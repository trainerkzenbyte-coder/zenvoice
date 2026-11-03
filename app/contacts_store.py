"""Contacts (CRM): one record per person, like Retell's Contacts page.

    data/crm/contacts.json   the contacts
    data/crm/fields.json     your own extra fields (Manage contact fields)

Each contact: contact_id, phone (+91...), first_name, last_name, tags, external_id,
fields {your custom fields}. "Do not call" is kept in campaigns/do_not_call.txt (the same file
the calls already respect), so marking a contact DNC also stops batch calls to that number.
Related conversations and the latest conversation are counted from logs/calls/*.json.

API (all need the admin token, like the rest of the dashboard):
    GET    /api/crm/contacts                 list (with conversation counts, DNC)
    POST   /api/crm/contacts                 add (merges by phone number)
    PUT    /api/crm/contacts/{id}            edit
    DELETE /api/crm/contacts/{id}            delete
    POST   /api/crm/contacts/delete          {"ids": [...]}  delete many
    POST   /api/crm/contacts/{id}/dnc        {"dnc": true|false}
    GET    /api/crm/contacts/{id}/calls      that contact's calls
    POST   /api/crm/import                   {"csv": "...", "tags": [...]}  Upload CSV
    POST   /api/crm/backfill                 {"contacts": [...]}  from batch lists + call history
    GET    /api/crm/export                   download all contacts as CSV
    GET    /api/crm/fields   PUT /api/crm/fields   {"fields": [{"key","label","type"}]}

Install: in app/server.py, right after the line that creates the app (app = FastAPI(...)), add
    from app.contacts_store import router as crm_router
    app.include_router(crm_router)
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from app.config import ROOT

CRM_DIR = Path(ROOT) / "data" / "crm"
CONTACTS_FILE = CRM_DIR / "contacts.json"
FIELDS_FILE = CRM_DIR / "fields.json"
DNC_FILE = Path(ROOT) / "campaigns" / "do_not_call.txt"
CALL_LOG_DIR = Path(ROOT) / "logs" / "calls"
MAX_CSV_BYTES = 5_000_000
FIELD_TYPES = ("text", "number", "date")
BASE_KEYS = {"contact_id", "phone", "first_name", "last_name", "tags", "external_id", "fields",
             "created", "updated", "dnc"}

_lock = threading.RLock()


# ---------- access: same admin token as the dashboard ----------
def _auth(request: Request) -> None:
    from app import auth  # recruiter / admin sessions or ADMIN_TOKEN via the tunnel (roles: app/auth.py)

    if not auth.identify(request):
        raise HTTPException(status_code=401, detail="Please sign in again")


router = APIRouter(prefix="/api/crm", dependencies=[Depends(_auth)])


# ---------- helpers ----------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def normalize_phone(raw) -> str:
    """'98765 43210', '09876543210', '919876543210', '+91-98765-43210' -> '+919876543210'."""
    s = str(raw or "").strip()
    digits = re.sub(r"\D", "", s)
    if not digits:
        return ""
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 10:
        return "+91" + digits
    if len(digits) >= 11 and (s.startswith("+") or digits.startswith("91")):
        return "+" + digits
    return "+" + digits if 8 <= len(digits) <= 15 else ""


def _key(phone: str) -> str:
    """Compare numbers by their last 10 digits."""
    return re.sub(r"\D", "", phone or "")[-10:]


def _clean_tags(tags) -> list[str]:
    if isinstance(tags, str):
        tags = re.split(r"[;,|]", tags)
    out = []
    for t in tags or []:
        t = str(t).strip()[:40]
        if t and t.lower() not in (x.lower() for x in out):
            out.append(t)
    return out[:20]


def _contacts() -> list[dict]:
    return _read(CONTACTS_FILE, [])


def _save(contacts: list[dict]) -> None:
    _write(CONTACTS_FILE, contacts)


def get_fields() -> list[dict]:
    return _read(FIELDS_FILE, [])


def _field_keys() -> set[str]:
    return {f["key"] for f in get_fields()}


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")[:40]


# ---------- Do Not Call (shared with the dialer) ----------
def _dnc_keys() -> set[str]:
    try:
        lines = DNC_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return {_key(ln) for ln in lines if ln.strip() and not ln.lstrip().startswith("#") and _key(ln)}


def set_dnc(phone: str, dnc: bool) -> None:
    k = _key(phone)
    if not k:
        return
    with _lock:
        try:
            lines = DNC_FILE.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        present = any(_key(ln) == k for ln in lines if not ln.lstrip().startswith("#"))
        if dnc and not present:
            lines.append(normalize_phone(phone) or phone)
        elif not dnc and present:
            lines = [ln for ln in lines if ln.lstrip().startswith("#") or _key(ln) != k]
        else:
            return
        DNC_FILE.parent.mkdir(parents=True, exist_ok=True)
        DNC_FILE.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------- conversations from call logs (cached per file) ----------
_log_cache: dict[str, tuple[float, dict]] = {}


def _call_summaries() -> list[dict]:
    out = []
    if not CALL_LOG_DIR.exists():
        return out
    seen = set()
    for f in CALL_LOG_DIR.glob("*.json"):
        name = str(f)
        seen.add(name)
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        hit = _log_cache.get(name)
        if not hit or hit[0] != mtime:
            rec = _read(f, {})
            outcome = rec.get("outcome") or {}
            hit = (mtime, {
                "call_id": rec.get("call_id") or f.stem,
                "phone": rec.get("phone") or "",
                "name": rec.get("candidate_name") or "",
                "started_at": rec.get("started_at") or rec.get("ended_at") or "",
                "duration_secs": rec.get("duration_secs"),
                "direction": rec.get("direction") or "",
                "outcome": (outcome.get("call_outcome") if isinstance(outcome, dict) else "") or "",
                "summary": (outcome.get("call_summary") or outcome.get("summary") or "") if isinstance(outcome, dict) else "",
            })
            _log_cache[name] = hit
        out.append(hit[1])
    for name in list(_log_cache):
        if name not in seen:
            del _log_cache[name]
    return out


def _calls_by_key() -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for c in _call_summaries():
        k = _key(c["phone"])
        if k:
            by.setdefault(k, []).append(c)
    for v in by.values():
        v.sort(key=lambda c: c["started_at"], reverse=True)
    return by


# ---------- create / update ----------
def _apply(c: dict, data: dict, fields: set[str]) -> None:
    for k in ("first_name", "last_name", "external_id"):
        if k in data and data[k] is not None:
            c[k] = str(data[k]).strip()[:120]
    if "tags" in data:
        c["tags"] = _clean_tags(data["tags"])
    if isinstance(data.get("fields"), dict):
        c.setdefault("fields", {})
        for k, v in data["fields"].items():
            if k in fields:
                v = "" if v is None else str(v).strip()[:500]
                if v:
                    c["fields"][k] = v
                else:
                    c["fields"].pop(k, None)
    c["updated"] = _now()


def upsert(data: dict, contacts: list[dict] | None = None, *, merge_tags: bool = True) -> tuple[dict, bool]:
    """Add a contact, or update the one with the same phone number. Returns (contact, created)."""
    phone = normalize_phone(data.get("phone"))
    if not phone:
        raise ValueError(f"Invalid phone number: {data.get('phone')!r}")
    own = contacts is None
    contacts = _contacts() if own else contacts
    fields = _field_keys()
    k = _key(phone)
    existing = next((c for c in contacts if _key(c["phone"]) == k), None)
    if existing:
        if merge_tags and "tags" in data:
            data = dict(data, tags=existing.get("tags", []) + _clean_tags(data["tags"]))
        # don't wipe names already known with blanks from a CSV
        data = {key: v for key, v in data.items() if not (key in ("first_name", "last_name") and not str(v or "").strip())}
        _apply(existing, data, fields)
        created, contact = False, existing
    else:
        contact = {"contact_id": "contact_" + secrets.token_hex(12), "phone": phone, "first_name": "",
                   "last_name": "", "tags": [], "external_id": "", "fields": {}, "created": _now()}
        _apply(contact, data, fields)
        contacts.append(contact)
        created = True
    if "dnc" in data:
        set_dnc(phone, _truthy(data["dnc"]))
    if own:
        _save(contacts)
    return contact, created


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "dnc", "on")


def _find(contacts: list[dict], cid: str) -> dict:
    c = next((c for c in contacts if c["contact_id"] == cid), None)
    if not c:
        raise HTTPException(status_code=404, detail="Contact not found")
    return c


def _with_activity(contacts: list[dict]) -> list[dict]:
    dnc, calls = _dnc_keys(), _calls_by_key()
    out = []
    for c in contacts:
        k = _key(c["phone"])
        mine = calls.get(k, [])
        out.append(dict(c, dnc=k in dnc, conversations=len(mine),
                        latest_conversation=mine[0]["started_at"] if mine else "",
                        latest_outcome=mine[0]["outcome"] if mine else ""))
    return out


# ---------- routes ----------
@router.get("/contacts")
def list_contacts():
    with _lock:
        contacts = _contacts()
    contacts.sort(key=lambda c: c.get("updated", ""), reverse=True)
    return {"contacts": _with_activity(contacts), "fields": get_fields()}


@router.post("/contacts")
async def add_contact(request: Request):
    data = await request.json()
    with _lock:
        try:
            contact, created = upsert(data, merge_tags=False)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"contact": _with_activity([contact])[0], "created": created}


@router.put("/contacts/{cid}")
async def edit_contact(cid: str, request: Request):
    data = await request.json()
    with _lock:
        contacts = _contacts()
        c = _find(contacts, cid)
        if data.get("phone") and _key(normalize_phone(data["phone"])) != _key(c["phone"]):
            new_phone = normalize_phone(data["phone"])
            if not new_phone:
                raise HTTPException(status_code=400, detail="Invalid phone number")
            if any(_key(o["phone"]) == _key(new_phone) for o in contacts if o is not c):
                raise HTTPException(status_code=400, detail="Another contact already has this number")
            was_dnc = _key(c["phone"]) in _dnc_keys()
            if was_dnc:
                set_dnc(c["phone"], False)
                set_dnc(new_phone, True)
            c["phone"] = new_phone
        _apply(c, data, _field_keys())
        if "dnc" in data:
            set_dnc(c["phone"], _truthy(data["dnc"]))
        _save(contacts)
    return {"contact": _with_activity([c])[0]}


@router.delete("/contacts/{cid}")
def delete_contact(cid: str):
    with _lock:
        contacts = _contacts()
        _find(contacts, cid)
        _save([c for c in contacts if c["contact_id"] != cid])
    return {"deleted": 1}


@router.post("/contacts/delete")
async def delete_many(request: Request):
    ids = set((await request.json()).get("ids") or [])
    with _lock:
        contacts = _contacts()
        keep = [c for c in contacts if c["contact_id"] not in ids]
        _save(keep)
    return {"deleted": len(contacts) - len(keep)}


@router.post("/contacts/{cid}/dnc")
async def contact_dnc(cid: str, request: Request):
    dnc = bool((await request.json()).get("dnc"))
    with _lock:
        c = _find(_contacts(), cid)
        set_dnc(c["phone"], dnc)
    return {"dnc": dnc}


@router.get("/contacts/{cid}/calls")
def contact_calls(cid: str):
    with _lock:
        c = _find(_contacts(), cid)
    return {"calls": _calls_by_key().get(_key(c["phone"]), [])}


PHONE_COLS = ("phone", "phone_number", "mobile", "mobile_number", "number", "contact_number", "phone_no", "to")
FIRST_COLS = ("first_name", "firstname", "name", "full_name", "candidate_name", "candidate")
LAST_COLS = ("last_name", "lastname", "surname")


@router.post("/import")
async def import_csv(request: Request):
    body = await request.json()
    text = str(body.get("csv") or "")
    if len(text.encode("utf-8")) > MAX_CSV_BYTES:
        raise HTTPException(status_code=400, detail="CSV is larger than 5 MB")
    text = text.lstrip("﻿")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="The CSV has no header row")
    cols = {_slug(h): h for h in reader.fieldnames if h}
    phone_col = next((cols[c] for c in PHONE_COLS if c in cols), None)
    if not phone_col:
        raise HTTPException(status_code=400, detail="No phone column found. Name a column 'phone' or 'phone_number'.")
    first_col = next((cols[c] for c in FIRST_COLS if c in cols), None)
    last_col = next((cols[c] for c in LAST_COLS if c in cols), None)
    tags_col, ext_col, dnc_col = cols.get("tags"), cols.get("external_id"), cols.get("dnc") or cols.get("do_not_call")
    used = {phone_col, first_col, last_col, tags_col, ext_col, dnc_col}
    extra_tags = _clean_tags(body.get("tags") or [])

    added = updated = skipped = 0
    errors, new_fields = [], []
    with _lock:
        fields = get_fields()
        known = {f["key"] for f in fields}
        extra_cols = {}
        for slug, original in cols.items():
            if original in used or slug in BASE_KEYS:
                continue
            if slug not in known:
                fields.append({"key": slug, "label": original.strip()[:40], "type": "text"})
                known.add(slug)
                new_fields.append(original.strip())
            extra_cols[slug] = original
        if new_fields:
            _write(FIELDS_FILE, fields)
        contacts = _contacts()
        for n, row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            data = {"phone": row.get(phone_col), "tags": _clean_tags(row.get(tags_col) or "") + extra_tags,
                    "fields": {k: row.get(orig) for k, orig in extra_cols.items()}}
            if first_col:
                data["first_name"] = row.get(first_col) or ""
            if last_col:
                data["last_name"] = row.get(last_col) or ""
            if ext_col:
                data["external_id"] = row.get(ext_col) or ""
            if dnc_col and str(row.get(dnc_col) or "").strip():
                data["dnc"] = row.get(dnc_col)
            try:
                _, created = upsert(data, contacts)
                added += created
                updated += not created
            except ValueError as e:
                skipped += 1
                if len(errors) < 10:
                    errors.append(f"Row {n}: {e}")
        _save(contacts)
    return {"added": added, "updated": updated, "skipped": skipped, "errors": errors, "new_fields": new_fields}


@router.post("/backfill")
async def backfill(request: Request):
    """Create contacts for everyone in your batch lists (sent by the dashboard) and call history."""
    listed = (await request.json()).get("contacts") or []
    added = 0
    with _lock:
        contacts = _contacts()
        have = {_key(c["phone"]) for c in contacts}
        people = [(p.get("phone"), p.get("name") or "", p.get("list") or "") for p in listed]
        people += [(c["phone"], c["name"], "") for c in _call_summaries()]
        for phone, name, lst in people:
            p = normalize_phone(phone)
            if not p or _key(p) in have or "browser" in str(phone):
                continue
            upsert({"phone": p, "first_name": name, "tags": [lst.rsplit(".", 1)[0]] if lst else []}, contacts)
            have.add(_key(p))
            added += 1
        _save(contacts)
    return {"added": added}


@router.get("/export")
def export_csv():
    with _lock:
        rows = _with_activity(_contacts())
        fields = get_fields()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["phone", "first_name", "last_name", "tags", "contact_id", "external_id", "do_not_call",
                "conversations", "latest_conversation"] + [f["key"] for f in fields])
    for c in rows:
        w.writerow([c["phone"], c.get("first_name", ""), c.get("last_name", ""), ";".join(c.get("tags", [])),
                    c["contact_id"], c.get("external_id", ""), "yes" if c["dnc"] else "no", c["conversations"],
                    c["latest_conversation"]] + [c.get("fields", {}).get(f["key"], "") for f in fields])
    return Response(buf.getvalue().encode("utf-8-sig"), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="contacts-{time.strftime("%Y%m%d")}.csv"'})


@router.get("/fields")
def list_fields():
    return {"fields": get_fields()}


@router.put("/fields")
async def save_fields(request: Request):
    out, seen = [], set()
    for f in (await request.json()).get("fields") or []:
        label = str(f.get("label") or "").strip()[:40]
        key = _slug(f.get("key") or label)
        if not key or key in seen or key in BASE_KEYS:
            continue
        seen.add(key)
        out.append({"key": key, "label": label or key, "type": f.get("type") if f.get("type") in FIELD_TYPES else "text"})
    with _lock:
        _write(FIELDS_FILE, out[:30])
    return {"fields": out[:30]}
