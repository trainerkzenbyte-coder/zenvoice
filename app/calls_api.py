"""Call History (Retell-style table + call details drawer), built only from your own files:

    logs/calls/*.json         one record per connected call (transcript, outcome, latency, tokens)
    logs/call_events.jsonl    Vobiz hang-up reports: used for calls that never connected
    logs/recordings/*.wav     call audio, only when recording is switched on
    logs/server.log           the "Detail Logs" tab (lines for that call)

Same access rules as the rest of the console: this computer only, ADMIN_TOKEN required.
"""

import datetime as dt
import json
import math
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app import campaign as cm
from app import dashboard as core
from app import qa
from app.config import ROOT, settings

router = APIRouter()
RECORDING_DIR = Path(ROOT) / "logs" / "recordings"
SERVER_LOG = Path(ROOT) / "logs" / "server.log"

END_REASONS = {"candidate": "user hangup", "agent": "agent hangup", "time_limit": "max duration reached",
               "silence": "inactivity", "error": "error", "transfer": "call transfer"}
DIAL_FAILURES = {"NO_ANSWER": "dial no answer", "NO_USER_RESPONSE": "dial no answer", "USER_BUSY": "dial busy",
                 "CALL_REJECTED": "dial rejected", "ORIGINATOR_CANCEL": "dial cancelled",
                 "UNALLOCATED_NUMBER": "invalid number", "INVALID_NUMBER_FORMAT": "invalid number"}


def _iso(when: dt.datetime) -> str:
    return when.astimezone(cm.IST).isoformat(timespec="seconds")


def _successful(out: dict) -> bool | None:
    """Successful = the screening was completed, or a callback was booked with a time."""
    o = out.get("call_outcome")
    if not o:
        return None
    return o == "completed" or (o == "callback_requested" and bool((out.get("data") or {}).get("callback_time")))


def _end_reason(rec: dict) -> str:
    out = rec.get("outcome") or {}
    if out.get("call_outcome") == "voicemail":
        return "voicemail reached"
    ended_by = out.get("ended_by") or ""
    if not ended_by:
        m = re.search(r"ended by (\w+)", str(rec.get("end_reason", "")))
        ended_by = m.group(1) if m else ""
    return END_REASONS.get(ended_by, ended_by or "unknown")


def _cost(duration: float, connected: bool) -> float | None:
    from app.dashboard_ext import billing_rates

    r = billing_rates()
    if r.get("vobiz_inr_per_min") is None:
        return None
    cost = math.ceil(duration / 60) * float(r["vobiz_inr_per_min"]) if connected and duration else 0.0
    if connected and r.get("deepgram_usd_per_min") is not None and r.get("usd_to_inr"):
        cost += duration / 60 * float(r["deepgram_usd_per_min"]) * float(r["usd_to_inr"])
    if connected and r.get("gemini_inr_per_call") is not None:
        cost += float(r["gemini_inr_per_call"])
    return round(cost, 2)


def _row_from_log(path: Path, rec: dict) -> dict:
    out = rec.get("outcome") or {}
    lat = rec.get("latency") or {}
    when = core._when(rec, path)
    duration = float(rec.get("duration_secs") or 0)
    web = str(rec.get("call_id", "")).startswith("browser-")
    sentiment = out.get("user_sentiment") or (out.get("data") or {}).get("user_sentiment")
    rid = str(rec.get("call_id") or path.stem)
    return {
        "id": rid, "file": path.name, "time": _iso(when), "duration_secs": round(duration),
        "channel": "web_call" if web else "phone_call", "cost_inr": None if web else _cost(duration, True),
        "end_reason": _end_reason(rec), "status": "ended", "sentiment": sentiment,
        "from": "" if web else (rec.get("from_number") or os.getenv("VOBIZ_FROM_NUMBER", "").strip()),
        "to": "" if web else rec.get("phone", ""), "direction": rec.get("direction") or ("web" if web else "outbound"),
        "successful": _successful(out), "latency_ms": lat.get("p50_ms"), "candidate_name": rec.get("candidate_name", ""),
        "role": rec.get("role", ""), "call_outcome": out.get("call_outcome"), "summary": out.get("summary", ""),
        "has_recording": bool(rec.get("recording")), "simulated": core._is_test(rec), "ref": rec.get("ref", ""),
        "agent": rec.get("agent") or settings.get("agent", {}).get("name", "Priya"),
    }


def _ref_names() -> dict[str, dict]:
    """Campaign reference -> candidate (for calls that never connected)."""
    out: dict[str, dict] = {}
    for path in cm.CAMPAIGN_DIR.glob("*.state.json") if cm.CAMPAIGN_DIR.exists() else []:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for c in (state.get("candidates") or {}).values():
            refs = [c.get("last_ref")] + [h.get("ref") for h in c.get("history") or []]
            for ref in refs:
                if ref:
                    out[ref] = {"name": c.get("name", ""), "phone": c.get("phone", ""), "role": c.get("role", "")}
    return out


def _rows(include_tests: bool = False) -> list[dict]:
    rows, seen_ids, seen_refs = [], set(), set()
    for path, rec in core._load_logs():
        ch = core._channel(rec)
        if ch in core.TEXT_CHANNELS and ch != "browser test":
            continue  # text chats live in Chat History
        if core._is_test(rec) and not include_tests:
            continue
        row = _row_from_log(path, rec)
        rows.append(row)
        seen_ids.add(row["id"])
        if row["ref"]:
            seen_refs.add(row["ref"])
    names = None
    for e in core._events():
        cid, ref = str(e.get("call_uuid") or ""), str(e.get("ref") or "")
        try:
            answered = float(e.get("bill_duration") or e.get("duration") or 0) > 0
        except ValueError:
            answered = False
        if answered or (cid and cid in seen_ids) or (ref and ref in seen_refs):
            continue  # connected calls come from their call log
        names = names if names is not None else _ref_names()
        who = names.get(ref, {})
        cause = str(e.get("cause") or "").upper()
        when = dt.datetime.fromtimestamp(float(e.get("t", 0)), cm.IST)
        rows.append({
            "id": cid or f"ref-{ref or int(float(e.get('t', 0)))}", "file": "", "time": _iso(when), "duration_secs": 0,
            "channel": "phone_call", "cost_inr": _cost(0, False), "end_reason": DIAL_FAILURES.get(cause, f"dial failed ({cause.lower() or 'unknown'})"),
            "status": "not_connected", "sentiment": None, "from": os.getenv("VOBIZ_FROM_NUMBER", "").strip(),
            "to": str(e.get("to") or who.get("phone", "")), "direction": "outbound", "successful": None, "latency_ms": None,
            "candidate_name": who.get("name", ""), "role": who.get("role", ""), "call_outcome": None, "summary": "",
            "has_recording": False, "simulated": False, "ref": ref, "agent": settings.get("agent", {}).get("name", "Priya"),
        })
        seen_ids.add(rows[-1]["id"])
    rows.sort(key=lambda r: r["time"], reverse=True)
    return rows


@router.get("/api/calls")
async def calls(request: Request, start: str = "", end: str = "", include_tests: bool = False, limit: int = 1000):
    core._guard(request)
    rows = _rows(include_tests)
    if start:
        rows = [r for r in rows if r["time"][:10] >= start]
    if end:
        rows = [r for r in rows if r["time"][:10] <= end]
    return {"calls": rows[: max(1, min(limit, 5000))], "currency": "INR",
            "cost_note": "Estimated from your rates in Settings" if rows and rows[0]["cost_inr"] is not None else "Set your rates in Settings to see costs"}


def _server_log_lines(call_id: str, max_lines: int = 300) -> list[str]:
    if not SERVER_LOG.exists() or not call_id:
        return []
    tag = f"[{call_id[-6:]}]"
    size = SERVER_LOG.stat().st_size
    with SERVER_LOG.open("rb") as f:
        f.seek(max(0, size - 5_000_000))
        data = f.read().decode("utf-8", errors="replace")
    return [ln for ln in data.splitlines() if tag in ln][-max_lines:]


@router.get("/api/calls/{call_id}")
async def call_detail(request: Request, call_id: str):
    core._guard(request)
    rows = _rows(include_tests=True)
    row = next((r for r in rows if r["id"] == call_id), None)
    if not row:
        raise HTTPException(404, "Call not found")
    rec = {}
    if row["file"]:
        rec = json.loads((core.CALL_LOG_DIR / row["file"]).read_text(encoding="utf-8"))
    out = rec.get("outcome") or {}
    lat = rec.get("latency") or {}
    transcript = rec.get("transcript") or []
    times = [t["t"] for t in transcript if isinstance(t.get("t"), (int, float))]
    base = min(times) if times else 0
    turns = [{"role": t.get("role"), "text": t.get("text", ""), "offset": round(t["t"] - base, 1) if isinstance(t.get("t"), (int, float)) else None}
             for t in transcript]
    questions = [{"question": q, "resolved": ok} for _, q, ok in qa.questions(transcript)]
    key = cm.phone_key(row["to"])
    name = (row["candidate_name"] or "").strip()
    first, _, last = name.partition(" ")
    lists = []
    for path in sorted(cm.CAMPAIGN_DIR.glob("*.csv")) if cm.CAMPAIGN_DIR.exists() and key else []:
        try:
            c = cm.load_state(path)["candidates"].get(key)
        except Exception:  # noqa: BLE001
            c = None
        if c:
            lists.append({"file": path.name, "status": core.DISPLAY_STATUS.get(c["status"], c["status"]), "attempts": c.get("attempts", 0)})
    convs = [r for r in rows if key and cm.phone_key(r["to"]) == key and (not r["simulated"] or r["id"] == row["id"])]
    actions = out.get("actions") or []
    return {
        **row,
        "title": f"{row['time'][5:7]}/{row['time'][8:10]}/{row['time'][:4]} {row['time'][11:16]} {row['channel']}",
        "tokens": (lat.get("tokens") or {}),
        "analysis": {
            "call_successful": row["successful"], "call_status": row["status"], "user_sentiment": row["sentiment"],
            "disconnection_reason": row["end_reason"], "latency_ms": row["latency_ms"],
            "first_audio_ms": lat.get("first_audio_ms"), "p95_ms": lat.get("p95_ms"), "call_outcome": out.get("call_outcome"),
        },
        "custom": {**(out.get("data") or {}), "candidate_showed_interest": (out.get("data") or {}).get("interested")},
        "unscripted_questions": questions,
        "summary": out.get("summary", ""), "recruiter_notes": out.get("recruiter_notes", ""), "checks": out.get("checks") or [],
        "transcript": turns,
        "data": {"outcome": out, "latency": lat, "record": {k: v for k, v in rec.items() if k not in ("transcript", "outcome", "latency")}},
        "detail_logs": {"actions": actions, "turn_ms": lat.get("turn_ms") or [], "server_log": _server_log_lines(row["id"])},
        "contact": {"first_name": first, "last_name": last, "phone": row["to"], "contact_id": f"ct_{key}" if key else "",
                    "lists": lists, "dnc": key in cm.load_dnc() if key else False},
        "conversations": [{"id": r["id"], "time": r["time"], "duration_secs": r["duration_secs"], "status": r["status"],
                           "end_reason": r["end_reason"], "successful": r["successful"], "current": r["id"] == row["id"]} for r in convs],
    }


@router.get("/api/calls/{call_id}/recording")
async def call_recording(request: Request, call_id: str):
    core._guard(request)
    rows = _rows(include_tests=True)
    row = next((r for r in rows if r["id"] == call_id and r["file"]), None)
    if not row:
        raise HTTPException(404, "Call not found")
    rec = json.loads((core.CALL_LOG_DIR / row["file"]).read_text(encoding="utf-8"))
    name = rec.get("recording")
    path = RECORDING_DIR / str(name or "")
    if not name or not re.fullmatch(r"[\w.-]+\.wav", name) or not path.exists():
        raise HTTPException(404, "No recording for this call")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})
