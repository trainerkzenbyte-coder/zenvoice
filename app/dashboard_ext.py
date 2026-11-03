"""Console v3 routes: agent profiles, knowledge bases, phone numbers, batch calls, AI QA,
alerting, integrations, billing, settings and the getting-started checklist.

Same rules as app/dashboard.py: only answers on this computer, needs ADMIN_TOKEN.
Everything shown is computed from your own files (logs/calls, logs/call_events.jsonl,
campaigns/, config/); nothing is invented.
"""

import asyncio
import datetime as dt
import json
import math
import os
import re
import time
from pathlib import Path

import httpx
import yaml
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from loguru import logger

from app import agents_store, kb_store, qa, sheet_sync, vobiz
from app import campaign as cm
from app import dashboard as core
from app.config import ROOT, settings

router = APIRouter()
CONSOLE_FILE = Path(ROOT) / "config" / "console.yaml"
ALERTS_FILE = Path(ROOT) / "config" / "alerts.yaml"
BILLING_FILE = Path(ROOT) / "config" / "billing.yaml"
_checks: dict[str, dict] = {}
_alert_state: dict[str, bool] = {}
ALERT_HISTORY: list[dict] = []

guard = core._guard


def _yaml(path: Path) -> dict:
    return core._yaml(path)


def _save_yaml(path: Path, data: dict, header: str) -> None:
    path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def console_cfg() -> dict:
    c = _yaml(CONSOLE_FILE)
    return {"workspace_name": c.get("workspace_name") or "Zenbyte",
            "user_label": c.get("user_label") or os.getenv("DASHBOARD_USER", "").strip() or "Admin"}


def gemini_tier() -> str:
    return str(_yaml(core.CAMPAIGN_FILE).get("gemini_tier", "free")).lower()


def max_allowed_concurrency() -> int:
    return 1 if gemini_tier() == "free" else 5


# ======================= Agents =======================
async def _live_settings() -> dict:
    return await core._config_snapshot()


@router.get("/api/agents")
async def agents_list(request: Request):
    guard(request)
    agents_store.ensure_default(await _live_settings())
    rows = []
    for a in agents_store.list_agents():
        s = a.get("settings", {})
        rows.append({"id": a["id"], "name": a["name"], "type": a.get("type", "Single Prompt"), "folder": a.get("folder", ""),
                     "voice": s.get("voice", ""), "model": s.get("model", ""), "active": a["active"],
                     "phone": os.getenv("VOBIZ_FROM_NUMBER", "").strip() if a["active"] else "",
                     "updated": a.get("updated", ""), "created": a.get("created", "")})
    return {"agents": rows, "folders": agents_store.meta()["folders"]}


@router.get("/api/agents/{agent_id}")
async def agents_get(request: Request, agent_id: str):
    guard(request)
    try:
        a = agents_store.get(agent_id)
    except KeyError:
        raise HTTPException(404, "Agent not found") from None
    if a["active"]:
        a["settings"] = await _live_settings()  # the live files are the truth for the active agent
    cfg = await core._config_dict()
    a["protected_fields"] = cfg["protected_fields"]
    a["prompt_file"] = cfg["prompt_file"]
    a["knowledge_chars"] = cfg["knowledge_chars"]
    a["sheet_system_columns"] = cfg["sheet_system_columns"]
    a["studio"] = _studio_info()
    return a


def _studio_info() -> dict:
    """Header pills and side-panel facts for the Agent Studio, all from real data."""
    from app.call_actions import active_tool_specs

    p50s, firsts = [], []
    for path, rec in core._load_logs()[:200]:
        if core._is_test(rec) or core._channel(rec) in core.TEXT_CHANNELS:
            continue
        lat = rec.get("latency") or {}
        if lat.get("p50_ms"):
            p50s.append(lat["p50_ms"])
        if lat.get("first_audio_ms"):
            firsts.append(lat["first_audio_ms"])
        if len(p50s) >= 20:
            break
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    rates = billing_rates()
    per_min = None
    if rates.get("vobiz_inr_per_min") is not None:
        per_min = float(rates["vobiz_inr_per_min"])
        if rates.get("deepgram_usd_per_min") is not None and rates.get("usd_to_inr"):
            per_min += float(rates["deepgram_usd_per_min"]) * float(rates["usd_to_inr"])
    kbs = kb_store.list_kbs()
    used = kb_store.prompt_text()[1]
    hook = os.getenv("SHEET_WEBHOOK_URL", "").strip()
    return {
        "median_reply_ms": med(p50s), "median_first_audio_ms": med(firsts), "calls_measured": len(p50s),
        "cost_per_min_inr": round(per_min, 2) if per_min is not None else None, "gemini_tier": gemini_tier(),
        "tools": [{"name": t["name"], "description": t["description"], "params": list(t["properties"])} for t in active_tool_specs()],
        "kb": {"enabled": sum(1 for k in kbs if k.get("enabled", True)), "total": len(kbs),
               "sources": sum(k.get("source_count", 0) for k in kbs if k.get("enabled", True)),
               "faqs": sum(1 for k in kbs if k.get("enabled", True) for src in kb_store.get_sources(k["id"]) if src["type"] == "faq"),
               "chars": used, "max_chars": kb_store.prompt_limit()},
        "webhook": {"sheet_configured": bool(hook), "sheet_url_masked": (re.sub(r"/macros/s/[^/]+/", "/macros/s/•••/", hook) if hook else ""),
                    "public_url": core._public.get("url") or os.getenv("PUBLIC_BASE_URL", "").strip(), "public_ok": core._public.get("ok")},
    }


@router.post("/api/agents/save")
async def agents_publish(request: Request):
    """Agent Studio "Publish Changes". Body: {agent_id (default: live agent), name?, folder?, settings: {...}}.
    For the live agent this writes config/agent.yaml, config/screening.yaml and the prompt file
    (each with a .bak copy of the previous version)."""
    guard(request)
    body = await request.json()
    agents_store.ensure_default(await _live_settings())
    agent_id = body.get("agent_id") or agents_store.meta()["active"]
    return await _save_agent(agent_id, body)


@router.get("/api/prompt-preview")
async def prompt_preview(request: Request):
    """Exactly what the live agent's AI gets as instructions on the next call (with knowledge base)."""
    guard(request)
    from app.prompts import load_prompt

    text = load_prompt(candidate_name="Karthik")
    return {"prompt": text, "chars": len(text), "approx_tokens": len(text) // 4, "knowledge": kb_store.report()}


@router.get("/api/prompt-templates")
async def prompt_templates(request: Request):
    guard(request)
    folder = Path(ROOT) / "config" / "prompts"
    out = []
    for name, label in (("zenbyte_recruiter_natural.txt", "Natural recruiter call (Sprint 9, recommended)"),
                        ("zenbyte_talent_advisor.txt", "Zenbyte Talent Advisor (structured)"),
                        ("sprint3_screening.txt", "Original screening script")):
        path = folder / name
        if path.exists():
            out.append({"file": name, "label": label, "text": path.read_text(encoding="utf-8")})
    return {"templates": out}


@router.get("/api/test-audio")
async def test_audio(request: Request, name: str = "", role: str = ""):
    """Browser microphone test: the page connects to the same audio WebSocket Vobiz uses.
    Uses the PUBLISHED agent. No phone call, no Sheet row; logged under Chat History."""
    guard(request)
    import secrets as _s

    ctx = vobiz.encode_context(name.strip()[:60], "", "", role.strip()[:80])
    return {"ws_path": f"/ws/{vobiz.stream_token()}/{ctx}", "call_id": f"browser-{_s.token_hex(6)}"}


@router.post("/api/agents")
async def agents_create(request: Request):
    guard(request)
    body = await request.json()
    template = body.get("template", "blank")
    base = await _live_settings()
    if template == "duplicate" and body.get("from"):
        try:
            base = dict(agents_store.get(body["from"])["settings"])
        except KeyError:
            raise HTTPException(404, "Agent to copy not found") from None
    elif template == "blank":
        base = dict(base, prompt="You are {agent_name}, a friendly recruiter at {company}, calling {candidate_name} "
                                 "about the {job_title} role in {job_location}.\n\nKeep every reply short. Ask one question "
                                 "at a time. Say goodbye and call end_call when done.")
    name = core._clean_str(body.get("name") or "New agent", "Agent name", 80)
    a = agents_store.create(name, base, folder=str(body.get("folder", ""))[:60])
    return {"ok": True, "agent": a}


@router.post("/api/agents/import")
async def agents_import(request: Request, file: UploadFile = File(...)):
    guard(request)
    try:
        data = json.loads((await file.read(500_000)).decode("utf-8"))
        settings_in = data.get("settings") or {}
        core.parse_config({k: v for k, v in settings_in.items() if k in agents_store.SETTING_KEYS})  # validate
    except HTTPException as e:
        raise HTTPException(400, f"Invalid agent file: {e.detail}") from None
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Not an exported agent file: {e}") from None
    a = agents_store.create(str(data.get("name", "Imported agent")) + " (imported)", settings_in)
    return {"ok": True, "agent": a}


@router.put("/api/agents/{agent_id}")
async def agents_save(request: Request, agent_id: str):
    guard(request)
    return await _save_agent(agent_id, await request.json())


async def _save_agent(agent_id: str, body: dict) -> dict:
    try:
        a = agents_store.get(agent_id)
    except KeyError:
        raise HTTPException(404, "Agent not found") from None
    changes = {k: v for k, v in (body.get("settings") or {}).items() if k in agents_store.SETTING_KEYS}
    if a["active"] and changes:
        core.apply_config(changes)  # writes the live files
        agents_store.update(agent_id, settings=await _live_settings())
    elif changes:
        core.parse_config(changes)  # validate only
        agents_store.update(agent_id, settings=changes)
    if "name" in body or "folder" in body:
        agents_store.update(agent_id, name=body.get("name"), folder=body.get("folder"))
    return {"ok": True, "live": a["active"]}


@router.post("/api/agents/{agent_id}/activate")
async def agents_activate(request: Request, agent_id: str):
    guard(request)
    try:
        a = agents_store.get(agent_id)
    except KeyError:
        raise HTTPException(404, "Agent not found") from None
    agents_store.snapshot_active(await _live_settings())  # keep the current live agent's latest settings
    core.apply_config(dict(a["settings"]))
    agents_store.set_active(agent_id)
    logger.info(f"Console: agent '{a['name']}' is now live")
    return {"ok": True}


@router.delete("/api/agents/{agent_id}")
async def agents_delete(request: Request, agent_id: str):
    guard(request)
    try:
        agents_store.delete(agent_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except KeyError:
        raise HTTPException(404, "Agent not found") from None
    return {"ok": True}


@router.post("/api/agent-folders")
async def folders_save(request: Request):
    guard(request)
    body = await request.json()
    m = agents_store.meta()
    if body.get("add"):
        name = core._clean_str(body["add"], "Folder name", 40)
        if name not in m["folders"]:
            m["folders"].append(name)
    if body.get("remove"):
        m["folders"] = [f for f in m["folders"] if f != body["remove"]]
        for a in agents_store.list_agents():
            if a.get("folder") == body["remove"]:
                agents_store.update(a["id"], folder="")
    agents_store.save_meta(m)
    return {"ok": True, "folders": m["folders"]}


# ======================= Knowledge bases =======================
def _kb_or_404(kb_id: str) -> dict:
    kb = kb_store.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return kb


@router.get("/api/kbs")
async def kbs_list(request: Request):
    guard(request)
    _, used, total = kb_store.prompt_text()
    return {"kbs": kb_store.list_kbs(), "prompt_used": used, "prompt_total": total, "max_chars": kb_store.prompt_limit()}


@router.post("/api/kbs")
async def kbs_create(request: Request):
    guard(request)
    name = core._clean_str((await request.json()).get("name"), "Name", 80)
    return {"ok": True, "kb": kb_store.create_kb(name)}


@router.get("/api/kbs/{kb_id}")
async def kbs_get(request: Request, kb_id: str):
    guard(request)
    kb = _kb_or_404(kb_id)
    sources = kb_store.get_sources(kb_id)
    for s in sources:
        s["chars"] = len(kb_store._source_text(s))
    return {"kb": kb, "sources": sources}


@router.put("/api/kbs/{kb_id}")
async def kbs_update(request: Request, kb_id: str):
    guard(request)
    _kb_or_404(kb_id)
    body = await request.json()
    changes = {}
    if "name" in body:
        changes["name"] = core._clean_str(body["name"], "Name", 80)
    if "enabled" in body:
        changes["enabled"] = bool(body["enabled"])
    return {"ok": True, "kb": kb_store.update_kb(kb_id, **changes)}


@router.delete("/api/kbs/{kb_id}")
async def kbs_delete(request: Request, kb_id: str):
    guard(request)
    _kb_or_404(kb_id)
    kb_store.delete_kb(kb_id)
    return {"ok": True}


@router.post("/api/kbs/{kb_id}/sources")
async def kbs_add_source(request: Request, kb_id: str):
    guard(request)
    _kb_or_404(kb_id)
    body = await request.json()
    kind = body.get("type", "text")
    if kind == "url":
        url = core._clean_str(body.get("url"), "Web address", 500)
        try:
            src = await asyncio.to_thread(kb_store.add_url, kb_id, url, int(body.get("max_pages", 1)))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True, "source": {k: v for k, v in src.items() if k != "pages"} | {"page_count": len(src["pages"])}}
    title = core._clean_str(body.get("title"), "Title" if kind != "faq" else "Question", 200)
    text = core._clean_str(body.get("text"), "Text" if kind != "faq" else "Answer", kb_store.MAX_PAGE_CHARS)
    return {"ok": True, "source": kb_store.add_text(kb_id, title, text, kind="faq" if kind == "faq" else "snippet")}


@router.post("/api/kbs/{kb_id}/upload")
async def kbs_upload(request: Request, kb_id: str, file: UploadFile = File(...)):
    guard(request)
    _kb_or_404(kb_id)
    name = Path(file.filename or "document.txt").name
    if not re.search(r"\.(txt|md|csv|html?|pdf)$", name, re.I):
        raise HTTPException(400, "Upload a .pdf, .txt, .md, .csv or .html document (for Word, save as PDF first)")
    if name.lower().endswith(".pdf"):
        text = _pdf_text(await file.read(20 * 1024 * 1024 + 1))
        if not text.strip():
            raise HTTPException(400, "No text found in this PDF (a scanned image PDF has no text to read)")
        return {"ok": True, "source": kb_store.add_text(kb_id, name, text[: kb_store.MAX_PAGE_CHARS])}
    raw = (await file.read(kb_store.MAX_DOWNLOAD + 1))[: kb_store.MAX_DOWNLOAD]
    text = raw.decode("utf-8", errors="replace")
    if name.lower().endswith((".htm", ".html")):
        p = kb_store._TextExtractor()
        p.feed(text)
        text = p.text()
    if not text.strip():
        raise HTTPException(400, "The document is empty")
    return {"ok": True, "source": kb_store.add_text(kb_id, name, text)}


def _pdf_text(data: bytes) -> str:
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "PDF too large (max 20 MB)")
    try:
        from pypdf import PdfReader
    except ImportError:
        raise HTTPException(400, "To upload PDFs, install the PDF reader once: pip install pypdf") from None
    import io

    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages[:200])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read this PDF: {e}") from None


@router.put("/api/kbs/{kb_id}/sources/{src_id}")
async def kbs_update_source(request: Request, kb_id: str, src_id: str):
    guard(request)
    body = await request.json()
    try:
        return {"ok": True, "source": kb_store.update_source(kb_id, src_id, **{k: body[k] for k in ("title", "text", "enabled") if k in body})}
    except KeyError:
        raise HTTPException(404, "Source not found") from None


@router.delete("/api/kbs/{kb_id}/sources/{src_id}")
async def kbs_delete_source(request: Request, kb_id: str, src_id: str):
    guard(request)
    kb_store.delete_source(kb_id, src_id)
    return {"ok": True}


@router.post("/api/kbs/{kb_id}/sync")
async def kbs_sync(request: Request, kb_id: str):
    guard(request)
    _kb_or_404(kb_id)
    return {"ok": True, **(await asyncio.to_thread(kb_store.sync_kb, kb_id))}


# ======================= Phone numbers =======================
@router.get("/api/numbers")
async def numbers(request: Request):
    guard(request)
    base = await core.api_phone_numbers(request)
    agents_store.ensure_default(await _live_settings())
    active = next((a for a in agents_store.list_agents() if a["active"]), None)
    auth_id = os.getenv("VOBIZ_AUTH_ID", "").strip()
    return {**base, "agent": active["name"] if active else "", "auth_id_masked": (auth_id[:4] + "•••" + auth_id[-3:]) if len(auth_id) > 8 else "",
            "routing": "Outbound only: test calls, Dial now and batch calls go out through Vobiz from this number. "
                       "When the candidate answers, Vobiz streams the audio to the answer URL below."}


# ======================= Batch calls =======================
def _meta_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".batch.json")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_meta(csv_path: Path, meta: dict) -> None:
    _meta_path(csv_path).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _batch_row(csv_path: Path) -> dict:
    meta = _read_json(_meta_path(csv_path))
    try:
        state = cm.load_state(csv_path)
    except Exception as e:  # noqa: BLE001
        return {"file": csv_path.name, "name": meta.get("name", csv_path.stem), "status": "error", "error": str(e)[:120]}
    cands = list(state["candidates"].values())
    total = len(cands)
    finished = sum(1 for c in cands if c["status"] in cm.DONE)
    started = any(c.get("attempts", 0) or c.get("dials", 0) for c in cands)
    info = core._campaign_info()
    running = info["running"] and info["file"] == csv_path.name
    if running:
        status = "pausing" if info.get("pausing") else "in_progress"
    elif meta.get("status") == "scheduled":
        status = "scheduled"
    elif meta.get("status") == "draft" and not started:
        status = "draft"
    elif total and finished == total and started:
        status = "completed"
    elif started:
        status = "paused"
    else:
        status = "draft"
    counts: dict[str, int] = {}
    for c in cands:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {"file": csv_path.name, "name": meta.get("name") or csv_path.stem.replace("_", " "), "status": status,
            "total": total, "finished": finished, "progress": round(100 * finished / total) if total else 0,
            "counts": counts, "created": meta.get("created") or time.strftime("%Y-%m-%d %H:%M", time.localtime(csv_path.stat().st_mtime)),
            "schedule_at": meta.get("schedule_at"), "concurrency": meta.get("concurrency", 1),
            "from_number": meta.get("from_number") or os.getenv("VOBIZ_FROM_NUMBER", "").strip(),
            "stopped_because": info.get("stopped_because") if info["file"] == csv_path.name else None}


@router.get("/api/batches")
async def batches(request: Request):
    guard(request)
    cm.CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_batch_row(p) for p in sorted(cm.CAMPAIGN_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)]
    return {"batches": rows, "tier": gemini_tier(), "max_concurrency": max_allowed_concurrency(),
            "calling_hours": (settings.get("call_limits", {}) or {}).get("calling_hours", ["09:00", "21:00"]),
            "from_numbers": [n for n in [os.getenv("VOBIZ_FROM_NUMBER", "").strip()] if n]}


def start_batch(csv_path: Path, concurrency: int) -> None:
    task = core._campaign["task"]
    if task and not task.done():
        raise HTTPException(409, f"Batch {core._campaign['file']} is running. Pause it first (one batch at a time).")
    runner = cm.CampaignRunner(csv_path, max_concurrent=min(max(1, concurrency), max_allowed_concurrency()))

    async def run():
        try:
            await runner.run()
        except Exception as e:  # noqa: BLE001
            core._campaign["error"] = str(e)[:300]
            logger.exception(f"Batch crashed: {e}")

    core._campaign.update(runner=runner, file=csv_path.name, started=cm.iso(cm.now_ist()), error="",
                          task=asyncio.create_task(run()))
    meta = _read_json(_meta_path(csv_path))
    meta.update(status="launched", launched=time.strftime("%Y-%m-%d %H:%M"))
    _save_meta(csv_path, meta)


@router.post("/api/batches")
async def batches_create(request: Request, name: str = Form(...), mode: str = Form("draft"),
                         concurrency: int = Form(1), schedule_at: str = Form(""), file: UploadFile = File(...)):
    guard(request)
    name = core._clean_str(name, "Batch call name", 60)
    if mode not in ("draft", "now", "schedule"):
        raise HTTPException(400, "mode must be draft, now or schedule")
    if not 1 <= concurrency <= max_allowed_concurrency():
        raise HTTPException(400, f"Concurrency must be 1 to {max_allowed_concurrency()} on the Gemini {gemini_tier()} tier")
    when = None
    if mode == "schedule":
        when = cm.parse_iso(schedule_at)
        if not when or when <= cm.now_ist():
            raise HTTPException(400, "Pick a schedule time in the future (India time)")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:50] or "batch"
    path = cm.CAMPAIGN_DIR / f"{slug}.csv"
    n = 2
    while path.exists():
        path = cm.CAMPAIGN_DIR / f"{slug}_{n}.csv"
        n += 1
    data = await file.read(1_000_001)
    if len(data) > 1_000_000:
        raise HTTPException(400, "CSV larger than 1 MB")
    cm.CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".upload")
    tmp.write_bytes(data)
    try:
        rows = cm.read_candidates(tmp)
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read the CSV: {e}") from None
    if not rows or not any(r["phone"] for r in rows):
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "The CSV needs a header row with name and phone columns (optional: role)")
    tmp.replace(path)
    meta = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "concurrency": concurrency,
            "from_number": os.getenv("VOBIZ_FROM_NUMBER", "").strip(),
            "status": {"draft": "draft", "now": "launching", "schedule": "scheduled"}[mode],
            "schedule_at": cm.iso(when) if when else None}
    _save_meta(path, meta)
    cm.save_state(path, cm.load_state(path))
    if mode == "now":
        start_batch(path, concurrency)
    return {"ok": True, "file": path.name, "rows": len(rows), "status": meta["status"]}


@router.post("/api/batches/{file}/launch")
async def batches_launch(request: Request, file: str):
    guard(request)
    path = core._csv(file)
    meta = _read_json(_meta_path(path))
    start_batch(path, int(meta.get("concurrency", 1)))
    return {"ok": True}


@router.post("/api/batches/{file}/schedule")
async def batches_schedule(request: Request, file: str):
    guard(request)
    path = core._csv(file)
    body = await request.json()
    meta = _read_json(_meta_path(path))
    if body.get("cancel"):
        meta.update(status="draft", schedule_at=None)
    else:
        when = cm.parse_iso(body.get("schedule_at"))
        if not when or when <= cm.now_ist():
            raise HTTPException(400, "Pick a time in the future")
        meta.update(status="scheduled", schedule_at=cm.iso(when))
    _save_meta(path, meta)
    return {"ok": True}


@router.delete("/api/batches/{file}")
async def batches_delete(request: Request, file: str):
    guard(request)
    path = core._csv(file)
    if core._campaign["file"] == path.name and core._campaign["task"] and not core._campaign["task"].done():
        raise HTTPException(409, "Pause the batch before deleting it")
    for p in (path, cm.state_path(path), _meta_path(path)):
        p.unlink(missing_ok=True)
    return {"ok": True}


async def scheduler_loop() -> None:
    """Starts scheduled batches when their time comes (checked every 20 s)."""
    while True:
        try:
            for path in cm.CAMPAIGN_DIR.glob("*.csv") if cm.CAMPAIGN_DIR.exists() else []:
                meta = _read_json(_meta_path(path))
                when = cm.parse_iso(meta.get("schedule_at")) if meta.get("status") == "scheduled" else None
                if when and when <= cm.now_ist():
                    task = core._campaign["task"]
                    if task and not task.done():
                        continue  # another batch is running; try again next round
                    logger.info(f"Scheduled batch '{meta.get('name', path.stem)}' starting")
                    start_batch(path, int(meta.get("concurrency", 1)))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Batch scheduler: {e}")
        await asyncio.sleep(20)


# ======================= AI Quality Assurance =======================
def _phone_calls(start: dt.date | None, end: dt.date | None) -> list[tuple[dt.datetime, Path, dict]]:
    out = []
    for path, rec in core._load_logs():
        if core._is_test(rec) or core._channel(rec) in core.TEXT_CHANNELS:
            continue
        when = core._when(rec, path)
        if (start and when.date() < start) or (end and when.date() > end):
            continue
        out.append((when, path, rec))
    return out


@router.get("/api/qa")
async def qa_dashboard(request: Request, start: str = "", end: str = ""):
    guard(request)
    try:
        d0 = dt.date.fromisoformat(start) if start else (cm.now_ist() - dt.timedelta(days=30)).date()
        d1 = dt.date.fromisoformat(end) if end else cm.now_ist().date()
    except ValueError:
        raise HTTPException(400, "Dates must be YYYY-MM-DD") from None
    calls = _phone_calls(d0, d1)
    scored, per_day = [], {}
    for when, path, rec in calls:
        s = qa.score_call(rec)
        if not s:
            continue
        scored.append({"file": path.name, "date": when.strftime("%Y-%m-%d %H:%M"), "candidate_name": rec.get("candidate_name", ""),
                       "outcome": (rec.get("outcome") or {}).get("call_outcome"), **s})
        per_day.setdefault(when.date().isoformat(), []).append(s["score"])
    trend = [{"date": d, "avg": round(sum(v) / len(v), 1), "calls": len(v)} for d, v in sorted(per_day.items())]
    avg = round(sum(r["score"] for r in scored) / len(scored), 2) if scored else None
    resolved = sum(1 for r in scored if r["resolved"])
    parts = {k: round(sum(r["parts"][k] for r in scored) / len(scored), 1) for k in ("result", "data", "speed", "reliability", "clean")} if scored else {}
    return {"start": d0.isoformat(), "end": d1.isoformat(), "total_calls": len(calls), "analysed": len(scored),
            "average_score": avg, "resolution_rate": round(100 * resolved / len(scored), 2) if scored else None,
            "resolved": resolved, "parts": parts, "trend": trend,
            "top_questions": qa.top_questions([rec for _, _, rec in calls]),
            "lowest": sorted(scored, key=lambda r: r["score"])[:8],
            "transfers": None}  # Priya never transfers calls


# ======================= Alerting =======================
DEFAULT_RULES = [
    {"id": "webhook_down", "name": "Public URL (ngrok) unreachable", "metric": "webhook_down", "threshold": 1, "enabled": True},
    {"id": "gemini_errors", "name": "Gemini errors in the last hour", "metric": "gemini_errors_hour", "threshold": 3, "enabled": True},
    {"id": "sheet_outbox", "name": "Sheet rows waiting to resend", "metric": "sheet_outbox", "threshold": 5, "enabled": True},
    {"id": "cpu", "name": "Server CPU busy (%)", "metric": "cpu_percent", "threshold": 85, "enabled": True},
    {"id": "batch_stopped", "name": "Batch stopped by an error", "metric": "batch_stopped", "threshold": 1, "enabled": True},
    {"id": "pickup_low", "name": "Pick-up rate today below (%), after 10 dials", "metric": "pickup_today", "threshold": 20, "enabled": False},
]


def alert_rules() -> list[dict]:
    saved = {r["id"]: r for r in (_yaml(ALERTS_FILE).get("rules") or []) if isinstance(r, dict) and r.get("id")}
    return [{**d, **{k: saved[d["id"]][k] for k in ("threshold", "enabled") if k in saved.get(d["id"], {})}} for d in DEFAULT_RULES]


def _metrics(status: dict) -> dict:
    hour_ago = time.time() - 3600
    gem = 0
    for path, rec in core._load_logs()[:200]:
        if path.stat().st_mtime < hour_ago:
            break
        gem += sum(1 for a in ((rec.get("outcome") or {}).get("actions") or []) if a.get("action") == "llm_error")
    today = cm.now_ist().date()
    ev = [e for e in core._events() if dt.datetime.fromtimestamp(float(e.get("t", 0)), cm.IST).date() == today]
    answered = sum(1 for e in ev if str(e.get("duration") or "0") not in ("0", "", "0.0"))
    camp = status.get("campaign") or {}
    stopped = camp.get("stopped_because") or ""
    return {"webhook_down": 1 if status.get("public_ok") is False else 0, "gemini_errors_hour": gem,
            "sheet_outbox": status.get("sheet_outbox_waiting", 0) or 0, "cpu_percent": status.get("cpu_percent") or 0,
            "batch_stopped": 1 if stopped and "dashboard" not in stopped else 0,
            "pickup_today": round(100 * answered / len(ev)) if len(ev) >= 10 else None}


def evaluate_alerts(status: dict) -> list[dict]:
    values = _metrics(status)
    out = []
    for r in alert_rules():
        v = values.get(r["metric"])
        if not r["enabled"] or v is None:
            firing = False
        elif r["metric"] == "pickup_today":
            firing = v < r["threshold"]
        else:
            firing = v >= r["threshold"]
        if firing != _alert_state.get(r["id"], False):
            _alert_state[r["id"]] = firing
            ALERT_HISTORY.insert(0, {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": r["name"],
                                     "state": "firing" if firing else "resolved", "value": v})
            del ALERT_HISTORY[100:]
            if firing:
                logger.warning(f"ALERT: {r['name']} (value {v}, threshold {r['threshold']})")
        out.append({**r, "value": v, "firing": firing})
    return out


@router.get("/api/alerts")
async def alerts(request: Request):
    guard(request)
    status = await core.api_status(request)
    return {"rules": evaluate_alerts(status), "history": ALERT_HISTORY}


@router.post("/api/alerts")
async def alerts_save(request: Request):
    guard(request)
    body = await request.json()
    rules = []
    known = {r["id"] for r in DEFAULT_RULES}
    for r in body.get("rules") or []:
        if r.get("id") in known:
            rules.append({"id": r["id"], "threshold": core._num(r.get("threshold", 0), "Threshold", 0, 100000),
                          "enabled": bool(r.get("enabled"))})
    _save_yaml(ALERTS_FILE, {"rules": rules}, "# Alert thresholds, edited from the console (MONITOR > Alerting).\n")
    return {"ok": True}


# ======================= Integrations =======================
async def _check(name: str, fn, ttl: int = 600, force: bool = False) -> dict:
    c = _checks.get(name)
    if c and not force and time.time() - c["t"] < ttl:
        return c
    try:
        status, detail = await fn()
    except Exception as e:  # noqa: BLE001
        status, detail = "error", str(e)[:160]
    _checks[name] = {"t": time.time(), "status": status, "detail": detail, "checked": time.strftime("%H:%M:%S")}
    return _checks[name]


async def _deepgram():
    key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not key:
        return "not_configured", "Add DEEPGRAM_API_KEY to .env"
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get("https://api.deepgram.com/v1/projects", headers={"Authorization": f"Token {key}"})
    region = (settings.get("deepgram") or {}).get("region", "in")
    return ("connected", f"Key valid · region {region}") if r.status_code == 200 else ("error", f"HTTP {r.status_code}")


async def _sarvam():
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        return "not_configured", "Optional: add SARVAM_API_KEY to .env for Indian-language voices and voice cloning"
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get("https://api.sarvam.ai/voices", params={"limit": 1}, headers={"api-subscription-key": key})
    from app.voice_config import plan

    p = plan(settings)
    use = "in use for this agent" if p["needs_sarvam"] else "not used by the live agent (English + Deepgram voice)"
    return ("connected", f"Key valid · {use}") if r.status_code == 200 else ("error", f"HTTP {r.status_code}: {r.text[:120]}")


async def _fast_llm():
    from app.llm_choice import llm_provider

    provider, model = llm_provider()
    urls = {"groq": "https://api.groq.com/openai/v1/models", "cerebras": "https://api.cerebras.ai/v1/models",
            "openai": "https://api.openai.com/v1/models"}
    key_names = {"groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY", "openai": "OPENAI_API_KEY"}
    if provider == "anthropic":
        ok = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
        return ("connected", f"live calls use Claude ({model})") if ok else ("error", "ANTHROPIC_API_KEY missing in .env")
    p = provider if provider in urls else "groq"
    key = os.getenv(key_names[p], "").strip()
    if not key:
        return "not_configured", "Optional: GROQ_API_KEY in .env for the fast call LLM (console.groq.com, free tier)"
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(urls[p], headers={"Authorization": f"Bearer {key}"})
    use = f"live calls use {provider}:{model}" if provider != "google" else "key valid, live calls still use Gemini"
    return ("connected", use) if r.status_code == 200 else ("error", f"{p} HTTP {r.status_code}")


async def _gemini():
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        return "not_configured", "Add GOOGLE_API_KEY to .env"
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get("https://generativelanguage.googleapis.com/v1beta/models", headers={"x-goog-api-key": key})
    model = (settings.get("llm") or {}).get("model", "")
    return ("connected", f"Key valid · {model} · {gemini_tier()} tier") if r.status_code == 200 else ("error", f"HTTP {r.status_code}")


async def _vobiz():
    if not os.getenv("VOBIZ_AUTH_ID") or not os.getenv("VOBIZ_AUTH_TOKEN"):
        return "not_configured", "Add VOBIZ_AUTH_ID and VOBIZ_AUTH_TOKEN to .env"
    core._vobiz["t"] = 0
    ok, detail = await core._vobiz_ok()
    return ("connected" if ok else "error"), f"{detail} · from {os.getenv('VOBIZ_FROM_NUMBER', '?')}"


async def _sheet():
    if not sheet_sync.enabled():
        return "not_configured", "Add SHEET_WEBHOOK_URL and SHEET_WEBHOOK_SECRET to .env"
    waiting = (core.status_provider() or {}).get("sheet_outbox_waiting", 0) if core.status_provider else 0
    return "connected", "Webhook configured" + (f" · {waiting} rows waiting" if waiting else "")


async def _ngrok():
    core._public["t"] = 0
    ok, url = await core._public_ok()
    return ("connected" if ok else "error"), url or "PUBLIC_BASE_URL not set"


INTEGRATIONS = [
    ("vobiz", "Vobiz", "Telephony · SIP trunk and outbound calls", _vobiz),
    ("deepgram", "Deepgram", "Speech-to-text (Nova-3) and voice (Aura-2)", _deepgram),
    ("sarvam", "Sarvam AI", "Indian-language speech-to-text, Bulbul voices, voice cloning", _sarvam),
    ("fastllm", "Fast call LLM (Groq / Cerebras / OpenAI)", "Live-call replies when selected in Agents", _fast_llm),
    ("gemini", "Google Gemini", "Conversation and post-call extraction", _gemini),
    ("sheets", "Google Sheets", "One row per call via Apps Script webhook", _sheet),
    ("ngrok", "ngrok", "Public HTTPS address Vobiz calls back", _ngrok),
]


@router.get("/api/integrations")
async def integrations(request: Request, refresh: bool = False):
    guard(request)
    out = []
    for key, name, desc, fn in INTEGRATIONS:
        c = await _check(key, fn, force=refresh)
        out.append({"id": key, "name": name, "description": desc, **{k: c[k] for k in ("status", "detail", "checked")}})
    return {"integrations": out}


# ======================= Billing (usage from your own logs) =======================
DEFAULT_RATES = {"vobiz_inr_per_min": None, "vobiz_platform_fee_inr_per_month": 500, "deepgram_usd_per_min": None,
                 "gemini_inr_per_call": None, "usd_to_inr": None}


def billing_rates() -> dict:
    return {**DEFAULT_RATES, **{k: v for k, v in _yaml(BILLING_FILE).items() if k in DEFAULT_RATES}}


@router.get("/api/billing")
async def billing(request: Request):
    guard(request)
    rates = billing_rates()
    months: dict[str, dict] = {}
    for when, _, rec in _phone_calls(None, None):
        m = months.setdefault(when.strftime("%Y-%m"), {"month": when.strftime("%Y-%m"), "calls": 0, "seconds": 0, "billed_minutes": 0})
        secs = float(rec.get("duration_secs") or 0)
        m["calls"] += 1
        m["seconds"] += secs
        m["billed_minutes"] += math.ceil(secs / 60) if secs else 0
    rows = []
    for m in sorted(months.values(), key=lambda m: m["month"], reverse=True):
        cost = {}
        if rates["vobiz_inr_per_min"] is not None:
            cost["vobiz"] = round(m["billed_minutes"] * float(rates["vobiz_inr_per_min"]) + float(rates["vobiz_platform_fee_inr_per_month"] or 0), 2)
        if rates["deepgram_usd_per_min"] is not None and rates["usd_to_inr"]:
            cost["deepgram"] = round(m["seconds"] / 60 * float(rates["deepgram_usd_per_min"]) * float(rates["usd_to_inr"]), 2)
        if rates["gemini_inr_per_call"] is not None:
            cost["gemini"] = round(m["calls"] * float(rates["gemini_inr_per_call"]), 2)
        rows.append({**m, "minutes": round(m["seconds"] / 60, 1), "cost_inr": cost, "total_inr": round(sum(cost.values()), 2) if cost else None})
    return {"months": rows, "rates": rates, "gemini_tier": gemini_tier(),
            "plan": "₹0 self-hosted · Gemini " + gemini_tier() + " tier · Deepgram credit · ngrok free"}


# ======================= Settings =======================
@router.get("/api/settings")
async def settings_get(request: Request):
    guard(request)
    env = {k: bool(os.getenv(k, "").strip()) for k in (
        "DEEPGRAM_API_KEY", "GOOGLE_API_KEY", "VOBIZ_AUTH_ID", "VOBIZ_AUTH_TOKEN", "VOBIZ_FROM_NUMBER", "PUBLIC_BASE_URL",
        "SHEET_WEBHOOK_URL", "SHEET_WEBHOOK_SECRET", "ADMIN_TOKEN", "SARVAM_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "DASHBOARD_USER", "DASHBOARD_ALLOW_REMOTE")}
    return {**console_cfg(), "gemini_tier": gemini_tier(), "rates": billing_rates(), "env": env,
            "retention_days": (settings.get("data") or {}).get("retention_days", 90),
            "allow_remote": os.getenv("DASHBOARD_ALLOW_REMOTE", "").strip() == "1"}


@router.post("/api/settings")
async def settings_save(request: Request):
    guard(request)
    body = await request.json()
    c = _yaml(CONSOLE_FILE)
    if "workspace_name" in body:
        c["workspace_name"] = core._clean_str(body["workspace_name"], "Workspace name", 40)
    if "user_label" in body:
        c["user_label"] = core._clean_str(body["user_label"], "Your name or email", 80)
    _save_yaml(CONSOLE_FILE, c, "# Console display settings (SYSTEM > Settings).\n")
    if "gemini_tier" in body:
        tier = "paid" if body["gemini_tier"] == "paid" else "free"
        core._write_yaml(core.CAMPAIGN_FILE, [(None, "gemini_tier", tier)] + ([(None, "max_concurrent", 1)] if tier == "free" else []))
    if "rates" in body:
        rates = {}
        for k in DEFAULT_RATES:
            v = (body["rates"] or {}).get(k)
            rates[k] = None if v in (None, "") else core._num(v, k, 0, 1_000_000)
        _save_yaml(BILLING_FILE, rates, "# Your own rates, used only to estimate costs on the Billing page.\n")
    if "retention_days" in body:
        days = core._num(body["retention_days"], "Keep call logs for (days)", 7, 3650, True)
        new = core._write_yaml(core.CONFIG_FILE, [("data", "retention_days", days)])
        settings.clear()
        settings.update(new)
    return {"ok": True}


# ======================= Getting started =======================
def onboarding(status: dict) -> list[dict]:
    has_kb = any(k.get("source_count") for k in kb_store.list_kbs())
    calls = [rec for _, rec in core._load_logs()[:300] if not core._is_test(rec) and core._channel(rec) not in core.TEXT_CHANNELS]
    batch_run = False
    for p in cm.CAMPAIGN_DIR.glob("*.state.json") if cm.CAMPAIGN_DIR.exists() else []:
        st = _read_json(p)
        if any(c.get("dials") or c.get("attempts") for c in (st.get("candidates") or {}).values()):
            batch_run = True
            break
    return [
        {"id": "keys", "label": "Add Deepgram and Gemini keys", "done": bool(os.getenv("DEEPGRAM_API_KEY") and os.getenv("GOOGLE_API_KEY"))},
        {"id": "vobiz", "label": "Connect your Vobiz number", "done": bool(os.getenv("VOBIZ_AUTH_ID") and os.getenv("VOBIZ_FROM_NUMBER"))},
        {"id": "webhook", "label": "Start ngrok (public URL reachable)", "done": status.get("public_ok") is True},
        {"id": "sheet", "label": "Connect the Google Sheet", "done": sheet_sync.enabled()},
        {"id": "kb", "label": "Add a knowledge source", "done": has_kb},
        {"id": "call", "label": "Make your first test call", "done": bool(calls)},
        {"id": "batch", "label": "Launch your first batch call", "done": batch_run},
    ]


async def status_extras(status: dict) -> dict:
    steps = onboarding(status)
    firing = [r["name"] for r in evaluate_alerts(status) if r["firing"]]
    return {**console_cfg(), "gemini_tier": gemini_tier(), "onboarding": steps,
            "onboarding_done": sum(1 for s in steps if s["done"]), "alerts_firing": firing}


# ======================= Webhook test (Agents > Webhook settings > Test) =======================
WEBHOOK_TEST_ID = "console-webhook-test"  # always the same id, so repeated tests update one row


@router.post("/api/webhook/test")
async def webhook_test(request: Request):
    guard(request)
    if not sheet_sync.enabled():
        raise HTTPException(400, "Set SHEET_WEBHOOK_URL and SHEET_WEBHOOK_SECRET in .env, then restart the server")
    record = {
        "call_id": WEBHOOK_TEST_ID, "candidate_name": "[TEST] Console webhook test", "phone": "",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_secs": 0,
        "outcome": {"call_outcome": "completed", "data": {}, "summary": "TEST ROW from the console. Safe to delete.",
                    "recruiter_notes": "", "checks": [], "ended_by": "agent"},
        "latency": {}, "transcript": [],
    }
    started = time.monotonic()
    try:
        body = await sheet_sync._post(sheet_sync.build_row(record))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:300]}
    return {"ok": True, "ms": round((time.monotonic() - started) * 1000), "action": body.get("action", "")}
