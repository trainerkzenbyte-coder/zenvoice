"""Web server: Vobiz talks to this (laptop with ngrok, or the cloud server).

Run from the project folder (virtual environment active):
    python -m app.server
"""

import asyncio
import hmac
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from loguru import logger

try:
    import psutil
except ImportError:
    psutil = None

from app import call_library, calls_api, dashboard, dashboard_ext, housekeeping, live, sheet_sync, transfer, vobiz, voices
from app.call_actions import LLM_ERROR_STATS, CallContext
from app.config import ROOT
from app.phone_bot import run_phone_bot

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> {message}")

(Path(ROOT) / "logs").mkdir(exist_ok=True)
logger.add(
    str(Path(ROOT) / "logs" / "server.log"),
    level="INFO",
    rotation="10 MB",
    retention="14 days",
    format="{time:YYYY-MM-DD HH:mm:ss} {message}",
    enqueue=True,
)

ACTIVE_CALLS: set[str] = set()
STATS = {"started": time.time(), "calls_total": 0, "calls_failed": 0, "peak_concurrent": 0}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if sheet_sync.enabled():
        sent, left = await sheet_sync.flush_outbox()
        logger.info(
            "Google Sheet sync: on"
            + (f" ({sent} queued row(s) resent, {left} waiting)" if sent or left else "")
        )
    else:
        logger.info("Google Sheet sync: off (set SHEET_WEBHOOK_URL and SHEET_WEBHOOK_SECRET in .env)")
    task = asyncio.create_task(housekeeping.run_forever())
    scheduler = asyncio.create_task(dashboard_ext.scheduler_loop())
    yield
    task.cancel()
    scheduler.cancel()


app = FastAPI(title="Voice agent", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


async def _params(request: Request) -> dict:
    try:
        if "json" in request.headers.get("content-type", ""):
            return await request.json()
        return dict(await request.form())
    except Exception:
        return dict(request.query_params)


@app.get("/health")
async def health():
    return PlainTextResponse("ok")


def _authorized(request: Request) -> bool:
    from app import auth

    who = auth.identify(request)
    return bool(who) and who["role"] == "admin"


@app.get("/status")
async def status(request: Request):
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return _status_dict()


# /api/live and /api/status come from app/dashboard.py (local-only + ADMIN_TOKEN). Do not add
# unprotected copies here: routes declared here win, and would publish candidates' names, numbers
# and transcripts to anyone who finds the ngrok address.


def _listener_allowed(websocket: WebSocket) -> bool:
    """Live audio is candidate PII: same rule as the console (this computer only, unless
    DASHBOARD_ALLOW_REMOTE=1) plus ADMIN_TOKEN, sent as ?token= because browsers can't set
    headers on a WebSocket."""
    if os.getenv("DASHBOARD_ALLOW_REMOTE", "").strip() != "1":
        host = websocket.headers.get("host", "").rsplit(":", 1)[0].strip("[]").lower()
        forwarded = any(h in websocket.headers for h in ("x-forwarded-for", "x-forwarded-host", "forwarded"))
        client = websocket.client.host if websocket.client else ""
        if host not in ("127.0.0.1", "localhost", "::1") or forwarded or client not in ("127.0.0.1", "::1", "localhost"):
            return False
    from app import auth  # any signed-in recruiter or admin may listen live

    return auth.identify_token(websocket.query_params.get("token", ""), websocket) is not None


@app.websocket("/ws/listen/{call_id}")
async def listen_live_audio(websocket: WebSocket, call_id: str):
    if not _listener_allowed(websocket):
        await websocket.close(code=1008)
        return
    if not live.is_live(call_id):
        await websocket.close(code=4404)  # call already ended
        return
    await websocket.accept()
    q = live.add_audio_listener(call_id)
    try:
        while True:
            packet = await q.get()
            if packet == live.END_OF_CALL:
                await websocket.close(code=1000)
                break
            await websocket.send_bytes(packet)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        live.remove_audio_listener(call_id, q)



def _status_dict() -> dict:
    calls = live.snapshot()
    return {
        "active_calls": len(ACTIVE_CALLS),
        "live_calls": calls,
        "calls": calls,
        "calls_since_start": STATS["calls_total"],
        "calls_crashed": STATS["calls_failed"],
        "peak_concurrent": STATS["peak_concurrent"],
        "uptime_minutes": round((time.time() - STATS["started"]) / 60),
        **_machine_stats(),
        "llm_errors": LLM_ERROR_STATS["total"],
        "last_llm_error": LLM_ERROR_STATS["last"],
        "sheet_sync": sheet_sync.enabled(),
        "sheet_outbox_waiting": _outbox_rows(),
    }


from app import auth  # noqa: E402  Sprint 12: recruiter logins

auth.install(app)                # role check + audit log for every /api call
app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(dashboard_ext.router)
app.include_router(calls_api.router)
app.include_router(voices.router)
app.include_router(transfer.router)
app.include_router(call_library.router)  # Sprint 11: Call Library (recruiter calls -> style guide)
dashboard.status_provider = _status_dict


def _outbox_rows() -> int:
    outbox = sheet_sync.OUTBOX
    if not outbox.exists():
        return 0
    with outbox.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def _machine_stats() -> dict:
    out: dict = {"cpu_count": os.cpu_count()}
    if hasattr(os, "getloadavg"):
        out["load_1min"] = round(os.getloadavg()[0], 2)
    if psutil is not None:
        out["cpu_percent"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        out["memory_used_mb"] = round((vm.total - vm.available) / 1048576)
        out["memory_total_mb"] = round(vm.total / 1048576)
        return out
    return out


@app.api_route("/answer", methods=["GET", "POST"])
async def answer(request: Request):
    p = await _params(request)
    ctx = vobiz.decode_context(request.query_params.get("ctx"))
    phone = ctx["phone"] or str(p.get("To", ""))
    logger.info(
        f"[{str(p.get('CallUUID', ''))[-6:]}] Answered: {ctx['candidate_name'] or 'candidate'} "
        f"({phone or '?'}), status {p.get('CallStatus', p.get('Status', '?'))}"
    )
    token = vobiz.encode_context(ctx["candidate_name"], phone, ctx.get("ref", ""), ctx.get("role", ""))
    return Response(content=vobiz.answer_xml(token), media_type="application/xml")


@app.api_route("/hangup", methods=["GET", "POST"])
async def hangup(request: Request):
    p = await _params(request)
    logger.info(
        f"[{str(p.get('CallUUID', ''))[-6:]}] Vobiz hang-up report: "
        f"duration {p.get('Duration', '?')} s, cause {p.get('HangupCause', p.get('Status', '?'))}"
    )
    event = {
        "t": time.time(),
        "ref": request.query_params.get("ref", ""),
        "call_uuid": str(p.get("CallUUID", "")),
        "to": str(p.get("To", "")),
        "duration": str(p.get("Duration", "")),
        "bill_duration": str(p.get("BillDuration", "")),
        "cause": str(p.get("HangupCause", p.get("Status", ""))),
        "status": str(p.get("CallStatus", "")),
    }
    try:
        with (Path(ROOT) / "logs" / "call_events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except OSError as e:
        logger.warning(f"Could not record hang-up event: {e}")
    return PlainTextResponse("ok")


@app.api_route("/stream-status", methods=["GET", "POST"])
async def stream_status(request: Request):
    p = await _params(request)
    event = p.get("Event") or p.get("event") or "?"
    logger.info(f"[{str(p.get('CallUUID', ''))[-6:]}] Stream status: {event}")
    return PlainTextResponse("ok")


@app.websocket("/ws/{token}")
@app.websocket("/ws/{token}/{ctx}")
async def media_stream(websocket: WebSocket, token: str, ctx: str = ""):
    if token != vobiz.stream_token():
        logger.warning("Rejected a WebSocket with a wrong token")
        await websocket.close(code=1008)
        return
    await websocket.accept()

    ids = None
    try:
        for _ in range(5):
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10)
            ids = vobiz.parse_start(json.loads(raw))
            if ids:
                break
    except (asyncio.TimeoutError, WebSocketDisconnect, json.JSONDecodeError) as e:
        logger.warning(f"No valid start message from Vobiz: {e!r}")
    if not ids:
        await websocket.close()
        return

    call_id, stream_id = ids
    ACTIVE_CALLS.add(call_id or stream_id)
    STATS["calls_total"] += 1
    STATS["peak_concurrent"] = max(STATS["peak_concurrent"], len(ACTIVE_CALLS))
    logger.info(f"[{(call_id or stream_id)[-6:]}] Audio stream open ({len(ACTIVE_CALLS)} active call(s))")
    try:
        info = vobiz.decode_context(ctx)
        candidate = CallContext(
            candidate_name=info["candidate_name"],
            phone=info["phone"],
            ref=info.get("ref", ""),
            role=info.get("role", ""),
        )
        await run_phone_bot(websocket, call_id, stream_id, candidate)
    except Exception as e:
        logger.exception(f"Call {call_id} crashed: {e}")
        STATS["calls_failed"] += 1
        if not (call_id or "").startswith("loadtest-"):
            await vobiz.hang_up(call_id)
    finally:
        ACTIVE_CALLS.discard(call_id or stream_id)


def main():
    url = vobiz.public_base_url()
    host = os.getenv("HOST", "127.0.0.1")
    logger.info(f"Server starting on {host}:8000  (public: {url})")
    logger.info("To place a call: python scripts/make_call.py <mobile> --name <name>")
    logger.info("Dashboard: http://127.0.0.1:8000/  (sign in with ADMIN_TOKEN from .env)")
    uvicorn.run(app, host=host, port=8000, log_level="warning", ws_ping_interval=20, ws_ping_timeout=20)


if __name__ == "__main__":
    main()