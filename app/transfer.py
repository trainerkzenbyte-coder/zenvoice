"""Call transfer to a human recruiter (Agents -> Functions -> transfer_call).

How it works on a Vobiz call:
    1. Priya says the hand-off line ("Please stay on the line, I'm connecting you ...").
    2. When she has finished speaking, we ask Vobiz to transfer the candidate's leg:
       POST /Account/{id}/Call/{uuid}/  {"legs": "aleg", "aleg_url": <our /transfer/<token> XML>}
       (https://vobiz.ai/docs/call/transfer-call). That replaces the audio stream, so our
       pipeline ends WITHOUT hanging up.
    3. Vobiz fetches /transfer/<token>: <Dial> the recruiter's number.
         cold  = the recruiter just hears the call ring.
         warm  = when the recruiter answers, they first hear a short spoken briefing
                 (confirmSound -> /transfer/<token>/whisper): candidate name, role, what
                 they asked. Then the call is bridged.
    4. /transfer/<token>/done (Dial "action"): if the recruiter didn't answer, the candidate
       hears "our recruiter will call you back" and the call ends.

The /transfer URLs must be public (Vobiz calls them through ngrok); each one only works with a
random one-time token that expires after 15 minutes.
"""

from __future__ import annotations

import secrets
import time
from html import escape

from fastapi import APIRouter, Request
from fastapi.responses import Response
from loguru import logger


router = APIRouter()
TOKEN_TTL = 15 * 60
_TRANSFERS: dict[str, dict] = {}

from app.transfer_cfg import DEFAULTS, available, cfg  # noqa: E402,F401


def _cleanup():
    now = time.time()
    for t in [t for t, v in _TRANSFERS.items() if v["expires"] < now]:
        _TRANSFERS.pop(t, None)


def create(call_id: str, candidate_name: str, role: str, summary: str, reason: str) -> str:
    _cleanup()
    token = secrets.token_urlsafe(18)
    _TRANSFERS[token] = {"call_id": call_id, "name": candidate_name, "role": role, "summary": summary,
                         "reason": reason, "expires": time.time() + TOKEN_TTL, "cfg": cfg()}
    return token


def briefing(t: dict) -> str:
    who = t["name"] or "a candidate"
    role = f" for the {t['role']} role" if t["role"] else ""
    about = f" They said: {t['summary']}." if t["summary"] else ""
    return f"Zenbyte call transfer. You are being connected to {who}{role}.{about} Connecting now."


def _xml(body: str) -> Response:
    return Response(f'<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n{body}\n</Response>\n',
                    media_type="application/xml")


@router.api_route("/transfer/{token}", methods=["GET", "POST"])
async def transfer_xml(token: str, request: Request):
    from app import vobiz

    t = _TRANSFERS.get(token)
    if not t or t["expires"] < time.time():
        return _xml("  <Hangup/>")
    c = t["cfg"]
    base = vobiz.public_base_url()
    number = vobiz.normalize_number(str(c["number"]))
    caller = vobiz.normalize_number(__import__("os").getenv("VOBIZ_FROM_NUMBER", ""))
    attrs = [f'callerId="{escape(caller)}"', f'timeout="{int(c["ring_timeout_secs"])}"',
             f'action="{escape(f"{base}/transfer/{token}/done")}"', 'method="POST"']
    if c["mode"] == "warm":
        attrs += [f'confirmSound="{escape(f"{base}/transfer/{token}/whisper")}"', 'confirmTimeout="10"']
    logger.info(f"[{t['call_id'][-6:]}] Transfer: dialling recruiter {number[:4]}…{number[-3:]} ({c['mode']})")
    return _xml(f"  <Dial {' '.join(attrs)}>\n    <Number>{escape(number)}</Number>\n  </Dial>")


@router.api_route("/transfer/{token}/whisper", methods=["GET", "POST"])
async def transfer_whisper(token: str):
    t = _TRANSFERS.get(token)
    if not t:
        return _xml("  <Wait length=\"1\"/>")
    return _xml(f"  <Speak>{escape(briefing(t), quote=False)}</Speak>")


@router.api_route("/transfer/{token}/done", methods=["GET", "POST"])
async def transfer_done(token: str, request: Request):
    t = _TRANSFERS.pop(token, None)
    try:
        form = dict(await request.form())
    except Exception:  # noqa: BLE001
        form = dict(request.query_params)
    status = str(form.get("DialStatus") or form.get("DialBLegStatus") or "").lower()
    label = t["call_id"][-6:] if t else "??????"
    logger.info(f"[{label}] Transfer result: {status or 'unknown'}")
    if status in ("completed", "answer", "answered"):
        return _xml("  <Hangup/>")  # the recruiter talked to the candidate; call is over
    line = (t["cfg"] if t else DEFAULTS)["unavailable"]
    return _xml(f"  <Speak>{escape(line, quote=False)}</Speak>\n  <Hangup/>")
