"""Everything Vobiz-specific lives here: place a call, hang up, answer XML, audio connector.



Vobiz API reference: https://vobiz.ai/docs/call/make-call

Streaming protocol:  https://vobiz.ai/docs/concepts/streaming-websockets



Audio connector: Vobiz's WebSocket protocol (start / media / playAudio /

clearAudio / dtmf) matches Plivo's, so we reuse Pipecat's maintained

PlivoFrameSerializer for audio and barge-in. Only the hang-up address differs,

so its built-in auto hang-up is switched off and we hang up via Vobiz below.

"""



import base64

import hashlib

import json

from urllib.parse import quote, urlparse

from xml.sax.saxutils import escape



import httpx

from loguru import logger



from pipecat.serializers.plivo import PlivoFrameSerializer



from app.config import secret



API_BASE = "https://api.vobiz.ai/api/v1"





def _auth() -> tuple[str, str]:

    return secret("VOBIZ_AUTH_ID"), secret("VOBIZ_AUTH_TOKEN")





def _headers() -> dict:

    auth_id, auth_token = _auth()

    return {"X-Auth-ID": auth_id, "X-Auth-Token": auth_token, "Content-Type": "application/json"}





def public_base_url() -> str:

    """Public HTTPS address of this server (your ngrok static domain while developing)."""

    url = secret("PUBLIC_BASE_URL").rstrip("/")

    if not url.startswith("https://"):

        raise RuntimeError("PUBLIC_BASE_URL must start with https:// (your ngrok domain)")

    return url





def stream_token() -> str:

    """Secret token in the WebSocket URL so only Vobiz (via our answer XML) can connect.

    Derived from your Vobiz token, so there is no extra secret to manage."""

    return hashlib.sha256(("ws:" + _auth()[1]).encode()).hexdigest()[:32]





def normalize_number(number: str) -> str:

    """'98765 43210' / '+91 98765-43210' / '919876543210' -> '919876543210' (E.164 digits)."""

    digits = "".join(ch for ch in number if ch.isdigit())

    if len(digits) == 10:  # Indian mobile without country code

        digits = "91" + digits

    if len(digits) == 11 and digits.startswith("0"):  # 0 + 10 digits

        digits = "91" + digits[1:]

    if len(digits) < 11:

        raise ValueError(f"'{number}' does not look like a full phone number")

    return digits





# ---------- Candidate context (travels with the call: make_call -> /answer -> WebSocket) ----------

def encode_context(candidate_name: str = "", phone: str = "", ref: str = "", role: str = "") -> str:

    """ref: campaign reference (Sprint 6) so the call log and hang-up report can be matched.

    role: job title for this call only (dashboard dialer); empty = job.title from screening.yaml."""

    d = {"n": candidate_name.strip()[:60], "p": phone}

    if ref:

        d["r"] = ref[:40]

    if role.strip():

        d["j"] = role.strip()[:80]

    raw = json.dumps(d, separators=(",", ":"))

    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")





def decode_context(token: str | None) -> dict:

    if not token:

        return {"candidate_name": "", "phone": "", "ref": "", "role": ""}

    try:

        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()

        d = json.loads(raw)

        return {"candidate_name": str(d.get("n", ""))[:60], "phone": str(d.get("p", ""))[:20],

                "ref": str(d.get("r", ""))[:40], "role": str(d.get("j", ""))[:80]}

    except Exception:  # noqa: BLE001  (bad context never blocks a call)

        return {"candidate_name": "", "phone": "", "ref": "", "role": ""}





# ---------- REST API ----------

async def make_call(

    to: str, *, time_limit_secs: int, ring_timeout_secs: int = 30, candidate_name: str = "",

    ref: str = "", role: str = "",

) -> dict:

    auth_id, _ = _auth()

    base = public_base_url()

    to_number = normalize_number(to)

    ctx = encode_context(candidate_name, to_number, ref, role)

    body = {

        "from": normalize_number(secret("VOBIZ_FROM_NUMBER")),

        "to": to_number,

        "answer_url": f"{base}/answer?ctx={quote(ctx)}",

        "answer_method": "POST",

        "hangup_url": f"{base}/hangup" + (f"?ref={quote(ref)}" if ref else ""),

        "hangup_method": "POST",

        "time_limit": time_limit_secs,          # Vobiz cuts the call after this (backstop)

        "ring_timeout": ring_timeout_secs,    # stop ringing after this many seconds

    }

    async with httpx.AsyncClient(timeout=20) as client:

        r = await client.post(f"{API_BASE}/Account/{auth_id}/Call/", headers=_headers(), json=body)

    if r.status_code >= 300:

        raise RuntimeError(f"Vobiz rejected the call (HTTP {r.status_code}): {r.text[:300]}")

    return r.json()





async def hang_up(call_uuid: str) -> None:

    """Hang up a live call. Safe to call twice (404 = already ended)."""

    if not call_uuid:

        return

    auth_id, _ = _auth()

    try:

        async with httpx.AsyncClient(timeout=10) as client:

            r = await client.delete(

                f"{API_BASE}/Account/{auth_id}/Call/{call_uuid}/", headers=_headers()

            )

        if r.status_code in (200, 202, 204, 404):

            logger.debug(f"Hang-up {call_uuid}: HTTP {r.status_code}")

        else:

            logger.warning(f"Hang-up {call_uuid} failed: HTTP {r.status_code} {r.text[:200]}")

    except httpx.HTTPError as e:

        logger.warning(f"Hang-up {call_uuid} failed: {e}")





async def transfer(call_uuid: str, aleg_url: str) -> None:

    """Replace the live call's XML (our audio stream) with aleg_url's XML, e.g. <Dial> a recruiter.

    https://vobiz.ai/docs/call/transfer-call  -- the call must be in progress."""

    auth_id, _ = _auth()

    async with httpx.AsyncClient(timeout=10) as client:

        r = await client.post(f"{API_BASE}/Account/{auth_id}/Call/{call_uuid}/", headers=_headers(),

                              json={"legs": "aleg", "aleg_url": aleg_url, "aleg_method": "POST"})

    if r.status_code >= 300:

        raise RuntimeError(f"Vobiz transfer failed (HTTP {r.status_code}): {r.text[:200]}")





# ---------- Answer XML ----------

def answer_xml(ctx: str | None = None) -> str:

    """Tells Vobiz: open a two-way audio stream to our WebSocket for this call.

    ctx (candidate name + phone) rides in the WebSocket path so the bot knows who it called."""

    base = public_base_url()

    host = urlparse(base).netloc

    ws_url = f"wss://{host}/ws/{stream_token()}" + (f"/{quote(ctx)}" if ctx else "")

    status_url = f"{base}/stream-status"

    return (

        '<?xml version="1.0" encoding="UTF-8"?>\n'

        "<Response>\n"

        f'  <Stream bidirectional="true" streamTimeout="1800" audioTrack="inbound" keepCallAlive="true" '

        f'contentType="audio/x-mulaw;rate=8000" '

        f'statusCallbackUrl="{escape(status_url)}" statusCallbackMethod="POST">'

        f"{escape(ws_url)}</Stream>\n"

        "</Response>\n"

    )





# ---------- Audio connector ----------

def make_serializer(stream_id: str) -> PlivoFrameSerializer:

    """8 kHz mu-law phone audio <-> Pipecat audio (resampled to the pipeline's 16 kHz)."""

    params = PlivoFrameSerializer.InputParams(plivo_sample_rate=8000, auto_hang_up=False)

    return PlivoFrameSerializer(stream_id=stream_id, params=params)





def parse_start(message: dict) -> tuple[str, str] | None:

    """Return (call_id, stream_id) from Vobiz's first WebSocket 'start' message."""

    if message.get("event") != "start":

        return None

    start = message.get("start", {}) or {}

    call_id = start.get("callId") or start.get("callUUID") or start.get("call_uuid")

    stream_id = start.get("streamId") or start.get("streamSid") or start.get("stream_id")

    if not stream_id:

        return None

    return call_id or "", stream_id

