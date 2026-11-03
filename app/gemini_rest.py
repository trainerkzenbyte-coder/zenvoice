"""Small direct client for Gemini's REST API (used outside the live voice pipeline).

Used for post-call extraction, the text chat simulator and automated call-flow
tests. The live call itself uses Pipecat's Gemini service.
"""

import asyncio
import json
import re
import time

import httpx
from loguru import logger

from app.config import secret, settings

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 502, 503, 504}   # worth trying again; anything else is not
BACKOFF_SECS = [2, 5, 10, 20, 30]          # used when Gemini doesn't say how long to wait


# ---------- explaining Gemini errors in plain words ----------
def _quota_details(text: str) -> tuple[str, str]:
    qid = re.search(r"quotaId['\"]?\s*:\s*['\"]([\w\-]+)", text)
    qval = re.search(r"quotaValue['\"]?\s*:\s*['\"]?(\d+)", text)
    if not qval:  # "... limit: 15, model: ..." in the human-readable message
        qval = re.search(r"limit:\s*(\d+)", text)
    return (qid.group(1) if qid else ""), (qval.group(1) if qval else "")


def describe_error(text: str) -> str:
    """One readable line for a Gemini error (from Pipecat or this REST client)."""
    t = text or ""
    low = t.lower()
    if "429" in t or "resource_exhausted" in low or "quota" in low:
        quota_id, limit = _quota_details(t)
        free = "freetier" in quota_id.lower() or "free_tier" in low
        who = "Gemini free tier" if free else "Gemini"
        per = ("per minute" if "perminute" in quota_id.lower() else
               "per day" if "perday" in quota_id.lower() else "")
        what = "input tokens" if "token" in quota_id.lower() else "requests"
        if limit:
            return f"rate limit reached: {who} allows {limit} {what} {per}".strip()
        return f"rate limit reached (429) on {who}"
    if "503" in t or "unavailable" in low or "overloaded" in low:
        return "Gemini is overloaded (503); usually clears within seconds"
    if "timeout" in low or "timed out" in low:
        return "Gemini did not answer in time"
    if "401" in t or "403" in t or "api key" in low or "permission_denied" in low:
        return "Gemini refused the API key (check GOOGLE_API_KEY in .env)"
    if "404" in t and "model" in low:
        return "Gemini model not found (check llm.model in config/agent.yaml)"
    if "500" in t or "internal" in low:
        return "Gemini internal error (500)"
    if "400" in t or "invalid_argument" in low:
        return f"Gemini rejected the request (400): {t[:160]}"
    return t[:200] or "unknown error"


def _retry_after(r: httpx.Response) -> float | None:
    """Seconds Gemini asks us to wait (RetryInfo 'retryDelay' or the Retry-After header)."""
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", r.text) or re.search(
        r"retry in (\d+(?:\.\d+)?)\s*s", r.text, re.IGNORECASE
    )
    if m:
        return float(m.group(1))
    try:
        return float(r.headers.get("retry-after", ""))
    except ValueError:
        return None


async def _post(client: httpx.AsyncClient, url: str, body: dict, max_wait_secs: float) -> httpx.Response:
    """POST, retrying rate limits, overloads and network errors for up to max_wait_secs."""
    deadline = time.monotonic() + max_wait_secs
    attempt = 0
    while True:
        try:
            r = await client.post(url, headers={"x-goog-api-key": secret("GOOGLE_API_KEY")}, json=body)
            if r.status_code not in RETRY_STATUS:
                return r
            problem, wait = describe_error(f"{r.status_code} {r.text}"), _retry_after(r)
        except httpx.TransportError as e:  # network blip, DNS, read timeout
            r, problem, wait = None, f"network error: {e!r}"[:120], None
        if wait is None:
            wait = BACKOFF_SECS[min(attempt, len(BACKOFF_SECS) - 1)]
        wait = min(wait + 0.5, 60.0)
        if time.monotonic() + wait > deadline:
            if r is not None:
                return r  # caller turns it into a clear error
            raise RuntimeError(f"Gemini unreachable: {problem}")
        attempt += 1
        logger.info(f"Gemini busy ({problem}); retrying in {wait:.0f} s")
        await asyncio.sleep(wait)


def _thinking_variants(cfg: dict) -> list[dict | None]:
    """Gemini 2.5: thinkingBudget. Gemini 3.x: thinkingLevel (some API versions want upper case)."""
    if cfg.get("thinking_budget") is not None:
        return [{"thinkingBudget": int(cfg["thinking_budget"])}, None]
    level = str(cfg.get("thinking_level", "minimal")).lower()
    if level == "none":
        return [None]
    return [{"thinkingLevel": level}, {"thinkingLevel": level.upper()}, None]


def tool_declarations(specs: list[dict]) -> list[dict]:
    """Our TOOL_SPECS -> Gemini REST functionDeclarations."""
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "parameters": {"type": "object", "properties": s["properties"], "required": s["required"]},
        }
        for s in specs
    ]


async def generate_parts(
    contents: list[dict],
    *,
    system: str | None = None,
    model: str | None = None,
    thinking_cfg: dict | None = None,
    json_schema: dict | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    max_wait_secs: float = 60,
) -> list[dict]:
    """Return the raw content parts of the reply (text and/or functionCall parts).

    contents = [{"role": "user"|"model", "parts": [...]}]. When the reply contains
    function calls, append these parts unchanged as a "model" turn before sending
    the function results back (Gemini 3 requires its thought signatures echoed).
    Rate limits (429) and overloads (503) are retried for up to max_wait_secs."""
    llm_cfg = settings.get("llm", {})
    model = model or llm_cfg.get("model", "gemini-3.5-flash-lite")
    thinking_cfg = thinking_cfg if thinking_cfg is not None else llm_cfg

    base_cfg: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
    schema_variants: list[dict] = [{}]
    if json_schema:
        # Strict schema first; if this API version rejects it, fall back to plain JSON mode
        # (the prompt still describes the fields and Python validates the result).
        schema_variants = [
            {"responseMimeType": "application/json", "responseJsonSchema": json_schema},
            {"responseMimeType": "application/json"},
        ]

    last_error = ""
    async with httpx.AsyncClient(timeout=60) as client:
        for schema_part in schema_variants:
            for thinking in _thinking_variants(thinking_cfg):
                gen = dict(base_cfg, **schema_part)
                if thinking:
                    gen["thinkingConfig"] = thinking
                body: dict = {"contents": contents, "generationConfig": gen}
                if system:
                    body["systemInstruction"] = {"parts": [{"text": system}]}
                if tools:
                    body["tools"] = [{"functionDeclarations": tool_declarations(tools)}]
                r = await _post(client, URL.format(model=model), body, max_wait_secs)
                if r.status_code == 400:
                    last_error = r.text[:300]
                    continue  # try the next variant
                if r.status_code != 200:
                    raise RuntimeError(
                        f"Gemini HTTP {r.status_code}: {describe_error(f'{r.status_code} {r.text}')}"
                    )
                data = r.json()
                cand = (data.get("candidates") or [{}])[0]
                parts = cand.get("content", {}).get("parts", [])
                if not any(p.get("text") or p.get("functionCall") for p in parts if not p.get("thought")):
                    raise RuntimeError(
                        f"Gemini returned nothing usable (finish reason: {cand.get('finishReason', 'unknown')})"
                    )
                return parts
    raise RuntimeError(f"Gemini rejected every request variant: {last_error}")


def text_of(parts: list[dict]) -> str:
    return "".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought")).strip()


def calls_of(parts: list[dict]) -> list[dict]:
    """[{'name': ..., 'args': {...}}, ...] for every function call in the reply."""
    return [
        {"name": p["functionCall"].get("name"), "args": p["functionCall"].get("args") or {}}
        for p in parts
        if p.get("functionCall")
    ]


async def generate(contents: list[dict], **kwargs) -> str:
    """Text-only convenience wrapper."""
    text = text_of(await generate_parts(contents, **kwargs))
    if not text:
        raise RuntimeError("Gemini returned no text")
    return text


def parse_json(text: str) -> dict:
    """Parse model JSON, tolerating ```json fences."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    return json.loads(t[start : end + 1])
