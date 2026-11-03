"""Call Library: real recruiter calls -> transcripts -> a speaking-style guide for the agent's prompt.

Console page: BUILD > Call Library.

  1. Upload recordings (mp3, wav, m4a, aac, ogg, flac, webm) or ready-made transcripts (.txt).
  2. Transcribe: Sarvam's Batch speech-to-text (saaras:v4) with speaker labels (diarization).
     Up to 20 files per Sarvam job; long recordings are fine (up to 2 hours each).
  3. Check each transcript: which speaker is the recruiter, fix any line, untick calls to leave out.
  4. Analyse style: the AI studies HOW the recruiters talk (rhythm, acknowledgements, bridges,
     fillers, how they handle doubts) and drafts the prompt's "How to talk" sections, a phrase bank
     and short example exchanges. Nothing is copied into the prompt automatically: you review it.

Privacy: recordings and transcripts are candidate personal data. They are stored only on this
server (data/call_library/, never in git), phone numbers and e-mail addresses are masked in every
transcript, and you can add names/companies to mask too. Only use calls where people were told the
call may be recorded. Delete calls you no longer need from the page.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from app.config import ROOT, settings

router = APIRouter()
LIB = Path(ROOT) / "data" / "call_library"
ANALYSIS = LIB / "_analysis.json"
SETTINGS_FILE = LIB / "_settings.json"
SARVAM = "https://api.sarvam.ai"
# File type Sarvam's storage must be told on upload (without it the file is stored as
# binary/octet-stream and Sarvam rejects the job).
AUDIO_MIME = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/x-m4a", ".aac": "audio/aac",
              ".ogg": "audio/ogg", ".opus": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm"}
AUDIO_EXT = set(AUDIO_MIME)
TEXT_EXT = {".txt"}
MAX_UPLOAD_MB = 150
MAX_FILES_PER_JOB = 20
POLL_SECS = 10  # how often Sarvam's job status is checked
_tasks: set[asyncio.Task] = set()  # keep background jobs referenced


def _guard(request: Request) -> None:
    from app import dashboard as core

    core._guard(request)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def _cfg() -> dict:
    """call_library: in config/agent.yaml (all optional)."""
    c = settings.get("call_library") or {}
    return {
        "language": str(c.get("language", "en-IN")),      # or "unknown" for auto-detect, ta-IN, hi-IN...
        "mode": str(c.get("mode", "codemix")),            # codemix keeps English words in English
        "speakers": int(c.get("speakers", 2)),
        "analysis_model": str(c.get("analysis_model", "")),  # e.g. "cerebras:gpt-oss-120b"; default: llm.*
        "max_chars_per_call": int(c.get("max_chars_per_call", 24000)),
    }


# ---------------------------------------------------------------- storage
def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _item_dir(item_id: str) -> Path:
    if not re.fullmatch(r"cl_[0-9a-f]{10}", item_id or ""):
        raise HTTPException(400, "bad id")
    d = LIB / item_id
    if not d.is_dir():
        raise HTTPException(404, "Not found (deleted?)")
    return d


def _meta(item_id: str) -> dict:
    return _read(_item_dir(item_id) / "meta.json", {})


def _save_meta(meta: dict) -> None:
    _write(LIB / meta["id"] / "meta.json", meta)


def _items() -> list[dict]:
    if not LIB.exists():
        return []
    out = []
    for d in sorted(LIB.glob("cl_*"), reverse=True):
        m = _read(d / "meta.json", None)
        if m:
            out.append(m)
    out.sort(key=lambda m: m.get("uploaded", ""), reverse=True)
    return out


def _transcript(item_id: str) -> dict:
    return _read(LIB / item_id / "transcript.json", {"turns": []})


# ---------------------------------------------------------------- redaction
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s\-]{8,}\d)(?!\w)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _mask_words() -> list[str]:
    return [w for w in (_read(SETTINGS_FILE, {}).get("mask_words") or []) if str(w).strip()]


def redact(text: str, words: list[str] | None = None) -> str:
    text = EMAIL.sub("[EMAIL]", str(text))
    text = PHONE.sub("[PHONE]", text)
    for w in sorted({str(w).strip() for w in (words if words is not None else _mask_words())}, key=len, reverse=True):
        if len(w) >= 2:
            text = re.sub(rf"(?<!\w){re.escape(w)}(?!\w)", "[NAME]", text, flags=re.I)
    return text


def _redact_turns(turns: list[dict]) -> list[dict]:
    words = _mask_words()
    return [{**t, "text": redact(t.get("text", ""), words)} for t in turns]


# ---------------------------------------------------------------- transcripts
SPEAKER_LINE = re.compile(r"^\s*(?:\[?[\d:.]+\]?\s*)?([A-Za-z][\w .'-]{0,30}?)\s*[:\-–]\s+(.+)$")


def parse_text_transcript(text: str) -> list[dict]:
    """'Recruiter: ...' / 'Candidate: ...' lines (any speaker names). Lines without a speaker
    continue the previous turn. A transcript with no speaker names becomes one turn per line."""
    turns: list[dict] = []
    for raw in str(text).replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = SPEAKER_LINE.match(line)
        if m and len(m.group(1).split()) <= 3:
            turns.append({"speaker": m.group(1).strip(), "text": m.group(2).strip()})
        elif turns:
            turns[-1]["text"] += " " + line
        else:
            turns.append({"speaker": "Speaker", "text": line})
    return turns


def turns_from_sarvam(result: dict) -> list[dict]:
    """Sarvam batch output -> [{speaker_id, text, start, end}]. Uses diarized entries when present."""
    entries = ((result.get("diarized_transcript") or {}).get("entries")) or []
    turns: list[dict] = []
    for e in entries:
        text = str(e.get("transcript") or "").strip()
        if not text:
            continue
        sid = str(e.get("speaker_id", "0"))
        if turns and turns[-1]["speaker_id"] == sid:  # merge consecutive chunks from the same person
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = e.get("end_time_seconds", turns[-1].get("end"))
        else:
            turns.append({"speaker_id": sid, "text": text, "start": e.get("start_time_seconds"),
                          "end": e.get("end_time_seconds")})
    if not turns and str(result.get("transcript") or "").strip():
        turns = [{"speaker_id": "0", "text": str(result["transcript"]).strip()}]
    return turns


def guess_speakers(turns: list[dict]) -> dict:
    """Which speaker id is the recruiter? In a screening call the recruiter usually talks most
    (and asks the questions). You can swap it on the page."""
    words: dict[str, int] = {}
    questions: dict[str, int] = {}
    for t in turns:
        sid = str(t.get("speaker_id", t.get("speaker", "0")))
        words[sid] = words.get(sid, 0) + len(t.get("text", "").split())
        questions[sid] = questions.get(sid, 0) + t.get("text", "").count("?")
    if not words:
        return {}
    ranked = sorted(words, key=lambda s: (questions.get(s, 0) * 20 + words[s]), reverse=True)
    out = {ranked[0]: "Recruiter"}
    for s in ranked[1:]:
        out[s] = "Candidate" if len(ranked) == 2 else f"Speaker {int(s) + 1 if s.isdigit() else s}"
    return out


def labelled(item_id: str) -> list[dict]:
    """Turns with readable speaker names (Recruiter / Candidate), redacted."""
    meta = _meta(item_id)
    smap = meta.get("speaker_map") or {}
    out = []
    for t in _transcript(item_id).get("turns", []):
        who = t.get("speaker") or smap.get(str(t.get("speaker_id", "0")), f"Speaker {t.get('speaker_id', '?')}")
        out.append({"speaker": who, "text": t.get("text", "")})
    return out


def as_text(item_id: str, limit: int | None = None) -> str:
    text = "\n".join(f"{t['speaker']}: {t['text']}" for t in labelled(item_id))
    return text if not limit or len(text) <= limit else text[:limit] + "\n[... rest of call cut for length ...]"


# ---------------------------------------------------------------- Sarvam batch transcription
def _key() -> str:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Add SARVAM_API_KEY to .env")
    return key


async def _sarvam(client: httpx.AsyncClient, method: str, path: str, **kw) -> dict:
    r = await client.request(method, f"{SARVAM}{path}", headers={"api-subscription-key": _key()}, **kw)
    if r.status_code >= 300:
        raise RuntimeError(f"Sarvam {path}: HTTP {r.status_code} {r.text[:240]}")
    return r.json() if r.content else {}


async def transcribe_batch(item_ids: list[str]) -> None:
    """One Sarvam batch job for up to 20 recordings: create -> upload -> start -> poll -> download."""
    c = _cfg()
    metas = [_meta(i) for i in item_ids]
    for m in metas:
        m.update(status="transcribing", error="", progress="creating job")
        _save_meta(m)
    names = {}  # file name in the job -> item id
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            params = {"model": "saaras:v4", "mode": c["mode"], "language_code": c["language"],
                      "with_diarization": True, "num_speakers": c["speakers"]}
            job = await _sarvam(client, "POST", "/speech-to-text/job/v1", json={"job_parameters": params})
            job_id = job["job_id"]
            for m in metas:
                fname = f"{m['id']}{Path(m['audio']).suffix.lower()}"
                names[fname] = m["id"]
                m.update(job_id=job_id, progress="uploading")
                _save_meta(m)
            links = await _sarvam(client, "POST", "/speech-to-text/job/v1/upload-files",
                                  json={"job_id": job_id, "files": list(names)})
            azure = "azure" in str(links.get("storage_container_type", job.get("storage_container_type", ""))).lower()
            for fname, iid in names.items():
                url = ((links.get("upload_urls") or {}).get(fname) or {}).get("file_url")
                if not url:
                    raise RuntimeError(f"Sarvam gave no upload link for {fname}")
                data = (LIB / iid / _meta(iid)["audio"]).read_bytes()
                headers = {"Content-Type": AUDIO_MIME.get(Path(fname).suffix.lower(), "audio/mpeg")}
                if azure:
                    headers["x-ms-blob-type"] = "BlockBlob"
                put = await client.put(url, content=data, headers=headers, timeout=600)
                if put.status_code >= 300:
                    raise RuntimeError(f"Upload of {fname} failed: HTTP {put.status_code} {put.text[:160]}")
            await _sarvam(client, "POST", f"/speech-to-text/job/v1/{job_id}/start")
            for m in metas:
                m.update(progress="Sarvam is transcribing")
                _save_meta(m)
            status: dict = {}
            for _ in range(int(7200 / POLL_SECS)):  # up to ~2 hours
                await asyncio.sleep(POLL_SECS)
                status = await _sarvam(client, "GET", f"/speech-to-text/job/v1/{job_id}/status")
                state = str(status.get("job_state", ""))
                if state in ("Completed", "Failed", "PartiallyCompleted"):
                    break
            else:
                raise RuntimeError("Sarvam job still running after 2 hours")
            outputs: dict[str, str] = {}  # item id -> output file name
            failed: dict[str, str] = {}
            for d in status.get("job_details") or []:
                ins = [str(x.get("file_name", x) if isinstance(x, dict) else x) for x in d.get("inputs") or []]
                outs = [str(x.get("file_name", x) if isinstance(x, dict) else x) for x in d.get("outputs") or []]
                for i, inp in enumerate(ins):
                    iid = names.get(Path(inp).name)
                    if not iid:
                        continue
                    if str(d.get("state", "")).lower() in ("success", "completed") and i < len(outs):
                        outputs[iid] = outs[i]
                    else:
                        failed[iid] = str(d.get("error_message") or d.get("exception_name") or "failed")
            if outputs:
                dl = await _sarvam(client, "POST", "/speech-to-text/job/v1/download-files",
                                   json={"job_id": job_id, "files": list(outputs.values())})
                for iid, fname in outputs.items():
                    url = ((dl.get("download_urls") or {}).get(fname) or {}).get("file_url")
                    if not url:
                        failed[iid] = "no download link"
                        continue
                    res = (await client.get(url, timeout=120)).json()
                    turns = turns_from_sarvam(res)
                    if not turns:
                        failed[iid] = "empty transcript (silent or unsupported audio?)"
                        continue
                    _write(LIB / iid / "transcript.json", {"turns": _redact_turns(turns), "source": "sarvam",
                                                            "job_id": job_id, "created": _now()})
                    m = _meta(iid)
                    m.update(status="transcribed", progress="", error="", transcribed=_now(),
                             speaker_map=guess_speakers(turns), turns=len(turns),
                             words=sum(len(t["text"].split()) for t in turns))
                    _save_meta(m)
            for iid in item_ids:
                if iid not in outputs or iid in failed:
                    m = _meta(iid)
                    if m.get("status") != "transcribed":
                        m.update(status="failed", progress="",
                                 error=failed.get(iid) or status.get("error_message") or "Sarvam returned no result")
                        _save_meta(m)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Call Library: transcription failed: {e}")
        for iid in item_ids:
            try:
                m = _meta(iid)
                if m.get("status") == "transcribing":
                    m.update(status="failed", progress="", error=str(e)[:300])
                    _save_meta(m)
            except HTTPException:
                pass


def _start(coro) -> None:
    t = asyncio.create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


# ---------------------------------------------------------------- style analysis (AI)
PER_CALL_SYSTEM = """You study how real Indian recruiters speak on phone screening calls, so a voice agent can
learn their STYLE. Ignore what is offered or promised; only study HOW they talk. Reply with JSON only."""

PER_CALL_USER = """Transcript of one real call (speaker names: Recruiter / Candidate; [PHONE] [EMAIL] [NAME] are masked):

{transcript}

Return JSON with these keys (short strings, copy wording EXACTLY as the recruiter said it):
{{
 "openings": [how the recruiter opened the call],
 "acknowledgements": [how they reacted to answers, e.g. "Okay okay, nice"],
 "bridges": [how they moved to the next topic],
 "fillers": [filler/discourse words they use: "Actually", "So basically", "Achha"...],
 "clarifying": [how they asked again / checked when unclear],
 "doubts": [how they handled a doubt, objection or confusion],
 "closing": [how they ended],
 "turn_length": "typical words per recruiter turn and how many questions per turn",
 "habits": [3-6 notable habits of rhythm and tone],
 "best_moments": [2-3 short exchanges (2-4 lines, "Recruiter: ... / Candidate: ...") that show a natural move well]
}}"""

COMBINE_SYSTEM = """You write the speaking-style part of a system prompt for an Indian voice agent named after a
human recruiter. You turn notes from many real recruiter calls into clear, enforceable instructions.
Style only: never take facts, offers, prices or promises from the calls. Reply with JSON only."""

COMBINE_USER = """Notes from {n} real recruiter calls (JSON per call):

{notes}
{agent_part}
Write JSON with:
{{
 "style_summary": [6-8 bullets: rhythm, turn length, how they react, bridge, clarify, handle doubts, close],
 "phrase_bank": {{"acknowledge": [...], "bridge": [...], "clarify": [...], "reassure": [...], "close": [...]}},
   (8-12 real phrases per group across all calls; drop any that would sound odd from a text-to-speech
    voice, such as "umm", "uh", laughter or sounds)
 "examples": [5 short exchanges, 2-4 lines each, "Recruiter: ... / Candidate: ...", showing: answer + bridge,
    "who is this?", fresher or not working, a doubt, the close. Light clean-up only; keep their Indian English],
 "banned_phrases": [phrases a robotic AI agent uses that these recruiters never do; always include
    "Does that make sense?", "Great question", "Absolutely", "I understand your concern",
    "How can I assist you"{agent_banned}],
 "prompt_sections": "the full text of two prompt sections, '# HOW TO TALK' and '# ONE THING PER TURN', under
    450 words, written as direct instructions to the agent (you/your). It is pasted into the prompt ON ITS OWN,
    so it must be self-contained: write the phrase bank (grouped: acknowledge / bridge / clarify / reassure /
    close) and the banned phrases OUT IN FULL inside it; never refer to lists 'below' or by key name. Include a
    rule never to use the same acknowledgement twice in a row, and the 5 examples as 'examples of the style,
    never to be read word for word'. Describe the style as natural tendencies ('you usually...'), not
    mechanical steps. Keep: at most one question per turn; answer only what they asked, then stop; react to
    what they said before asking. Do NOT write rules for the greeting or confirming their name: the opening
    line is spoken automatically before you speak."
}}"""


def _llm_choice() -> tuple[str, str]:
    from app.llm_choice import llm_provider

    override = _cfg()["analysis_model"]
    if override and ":" in override:
        p, m = override.split(":", 1)
        return p, m
    p, m = llm_provider(settings.get("llm") or {})
    if p == "anthropic":  # call analysis uses Gemini when calls run on Claude
        return "google", "gemini-3.5-flash"
    return p, m


async def _ask(system: str, user: str, max_tokens: int = 4000) -> dict:
    """One JSON answer from the configured AI (Cerebras / Groq / OpenAI / Gemini)."""
    provider, model = _llm_choice()
    text = ""
    if provider == "google":
        from app import gemini_rest

        text = await gemini_rest.generate([{"role": "user", "parts": [{"text": user}]}], system=system,
                                          model=model, temperature=0.3, max_tokens=max_tokens, max_wait_secs=120)
    else:
        from app.config import secret
        from app.llm_choice import FAST_LLMS, REASONING_MODELS

        base = {"groq": "https://api.groq.com/openai/v1", "cerebras": "https://api.cerebras.ai/v1",
                "openai": "https://api.openai.com/v1"}[provider]
        body = {"model": model, "temperature": 0.3, "max_completion_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        if any(r in model for r in REASONING_MODELS):
            body["reasoning_effort"] = "medium"
        async with httpx.AsyncClient(timeout=180) as client:
            for attempt in range(4):
                r = await client.post(f"{base}/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {secret(FAST_LLMS[provider][0])}"})
                if r.status_code == 429 and attempt < 3:  # rate limit: wait and try again
                    await asyncio.sleep(float(r.headers.get("retry-after") or 20))
                    continue
                if r.status_code >= 300:
                    raise RuntimeError(f"{provider}: HTTP {r.status_code} {r.text[:200]}")
                text = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                break
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0) if m else text)
    except ValueError:
        raise RuntimeError("The AI did not return valid JSON; try again") from None


def _agent_sample(limit: int = 6000) -> str:
    """The agent's own most recent real call, for the 'banned phrases' comparison."""
    log_dir = Path(ROOT) / "logs" / "calls"
    for path in sorted(log_dir.glob("*.json"), reverse=True)[:30] if log_dir.exists() else []:
        rec = _read(path, {})
        tr = rec.get("transcript") or []
        if len(tr) >= 8 and not str(rec.get("call_id", "")).startswith("loadtest-"):
            lines = [f"{'Agent' if t.get('role') == 'bot' else 'Candidate'}: {t.get('text', '')}" for t in tr]
            return redact("\n".join(lines))[:limit]
    return ""


async def run_analysis(item_ids: list[str]) -> None:
    started = time.time()
    state = {"status": "running", "started": _now(), "calls": item_ids, "progress": "", "error": ""}
    _write(ANALYSIS, {**_read(ANALYSIS, {}), **state})
    try:
        c = _cfg()
        notes = []
        for n, iid in enumerate(item_ids, 1):
            _write(ANALYSIS, {**_read(ANALYSIS, {}), "progress": f"reading call {n} of {len(item_ids)}"})
            m = _meta(iid)
            cache = LIB / iid / "style_notes.json"
            cached = _read(cache, None)
            if cached and cached.get("_transcript_mtime") == (LIB / iid / "transcript.json").stat().st_mtime:
                notes.append(cached)
                continue
            text = as_text(iid, c["max_chars_per_call"])
            if len(text) < 200:
                continue
            note = await _ask(PER_CALL_SYSTEM, PER_CALL_USER.format(transcript=text), max_tokens=3000)
            note["_call"] = m.get("title") or iid
            note["_transcript_mtime"] = (LIB / iid / "transcript.json").stat().st_mtime
            _write(cache, note)
            notes.append(note)
        if not notes:
            raise RuntimeError("No usable transcripts (each call needs a few lines of conversation)")
        _write(ANALYSIS, {**_read(ANALYSIS, {}), "progress": "writing the style guide"})
        agent = _agent_sample()
        slim = [{k: v for k, v in n.items() if not k.startswith("_")} for n in notes]
        notes_text = json.dumps(slim, ensure_ascii=False)
        if len(notes_text) > 60000:  # many calls: keep the combine step within limits
            notes_text = notes_text[:60000]
        result = await _ask(COMBINE_SYSTEM, COMBINE_USER.format(
            n=len(notes), notes=notes_text,
            agent_part=(f"\nFor comparison, a recent call by our AI agent (find its robotic habits):\n{agent}\n" if agent else ""),
            agent_banned=" plus the agent's own robotic phrases from the comparison call" if agent else ""),
            max_tokens=6000)
        _write(ANALYSIS, {"status": "done", "finished": _now(), "secs": round(time.time() - started),
                          "calls": item_ids, "result": result, "progress": "", "error": ""})
        logger.info(f"Call Library: style analysis of {len(notes)} call(s) done")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Call Library: analysis failed: {e}")
        _write(ANALYSIS, {**_read(ANALYSIS, {}), "status": "failed", "error": str(e)[:300], "progress": ""})


# ---------------------------------------------------------------- routes
@router.get("/api/library")
async def api_library(request: Request):
    _guard(request)
    a = _read(ANALYSIS, {})
    return {"items": _items(), "analysis": {k: a.get(k) for k in ("status", "finished", "progress", "error", "calls")},
            "mask_words": _mask_words(), "sarvam_key_set": bool(os.getenv("SARVAM_API_KEY", "").strip()),
            "config": _cfg()}


@router.post("/api/library/upload")
async def api_library_upload(request: Request, files: list[UploadFile] = File(...), recruiter: str = Form("")):
    _guard(request)
    added, skipped = [], []
    for f in files[:50]:
        name = Path(f.filename or "call").name
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXT | TEXT_EXT:
            skipped.append(f"{name}: not a supported audio or .txt file")
            continue
        data = await f.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            skipped.append(f"{name}: larger than {MAX_UPLOAD_MB} MB")
            continue
        if not data:
            skipped.append(f"{name}: empty file")
            continue
        iid = "cl_" + secrets.token_hex(5)
        d = LIB / iid
        d.mkdir(parents=True)
        meta = {"id": iid, "title": Path(name).stem[:80], "recruiter": recruiter.strip()[:60], "filename": name,
                "uploaded": _now(), "size": len(data), "include": True}
        if ext in TEXT_EXT:
            turns = parse_text_transcript(data.decode("utf-8", errors="replace"))
            names = sorted({t["speaker"] for t in turns})
            _write(d / "transcript.json", {"turns": _redact_turns(turns), "source": "text", "created": _now()})
            meta.update(kind="text", status="transcribed", turns=len(turns),
                        words=sum(len(t["text"].split()) for t in turns), speakers=names)
        else:
            (d / f"audio{ext}").write_bytes(data)
            meta.update(kind="audio", audio=f"audio{ext}", status="uploaded")
        _save_meta(meta)
        added.append(iid)
    return {"ok": True, "added": added, "skipped": skipped}


@router.post("/api/library/transcribe")
async def api_library_transcribe(request: Request):
    _guard(request)
    body = await request.json()
    ids = [i for i in (body.get("ids") or []) if isinstance(i, str)]
    if not ids:
        ids = [m["id"] for m in _items() if m.get("kind") == "audio" and m.get("status") in ("uploaded", "failed")]
    metas = [_meta(i) for i in ids]
    todo = [m["id"] for m in metas if m.get("kind") == "audio" and m.get("status") != "transcribing"]
    if not todo:
        raise HTTPException(400, "Nothing to transcribe")
    _key()  # fail now with a clear message if the key is missing
    for k in range(0, len(todo), MAX_FILES_PER_JOB):
        _start(transcribe_batch(todo[k:k + MAX_FILES_PER_JOB]))
    return {"ok": True, "started": len(todo), "jobs": (len(todo) + MAX_FILES_PER_JOB - 1) // MAX_FILES_PER_JOB}


@router.get("/api/library/analysis")
async def api_library_analysis(request: Request):
    _guard(request)
    return _read(ANALYSIS, {"status": "none"})


@router.post("/api/library/analyze")
async def api_library_analyze(request: Request):
    _guard(request)
    if _read(ANALYSIS, {}).get("status") == "running":
        raise HTTPException(409, "An analysis is already running")
    body = await request.json()
    ids = [i for i in (body.get("ids") or []) if isinstance(i, str)]
    if not ids:
        ids = [m["id"] for m in _items() if m.get("status") == "transcribed" and m.get("include", True)]
    ids = [i for i in ids if _meta(i).get("status") == "transcribed"]
    if not ids:
        raise HTTPException(400, "No transcribed calls selected")
    _start(run_analysis(ids[:60]))
    return {"ok": True, "calls": len(ids[:60])}


@router.post("/api/library/settings")
async def api_library_settings(request: Request):
    _guard(request)
    body = await request.json()
    words = [str(w).strip()[:60] for w in (body.get("mask_words") or []) if str(w).strip()][:300]
    _write(SETTINGS_FILE, {**_read(SETTINGS_FILE, {}), "mask_words": words})
    # re-mask every stored transcript with the new list
    for m in _items():
        p = LIB / m["id"] / "transcript.json"
        tr = _read(p, None)
        if tr:
            tr["turns"] = _redact_turns(tr.get("turns", []))
            _write(p, tr)
    return {"ok": True, "mask_words": words}


@router.get("/api/library/{item_id}")
async def api_library_item(request: Request, item_id: str):
    _guard(request)
    meta = _meta(item_id)
    return {"meta": meta, "turns": _transcript(item_id).get("turns", []), "labelled": labelled(item_id)
            if meta.get("status") == "transcribed" else []}


@router.post("/api/library/{item_id}")
async def api_library_update(request: Request, item_id: str):
    _guard(request)
    meta = _meta(item_id)
    body = await request.json()
    for key, limit in (("title", 80), ("recruiter", 60)):
        if key in body:
            meta[key] = str(body[key] or "").strip()[:limit]
    if "include" in body:
        meta["include"] = bool(body["include"])
    if isinstance(body.get("speaker_map"), dict):
        meta["speaker_map"] = {str(k)[:8]: str(v).strip()[:30] or "Speaker" for k, v in body["speaker_map"].items()}
    if isinstance(body.get("turns"), list):  # edited transcript lines
        tr = _transcript(item_id)
        old = tr.get("turns", [])
        new = []
        for i, t in enumerate(body["turns"][:2000]):
            text = str((t or {}).get("text", "")).strip()[:4000]
            if not text:
                continue
            base = old[i] if i < len(old) else {}
            new.append({**base, "text": text, **({"speaker": str(t["speaker"])[:30]} if t.get("speaker") else {})})
        tr["turns"] = _redact_turns(new)
        _write(LIB / item_id / "transcript.json", tr)
        meta["turns"] = len(new)
    _save_meta(meta)
    return {"ok": True, "meta": meta}


@router.get("/api/library/{item_id}/audio")
async def api_library_audio(request: Request, item_id: str):
    _guard(request)
    meta = _meta(item_id)
    if not meta.get("audio"):
        raise HTTPException(404, "No audio for this call")
    return FileResponse(LIB / item_id / meta["audio"])


@router.delete("/api/library/{item_id}")
async def api_library_delete(request: Request, item_id: str):
    _guard(request)
    d = _item_dir(item_id)
    if _meta(item_id).get("status") == "transcribing":
        raise HTTPException(409, "Still transcribing; delete it when Sarvam has finished")
    shutil.rmtree(d)
    return {"ok": True}
