"""Voice library for the console (Agents -> Voice), like Retell's "Select Voice" dialog.

    GET    /api/voices                    Deepgram Aura-2 + Sarvam Bulbul v3 voices + your cloned voices
    GET    /api/voices/preview?voice=&lang=   a short sample in that voice (cached in data/voice_previews/)
    POST   /api/voices/clone              clone your voice with Sarvam (10-15 s recording, your consent)
    DELETE /api/voices/clone/{voice_id}   delete a cloned voice

API keys stay on the server (.env); the browser never sees them. Previews cost a few paise
once per voice and language (about 120 characters), then play from the cache for free.
"""

from __future__ import annotations

import base64
import hashlib
import re
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from loguru import logger

from app import dashboard as core
from app import voice_config as vc
from app.config import ROOT, settings

router = APIRouter()
PREVIEW_DIR = Path(ROOT) / "data" / "voice_previews"
SARVAM = "https://api.sarvam.ai"
_cache: dict[str, tuple[float, object]] = {}

SAMPLE_TEXT = {
    "en-IN": "Hi, this is {agent} from {company}. Do you have two minutes to talk about a job opening?",
    "ta-IN": "வணக்கம், நான் {company} நிறுவனத்திலிருந்து பேசுகிறேன். ஒரு வேலை வாய்ப்பு பற்றி இரண்டு நிமிடம் பேசலாமா?",
    "hi-IN": "नमस्ते, मैं {company} से हूँ। क्या आपके पास एक नौकरी के बारे में बात करने के लिए दो मिनट हैं?",
    "te-IN": "నమస్తే, నేను {company} నుండి మాట్లాడుతున్నాను. ఒక ఉద్యోగ అవకాశం గురించి రెండు నిమిషాలు మాట్లాడవచ్చా?",
    "kn-IN": "ನಮಸ್ಕಾರ, ನಾನು {company} ಇಂದ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ಒಂದು ಉದ್ಯೋಗಾವಕಾಶದ ಬಗ್ಗೆ ಎರಡು ನಿಮಿಷ ಮಾತನಾಡಬಹುದೇ?",
    "ml-IN": "നമസ്കാരം, ഞാൻ {company} ൽ നിന്നാണ് വിളിക്കുന്നത്. ഒരു ജോലി അവസരത്തെക്കുറിച്ച് രണ്ട് മിനിറ്റ് സംസാരിക്കാമോ?",
    "bn-IN": "নমস্কার, আমি {company} থেকে বলছি। একটা চাকরির সুযোগ নিয়ে দুই মিনিট কথা বলা যাবে?",
    "mr-IN": "नमस्कार, मी {company} मधून बोलत आहे. एका नोकरीच्या संधीबद्दल बोलायला दोन मिनिटे वेळ आहे का?",
    "gu-IN": "નમસ્તે, હું {company} માંથી બોલું છું. એક નોકરીની તક વિશે બે મિનિટ વાત કરી શકીએ?",
}

# Used when Deepgram's model list can't be fetched. Listen before choosing.
DEEPGRAM_FALLBACK = [
    ("thalia", "female", "American"), ("andromeda", "female", "American"), ("helena", "female", "American"),
    ("asteria", "female", "American"), ("luna", "female", "American"), ("athena", "female", "American"),
    ("hera", "female", "American"), ("aurora", "female", "American"), ("apollo", "male", "American"),
    ("arcas", "male", "American"), ("orion", "male", "American"), ("zeus", "male", "American"),
]


def _key(name: str) -> str:
    return __import__("os").getenv(name, "").strip()


async def _cached(name: str, ttl: float, fn):
    hit = _cache.get(name)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    value = await fn()
    _cache[name] = (time.time(), value)
    return value


# ---------------------------------------------------------------- catalogue
async def _deepgram_voices() -> list[dict]:
    async def fetch():
        from app.voice_config import deepgram_host  # India endpoint, same as calls

        key = _key("DEEPGRAM_API_KEY")
        if not key:
            return None
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(f"https://{deepgram_host()}/v1/models", headers={"Authorization": f"Token {key}"})
        r.raise_for_status()
        out = []
        for m in r.json().get("tts") or []:
            cid = str(m.get("canonical_name") or "")
            langs = [str(x) for x in (m.get("languages") or [])]
            if not cid.startswith("aura-2-") or not any(x.startswith("en") for x in langs):
                continue
            meta = m.get("metadata") or {}
            tags = [str(t).lower() for t in (meta.get("tags") or [])]
            gender = "female" if any(t in ("feminine", "female") for t in tags) else (
                "male" if any(t in ("masculine", "male") for t in tags) else "")
            out.append({"id": cid, "provider": "deepgram", "name": str(m.get("name") or cid).title(),
                        "gender": gender, "accent": str(meta.get("accent") or ""), "languages": ["en-IN"],
                        "tags": tags[:4], "cost": "Deepgram credit"})
        return out or None

    try:
        voices = await _cached("deepgram", 86400, fetch)
    except Exception as e:  # noqa: BLE001
        logger.info(f"Voice list: Deepgram model list unavailable ({e}); using the built-in list")
        voices = None
    return voices or [{"id": f"aura-2-{n}-en", "provider": "deepgram", "name": n.title(), "gender": g, "accent": a,
                       "languages": ["en-IN"], "tags": [], "cost": "Deepgram credit"} for n, g, a in DEEPGRAM_FALLBACK]


def _sarvam_voices() -> list[dict]:
    from app import sarvam_v4 as v4

    v4_list = [{"id": f"sarvam:{v}", "provider": "sarvam", "name": v4.label(v), "gender": v4.gender(v),
                "accent": "Indian", "languages": [lang], "tags": ["bulbul v4 flash"],
                "model_label": "Sarvam Bulbul v4 Flash", "cost": "Sarvam credit"}
               for lang, vs in v4.V4_VOICES.items() for v in vs]
    v3_list = [{"id": f"sarvam:{v}", "provider": "sarvam", "name": v.title(), "gender": g, "accent": "Indian",
                "languages": sorted(vc.SARVAM_TTS_LANGUAGES & set(vc.LANGUAGES)), "tags": ["bulbul v3"],
                "model_label": "Sarvam Bulbul v3", "cost": "₹30 / 10k characters"}
               for g, vs in vc.SARVAM_VOICES.items() for v in vs]
    return v4_list + v3_list


async def _clones() -> tuple[list[dict], str]:
    key = _key("SARVAM_API_KEY")
    if not key:
        return [], ""

    async def fetch():
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(f"{SARVAM}/voices", params={"limit": 200}, headers={"api-subscription-key": key})
        if r.status_code != 200:
            raise RuntimeError(f"Sarvam {r.status_code}: {r.text[:160]}")
        voices = ((r.json().get("data") or {}).get("voices")) or []
        return [{"id": f"clone:{v['id']}", "provider": "clone", "name": v.get("name") or "My voice",
                 "gender": v.get("gender") or "", "accent": v.get("accent") or "",
                 "languages": sorted(vc.SARVAM_TTS_LANGUAGES & set(vc.LANGUAGES)), "tags": ["cloned"],
                 "cost": "₹30 / 10k characters", "created": v.get("created_at", ""),
                 "reference_language": v.get("language", "")}
                for v in voices if v.get("status", "active") == "active" and vc.CLONE_ID.fullmatch(str(v.get("id", "")))]

    try:
        return await _cached("clones", 60, fetch), ""
    except Exception as e:  # noqa: BLE001
        return [], str(e)[:200]


@router.get("/api/voices")
async def list_voices(request: Request):
    core._guard(request)
    clones, clone_err = await _clones()
    return {
        "voices": clones + _sarvam_voices() + await _deepgram_voices(),
        "current": (settings.get("tts") or {}).get("voice", vc.DEFAULT_DEEPGRAM_VOICE),
        "languages": vc.LANGUAGES,
        "sarvam_key_set": bool(_key("SARVAM_API_KEY")), "deepgram_key_set": bool(_key("DEEPGRAM_API_KEY")),
        "clone_error": clone_err,
    }


# ---------------------------------------------------------------- previews
def _sample(lang: str) -> str:
    agent = settings.get("agent") or {}
    return SAMPLE_TEXT.get(lang, SAMPLE_TEXT["en-IN"]).format(
        agent=agent.get("name") or "Priya", company=agent.get("company") or "our company")


async def _synth(provider: str, vid: str, lang: str, text: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=25) as c:
        if provider == "deepgram":
            from app.voice_config import deepgram_host

            key = _key("DEEPGRAM_API_KEY")
            if not key:
                raise HTTPException(400, "Add DEEPGRAM_API_KEY to .env")
            r = await c.post(f"https://{deepgram_host()}/v1/speak", params={"model": vid, "encoding": "linear16",
                             "container": "wav", "sample_rate": 24000},
                             headers={"Authorization": f"Token {key}"}, json={"text": text})
            if r.status_code != 200:
                raise HTTPException(502, f"Deepgram: {r.text[:200]}")
            return r.content, "audio/wav"
        key = _key("SARVAM_API_KEY")
        if not key:
            raise HTTPException(400, "Add SARVAM_API_KEY to .env to hear Sarvam voices")
        headers = {"api-subscription-key": key}
        if provider == "sarvam":
            from app import sarvam_v4

            r = await c.post(f"{SARVAM}/text-to-speech", headers=headers, json={
                "text": text, "target_language_code": sarvam_v4.V4_LANG.get(vid, lang), "speaker": vid,
                "model": sarvam_v4.model_for(vid),
                "speech_sample_rate": 24000, "pace": 1.0})
            if r.status_code != 200:
                raise HTTPException(502, f"Sarvam: {r.text[:200]}")
            audios = r.json().get("audios") or []
            if not audios:
                raise HTTPException(502, "Sarvam returned no audio")
            return base64.b64decode(audios[0]), "audio/wav"
        r = await c.post(f"{SARVAM}/voices/clone/stream", headers=headers, data={
            "text": text, "language_code": lang, "voice_id": vid, "output_audio_codec": "wav",
            "speech_sample_rate": "24000"})
        if r.status_code != 200:
            raise HTTPException(502, f"Sarvam cloned voice: {r.text[:200]}")
        return r.content, "audio/wav"


@router.get("/api/voices/preview")
async def preview(request: Request, voice: str, lang: str = "en-IN"):
    core._guard(request)
    if not vc.valid_voice(voice):
        raise HTTPException(400, "Unknown voice")
    provider, vid = vc.parse_voice(voice)
    if lang not in SAMPLE_TEXT or (provider == "deepgram" and lang != "en-IN"):
        lang = "en-IN"
    text = _sample(lang)
    name = hashlib.sha1(f"{voice}|{lang}|{text}".encode()).hexdigest()[:20] + ".wav"
    path = PREVIEW_DIR / name
    if not path.exists():
        audio, _ = await _synth(provider, vid, lang, text)
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
    return Response(path.read_bytes(), media_type="audio/wav", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- cloning
@router.post("/api/voices/clone")
async def clone_voice(request: Request, file: UploadFile = File(...), name: str = Form(...),
                      language: str = Form("en-IN"), consent: str = Form("")):
    core._guard(request)
    key = _key("SARVAM_API_KEY")
    if not key:
        raise HTTPException(400, "Add SARVAM_API_KEY to .env first")
    if consent.lower() not in ("true", "1", "yes", "on"):
        raise HTTPException(400, "Confirm that this is your own voice (or you have the speaker's permission)")
    name = core._clean_str(name, "Voice name", 100)
    if language not in vc.SARVAM_TTS_LANGUAGES:
        raise HTTPException(400, "Recording language must be an Indian language or English (India)")
    data = await file.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "Recording too large (max 20 MB; 10-15 seconds is enough)")
    if len(data) < 20000:
        raise HTTPException(400, "Recording too short: record 10-15 seconds of clear speech")
    fname = re.sub(r"[^\w.-]", "_", Path(file.filename or "voice.wav").name) or "voice.wav"
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{SARVAM}/voices/create", headers={"api-subscription-key": key},
                         data={"name": name, "language": language},
                         files={"file": (fname, data, file.content_type or "audio/wav")})
    if r.status_code not in (200, 201):
        raise HTTPException(502, f"Sarvam could not clone the voice: {r.text[:300]}")
    info = (r.json().get("data") or {})
    vid = str(info.get("voice_id") or "")
    if not vc.CLONE_ID.fullmatch(vid):
        raise HTTPException(502, "Sarvam did not return a voice id")
    _cache.pop("clones", None)
    logger.info(f"Voice cloned: {name} ({vid})")
    return {"ok": True, "voice": f"clone:{vid}", "reference_text": info.get("reference_text", "")}


@router.delete("/api/voices/clone/{voice_id}")
async def delete_clone(request: Request, voice_id: str):
    core._guard(request)
    if not vc.CLONE_ID.fullmatch(voice_id):
        raise HTTPException(400, "Unknown voice")
    if (settings.get("tts") or {}).get("voice") == f"clone:{voice_id}":
        raise HTTPException(400, "This voice is used by the live agent: pick another voice first")
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.delete(f"{SARVAM}/voices/delete/{voice_id}", headers={"api-subscription-key": _key("SARVAM_API_KEY")})
    if r.status_code not in (200, 204):
        raise HTTPException(502, f"Sarvam: {r.text[:200]}")
    _cache.pop("clones", None)
    return {"ok": True}
