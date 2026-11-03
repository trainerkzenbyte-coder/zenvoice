"""Which speech-to-text and voice provider a call uses (no Pipecat imports: easy to test).

One setting picks the call language (config/agent.yaml -> agent.language), one picks the voice
(tts.voice). The voice id says which provider speaks:

    aura-2-thalia-en        Deepgram Aura-2 (English; uses your Deepgram credit)
    sarvam:priya            Sarvam Bulbul v3 preset voice (Indian languages + Indian English; paid after free credit)
    sarvam:ishita_en_customer  Sarvam Bulbul v4 Flash persona voice (one language each; app/sarvam_v4.py)
    clone:svc-<uuid>        your own voice, cloned with Sarvam Instant Voice Cloning (beta)

Routing (your choice: free Deepgram voice for English, Sarvam for Indian languages):

    language   speech-to-text                     voice
    en-IN      Deepgram Nova-3, or Sarvam Saaras  the chosen voice (Deepgram, or a Sarvam/cloned voice if you pick one)
               v4 with stt.provider: "sarvam"
    ta-IN ...  Sarvam Saaras v4 (code-mix aware)  Sarvam (a Deepgram voice can't speak Tamil, so the
                                                  default Sarvam voice is used instead)
    auto       Sarvam Saaras v4, detects language Sarvam, switches language to follow the candidate
"""

from __future__ import annotations

import re

LANGUAGES = {
    "en-IN": "English (India)",
    "auto": "Auto-detect Indian languages",
    "ta-IN": "Tamil",
    "hi-IN": "Hindi",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "bn-IN": "Bengali",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
}
# Languages Sarvam's voice can speak (and that the candidate's speech can switch to in "auto").
SARVAM_TTS_LANGUAGES = {"en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", "bn-IN", "mr-IN", "gu-IN", "pa-IN", "od-IN"}

# Bulbul v3 preset voices (Pipecat 1.12 SarvamTTSSpeakerV3). Every v3 voice can speak every
# supported language. Gender from the name; listen with the preview button before choosing.
SARVAM_VOICES = {
    "female": ["priya", "ritu", "neha", "pooja", "simran", "kavya", "ishita", "shreya", "roopa", "amelia", "sophia"],
    "male": ["shubh", "aditya", "rahul", "rohan", "amit", "dev", "ratan", "varun", "manan", "sumit", "kabir",
             "aayan", "ashutosh", "advait"],
}
SARVAM_V3_ALL = {v for vs in SARVAM_VOICES.values() for v in vs}
# Bulbul v4 Flash persona voices (app/sarvam_v4.py): each speaks ONE language, e.g. ishita_en_customer.
from app import sarvam_v4 as _v4  # noqa: E402

SARVAM_ALL = SARVAM_V3_ALL | _v4.V4_ALL
DEFAULT_SARVAM_VOICE = "priya"
DEFAULT_DEEPGRAM_VOICE = "aura-2-thalia-en"
CLONE_ID = re.compile(r"svc-[0-9a-fA-F-]{8,64}")
DEEPGRAM_ID = re.compile(r"aura-2-[a-z]+-[a-z]{2}")


DEEPGRAM_HOSTS = {"in": "api.in.deepgram.com", "global": "api.deepgram.com",
                  "eu": "api.eu.deepgram.com", "au": "api.au.deepgram.com"}


def deepgram_host() -> str:
    """India endpoint by default: same key and price, servers in India instead of the US.
    Override in config/agent.yaml:  deepgram: {region: global}"""
    from app.config import settings

    region = str((settings.get("deepgram") or {}).get("region", "in")).lower()
    return DEEPGRAM_HOSTS.get(region, DEEPGRAM_HOSTS["in"])


def parse_voice(voice: str) -> tuple[str, str]:
    """'sarvam:priya' -> ('sarvam', 'priya'); 'clone:svc-..' -> ('clone', 'svc-..');
    'aura-2-thalia-en' -> ('deepgram', 'aura-2-thalia-en')."""
    voice = str(voice or "").strip()
    if voice.startswith("sarvam:"):
        return "sarvam", voice.split(":", 1)[1]
    if voice.startswith("clone:"):
        return "clone", voice.split(":", 1)[1]
    return "deepgram", voice or DEFAULT_DEEPGRAM_VOICE


def valid_voice(voice: str) -> bool:
    provider, vid = parse_voice(voice)
    if provider == "sarvam":
        return vid in SARVAM_ALL
    if provider == "clone":
        return bool(CLONE_ID.fullmatch(vid))
    return bool(DEEPGRAM_ID.fullmatch(vid))


def call_language(settings: dict) -> str:
    lang = str((settings.get("agent") or {}).get("language") or "en-IN")
    return lang if lang in LANGUAGES else "en-IN"


def plan(settings: dict) -> dict:
    """Everything build_stt/build_tts need, decided in one place."""
    lang = call_language(settings)
    tts_cfg = settings.get("tts") or {}
    provider, vid = parse_voice(tts_cfg.get("voice", DEFAULT_DEEPGRAM_VOICE))
    if tts_cfg.get("provider") == "deepgram_flux" and provider == "deepgram":
        provider = "deepgram_flux"
    note = ""
    if lang != "en-IN" and provider in ("deepgram", "deepgram_flux"):
        note = (f"Deepgram voices only speak English, so this {LANGUAGES[lang]} call uses the Sarvam voice "
                f"'{tts_cfg.get('sarvam_fallback_voice') or DEFAULT_SARVAM_VOICE}'. Pick a Sarvam voice in Agents to choose another.")
        provider, vid = "sarvam", str(tts_cfg.get("sarvam_fallback_voice") or DEFAULT_SARVAM_VOICE)
    if provider == "sarvam" and vid not in SARVAM_ALL:
        note = f"Unknown Sarvam voice '{vid}', using '{DEFAULT_SARVAM_VOICE}'."
        vid = DEFAULT_SARVAM_VOICE
    if provider == "sarvam" and _v4.is_v4(vid) and (lang == "auto" or _v4.V4_LANG[vid] != lang):
        # v4 voices speak only their own language; other languages and auto-detect need a v3 voice.
        fallback = str(tts_cfg.get("sarvam_fallback_voice") or DEFAULT_SARVAM_VOICE)
        note = (f"The v4 voice '{vid}' only speaks {LANGUAGES.get(_v4.V4_LANG[vid], _v4.V4_LANG[vid])}, "
                f"so this {LANGUAGES[lang]} call uses the v3 voice '{fallback}'.")
        vid = fallback
    # English calls use Deepgram unless stt.provider is "sarvam" (Sarvam Saaras v4 for English too:
    # trained on Indian speech and only one API key to pay for). Other languages always use Sarvam.
    stt_choice = str((settings.get("stt") or {}).get("provider") or "deepgram").lower()
    stt_provider = "sarvam" if lang != "en-IN" or stt_choice == "sarvam" else "deepgram"
    speed = tts_cfg.get("speed", 1.0)
    try:
        speed = min(2.0, max(0.5, float(speed)))
    except (TypeError, ValueError):
        speed = 1.0
    return {
        "language": lang,
        "stt": stt_provider,
        "stt_language": "auto" if lang == "auto" else lang,
        "tts": provider,
        "voice": vid,
        "sarvam_model": _v4.model_for(vid),  # bulbul:v4-flash for v4 voices, bulbul:v3 otherwise
        # Sarvam needs a language for the voice: in auto mode start in English and follow the candidate.
        "tts_language": "en-IN" if lang == "auto" else lang,
        "follow_language": lang == "auto",
        "speed": speed,
        "note": note,
        "needs_sarvam": stt_provider == "sarvam" or provider in ("sarvam", "clone"),
    }


def sarvam_language(value) -> str | None:
    """Pipecat Language / 'ta-IN' / 'ta' -> a code Sarvam's voice accepts, or None."""
    code = str(getattr(value, "value", value) or "").strip()
    if not code or code == "auto":
        return None
    if "-" not in code:
        code = f"{code.lower()}-IN"
    code = code[:2].lower() + code[2:]
    if code == "or-IN":
        code = "od-IN"  # Sarvam's code for Odia
    return code if code in SARVAM_TTS_LANGUAGES else None


def language_instruction(lang: str) -> str:
    """Added to the prompt so the reply text matches the voice's language."""
    if lang == "en-IN":
        return ""
    if lang == "auto":
        return ("\n\n# LANGUAGE\nReply in the language the candidate is speaking (English, Tamil, Hindi, Telugu, "
                "Kannada, Malayalam, Bengali, Marathi or Gujarati). Write Indian languages in their own script "
                "(Tamil in Tamil script, Hindi in Devanagari), and keep English words such as job titles, "
                "company names and technologies in English. If the candidate mixes languages, mix the same way.")
    name = LANGUAGES.get(lang, lang)
    return (f"\n\n# LANGUAGE\nSpeak {name} by default, written in {name} script, the way people speak it every day: "
            "keep English words for job titles, technologies, company names, numbers like CTC and notice period "
            f"if that is more natural. If the candidate switches to English, reply in English; if they switch back, "
            f"reply in {name}.")
