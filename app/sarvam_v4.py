"""Sarvam Bulbul v4 Flash support, added from OUR code (never edit Pipecat's site-packages).

Pipecat 1.12 only knows bulbul:v3. Bulbul v4 Flash uses the same request format as v3; only the
model name and the voice ids differ. v4 voices are personas tied to one language:
    <name>_<lang>_<style>     e.g. ishita_en_customer, simran_en_sales, aravind_ta_ads
Source: https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/voices
"""

from __future__ import annotations

V4_MODEL = "bulbul:v4-flash"
V3_MODEL = "bulbul:v3"

# Voice ids per call language (exact ids from Sarvam's docs).
V4_VOICES: dict[str, list[str]] = {
    "en-IN": [
        "aditi_en_stories", "aparna_en_companion", "aparna_en_edtech", "ashwin_en_sports",
        "ashwin_en_sports_energetic", "chandrika_en_stories", "dev_en_recovery", "dev_en_conversational",
        "deven_en_conversation", "ishita_en_customer", "ishita_en_medical", "ishita_en_numbers",
        "ishita_en_social", "ishita_en_stories", "kalpit_en_edtech", "nachiket_en_ads", "neha_en_customer",
        "neha_en_latenight", "nupur_en_kids", "ojas_en_social", "ritu_en_edtech", "ritu_en_latenight",
        "ritu_en_medical", "ritu_en_reels", "rohan_en_recovery", "roopa_en_conversational",
        "rustom_en_suspense", "sanchita_en_companion", "sanchita_en_insurance", "sanchita_en_recovery",
        "sanchita_en_market", "sanchita_en_social", "shabana_en_edtech", "shalini_en_companion",
        "shalini_en_customer", "shubh_en_narration", "shubh_en_numbers", "shubh_en_ads", "shubh_en_recovery",
        "shubh_en_audiobook", "shubh_en_narration_gentle", "shubh_en_sports", "simran_en_narration",
        "simran_en_automobile", "simran_en_conversation", "simran_en_customer", "simran_en_edtech",
        "simran_en_edtech_bot", "simran_en_sales", "simran_en_recovery", "simran_en_ads",
        "simran_en_therapist", "sunny_en_social", "varun_en_ads", "varun_en_suspense",
        "zarina_en_conversation", "amelia_en_conversational", "sophia_en_conversational",
        "girish_en_documentary", "girish_en_devotional", "payal_en_edtech", "sarang_en_narration",
    ],
    "ta-IN": ["aravind_ta_ads", "aravind_ta_suspense", "gokul_ta_narration", "vijay_ta_narration"],
}
V4_ALL = {v for vs in V4_VOICES.values() for v in vs}
V4_LANG = {v: lang for lang, vs in V4_VOICES.items() for v in vs}

# Sarvam's docs don't give genders; these come from the persona names. Preview before choosing.
_FEMALE = {"aditi", "aparna", "chandrika", "ishita", "neha", "nupur", "ritu", "roopa", "sanchita", "shabana",
           "shalini", "simran", "zarina", "amelia", "sophia", "payal"}


def is_v4(voice_id: str) -> bool:
    return voice_id in V4_ALL


def model_for(voice_id: str) -> str:
    """The voice decides the model: v4 persona ids -> bulbul:v4-flash, short v3 names -> bulbul:v3."""
    return V4_MODEL if is_v4(voice_id) else V3_MODEL


def gender(voice_id: str) -> str:
    return "female" if voice_id.split("_", 1)[0] in _FEMALE else "male"


def label(voice_id: str) -> str:
    """'simran_en_edtech_bot' -> 'Simran · edtech bot'."""
    name, _lang, *style = voice_id.split("_")
    return f"{name.title()} · {' '.join(style)}" if style else name.title()


def register() -> None:
    """Teach Pipecat's Sarvam service the v4 Flash model (same settings shape as v3). Safe to call twice."""
    from pipecat.services.sarvam import tts as sarvam_tts

    if V4_MODEL in sarvam_tts.TTS_MODEL_CONFIGS:
        return
    v3 = sarvam_tts.TTS_MODEL_CONFIGS[V3_MODEL]
    sarvam_tts.TTS_MODEL_CONFIGS[V4_MODEL] = sarvam_tts.TTSModelConfig(
        supports_pitch=False,
        supports_loudness=False,
        supports_temperature=True,
        default_sample_rate=22050,
        default_speaker="ishita_en_customer",
        pace_range=(0.5, 2.0),
        preprocessing_always_enabled=v3.preprocessing_always_enabled,
        speakers=tuple(sorted(V4_ALL)),
    )
