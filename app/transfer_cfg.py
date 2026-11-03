"""transfer_call settings (config/agent.yaml -> transfer:), without web-server imports."""

from app.config import settings

DEFAULTS = {
    "enabled": False,
    "number": "",                 # recruiter's mobile or desk number, e.g. 919876543210
    "mode": "warm",               # warm (briefing first) | cold
    "announce": "Sure, please stay on the line. I'm connecting you to our recruiter now.",
    "unavailable": "Sorry, our recruiter couldn't take the call right now. They will call you back shortly. Thank you!",
    "ring_timeout_secs": 25,
}


def cfg() -> dict:
    return {**DEFAULTS, **(settings.get("transfer") or {})}


def available() -> bool:
    c = cfg()
    return bool(c["enabled"]) and len("".join(ch for ch in str(c["number"]) if ch.isdigit())) >= 10
