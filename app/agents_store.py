"""Agent profiles for the console's Agents page.

The LIVE agent is whatever is in config/agent.yaml, config/screening.yaml and the prompt
file: that is what every call uses. Profiles (config/agents/<id>.json) let you keep several
agents (e.g. one per role) and switch which one is live. Exactly one profile is "active";
editing it edits the live files, editing another profile only edits its JSON until you
make it active.
"""

import json
import re
import secrets
import time
from pathlib import Path

from app.config import ROOT

AGENTS_DIR = Path(ROOT) / "config" / "agents"
META = AGENTS_DIR / "_meta.json"
SETTING_KEYS = [
    "agent_name", "company", "model", "temperature", "voice", "call_language", "voice_speed", "greeting_with_name", "greeting_without_name",
    "max_call_seconds", "silence_hangup_seconds", "calling_start", "calling_end", "vad_confidence",
    "vad_min_volume", "interrupt_min_words", "job_title", "job_location", "job_key_skills",
    "job_experience_range", "fields", "prompt", "greeting_mode", "greeting_delay_secs", "stt_language",
    "keyterms", "vad_stop_secs", "ring_timeout_secs", "sheet_timeout_secs", "record_calls",
    "transfer_enabled", "transfer_number", "transfer_mode", "transfer_announce", "reminder_count", "reminder_line",
    "natural_style", "contact_memory", "knowledge_max_chars", "stt_provider", "history_messages", "smart_turn_max_wait_secs", "off_limits", "deflect_line", "pronunciation",
]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _path(agent_id: str) -> Path:
    if not re.fullmatch(r"ag_[0-9a-f]{8}", agent_id or ""):
        raise KeyError(agent_id)
    return AGENTS_DIR / f"{agent_id}.json"


def meta() -> dict:
    m = _read(META, {})
    m.setdefault("active", None)
    m.setdefault("folders", [])
    return m


def save_meta(m: dict) -> None:
    _write(META, m)


def ensure_default(live_settings: dict) -> None:
    """First visit: turn the current live configuration into the first (active) profile."""
    if meta()["active"] and _path(meta()["active"]).exists():
        return
    existing = list_agents()
    if existing:
        m = meta()
        m["active"] = existing[0]["id"]
        save_meta(m)
        return
    name = f"{live_settings.get('agent_name', 'Priya')} - {live_settings.get('company') or 'Zenbyte'} Screening"
    agent = create(name, live_settings)
    m = meta()
    m["active"] = agent["id"]
    save_meta(m)


def list_agents() -> list[dict]:
    if not AGENTS_DIR.exists():
        return []
    active = meta()["active"]
    out = []
    for path in sorted(AGENTS_DIR.glob("ag_*.json")):
        a = _read(path, None)
        if a:
            a["active"] = a["id"] == active
            out.append(a)
    return sorted(out, key=lambda a: (not a["active"], a.get("name", "")))


def get(agent_id: str) -> dict:
    a = _read(_path(agent_id), None)
    if not a:
        raise KeyError(agent_id)
    a["active"] = a["id"] == meta()["active"]
    return a


def create(name: str, settings: dict, folder: str = "") -> dict:
    a = {"id": f"ag_{secrets.token_hex(4)}", "name": name.strip()[:80] or "Untitled agent", "type": "Single Prompt",
         "folder": folder, "created": _now(), "updated": _now(),
         "settings": {k: settings[k] for k in SETTING_KEYS if k in settings}}
    _write(_path(a["id"]), a)
    return a


def update(agent_id: str, *, name: str | None = None, folder: str | None = None, settings: dict | None = None) -> dict:
    a = get(agent_id)
    if name is not None:
        a["name"] = name.strip()[:80] or a["name"]
    if folder is not None:
        a["folder"] = folder
    if settings is not None:
        a["settings"].update({k: v for k, v in settings.items() if k in SETTING_KEYS})
    a["updated"] = _now()
    a.pop("active", None)
    _write(_path(agent_id), a)
    return get(agent_id)


def delete(agent_id: str) -> None:
    if meta()["active"] == agent_id:
        raise ValueError("The live agent can't be deleted. Make another agent live first.")
    _path(agent_id).unlink(missing_ok=True)


def set_active(agent_id: str) -> None:
    get(agent_id)
    m = meta()
    m["active"] = agent_id
    save_meta(m)


def snapshot_active(live_settings: dict) -> None:
    """After the live settings change, store them in the active profile too."""
    active = meta()["active"]
    if active:
        try:
            update(active, settings=live_settings)
        except KeyError:
            pass
