"""Loads secrets from .env and settings from config/agent.yaml.

Usage:
    from app.config import settings, secret
    settings["llm"]["model"]        -> "gemini-3.1-flash-lite"
    secret("DEEPGRAM_API_KEY")      -> the key, or raises a clear error
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
CONFIG_FILE = ROOT / "config" / "agent.yaml"

load_dotenv(ENV_FILE)
# Either name works for the Gemini key (Google's own docs call it GEMINI_API_KEY).
if not os.getenv("GOOGLE_API_KEY", "").strip() and os.getenv("GEMINI_API_KEY", "").strip():
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"].strip()

with open(CONFIG_FILE, encoding="utf-8") as f:
    settings: dict = yaml.safe_load(f)


def secret(name: str) -> str:
    """Return a required secret from .env, or fail with a helpful message."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is missing. Add it to {ENV_FILE} (copy .env.example first)."
        )
    return value
