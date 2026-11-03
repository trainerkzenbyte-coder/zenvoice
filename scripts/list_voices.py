"""Sprint 1, Step 1: audition Deepgram Aura-2 English voices.

Run from the project folder (virtual environment active):
    python scripts/list_voices.py            # list voices, Indian-accent first
    python scripts/list_voices.py --samples  # also save a short MP3 per shortlisted voice

Samples are saved to voice_samples/ (ignored by git). Each sample costs a
fraction of a cent from your Deepgram free credit.
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
if not KEY:
    sys.exit("DEEPGRAM_API_KEY is empty in .env")

HEADERS = {"Authorization": f"Token {KEY}"}
SAMPLE_TEXT = (
    "Hi, this is Priya calling from the hiring team. "
    "I saw your application for the software developer role. "
    "Is this a good time to talk for two minutes? "
    "Your notice period is thirty days and your expected CTC is twelve lakhs, right?"
)

r = httpx.get("https://api.deepgram.com/v1/models", headers=HEADERS, timeout=20)
r.raise_for_status()
voices = []
for m in r.json().get("tts", []):
    name = m.get("canonical_name") or m.get("name", "")
    if not name.startswith("aura-2-") or not name.endswith("-en"):
        continue
    meta = m.get("metadata") or {}
    voices.append(
        {
            "name": name,
            "accent": str(meta.get("accent", "?")),
            "age": str(meta.get("age", "")),
            "tags": ", ".join(meta.get("tags", []) or [])[:60],
        }
    )

if not voices:
    sys.exit("No Aura-2 English voices returned. Check your Deepgram key.")

# Indian-accent voices first, then the rest alphabetically
voices.sort(key=lambda v: ("india" not in v["accent"].lower(), v["name"]))

print(f"{'VOICE':<28} {'ACCENT':<14} {'AGE':<12} TAGS")
for v in voices:
    print(f"{v['name']:<28} {v['accent']:<14} {v['age']:<12} {v['tags']}")

indian = [v["name"] for v in voices if "india" in v["accent"].lower()]
print(f"\n{len(voices)} voices, {len(indian)} with an Indian accent.")

if "--samples" in sys.argv:
    shortlist = indian or [v["name"] for v in voices[:6]]
    out = ROOT / "voice_samples"
    out.mkdir(exist_ok=True)
    print(f"\nSaving samples for {len(shortlist)} voice(s) to {out} ...")
    for name in shortlist:
        resp = httpx.post(
            "https://api.deepgram.com/v1/speak",
            params={"model": name},
            headers=HEADERS,
            json={"text": SAMPLE_TEXT},
            timeout=30,
        )
        if resp.status_code == 200:
            (out / f"{name}.mp3").write_bytes(resp.content)
            print(f"  saved {name}.mp3")
        else:
            print(f"  {name}: failed (HTTP {resp.status_code})")
    print("\nPlay the MP3s, pick a voice, and put its name in config/agent.yaml under tts: voice:")
