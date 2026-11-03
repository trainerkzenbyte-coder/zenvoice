"""Measure which Gemini model answers fastest from YOUR laptop, plus Deepgram voice speed.

Run from the project folder (virtual environment active):
    python scripts/bench_latency.py

Takes about 1 minute and costs a fraction of a rupee. Keys are never printed.
Prints the median "time to first word" per model and the exact lines to put in
config/agent.yaml.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
DG_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
if not GOOGLE_KEY or not DG_KEY:
    sys.exit("GOOGLE_API_KEY and DEEPGRAM_API_KEY must be set in .env")

RUNS = 5
SYSTEM = (
    "You are Priya, a recruiter on a phone call. Reply in one or two short sentences "
    "in simple Indian English."
)
USER = "Yes I can talk. I'm a Java developer at Infosys with four years of experience."

# (label, model, thinkingConfig, yaml lines)
CANDIDATES = [
    ("2.5 Flash-Lite, thinking off", "gemini-2.5-flash-lite", {"thinkingBudget": 0},
     'model: "gemini-2.5-flash-lite"\n  thinking_budget: 0'),
    ("3.1 Flash-Lite, minimal", "gemini-3.1-flash-lite", {"thinkingLevel": "minimal"},
     'model: "gemini-3.1-flash-lite"\n  thinking_level: "minimal"'),
    ("3.5 Flash-Lite, minimal", "gemini-3.5-flash-lite", {"thinkingLevel": "minimal"},
     'model: "gemini-3.5-flash-lite"\n  thinking_level: "minimal"'),
    ("2.5 Flash, thinking off", "gemini-2.5-flash", {"thinkingBudget": 0},
     'model: "gemini-2.5-flash"\n  thinking_budget: 0'),
]


def gemini_once(client: httpx.Client, model: str, thinking: dict) -> tuple[float, float, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": USER}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 120,
            "thinkingConfig": thinking,
        },
    }
    start = time.perf_counter()
    first = None
    text = ""
    with client.stream(
        "POST", url, params={"alt": "sse"}, headers={"x-goog-api-key": GOOGLE_KEY},
        json=body, timeout=30,
    ) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.read().decode()[:160]}")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            for cand in chunk.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if part.get("text") and not part.get("thought"):
                        if first is None:
                            first = time.perf_counter()
                        text += part["text"]
    end = time.perf_counter()
    if first is None:
        raise RuntimeError("no text returned")
    return (first - start) * 1000, (end - start) * 1000, text.strip()


def bench_gemini(client: httpx.Client):
    results = []
    print(f"Gemini: {RUNS} runs per model (first run is a warm-up and not counted)\n")
    for label, model, thinking, yaml_lines in CANDIDATES:
        variants = [thinking]
        if "thinkingLevel" in thinking:  # some API versions want upper-case enum names
            variants.append({"thinkingLevel": thinking["thinkingLevel"].upper()})
        ttfts, totals, sample, error = [], [], "", None
        for variant in variants:
            ttfts, totals, error = [], [], None
            try:
                for i in range(RUNS + 1):
                    ttft, total, sample = gemini_once(client, model, variant)
                    if i > 0:
                        ttfts.append(ttft)
                        totals.append(total)
                break
            except Exception as e:  # noqa: BLE001
                error = str(e)
        if error:
            print(f"  {label:<30} not available: {error[:90]}")
            continue
        med_ttft, med_total = statistics.median(ttfts), statistics.median(totals)
        results.append((med_ttft, label, yaml_lines))
        print(f"  {label:<30} first word {med_ttft:5.0f} ms | full reply {med_total:5.0f} ms")
        print(f"  {'':<30} e.g. \"{sample[:90]}\"")
    return sorted(results)


def bench_deepgram_tts(client: httpx.Client):
    """First-audio time from Deepgram's global (US) endpoint vs the India endpoint."""
    import yaml

    cfg = yaml.safe_load((ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
    voice = cfg.get("tts", {}).get("voice", "aura-2-thalia-en")
    print(f"\nDeepgram voice {voice}: first audio, median of {RUNS} (first run not counted)")
    for label, host in (("global (US)", "api.deepgram.com"), ("India", "api.in.deepgram.com")):
        times = []
        for i in range(RUNS + 1):
            start = time.perf_counter()
            try:
                with client.stream(
                    "POST", f"https://{host}/v1/speak",
                    params={"model": voice, "encoding": "linear16", "sample_rate": 24000},
                    headers={"Authorization": f"Token {DG_KEY}"},
                    json={"text": "Great, thanks. What is your current notice period?"},
                    timeout=30,
                ) as r:
                    if r.status_code != 200:
                        print(f"  {label:<12} HTTP {r.status_code}")
                        break
                    for _ in r.iter_bytes():
                        if i > 0:
                            times.append((time.perf_counter() - start) * 1000)
                        break
            except httpx.HTTPError as e:
                print(f"  {label:<12} failed: {e}")
                break
        if times:
            print(f"  {label:<12} {statistics.median(times):5.0f} ms")
    print("  The agent uses the India endpoint by default (config/agent.yaml: deepgram: region: in).")


with httpx.Client(http2=False) as client:
    ranked = bench_gemini(client)
    bench_deepgram_tts(client)

if ranked:
    best = ranked[0]
    print(f"\nFastest: {best[1]} ({best[0]:.0f} ms to first word).")
    print("Put these two lines in config/agent.yaml under  llm:  (replace the old model/thinking lines):\n")
    print("  " + best[2])
else:
    print("\nNo Gemini model responded. Check GOOGLE_API_KEY and your internet connection.")
