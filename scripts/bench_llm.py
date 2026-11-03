"""Measure time-to-first-word of each call LLM from THIS computer (India -> provider).

    python scripts/bench_llm.py            # every provider whose key is in .env, 5 runs each
    python scripts/bench_llm.py --runs 10

Prints the median and worst "first spoken word" time. That is the number that matters on a
call: the voice starts as soon as the first words arrive. Uses a realistic call prompt
(your live prompt file) and a short candidate reply, streaming, like a real call.
Costs a few paise at most; Groq's free tier is enough.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.prompts import load_prompt  # noqa: E402

CANDIDATES = [
    ("groq", "openai/gpt-oss-20b", "GROQ_API_KEY", "https://api.groq.com/openai/v1", {"reasoning_effort": "low"}),
    ("groq", "llama-3.1-8b-instant", "GROQ_API_KEY", "https://api.groq.com/openai/v1", {}),
    ("cerebras", "gpt-oss-120b", "CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", {"reasoning_effort": "low"}),
    ("openai", "gpt-4.1-mini", "OPENAI_API_KEY", "https://api.openai.com/v1", {}),
]
GEMINI = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
USER = "Yeah, I'm a Java developer with four years of experience, currently in Chennai."


def openai_compatible(base: str, key: str, model: str, extra: dict, system: str) -> float:
    body = {"model": model, "stream": True, "temperature": 0.4, "max_completion_tokens": 400,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": USER}], **extra}
    t0 = time.perf_counter()
    with httpx.stream("POST", f"{base}/chat/completions", json=body, timeout=30,
                      headers={"Authorization": f"Bearer {key}"}) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.read()[:200]!r}")
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                return (time.perf_counter() - t0) * 1000
    raise RuntimeError("no spoken text in the reply (raise max tokens?)")


def gemini(key: str, model: str, system: str) -> float:
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": USER}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 120, "thinkingConfig": {"thinkingLevel": "minimal"}}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
    t0 = time.perf_counter()
    with httpx.stream("POST", url, json=body, timeout=30, headers={"x-goog-api-key": key}) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.read()[:200]!r}")
        for line in r.iter_lines():
            if line.startswith("data: ") and '"text"' in line:
                return (time.perf_counter() - t0) * 1000
    raise RuntimeError("no text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    system = load_prompt(candidate_name="Karthik")
    print(f"Prompt: {len(system):,} characters. {args.runs} runs per model, first spoken word (ms):\n")
    jobs = [(f"{p}:{m}", (lambda b=b, k=k, m=m, e=e: openai_compatible(b, os.getenv(k, "").strip(), m, e, system)))
            for p, m, k, b, e in CANDIDATES if os.getenv(k, "").strip()]
    gkey = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    jobs += [(m, (lambda m=m: gemini(gkey, m, system))) for m in GEMINI if gkey]
    if not jobs:
        sys.exit("No LLM keys in .env (GROQ_API_KEY, CEREBRAS_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)")
    for name, fn in jobs:
        times, errors = [], []
        for _ in range(args.runs):
            try:
                times.append(fn())
            except Exception as e:  # noqa: BLE001
                errors.append(str(e)[:120])
            time.sleep(2.5)  # stay under free-tier per-minute limits
        if times:
            print(f"{name:38s} median {statistics.median(times):6.0f}   worst {max(times):6.0f}   ({len(times)} ok)")
        for e in errors[:2]:
            print(f"{'':38s} error: {e}")
    print("\nUnder ~500 ms median is what Retell-class turn gaps need (plus ~250 ms voice, ~300 ms turn detection).")


if __name__ == "__main__":
    main()
