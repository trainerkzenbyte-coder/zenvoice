"""Compare Sarvam speech recognition settings on YOUR audio, side by side.

Plays the same recording to Sarvam's live speech-to-text three times at once, each with a different
way of giving it your keyterms, and prints what each one heard:

  A  prompt    the way calls worked until now (keyterms written into a text hint)
  B  keyterms  Saaras v4's own keyterm list (stt.sarvam_bias: "keyterms")
  C  none      no hint at all

It also tells you whether Sarvam accepts the keyterm list. Run from the project folder:

    python scripts/stt_compare.py logs/recordings/<a call>.wav
    python scripts/stt_compare.py my_test.wav --secs 90

The WAV must be 8 kHz or 16 kHz, 16-bit (call recordings are 8 kHz mono: fine). Call recordings
contain the agent's voice too; that's fine for comparing. Uses SARVAM_API_KEY from .env and your
stt settings from config/agent.yaml. Audio is streamed at normal speed, so 60 s takes about 60 s.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import wave
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import secret, settings  # noqa: E402
from app.services import sarvam_keyterms  # noqa: E402

URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
CHUNK_MS = 50


def read_wav(path: Path, secs: float) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            sys.exit("The WAV must be 16-bit.")
        if rate not in (8000, 16000):
            sys.exit(f"The WAV is {rate} Hz; Sarvam takes 8000 or 16000 Hz. Convert it first.")
        audio = w.readframes(int(min(w.getnframes(), secs * rate)))
    if ch == 2:  # keep the left channel
        audio = b"".join(audio[i:i + 2] for i in range(0, len(audio), 4))
    elif ch != 1:
        sys.exit("The WAV must be mono or stereo.")
    return audio, rate


async def run(name: str, extra: dict, audio: bytes, rate: int) -> dict:
    import websockets

    stt = settings.get("stt") or {}
    params = {"language_code": (settings.get("agent") or {}).get("language", "en-IN"), "stream_type": "balanced",
              "endpointing": "vad", "encoding": "linear16", "sample_rate": rate,
              "model": stt.get("sarvam_model", "saaras:v4"), "mode": stt.get("sarvam_mode", "transcribe"),
              "prefix_padding_ms": int(stt.get("sarvam_prefix_ms", 300)),
              "silence_duration_ms": int(stt.get("sarvam_silence_ms", 500)), **extra}
    headers = {"API-SUBSCRIPTION-KEY": secret("SARVAM_API_KEY")}
    out = {"name": name, "finals": [], "error": ""}
    try:
        try:
            ws = await websockets.connect(f"{URL}?{urlencode(params)}", additional_headers=headers, max_size=None)
        except TypeError:  # older websockets
            ws = await websockets.connect(f"{URL}?{urlencode(params)}", extra_headers=headers, max_size=None)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"connection refused: {e}"
        return out

    async def listen():
        async for msg in ws:
            try:
                m = json.loads(msg)
            except ValueError:
                continue
            if m.get("event") == "transcript.final" and (m.get("text") or "").strip():
                out["finals"].append(m["text"].strip())
            elif m.get("event") == "error":
                out["error"] = json.dumps(m)[:300]
            elif m.get("event") == "session.end":
                return

    listener = asyncio.create_task(listen())
    step = int(rate * 2 * CHUNK_MS / 1000)
    start = time.monotonic()
    try:
        for i, pos in enumerate(range(0, len(audio), step)):
            await ws.send(json.dumps({"event": "audio_input", "audio": base64.b64encode(audio[pos:pos + step]).decode()}))
            await asyncio.sleep(max(0.0, start + (i + 1) * CHUNK_MS / 1000 - time.monotonic()))  # real time
        silence = b"\0" * step
        for _ in range(30):  # 1.5 s of silence so the last sentence ends
            await ws.send(json.dumps({"event": "audio_input", "audio": base64.b64encode(silence).decode()}))
            await asyncio.sleep(CHUNK_MS / 1000)
        await ws.send(json.dumps({"event": "end"}))
        await asyncio.wait_for(listener, timeout=15)
    except asyncio.TimeoutError:
        pass
    except Exception as e:  # noqa: BLE001
        out["error"] = out["error"] or f"closed during the test: {e}"
    finally:
        listener.cancel()
        await ws.close()
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--secs", type=float, default=60, help="how much of the recording to use (default 60)")
    a = ap.parse_args()
    audio, rate = read_wav(Path(a.wav), a.secs)
    terms = sarvam_keyterms(settings.get("stt") or {})
    hint = "Phone call about IT training and interviews. Words that may come up: " + ", ".join(terms)[:400]
    print(f"Sending {len(audio) / (2 * rate):.0f} s of audio ({rate} Hz) to Sarvam three times at once, "
          f"{len(terms)} keyterms...")
    results = await asyncio.gather(
        run("A prompt (old way)", {"prompt": hint}, audio, rate),
        run("B keyterms", {"keyterms": json.dumps(terms, ensure_ascii=False)}, audio, rate),
        run("C no hint", {}, audio, rate),
    )
    for r in results:
        print(f"\n===== {r['name']} =====")
        if r["error"]:
            print(f"  PROBLEM: {r['error']}")
        for i, t in enumerate(r["finals"], 1):
            print(f"  {i:>2}. {t}")
    b = results[1]
    print("\nKeyterms accepted by Sarvam:", "NO -> keep stt.sarvam_bias: \"prompt\""
          if b["error"] or not b["finals"] else "yes")
    print("Read the three lists against what was really said and keep the setting that got the most right.")


if __name__ == "__main__":
    asyncio.run(main())
