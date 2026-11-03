"""Simulate many phone calls at once against the server. No Vobiz, no phone bill.

Each simulated call connects to the server exactly like Vobiz does (same WebSocket,
same 8 kHz phone audio), waits for Priya's greeting, speaks 4 recorded candidate
answers, and measures how long Priya takes to start replying after each one.

Run from your laptop (virtual environment active), with the server running:
    python scripts/load_test.py                 # 10 calls at once (Sprint 5 exit test)
    python scripts/load_test.py --calls 1       # one call: a quick latency check
    python scripts/load_test.py --calls 20      # find the server's limit

Uses real Deepgram and Gemini (about Rs 2-3 per simulated call). Simulated calls are
never sent to Vobiz or your Google Sheet. Set ADMIN_TOKEN in .env to also see the
server's CPU and memory during the test.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import vobiz  # noqa: E402
from app.config import secret, settings  # noqa: E402

AUDIO_DIR = ROOT / "logs" / "loadtest_audio"
CHUNK = 160                      # 20 ms of 8 kHz mu-law
SILENCE = b"\xff" * CHUNK        # mu-law silence
CANDIDATE = "Karthik"            # name Priya asks for in the greeting
# One sentence each, no full stops in the middle: a long pause inside an answer makes
# Priya reply to the first half (realistic, but it spoils the latency measurement).
UTTERANCES = [
    "Yes, this is Karthik speaking",
    "Yes, I have a few minutes, please go ahead",
    "I am a Java developer at Infosys with five years of experience",
    "My current CTC is nine lakhs and I am expecting fourteen lakhs",
]
REPLY_TIMEOUT = 15.0             # seconds to wait for Priya before giving up on a turn
BOT_DONE_GAP = 2.5               # no new bot audio for this long = she has really finished
MIN_REAL_DELAY = 0.3             # a reply faster than this began before the candidate finished


# ---------- candidate audio (made once with Deepgram, then cached) ----------
def prepare_audio() -> list[bytes]:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    voice = (settings.get("tts") or {}).get("voice", "aura-2-thalia-en")
    clips = []
    for i, text in enumerate(UTTERANCES):
        tag = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:8]
        path = AUDIO_DIR / f"turn{i + 1}_{tag}.ulaw"
        if not path.exists():
            r = httpx.post(
                "https://api.deepgram.com/v1/speak",
                params={"model": voice, "encoding": "mulaw", "sample_rate": 8000, "container": "none"},
                headers={"Authorization": f"Token {secret('DEEPGRAM_API_KEY')}"},
                json={"text": text}, timeout=30,
            )
            r.raise_for_status()
            path.write_bytes(r.content)
        clips.append(path.read_bytes())
    return clips


# ---------- one simulated call ----------
class SimCall:
    def __init__(self, idx: int, run_id: str, ws_url: str, clips: list[bytes]):
        self.idx, self.ws_url, self.clips = idx, ws_url, clips
        self.call_id = f"loadtest-{run_id}-{idx}"
        self.greeting_s: float | None = None
        self.latencies: list[float | None] = []  # None = no reply in time
        self.overlaps = 0          # Priya started talking while the candidate was speaking
        self.status = "ok"
        self._bot_first = 0.0       # first bot audio of the current reply
        self._bot_last = 0.0        # last bot audio message
        self._bot_bytes = 0         # audio bytes in the current reply
        self._got_audio = asyncio.Event()
        self._closed = False

    def bot_done(self) -> bool:
        if not self._bot_first:
            return False
        playback_end = self._bot_first + self._bot_bytes / 8000
        now = time.perf_counter()
        return now - self._bot_last > BOT_DONE_GAP and now > playback_end + 0.3

    def _new_reply(self):
        self._bot_first, self._bot_last, self._bot_bytes = 0.0, 0.0, 0
        self._got_audio.clear()

    async def _reader(self, ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("event") == "playAudio":
                    now = time.perf_counter()
                    if not self._bot_first:
                        self._bot_first = now
                    self._bot_last = now
                    self._bot_bytes += len(base64.b64decode(msg.get("media", {}).get("payload", "")))
                    self._got_audio.set()
        except websockets.ConnectionClosed:
            pass
        finally:
            self._closed = True

    async def _send_audio(self, ws, audio: bytes, until=None, max_secs=30.0):
        """Send audio (or silence) in real time, 20 ms per frame."""
        t0 = time.perf_counter()
        nxt = t0
        pos = 0
        while not self._closed:
            if audio:
                frame = audio[pos:pos + CHUNK]
                if not frame:
                    return
                frame = frame.ljust(CHUNK, b"\xff")
                pos += CHUNK
            else:
                if until and until():
                    return
                if time.perf_counter() - t0 > max_secs:
                    raise TimeoutError
                frame = SILENCE
            try:
                await ws.send(json.dumps({"event": "media", "media": {"payload": base64.b64encode(frame).decode()}}))
            except websockets.ConnectionClosed:
                self._closed = True  # Priya (the server) ended the call
                return
            nxt += 0.02
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))

    async def run(self):
        try:
            async with websockets.connect(self.ws_url, max_size=None, open_timeout=15) as ws:
                await ws.send(json.dumps({"event": "start", "start": {
                    "callId": self.call_id, "streamId": f"stream-{self.call_id}",
                    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000}}}))
                reader = asyncio.create_task(self._reader(ws))
                t_connect = time.perf_counter()

                # Greeting: send silence until Priya has started AND finished speaking.
                self._new_reply()
                await self._send_audio(ws, b"", until=lambda: self._got_audio.is_set(), max_secs=REPLY_TIMEOUT)
                if not self._got_audio.is_set():
                    raise TimeoutError
                self.greeting_s = self._bot_first - t_connect
                await self._send_audio(ws, b"", until=self.bot_done, max_secs=30)

                for clip in self.clips:
                    if self._closed:
                        break
                    self._new_reply()
                    await self._send_audio(ws, clip)              # candidate speaks
                    t_end = time.perf_counter()
                    if self._got_audio.is_set():
                        # Priya talked while the candidate was still speaking: not a valid
                        # reply-delay sample. Let her finish, then go on.
                        self.overlaps += 1
                        self.latencies.append(float("nan"))
                        await self._send_audio(ws, b"", until=self.bot_done, max_secs=30)
                        continue
                    try:
                        await self._send_audio(ws, b"", until=lambda: self._got_audio.is_set(),
                                               max_secs=REPLY_TIMEOUT)
                        if not self._got_audio.is_set():  # call ended before she replied
                            self.status = "ended early"
                            break
                        delay = self._bot_first - t_end
                        if delay < MIN_REAL_DELAY:  # she was answering the first half
                            self.overlaps += 1
                            delay = float("nan")
                        self.latencies.append(delay)
                    except TimeoutError:
                        self.latencies.append(None)
                        self.status = "no reply"
                        break
                    await self._send_audio(ws, b"", until=self.bot_done, max_secs=30)
                reader.cancel()
        except TimeoutError:
            self.status = "no greeting"
        except Exception as e:  # noqa: BLE001
            self.status = f"error: {str(e)[:60]}"


# ---------- server stats while the test runs ----------
async def get_status(client: httpx.AsyncClient, base: str) -> dict | None:
    token = os.getenv("ADMIN_TOKEN", "").strip()
    if not token:
        return None
    try:
        r = await client.get(f"{base}/status", headers={"Authorization": f"Bearer {token}"})
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


async def watch_server(base: str, stop: asyncio.Event, peaks: dict):
    async with httpx.AsyncClient(timeout=5) as client:
        while not stop.is_set():
            s = await get_status(client, base)
            if s:
                for k in ("active_calls", "load_1min", "cpu_percent", "memory_used_mb"):
                    if s.get(k) is not None:
                        peaks[k] = max(peaks.get(k, 0), s[k])
                peaks["cpu_count"] = s.get("cpu_count")
                peaks["memory_total_mb"] = s.get("memory_total_mb")
            await asyncio.sleep(2)


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def cell(x: float | None) -> str:
    if x is None:
        return "no reply"
    if x != x:  # NaN: Priya talked over the candidate
        return "overlap "
    return f"{x:5.2f} s "


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=10, help="simulated calls at the same time")
    ap.add_argument("--stagger", type=float, default=1.0, help="seconds between call starts")
    ap.add_argument("--url", help="server base URL (default: PUBLIC_BASE_URL from .env)")
    args = ap.parse_args()

    base = (args.url or vobiz.public_base_url()).rstrip("/")
    host = urlparse(base).netloc
    print("Preparing candidate audio... ", end="", flush=True)
    clips = prepare_audio()
    print("done")

    async with httpx.AsyncClient(timeout=5) as client:
        before = await get_status(client, base)

    run_id = time.strftime("%H%M%S")
    calls = []
    for i in range(1, args.calls + 1):
        ctx = vobiz.encode_context(CANDIDATE, "910000000000")
        scheme = "ws" if base.startswith("http://") else "wss"
        ws_url = f"{scheme}://{host}/ws/{vobiz.stream_token()}/{quote(ctx)}"
        calls.append(SimCall(i, run_id, ws_url, clips))

    print(f"Starting {args.calls} simulated call(s) against {base} ({len(clips)} candidate turns each)...\n")
    stop, peaks = asyncio.Event(), {}
    watcher = asyncio.create_task(watch_server(base, stop, peaks))

    async def start_later(c: SimCall, delay: float):
        await asyncio.sleep(delay)
        await c.run()

    t0 = time.perf_counter()
    await asyncio.gather(*(start_later(c, i * args.stagger) for i, c in enumerate(calls)))
    stop.set()
    await watcher
    async with httpx.AsyncClient(timeout=5) as client:
        after = await get_status(client, base)

    print(f"{'call':>4}  {'greeting':>8}  " + "  ".join(f"turn {i + 1:<3}" for i in range(len(clips))) + "  status")
    all_lat, greetings, overlaps = [], [], 0
    for c in calls:
        cells = [cell(x) for x in c.latencies]
        cells += ["        "] * (len(clips) - len(cells))
        g = f"{c.greeting_s:6.2f} s" if c.greeting_s is not None else "     -  "
        print(f"{c.idx:>4}  {g}  " + "  ".join(cells) + f"  {c.status}")
        all_lat += [x for x in c.latencies if x is not None and x == x]
        greetings += [c.greeting_s] if c.greeting_s is not None else []
        overlaps += c.overlaps

    ok = sum(1 for c in calls if c.status == "ok")
    print(f"\nCompleted: {ok}/{len(calls)} calls in {time.perf_counter() - t0:.0f} s")
    if greetings:
        print(f"Greeting (call connected -> Priya's first audio): median {statistics.median(greetings):.2f} s, "
              f"worst {max(greetings):.2f} s   (target: under 1.5 s)")
    if all_lat:
        print(f"Reply delay (end of candidate speech -> Priya's first audio), {len(all_lat)} turns:")
        print(f"  p50 {statistics.median(all_lat):.2f} s | p95 {pct(all_lat, 0.95):.2f} s | "
              f"worst {max(all_lat):.2f} s   (Gate A target: p50 < 0.9 s, p95 < 1.5 s)")
        print("  Includes the pause Priya waits to be sure you finished (turn.vad_stop_secs).")
    if overlaps:
        print(f"Overlaps: {overlaps} time(s) Priya started talking before the candidate finished "
              "(not counted in the delay figures).")

    if before is not None and after is not None and "llm_errors" in after:
        errors = after["llm_errors"] - before.get("llm_errors", 0)
        if errors:
            print(f"\nGemini errors during the test: {errors}. Last: {after.get('last_llm_error', '?')}")
            print("  Priya covered each one with 'Sorry, could you say that again?' or a polite goodbye,")
            print("  so those turns show a longer delay or the call ends early ('ended early').")
            if "free tier" in str(after.get("last_llm_error", "")):
                print("  This is the free-tier cap: test with fewer calls at once (see README, 'Gemini plan').")
        else:
            print("\nGemini errors during the test: none")
    elif any(c.status == "no reply" for c in calls):
        print("\n'no reply' usually means Gemini refused the request (rate limit). Check the server window")
        print("or logs/server.log for 'Gemini failed' / '429'.")

    if peaks:
        parts = [f"{peaks.get('active_calls', '?')} active calls"]
        if peaks.get("cpu_percent") is not None:
            parts.append(f"CPU {peaks['cpu_percent']:.0f}% busy")
        if peaks.get("load_1min") is not None:
            parts.append(f"load {peaks['load_1min']} on {peaks.get('cpu_count', '?')} CPUs")
        if peaks.get("memory_used_mb"):
            parts.append(f"memory {peaks['memory_used_mb']}/{peaks.get('memory_total_mb', '?')} MB")
        print("Server peak: " + ", ".join(parts))
        print("  Rule of thumb: CPU under 70% (or load under the CPU count) = headroom left.")
    else:
        print("(Set ADMIN_TOKEN in .env to also see the server's CPU, memory and Gemini errors.)")
    sys.exit(0 if ok == len(calls) else 1)


if __name__ == "__main__":
    asyncio.run(main())
