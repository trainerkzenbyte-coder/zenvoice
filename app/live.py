"""Calls in progress + live audio streaming for the dashboard's Live Monitoring page.

Live audio packets (binary WebSocket messages on /ws/listen/<call_id>):

    bytes 0-3  sample rate (uint32, little endian)
    bytes 4-7  channel     (uint32): 0 = candidate, 1 = agent, 2 = "agent interrupted" (no audio)
    bytes 8-   raw 16-bit mono PCM

The browser plays the candidate and the agent on two separate timelines, so they overlap
naturally instead of being queued one after the other (which made playback fall behind).
An empty packet (b"") means the call ended.
"""

import asyncio
import struct
import time
from typing import Any, Dict, List, Set

from pipecat.frames.frames import Frame, InputAudioRawFrame, InterruptionFrame, OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

LIVE_CALLS: Dict[str, Dict[str, Any]] = {}
AUDIO_LISTENERS: Dict[str, Set[asyncio.Queue]] = {}

CH_CANDIDATE, CH_AGENT, CH_INTERRUPT = 0, 1, 2
END_OF_CALL = b""


def register(
    call_id: str,
    *,
    name: str = "",
    phone: str = "",
    role: str = "",
    transcript: list | None = None,
    simulated: bool = False,
) -> None:
    if not call_id:
        return
    LIVE_CALLS[call_id] = {
        "call_id": call_id,
        "candidate_name": name or "Candidate",
        "phone": phone or "",
        "role": role or "",
        "status": "in-progress",
        "started": time.time(),
        "transcript": transcript if transcript is not None else [],
        "simulated": simulated,
    }


def unregister(call_id: str) -> None:
    if not call_id:
        return
    LIVE_CALLS.pop(call_id, None)
    for q in AUDIO_LISTENERS.pop(call_id, set()):
        _put(q, END_OF_CALL, force=True)  # tell the listener's WebSocket to close


def is_live(call_id: str) -> bool:
    return call_id in LIVE_CALLS


def add_audio_listener(call_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=150)  # ~3 s of 20 ms packets per channel; older ones drop
    AUDIO_LISTENERS.setdefault(call_id, set()).add(q)
    return q


def remove_audio_listener(call_id: str, q: asyncio.Queue) -> None:
    listeners = AUDIO_LISTENERS.get(call_id)
    if listeners is not None:
        listeners.discard(q)


def _put(q: asyncio.Queue, packet: bytes, force: bool = False) -> None:
    try:
        q.put_nowait(packet)
    except asyncio.QueueFull:
        if force:  # make room for the end-of-call marker
            try:
                q.get_nowait()
                q.put_nowait(packet)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass


def broadcast_audio(call_id: str, pcm_bytes: bytes, sample_rate: int, channel: int = CH_CANDIDATE) -> None:
    """Send one packet (8-byte header + 16-bit PCM) to every browser listening to this call."""
    listeners = AUDIO_LISTENERS.get(call_id)
    if not listeners:
        return
    packet = struct.pack("<II", int(sample_rate), int(channel)) + pcm_bytes
    for q in list(listeners):
        _put(q, packet)


class LiveAudioTap(FrameProcessor):
    """Taps candidate audio (input) or agent audio (output) and broadcasts it. Costs nothing
    when nobody is listening."""

    def __init__(self, call_ids: list[str], tap_input: bool = True, tap_output: bool = True):
        super().__init__()
        self._call_ids = list(dict.fromkeys(c for c in call_ids if c))  # unique, keeps order
        self._tap_input = tap_input
        self._tap_output = tap_output

    def _send(self, audio: bytes, rate: int, channel: int) -> None:
        for cid in self._call_ids:
            broadcast_audio(cid, audio, rate, channel)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._tap_input and isinstance(frame, InputAudioRawFrame):
            self._send(frame.audio, frame.sample_rate, CH_CANDIDATE)
        elif self._tap_output and isinstance(frame, OutputAudioRawFrame):
            self._send(frame.audio, frame.sample_rate, CH_AGENT)
        elif self._tap_output and isinstance(frame, InterruptionFrame):
            # The candidate cut the agent off: the browser drops agent audio it has queued.
            self._send(b"", 16000, CH_INTERRUPT)

        await self.push_frame(frame, direction)


def _format_transcript_item(t: Any) -> Dict[str, str]:
    if isinstance(t, dict):
        speaker = t.get("role") or t.get("speaker") or "system"
        text = t.get("text") or t.get("content") or ""
        return {"role": str(speaker), "text": str(text)}
    if isinstance(t, (list, tuple)) and len(t) >= 2:
        return {"role": str(t[0]), "text": str(t[1])}
    return {"role": "system", "text": str(t)}


def snapshot(max_turns: int = 60) -> List[Dict[str, Any]]:
    """One entry per call. The phone bot registers a call under both its call id and its stream
    id (so either can be listened to); the shared transcript list identifies the duplicate."""
    now = time.time()
    results = []
    seen: set[int] = set()
    for c in list(LIVE_CALLS.values()):
        transcript = c.get("transcript", [])
        if id(transcript) in seen:
            continue
        seen.add(id(transcript))
        try:
            raw_transcript = list(transcript)[-max_turns:]
            elapsed = max(0, round(now - c.get("started", now)))
            results.append({
                "call_id": c.get("call_id", ""),
                "candidate_name": c.get("candidate_name", "Candidate"),
                "phone": c.get("phone", ""),
                "role": c.get("role", ""),
                "status": "in-progress",
                "simulated": c.get("simulated", False),
                "started": c.get("started", now),
                "elapsed_secs": elapsed,
                "duration": elapsed,
                "transcript": [_format_transcript_item(t) for t in raw_transcript],
            })
        except Exception:
            continue
    return results
