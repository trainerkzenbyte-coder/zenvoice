"""Shows what the bot hears and decides, measures reply delay, and keeps a transcript.

Per turn it prints, for example:
    LATENCY 1450 ms [SLOW]  turn-wait 780 | LLM 420 | voice 250

- turn-wait: from "you went quiet" until the bot decided you had finished
- LLM:       from that decision until Gemini's first words
- voice:     from Gemini's first words until audio starts playing
The three parts always add up to the total.

Why logs used to appear twice: Pipecat sends some events (speech started,
interruption, bot speaking...) in BOTH directions through the pipeline as two
separate copies. We now only look at the downstream copy, so each event is
logged once. Errors travel upstream, so they are always shown.
"""

import statistics
import time

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    MetricsFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

WATCHED = (
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    InterruptionFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    TTSSpeakFrame,
    ErrorFrame,
    MetricsFrame,
)


class LatencyObserver(BaseObserver):
    def __init__(self, vad_stop_secs: float, verbose: bool = False, label: str = "", **kwargs):
        super().__init__(**kwargs)
        self._vad_stop_secs = vad_stop_secs
        self._verbose = verbose
        self._label = f"[{label}] " if label else ""
        self._seen: set[int] = set()
        self._bot_text = ""
        self._bot_speaking = False
        self._reset_turn()
        self.samples_ms: list[float] = []
        self.parts: dict[str, list[float]] = {"turn": [], "llm": [], "voice": []}
        self.transcript: list[dict] = []  # [{"role": "user"|"bot", "text": ..., "t": ...}]
        self._created = time.perf_counter()
        self.first_audio_ms: float | None = None  # call connected -> Priya's first audio (greeting)
        self.tokens = {"prompt": 0, "completion": 0, "total": 0, "requests": 0}  # Gemini usage on this call

    # ---------- helpers ----------
    def _log(self, msg: str, always: bool = False):
        if always or self._verbose:
            logger.info(f"{self._label}{msg}")

    def _reset_turn(self):
        self._vad_stopped_at: float | None = None
        self._turn_ended_at: float | None = None
        self._first_llm_at: float | None = None

    def _first_time(self, frame) -> bool:
        # A frame passes through many processors; handle each frame object once.
        if frame.id in self._seen:
            return False
        self._seen.add(frame.id)
        if len(self._seen) > 20000:
            self._seen.clear()
        return True

    # ---------- main hook ----------
    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if not isinstance(frame, WATCHED):
            return
        # Errors travel upstream; everything else: downstream copy only (no duplicates).
        if not isinstance(frame, ErrorFrame) and data.direction != FrameDirection.DOWNSTREAM:
            return
        if not self._first_time(frame):
            return
        now = time.perf_counter()

        if isinstance(frame, MetricsFrame):
            for d in getattr(frame, "data", None) or []:
                usage = getattr(d, "value", None)
                if usage is not None and hasattr(usage, "prompt_tokens"):
                    self.tokens["prompt"] += int(usage.prompt_tokens or 0)
                    self.tokens["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)
                    self.tokens["total"] += int(getattr(usage, "total_tokens", 0) or 0)
                    self.tokens["requests"] += 1
            return
        if isinstance(frame, ErrorFrame):
            src = getattr(getattr(frame, "processor", None), "name", "pipeline")
            logger.error(f"{self._label}ERROR from {src}: {frame.error}")
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._log("  [mic] speech detected")
            self._reset_turn()  # still talking: start the measurement over
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._log("  [mic] pause, checking if you finished")
            if self._vad_stopped_at is None:
                self._vad_stopped_at = now
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._log("  [turn] your turn started")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._log("  [turn] your turn ended, bot is thinking")
            if self._turn_ended_at is None:
                self._turn_ended_at = now
        elif isinstance(frame, InterimTranscriptionFrame):
            self._log(f"  [hearing] {frame.text}")
        elif isinstance(frame, TranscriptionFrame):
            self._log(f"YOU   : {frame.text}", always=True)
            self.transcript.append({"role": "user", "text": frame.text, "t": time.time()})
        elif isinstance(frame, InterruptionFrame):
            if self._bot_speaking:  # only a real barge-in, not every new user turn
                self._log("  [bot] interrupted by you, stopped talking")
        elif isinstance(frame, LLMTextFrame):
            if self._first_llm_at is None:
                self._first_llm_at = now
            self._bot_text += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            text = self._bot_text.strip()
            if text:
                self._log(f"BOT   : {text}", always=True)
                self.transcript.append({"role": "bot", "text": text, "t": time.time()})
            self._bot_text = ""
        elif isinstance(frame, TTSSpeakFrame):  # fixed lines: silence check, fallback goodbye
            self._log(f"BOT   : {frame.text}", always=True)
            self.transcript.append({"role": "bot", "text": frame.text, "t": time.time()})
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if self.first_audio_ms is None:
                self.first_audio_ms = (now - self._created) * 1000
            if self._vad_stopped_at is not None:
                self._record_turn(now)
            self._reset_turn()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._log("  [bot] finished speaking, listening")

    def _record_turn(self, now: float):
        pause_ms = self._vad_stop_secs * 1000
        start = self._vad_stopped_at
        total = (now - start) * 1000 + pause_ms
        turn = llm = voice = None
        if self._turn_ended_at:
            turn = (self._turn_ended_at - start) * 1000 + pause_ms
            if self._first_llm_at and self._first_llm_at >= self._turn_ended_at:
                llm = (self._first_llm_at - self._turn_ended_at) * 1000
                voice = (now - self._first_llm_at) * 1000
        self.samples_ms.append(total)
        for key, val in (("turn", turn), ("llm", llm), ("voice", voice)):
            if val is not None:
                self.parts[key].append(val)
        fmt = lambda v: f"{v:.0f}" if v is not None else "n/a"  # noqa: E731
        flag = "OK" if total < 900 else ("SLOW" if total < 1500 else "TOO SLOW")
        self._log(
            f"LATENCY {total:.0f} ms [{flag}]  turn-wait {fmt(turn)} | LLM {fmt(llm)} | "
            f"voice {fmt(voice)}",
            always=True,
        )

    # ---------- reporting ----------
    def stats(self) -> dict:
        s = sorted(self.samples_ms)
        first = round(self.first_audio_ms) if self.first_audio_ms is not None else None
        if not s:
            return {"turns": 0, "first_audio_ms": first, "tokens": dict(self.tokens)}
        p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
        med = lambda xs: round(statistics.median(xs)) if xs else None  # noqa: E731
        return {
            "turns": len(s),
            "p50_ms": round(statistics.median(s)),
            "p95_ms": round(p95),
            "worst_ms": round(s[-1]),
            "median_turn_wait_ms": med(self.parts["turn"]),
            "median_llm_ms": med(self.parts["llm"]),
            "median_voice_ms": med(self.parts["voice"]),
            "first_audio_ms": first,
            "turn_ms": [round(x) for x in self.samples_ms],  # each reply delay, in call order
            "tokens": dict(self.tokens),
        }

    def summary(self) -> str:
        st = self.stats()
        if not st["turns"]:
            return f"{self._label}No bot replies measured."
        return (
            f"{self._label}Turns measured: {st['turns']} | p50 {st['p50_ms']} ms | "
            f"p95 {st['p95_ms']} ms | worst {st['worst_ms']} ms\n"
            f"{self._label}Median parts: turn-wait {st['median_turn_wait_ms']} | "
            f"LLM {st['median_llm_ms']} | voice {st['median_voice_ms']} ms"
        )
