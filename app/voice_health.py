"""Voice health: finds out WHY the agent's voice breaks on a call, instead of guessing.

The agent's voice can break for four different reasons, each with a different fix:
  1. gaps:          the voice (Sarvam TTS) sends audio slower than it plays, or the AI's text arrives
                    slowly, so the line goes silent for a moment in the middle of a sentence
  2. server stalls: this program froze for a moment (laptop busy, something slow in the call code),
                    so audio that was ready wasn't sent in time
  3. cut-offs:      the agent was interrupted (a barge-in word, the candidate talking over it, or its
                    own voice echoing back into the phone) and stopped mid-sentence
  4. the network:   none of the above happened here, but the audio still broke on the way to the phone
                    (laptop Wi-Fi, ngrok, mobile signal). Then the fix is the cloud server, not the code.

During the call each problem is logged as it happens ("[voice] ..."); at the end you get one summary
with the verdict, and the numbers are saved in the call log (logs/calls/*.json -> latency.voice_health).
Nothing here changes what the agent says or when: it only measures.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger

GAP_SECS = 0.15         # silence inside a reply longer than this is audible as a break
NEW_REPLY_SECS = 1.5    # longer than this: it's a new reply, not a break
STALL_SECS = 0.15       # the program froze this long: audio can't be sent on time
TICK = 0.05


class VoiceHealth:
    def __init__(self, state, label: str = ""):
        self.state = state
        self.label = label
        self.gaps: list[dict] = []
        self.stalls: list[dict] = []
        self.cuts: list[dict] = []
        self.audio_secs = 0.0
        self._t0 = 0.0          # when the current stretch of agent audio started
        self._queued = 0.0      # seconds of audio produced since _t0
        self._last = 0.0
        self._start = time.monotonic()

    # ---------- 1 + 3: audio gaps and cut-offs, seen right after the voice (TTS) ----------
    def meter(self):
        from pipecat.frames.frames import InterruptionFrame, TTSAudioRawFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        health = self

        class VoiceMeter(FrameProcessor):
            async def process_frame(self, frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                try:
                    if isinstance(frame, TTSAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
                        health._audio(len(frame.audio), frame.sample_rate, getattr(frame, "num_channels", 1) or 1)
                    elif isinstance(frame, InterruptionFrame):
                        health._cut()
                except Exception:  # noqa: BLE001  measuring must never break a call
                    pass
                await self.push_frame(frame, direction)

        return VoiceMeter()

    def _audio(self, nbytes: int, rate: int, channels: int, now: float | None = None):
        now = time.monotonic() if now is None else now
        dur = nbytes / float(2 * max(1, rate) * max(1, channels))
        self.audio_secs += dur
        if not self._t0:
            self._t0, self._queued, self._last = now, dur, now
            return
        play_end = self._t0 + self._queued  # when the audio sent so far finishes playing
        silent = now - play_end
        if silent > NEW_REPLY_SECS:  # the previous reply had finished: this is a new one
            self._t0, self._queued, self._last = now, dur, now
            return
        if silent > GAP_SECS:
            self._gap(silent, now)
            self._t0, self._queued = now, dur
        else:
            self._queued += dur
        self._last = now

    def _gap(self, secs: float, now: float):
        recent_stall = any(now - s["at"] <= secs + 0.3 for s in self.stalls)
        if recent_stall:
            cause = "server stall"
        elif getattr(self.state, "llm_busy", False) and now - (getattr(self.state, "last_bot_text_at", 0) or 0) < secs:
            cause = "AI text slow"  # the words for this audio only just arrived from the AI
        else:
            cause = "voice (TTS) slow"
        g = {"at": now, "secs": round(secs, 2), "cause": cause, "t": round(now - self._start, 1)}
        self.gaps.append(g)
        logger.info(f"[{self.label}]   [voice] {secs:.2f} s silence in the middle of a reply ({cause}) "
                    f"at {g['t']} s")

    def _cut(self, now: float | None = None):
        now = time.monotonic() if now is None else now
        playing = self._t0 and now < self._t0 + self._queued + 0.2
        if not playing:
            return
        heard = str(getattr(self.state, "last_barge_in", "") or (self.state.recent_user or [""])[-1])
        c = {"t": round(now - self._start, 1), "heard": heard[:60]}
        self.cuts.append(c)
        self._t0 = self._queued = 0.0
        logger.info(f"[{self.label}]   [voice] agent cut off mid-sentence at {c['t']} s (heard: {c['heard']!r})")

    # ---------- 2: program stalls ----------
    def start(self) -> asyncio.Task:
        return asyncio.create_task(self._watch_loop())

    async def _watch_loop(self):
        try:
            while True:
                t = time.monotonic()
                await asyncio.sleep(TICK)
                late = time.monotonic() - t - TICK
                if late > STALL_SECS:
                    now = time.monotonic()
                    s = {"at": now, "secs": round(late, 2), "t": round(now - self._start, 1),
                         "speaking": bool(getattr(self.state, "bot_speaking", False))}
                    self.stalls.append(s)
                    logger.info(f"[{self.label}]   [voice] program froze for {late:.2f} s at {s['t']} s"
                                + (" while the agent was talking" if s["speaking"] else ""))
        except asyncio.CancelledError:
            pass

    # ---------- summary ----------
    def report(self) -> dict:
        def strip(items):
            return [{k: v for k, v in i.items() if k != "at"} for i in items]

        return {"agent_audio_secs": round(self.audio_secs, 1), "gaps": strip(self.gaps)[:50],
                "stalls": strip(self.stalls)[:50], "cut_offs": self.cuts[:50], "verdict": self.verdict()}

    def verdict(self) -> str:
        speaking_stalls = [s for s in self.stalls if s["speaking"]]
        causes = {}
        for g in self.gaps:
            causes[g["cause"]] = causes.get(g["cause"], 0) + 1
        if not self.gaps and not speaking_stalls and not self.cuts:
            return ("clean: the agent's audio left this program without gaps or cut-offs. If it still sounded "
                    "broken on the phone, the cause is the network (laptop Wi-Fi, ngrok, mobile signal): "
                    "the cloud server fixes that")
        parts = []
        if causes.get("server stall") or speaking_stalls:
            parts.append(f"{len(speaking_stalls)} program freeze(s) while talking: the computer running the "
                         "agent is too busy (close other programs, or run it on the cloud server)")
        if causes.get("voice (TTS) slow"):
            parts.append(f"{causes['voice (TTS) slow']} gap(s) where the voice (Sarvam) was slow: raise "
                         "tts.sarvam_min_buffer (e.g. 60) so it gets longer pieces of text")
        if causes.get("AI text slow"):
            parts.append(f"{causes['AI text slow']} gap(s) waiting for the AI's text: use llm.reasoning_effort "
                         "\"low\" or check the LLM provider")
        if self.cuts:
            parts.append(f"{len(self.cuts)} cut-off(s): the agent was interrupted; see the '[turn] candidate "
                         "cut in' lines for what triggered it")
        return "; ".join(parts)

    def summary_lines(self) -> list[str]:
        longest = max((g["secs"] for g in self.gaps), default=0)
        return [
            f"[voice] health: {len(self.gaps)} gap(s) in replies (longest {longest:.2f} s), "
            f"{len(self.stalls)} program freeze(s), {len(self.cuts)} cut-off(s), "
            f"{self.audio_secs:.0f} s of agent audio",
            f"[voice] verdict: {self.verdict()}",
        ]


# ---------- slow AI replies: where the time went ----------
SLOW_REPLY_SECS = 3.0
_http_logs_installed = False


def forward_http_logs():
    """Show LLM provider trouble in the terminal: rate limits (HTTP 429), server errors and the OpenAI
    client's automatic retries ("waiting 8 s, then trying again") are otherwise invisible."""
    global _http_logs_installed
    if _http_logs_installed:
        return
    import logging
    import re as _re

    class _Forward(logging.Handler):
        def emit(self, record):
            try:
                msg = record.getMessage()
            except Exception:  # noqa: BLE001
                return
            if record.name.startswith("openai") and "Retrying request" in msg:
                logger.warning(f"[llm] provider didn't answer properly, the client is retrying: {msg}")
            elif record.name.startswith("httpx"):
                m = _re.search(r'"HTTP/[\d.]+ (\d{3})', msg)
                if m and m.group(1) != "200" and ("chat/completions" in msg or "generateContent" in msg):
                    hint = " (RATE LIMIT: too many requests or tokens per minute)" if m.group(1) == "429" else ""
                    logger.warning(f"[llm] {msg}{hint}")

    h = _Forward(level=logging.INFO)
    for name in ("openai._base_client", "openai", "httpx"):
        lg = logging.getLogger(name)
        lg.addHandler(h)
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
        lg.propagate = False  # don't print every normal request
    _http_logs_installed = True


class LLMTimer:
    """Splits a slow reply into: waiting before the AI got the turn, the provider's first byte
    (network, queue, rate limit), and the AI thinking before its first spoken word.
    Logs one "[llm] slow reply" line for any reply slower than 3 s."""

    def __init__(self, label: str = ""):
        self.label = label
        self.sent = self.first_byte = self.first_text = 0.0
        self.ttfb = None
        self.usage = None
        self.slow: list[dict] = []
        self.heard_at = 0.0

    def before(self):
        from pipecat.frames.frames import LLMContextFrame, TranscriptionFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        timer = self

        class BeforeLLM(FrameProcessor):
            async def process_frame(self, frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM and not timer.sent:
                    # (a watchdog retry of the same turn keeps the original start time)
                    timer.sent, timer.first_byte, timer.first_text, timer.ttfb, timer.usage = (
                        time.monotonic(), 0.0, 0.0, None, None)
                await self.push_frame(frame, direction)

        return BeforeLLM()

    def after(self):
        from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame, MetricsFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        timer = self

        class AfterLLM(FrameProcessor):
            async def process_frame(self, frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                try:
                    now = time.monotonic()
                    if isinstance(frame, LLMFullResponseStartFrame) and not timer.first_byte:
                        timer.first_byte = now
                    elif isinstance(frame, LLMTextFrame) and timer.sent and not timer.first_text:
                        timer.first_text = now
                    elif isinstance(frame, MetricsFrame):
                        for d in getattr(frame, "data", None) or []:
                            kind = type(d).__name__
                            if kind == "TTFBMetricsData" and "LLM" in str(getattr(d, "processor", "")):
                                timer.ttfb = float(getattr(d, "value", 0) or 0)
                            elif kind == "LLMUsageMetricsData":
                                timer.usage = getattr(d, "value", None)
                    elif isinstance(frame, LLMFullResponseEndFrame):
                        timer._done(now)
                except Exception:  # noqa: BLE001
                    pass
                await self.push_frame(frame, direction)

        return AfterLLM()

    def _done(self, now: float):
        if not self.sent:
            return
        total = (self.first_text or now) - self.sent
        if total >= SLOW_REPLY_SECS:
            u = self.usage
            prompt = getattr(u, "prompt_tokens", None)
            think = getattr(u, "reasoning_tokens", None)
            ttfb = self.ttfb if self.ttfb is not None else ((self.first_byte - self.sent) if self.first_byte else None)
            thinking = (total - ttfb) if ttfb is not None else None
            row = {"secs": round(total, 1), "first_byte_secs": round(ttfb, 1) if ttfb is not None else None,
                   "thinking_secs": round(thinking, 1) if thinking is not None else None,
                   "prompt_tokens": prompt, "reasoning_tokens": think}
            self.slow.append(row)
            if ttfb is not None and ttfb >= 0.7 * total:
                why = "the provider was slow to start (busy, rate limit or network: see any [llm] warnings above)"
            elif thinking is not None and thinking >= 0.6 * total:
                why = "the AI thought for a long time (llm.reasoning_effort \"low\" makes this shorter)"
            else:
                why = "mixed"
            logger.info(f"[{self.label}]   [llm] slow reply {total:.1f} s: provider first byte "
                        f"{row['first_byte_secs']} s, thinking {row['thinking_secs']} s, prompt {prompt} tokens"
                        f"{f', {think} thinking tokens' if think else ''} -> {why}")
        self.sent = 0.0
