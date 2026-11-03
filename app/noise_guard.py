"""Keep background sounds and background PEOPLE out of what the agent hears.

Three layers, all on the candidate's side of the call:

1. Voice gate (before Sarvam; audio.voice_gate): the candidate talks right into the phone, so their
   voice is louder than a TV, people across the room or traffic. The gate learns how loud the
   candidate is on this call and mutes anything much quieter (audio.voice_gate_db below it).

2. Background speech filter (after Sarvam; audio.background_filter, NEW): every piece of text Sarvam
   hears is checked against the sound it came from. If, while those words were being said, there was
   no sound about as loud as the candidate's own voice (within audio.background_near_db), the words
   came from someone else (a crowd, traffic, a lecturer, a TV) and are thrown away: they can't stop
   the agent mid-sentence and the agent never replies to them. This is what stops crowd / traffic /
   lecture-hall voices interrupting the agent.

3. Voice Focus (optional, paid, off): ai-coustics "Quail Voice Focus", an AI model that keeps the main
   speaker and suppresses other voices - the only layer that can separate someone standing right next
   to the candidate talking just as loudly. Needs an ai-coustics account + AIC_LICENSE_KEY in .env +
   pip install "pipecat-ai[aic]==1.12.0".

config/agent.yaml:

    audio:
      voice_gate: true                # layer 1
      voice_gate_db: 20               # mute what is this much quieter than the candidate (15 stricter, 25 gentler)
      background_filter: true         # layer 2
      background_near_db: 10          # "the candidate" = sound within this many dB of their voice level
                                      # (8 = stricter: more background ignored; 14 = gentler)
      background_min_speech_ms: 200   # how much candidate-loud sound the words need (150-400)
      voice_focus: false              # layer 3
      voice_focus_model: "quail-vf-2.2-s-16khz"

The dashboard's live audio keeps the ORIGINAL sound. The call recording has the candidate's side as
Sarvam heard it (muted parts are silent).
"""

from __future__ import annotations

import collections
import math
import time

from loguru import logger

from app.config import settings


def _cfg() -> dict:
    return settings.get("audio") or {}


def _num(key: str, default: float, lo: float, hi: float) -> float:
    try:
        return min(hi, max(lo, float(_cfg().get(key, default))))
    except (TypeError, ValueError):
        return default


def gate_enabled() -> bool:
    return _cfg().get("voice_gate", True) is not False


def gate_db() -> float:
    return _num("voice_gate_db", 20, 6, 40)


def filter_enabled() -> bool:
    return _cfg().get("background_filter", True) is not False


def near_db() -> float:
    return _num("background_near_db", 10, 4, 25)


def min_speech_secs() -> float:
    return _num("background_min_speech_ms", 200, 80, 1500) / 1000.0


def frame_db(pcm: bytes) -> float:
    n = len(pcm) // 2
    if not n:
        return -120.0
    try:
        import numpy as np

        x = np.frombuffer(pcm[: n * 2], dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x)))
    except ImportError:  # pragma: no cover
        import array

        a = array.array("h", pcm[: n * 2])
        rms = math.sqrt(sum(v * v for v in a) / n)
    return 20 * math.log10(rms / 32768.0 + 1e-9)


class GateLogic:
    """Learns the candidate's speaking loudness and decides what to keep (no Pipecat inside)."""

    FLOOR_DB = -55.0      # quieter than this is silence: never used to learn the voice level
    WINDOW = 750          # frames remembered for the voice level (~15 s of 20 ms frames)
    WARMUP = 40           # voiced frames needed before the level is known (~0.8 s of speech)
    HOLD = 15             # frames kept open after a loud one (~300 ms: word endings aren't cut)
    RELEARN = 2250        # nothing as loud as the candidate for ~45 s: forget the level, learn again

    frame_db = staticmethod(frame_db)

    def __init__(self, below_db: float = 20.0):
        self.below = below_db
        self.recent: collections.deque = collections.deque(maxlen=self.WINDOW)
        self.level = None
        self.hold = 0
        self.frames = self.muted = 0
        self._since_update = 0
        self._since_open = 0

    def _update_level(self):
        loud = sorted(self.recent)
        self.level = loud[int(0.9 * (len(loud) - 1))]  # the candidate's normal speaking loudness

    def keep(self, db: float) -> bool:
        """True = pass this frame on; False = it's background (much quieter than the candidate)."""
        self.frames += 1
        if self.level is not None and self._since_open >= self.RELEARN:
            self.level, self._since_open = None, 0  # (a quieter phone, a new speaker position...)
            self.recent.clear()
        # Learn only from sounds that could be the candidate: background never pulls the level down.
        if db > self.FLOOR_DB and (self.level is None or db >= self.level - self.below):
            self.recent.append(db)
            self._since_update += 1
            if len(self.recent) >= self.WARMUP and (self.level is None or self._since_update >= 25):
                self._update_level()
                self._since_update = 0
        if self.level is None:
            return True  # still learning the candidate's voice: let everything through
        if db >= self.level - self.below:
            self.hold, self._since_open = self.HOLD, 0
            return True
        self._since_open += 1
        if self.hold > 0:
            self.hold -= 1
            return True
        self.muted += 1
        return False

    def report(self) -> dict:
        return {"voice_level_db": round(self.level, 1) if self.level is not None else None,
                "gate_db": self.below, "muted_pct": round(100 * self.muted / self.frames, 1) if self.frames else 0}


# ---------------- layer 1: voice gate (before Sarvam) ----------------
def make_voice_gate(label: str = ""):
    """FrameProcessor: put it between the caller's audio and speech-to-text."""
    from pipecat.frames.frames import CancelFrame, EndFrame, InputAudioRawFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class VoiceGate(FrameProcessor):
        """Audio is held back LOOKAHEAD frames (~80 ms), so the gate opens just BEFORE a word starts:
        the soft first sound of a word ("s", "f", "th", "h") is never cut off."""

        LOOKAHEAD = 4

        def __init__(self):
            super().__init__()
            self.logic = GateLogic(gate_db())
            self._pending: collections.deque = collections.deque()

        async def _release(self, direction):
            frame, _db = self._pending.popleft()
            try:
                ahead = max(d for _f, d in self._pending) if self._pending else -120.0
                if not self.logic.keep(max(_db, ahead)):
                    frame.audio = b"\0" * len(frame.audio)  # quiet background: silence for Sarvam
            except Exception:  # noqa: BLE001  the gate must never break a call
                pass
            await self.push_frame(frame, direction)

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
                try:
                    db = frame_db(frame.audio)
                except Exception:  # noqa: BLE001
                    db = 0.0
                self._pending.append((frame, db))
                if len(self._pending) > self.LOOKAHEAD:
                    await self._release(direction)
                return
            if isinstance(frame, (EndFrame, CancelFrame)):
                while self._pending:
                    await self._release(FrameDirection.DOWNSTREAM)
            await self.push_frame(frame, direction)

    return VoiceGate()


# ---------------- layer 2: background speech filter (after Sarvam) ----------------
class NearTalk:
    """Remembers WHEN there was sound about as loud as the candidate's own voice ("near talk").
    Words heard while there was none came from somebody else."""

    STEP = 0.02      # one audio frame (20 ms)
    KEEP_SECS = 60.0

    def __init__(self, near: float = 10.0, need_secs: float = 0.2):
        self.level = GateLogic(near)   # the same level-learning as the gate, but stricter
        self.near = near
        self.need = need_secs
        self.marks: collections.deque = collections.deque()  # times of candidate-loud frames
        self.turn_start = None
        self.dropped = self.kept = 0

    def audio(self, db: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.level.keep(db)
        lvl = self.level.level
        if db > GateLogic.FLOOR_DB and (lvl is None or db >= lvl - self.near):
            self.marks.append(now)
        while self.marks and now - self.marks[0] > self.KEEP_SECS:
            self.marks.popleft()

    def near_secs(self, start: float, end: float) -> float:
        return sum(1 for t in self.marks if start <= t <= end) * self.STEP

    def learning(self) -> bool:
        return self.level.level is None

    def is_candidate(self, final: bool, now: float | None = None) -> bool:
        """Did the words that just arrived come with candidate-loud sound?"""
        now = time.monotonic() if now is None else now
        if self.learning():
            return True
        if final:
            start = (self.turn_start or now - 3.0) - 0.8  # the whole sentence (+ Sarvam's delay)
            ok = self.near_secs(start, now) >= self.need
        else:
            ok = self.near_secs(now - 1.0, now) >= min(self.need, 0.2)  # live words: the last second
        return ok


def make_background_filter(label: str = ""):
    """(meter, filter): the meter goes right before speech-to-text, the filter right after it
    (after the transcript fixer). Returns (None, None) when audio.background_filter is false."""
    if not filter_enabled():
        return None, None
    from pipecat.frames.frames import (
        CancelFrame,
        EndFrame,
        InputAudioRawFrame,
        InterimTranscriptionFrame,
        TranscriptionFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    talk = NearTalk(near_db(), min_speech_secs())

    class LoudnessMeter(FrameProcessor):
        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
                try:
                    talk.audio(frame_db(frame.audio))
                except Exception:  # noqa: BLE001  measuring must never break a call
                    pass
            await self.push_frame(frame, direction)

    class BackgroundSpeechFilter(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.talk = talk
            self._logged_interim = ""

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
                final = isinstance(frame, TranscriptionFrame)
                text = (frame.text or "").strip()
                try:
                    if talk.turn_start is None:
                        talk.turn_start = time.monotonic()
                    ok = not text or talk.is_candidate(final)
                except Exception:  # noqa: BLE001
                    ok = True
                if final:
                    talk.turn_start = None
                if not ok:
                    if final:
                        talk.dropped += 1
                        logger.info(f"[{label}]   [noise] ignored background speech (not the candidate): {text[:60]!r}")
                    elif text.split()[:2] != self._logged_interim.split()[:2]:
                        self._logged_interim = text
                        logger.info(f"[{label}]   [noise] background voices, not stopping the agent: {text[:40]!r}")
                    return  # never reaches the interruption check or the AI
                if final:
                    talk.kept += 1
            elif isinstance(frame, (EndFrame, CancelFrame)) and (talk.dropped or talk.kept):
                logger.info(f"[{label}] [noise] background filter: ignored {talk.dropped} background sentence(s), "
                            f"kept {talk.kept} from the candidate")
            await self.push_frame(frame, direction)

    return LoudnessMeter(), BackgroundSpeechFilter()


# ---------------- layer 3: Voice Focus (optional) ----------------
def voice_focus_filter(label: str = ""):
    """ai-coustics Voice Focus as the transport's input filter, or None (off, or not set up).
    Never stops a call: if anything is missing it logs why and the call runs without it."""
    c = _cfg()
    if not c.get("voice_focus"):
        return None
    import os

    key = os.getenv("AIC_LICENSE_KEY", "").strip()
    if not key:
        logger.warning(f"[{label}] audio.voice_focus is on but AIC_LICENSE_KEY is not in .env: running without it")
        return None
    try:
        from pipecat.audio.filters.aic_filter import AICFilter
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{label}] Voice Focus not installed ({e}): pip install \"pipecat-ai[aic]==1.12.0\"")
        return None
    try:
        return AICFilter(license_key=key, model_id=str(c.get("voice_focus_model") or "quail-vf-2.2-s-16khz"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{label}] Voice Focus could not start ({e}): running without it")
        return None
