"""Auto-detect calls: make Sarvam's voice follow the language the candidate speaks.

Placed right after speech-to-text. Sarvam STT reports the detected language on every final
transcript; when it changes (say English -> Tamil), this sends one settings update down the
pipeline, and Sarvam's voice reconfigures before Priya's next sentence.
"""

from loguru import logger

from pipecat.frames.frames import TranscriptionFrame, TTSUpdateSettingsFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.voice_config import sarvam_language


class LanguageFollower(FrameProcessor):
    def __init__(self, start: str = "en-IN", **kwargs):
        super().__init__(**kwargs)
        self.current = start

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if not isinstance(frame, TranscriptionFrame):
            return
        text = (frame.text or "").strip()
        if len(text.split()) < 2 and len(text) < 8:
            return  # "ok", "haan", "seri": too short to judge the language
        lang = sarvam_language(getattr(frame, "language", None))
        if lang and lang != self.current:
            logger.info(f"Candidate switched language: {self.current} -> {lang}; voice follows")
            self.current = lang
            from pipecat.services.sarvam.tts import SarvamTTSService

            await self.push_frame(TTSUpdateSettingsFrame(delta=SarvamTTSService.Settings(language=lang)))
