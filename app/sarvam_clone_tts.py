"""Your own cloned voice in calls (Sarvam Instant Voice Cloning, beta).

Sarvam's cloned voices use their own endpoint (POST /voices/clone/stream), which Pipecat's
Sarvam services don't call, so this small service does: one request per sentence, audio
streamed back as raw 16-bit PCM at the pipeline's sample rate, so playback starts before
the whole sentence is ready. Interruptions work as with any Pipecat voice.

Price: billed like Bulbul (per character). Create the voice in Agents -> Voice -> My voices.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.sarvam.tts import SarvamHttpTTSService

CLONE_STREAM_URL = "https://api.sarvam.ai/voices/clone/stream"
MAX_CHARS = 1000  # Sarvam's limit per request


class SarvamCloneTTSService(SarvamHttpTTSService):
    def __init__(self, *, api_key: str, voice_id: str, language: str = "en-IN", pace: float = 1.0,
                 sample_rate: int = 24000, **kwargs):
        # The parent class validates a preset model/voice; the cloned voice id is kept separately.
        super().__init__(api_key=api_key, aiohttp_session=None, sample_rate=sample_rate,
                         settings=SarvamHttpTTSService.Settings(model="bulbul:v3", language=language, pace=pace),
                         **kwargs)
        self._clone_voice_id = voice_id
        self._own_session: aiohttp.ClientSession | None = None

    async def _http(self) -> aiohttp.ClientSession:
        if self._own_session is None or self._own_session.closed:
            self._own_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30, sock_read=10))
        return self._own_session

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = (text or "").strip()[:MAX_CHARS]
        if not text:
            return
        form = aiohttp.FormData()
        form.add_field("text", text)
        form.add_field("language_code", str(self._settings.language or "en-IN"))
        form.add_field("voice_id", self._clone_voice_id)
        form.add_field("pace", str(self._settings.pace or 1.0))
        form.add_field("output_audio_codec", "linear16")  # raw PCM, no header
        form.add_field("speech_sample_rate", str(self.sample_rate))
        headers = {"api-subscription-key": self._api_key}
        try:
            session = await self._http()
            async with session.post(CLONE_STREAM_URL, data=form, headers=headers) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:300]
                    logger.warning(f"Sarvam cloned voice error {resp.status}: {detail}")
                    yield ErrorFrame(error=f"Sarvam cloned voice error {resp.status}: {detail}")
                    return
                await self.start_tts_usage_metrics(text)
                leftover = b""
                first = True
                async for chunk in resp.content.iter_chunked(4096):
                    data = leftover + chunk
                    cut = len(data) - (len(data) % 2)  # whole 16-bit samples only
                    data, leftover = data[:cut], data[cut:]
                    if not data:
                        continue
                    if first:
                        await self.stop_ttfb_metrics()
                        first = False
                    yield TTSAudioRawFrame(audio=data, sample_rate=self.sample_rate, num_channels=1,
                                           context_id=context_id)
        except Exception as e:  # noqa: BLE001
            yield ErrorFrame(error=f"Sarvam cloned voice failed: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()

    async def cleanup(self):
        if self._own_session is not None and not self._own_session.closed:
            await self._own_session.close()
        await super().cleanup()
