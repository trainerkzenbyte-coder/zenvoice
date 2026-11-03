"""Agents > Speech settings > Pronunciation: say a word the way you want, on every voice.

Retell uses IPA/CMU phonemes; Deepgram Aura and Sarvam Bulbul don't take phonemes, so we
rewrite the word into a spelling the voice reads correctly, just before speech is generated:
    Zenbyte -> "Zen byte",  CTC -> "C T C",  Velachery -> "Vela-cheri"
Only the audio changes; the transcript and the AI's memory keep the normal spelling.
"""

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter


def rules_from(settings: dict) -> list[tuple[re.Pattern, str]]:
    rules = []
    for item in ((settings.get("tts") or {}).get("pronunciation") or [])[:100]:
        word, say = str(item.get("word", "")).strip(), str(item.get("say", "")).strip()
        if word and say:
            rules.append((re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE), say))
    return rules


class PronunciationFilter(BaseTextFilter):
    def __init__(self, rules: list[tuple[re.Pattern, str]]):
        super().__init__()
        self._rules = rules

    async def filter(self, text: str) -> str:
        for pattern, say in self._rules:
            text = pattern.sub(say, text)
        return text
