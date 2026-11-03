"""One question per turn, enforced in code (not only in the prompt).

Sits between the LLM and the voice (text-to-speech). Small models sometimes write the whole
interview in one reply: "What is your expected CTC? What is your notice period? Are you in
Chennai? Let me read back: ... notice period ___". This filter lets Priya say everything up to
and including her FIRST question, then drops the rest of that reply. So she must stop and wait
for the candidate's answer, like a person would.

It also stops a reply at:
  - "[" ......... (the model writing "[End call]" or other notes as text)
  - "..." / "___" runs (the model padding or leaving blanks for answers it does not have)

Function calls (end_call, schedule_callback) are separate frames and always pass through.
Only what was actually spoken is saved in the conversation history (the assistant aggregator
sits after the voice), so the model sees that it asked one question and waits.
"""

from __future__ import annotations

import re

from loguru import logger

from pipecat.frames.frames import LLMContextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Where a reply must stop: after the first "?", or before "[", "....", "___"
STOP_AT = re.compile(r"\?|\[|\.{4,}|_{3,}|…{2,}")
MAX_SENTENCES = 3   # a phone reply longer than this turns into a speech: cut it
SENTENCE_END = re.compile(r"[.!](?=\s|$)")
MARKDOWN = re.compile(r"\*+|`+|^#+\s*|^\s*[-•]\s+", re.M)


class OneQuestionPerTurn(FrameProcessor):
    def __init__(self, label: str = "", **kwargs):
        super().__init__(**kwargs)
        self._cut = False
        self._dropped = ""
        self._sentences = 0
        self._prefix = f"[{label}] " if label else ""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._cut, self._dropped, self._sentences = False, "", 0
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._dropped.strip():
                logger.info(f"{self._prefix}  [guard] one question per turn: dropped "
                            f"{len(self._dropped)} chars ({self._dropped.strip()[:70]}...)")
            self._cut, self._dropped = False, ""
        elif isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._cut:
                self._dropped += frame.text or ""
                return  # swallow the rest of this reply
            # markdown is read aloud by the voice ("star star plus nine one..."): remove it
            text = MARKDOWN.sub("", frame.text or "")
            if not text:
                return
            frame.text = text
            m = STOP_AT.search(text)
            for e in SENTENCE_END.finditer(text):
                if m and e.start() >= m.start():
                    break
                self._sentences += 1
                if self._sentences >= MAX_SENTENCES:  # end the reply after this sentence
                    self._cut = True
                    self._dropped += text[e.end():]
                    frame.text = text[: e.end()]
                    await self.push_frame(frame, direction)
                    return
            if m:
                keep = text[: m.end()] if m.group() == "?" else text[: m.start()]
                self._cut = True
                self._dropped += text[len(keep):]
                if keep.strip():
                    frame.text = keep  # same frame (keeps its spacing flags), shortened
                    await self.push_frame(frame, direction)
                return

        await self.push_frame(frame, direction)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


class NoReplyWithoutNewWords(FrameProcessor):
    """Stops Suguna answering twice in a row.

    When her reply is cut by a noise or a very short sound, the turn can "end" again with no new
    words from the candidate, and the AI is asked again with the same conversation: it then says
    the same thing a second time ("I'm sorry... could you share..." / "I'm really sorry... please
    let me know..."). This sits before the AI and skips a request whose last message is already
    Suguna's own reply (nothing new from the candidate). Tool results still get an answer."""

    def __init__(self, label: str = "", **kwargs):
        super().__init__(**kwargs)
        self._prefix = f"[{label}] " if label else ""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            try:
                ctx = frame.context
                msgs = ctx.get_messages() if hasattr(ctx, "get_messages") else ctx.messages
                last = next((m for m in reversed(msgs) if isinstance(m, dict) and m.get("role") != "system"), None)
                if last and last.get("role") == "assistant" and not last.get("tool_calls") and _text(last.get("content")).strip():
                    logger.info(f"{self._prefix}  [guard] no new words from the candidate: not replying twice")
                    return
            except Exception:  # noqa: BLE001  never block a reply because of this check
                pass
        await self.push_frame(frame, direction)
