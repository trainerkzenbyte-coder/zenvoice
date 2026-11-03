"""Rolling conversation window for long calls (30-45+ minutes).

Without it every turn re-sends the whole conversation: requests get slower, cost more, and
hit the provider's tokens-per-minute limit. The system prompt (script + knowledge base +
guardrails) is NOT part of this history: it is sent separately on every turn, so it is always
complete. Only the back-and-forth is windowed:

    [opening note + greeting]  [what the candidate said earlier, as one short note]  [last N messages]

The note keeps facts the candidate gave long ago (notice period, CTC...), so the AI doesn't
re-ask them, without re-sending every old turn.
"""

from __future__ import annotations

from loguru import logger

from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

MARK = "(Earlier in this call"


def _text(msg) -> str:
    if not isinstance(msg, dict):
        return ""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return ""


def _is_plain_user(msg) -> bool:
    return isinstance(msg, dict) and msg.get("role") == "user" and not msg.get("tool_call_id")


class ContextWindow(FrameProcessor):
    def __init__(self, max_messages: int = 24, keep_first: int = 2, memory_chars: int = 2500, **kwargs):
        super().__init__(**kwargs)
        self.max_messages = max(8, max_messages)
        self.keep_first = keep_first
        self.memory_chars = memory_chars
        self.earlier: list[str] = []  # what the candidate said in trimmed turns

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            try:
                self.trim(frame.context)
            except Exception as e:  # noqa: BLE001  never block a reply because of trimming
                logger.warning(f"Context window: could not trim ({e})")
        await self.push_frame(frame, direction)

    def trim(self, context) -> None:
        msgs = list(context.get_messages())
        head = msgs[: self.keep_first]
        body = [m for m in msgs[self.keep_first:] if not _text(m).startswith(MARK)]
        if body and isinstance(body[0], dict) and body[0].get("role") == "assistant" and _text(body[0]) == "Okay.":
            body = body[1:]  # the acknowledgement that followed our note
        if len(body) <= self.max_messages:
            return
        cut = len(body) - self.max_messages
        while cut < len(body) and not _is_plain_user(body[cut]):
            cut += 1  # start the window at a candidate turn (never split a tool call from its result)
        if cut >= len(body):
            return
        for m in body[:cut]:
            if _is_plain_user(m) and _text(m).strip():
                self.earlier.append(_text(m).strip())
        memory = " | ".join(self.earlier)
        if len(memory) > self.memory_chars:
            memory = "…" + memory[-self.memory_chars:]
        note = [
            {"role": "user", "content": f"{MARK}, older turns were shortened. The candidate said: {memory}. "
                                        "Do not ask again for anything they already told you.)"},
            {"role": "assistant", "content": "Okay."},
        ]
        context.set_messages(head + note + body[cut:])
        logger.debug(f"Context window: kept {len(body) - cut} recent messages, {len(self.earlier)} earlier candidate turns noted")
