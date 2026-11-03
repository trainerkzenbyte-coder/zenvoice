"""Make the AI notice questions and stop repeating itself.

Speech recognition on phone calls rarely writes "?" (Indian English questions often have no rising
tone, and Sarvam punctuates from the words, not the tone). So "for what I want to visit the office"
reaches the AI as a statement, and the AI pushes its script forward instead of answering.

This sits just before the AI, once per turn:
  1. QUESTION MARK: if what the candidate said looks like a question, a "?" is added to it.
  2. ANSWER FIRST: a one-turn note tells the AI the candidate asked something and must get a real,
     direct answer first, without a new script question stuck on the end in the same breath.
  3. NO REPEATING: if the agent's last replies ended with the same question (e.g. "Which location
     works for you?" three times), a note tells it to stop asking that and deal with their concern.

Notes are replaced every turn (never pile up) and are never spoken. Settings (config/agent.yaml):

    turn:
      question_help: true     # false = switch all of this off
"""

from __future__ import annotations

import difflib
import re

from loguru import logger

from app.config import settings

NOTE_MARK = "(Note for this turn only:"

_OPENERS = r"(?:ok(?:ay)?|yeah|yes|ya|haan|so|but|and|sir|madam|ma'?am|then|i mean|actually|one doubt|hello|no|oh)"
_STARTS = re.compile(
    rf"^(?:{_OPENERS}[\s,]+){{0,3}}"
    r"(what|what's|whats|why|how|when|where|which|who|whom|whose|"
    r"is|are|am|do|does|did|can|could|will|would|should|shall|may|"
    r"isn't|aren't|don't|doesn't|can't|won't|how's)\b", re.I)
_ANYWHERE = re.compile(
    r"\b(for what|what for|why|how much|how many|how long|or not|is it only|only for|is that only|"
    r"do i (need|have)|should i|can i|will i|what is|what are|what's the|what about|how about|"
    r"means what|meaning of|i (want|need) to know|tell me|can you (explain|tell)|explain|"
    r"i (have|had) a (doubt|question)|my (doubt|question)|i asked|is it (mandatory|compulsory|necessary|free)|"
    r"is there|are there|kya|enna|yenna|eppadi|evlo|ethuku|edhukku)\b", re.I)


def enabled() -> bool:
    return (settings.get("turn") or {}).get("question_help", True) is not False


def looks_like_question(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    words = re.sub(r"[^\w' ]+", " ", t.lower()).split()
    if len(words) < 2:
        return False
    clean = " ".join(words)
    if words[0] in ("and", "also", "plus") and len(words) <= 5:
        return True  # "and the fees", "also the timing": a short follow-up question
    return bool(_STARTS.match(clean) or _ANYWHERE.search(clean))


def add_question_mark(text: str) -> str:
    t = str(text or "").rstrip()
    if not t or t.endswith("?"):
        return t
    return t.rstrip(".!,;: ") + "?"


def _last_question(text: str) -> str:
    """The question the agent ended its reply with, if any ('Which location works for you?')."""
    parts = re.split(r"(?<=[.!?])\s+|\s+[-—]\s*", str(text or "").strip())
    for p in reversed([p for p in parts if p.strip()]):
        return p.strip() if p.strip().endswith("?") else ""
    return ""


def repeated_question(assistant_texts: list[str]) -> str:
    """The agent's closing question if its last two replies both ended with (nearly) the same one."""
    qs = [_last_question(t) for t in assistant_texts[-2:]]
    if len(qs) < 2 or not all(qs):
        return ""
    a, b = (re.sub(r"[^\w ]+", "", q.lower()) for q in qs)
    return qs[-1] if difflib.SequenceMatcher(None, a, b).ratio() >= 0.6 else ""


def _text(m) -> str:
    c = (m or {}).get("content", "")
    return c if isinstance(c, str) else ""


def build_note(msgs: list) -> tuple[str, bool, str]:
    """(note text or "", is the last candidate message a question, repeated question)."""
    if not msgs or msgs[-1].get("role") != "user" or msgs[-1].get("tool_call_id"):
        return "", False, ""
    said = _text(msgs[-1])
    asked = looks_like_question(said)
    agent_lines = [_text(m) for m in msgs if m.get("role") == "assistant" and _text(m).strip()]
    repeat = repeated_question(agent_lines)
    parts = []
    if asked:
        parts.append(
            f'the candidate is ASKING you something: "{said.strip()[:200]}". Work out what they really want to '
            "know (speech recognition drops words, so read it generously) and answer THAT, directly and honestly, "
            "in one or two sentences, from your script and knowledge. If the answer isn't there, say you'll check "
            "and get back to them. Don't add your next script question in this same reply; at most end with a "
            "short check like \"Does that clear it up?\"")
    if repeat:
        parts.append(
            f'you have already asked "{repeat}" in your last two replies. Do NOT ask it again now. They haven\'t '
            "answered because they have a doubt or concern: deal with that first, or ask plainly what's "
            "worrying them.")
    if not parts:
        return "", asked, repeat
    return NOTE_MARK + " " + " Also, ".join(parts) + ")", asked, repeat


def make_question_note(label: str = ""):
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    from app.llm_choice import llm_provider

    role = "user" if llm_provider(settings.get("llm") or {})[0] == "google" else "system"

    class QuestionNote(FrameProcessor):
        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM and enabled():
                try:
                    msgs = [m for m in frame.context.get_messages() if not _text(m).startswith(NOTE_MARK)]
                    note, asked, repeat = build_note(msgs)
                    if asked and isinstance(msgs[-1].get("content"), str):
                        msgs[-1]["content"] = add_question_mark(msgs[-1]["content"])  # same message, now with "?"
                    if note:
                        msgs.insert(len(msgs) - 1, {"role": role, "content": note})
                        logger.info(f"[{label}]   [turn] " + ("question heard: answer first" if asked else "")
                                    + (" | " if asked and repeat else "")
                                    + (f"agent repeating {repeat[:40]!r}: told to stop" if repeat else ""))
                    frame.context.set_messages(msgs)
                except Exception as e:  # noqa: BLE001  never block a reply
                    logger.warning(f"[{label}] question note failed: {e}")
            await self.push_frame(frame, direction)

    return QuestionNote()
