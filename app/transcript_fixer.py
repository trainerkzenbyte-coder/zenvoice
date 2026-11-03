"""Fix words speech recognition commonly mishears BEFORE the AI reads them.

Sits right after speech-to-text in the call pipeline. Example: Deepgram writes "merger stack"
when the candidate says "MERN stack"; this turns it back into "MERN stack", so the AI, the
transcript and the Google Sheet all get the right word.

Built-in fixes cover common IT terms on Indian phone lines. Add your own in config/agent.yaml:

    stt:
      corrections:
        - {heard: "zen bite", write: "Zenbyte"}
        - {heard: "vela cherry", write: "Velachery"}

'heard' is matched as whole words, ignoring case. Several spellings: "a|b|c".
Look at the "[hearing]" lines in the server log to find new mishearings to add.
"""

from __future__ import annotations

import re

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# (what speech recognition writes, what the candidate said). Only fixes that are safe in context:
# "merger" alone is left alone; "merger stack" is not.
DEFAULT_CORRECTIONS: list[tuple[str, str]] = [
    (r"merger stack|merge stack|merged stack|mern stuck|man stack|mon stack|mernst stack|mirn stack|m e r n stack", "MERN stack"),
    (r"merger developer|merge developer|mon developer|man developer|m e r n developer", "MERN developer"),
    (r"mean stuck|m e a n stack", "MEAN stack"),
    (r"java script", "JavaScript"),
    (r"type script", "TypeScript"),
    (r"node j s|node jay s|node js|nod js", "Node.js"),
    (r"react j s|react js|re act js", "React JS"),
    (r"express j s|express js", "Express.js"),
    (r"mongo d b|mongo db|mango db|mongo debby", "MongoDB"),
    (r"my s q l|my sequel|my sql", "MySQL"),
    (r"post grey s q l|postgre sql|post gress", "PostgreSQL"),
    (r"spring boat|spring bot|spring boots", "Spring Boot"),
    (r"micro services|micro service", "microservices"),
    (r"dev ops|deb ops", "DevOps"),
    (r"sales force", "Salesforce"),
    (r"power b i|power bee i", "Power BI"),
    (r"full stuck", "full stack"),
    (r"dot net|dotnet", ".NET"),
    (r"c sharp|see sharp", "C#"),
    (r"angular j s", "AngularJS"),
    (r"jango|d jango", "Django"),
    (r"pie thon|pyton", "Python"),
    (r"a w s", "AWS"),
    (r"cuber net is|kuber netes|cuba net is", "Kubernetes"),
    (r"selenium s", "Selenium"),
    (r"data sign tist", "data scientist"),
    (r"c t c", "CTC"),
    (r"l p a", "LPA"),
    (r"nakri|naukari|nauk ri|knock ri", "Naukri"),
    (r"zen byte|zen bite|zenbite|xen byte|send bite", "Zenbyte"),
    (r"fresh er", "fresher"),
    (r"coimbator|coimbature", "Coimbatore"),
]


def _compile(pairs: list[tuple[str, str]]) -> list[tuple[re.Pattern, str]]:
    out = []
    for heard, write in pairs:
        heard, write = str(heard or "").strip(), str(write or "").strip()
        if not heard or not write:
            continue
        alts = "|".join(a.strip().replace(" ", r"\s+") for a in heard.split("|") if a.strip())
        try:
            out.append((re.compile(rf"(?<![\w]){'(?:' + alts + ')'}(?![\w])", re.I), write))
        except re.error:
            continue  # a bad custom entry never breaks a call
    return out


def rules_from(settings: dict) -> list[tuple[re.Pattern, str]]:
    stt = settings.get("stt") or {}
    custom = []
    for r in stt.get("corrections") or []:
        if isinstance(r, dict):
            custom.append((r.get("heard", ""), r.get("write", "")))
    # Your own fixes first, so they win over the built-in ones.
    return _compile(custom + ([] if stt.get("builtin_corrections") is False else DEFAULT_CORRECTIONS))


def fix_text(text: str, rules: list[tuple[re.Pattern, str]]) -> str:
    for pat, write in rules:
        text = pat.sub(write, text)
    return text


class TranscriptFixer(FrameProcessor):
    def __init__(self, rules: list[tuple[re.Pattern, str]], log_label: str = ""):
        super().__init__()
        self._rules = rules
        self._label = log_label

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)) and frame.text:
            fixed = fix_text(frame.text, self._rules)
            if fixed != frame.text:
                if isinstance(frame, TranscriptionFrame):
                    from loguru import logger

                    logger.info(f"[{self._label}] [heard-fix] {frame.text!r} -> {fixed!r}")
                frame.text = fixed
        await self.push_frame(frame, direction)
