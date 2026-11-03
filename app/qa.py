"""AI quality assurance: a transparent, rule-based score for every answered phone call.

Score out of 100 (voicemails are not scored):
    30  the call reached a clear result (completed, not interested, wrong person, do not call,
        language callback, or a callback WITH a time). Dropped / technical failure = 0
    25  data captured: on completed calls, share of the key fields filled
        (experience, current CTC, expected CTC, notice period, location); other results = 25
    20  responsiveness: median reply delay <= 1.5 s = 20, <= 2.5 s = 10, slower = 0 (unknown = 10)
    15  no Gemini errors during the call
    10  nothing flagged for the recruiter to verify (1 flag = 5, more = 0)

Candidate questions are detected from the transcript (sentences with "?" or starting with a
question word). A question counts as resolved when Priya's next reply is not a deferral such
as "the recruiter will share that".
"""

import re
from collections import defaultdict

KEY_FIELDS = ["total_experience_years", "current_ctc_lpa", "expected_ctc_lpa", "notice_period_days", "current_location"]
RESOLVED = {"completed", "not_interested", "wrong_person", "do_not_call", "language_barrier"}
QUESTION_START = re.compile(
    r"^(what|when|where|which|who|whom|why|how|is|are|can|could|will|would|do|does|did|should|may|any|kya|kab|kaun|kitna)\b",
    re.I)
DEFERRAL = re.compile(r"recruiter (will|can|would) (share|tell|explain|confirm|get back)|i don't have|i do not have|"
                      r"not sure|don't know|do not know|get back to you|check and (let|tell)", re.I)


def score_call(rec: dict) -> dict | None:
    out = rec.get("outcome") or {}
    outcome = out.get("call_outcome")
    if not outcome or outcome == "voicemail":
        return None
    data = out.get("data") or {}
    actions = out.get("actions") or []
    errors = sum(1 for a in actions if a.get("action") == "llm_error")
    ended_error = out.get("ended_by") == "error"

    if ended_error or outcome == "dropped":
        result = 0
    elif outcome in RESOLVED or (outcome == "callback_requested" and data.get("callback_time")):
        result = 30
    else:
        result = 10
    if outcome == "completed":
        captured = sum(1 for f in KEY_FIELDS if data.get(f) not in (None, ""))
        capture = round(25 * captured / len(KEY_FIELDS))
    else:
        capture = 25 if not (ended_error or outcome == "dropped") else 0
    p50 = (rec.get("latency") or {}).get("p50_ms")
    speed = 10 if p50 is None else 20 if p50 <= 1500 else 10 if p50 <= 2500 else 0
    reliability = 15 if errors == 0 else 0
    checks = len(out.get("checks") or [])
    clean = 10 if checks == 0 else 5 if checks == 1 else 0
    total = result + capture + speed + reliability + clean
    return {"score": total, "parts": {"result": result, "data": capture, "speed": speed, "reliability": reliability,
                                      "clean": clean},
            "resolved": result == 30, "llm_errors": errors}


def _norm(q: str) -> str:
    q = re.sub(r"[^\w\s]", " ", q.lower())
    q = re.sub(r"\b(please|sir|madam|ma'am|ok|okay|actually|so|and|but|um|uh)\b", " ", q)
    return " ".join(q.split()[:10])


def questions(transcript: list[dict]) -> list[tuple[str, str, bool]]:
    """[(normalized key, original text, resolved)] for each candidate question."""
    out = []
    for i, turn in enumerate(transcript or []):
        if turn.get("role") != "user":
            continue
        for sentence in re.split(r"(?<=[?.!])\s+", str(turn.get("text", ""))):
            s = sentence.strip()
            if len(s.split()) < 3 or not ("?" in s or QUESTION_START.match(s)):
                continue
            reply = next((t.get("text", "") for t in transcript[i + 1:] if t.get("role") == "bot"), "")
            out.append((_norm(s), s.rstrip("?") + "?", bool(reply) and not DEFERRAL.search(reply)))
    return out


def top_questions(records: list[dict], limit: int = 10) -> list[dict]:
    groups: dict[str, dict] = defaultdict(lambda: {"count": 0, "resolved": 0, "example": ""})
    for rec in records:
        for key, text, ok in questions(rec.get("transcript") or []):
            g = groups[key]
            g["count"] += 1
            g["resolved"] += int(ok)
            g["example"] = g["example"] or text
    rows = [{"question": g["example"], "count": g["count"], "resolved": g["resolved"],
             "resolution_rate": round(100 * g["resolved"] / g["count"], 1)} for g in groups.values()]
    return sorted(rows, key=lambda r: (-r["count"], r["question"]))[:limit]
