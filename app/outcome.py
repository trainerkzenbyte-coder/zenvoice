"""Post-call extraction: transcript -> structured screening outcome (JSON).

Runs AFTER the call ends, so it adds no delay to the conversation.

    outcome = await extract_outcome(transcript)
    outcome["call_outcome"]      -> "completed"
    outcome["data"]              -> {"total_experience_years": 4.5, "expected_ctc_lpa": 12, ...}
    outcome["summary"]           -> "Java developer at Infosys, 4.5 years..."
    outcome["checks"]            -> ["expected CTC is 2.6x current - please verify"]

Two layers of safety:
1. Gemini fills a strict JSON schema built from config/screening.yaml and is told
   never to guess (missing = left out).
2. Python re-validates every value: types, ranges, allowed values, dates, and
   sanity checks a recruiter would want flagged.
"""

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from app import gemini_rest
from app.config import ROOT

SCREENING_FILE = Path(ROOT) / "config" / "screening.yaml"


def load_screening() -> dict:
    return yaml.safe_load(SCREENING_FILE.read_text(encoding="utf-8"))


# ---------- transcript ----------
def format_transcript(transcript: list[dict], agent_name: str = "Priya") -> str:
    lines = []
    for turn in transcript:
        who = agent_name if turn.get("role") == "bot" else "Candidate"
        text = str(turn.get("text", "")).strip()
        if text:
            lines.append(f"{who}: {text}")
    return "\n".join(lines)


# ---------- schema + prompt ----------
_JSON_TYPES = {"string": "string", "number": "number", "integer": "integer",
               "boolean": "boolean", "enum": "string"}


SENTIMENTS = ["positive", "neutral", "negative"]


def build_schema(cfg: dict) -> dict:
    data_props = {}
    for name, f in cfg["fields"].items():
        prop = {"type": _JSON_TYPES[f["type"]], "description": f.get("description", "")}
        if f["type"] == "enum":
            prop["enum"] = list(f["values"])
        data_props[name] = prop
    return {
        "type": "object",
        "properties": {
            "call_outcome": {"type": "string", "enum": list(cfg["call_outcomes"].keys())},
            "data": {"type": "object", "properties": data_props},
            "summary": {"type": "string", "description": "2-3 sentences for the recruiter"},
            "recruiter_notes": {"type": "string", "description": "Concerns, ranges, follow-ups; empty if none"},
            "user_sentiment": {"type": "string", "enum": SENTIMENTS,
                               "description": "The candidate's overall mood in the call: positive, neutral or negative"},
        },
        "required": ["call_outcome", "data", "summary"],
    }


def build_prompt(cfg: dict, transcript_text: str, now: dt.datetime) -> str:
    job = cfg.get("job", {})
    outcomes = "\n".join(f"- {k}: {v}" for k, v in cfg["call_outcomes"].items())
    fields = "\n".join(
        f"- {k} ({f['type']}{': ' + '/'.join(f['values']) if f['type'] == 'enum' else ''}): "
        f"{f.get('description', '')}"
        for k, f in cfg["fields"].items()
    )
    return f"""You extract structured screening data from a recruiter phone call transcript.

Job being screened for: {job.get('title', '')} in {job.get('location', '')}; key skills {job.get('key_skills', '')}.
Current date and time in India: {now.strftime('%A %Y-%m-%d %H:%M')}. Use it to resolve "tomorrow", "Monday", "after 6".

Rules:
1. Fill a field ONLY if the candidate clearly stated it in this call. If not stated, leave the field out. Never guess or infer from the job description.
2. Convert units exactly as described in each field. Salaries are lakhs per annum (LPA): monthly amount x 12 / 100000. "Thousand" = 1,000. "Lakh" = 100,000. "Crore" = 100 lakh.
3. If the candidate corrects themselves, use the corrected value.
4. Transcripts come from speech recognition and may contain small errors (Hinglish, Indian accents). Read for meaning.
5. Mention ranges, uncertainty, or anything a recruiter should double-check in recruiter_notes.
6. The summary is 2-3 plain sentences for a recruiter: who they are, key numbers, interest and next step.

Call outcomes (choose exactly one):
{outcomes}

Fields inside "data":
{fields}

Transcript:
{transcript_text}

Return JSON only."""


# ---------- validation ----------
def _to_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def validate(raw: dict, cfg: dict) -> tuple[dict, list[str]]:
    """Clean the model output. Returns (clean_outcome, checks_for_recruiter)."""
    checks: list[str] = []
    fields = cfg["fields"]
    outcome = raw.get("call_outcome")
    if outcome not in cfg["call_outcomes"]:
        checks.append(f"unknown call outcome '{outcome}' replaced with 'dropped'")
        outcome = "dropped"

    data_in = raw.get("data") or {}
    data: dict = {}
    for name, value in data_in.items():
        f = fields.get(name)
        if f is None or value is None or value == "":
            continue  # unknown field or empty: drop silently
        t = f["type"]
        if t in ("number", "integer"):
            num = _to_number(value)
            if num is None:
                checks.append(f"{name}: '{value}' is not a number, dropped")
                continue
            if t == "integer":
                num = int(round(num))
            if ("min" in f and num < f["min"]) or ("max" in f and num > f["max"]):
                checks.append(f"{name}: {num} is outside {f.get('min')}-{f.get('max')}, please verify")
            data[name] = round(num, 2) if t == "number" else num
        elif t == "boolean":
            if isinstance(value, bool):
                data[name] = value
            elif str(value).lower() in ("true", "yes"):
                data[name] = True
            elif str(value).lower() in ("false", "no"):
                data[name] = False
        elif t == "enum":
            v = str(value).strip().lower()
            if v in f["values"]:
                data[name] = v
            else:
                checks.append(f"{name}: '{value}' not one of {f['values']}, dropped")
        else:
            data[name] = str(value).strip()

    # Date fields: must parse
    for name, fmt in (("last_working_day", "%Y-%m-%d"), ("callback_time", "%Y-%m-%dT%H:%M")):
        if name in data:
            try:
                dt.datetime.strptime(data[name][: len(fmt) + 2].replace(" ", "T"), fmt)
            except ValueError:
                checks.append(f"{name}: '{data[name]}' is not a clear date/time, please verify")

    # Recruiter sanity checks
    cur, exp = data.get("current_ctc_lpa"), data.get("expected_ctc_lpa")
    if cur and exp:
        ratio = exp / cur
        if ratio > 2.0:
            checks.append(f"expected CTC is {ratio:.1f}x current ({cur} -> {exp} LPA), please verify")
        elif ratio < 0.9:
            checks.append(f"expected CTC is below current ({cur} -> {exp} LPA), please verify")
    tot, rel = data.get("total_experience_years"), data.get("relevant_experience_years")
    if tot is not None and rel is not None and rel > tot:
        checks.append(f"relevant experience ({rel}) is more than total ({tot}), please verify")
    if outcome == "callback_requested" and "callback_time" not in data:
        checks.append("callback requested but no time given")

    clean = {
        "call_outcome": outcome,
        "data": data,
        "summary": str(raw.get("summary", "")).strip(),
        "recruiter_notes": str(raw.get("recruiter_notes", "") or "").strip(),
        "checks": checks,
    }
    return clean, checks


# ---------- main entry ----------
async def extract_outcome(
    transcript: list[dict], *, now: dt.datetime | None = None, model: str | None = None,
    agent_name: str = "Priya",
) -> dict:
    cfg = load_screening()
    ex = cfg.get("extraction", {})
    now = now or dt.datetime.now(ZoneInfo(ex.get("timezone", "Asia/Kolkata")))

    user_turns = [t for t in transcript if t.get("role") == "user" and str(t.get("text", "")).strip()]
    if not user_turns:
        # Nobody spoke back: no need to spend an LLM call.
        return {"call_outcome": "dropped", "data": {}, "summary": "No response from the person called.",
                "recruiter_notes": "", "checks": [], "extraction_model": None}

    text = format_transcript(transcript, agent_name)
    raw_text = await gemini_rest.generate(
        [{"role": "user", "parts": [{"text": build_prompt(cfg, text, now)}]}],
        model=model or ex.get("model"),
        thinking_cfg={"thinking_level": ex.get("thinking_level", "low")},
        json_schema=build_schema(cfg),
        temperature=0.0,
        max_tokens=1500,
        max_wait_secs=180,  # runs after the call, so waiting out a rate limit is fine
    )
    try:
        raw = gemini_rest.parse_json(raw_text)
    except Exception as e:  # noqa: BLE001
        return {"call_outcome": "dropped", "data": {}, "summary": "", "recruiter_notes": "",
                "checks": [f"extraction failed: could not read model JSON ({e})"],
                "extraction_model": model or ex.get("model")}
    clean, _ = validate(raw, cfg)
    sentiment = str(raw.get("user_sentiment") or clean.get("data", {}).get("user_sentiment") or "").lower()
    clean["user_sentiment"] = sentiment if sentiment in SENTIMENTS else None  # Call History "User Sentiment"
    clean["extraction_model"] = model or ex.get("model")
    return clean


def reconcile(outcome: dict, state) -> dict:
    """Merge what Priya DID during the call (app.call_actions.CallState) into the outcome.

    - A validated callback time from schedule_callback wins over the transcript reading.
    - A do-not-call request is always honoured (compliance).
    - Any other disagreement is kept as extracted but flagged for the recruiter.
    """
    if state is None:
        return outcome
    data = outcome.setdefault("data", {})
    checks = outcome.setdefault("checks", [])
    outcome["actions"] = list(state.actions)
    outcome["ended_by"] = state.ended_by

    if state.callback_time:
        data["callback_time"] = state.callback_time
        checks[:] = [
            c for c in checks
            if not c.startswith(("callback requested but no time", "callback_time:"))
        ]
    if state.callback_language and "preferred_language" not in data:
        data["preferred_language"] = state.callback_language

    if state.ended_by == "transfer":
        outcome["recruiter_notes"] = ((outcome.get("recruiter_notes") or "") +
                                      f" Transferred live to {getattr(state, 'transferred_to', '') or 'a recruiter'}.").strip()
    agent_reason = state.end_reason
    if agent_reason == "do_not_call":
        if outcome.get("call_outcome") != "do_not_call":
            checks.append(f"Priya recorded do-not-call; outcome changed from '{outcome.get('call_outcome')}'")
        outcome["call_outcome"] = "do_not_call"
    elif agent_reason and agent_reason not in ("other",) and agent_reason != outcome.get("call_outcome"):
        checks.append(
            f"Priya ended the call as '{agent_reason}' but the transcript reads as "
            f"'{outcome.get('call_outcome')}', please verify"
        )
    return outcome


def fallback_outcome(error: str, state=None) -> dict:
    """Used when the automatic summary fails (e.g. Gemini rate limit all day).

    The call is still logged and sent to the Sheet, clearly flagged, with whatever
    Priya decided during the call (end reason, callback time). The full transcript
    stays in logs/calls/ for the recruiter."""
    outcomes = set((load_screening().get("call_outcomes") or {}).keys())
    reason = getattr(state, "end_reason", None)
    outcome = {
        "call_outcome": reason if reason in outcomes else "dropped",
        "data": {},
        "summary": "",
        "recruiter_notes": "",
        "checks": [f"Automatic summary failed ({error[:120]}). Read the transcript in the call log."],
        "extraction_model": None,
    }
    return reconcile(outcome, state)


def describe(outcome: dict) -> str:
    """Short, readable printout for terminals."""
    lines = [f"OUTCOME : {outcome['call_outcome']}"]
    for k, v in outcome.get("data", {}).items():
        lines.append(f"  {k:<26} {v}")
    if outcome.get("summary"):
        lines.append(f"SUMMARY : {outcome['summary']}")
    if outcome.get("recruiter_notes"):
        lines.append(f"NOTES   : {outcome['recruiter_notes']}")
    for a in outcome.get("actions", []) or []:
        details = ", ".join(f"{k}={v}" for k, v in a.items() if k not in ("t", "action") and v)
        lines.append(f"ACTION  : {a.get('t', '')} {a.get('action')}({details})")
    if outcome.get("ended_by"):
        lines.append(f"ENDED BY: {outcome['ended_by']}")
    for c in outcome.get("checks", []):
        lines.append(f"CHECK   : {c}")
    return "\n".join(lines)
