"""Call resumption: when a recent call with this number ended before it was finished, the next call
picks up where it stopped.

How a call counts as unfinished: the line dropped or the candidate hung up before the agent closed
the call (end_call), or they asked to be called back. NOT resumed: the agent finished the call, the
candidate refused / asked not to be called / was the wrong person, voicemail, or nobody spoke.

On a resumed call:
  - the opening line acknowledges the drop (resume.greeting), spoken instantly
  - a "# THIS IS A FOLLOW-UP CALL" section is added to the NORMAL prompt with when it was, the summary,
    what we already know, the last few lines, and how to behave (short recap, then continue)
  - set resume.separate_prompt: true to use config/prompts/<resume.prompt_file> instead
    (with {{previous_conversation_context}}), the older two-prompt setup.

Settings (config/agent.yaml, all optional):

    resume:
      enabled: true
      window_hours: 48                 # only calls this recent count
      prompt_file: "callback_screening.txt"
      last_lines: 8                    # transcript lines carried over
      outcomes: ["dropped", "callback_requested", "other"]   # which endings get resumed
      greeting: "Hi {candidate_name}, it's {agent_name} from {company} again. Sorry, our call got cut earlier. Is now a good time to continue?"

Finished calls (completed, not interested, do-not-call, wrong person) are NOT resumed: the next call
starts fresh (contact memory still tells the AI what it already knows).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from app.config import ROOT, settings

LOG_DIR = Path(ROOT) / "logs" / "calls"
PROMPTS_DIR = Path(ROOT) / "config" / "prompts"
DEFAULTS = {
    "enabled": True,
    "window_hours": 48,
    "prompt_file": "callback_screening.txt",
    "last_lines": 8,
    "separate_prompt": False,
    "never_for": ["not_interested", "do_not_call", "wrong_person", "voicemail"],
    "greeting": ("Hi {candidate_name}, it's {agent_name} from {company} again. Sorry, our call got cut "
                 "earlier. Is now a good time to continue?"),
}
SKIP_IDS = ("loadtest-", "browser-", "chat-")


def cfg() -> dict:
    return {**DEFAULTS, **(settings.get("resume") or {})}


def _digits(phone: str) -> str:
    return re.sub(r"\D", "", str(phone or ""))[-10:]


# ---- helpers also used by app/prompts.py (contact memory) ----
def _name(value=None, *args, **kwargs) -> str:
    """Normalised person name for comparisons: 'Karthik  R.' -> 'karthik r'. Accepts a name or a call log."""
    if isinstance(value, dict):
        value = value.get("candidate_name") or value.get("name") or ""
    return " ".join(re.sub(r"[^a-z\s]", " ", str(value or "").lower()).split())


def _same_person(*args, **kwargs) -> bool:
    """Is a call log about the same candidate? Same phone number (last 10 digits) and, when both sides
    have a name, the same first name (a shared family number shouldn't mix two people up).
    Accepts the call log and/or phone and name in any order, positionally or by keyword."""
    recs = [a for a in list(args) + list(kwargs.values()) if isinstance(a, dict)]
    texts = [str(a) for a in list(args) + list(kwargs.values()) if isinstance(a, (str, int)) and str(a).strip()]
    phones = [_digits(t) for t in texts if len(_digits(t)) >= 10]
    names = [_name(t) for t in texts if len(_digits(t)) < 10 and _name(t)]
    for k in ("phone", "number"):
        if kwargs.get(k):
            phones.append(_digits(kwargs[k]))
    for k in ("name", "candidate_name"):
        if kwargs.get(k):
            names.append(_name(kwargs[k]))
    for r in recs:
        if r.get("phone"):
            phones.append(_digits(r["phone"]))
        if _name(r):
            names.append(_name(r))
    phones = [p for p in phones if p]
    if len(set(phones)) > 1 or not phones:
        return False
    firsts = {n.split()[0] for n in names if n}
    return len(firsts) <= 1


def check_previous_call(phone: str, *, now: float | None = None) -> tuple[dict | None, str]:
    """(call log to resume or None, why). The reason is printed in the terminal on every call, so
    when a follow-up doesn't happen you can see exactly which call log decided it and why."""
    c = cfg()
    digits = _digits(phone)
    if not c["enabled"]:
        return None, "resume.enabled is false"
    if len(digits) < 10:
        return None, f"no usable phone number for this call ({phone!r})"
    if not LOG_DIR.exists():
        return None, "no call logs yet"
    cutoff = (now or time.time()) - float(c["window_hours"]) * 3600
    skipped = 0
    # Newest first (file names start with YYYYMMDD-HHMMSS); stop at the first one outside the window.
    for path in sorted(LOG_DIR.glob("*.json"), reverse=True)[:500]:
        try:
            if path.stat().st_mtime < cutoff:
                break
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(rec.get("call_id", "")).startswith(SKIP_IDS) or _digits(rec.get("phone", "")) != digits:
            continue
        said = [str(t.get("text", "")).strip() for t in rec.get("transcript") or []
                if t.get("role") == "user" and str(t.get("text", "")).strip()]
        if not said:
            skipped += 1  # unanswered redial, voicemail, a call that failed to start: look further back
            continue
        # The newest call where the candidate spoke decides: if it was finished, don't resume an older one.
        outcome = (rec.get("outcome") or {}).get("call_outcome") or "dropped"
        ended = str(rec.get("end_reason") or "").lower()
        where = f"{path.name} ({outcome}, {ended or 'no end reason'})"
        if outcome in c["never_for"]:
            return None, f"newest call {where}: outcome '{outcome}' is in resume.never_for"
        agent_closed = ended.startswith("ended by agent") or ended.startswith("ended by transfer")
        if agent_closed and outcome != "callback_requested":
            return None, f"newest call {where}: the agent closed it properly, so this call starts fresh"
        if outcome != "callback_requested" and _GOODBYE.search(said[-1]):
            return None, f"newest call {where}: the candidate said goodbye ({said[-1][:40]!r}) and hung up"
        rec["_file"] = path.name
        note = f" (skipped {skipped} newer call(s) where nobody spoke)" if skipped else ""
        return rec, f"resuming {where}{note}"
    return None, (f"no call with this number in the last {c['window_hours']} h"
                  + (f" where the candidate spoke ({skipped} without speech)" if skipped else ""))


# Only a real goodbye counts. "Okay, thank you" is said all through a call, so it does NOT block a resume.
_GOODBYE = re.compile(r"\b(bye|good ?bye|bye bye|see you|talk (to you )?later|that'?s all)\b\W*$", re.I)


def find_previous_call(phone: str, *, now: float | None = None) -> dict | None:
    """The most recent call log with this number inside the window, if it should be resumed."""
    return check_previous_call(phone, now=now)[0]


def where_it_stopped(rec: dict) -> str:
    """The agent's and the candidate's last lines, so the AI knows the exact point to continue from."""
    agent = (settings.get("agent") or {}).get("name") or "Agent"
    turns = [t for t in (rec.get("transcript") or []) if str(t.get("text") or "").strip()]
    last_agent = next((str(t["text"]).strip() for t in reversed(turns) if t.get("role") != "user"), "")
    last_user = next((str(t["text"]).strip() for t in reversed(turns) if t.get("role") == "user"), "")
    out = []
    if last_agent:
        out.append(f'Your last words before the line went: "{last_agent[:300]}"')
    if last_user:
        out.append(f'The candidate\'s last words: "{last_user[:300]}"')
    if turns and turns[-1].get("role") != "user":
        out.append(f"The call cut off while {agent} (you) was speaking or waiting for an answer to that.")
    return "\n".join(out)


def previous_context(rec: dict) -> str:
    """Plain-text block for {{previous_conversation_context}}."""
    c = cfg()
    out = rec.get("outcome") or {}
    lines = [f"When: {str(rec.get('started_at') or '')[:16] or 'recently'}"
             f" ({out.get('call_outcome') or 'ended early'}, {rec.get('end_reason') or 'reason unknown'})"]
    if out.get("summary"):
        lines.append(f"Summary: {str(out['summary']).strip()}")
    known = {k: v for k, v in (out.get("data") or {}).items() if v not in (None, "", [], "unknown")}
    if known:
        lines.append("Already known: " + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in list(known.items())[:12]))
    agent = (settings.get("agent") or {}).get("name") or "Agent"
    turns = [t for t in (rec.get("transcript") or []) if t.get("text")][-int(c["last_lines"]):]
    if turns:
        lines.append("Last part of that call:")
        for t in turns:
            who = "Candidate" if t.get("role") == "user" else agent
            lines.append(f"  {who}: {str(t['text']).strip()[:300]}")
    return "\n".join(lines)


def callback_template(rec: dict) -> str | None:
    """The callback prompt with the history filled in, or None if the file is missing
    (then the call falls back to the standard prompt)."""
    name = str(cfg()["prompt_file"])
    path = (PROMPTS_DIR / name).resolve()
    if PROMPTS_DIR.resolve() not in path.parents or not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    ctx = previous_context(rec)
    # Fill this one placeholder here: load_prompt() caps variables at 200 characters.
    return re.sub(r"\{\{\s*previous_conversation_context\s*\}\}", lambda _m: ctx, text)


RESUME_RULES = """
# THIS IS A FOLLOW-UP CALL (this section overrides the opening steps of the call flow above)
You spoke to {{candidate_name}} recently and that call ended before it was finished (the line dropped
or they had to go). What happened on that call:

{context}

WHERE IT STOPPED
{stopped}

How to handle it:
- Your opening line has already apologised for the disconnection and asked if now is a good time.
  Do NOT greet again, do NOT introduce yourself or the company again, do NOT confirm their name again.
- When they say it's fine, give a very short recap in ONE or TWO sentences (under 40 words), using only
  what is written above, e.g. "So last time you told me you're a React developer with two years, and I
  was explaining our mock interviews." Then continue from WHERE IT STOPPED: if you had asked a question
  they never answered, ask it again; otherwise go on to the next step of the call flow after that point.
- Never start the call flow from the beginning. Never ask again for anything listed under "Already known"
  or anything they already told you in the lines above.
- If they don't remember the call, one plain sentence on who you are and why you called, then continue.
- If it's not a good time, ask when to call back and end politely.
- Only use what is written above about the last call. If something isn't there, don't guess it."""

RESUME_TOP = ("# FOLLOW-UP CALL: you already spoke to this candidate and the call was cut. Read the section "
              "'THIS IS A FOLLOW-UP CALL' at the end of these instructions before anything else: it decides "
              "how you open and where you continue.\n\n")

# Replaces the normal "call connected, follow the call flow" note at the start of the conversation.
RESUME_OPENING_NOTE = ("(The call has connected: this is the follow-up call to the one that was cut. You have "
                       "already said the opening line below. When they reply, follow the section 'THIS IS A "
                       "FOLLOW-UP CALL': short recap, then continue from where it stopped. Do not restart the "
                       "call flow.)")


def resume_block(rec: dict) -> str:
    """The follow-up section added to the normal prompt."""
    return RESUME_RULES.replace("{context}", previous_context(rec)).replace("{stopped}", where_it_stopped(rec) or "(not recorded)")


def resume_opening_messages(greeting: str) -> list[dict]:
    """Conversation history at the start of a follow-up call (instead of prompts.opening_messages)."""
    if not greeting:
        return []
    return [{"role": "user", "content": RESUME_OPENING_NOTE}, {"role": "assistant", "content": greeting}]


def resume_greeting(candidate_name: str) -> str:
    agent = settings.get("agent") or {}
    line = str(cfg()["greeting"] or "")
    values = {"candidate_name": candidate_name or "there", "agent_name": agent.get("name") or "Priya",
              "company": agent.get("company") or "our company"}
    for k, v in values.items():
        line = line.replace("{{" + k + "}}", str(v)).replace("{" + k + "}", str(v))
    return line
