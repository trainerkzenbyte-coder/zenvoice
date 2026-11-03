"""Loads Priya's call script and fills in agent, job, candidate and date details.

No Pipecat imports here, so the text simulator and tests can use it too.
"""

import datetime as dt
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from app.config import ROOT, settings


def screening_cfg() -> dict:
    path = Path(ROOT) / "config" / "screening.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def render(text: str, values: dict) -> str:
    """Fill {{name}} (Retell style) and {name} placeholders. Unknown names are left as they are."""
    text = VAR_RE.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), text)
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def prompt_values(*, candidate_name: str = "", now: dt.datetime | None = None, job_title: str = "",
                  variables: dict | None = None) -> dict:
    agent = settings.get("agent", {})
    job = screening_cfg().get("job", {}) or {}
    now = now or dt.datetime.now(ZoneInfo("Asia/Kolkata"))
    values = {
        "agent_name": agent.get("name", "Suguna"),
        "company": agent.get("company", "Zenbyte Technologies"),
        "job_title": job_title.strip() or job.get("title", "the open role"),
        "job_location": job.get("location", "the job location"),
        "job_key_skills": job.get("key_skills", "the key skills"),
        "job_experience_range": job.get("experience_range", "the required experience"),
        "candidate_name": candidate_name.strip() or "unknown (ask politely who you are speaking with)",
        "today": now.strftime("%A %d %B %Y, %I:%M %p").replace(" 0", " "),
    }
    values["role"] = values["job_title"]
    for k, v in (variables or {}).items():  # test variables from the console (strings only)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,40}", str(k)) and str(v).strip():
            values[str(k)] = str(v)[:200]
    return values


def load_prompt(
    filename: str | None = None, *, candidate_name: str = "", now: dt.datetime | None = None,
    job_title: str = "", template: str | None = None, variables: dict | None = None,
    greeting_cfg: dict | None = None, phone: str = "",
) -> str:
    """Call script from config/prompts/ with placeholders filled in, plus the knowledge base.

    Which file: agent.prompt_file in config/agent.yaml (default: sprint3_screening.txt).
    template: use this text instead of the file (console "Test LLM" on unpublished changes).
    Placeholders, as {name} or {{name}}: agent_name company candidate_name job_title (= role)
    job_location job_key_skills job_experience_range today
    """
    agent = settings.get("agent", {})
    if template is None:
        filename = filename or agent.get("prompt_file", "sprint3_screening.txt")
        template = (Path(ROOT) / "config" / "prompts" / filename).read_text(encoding="utf-8")
    values = prompt_values(candidate_name=candidate_name, now=now, job_title=job_title, variables=variables)
    text = render(template, values)
    gcfg = greeting_cfg if greeting_cfg is not None else (screening_cfg().get("greeting") or {})
    if gcfg.get("mode") == "user_first":
        line = greeting_line(candidate_name, cfg=dict(gcfg, mode="ai_first"))
        if line:
            text += ("\n\nOpening: the candidate speaks first when the call connects (usually 'hello'). "
                     f"Reply to that with your opening line: \"{line}\"")
    from app.knowledge import prompt_section  # dashboard: Knowledge Base page
    from app.voice_config import call_language, language_instruction  # Agents > Language

    return (text + conversation_rules() + language_instruction(call_language(settings))
            + contact_memory(phone) + prompt_section() + knowledge_grounding())


# ---------- Sprint 9: sound human + what NOT to talk about (Agents > Guardrails) ----------
DEFAULT_DEFLECT = "That's a good question. My team will reach out to you with those details in the next step."
DEFAULT_OFF_LIMITS = [
    "salary, fees or payment details beyond what the knowledge base says",
    "legal, visa, tax or medical advice",
    "guarantees of a job, offer, interview result or placement",
    "other candidates or other companies' internal information",
    "personal questions about you, the recruiter or the team",
    "politics, religion or anything offensive",
    "anything not covered in this script or the knowledge base",
]
NATURAL_STYLE = """
# HOW TO TALK (most important; this beats any script below)
You are having a real phone conversation, not filling in a form. The candidate leads; you follow.

Their question or doubt always comes first:
- If they ask anything, answer it properly first. Do NOT tack your next question onto the end of your answer.
- Only go back to your own questions once they seem satisfied ("okay", "got it", silence, or moving on).
- If they still seem unsure, ask "Does that make sense?" or "Anything else you want to check?"

Never ask the same thing twice in a row:
- If they skipped or dodged a question, drop it for now and come back to it at most once later, or let it go.
- If they gave an answer you didn't expect (a different city, "not sure", "depends"), accept it gracefully.

Handle questions about the conversation like a person would:
- "What did you ask?" / "What are you asking?": say sorry lightly and repeat your last question naturally, e.g., "Ah sorry, I was just asking how many years you've worked."
- "Do you understand English/Tamil?": answer warmly and switch to the language they prefer.
- "Who are you?" / "Why are you calling?": one plain sentence about who you are and why you're calling.
- If they sound annoyed or confused ("stop it", "I don't understand", sighing): slow down immediately, apologize, and check what they need. Don't push your next question in that turn.

Adapt to the person:
- If they say they're a fresher, a student or not working, never ask about notice period or current company. Ask what they want next instead.
- Skip any question their earlier answers already make irrelevant or have already answered.

Short replies:
- "Yes", "yeah", "okay", "hmm", "haan": take it as a go-ahead and continue. Never repeat your greeting if they say this.
- "Yes?" or "Hello?" right after your greeting means they didn't catch you. Say who you are in one short line.

Sound human:
- One or two short sentences, then stop and let them talk. No lists, no reading long text blocks.
- React to what they said before moving on ("Oh nice.", "Ah, okay.", "That's fair.") and use natural fillers ("actually", "sure").
- Spoken Indian English with contractions. Numbers as spoken: "five lakhs", "thirty days", "two point seven five lakhs".
- If you didn't catch something: "Sorry, the line broke a bit, could you say that again?"
"""

def conversation_cfg() -> dict:
    return settings.get("conversation") or {}


def conversation_rules() -> str:
    # Only keep natural conversational rhythm; strip out forced deflections
    cfg = conversation_cfg()
    return NATURAL_STYLE if cfg.get("natural_style", True) else ""


def knowledge_grounding() -> str:
    # Do not inject the forced 'my team will reach out' sentence
    return ""


# ---------- the opening line (spoken instantly, no AI wait) ----------
# Put in the conversation history before the greeting, so the AI knows the call has
# started and that it already said the opening line. Text tests use the same note.
CALL_CONNECTED_NOTE = (
    "(The call has connected and the candidate picked up. You have already said the opening "
    "line below. Continue the call from there, following the call flow.)"
)
DEFAULT_GREETING = {
    "with_name": "Hello, this is Suguna from Zenbyte Technologies. Am I speaking with {candidate_name}?",
    "without_name": "Hello, this is Suguna from Zenbyte Technologies. Am I speaking with you regarding your job search?",
}


def greeting_cfg() -> dict:
    return screening_cfg().get("greeting") or {}


def greeting_mode(cfg: dict | None = None) -> str:
    """ai_first (default): Priya speaks the moment the call connects.
    user_first: Priya waits for the candidate's "hello" (falls back to speaking after a few seconds)."""
    cfg = greeting_cfg() if cfg is None else cfg
    return "user_first" if cfg.get("mode") == "user_first" else "ai_first"


def greeting_delay(cfg: dict | None = None) -> float:
    """Pause after the call connects before Priya's greeting (0-5 s)."""
    cfg = greeting_cfg() if cfg is None else cfg
    try:
        return max(0.0, min(5.0, float(cfg.get("delay_secs", 0) or 0)))
    except (TypeError, ValueError):
        return 0.0


def greeting_line(candidate_name: str = "", cfg: dict | None = None, variables: dict | None = None) -> str:
    """Priya's first sentence (config/screening.yaml -> greeting).

    Spoken the moment the call connects instead of waiting ~1-3 s for the AI to write it.
    Set greeting.enabled: false to go back to an AI-written greeting (returns "").
    Returns "" in user_first mode (nothing is spoken until the candidate talks)."""
    cfg = greeting_cfg() if cfg is None else cfg
    if cfg.get("enabled") is False or greeting_mode(cfg) == "user_first":
        return ""
    name = candidate_name.strip()
    template = str(cfg.get("with_name" if name else "without_name")
                   or DEFAULT_GREETING["with_name" if name else "without_name"])
    values = prompt_values(candidate_name=name, variables=variables)
    values["candidate_name"] = name
    return " ".join(render(template, values).split())


def fallback_greeting(candidate_name: str = "") -> str:
    """User-first mode: what Priya says if the candidate stays silent after picking up."""
    cfg = dict(greeting_cfg(), mode="ai_first")
    return greeting_line(candidate_name, cfg=cfg)


def opening_messages(greeting: str) -> list[dict]:
    """Conversation history at call start: the 'call connected' note + the greeting."""
    if not greeting:
        return []
    return [
        {"role": "user", "content": CALL_CONNECTED_NOTE},
        {"role": "assistant", "content": greeting},
    ]


def test_candidate_name() -> str:
    """Name used for laptop and text tests (config/screening.yaml -> test_candidate)."""
    return str((screening_cfg().get("test_candidate") or {}).get("name", "")).strip()



# ---------- Sprint 9: contact memory (like Retell's "Use contact's memory") ----------
def contact_memory(phone: str, days: int = 14) -> str:
    """If we spoke to this number recently, tell the AI what happened, so a dropped or callback
    call continues naturally ("Hi Karthik, sorry, we got cut off. We were talking about...")."""
    if not phone or not conversation_cfg().get("contact_memory", True):
        return ""
    import json
    import time as _t

    digits = re.sub(r"\D", "", phone)[-10:]
    log_dir = Path(ROOT) / "logs" / "calls"
    if len(digits) < 10 or not log_dir.exists():
        return ""
    cutoff = _t.time() - days * 86400
    for path in sorted(log_dir.glob("*.json"), reverse=True)[:400]:
        if path.stat().st_mtime < cutoff:
            break
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if re.sub(r"\D", "", str(rec.get("phone", "")))[-10:] != digits or str(rec.get("call_id", "")).startswith(("loadtest-", "browser-")):
            continue
        out = rec.get("outcome") or {}
        summary = str(out.get("summary") or "").strip()
        data = {k: v for k, v in (out.get("data") or {}).items() if v not in (None, "", [], "unknown")}
        if not summary and not data:
            return ""
        facts = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in list(data.items())[:12])
        when = str(rec.get("started_at") or "")[:16]
        return ("\n\n# PREVIOUS CALL WITH THIS CANDIDATE" + (f" ({when})" if when else "") + "\n"
                + (f"Summary: {summary}\n" if summary else "") + (f"Already known: {facts}\n" if facts else "")
                + f"How it ended: {out.get('call_outcome') or rec.get('end_reason') or 'unknown'}.\n"
                "Don't ask again for what is already known; briefly confirm it if it matters, and continue from where "
                "you left off. If the last call dropped, open with a quick apology for the disconnection.")
    return ""
