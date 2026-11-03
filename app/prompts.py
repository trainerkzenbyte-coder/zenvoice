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
        "agent_name": agent.get("name", "Priya"),
        "company": agent.get("company", "our company"),
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
    greeting_cfg: dict | None = None, phone: str = "", memory_name: str = "", browser_test: bool = False,
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
            + contact_memory(phone, name=memory_name, browser=browser_test) + prompt_section() + knowledge_grounding())


# ---------- Sprint 9: sound human + what NOT to talk about (Agents > Guardrails) ----------
# ---------- Sprint 9: sound human + what NOT to talk about (Agents > Guardrails) ----------
DEFAULT_DEFLECT = "Our HR operations team handles those specific details during onboarding, and I can have them share that with you."
DEFAULT_OFF_LIMITS = [
    "legal, visa, tax (like Form 16) or medical advice",
    "other candidates' personal contact information before signing the agreement",
    "personal questions about you, the recruiter or the team",
    "politics, religion or anything offensive",
]
NATURAL_STYLE = """

# HOW TO SOUND LIKE A REAL PERSON ON THE PHONE (MICRO-CHUNKING)
- MAXIMUM 2 SHORT SENTENCES PER TURN (under 30 words total). Never speak a third sentence in a single turn.
- If a topic (like the placement process or timeline) has many details, give ONLY the first 1-2 sentences, then pause and ask a quick check-in like "Does that make sense?" or "Should I share how our live interview support works?"
- First react briefly to what they just said ("Got it.", "Makes sense.", "Sure thing."), then say your 1-2 sentences. Never use "That's a good question" or "Great question".
- Everyday spoken Indian English with contractions. No numbered lists, no bullet points, no "etc." or "et cetera".
- Ask ONLY ONE question at a time. Never combine tech stack, experience, and notice period into one turn.
- If they answer several questions at once, accept all of it and only ask what is still missing."""

def conversation_cfg() -> dict:
    return settings.get("conversation") or {}


def conversation_rules() -> str:
    cfg = conversation_cfg()
    out = NATURAL_STYLE if cfg.get("natural_style", True) else ""
    topics = [str(t).strip() for t in (cfg.get("off_limits") or DEFAULT_OFF_LIMITS) if str(t).strip()]
    deflect = str(cfg.get("deflect_line") or DEFAULT_DEFLECT).strip()
    out += ("\n\n# WHAT YOU MUST NOT TALK ABOUT\n"
            "If the candidate asks about any of these, do not guess, explain or make anything up:\n"
            + "\n".join(f"- {t}" for t in topics)
            + f'\nInstead say, in your own natural words close to: "{deflect}" '
              "Then continue the conversation (or ask if there's anything else). Never promise anything you were not told.")
    return out


def knowledge_grounding() -> str:
    deflect = str(conversation_cfg().get("deflect_line") or DEFAULT_DEFLECT).strip()
    return (f'\n\nOnly state facts that are in this script or the knowledge base. If the answer is not there, say: "{deflect}"')


# ---------- the opening line (spoken instantly, no AI wait) ----------
# Put in the conversation history before the greeting, so the AI knows the call has
# started and that it already said the opening line. Text tests use the same note.
CALL_CONNECTED_NOTE = (
    "(The call has connected and the candidate picked up. You have already said the opening "
    "line below. Continue the call from there, following the call flow.)"
)
DEFAULT_GREETING = {
    "with_name": "Hello, this is {agent_name} from {company}. Am I speaking with {candidate_name}?",
    "without_name": "Hello, this is {agent_name} from {company}. May I know who I'm speaking with?",
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
def contact_memory(phone: str, days: int = 14, *, name: str = "", browser: bool = False) -> str:
    """If we spoke to this person recently, tell the AI what happened, so a dropped or callback
    call continues naturally ("Hi Karthik, sorry, we got cut off. We were talking about...").

    Phone calls match by number. Browser tests (no number) match earlier browser tests by name."""
    if not conversation_cfg().get("contact_memory", True):
        return ""
    import json
    import time as _t

    from app.call_resume import _digits, _name, _same_person

    digits = _digits(phone)
    if len(digits) < 10:
        digits = ""
    key_name = _name(name)
    log_dir = Path(ROOT) / "logs" / "calls"
    if not (digits or (browser and key_name)) or not log_dir.exists():
        return ""
    cutoff = _t.time() - days * 86400
    for path in sorted(log_dir.glob("*.json"), reverse=True)[:400]:
        if path.stat().st_mtime < cutoff:
            break
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not _same_person(rec, digits, key_name, browser):
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
                "Use only these facts about the candidate; never guess their role, stack or experience. "
                "If the candidate says something different now, believe what they say now. "
                "Don't ask again for what is already known; briefly confirm it if it matters, and continue from where "
                "you left off. If the last call dropped, open with a quick apology for the disconnection.")
    return ""
