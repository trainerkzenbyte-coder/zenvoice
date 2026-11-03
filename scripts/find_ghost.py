"""Find where a phrase the agent keeps saying (e.g. old pricing) is coming from.

Run from the project folder (virtual environment active):

    python scripts/find_ghost.py "2.5" "3.5" --phone 9876543210

--phone: the number you test-call (contact memory is per number). Without it, that part is skipped.

It does two things:
  1. Rebuilds the EXACT system prompt a phone call gets, splits it into its parts (prompt file,
     conversation rules, language rule, previous-call memory, knowledge base, grounding rule) and
     says which part contains the phrase.
  2. Searches every file in the project (configs, prompts, knowledge stores, drop folders incl.
     PDFs, call logs, agent profiles, .bak files) and lists the matches.
If the phrase is in neither, the AI is inventing it: see the advice printed at the end.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKIP_DIRS = {".venv", "venv", "env", ".git", "__pycache__", "node_modules", "site-packages",
             "scripts", "tests", "static", "deploy"}  # code/test data the AI never reads
TEXT_EXT = {".py", ".txt", ".md", ".yaml", ".yml", ".json", ".jsonl", ".csv", ".bak", ".html", ".log", ".env.example"}


def patterns(phrases: list[str]) -> list[re.Pattern]:
    out = []
    for p in phrases:
        words = {"2.5": "two point five", "3.5": "three point five"}
        # A bare number must stand alone: matches "2.5 lakhs" / "2.5-3.5", not "gemini-2.5" or "12.5px".
        num = re.fullmatch(r"[\d.,]+", p)
        alts = [rf"(?<![\w.\-]){re.escape(p)}(?![\w.%])" if num else re.escape(p)]
        alts += [re.escape(words[p])] if p in words else []
        out.append(re.compile("|".join(alts), re.I))
    return out


def hits(text: str, pats: list[re.Pattern], width: int = 70) -> list[str]:
    found = []
    for pat in pats:
        for m in pat.finditer(text):
            a, b = max(0, m.start() - width), min(len(text), m.end() + width)
            found.append("…" + text[a:b].replace("\n", " ") + "…")
            if len(found) >= 5:
                return found
    return found


def check_prompt(pats, phone: str) -> bool:
    print("\n=== 1. What the AI is actually given on a call ===")
    from app import prompts
    from app.config import settings
    from app.knowledge import prompt_section
    from app.voice_config import call_language, language_instruction

    agent = settings.get("agent", {})
    pfile = agent.get("prompt_file", "sprint3_screening.txt")
    print(f"Prompt file in use: config/prompts/{pfile}   (agent.prompt_file in config/agent.yaml)")
    template = (ROOT / "config" / "prompts" / pfile).read_text(encoding="utf-8")
    parts = {
        f"prompt file + placeholders (config/prompts/{pfile}, job details from config/screening.yaml)":
            prompts.render(template, prompts.prompt_values()),
        "conversation rules + guardrails (app/prompts.py)": prompts.conversation_rules(),
        "language rule": language_instruction(call_language(settings)),
        f"previous-call memory for {phone or '(no --phone given)'} (logs/calls/*.json)":
            prompts.contact_memory(phone) if phone else "",
        "knowledge base (data/knowledge/*, data/knowledge_files/, config/kb/)": prompt_section(),
        "grounding rule": prompts.knowledge_grounding() if hasattr(prompts, "knowledge_grounding") else "",
    }
    any_hit = False
    for name, text in parts.items():
        h = hits(text or "", pats)
        print(f"  [{'FOUND' if h else ' ok  '}] {name}: {len(text or ''):,} chars")
        for line in h:
            print(f"           {line}")
        any_hit |= bool(h)
    try:
        from app import kb_store

        print("\n  Knowledge sources the AI reads right now:")
        for s in kb_store.compile_sources():
            body = kb_store._source_text(s)
            flag = "FOUND" if hits(body, pats) else " ok  "
            print(f"    [{flag}] {s.get('type')}: {s.get('title') or s.get('url') or '?'} ({len(body):,} chars)")
    except Exception as e:  # noqa: BLE001
        print(f"  (could not list knowledge sources: {e})")
    return any_hit


def check_files(pats) -> None:
    print("\n=== 2. Every project file containing it ===")
    count = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.name == ".env" or path.name == Path(__file__).name:
            continue  # never read secrets
        try:
            if path.suffix.lower() == ".pdf":
                from pypdf import PdfReader

                text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages[:200])
            elif path.suffix.lower() in TEXT_EXT or path.name.endswith(".bak"):
                if path.stat().st_size > 20_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            else:
                continue
        except Exception:  # noqa: BLE001
            continue
        h = hits(text, pats, width=50)
        if h:
            count += 1
            print(f"  {path.relative_to(ROOT)}")
            for line in h[:2]:
                print(f"      {line}")
    print(f"  {count} file(s)." if count else "  None.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phrases", nargs="+")
    ap.add_argument("--phone", default="")
    a = ap.parse_args()
    pats = patterns(a.phrases)
    in_prompt = check_prompt(pats, a.phone)
    check_files(pats)
    print("\n=== Verdict ===")
    if in_prompt:
        print("The phrase IS in what the AI is given: fix or delete the part marked FOUND above, then restart.")
    else:
        print("The phrase is NOT in anything the AI is given. Then either:\n"
              " - you tested from a number with old call logs and didn't pass --phone (rerun with it), or\n"
              " - the model is making it up: it answers pricing questions from general knowledge when the\n"
              "   knowledge base has no price. Add the real price as an FAQ, or tell it in the prompt to say\n"
              "   the team will confirm fees, and keep temperature at or below 0.7.")


if __name__ == "__main__":
    main()
