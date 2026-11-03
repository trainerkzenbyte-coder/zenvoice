"""Automated call-flow tests: an AI plays 8 candidates, Priya runs the real script + tools.

Run from the project folder (virtual environment active):
    python scripts/test_call_flow.py                  # all personas (~2 min, about Rs 10)
    python scripts/test_call_flow.py --only busy      # personas whose name contains "busy"
    python scripts/test_call_flow.py --show           # print every conversation

Each persona in tests/call_flow/personas.yaml checks that Priya:
  - ends the call herself with the right reason (end_call)
  - saves a valid callback time when asked (schedule_callback), within calling hours
  - produces the right outcome and numbers after extraction
  - keeps replies short and never invents a salary budget
"""

import argparse
import asyncio
import datetime as dt
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import gemini_rest  # noqa: E402
from app.outcome import describe, extract_outcome, reconcile  # noqa: E402
from app.text_sim import TextCall  # noqa: E402

PERSONAS = ROOT / "tests" / "call_flow" / "personas.yaml"
NOW = dt.datetime(2026, 10, 5, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # Monday 10:00 IST
MAX_TURNS = 14
MAX_WORDS = 45  # a phone reply longer than this feels like a speech

CANDIDATE_RULES = (
    "You are on a phone call with a recruiter named Priya. Stay in character. Reply as the "
    "candidate only, in one or two short natural sentences of Indian English, like real speech. "
    "Never describe actions, never write the recruiter's lines. If she says goodbye, say a short "
    "goodbye.\n\nYour character:\n"
)


async def candidate_says(persona: str, transcript: list[dict]) -> str:
    # From the candidate's point of view, Priya is the "user" and the candidate is the "model".
    contents = []
    for t in transcript:
        role = "user" if t["role"] == "bot" else "model"
        contents.append({"role": role, "parts": [{"text": t["text"]}]})
    return await gemini_rest.generate(
        contents, system=CANDIDATE_RULES + persona, temperature=0.6, max_tokens=150,
        thinking_cfg={"thinking_level": "minimal"},
    )


async def run_persona(key: str, p: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        call = TextCall(candidate_name=p.get("name", ""), now=NOW)
        try:
            await call.start()
            for _ in range(MAX_TURNS):
                if call.ended:
                    break
                said = await candidate_says(p["persona"], call.transcript)
                await call.reply(said)
            outcome = reconcile(await extract_outcome(call.transcript, now=NOW), call.state)
        except Exception as e:  # noqa: BLE001
            return {"key": key, "error": str(e), "call": call}
    return {"key": key, "call": call, "outcome": outcome, "problems": check(p, call, outcome)}


def check(p: dict, call: TextCall, outcome: dict) -> list[str]:
    problems = []
    st = call.state
    exp_end = p.get("end_reason")
    if exp_end is None:
        if call.ended:
            problems.append(f"Priya hung up ({st.end_reason}) but should have kept talking")
    elif not call.ended:
        problems.append(f"Priya never hung up (expected end_call '{exp_end}')")
    elif st.end_reason != exp_end:
        problems.append(f"end_call reason '{st.end_reason}', expected '{exp_end}'")

    cb = p.get("callback")
    if cb:
        if not st.callback_time:
            problems.append("no callback saved")
        else:
            when = dt.datetime.strptime(st.callback_time, "%Y-%m-%dT%H:%M")
            want_day = (NOW + dt.timedelta(days=cb["day_offset"])).date()
            if when.date() != want_day or when.hour != cb["hour"]:
                problems.append(f"callback {st.callback_time}, expected {want_day} {cb['hour']:02d}:00")
    elif cb is False and st.callback_time:
        problems.append(f"unexpected callback saved ({st.callback_time})")

    if p.get("outcome") and outcome["call_outcome"] != p["outcome"]:
        problems.append(f"outcome '{outcome['call_outcome']}', expected '{p['outcome']}'")
    for field, want in (p.get("data") or {}).items():
        got = outcome["data"].get(field)
        if got is None or abs(float(got) - float(want)) > 0.05:
            problems.append(f"{field} = {got!r}, expected {want}")

    bot_lines = [t["text"] for t in call.transcript if t["role"] == "bot"]
    long = [b for b in bot_lines if len(b.split()) > MAX_WORDS]
    if long:
        problems.append(f"{len(long)} reply(ies) over {MAX_WORDS} words, e.g. '{long[0][:80]}...'")
    for word in p.get("forbid_bot_numbers_near", []):
        for b in bot_lines:
            for sentence in re.split(r"(?<=[.?!])\s+", b):
                if word in sentence.lower() and re.search(r"\d|lakh", sentence.lower()):
                    problems.append(f"Priya may have stated a {word} figure: '{sentence[:90]}'")
    return problems


def print_conversation(call: TextCall):
    for t in call.transcript:
        who = "PRIYA" if t["role"] == "bot" else "CAND."
        print(f"      {who}: {t['text']}")
    for a in call.state.actions:
        details = ", ".join(f"{k}={v}" for k, v in a.items() if k not in ("t", "action") and v)
        print(f"      [action] {a['action']}({details})")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run personas whose key contains this text")
    ap.add_argument("--show", action="store_true", help="print every conversation and outcome")
    args = ap.parse_args()

    personas = yaml.safe_load(PERSONAS.read_text(encoding="utf-8"))
    keys = [k for k in personas if not args.only or args.only in k]
    print(f"Running {len(keys)} simulated call(s)... (test clock: Monday 5 Oct 2026, 10:00 IST)\n")
    sem = asyncio.Semaphore(3)
    runs = await asyncio.gather(*(run_persona(k, personas[k], sem) for k in keys))

    passed = 0
    for r in runs:
        if r.get("error"):
            print(f"ERROR {r['key']}: {r['error']}\n")
            continue
        ok = not r["problems"]
        passed += ok
        turns = sum(1 for t in r["call"].transcript if t["role"] == "user")
        print(f"{'PASS' if ok else 'FAIL'}  {r['key']:<24} {turns} candidate turns, "
              f"ended: {r['call'].state.end_reason or 'no'}")
        for prob in r["problems"]:
            print(f"      - {prob}")
        if args.show or not ok:
            print_conversation(r["call"])
            for line in describe(r["outcome"]).splitlines():
                print(f"      | {line}")
        print()

    print(f"Passed {passed}/{len(keys)}. Target before live calls: all pass on 2 runs in a row.")
    print("(The AI candidate varies a little each run, so an occasional single failure is worth "
          "re-running before changing anything.)")
    sys.exit(0 if passed == len(keys) else 1)


if __name__ == "__main__":
    asyncio.run(main())
