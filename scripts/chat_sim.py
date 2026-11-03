"""Text-chat with Priya as if you were the candidate, then see the extracted data.

Run from the project folder (virtual environment active):
    python scripts/chat_sim.py                 # uses test_candidate name from config/screening.yaml
    python scripts/chat_sim.py --name Divya    # pretend the candidate is Divya
    python scripts/chat_sim.py --sheet         # also send the result to the Google Sheet (marked [TEST])

Same call script, job details, tools (end_call, schedule_callback) and Gemini model
as real calls, but typed: fast and almost free. Priya can now hang up by herself.
Type /end to hang up yourself. The result is saved to logs/calls/ like a real call.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import sheet_sync  # noqa: E402
from app.config import settings  # noqa: E402
from app.outcome import describe, extract_outcome, reconcile  # noqa: E402
from app.prompts import test_candidate_name  # noqa: E402
from app.text_sim import TextCall  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="candidate name Priya should ask for")
    ap.add_argument("--sheet", action="store_true", help="send the result to the Google Sheet")
    args = ap.parse_args()
    name = args.name if args.name is not None else test_candidate_name()
    agent = settings.get("agent", {}).get("name", "Priya")

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    started = time.time()
    call = TextCall(candidate_name=name)
    print(f"Chat with {agent}. You are the candidate{f' ({name})' if name else ''}. Type /end to hang up.\n")
    print(f"{agent.upper():<9}: {await call.start()}")

    while not call.ended:
        try:
            you = input("YOU      : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if you.lower() in ("/end", "/quit", "/exit"):
            call.state.ended_by = "candidate"
            break
        if not you:
            continue
        before = len(call.state.actions)
        reply = await call.reply(you)
        for a in call.state.actions[before:]:
            details = ", ".join(f"{k}={v}" for k, v in a.items() if k not in ("t", "action") and v)
            print(f"  [action] {a['action']}({details})")
        if reply:
            print(f"{agent.upper():<9}: {reply}")

    if call.ended:
        print(f"\n{agent} hung up ({call.state.end_reason}).")
    print("\nExtracting screening data...\n")
    outcome = reconcile(await extract_outcome(call.transcript, agent_name=agent), call.state)
    print(describe(outcome))

    out_dir = ROOT / "logs" / "calls"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_chat-sim.json"
    record = {"call_id": "chat-sim", "channel": "text simulator", "candidate_name": name,
              "outcome": outcome, "transcript": call.transcript}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {path.relative_to(ROOT)}")

    if args.sheet:
        sheet_record = {
            "call_id": f"sim-{time.strftime('%Y%m%d-%H%M%S')}",
            "candidate_name": f"[TEST] {name or 'Text simulator'}",
            "phone": "",
            "started_at": started_at,
            "duration_secs": round(time.time() - started),
            "outcome": outcome,
        }
        await sheet_sync.push_call(sheet_record, "chat-sim")


if __name__ == "__main__":
    asyncio.run(main())
