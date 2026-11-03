"""Check the Google Sheet webhook end to end. No phone call needed.

Run from the project folder (virtual environment active):
    python scripts/test_sheet.py            # send a TEST row, then resend it (must update, not duplicate)
    python scripts/test_sheet.py --flush    # resend rows waiting in logs/sheet_outbox.jsonl
    python scripts/test_sheet.py --preview  # print the row that would be sent, send nothing

Rows from this script are marked [TEST]; delete them from the sheet afterwards.
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

SAMPLE = {
    "call_id": "",
    "candidate_name": "[TEST] Karthik Raman",
    "phone": "919876543210",
    "started_at": "",
    "duration_secs": 78,
    "outcome": {
        "call_outcome": "completed",
        "data": {
            "current_company": "Infosys", "total_experience_years": 5, "relevant_experience_years": 4,
            "current_ctc_lpa": 9.5, "expected_ctc_lpa": 14, "notice_period_days": 60,
            "current_location": "Chennai", "interested": "yes",
        },
        "summary": "TEST ROW. Senior engineer at Infosys, 5 years, 9.5 to 14 LPA, 60 days notice.",
        "recruiter_notes": "",
        "checks": [],
        "ended_by": "agent",
    },
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flush", action="store_true", help="resend rows waiting in the outbox")
    ap.add_argument("--preview", action="store_true", help="print the row, send nothing")
    args = ap.parse_args()

    record = dict(SAMPLE, call_id=f"test-{int(time.time())}", started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    if args.preview:
        print(json.dumps(sheet_sync.build_row(record), indent=2, ensure_ascii=False))
        return

    if not sheet_sync.enabled():
        sys.exit("SHEET_WEBHOOK_URL and SHEET_WEBHOOK_SECRET must both be set in .env")

    if args.flush:
        sent, left = await sheet_sync.flush_outbox()
        print(f"Outbox: {sent} sent, {left} still waiting")
        return

    row = sheet_sync.build_row(record)
    print("1/2 Sending a TEST row...")
    try:
        res = await sheet_sync.send_row(row)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"   FAIL: {e}\n   See integrations/google_sheet/README.md, 'If something fails'.")
    print(f"   OK: {res.get('action')} row {res.get('row')} in tab '{res.get('sheet')}'")
    if res.get("added_columns"):
        print(f"   Columns added to the sheet: {', '.join(res['added_columns'])}")

    print("2/2 Resending the same call with a changed summary (must UPDATE, not add a row)...")
    row["Summary"] = row.get("Summary", "") + " (updated by resend)"
    res2 = await sheet_sync.send_row(row)
    same = res2.get("action") == "updated" and res2.get("row") == res.get("row")
    print(f"   {'OK' if same else 'FAIL'}: {res2.get('action')} row {res2.get('row')}")
    print("\nOpen the sheet: you should see ONE [TEST] row ending in '(updated by resend)'. Delete it after.")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    asyncio.run(main())
