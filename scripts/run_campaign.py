"""Sprint 6: let Priya call a whole list of candidates.

Run from the project folder (virtual environment active), on the SAME computer as the
server (python -m app.server and ngrok must be running):

    python scripts/run_campaign.py campaigns/java_oct.csv --status     # progress, no calls
    python scripts/run_campaign.py campaigns/java_oct.csv --dry-run    # who would be called now
    python scripts/run_campaign.py campaigns/java_oct.csv              # start calling (Ctrl+C = pause)
    python scripts/run_campaign.py campaigns/java_oct.csv --once       # stop when nobody is due now

The CSV needs a header row with name and phone, e.g.:
    name,phone
    Karthik,9876543210

Stopping and starting again is safe: progress is saved in campaigns/<list>.state.json.
Settings: config/campaign.yaml. Results also go to the Google Sheet, one row per call.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import campaign, vobiz  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> {message}")


def show_status(csv_path: Path) -> None:
    state = campaign.load_state(csv_path)
    campaign.save_state(csv_path, state)
    print(f"Campaign {state['campaign']}: {campaign.summary_line(state)}\n")
    print(f"{'name':<22} {'phone':<14} {'status':<12} {'tries':>5}  {'next try':<16}  last result / note")
    for c in sorted(state["candidates"].values(), key=lambda c: (c["status"], c.get("next_at") or "")):
        print(f"{(c['name'] or '-')[:22]:<22} {c['phone'][:14]:<14} {c['status']:<12} {c['attempts']:>5}  "
              f"{(c.get('next_at') or '-'):<16}  {c.get('note') or c.get('last_result') or ''}")


def dry_run(csv_path: Path) -> None:
    state = campaign.load_state(csv_path)
    now = campaign.now_ist()
    due = campaign.due_candidates(state, now, campaign.load_dnc())
    if not campaign.in_calling_hours(now):
        print(f"Outside calling hours now; calling would start at {campaign.iso(campaign.next_calling_time(now))}.")
    print(f"{len(due)} candidate(s) due now, in this order:")
    for key, c in due:
        print(f"  {c['name'] or '-':<22} {c['phone']:<14} {c['status']}")
    print("\nNothing was dialled (dry run).")


def check_server() -> None:
    base = vobiz.public_base_url()
    try:
        r = httpx.get(f"{base}/health", timeout=10)
        ok = r.status_code == 200 and r.text.strip() == "ok"
    except httpx.HTTPError:
        ok = False
    if not ok:
        sys.exit(f"The server is not reachable at {base}/health. Start python -m app.server and ngrok first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="candidate list, e.g. campaigns/java_oct.csv")
    ap.add_argument("--status", action="store_true", help="show progress only")
    ap.add_argument("--dry-run", action="store_true", help="show who would be called now; no calls")
    ap.add_argument("--once", action="store_true", help="stop when nobody is due now (instead of waiting)")
    ap.add_argument("--max-concurrent", type=int, help="override config/campaign.yaml")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found. Copy campaigns/example.csv and fill in your candidates.")
    if args.status:
        return show_status(csv_path)
    if args.dry_run:
        return dry_run(csv_path)

    check_server()
    runner = campaign.CampaignRunner(csv_path, max_concurrent=args.max_concurrent)
    try:
        asyncio.run(runner.run(stop_when_idle=args.once))
    except KeyboardInterrupt:
        print("\nPaused. Calls in progress were saved; run the same command again to continue.")


if __name__ == "__main__":
    main()
