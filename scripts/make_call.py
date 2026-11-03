"""Place a test call: Priya calls the number you give.

Run from the project folder (virtual environment active), while the server
(python -m app.server) and ngrok are running in two other terminals:
    python scripts/make_call.py 9876543210
    python scripts/make_call.py 9876543210 --name Karthik

Before dialling it checks that ngrok forwards to the server, so a broken setup
fails here instead of as a silent call.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import vobiz  # noqa: E402
from app.config import settings  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("number", nargs="+", help="mobile number, e.g. 9876543210")
    ap.add_argument("--name", default="", help="candidate name Priya should ask for")
    args = ap.parse_args()
    number = " ".join(args.number)
    try:
        to = vobiz.normalize_number(number)
    except ValueError as e:
        sys.exit(str(e))

    base = vobiz.public_base_url()
    print(f"1/3 Checking the server through ngrok: {base}/health")
    try:
        r = httpx.get(f"{base}/health", timeout=10, headers={"ngrok-skip-browser-warning": "1"})
        ok = r.status_code == 200 and r.text.strip() == "ok"
    except httpx.HTTPError as e:
        ok, r = False, e
    if not ok:
        sys.exit(
            f"   FAIL: {r}\n"
            "   Is 'python -m app.server' running, and ngrok started with your static domain?\n"
            "   ngrok http --url=<your-domain>.ngrok-free.app 8000"
        )
    print("   OK")

    limits = settings.get("call_limits", {}) or {}
    phone = settings.get("phone", {}) or {}
    time_limit = int(limits.get("max_call_seconds", 240)) + 30  # Vobiz backstop
    ring = int(phone.get("ring_timeout_secs", 30))

    who = f"{args.name} at {to}" if args.name else to
    print(f"2/3 Asking Vobiz to call {who} (rings up to {ring} s)")
    try:
        result = await vobiz.make_call(
            to, time_limit_secs=time_limit, ring_timeout_secs=ring, candidate_name=args.name
        )
    except RuntimeError as e:
        hint = ""
        text = str(e)
        if "401" in text:
            hint = "\n   Check VOBIZ_AUTH_ID / VOBIZ_AUTH_TOKEN in .env."
        elif "402" in text:
            hint = "\n   Top up your Vobiz balance."
        elif "400" in text:
            hint = "\n   Check VOBIZ_FROM_NUMBER (your Vobiz number with country code) and the number you dialled."
        sys.exit(f"   FAIL: {text}{hint}")
    call_id = result.get("request_uuid") or result.get("call_uuid") or "?"
    print(f"   OK: {result.get('message', 'call requested')} (call id ...{str(call_id)[-6:]})")
    print("3/3 Your phone should ring now. Watch the server window for the conversation log.")


if __name__ == "__main__":
    asyncio.run(main())
