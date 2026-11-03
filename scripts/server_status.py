"""Show the server's live status: active calls, load, memory, Sheet outbox.

Run from your laptop or the server (virtual environment active, ADMIN_TOKEN in .env):
    python scripts/server_status.py              # one look
    python scripts/server_status.py --wait-idle  # wait until no calls are active (before a restart)

On the server (inside Docker):
    docker compose exec agent python scripts/server_status.py --url http://127.0.0.1:8000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import vobiz  # noqa: E402  (also loads .env)


def fetch(base: str, token: str) -> dict:
    r = httpx.get(f"{base}/status", headers={"Authorization": f"Bearer {token}"}, timeout=10)
    if r.status_code == 401:
        sys.exit("Unauthorized: ADMIN_TOKEN in this .env differs from the server's.")
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="server base URL (default: PUBLIC_BASE_URL from .env)")
    ap.add_argument("--wait-idle", action="store_true", help="wait until active calls = 0")
    args = ap.parse_args()
    token = os.getenv("ADMIN_TOKEN", "").strip()
    if not token:
        sys.exit("Set ADMIN_TOKEN in .env (same value as on the server).")
    base = (args.url or vobiz.public_base_url()).rstrip("/")

    s = fetch(base, token)
    if args.wait_idle:
        while s.get("active_calls", 0) > 0:
            print(f"{time.strftime('%H:%M:%S')}  {s['active_calls']} call(s) in progress, waiting...")
            time.sleep(10)
            s = fetch(base, token)
        print("No active calls. Safe to restart or deploy.")
        return

    mem = (f"{s['memory_used_mb']} / {s['memory_total_mb']} MB"
           if "memory_used_mb" in s else "n/a")
    print(f"Server       {base}")
    print(f"Active calls {s['active_calls']}  (peak {s['peak_concurrent']} since start)")
    print(f"Calls        {s['calls_since_start']} since start, {s['calls_crashed']} crashed")
    print(f"Uptime       {s['uptime_minutes']} min")
    print(f"CPU          {cpu_text(s)}")
    print(f"Memory       {mem}")
    if "llm_errors" in s:
        last = f"  (last: {s['last_llm_error']})" if s.get("last_llm_error") else ""
        print(f"Gemini errors {s['llm_errors']} since start{last}")
    print(f"Sheet sync   {'on' if s['sheet_sync'] else 'OFF'}, {s['sheet_outbox_waiting']} row(s) waiting to resend")


def cpu_text(s: dict) -> str:
    """CPU % where available (psutil, any OS); otherwise the Linux load average."""
    parts = []
    if s.get("cpu_percent") is not None:
        parts.append(f"{s['cpu_percent']:.0f}% busy (under 70% = fine)")
    if s.get("load_1min") is not None:
        parts.append(f"load {s['load_1min']} on {s['cpu_count']} CPUs (under {s['cpu_count']} = fine)")
    return ", ".join(parts) or f"{s.get('cpu_count')} CPUs (install psutil for usage: pip install psutil)"


if __name__ == "__main__":
    main()
