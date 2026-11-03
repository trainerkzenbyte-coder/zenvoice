"""Background jobs for the server (Sprint 5).

- Deletes call logs (they contain candidate speech) older than data.retention_days
  (config/agent.yaml, default 90).
- Resends Google Sheet rows that failed earlier, every 10 minutes.
"""

import asyncio
import time
from pathlib import Path

from loguru import logger

from app import sheet_sync
from app.config import ROOT, settings

CALL_LOG_DIR = Path(ROOT) / "logs" / "calls"


def retention_days() -> int:
    return int((settings.get("data") or {}).get("retention_days", 90))


def delete_old_call_logs() -> int:
    if not CALL_LOG_DIR.exists():
        return 0
    cutoff = time.time() - retention_days() * 86400
    removed = 0
    for f in CALL_LOG_DIR.glob("*.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    rec_dir = CALL_LOG_DIR.parent / "recordings"  # call recordings follow the same retention
    for f in rec_dir.glob("*.wav") if rec_dir.exists() else []:
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        logger.info(f"Housekeeping: deleted {removed} call log(s)/recording(s) older than {retention_days()} days")
    return removed


async def run_forever():
    """Started by the server; runs until the server stops."""
    last_cleanup = 0.0
    while True:
        try:
            if time.time() - last_cleanup > 6 * 3600:
                delete_old_call_logs()
                last_cleanup = time.time()
            if sheet_sync.enabled():
                await sheet_sync.flush_outbox()
        except Exception as e:  # noqa: BLE001  never let housekeeping kill the server
            logger.warning(f"Housekeeping error: {e}")
        await asyncio.sleep(600)
