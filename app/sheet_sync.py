"""Sprint 4: push each call's outcome to your Google Sheet (via the Apps Script webhook).

    ok = await push_call(record)      # after every call; never raises
    sent, left = await flush_outbox() # resend rows that failed earlier

Columns come from `sheet_columns` in config/screening.yaml (column name -> where the
value comes from). If Google can't be reached after 3 tries, the row is saved to
logs/sheet_outbox.jsonl and resent automatically when the server next starts
(or with: python scripts/test_sheet.py --flush).
"""

import asyncio
import json
import os
from pathlib import Path

import httpx
from loguru import logger

from app.config import settings, ROOT
from app.prompts import screening_cfg

OUTBOX = Path(ROOT) / "logs" / "sheet_outbox.jsonl"
KEY_COLUMN = "Call ID"
RETRY_DELAYS = (0, 3, 10)  # seconds before each attempt

# Extra columns always sent (added at the right of your sheet if missing).
DEFAULT_EXTRA_COLUMNS = {
    "Call ID": "call_id",
    "Called At": "called_at",
    "Duration (s)": "duration_secs",
    "Ended By": "ended_by",
    "Recruiter Checks": "checks",
}


def _config() -> tuple[str, str]:
    return (os.getenv("SHEET_WEBHOOK_URL", "").strip(), os.getenv("SHEET_WEBHOOK_SECRET", "").strip())


def enabled() -> bool:
    url, secret = _config()
    return bool(url and secret)


# ---------- building the row ----------
def format_phone(digits: str) -> str:
    d = "".join(ch for ch in str(digits or "") if ch.isdigit())
    if len(d) == 12 and d.startswith("91"):
        return f"+91 {d[2:7]} {d[7:]}"
    return f"+{d}" if d else ""


def outcome_label(value: str, cfg: dict) -> str:
    labels = cfg.get("sheet_outcome_labels") or {}
    if value in labels:
        return labels[value]
    return str(value or "").replace("_", " ").capitalize()


def _value(source: str, record: dict, cfg: dict):
    outcome = record.get("outcome") or {}
    data = outcome.get("data") or {}
    if source == "candidate_name":
        return record.get("candidate_name") or ""
    if source == "phone":
        return format_phone(record.get("phone", ""))
    if source == "call_id":
        return record.get("call_id", "")
    if source == "called_at":
        return record.get("started_at") or record.get("ended_at", "")
    if source == "duration_secs":
        return record.get("duration_secs", "")
    if source == "ended_by":
        return outcome.get("ended_by") or ""
    if source == "call_outcome":
        return outcome_label(outcome.get("call_outcome", ""), cfg)
    if source in ("summary", "recruiter_notes"):
        return outcome.get(source, "") or ""
    if source == "checks":
        return "; ".join(outcome.get("checks") or [])
    if source == "job.title" and record.get("role"):
        return record["role"]  # role typed in the dashboard dialer for this call
    if source.startswith("job."):
        return (cfg.get("job") or {}).get(source[4:], "")
    value = data.get(source)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if source == "callback_time":
        return str(value).replace("T", " ")
    return value


def build_row(record: dict, cfg: dict | None = None) -> dict:
    """Column name -> value, in your sheet's column order, then the extra columns."""
    cfg = cfg or screening_cfg()
    mapping = dict(cfg.get("sheet_columns") or {})
    for col, src in DEFAULT_EXTRA_COLUMNS.items():
        if col not in mapping and src not in mapping.values():
            mapping[col] = src
    mapping.setdefault(KEY_COLUMN, "call_id")  # needed so resends update the same row
    return {col: _value(str(src), record, cfg) for col, src in mapping.items()}


# ---------- sending ----------
async def _post(row: dict) -> dict:
    url, secret = _config()
    payload = {"secret": secret, "source": "zenbyte-voice", "version": 1,
               "key_column": KEY_COLUMN, "row": row}
    # Apps Script answers a POST with a redirect to the result: follow it.
    timeout = float((settings.get("sheet") or {}).get("timeout_secs", 30))  # Agents > Webhook settings
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.post(url, json=payload)
    try:
        body = r.json()
    except ValueError:
        hint = ""
        if "accounts.google.com" in str(r.url) or "<html" in r.text[:200].lower():
            hint = " (web app access must be 'Anyone'; redeploy if you changed it)"
        raise RuntimeError(f"Sheet webhook returned HTTP {r.status_code}, not JSON{hint}")
    if not body.get("ok"):
        raise RuntimeError(f"Sheet webhook refused the row: {body.get('error', body)}")
    return body


async def send_row(row: dict) -> dict:
    """Send with retries. Raises the last error if every attempt fails."""
    last: Exception | None = None
    for delay in RETRY_DELAYS:
        if delay:
            await asyncio.sleep(delay)
        try:
            return await _post(row)
        except (httpx.HTTPError, RuntimeError) as e:
            last = e
            if "unauthorized" in str(e) or "missing" in str(e):
                break  # configuration problem: retrying won't help
    raise last or RuntimeError("unknown error")


def _to_outbox(row: dict, error: str):
    OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with OUTBOX.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"row": row, "error": error}, ensure_ascii=False) + "\n")


async def push_call(record: dict, label: str = "") -> bool:
    """Push one call to the sheet. Never raises; failures go to the outbox."""
    tag = f"[{label}] " if label else ""
    if not enabled():
        logger.info(f"{tag}Sheet sync off (SHEET_WEBHOOK_URL / SHEET_WEBHOOK_SECRET not set)")
        return False
    if not record.get("outcome"):
        logger.warning(f"{tag}No outcome extracted; sending a minimal row so the call is not lost")
    try:
        row = build_row(record)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{tag}Could not build sheet row: {e}")
        return False
    try:
        res = await send_row(row)
        logger.info(f"{tag}Sheet: row {res.get('action')} (row {res.get('row')} in '{res.get('sheet')}')")
        if res.get("added_columns"):
            logger.info(f"{tag}Sheet: added columns {res['added_columns']}")
        return True
    except Exception as e:  # noqa: BLE001
        _to_outbox(row, str(e))
        logger.warning(f"{tag}Sheet push failed, saved to outbox for retry: {e}")
        return False


async def flush_outbox() -> tuple[int, int]:
    """Resend rows that failed earlier. Returns (sent, still_waiting)."""
    if not OUTBOX.exists() or not enabled():
        return 0, 0
    lines = [ln for ln in OUTBOX.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return 0, 0
    remaining, sent = [], 0
    for ln in lines:
        try:
            item = json.loads(ln)
            await send_row(item["row"])
            sent += 1
        except Exception as e:  # noqa: BLE001
            try:
                item["error"] = str(e)
                remaining.append(json.dumps(item, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                remaining.append(ln)
    OUTBOX.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
    if sent or remaining:
        logger.info(f"Sheet outbox: {sent} sent, {len(remaining)} still waiting")
    return sent, len(remaining)
