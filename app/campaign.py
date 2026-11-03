"""Sprint 6: campaign engine. Priya works through a list of candidates by herself.

    campaigns/java_oct.csv          your list: name,phone (one candidate per row)
    campaigns/java_oct.state.json   progress, written by the runner (who was called, result, next try)
    campaigns/do_not_call.txt       numbers that must never be called (added automatically)

Rules (config/campaign.yaml):
- only inside calling hours (config/agent.yaml -> call_limits.calling_hours)
- at most max_concurrent calls at once
- no answer / busy / voicemail / dropped -> try again after retry_gap_minutes, up to max_attempts
- candidate asked for a callback -> call again at that time
- completed / not interested / wrong person -> done
- do not call -> never again (also added to do_not_call.txt)
- Tamil / language request -> "needs_human" for a colleague

The runner must run on the same machine as the server: it reads logs/calls/*.json
(written after each answered call) and logs/call_events.jsonl (Vobiz hang-up reports).
"""

import asyncio
import csv
import datetime as dt
import json
import re
import time
from pathlib import Path

import yaml
from loguru import logger

from app.call_actions import IST, _calling_hours
from app.config import ROOT, settings

CAMPAIGN_DIR = Path(ROOT) / "campaigns"
DNC_FILE = CAMPAIGN_DIR / "do_not_call.txt"
CALL_LOG_DIR = Path(ROOT) / "logs" / "calls"
EVENTS_FILE = Path(ROOT) / "logs" / "call_events.jsonl"

DONE = {"done", "do_not_call", "needs_human", "failed", "invalid"}
WAITING = {"pending", "retry", "callback"}


# ---------- settings ----------
def campaign_cfg() -> dict:
    path = Path(ROOT) / "config" / "campaign.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = cfg or {}
    limits = settings.get("call_limits", {}) or {}
    return {
        "max_concurrent": int(cfg.get("max_concurrent", 1)),
        "gap_between_calls_secs": float(cfg.get("gap_between_calls_secs", 10)),
        "retry_gap_minutes": int(cfg.get("retry_gap_minutes", 180)),
        "error_retry_minutes": int(cfg.get("error_retry_minutes", 30)),
        "max_attempts": int(cfg.get("max_attempts", limits.get("max_attempts", 3))),
        "max_callbacks": int(cfg.get("max_callbacks", 2)),
    }


# ---------- small helpers ----------
def phone_key(number: str) -> str:
    """Last 10 digits: the same mobile written as +91 98765 43210 or 09876543210."""
    digits = re.sub(r"\D", "", str(number or ""))
    return digits[-10:] if len(digits) >= 10 else ""


def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)


def iso(t: dt.datetime) -> str:
    return t.astimezone(IST).strftime("%Y-%m-%dT%H:%M")


def parse_iso(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s)[:16])
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=IST)


def in_calling_hours(t: dt.datetime) -> bool:
    start, end = _calling_hours()
    return start <= t.astimezone(IST).time() < end


def next_calling_time(t: dt.datetime) -> dt.datetime:
    """t itself if inside calling hours, else the start of the next calling window."""
    t = t.astimezone(IST)
    start, end = _calling_hours()
    if t.time() < start:
        return t.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
    if t.time() >= end:
        nxt = t + dt.timedelta(days=1)
        return nxt.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
    return t


def load_dnc() -> set[str]:
    if not DNC_FILE.exists():
        return set()
    return {phone_key(line) for line in DNC_FILE.read_text(encoding="utf-8").splitlines() if phone_key(line)}


def add_dnc(number: str) -> None:
    key = phone_key(number)
    if key and key not in load_dnc():
        DNC_FILE.parent.mkdir(parents=True, exist_ok=True)
        with DNC_FILE.open("a", encoding="utf-8") as f:
            f.write(key + "\n")


# ---------- the list and its progress ----------
def state_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".state.json")


def read_candidates(csv_path: Path) -> list[dict]:
    """CSV with a header row containing 'name' and 'phone' (any letter case)."""
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        low = {str(k).strip().lower(): (v or "").strip() for k, v in row.items() if k}
        out.append({"name": low.get("name", ""), "phone": low.get("phone", low.get("mobile", "")),
                    "role": low.get("role", low.get("job", ""))})
    return out


def load_state(csv_path: Path) -> dict:
    sp = state_path(csv_path)
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {"candidates": {}}
    state.setdefault("campaign", re.sub(r"[^A-Za-z0-9]+", "-", csv_path.stem).strip("-")[:20] or "camp")
    cands = state.setdefault("candidates", {})
    for row in read_candidates(csv_path):
        key = phone_key(row["phone"])
        if not key:
            bad = f"invalid:{row['phone'] or row['name']}"
            cands.setdefault(bad, {"name": row["name"], "phone": row["phone"], "status": "invalid",
                                   "attempts": 0, "callbacks": 0, "history": []})
            continue
        if key not in cands:
            cands[key] = {"name": row["name"], "phone": "91" + key, "role": row.get("role", ""),
                          "status": "pending", "attempts": 0,
                          "callbacks": 0, "next_at": None, "last_ref": None, "last_result": None,
                          "history": []}
    return state


def save_state(csv_path: Path, state: dict) -> None:
    sp = state_path(csv_path)
    tmp = sp.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(sp)  # never leaves a half-written file


def due_candidates(state: dict, now: dt.datetime, dnc: set[str]) -> list[tuple[str, dict]]:
    """Who can be called right now: callbacks first, then new candidates, then retries."""
    order = {"callback": 0, "pending": 1, "retry": 2}
    due = []
    for key, c in state["candidates"].items():
        if c["status"] not in WAITING:
            continue
        if key in dnc:
            c["status"] = "do_not_call"
            continue
        at = parse_iso(c.get("next_at"))
        if at is None or at <= now:
            due.append((key, c))
    return sorted(due, key=lambda kc: (order[kc[1]["status"]], kc[1].get("next_at") or ""))


# ---------- what to do after a call ----------
def decide(c: dict, result: dict, cfg: dict, now: dt.datetime) -> None:
    """Update one candidate from a call result.

    result = {"kind": "answered", "outcome": {...call log outcome...}}
           | {"kind": "not_answered", "cause": "..."}
           | {"kind": "unknown"} (no report at all)
           | {"kind": "dial_failed", "error": "..."}
    """
    kind = result.get("kind")
    out = result.get("outcome") or {}
    call_outcome = out.get("call_outcome")
    ended_by = out.get("ended_by")
    note = ""

    def retry(minutes: int, why: str, count_attempt: bool = True):
        nonlocal note
        if count_attempt and c["attempts"] >= cfg["max_attempts"]:
            c["status"], c["next_at"] = "failed", None
            note = f"{why}; gave up after {c['attempts']} attempts"
            return
        if not count_attempt:
            c["attempts"] = max(0, c["attempts"] - 1)  # technical problem: doesn't use up a try
        c["status"] = "retry"
        c["next_at"] = iso(next_calling_time(now + dt.timedelta(minutes=minutes)))
        note = f"{why}; next try {c['next_at']}"

    if kind == "dial_failed":
        retry(cfg["error_retry_minutes"], f"could not dial ({result.get('error', '')[:80]})", count_attempt=False)
    elif kind == "not_answered":
        retry(cfg["retry_gap_minutes"], f"not answered ({result.get('cause') or 'no answer'})")
    elif kind == "unknown":
        retry(cfg["error_retry_minutes"], "no result received from the server", count_attempt=False)
    elif ended_by == "error":
        retry(cfg["error_retry_minutes"], "call ended by a technical problem", count_attempt=False)
    elif call_outcome == "do_not_call":
        c["status"], c["next_at"], note = "do_not_call", None, "asked not to be called again"
        add_dnc(c["phone"])
    elif call_outcome == "callback_requested":
        when = parse_iso((out.get("data") or {}).get("callback_time"))
        if when and when > now and c.get("callbacks", 0) < cfg["max_callbacks"]:
            c["callbacks"] = c.get("callbacks", 0) + 1
            c["attempts"] = max(0, c["attempts"] - 1)  # a promised callback is not a failed try
            c["status"], c["next_at"] = "callback", iso(next_calling_time(when))
            note = f"callback at {c['next_at']}"
        else:
            retry(cfg["retry_gap_minutes"], "asked for a callback without a usable time")
    elif call_outcome == "language_barrier":
        c["status"], c["next_at"], note = "needs_human", None, "wants another language: give to a colleague"
    elif call_outcome in ("completed", "not_interested", "wrong_person"):
        c["status"], c["next_at"], note = "done", None, call_outcome.replace("_", " ")
    else:  # voicemail, dropped, anything unexpected
        retry(cfg["retry_gap_minutes"], f"{call_outcome or 'no outcome'}")

    c["last_result"] = call_outcome or kind
    c["history"].append({"t": iso(now), "ref": c.get("last_ref"), "result": c["last_result"], "note": note})
    c["note"] = note


# ---------- finding the result of one call ----------
def find_call_log(ref: str, since: float) -> dict | None:
    if not CALL_LOG_DIR.exists():
        return None
    for path in sorted(CALL_LOG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.stat().st_mtime < since - 5:
            break  # older than this call: stop looking
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rec.get("ref") == ref:
            return rec
    return None


def find_hangup(ref: str) -> dict | None:
    if not EVENTS_FILE.exists():
        return None
    found = None
    with EVENTS_FILE.open(encoding="utf-8") as f:
        for line in f:
            if ref in line:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("ref") == ref:
                    found = ev
    return found


def _answered(ev: dict) -> bool:
    for k in ("bill_duration", "duration"):
        try:
            if float(ev.get(k) or 0) > 0:
                return True
        except ValueError:
            pass
    return False


async def wait_for_result(ref: str, since: float, max_wait_secs: float, poll_secs: float = 5) -> dict:
    """Waits for the call log (answered) or a hang-up report (not answered)."""
    deadline = time.time() + max_wait_secs
    hangup_seen_at = None
    while time.time() < deadline:
        rec = find_call_log(ref, since)
        if rec:
            return {"kind": "answered", "outcome": rec.get("outcome") or {}, "call_id": rec.get("call_id")}
        ev = find_hangup(ref)
        if ev and not _answered(ev):
            return {"kind": "not_answered", "cause": ev.get("cause", "")}
        if ev and hangup_seen_at is None:
            hangup_seen_at = time.time()
        if hangup_seen_at and time.time() - hangup_seen_at > 240:
            break  # answered but no call log after 4 min (server restarted mid-call?)
        await asyncio.sleep(poll_secs)
    return {"kind": "unknown"}


# ---------- the runner ----------
class CampaignRunner:
    def __init__(self, csv_path: Path, dial=None, max_concurrent: int | None = None, poll_secs: float = 5):
        from app import vobiz

        self.csv_path = Path(csv_path)
        self.cfg = campaign_cfg()
        if max_concurrent:
            self.cfg["max_concurrent"] = max_concurrent
        self.state = load_state(self.csv_path)
        self.dial = dial or vobiz.make_call
        self.poll_secs = poll_secs
        self.active: dict[str, asyncio.Task] = {}
        self.stop_reason = ""
        limits = settings.get("call_limits", {}) or {}
        phone = settings.get("phone", {}) or {}
        self.max_call = int(limits.get("max_call_seconds", 240))
        self.ring = int(phone.get("ring_timeout_secs", 30))
        self._recover()

    def _recover(self):
        """Calls that were in progress when the runner stopped: use their result if it arrived."""
        for c in self.state["candidates"].values():
            if c["status"] != "in_call":
                continue
            ref = c.get("last_ref") or ""
            rec = find_call_log(ref, 0) if ref else None
            ev = find_hangup(ref) if ref and not rec else None
            if rec:
                result = {"kind": "answered", "outcome": rec.get("outcome") or {}}
            elif ev and not _answered(ev):
                result = {"kind": "not_answered", "cause": ev.get("cause", "")}
            else:
                result = {"kind": "unknown"}
            decide(c, result, self.cfg, now_ist())
        save_state(self.csv_path, self.state)

    def save(self):
        save_state(self.csv_path, self.state)

    async def _call_one(self, key: str, c: dict):
        c["attempts"] += 1
        c["dials"] = c.get("dials", 0) + 1
        ref = f"{self.state['campaign']}-{key}-{c['dials']}"
        c["last_ref"], c["status"] = ref, "in_call"
        self.save()
        started = time.time()
        logger.info(f"Calling {c['name'] or '(no name)'} {c['phone']} (attempt {c['attempts']}, ref {ref})")
        try:
            await self.dial(c["phone"], time_limit_secs=self.max_call + 30, ring_timeout_secs=self.ring,
                            candidate_name=c["name"], ref=ref, **({"role": c["role"]} if c.get("role") else {}))
            result = await wait_for_result(ref, started, self.ring + self.max_call + 420, self.poll_secs)
        except RuntimeError as e:
            text = str(e)
            if "402" in text or "balance" in text.lower() or "401" in text:
                self.stop_reason = f"Vobiz refused the call: {text[:160]}"
            result = {"kind": "dial_failed", "error": text}
        except Exception as e:  # noqa: BLE001
            result = {"kind": "dial_failed", "error": repr(e)}
        decide(c, result, self.cfg, now_ist())
        logger.info(f"  {c['name'] or c['phone']}: {c['note']}")
        self.save()

    def _finished(self) -> bool:
        return not any(c["status"] in WAITING or c["status"] == "in_call"
                       for c in self.state["candidates"].values())

    async def run(self, stop_when_idle: bool = False):
        cfg = self.cfg
        logger.info(f"Campaign {self.state['campaign']}: {summary_line(self.state)} | "
                    f"{cfg['max_concurrent']} call(s) at once, up to {cfg['max_attempts']} attempts")
        said_outside = False
        try:
            while not self.stop_reason:
                self.active = {k: t for k, t in self.active.items() if not t.done()}
                now = now_ist()
                if self._finished() and not self.active:
                    logger.info("Campaign finished: nobody left to call.")
                    break
                if not in_calling_hours(now):
                    if not said_outside:
                        logger.info(f"Outside calling hours. Waiting until {iso(next_calling_time(now))}.")
                        said_outside = True
                    if stop_when_idle and not self.active:
                        break
                    await asyncio.sleep(self.poll_secs)
                    continue
                said_outside = False
                due = [kc for kc in due_candidates(self.state, now, load_dnc()) if kc[0] not in self.active]
                if not due and not self.active and stop_when_idle:
                    break
                if due and len(self.active) < cfg["max_concurrent"]:
                    key, c = due[0]
                    self.active[key] = asyncio.create_task(self._call_one(key, c))
                    await asyncio.sleep(cfg["gap_between_calls_secs"])
                    continue
                await asyncio.sleep(self.poll_secs)
        finally:
            if self.active:
                logger.info(f"Waiting for {len(self.active)} call(s) in progress to finish...")
                await asyncio.gather(*self.active.values(), return_exceptions=True)
            self.save()
        if self.stop_reason:
            logger.warning(f"Campaign paused: {self.stop_reason}")
        logger.info(f"Campaign {self.state['campaign']}: {summary_line(self.state)}")


def summary_line(state: dict) -> str:
    counts: dict[str, int] = {}
    for c in state["candidates"].values():
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    order = ["pending", "callback", "retry", "in_call", "done", "needs_human", "do_not_call", "failed", "invalid"]
    return ", ".join(f"{counts[s]} {s}" for s in order if counts.get(s)) or "empty list"
