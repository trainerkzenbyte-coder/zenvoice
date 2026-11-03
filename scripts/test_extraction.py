"""Score post-call extraction against 14 realistic call transcripts. No phone needed.

Run from the project folder (virtual environment active):
    python scripts/test_extraction.py                       # all cases
    python scripts/test_extraction.py --only 03             # one case
    python scripts/test_extraction.py --model gemini-3.5-flash   # compare a model
    python scripts/test_extraction.py --show                # print each full outcome

Each case in tests/extraction/ has a transcript and the exact answers expected.
A value of null in "expected" means the field must stay EMPTY (catches guessing).
Costs about Rs 2-3 for a full run.
"""

import argparse
import asyncio
import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.outcome import describe, extract_outcome  # noqa: E402

CASES_DIR = ROOT / "tests" / "extraction"
# Fixed "now" so callback dates are predictable: Monday 5 Oct 2026, 10:00 IST
NOW = dt.datetime(2026, 10, 5, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


def matches(field: str, expected, actual) -> bool:
    if expected is None:
        return actual is None
    if actual is None:
        return False
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= 0.05
        except (TypeError, ValueError):
            return False
    if field == "callback_time":  # same day and hour is enough
        return str(actual)[:13] == str(expected)[:13]
    return str(expected).lower() in str(actual).lower()


async def run_case(path: Path, model: str | None, sem: asyncio.Semaphore) -> dict:
    case = json.loads(path.read_text(encoding="utf-8"))
    async with sem:
        try:
            outcome = await extract_outcome(case["transcript"], now=NOW, model=model)
        except Exception as e:  # noqa: BLE001
            return {"name": path.stem, "case": case, "error": str(e), "results": {}}
    results = {}
    for field, expected in case["expected"].items():
        actual = outcome["call_outcome"] if field == "call_outcome" else outcome["data"].get(field)
        results[field] = (matches(field, expected, actual), expected, actual)
    return {"name": path.stem, "case": case, "outcome": outcome, "results": results}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run cases whose file name contains this text")
    ap.add_argument("--model", help="override extraction.model from config/screening.yaml")
    ap.add_argument("--show", action="store_true", help="print each full outcome")
    args = ap.parse_args()

    paths = sorted(p for p in CASES_DIR.glob("*.json") if not args.only or args.only in p.name)
    if not paths:
        sys.exit(f"No test cases found in {CASES_DIR}")
    print(f"Running {len(paths)} extraction case(s)...\n")
    sem = asyncio.Semaphore(4)
    runs = await asyncio.gather(*(run_case(p, args.model, sem) for p in paths))

    total = passed = 0
    per_field: dict[str, list[bool]] = {}
    for r in runs:
        if r.get("error"):
            print(f"ERROR {r['name']}: {r['error']}")
            continue
        ok_all = all(ok for ok, _, _ in r["results"].values())
        mark = "PASS" if ok_all else "FAIL"
        print(f"{mark}  {r['name']:<28} {r['case']['description']}")
        for field, (ok, exp, act) in r["results"].items():
            total += 1
            passed += ok
            per_field.setdefault(field, []).append(ok)
            if not ok:
                print(f"      {field}: expected {exp!r}, got {act!r}")
        if args.show or not ok_all:
            for line in describe(r["outcome"]).splitlines():
                print(f"      | {line}")
        print()

    if total:
        print(f"Field accuracy: {passed}/{total} = {100 * passed / total:.0f}%")
        weak = [f"{f} {sum(v)}/{len(v)}" for f, v in per_field.items() if not all(v)]
        if weak:
            print("Fields with misses: " + ", ".join(weak))
        print("Target before live calls: 95%+ overall and call_outcome correct in every case.")
    errors = sum(1 for r in runs if r.get("error"))
    sys.exit(0 if total and passed == total and not errors else 1)


if __name__ == "__main__":
    asyncio.run(main())
