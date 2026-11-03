"""Sprint 0 setup check.

Run from the project folder with the virtual environment active:
    python scripts/check_setup.py

Checks: Python version, required packages, config file, and that the
Deepgram and Gemini keys in .env are valid (read-only API calls, no cost).
Keys are never printed.
"""

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results: list[tuple[bool, str]] = []


def report(ok: bool, message: str) -> None:
    results.append((ok, message))
    print(f"[{'OK  ' if ok else 'FAIL'}] {message}")


# 1. Python version
v = sys.version_info
report(
    (3, 11) <= (v.major, v.minor) < (3, 14),
    f"Python {v.major}.{v.minor} (need 3.11 to 3.13)",
)

# 2. Packages
for module, pip_name in [
    ("pipecat", "pipecat-ai"),
    ("fastapi", "fastapi"),
    ("dotenv", "python-dotenv"),
    ("yaml", "pyyaml"),
    ("httpx", "httpx"),
    ("loguru", "loguru"),
    ("psutil", "psutil"),
]:
    try:
        importlib.import_module(module)
        report(True, f"Package {pip_name} installed")
    except ImportError:
        report(False, f"Package {pip_name} missing -> run: pip install -r requirements.txt")

# 3. Config + .env
try:
    from app.config import settings  # noqa: E402

    report(True, f"config/agent.yaml loaded (LLM: {settings['llm']['model']})")
except Exception as e:  # noqa: BLE001
    report(False, f"config/agent.yaml could not be loaded: {e}")

if not (ROOT / ".env").exists():
    report(False, ".env not found -> copy .env.example to .env and add your keys")

# 4. API keys (only if httpx is available)
try:
    import httpx

    dg = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not dg:
        report(False, "DEEPGRAM_API_KEY is empty in .env")
    else:
        try:
            r = httpx.get(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {dg}"},
                timeout=15,
            )
            report(r.status_code == 200, f"Deepgram key {'valid' if r.status_code == 200 else f'rejected (HTTP {r.status_code})'}")
        except httpx.HTTPError as e:
            report(False, f"Could not reach Deepgram: {e}")

    gk = os.getenv("GOOGLE_API_KEY", "").strip()
    if not gk:
        report(False, "GOOGLE_API_KEY is empty in .env")
    else:
        try:
            r = httpx.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": gk},
                timeout=15,
            )
            report(r.status_code == 200, f"Gemini key {'valid' if r.status_code == 200 else f'rejected (HTTP {r.status_code})'}")
        except httpx.HTTPError as e:
            report(False, f"Could not reach Gemini: {e}")

    # ---- Sprint 2: Vobiz + public URL (checked once any Vobiz value is filled in) ----
    vid = os.getenv("VOBIZ_AUTH_ID", "").strip()
    vtok = os.getenv("VOBIZ_AUTH_TOKEN", "").strip()
    vnum = os.getenv("VOBIZ_FROM_NUMBER", "").strip()
    pub = os.getenv("PUBLIC_BASE_URL", "").strip()
    if vid or vtok or vnum or pub:
        print("\nSprint 2 (phone) checks:")
        for module, pip_name in [("multipart", "python-multipart"), ("websockets", "pipecat-ai[websocket]")]:
            try:
                importlib.import_module(module)
                report(True, f"Package {pip_name} installed")
            except ImportError:
                report(False, f"Package {pip_name} missing -> run: pip install -r requirements.txt")
        report(bool(vnum), "VOBIZ_FROM_NUMBER set" if vnum else "VOBIZ_FROM_NUMBER is empty in .env")
        pub_ok = pub.startswith("https://")
        report(
            pub_ok,
            f"PUBLIC_BASE_URL looks right ({pub})" if pub_ok
            else "PUBLIC_BASE_URL must be your https:// ngrok domain",
        )
        if vid and vtok:
            try:
                r = httpx.get(
                    f"https://api.vobiz.ai/api/v1/Account/{vid}/Call/",  # read-only: live calls
                    headers={"X-Auth-ID": vid, "X-Auth-Token": vtok},
                    timeout=15,
                )
                ok = r.status_code == 200
                report(ok, "Vobiz credentials valid" if ok else f"Vobiz credentials rejected (HTTP {r.status_code})")
            except httpx.HTTPError as e:
                report(False, f"Could not reach Vobiz: {e}")
        else:
            report(False, "VOBIZ_AUTH_ID / VOBIZ_AUTH_TOKEN missing in .env")

    # ---- Sprint 4: Google Sheet webhook (checked once a value is filled in) ----
    surl = os.getenv("SHEET_WEBHOOK_URL", "").strip()
    ssec = os.getenv("SHEET_WEBHOOK_SECRET", "").strip()
    if surl or ssec:
        print("\nSprint 4 (Google Sheet) checks:")
        report(surl.startswith("https://script.google.com/") and surl.rstrip("/").endswith("/exec"),
               "SHEET_WEBHOOK_URL looks like an Apps Script web app (.../exec)" if surl
               else "SHEET_WEBHOOK_URL is empty in .env")
        report(len(ssec) >= 16, "SHEET_WEBHOOK_SECRET set" if len(ssec) >= 16
               else "SHEET_WEBHOOK_SECRET missing or shorter than 16 characters")
        if surl:
            try:
                r = httpx.get(surl, timeout=20, follow_redirects=True)
                ok = r.status_code == 200 and '"ok":true' in r.text.replace(" ", "")
                report(ok, "Apps Script web app reachable" if ok else
                       f"Apps Script web app did not answer as expected (HTTP {r.status_code}); "
                       "check it is deployed with access 'Anyone'")
            except httpx.HTTPError as e:
                report(False, f"Could not reach the Apps Script web app: {e}")

    # ---- Sprint 5: cloud server (checked once ADMIN_TOKEN or DOMAIN is set) ----
    admin = os.getenv("ADMIN_TOKEN", "").strip()
    domain = os.getenv("DOMAIN", "").strip()
    if admin or domain:
        print("\nSprint 5 (cloud server) checks:")
        report(len(admin) >= 16, "ADMIN_TOKEN set" if len(admin) >= 16
               else "ADMIN_TOKEN missing or shorter than 16 characters")
        if pub.startswith("https://"):
            try:
                r = httpx.get(pub.rstrip("/") + "/status", timeout=15,
                              headers={"Authorization": f"Bearer {admin}"})
                if r.status_code == 200:
                    s = r.json()
                    report(True, f"Server reachable at {pub}: {s.get('active_calls')} active call(s), "
                                 f"{s.get('cpu_count')} CPUs")
                elif r.status_code == 401:
                    report(False, "Server reachable but ADMIN_TOKEN differs from the server's .env")
                else:
                    report(False, f"{pub}/status answered HTTP {r.status_code} (server not updated to Sprint 5?)")
            except httpx.HTTPError as e:
                report(False, f"Could not reach {pub}: {e}")
except ImportError:
    pass

failed = sum(1 for ok, _ in results if not ok)
print()
if failed:
    print(f"{failed} check(s) failed. Fix them and run this script again.")
    sys.exit(1)
print("All checks passed.")
