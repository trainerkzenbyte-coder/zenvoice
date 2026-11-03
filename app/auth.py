"""Sprint 12: recruiter logins for the console, so it can be opened from anywhere in a browser.

Who can sign in, and with what:
  recruiter  username + password + TODAY'S CODE (6 digits, changes every day at midnight IST; the
             admin sees it on the Team page and shares it each morning)
  admin      username + password + the 6-digit code from an authenticator app on their phone
             (Google Authenticator / Microsoft Authenticator; set up once on the Team page)
  owner      ADMIN_TOKEN from .env, ONLY through the SSH tunnel (never from the internet). Used to
             create the first admin account.

What a recruiter can do: make calls, run campaigns, see calls / recordings / contacts / analytics /
live calls, use the Call Library. What they can't: change the agent, knowledge base, voices,
settings or keys, delete anything, export contacts, read server logs. The server enforces this
(hiding menu items is only for convenience).

Also: sessions end at midnight IST (everyone signs in again each day); 5 wrong tries lock that
username for 15 minutes (20 from one internet address lock that address); "New code" on the Team
page replaces today's code at once and signs out every recruiter; every action is written to
data/auth/audit.log (who, when, from where, what).

Files (data/auth/, never in git): users.json (passwords stored as scrypt hashes), sessions.json, audit.log
.env (optional): AUTH_SECRET - secret for the daily code (default: derived from ADMIN_TOKEN).
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import struct
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import ROOT

DIR = Path(ROOT) / "data" / "auth"
USERS = DIR / "users.json"
SESSIONS = DIR / "sessions.json"
STATE = DIR / "state.json"
AUDIT = DIR / "audit.log"
IST = ZoneInfo("Asia/Kolkata")
ROLES = ("admin", "recruiter")
USER_RE = re.compile(r"^[a-z0-9._-]{3,32}$")
MAX_FAILS, FAIL_WINDOW = 5, 15 * 60
MAX_IP_FAILS = 20

_lock = threading.Lock()
_fails: dict[str, list[float]] = {}


# ---------- small storage helpers ----------
def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def users() -> dict:
    return _read(USERS, {})


def _save_users(u: dict) -> None:
    _write(USERS, u)


# ---------- passwords ----------
def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(h).decode()


def check_password(pw: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=base64.b64decode(salt), n=2 ** 14, r=8, p=1, dklen=32)
        return hmac.compare_digest(got, base64.b64decode(h))
    except Exception:  # noqa: BLE001
        return False


def password_problem(pw: str) -> str:
    if len(pw or "") < 10:
        return "Password must be at least 10 characters"
    if pw.lower() == pw or not re.search(r"\d", pw):
        return "Use upper- and lower-case letters and at least one number"
    return ""


# ---------- today's code (recruiters) ----------
def _secret() -> bytes:
    s = os.getenv("AUTH_SECRET", "").strip() or os.getenv("ADMIN_TOKEN", "").strip()
    if not s:
        raise RuntimeError("Set ADMIN_TOKEN (or AUTH_SECRET) in .env")
    return hashlib.sha256(b"zenbyte-daily-code|" + s.encode()).digest()


def today(now: dt.datetime | None = None) -> str:
    return (now or dt.datetime.now(IST)).astimezone(IST).strftime("%Y-%m-%d")


def _generation(day: str) -> int:
    return int((_read(STATE, {}).get("rotations") or {}).get(day, 0))


def daily_code(day: str | None = None) -> str:
    day = day or today()
    mac = hmac.new(_secret(), f"{day}:{_generation(day)}".encode(), hashlib.sha256).digest()
    return f"{int.from_bytes(mac[:8], 'big') % 1_000_000:06d}"


def rotate_code() -> str:
    with _lock:
        st = _read(STATE, {})
        rot = st.setdefault("rotations", {})
        day = today()
        rot[day] = int(rot.get(day, 0)) + 1
        for d in [d for d in rot if d < day]:  # keep the file small
            rot.pop(d)
        _write(STATE, st)
    end_sessions(role="recruiter")
    return daily_code()


def midnight_ist(now: dt.datetime | None = None) -> float:
    n = (now or dt.datetime.now(IST)).astimezone(IST)
    return (n.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)).timestamp()


# ---------- authenticator app codes (admins) ----------
def new_totp_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def totp(secret: str, at: float | None = None, step: int = 30) -> str:
    key = base64.b32decode(secret + "=" * (-len(secret) % 8), casefold=True)
    counter = int((at or time.time()) // step)
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    return f"{(struct.unpack('>I', mac[off:off + 4])[0] & 0x7FFFFFFF) % 1_000_000:06d}"


def totp_ok(secret: str, code: str, at: float | None = None) -> bool:
    at = at or time.time()
    return any(hmac.compare_digest(totp(secret, at + d * 30), code) for d in (-1, 0, 1))


def totp_uri(username: str, secret: str) -> str:
    from urllib.parse import quote

    return f"otpauth://totp/{quote('Zenbyte Voice:' + username)}?secret={secret}&issuer={quote('Zenbyte Voice')}"


# ---------- sessions ----------
def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _sessions() -> dict:
    s = _read(SESSIONS, {})
    now = time.time()
    return {k: v for k, v in s.items() if v.get("expires", 0) > now}


def new_session(username: str, role: str) -> tuple[str, float]:
    token = secrets.token_urlsafe(32)
    exp = midnight_ist()
    with _lock:
        s = _sessions()
        s[_h(token)] = {"user": username, "role": role, "expires": exp, "created": time.time()}
        _write(SESSIONS, s)
    return token, exp


def end_sessions(*, token: str | None = None, user: str | None = None, role: str | None = None) -> None:
    with _lock:
        s = _sessions()
        if token:
            s.pop(_h(token), None)
        else:
            s = {k: v for k, v in s.items() if not ((user and v["user"] == user) or (role and v["role"] == role))}
        _write(SESSIONS, s)


def _forwarded(request) -> bool:
    """True when the request came through Caddy (the internet), not the SSH tunnel."""
    return any(h in request.headers for h in ("x-forwarded-for", "x-forwarded-host", "forwarded"))


def client_ip(request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:  # the last address is the one Caddy itself saw (earlier ones could be made up by the client)
        return xff.split(",")[-1].strip()[:64]
    return (request.client.host if request.client else "")[:64]


def identify_token(token: str, request) -> dict | None:
    """Who this token belongs to: {"user", "role"} or None."""
    token = (token or "").strip()
    if not token:
        return None
    owner = os.getenv("ADMIN_TOKEN", "").strip()
    if owner and hmac.compare_digest(token, owner):
        if _forwarded(request) and os.getenv("AUTH_ALLOW_ADMIN_TOKEN_REMOTE", "").strip() != "1":
            return None  # the master key never works from the internet
        return {"user": "owner", "role": "admin"}
    sess = _sessions().get(_h(token))
    if not sess:
        return None
    u = users().get(sess["user"])
    if not u or u.get("disabled") or u.get("role") != sess["role"]:
        return None
    return {"user": sess["user"], "role": sess["role"]}


def identify(request) -> dict | None:
    auth = request.headers.get("authorization", "")
    return identify_token(auth[7:] if auth.lower().startswith("bearer ") else "", request)


# ---------- what a recruiter may do ----------
_R_GET = [r"/api/status", r"/api/config", r"/api/auth/me", r"/api/calls(/.*)?", r"/api/live", r"/api/campaigns",
          r"/api/campaign/state", r"/api/contacts", r"/api/crm/contacts(/[^/]+(/calls)?)?", r"/api/crm/fields",
          r"/api/analytics", r"/api/library(/[^/]+(/audio)?)?", r"/api/library/analysis", r"/api/phone-numbers",
          r"/api/numbers", r"/api/batches", r"/api/qa", r"/api/agents(/[^/]+)?", r"/api/voices", r"/api/kbs(/[^/]+)?",
          r"/api/docs/[^/]+", r"/api/test-audio", r"/api/alerts", r"/api/prompt-preview", r"/api/prompt-templates",
          r"/api/logs", r"/api/logs/[^/]+", r"/api/an2/[^/]+"]  # (/api/logs = call logs and chats, not server logs)
_R_WRITE = [("POST", r"/api/call"), ("POST", r"/api/campaigns/upload"), ("POST", r"/api/campaign/(dry-run|start|pause)"),
            ("POST", r"/api/live/[^/]+/hangup"), ("POST", r"/api/chat/(start|[^/]+/send|[^/]+/end)"),
            ("POST", r"/api/library/(upload|transcribe|analyze)"), ("POST", r"/api/crm/contacts"),
            ("PUT", r"/api/crm/contacts/[^/]+"), ("POST", r"/api/crm/contacts/[^/]+/dnc"), ("POST", r"/api/crm/import"),
            ("POST", r"/api/batches"), ("POST", r"/api/batches/[^/]+/(launch|schedule)"), ("POST", r"/api/auth/logout")]
_R_GET_RE = [re.compile(p + "$") for p in _R_GET]
_R_WRITE_RE = [(m, re.compile(p + "$")) for m, p in _R_WRITE]


def recruiter_allowed(method: str, path: str) -> bool:
    if method in ("GET", "HEAD"):
        return any(r.match(path) for r in _R_GET_RE)
    return any(m == method and r.match(path) for m, r in _R_WRITE_RE)


_AUDIT_GETS = re.compile(r"/api/(calls/[^/]+/recording|library/[^/]+/audio|crm/export|logs/[^/]+)$")


def audit(who: str, ip: str, action: str, detail: str = "") -> None:
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        with AUDIT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"), "user": who, "ip": ip,
                                "action": action, "detail": detail[:200]}) + "\n")
    except OSError:
        pass


def install(app) -> None:
    """Role check + audit for every /api call (the routes' own guards still check the sign-in)."""

    @app.middleware("http")
    async def _roles(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") or path == "/status":
            who = identify(request)
            if who and who["role"] != "admin" and not recruiter_allowed(request.method, path):
                audit(who["user"], client_ip(request), "DENIED " + request.method, path)
                return JSONResponse({"detail": "Only an admin can do this"}, status_code=403)
            if who and (request.method not in ("GET", "HEAD") or _AUDIT_GETS.match(path)):
                audit(who["user"], client_ip(request), request.method, path)
        return await call_next(request)


# ---------- login ----------
def _too_many(key: str, limit: int) -> bool:
    now = time.time()
    hits = [t for t in _fails.get(key, []) if now - t < FAIL_WINDOW]
    _fails[key] = hits
    return len(hits) >= limit


def _failed(*keys: str) -> None:
    for k in keys:
        _fails.setdefault(k, []).append(time.time())


BAD_LOGIN = "Wrong username, password or security code"


def login(username: str, password: str, code: str, ip: str) -> dict:
    username = str(username or "").strip().lower()
    code = re.sub(r"\D", "", str(code or ""))
    ukey, ikey = "u:" + username, "ip:" + ip
    if _too_many(ukey, MAX_FAILS) or _too_many(ikey, MAX_IP_FAILS):
        audit(username or "?", ip, "LOGIN BLOCKED")
        raise HTTPException(status_code=429, detail="Too many wrong tries. Wait 15 minutes and try again.")
    u = users().get(username)
    ok = bool(u) and not u.get("disabled") and check_password(password or "", u.get("password", ""))
    if ok:
        if u["role"] == "admin":
            ok = bool(u.get("totp")) and totp_ok(u["totp"], code)
        else:
            ok = hmac.compare_digest(code, daily_code())
    if not u:
        check_password(password or "", hash_password("x"))  # same time whether or not the user exists
    if not ok:
        _failed(ukey, ikey)
        audit(username or "?", ip, "LOGIN FAILED")
        raise HTTPException(status_code=401, detail=BAD_LOGIN)
    _fails.pop(ukey, None)
    token, exp = new_session(username, u["role"])
    with _lock:
        all_u = users()
        if username in all_u:
            all_u[username]["last_login"] = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
            _save_users(all_u)
    audit(username, ip, "LOGIN")
    return {"token": token, "user": username, "name": u.get("name", username), "role": u["role"], "expires": exp}


# ---------- API ----------
router = APIRouter()


def _need(request: Request, admin: bool = False) -> dict:
    who = identify(request)
    if not who:
        raise HTTPException(status_code=401, detail="Please sign in")
    if admin and who["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only an admin can do this")
    return who


async def _body(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        return {}


@router.post("/auth/login")
async def api_login(request: Request):
    b = await _body(request)
    return login(b.get("username", ""), b.get("password", ""), b.get("code", ""), client_ip(request))


@router.post("/api/auth/logout")
async def api_logout(request: Request):
    auth = request.headers.get("authorization", "")
    end_sessions(token=auth[7:] if auth.lower().startswith("bearer ") else "")
    return {"ok": True}


@router.get("/api/auth/me")
async def api_me(request: Request):
    who = _need(request)
    u = users().get(who["user"], {})
    return {**who, "name": u.get("name", "Owner" if who["user"] == "owner" else who["user"]),
            "via_tunnel": not _forwarded(request)}


@router.get("/api/auth/today")
async def api_today(request: Request):
    _need(request, admin=True)
    return {"code": daily_code(), "date": today(), "valid_until": "midnight (IST)"}


@router.post("/api/auth/rotate")
async def api_rotate(request: Request):
    _need(request, admin=True)
    return {"code": rotate_code(), "date": today(), "note": "Old code stopped working; all recruiters were signed out."}


def _public(name: str, u: dict) -> dict:
    return {"username": name, "name": u.get("name", name), "role": u.get("role"), "disabled": bool(u.get("disabled")),
            "created": u.get("created", ""), "last_login": u.get("last_login", ""),
            "authenticator": bool(u.get("totp")) if u.get("role") == "admin" else None}


@router.get("/api/auth/users")
async def api_users(request: Request):
    _need(request, admin=True)
    return {"users": [_public(k, v) for k, v in sorted(users().items())]}


@router.post("/api/auth/users")
async def api_add_user(request: Request):
    who = _need(request, admin=True)
    b = await _body(request)
    name = str(b.get("username", "")).strip().lower()
    role = str(b.get("role", "recruiter"))
    pw = str(b.get("password", ""))
    if not USER_RE.match(name) or name == "owner":
        raise HTTPException(400, "Username: 3-32 characters, lower-case letters, numbers, . _ -")
    if role not in ROLES:
        raise HTTPException(400, "Role must be admin or recruiter")
    if (p := password_problem(pw)):
        raise HTTPException(400, p)
    with _lock:
        u = users()
        if name in u:
            raise HTTPException(400, "That username already exists")
        u[name] = {"name": str(b.get("name") or name)[:60], "role": role, "password": hash_password(pw),
                   "created": dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M"), "created_by": who["user"]}
        if role == "admin":
            u[name]["totp"] = new_totp_secret()
        _save_users(u)
    audit(who["user"], client_ip(request), "ADD USER", f"{name} ({role})")
    out = {"ok": True, "user": _public(name, u[name])}
    if role == "admin":
        out["authenticator_key"] = u[name]["totp"]
        out["authenticator_uri"] = totp_uri(name, u[name]["totp"])
    return out


@router.post("/api/auth/users/{name}")
async def api_edit_user(name: str, request: Request):
    who = _need(request, admin=True)
    b = await _body(request)
    out: dict = {"ok": True}
    with _lock:
        u = users()
        if name not in u:
            raise HTTPException(404, "No such user")
        if "disabled" in b:
            if name == who["user"] and b["disabled"]:
                raise HTTPException(400, "You can't disable yourself")
            u[name]["disabled"] = bool(b["disabled"])
        if b.get("password"):
            if (p := password_problem(str(b["password"]))):
                raise HTTPException(400, p)
            u[name]["password"] = hash_password(str(b["password"]))
        if b.get("new_authenticator") and u[name].get("role") == "admin":
            u[name]["totp"] = new_totp_secret()
            out["authenticator_key"] = u[name]["totp"]
            out["authenticator_uri"] = totp_uri(name, u[name]["totp"])
        _save_users(u)
    end_sessions(user=name)  # any change signs that person out
    audit(who["user"], client_ip(request), "EDIT USER", f"{name}: {', '.join(k for k in b if k != 'password') or 'password'}")
    return out


@router.delete("/api/auth/users/{name}")
async def api_del_user(name: str, request: Request):
    who = _need(request, admin=True)
    if name == who["user"]:
        raise HTTPException(400, "You can't delete yourself")
    with _lock:
        u = users()
        if u.pop(name, None) is None:
            raise HTTPException(404, "No such user")
        _save_users(u)
    end_sessions(user=name)
    audit(who["user"], client_ip(request), "DELETE USER", name)
    return {"ok": True}


@router.get("/api/auth/audit")
async def api_audit(request: Request, limit: int = 200):
    _need(request, admin=True)
    try:
        lines = AUDIT.read_text(encoding="utf-8").splitlines()[-max(1, min(limit, 1000)):]
    except OSError:
        lines = []
    rows = []
    for line in reversed(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return {"rows": rows}


if __name__ == "__main__":  # docker compose exec agent python -m app.auth today
    import sys

    if sys.argv[1:2] == ["today"]:
        print(f"Today's recruiter code ({today()}): {daily_code()}")
