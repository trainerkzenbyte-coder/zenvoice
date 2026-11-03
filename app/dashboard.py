"""Web console (Retell-style) served by the same FastAPI server.

Open http://127.0.0.1:8000/ on the computer running the server and sign in with ADMIN_TOKEN.

Security: the server is reachable from the internet through ngrok (Vobiz needs that), but
the console and every /api/* route only answer requests made ON this computer
(127.0.0.1 / localhost). Requests arriving through ngrok or Caddy get 404. Set
DASHBOARD_ALLOW_REMOTE=1 in .env to allow remote use (still needs ADMIN_TOKEN; use HTTPS).
"""

import asyncio
import datetime as dt
import hmac
import json
import os
import re
import secrets
import time
from collections import Counter
from pathlib import Path

import httpx
import yaml
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from loguru import logger

from app import campaign as cm
from app import knowledge, live, vobiz
from app.call_actions import LLM_ERROR_STATS
from app.config import CONFIG_FILE, ROOT, settings

router = APIRouter()
STATIC = Path(__file__).resolve().parent / "static"
CALL_LOG_DIR = Path(ROOT) / "logs" / "calls"
EVENTS_FILE = Path(ROOT) / "logs" / "call_events.jsonl"
SCREENING_FILE = Path(ROOT) / "config" / "screening.yaml"
CAMPAIGN_FILE = Path(ROOT) / "config" / "campaign.yaml"
PROMPTS_DIR = Path(ROOT) / "config" / "prompts"

status_provider = None          # set by app.server: returns the /status dict
_public = {"t": 0.0, "ok": None, "url": ""}
_vobiz = {"t": 0.0, "ok": None, "detail": ""}
_campaign = {"runner": None, "task": None, "file": "", "started": None, "error": ""}
_chats: dict[str, dict] = {}    # web chat test sessions

TEXT_CHANNELS = {"text simulator", "laptop", "web chat", "browser test"}
DISPLAY_STATUS = {"in_call": "calling", "done": "completed", "do_not_call": "dnc"}
NOISE_PRESETS = {
    "quiet": {"vad_confidence": 0.6, "vad_min_volume": 0.5, "interrupt_min_words": 2},
    "normal": {"vad_confidence": 0.7, "vad_min_volume": 0.6, "interrupt_min_words": 2},
    "noisy": {"vad_confidence": 0.8, "vad_min_volume": 0.7, "interrupt_min_words": 3},
}


# ---------- access control ----------
def _is_local(request: Request) -> bool:
    if os.getenv("DASHBOARD_ALLOW_REMOTE", "").strip() == "1":
        return True
    host = request.headers.get("host", "").rsplit(":", 1)[0].strip("[]").lower()
    forwarded = any(h in request.headers for h in ("x-forwarded-for", "x-forwarded-host", "forwarded"))
    client = request.client.host if request.client else ""
    return host in ("127.0.0.1", "localhost", "::1") and not forwarded and client in ("127.0.0.1", "::1", "localhost")


def _guard(request: Request) -> None:
    """Signed in? A recruiter / admin session (app/auth.py) or ADMIN_TOKEN through the SSH tunnel.
    What each role may do is checked once for every /api call in app.auth.install()."""
    if not _is_local(request):
        raise HTTPException(status_code=404)
    if not os.getenv("ADMIN_TOKEN", "").strip():
        raise HTTPException(status_code=503, detail="Set ADMIN_TOKEN in .env and restart the server")
    from app import auth

    if not auth.identify(request):
        raise HTTPException(status_code=401, detail="Please sign in again")


LIVE_AUDIO_INJECT = """
<script>
(function () {
  // Live Monitoring: hear a call as it happens. Packets: [rate u32][channel u32][PCM16];
  // channel 0 = candidate, 1 = agent, 2 = agent interrupted. The two voices play on separate
  // timelines so they overlap like a real call instead of queueing up and falling behind.
  let ws = null, ctx = null, activeId = null, agentSources = [];
  const next = [0, 0];
  const MAX_LAG = 1.0;  // seconds; if playback falls further behind, skip ahead
  const tok = () => { try { return sessionStorage.getItem("zv_token") || ""; } catch (e) { return ""; } };
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

  function stop() {
    if (ws) { try { ws.close(); } catch (e) {} }
    if (ctx) { try { ctx.close(); } catch (e) {} }
    ws = null; ctx = null; activeId = null; agentSources = [];
  }

  function play(buf) {
    if (!ctx || buf.byteLength < 8) return;
    const view = new DataView(buf);
    const rate = view.getUint32(0, true), ch = view.getUint32(4, true);
    if (ch === 2) {  // candidate interrupted the agent: drop agent audio still queued
      agentSources.forEach((s) => { try { s.stop(); } catch (e) {} });
      agentSources = []; next[1] = ctx.currentTime; return;
    }
    const n = (buf.byteLength - 8) >> 1;
    if (!n || ch > 1) return;
    const pcm = new Int16Array(buf, 8, n), f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = pcm[i] / 32768;
    const ab = ctx.createBuffer(1, n, rate);
    ab.getChannelData(0).set(f);
    const src = ctx.createBufferSource();
    src.buffer = ab; src.connect(ctx.destination);
    const now = ctx.currentTime;
    if (next[ch] < now) next[ch] = now + 0.05;
    if (ch === 0 && next[ch] > now + MAX_LAG) next[ch] = now + 0.05;  // candidate line: stay near real time
    src.start(next[ch]);
    next[ch] += ab.duration;
    if (ch === 1) {
      agentSources.push(src);
      src.onended = () => { agentSources = agentSources.filter((s) => s !== src); };
    }
  }

  window.toggleListenCall = function (callId) {
    if (ws && activeId === callId) { stop(); return; }
    stop();
    activeId = callId;
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    next[0] = next[1] = ctx.currentTime;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws/listen/${encodeURIComponent(callId)}?token=${encodeURIComponent(tok())}`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (ev) => play(ev.data);
    ws.onclose = () => { if (activeId === callId) stop(); };
  };

  setInterval(async () => {
    const t = tok();
    let bar = document.getElementById("live-audio-floating-bar");
    try {
      if (!t) { if (bar) bar.style.display = "none"; return; }
      const res = await fetch("/api/live", { headers: { Authorization: "Bearer " + t } });
      if (!res.ok) return;
      const calls = (await res.json()).calls || [];
      if (activeId && !calls.some((c) => c.call_id === activeId)) stop();
      if (!calls.length) { if (bar) bar.style.display = "none"; return; }
      if (!bar) {
        bar = document.createElement("div");
        bar.id = "live-audio-floating-bar";
        bar.style.cssText = "position:fixed;bottom:20px;right:20px;z-index:99999;background:#111827;color:#fff;padding:10px 14px;border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.3);display:flex;flex-direction:column;gap:8px;font-family:sans-serif;font-size:14px;border:1px solid #374151;max-width:calc(100vw - 40px)";
        bar.addEventListener("click", (e) => { const b = e.target.closest("button[data-call]"); if (b) { toggleListenCall(b.dataset.call); } });
        document.body.appendChild(bar);
      }
      bar.style.display = "flex";
      bar.innerHTML = calls.map((c) => {
        const on = activeId === c.call_id && ws;
        return `<div style="display:flex;align-items:center;gap:10px">
          <span style="width:10px;height:10px;background:#10b981;border-radius:50%;flex:none"></span>
          <div style="flex:1;min-width:0"><strong>${esc(c.candidate_name || "Live call")}</strong> (${esc(c.phone || "Web")})
            <span style="color:#9ca3af;margin-left:6px">${esc(c.elapsed_secs || 0)}s</span></div>
          <button data-call="${esc(c.call_id)}" style="background:${on ? "#ef4444" : "#2563eb"};color:#fff;border:none;padding:6px 12px;border-radius:6px;cursor:pointer;font-weight:600">
            ${on ? "Stop listening" : "Listen live"}</button></div>`;
      }).join("");
    } catch (e) {}
  }, 2000);
})();
</script>
"""


@router.get("/", include_in_schema=False)
@router.get("/dashboard", include_in_schema=False)
async def page(request: Request):
    if not _is_local(request):
        raise HTTPException(status_code=404)
    html_path = STATIC / "dashboard.html"
    html_text = html_path.read_text(encoding="utf-8")
    if "</body>" in html_text:
        html_text = html_text.replace("</body>", LIVE_AUDIO_INJECT + "\n</body>")
    else:
        html_text += LIVE_AUDIO_INJECT
    return HTMLResponse(html_text, headers={"Cache-Control": "no-store"})

# ---------- helpers ----------
def _yaml(path: Path) -> dict:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}


def _clean_str(v, name: str, limit: int, allow_empty: bool = False) -> str:
    s = str(v if v is not None else "").strip()
    if (not s and not allow_empty) or len(s) > limit:
        raise HTTPException(400, f"{name}: 1 to {limit} characters")
    return s


def _num(v, name: str, lo: float, hi: float, integer: bool = False):
    try:
        n = int(v) if integer else float(v)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{name} must be a {'whole ' if integer else ''}number") from None
    if not lo <= n <= hi:
        raise HTTPException(400, f"{name} must be between {lo:g} and {hi:g}")
    return n


def _load_logs() -> list[tuple[Path, dict]]:
    if not CALL_LOG_DIR.exists():
        return []
    out = []
    for path in sorted(CALL_LOG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return out


def _is_test(rec: dict) -> bool:
    return str(rec.get("call_id", "")).startswith("loadtest-")


def _channel(rec: dict) -> str:
    return rec.get("channel") or "phone"


def _when(rec: dict, path: Path) -> dt.datetime:
    for key in ("started_at", "ended_at"):
        try:
            return dt.datetime.fromisoformat(str(rec.get(key, ""))[:19]).replace(tzinfo=cm.IST)
        except ValueError:
            pass
    return dt.datetime.fromtimestamp(path.stat().st_mtime, cm.IST)


def _events() -> list[dict]:
    if not EVENTS_FILE.exists():
        return []
    out = []
    with EVENTS_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


# ---------- status ----------
async def _public_ok() -> tuple[bool | None, str]:
    try:
        url = vobiz.public_base_url()
    except Exception:  # noqa: BLE001
        return None, ""
    if time.time() - _public["t"] > 20 or _public["url"] != url:
        try:
            async with httpx.AsyncClient(timeout=6) as client:
                r = await client.get(f"{url}/health")
            _public["ok"] = r.status_code == 200 and r.text.strip() == "ok"
        except httpx.HTTPError:
            _public["ok"] = False
        _public["t"], _public["url"] = time.time(), url
    return _public["ok"], url


async def _vobiz_ok() -> tuple[bool | None, str]:
    """Read-only check that Vobiz accepts our credentials (cached 60 s)."""
    if time.time() - _vobiz["t"] > 60:
        try:
            auth_id, _ = vobiz._auth()
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(f"{vobiz.API_BASE}/Account/{auth_id}/Call/", headers=vobiz._headers())
            _vobiz["ok"] = r.status_code == 200
            _vobiz["detail"] = "Connected" if r.status_code == 200 else f"Vobiz answered HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            _vobiz["ok"], _vobiz["detail"] = False, f"Not reachable: {str(e)[:120]}"
        _vobiz["t"] = time.time()
    return _vobiz["ok"], _vobiz["detail"]


def _campaign_info() -> dict:
    runner, task = _campaign["runner"], _campaign["task"]
    running = bool(task and not task.done())
    info = {"running": running, "file": _campaign["file"], "started": _campaign["started"],
            "error": _campaign["error"], "pausing": bool(running and runner and runner.stop_reason)}
    if runner:
        info["summary"] = cm.summary_line(runner.state)
        info["active_calls"] = len([t for t in runner.active.values() if not t.done()])
        if runner.stop_reason and not running:
            info["stopped_because"] = runner.stop_reason
    return info


@router.get("/api/status")
async def api_status(request: Request):
    _guard(request)
    ok, url = await _public_ok()
    base = status_provider() if status_provider else {}
    limits = settings.get("call_limits", {}) or {}
    now = cm.now_ist()
    result = {**base, "public_ok": ok, "public_url": url, "campaign": _campaign_info(),
            "agent_name": settings.get("agent", {}).get("name", "Priya"),
            "company": settings.get("agent", {}).get("company", ""),
            "user_label": os.getenv("DASHBOARD_USER", "").strip() or "Admin",
            "model": settings.get("llm", {}).get("model", ""),
            "voice": settings.get("tts", {}).get("voice", ""),
            "max_concurrent": cm.campaign_cfg()["max_concurrent"],
            "live_calls": len(live.LIVE_CALLS),
            "calling_hours": limits.get("calling_hours", ["09:00", "21:00"]),
            "in_calling_hours": cm.in_calling_hours(now), "now": cm.iso(now)}
    from app import dashboard_ext  # console v3: workspace, onboarding, alerts

    try:
        result.update(await dashboard_ext.status_extras(result))
    except Exception as e:  # noqa: BLE001  (the status bar must always load)
        logger.warning(f"Console status extras failed: {e}")
    return result


# ---------- Agents: configuration ----------
def _prompt_path() -> Path:
    name = str(settings.get("agent", {}).get("prompt_file", "sprint3_screening.txt"))
    path = (PROMPTS_DIR / name).resolve()
    if PROMPTS_DIR.resolve() not in path.parents:
        raise HTTPException(400, "prompt_file must be inside config/prompts")
    return path


SHEET_SYSTEM_SOURCES = {"candidate_name", "phone", "call_id", "called_at", "duration_secs", "ended_by", "call_outcome",
                        "summary", "recruiter_notes", "checks"}


def _protected_fields(scr: dict) -> set[str]:
    """Extraction fields the call logic relies on (callbacks are scheduled from callback_time)."""
    return {"callback_time"}


def _sheet_column_for(scr: dict) -> dict[str, str]:
    """Extraction field -> the Google Sheet column it fills (first match)."""
    out: dict[str, str] = {}
    for col, src in (scr.get("sheet_columns") or {}).items():
        out.setdefault(str(src), str(col))
    return out


def _rebuild_sheet_columns(scr: dict, fields: list, kept: dict) -> dict:
    """New sheet_columns mapping: system columns stay; each extraction field with a column name
    keeps its place (renamed if changed); new ones are added at the end; removed ones are dropped."""
    old = dict(scr.get("sheet_columns") or {})
    wanted: dict[str, str] = {}
    for f in fields:
        name = str(f.get("name", "")).strip()
        if name not in kept:
            continue
        if "sheet_column" in f:
            col = str(f.get("sheet_column") or "").strip()[:60]
        else:  # older client: keep whatever this field had
            col = next((str(c) for c, src in old.items() if src == name), "")
        if col:
            wanted[name] = col
    out: dict[str, str] = {}

    def put(col: str, src: str):
        if col in out:
            raise HTTPException(400, f"The Sheet column '{col}' is used twice")
        out[col] = src

    for col, src in old.items():
        src = str(src)
        if src in SHEET_SYSTEM_SOURCES or src.startswith("job."):
            put(str(col), src)
        elif src in wanted:
            put(wanted.pop(src), src)
    for src, col in wanted.items():
        put(col, src)
    return out


@router.get("/api/config")
async def api_config(request: Request):
    _guard(request)
    return await _config_dict()


def _sprint9_cfg() -> dict:
    from app import prompts
    from app import transfer_cfg as transfer

    t, conv = transfer.cfg(), prompts.conversation_cfg()
    limits = settings.get("call_limits", {}) or {}
    return {
        "transfer_enabled": bool(t["enabled"]), "transfer_number": str(t["number"] or ""),
        "transfer_mode": t["mode"], "transfer_announce": t["announce"],
        "reminder_count": int(limits.get("reminder_count", 1)),
        "reminder_line": limits.get("reminder_line") or "Hello, are you still there?",
        "natural_style": bool(conv.get("natural_style", True)),
        "contact_memory": bool(conv.get("contact_memory", True)),
        "knowledge_max_chars": __import__("app.kb_store", fromlist=["x"]).prompt_limit(),
        "stt_provider": (settings.get("stt") or {}).get("provider", "deepgram"),
        "history_messages": int((settings.get("llm") or {}).get("max_history_messages", 24)),
        "smart_turn_max_wait_secs": float((settings.get("turn") or {}).get("smart_turn_max_wait_secs", 0.8)),
        "off_limits": list(conv.get("off_limits") or prompts.DEFAULT_OFF_LIMITS),
        "deflect_line": conv.get("deflect_line") or prompts.DEFAULT_DEFLECT,
        "pronunciation": list((settings.get("tts") or {}).get("pronunciation") or []),
    }


async def _config_dict() -> dict:
    agent = settings.get("agent", {}) or {}
    limits = settings.get("call_limits", {}) or {}
    llm = settings.get("llm", {}) or {}
    turn = settings.get("turn", {}) or {}
    scr = _yaml(SCREENING_FILE)
    camp = cm.campaign_cfg()
    pp = _prompt_path()
    sheet_col = _sheet_column_for(scr)
    fields = [{"name": k, "type": f.get("type", "string"), "description": f.get("description", ""),
               "values": f.get("values", []), "min": f.get("min"), "max": f.get("max"),
               "sheet_column": sheet_col.get(k, "")}
              for k, f in (scr.get("fields") or {}).items()]
    greet = scr.get("greeting") or {}
    stt = settings.get("stt", {}) or {}
    from app.voice_config import plan as _voice_plan

    voice_plan = _voice_plan(settings)
    noise = {"vad_confidence": turn.get("vad_confidence", 0.7), "vad_min_volume": turn.get("vad_min_volume", 0.6),
             "interrupt_min_words": turn.get("interrupt_min_words", 2)}
    preset = next((n for n, p in NOISE_PRESETS.items() if all(float(noise[k]) == float(v) for k, v in p.items())),
                  "custom")
    return {
        "agent_name": agent.get("name", ""), "company": agent.get("company", ""),
        "model": (llm.get("model", "") if llm.get("provider", "google") == "google"
                  else f'{llm.get("provider")}:{llm.get("model", "")}'), "temperature": llm.get("temperature", 0.4),
        "voice": (settings.get("tts") or {}).get("voice", ""),
        "call_language": voice_plan["language"], "voice_speed": voice_plan["speed"],
        "voice_plan": {k: voice_plan[k] for k in ("stt", "tts", "voice", "note", "needs_sarvam")},
        "sarvam_key_set": bool(os.getenv("SARVAM_API_KEY", "").strip()),
        **_sprint9_cfg(),
        "max_call_seconds": limits.get("max_call_seconds", 240),
        "silence_hangup_seconds": limits.get("silence_hangup_seconds", 8),
        "calling_start": (limits.get("calling_hours") or ["09:00", "21:00"])[0],
        "calling_end": (limits.get("calling_hours") or ["09:00", "21:00"])[1],
        **noise, "noise_preset": preset,
        "job_title": (scr.get("job") or {}).get("title", ""),
        "job_location": (scr.get("job") or {}).get("location", ""),
        "job_key_skills": (scr.get("job") or {}).get("key_skills", ""),
        "job_experience_range": (scr.get("job") or {}).get("experience_range", ""),
        "greeting_with_name": greet.get("with_name", ""),
        "greeting_without_name": greet.get("without_name", ""),
        "greeting_mode": "user_first" if greet.get("mode") == "user_first" else "ai_first",
        "greeting_delay_secs": greet.get("delay_secs", 0) or 0,
        "stt_language": stt.get("language", "multi"), "stt_model": stt.get("model", "nova-3"),
        "keyterms": list(stt.get("keyterms") or []),
        "vad_stop_secs": turn.get("vad_stop_secs", 0.7),
        "record_calls": bool((settings.get("recording") or {}).get("enabled", False)),
        "ring_timeout_secs": (settings.get("phone") or {}).get("ring_timeout_secs", 30),
        "sheet_timeout_secs": (settings.get("sheet") or {}).get("timeout_secs", 30),
        "sheet_system_columns": {c: s for c, s in (scr.get("sheet_columns") or {}).items()
                                 if str(s) in SHEET_SYSTEM_SOURCES or str(s).startswith("job.")},
        "fields": fields, "protected_fields": sorted(_protected_fields(scr)),
        "max_concurrent": camp["max_concurrent"], "retry_gap_minutes": camp["retry_gap_minutes"],
        "max_attempts": camp["max_attempts"],
        "prompt_file": pp.name, "prompt": pp.read_text(encoding="utf-8") if pp.exists() else "",
        "knowledge_chars": len(knowledge.prompt_section()),
    }


_VALUE = r'("(?:[^"\\]|\\.)*"|\'[^\']*\'|\[[^\]]*\]|[^#\n]*?)'


def set_yaml_value(text: str, section: str | None, key: str, value) -> str:
    """Change one value in a YAML file, keeping every comment and the layout."""
    rendered = json.dumps(value, ensure_ascii=False)
    lines = text.splitlines(keepends=True)
    if section is None:
        start, end, indent = 0, len(lines), ""
    else:
        start = next((i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(section)}:\s*(#.*)?$", ln)), None)
        if start is None:
            sep = "" if text.endswith("\n") or not text else "\n"
            return f"{text}{sep}\n{section}:\n  {key}: {rendered}\n"
        start += 1
        end = start
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t"))):
            end += 1
        indent = "  "
    pat = re.compile(rf"^(\s*{re.escape(key)}:[ \t]*){_VALUE}([ \t]+#.*)?$")
    for i in range(start, end):
        stripped = lines[i].rstrip("\r\n")
        if section is None and lines[i][:1] in (" ", "\t"):
            continue
        if section is not None and len(stripped) - len(stripped.lstrip()) != 2:
            continue  # only direct children of the section
        m = pat.match(stripped)
        if m:
            newline = lines[i][len(stripped):]
            lines[i] = f"{m.group(1)}{rendered}{m.group(3) or ''}{newline or chr(10)}"
            return "".join(lines)
    lines.insert(start, f"{indent}{key}: {rendered}\n")
    return "".join(lines)


def replace_block(text: str, key: str, data: dict) -> str:
    """Replace a whole top-level block (e.g. 'fields:') with freshly written YAML."""
    block = yaml.safe_dump({key: data}, sort_keys=False, allow_unicode=True, width=1000, default_flow_style=None)
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(key)}:\s*(#.*)?$", ln)), None)
    if start is None:
        return text.rstrip("\n") + "\n\n" + block
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t"))):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1  # keep the blank line(s) before the next section
    return "".join(lines[:start]) + block + "".join(lines[end:])


def _write_yaml(path: Path, changes: list[tuple[str | None, str, object]], blocks: dict | None = None) -> dict:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    for section, key, value in changes:
        text = set_yaml_value(text, section, key, value)
    for key, data in (blocks or {}).items():
        text = replace_block(text, key, data)
    data = yaml.safe_load(text) or {}
    for section, key, value in changes:  # verify before saving
        got = (data.get(section) or {}).get(key) if section else data.get(key)
        if got != value:
            raise HTTPException(500, f"Could not update {key} in {path.name} safely; edit it by hand")
    for key, value in (blocks or {}).items():
        if data.get(key) != value:
            raise HTTPException(500, f"Could not update {key} in {path.name} safely")
    if path.exists():
        path.with_suffix(path.suffix + ".bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    return data


def _clean_fields(raw: list, protected: set[str]) -> dict:
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, "Keep at least one extraction field")
    out: dict = {}
    for f in raw:
        name = str(f.get("name", "")).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", name):
            raise HTTPException(400, f"Field name '{name}': lowercase letters, numbers and _ only")
        if name in out:
            raise HTTPException(400, f"Field '{name}' appears twice")
        ftype = str(f.get("type", "string"))
        if ftype not in ("string", "number", "integer", "boolean", "enum"):
            raise HTTPException(400, f"Field '{name}': unknown type {ftype}")
        entry: dict = {"type": ftype}
        if ftype == "enum":
            values = f.get("values") or []
            if isinstance(values, str):
                values = [v.strip() for v in values.split(",")]
            values = [str(v).strip() for v in values if str(v).strip()]
            if len(values) < 2:
                raise HTTPException(400, f"Field '{name}': give at least two choices, comma separated")
            entry["values"] = values
        if ftype in ("number", "integer"):
            for bound in ("min", "max"):
                if f.get(bound) not in (None, ""):
                    entry[bound] = _num(f[bound], f"{name} {bound}", -1e9, 1e9, integer=ftype == "integer")
        entry["description"] = _clean_str(f.get("description", ""), f"Field '{name}' description", 300)
        out[name] = entry
    missing = sorted(p for p in protected if p not in out)
    if missing:
        raise HTTPException(400, f"These fields are used by the Google Sheet or callbacks and can't be removed: "
                                 f"{', '.join(missing)}")
    return out


def parse_config(body: dict) -> dict:
    """Validate a settings change (raises HTTPException 400). Writes nothing."""
    agent_changes, screening_changes, campaign_changes, blocks = [], [], [], {}

    if "agent_name" in body:
        agent_changes.append(("agent", "name", _clean_str(body["agent_name"], "Agent name", 40)))
    if "company" in body:
        agent_changes.append(("agent", "company", _clean_str(body["company"], "Company", 80)))
    if "model" in body:
        model = _clean_str(body["model"], "Model", 80)
        if re.fullmatch(r"gemini-[a-z0-9.\-]+", model):
            agent_changes += [("llm", "provider", "google"), ("llm", "model", model)]
        elif re.fullmatch(r"(groq|cerebras|openai):[A-Za-z0-9._/\-]{2,60}", model):
            provider, name = model.split(":", 1)
            agent_changes += [("llm", "provider", provider), ("llm", "model", name)]
        else:
            raise HTTPException(400, "Model: a Gemini id (gemini-3.5-flash-lite) or groq:/cerebras:/openai:<model>")
    if "temperature" in body:
        agent_changes.append(("llm", "temperature", round(_num(body["temperature"], "Temperature", 0, 1.5), 2)))
    if "voice" in body:
        from app.voice_config import valid_voice

        voice = _clean_str(body["voice"], "Voice", 80)
        if not valid_voice(voice):
            raise HTTPException(400, "Pick a voice from the voice list (aura-2-…-en, sarvam:<name> or a cloned voice)")
        agent_changes.append(("tts", "voice", voice))
    if "call_language" in body:
        from app.voice_config import LANGUAGES

        lang = str(body["call_language"])
        if lang not in LANGUAGES:
            raise HTTPException(400, "Unknown call language")
        agent_changes.append(("agent", "language", lang))
    if "voice_speed" in body:
        agent_changes.append(("tts", "speed", round(_num(body["voice_speed"], "Voice speed", 0.5, 2.0), 2)))
    if "max_call_seconds" in body:
        agent_changes.append(("call_limits", "max_call_seconds",
                              _num(body["max_call_seconds"], "Max call length", 60, 3600, True)))
    if "silence_hangup_seconds" in body:
        agent_changes.append(("call_limits", "silence_hangup_seconds",
                              _num(body["silence_hangup_seconds"], "Silence timeout", 3, 60, True)))
    if "calling_start" in body or "calling_end" in body:
        hours = [str(body.get("calling_start", "")).strip(), str(body.get("calling_end", "")).strip()]
        if not all(re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", h) for h in hours) or hours[0] >= hours[1]:
            raise HTTPException(400, "Calling hours: HH:MM to HH:MM, start before end")
        agent_changes.append(("call_limits", "calling_hours", hours))
    if "vad_confidence" in body:
        agent_changes.append(("turn", "vad_confidence", round(_num(body["vad_confidence"], "Speech confidence", 0.3, 0.95), 2)))
    if "vad_min_volume" in body:
        agent_changes.append(("turn", "vad_min_volume", round(_num(body["vad_min_volume"], "Minimum volume", 0.1, 0.95), 2)))
    if "vad_stop_secs" in body:
        agent_changes.append(("turn", "vad_stop_secs", round(_num(body["vad_stop_secs"], "Response wait", 0.2, 1.5), 2)))
    if "stt_language" in body:
        lang = str(body["stt_language"])
        if lang not in ("multi", "en-IN", "en", "hi"):
            raise HTTPException(400, "Language must be multi, en-IN, en or hi")
        agent_changes.append(("stt", "language", lang))
    if "keyterms" in body:
        raw = body["keyterms"]
        terms = raw if isinstance(raw, list) else str(raw or "").split(",")
        terms = [str(t).strip() for t in terms if str(t).strip()]
        if len(terms) > 50 or any(len(t) > 40 or not re.fullmatch(r"[\w .&'+#-]+", t) for t in terms):
            raise HTTPException(400, "Boosted keywords: up to 50 words or short phrases, letters and numbers only")
        agent_changes.append(("stt", "keyterms", terms))
    if "record_calls" in body:
        agent_changes.append(("recording", "enabled", str(body["record_calls"]).lower() in ("true", "1", "yes", "on")))
    if "ring_timeout_secs" in body:
        agent_changes.append(("phone", "ring_timeout_secs", _num(body["ring_timeout_secs"], "Ring duration", 10, 60, True)))
    if "sheet_timeout_secs" in body:
        agent_changes.append(("sheet", "timeout_secs", _num(body["sheet_timeout_secs"], "Webhook timeout", 3, 60, True)))
    if "greeting_mode" in body:
        if body["greeting_mode"] not in ("ai_first", "user_first"):
            raise HTTPException(400, "Welcome message: ai_first or user_first")
        screening_changes.append(("greeting", "mode", body["greeting_mode"]))
    if "greeting_delay_secs" in body:
        screening_changes.append(("greeting", "delay_secs", round(_num(body["greeting_delay_secs"], "Pause before speaking", 0, 5), 1)))
    # ---- Sprint 9: transfer, reminders, guardrails, pronunciation ----
    def _bool(v):
        return str(v).lower() in ("true", "1", "yes", "on")
    if "transfer_enabled" in body:
        agent_changes.append(("transfer", "enabled", _bool(body["transfer_enabled"])))
    if "transfer_number" in body:
        num = re.sub(r"[^\d]", "", str(body["transfer_number"] or ""))
        if num and not 10 <= len(num) <= 13:
            raise HTTPException(400, "Transfer number: 10-digit mobile or with country code, e.g. 919876543210")
        agent_changes.append(("transfer", "number", num))
    if "transfer_mode" in body:
        if body["transfer_mode"] not in ("warm", "cold"):
            raise HTTPException(400, "Transfer type: warm or cold")
        agent_changes.append(("transfer", "mode", body["transfer_mode"]))
    if "transfer_announce" in body:
        agent_changes.append(("transfer", "announce", _clean_str(body["transfer_announce"], "Transfer line", 200)))
    if "reminder_count" in body:
        agent_changes.append(("call_limits", "reminder_count", _num(body["reminder_count"], "Reminders", 0, 5, True)))
    if "reminder_line" in body:
        agent_changes.append(("call_limits", "reminder_line", _clean_str(body["reminder_line"], "Reminder message", 150)))
    if "stt_provider" in body:
        if body["stt_provider"] not in ("deepgram", "deepgram_flux", "sarvam"):
            raise HTTPException(400, "Speech recognition: deepgram, deepgram_flux or sarvam")
        agent_changes.append(("stt", "provider", body["stt_provider"]))
    if "history_messages" in body:
        agent_changes.append(("llm", "max_history_messages", _num(body["history_messages"], "History kept", 0, 80, True)))
    if "smart_turn_max_wait_secs" in body:
        agent_changes.append(("turn", "smart_turn_max_wait_secs", round(_num(body["smart_turn_max_wait_secs"], "Max wait for unfinished sentence", 0.3, 3), 2)))
    if "knowledge_max_chars" in body:
        agent_changes.append(("knowledge", "max_chars", _num(body["knowledge_max_chars"], "Knowledge base size", 2000, 40000, True)))
    if "contact_memory" in body:
        agent_changes.append(("conversation", "contact_memory", _bool(body["contact_memory"])))
    if "natural_style" in body:
        agent_changes.append(("conversation", "natural_style", _bool(body["natural_style"])))
    if "deflect_line" in body:
        agent_changes.append(("conversation", "deflect_line", _clean_str(body["deflect_line"], "Deflection line", 250)))
    if "off_limits" in body:
        raw = body["off_limits"]
        items = raw if isinstance(raw, list) else str(raw or "").splitlines()
        items = [str(i).strip(" -•\t") for i in items if str(i).strip(" -•\t")]
        if len(items) > 30 or any(len(i) > 200 for i in items):
            raise HTTPException(400, "Topics not to discuss: up to 30 lines, 200 characters each")
        agent_changes.append(("conversation", "off_limits", items))
    if "pronunciation" in body:
        rules = []
        for r in (body["pronunciation"] or [])[:100]:
            w, say = str(r.get("word", "")).strip(), str(r.get("say", "")).strip()
            if not w and not say:
                continue
            if not (w and say) or len(w) > 60 or len(say) > 120:
                raise HTTPException(400, "Pronunciation: each row needs a word and how to say it")
            rules.append({"word": w, "say": say})
        agent_changes.append(("tts", "pronunciation", rules))
    if "interrupt_min_words" in body:
        agent_changes.append(("turn", "interrupt_min_words",
                              _num(body["interrupt_min_words"], "Words to interrupt", 1, 5, True)))

    for field, key, limit in (("job_title", "title", 80), ("job_location", "location", 120),
                              ("job_key_skills", "key_skills", 200),
                              ("job_experience_range", "experience_range", 60)):
        if field in body:
            screening_changes.append(("job", key, _clean_str(body[field], field.replace("_", " "), limit)))
    for field, key in (("greeting_with_name", "with_name"), ("greeting_without_name", "without_name")):
        if field in body:
            screening_changes.append(("greeting", key, _clean_str(body[field], "Greeting", 300)))
    if "fields" in body:
        scr = _yaml(SCREENING_FILE)
        blocks["fields"] = _clean_fields(body["fields"], _protected_fields(scr))
        new_cols = _rebuild_sheet_columns(scr, body["fields"], blocks["fields"])
        if new_cols != (scr.get("sheet_columns") or {}):
            blocks["sheet_columns"] = new_cols

    if "max_concurrent" in body:
        campaign_changes.append((None, "max_concurrent", _num(body["max_concurrent"], "Calls at once", 1, 20, True)))
    if "retry_gap_minutes" in body:
        campaign_changes.append((None, "retry_gap_minutes", _num(body["retry_gap_minutes"], "Retry gap", 5, 2880, True)))
    if "max_attempts" in body:
        campaign_changes.append((None, "max_attempts", _num(body["max_attempts"], "Attempts", 1, 10, True)))

    if "prompt" in body:
        text = str(body["prompt"] or "")
        if not text.strip() or len(text) > 30000:
            raise HTTPException(400, "Prompt: 1 to 30,000 characters")
    return {"agent": agent_changes, "screening": screening_changes, "campaign": campaign_changes,
            "blocks": blocks, "prompt": body.get("prompt")}


def apply_config(body: dict) -> None:
    """Validate, then write config/agent.yaml, screening.yaml, campaign.yaml and the prompt file."""
    parsed = parse_config(body)
    agent_changes, screening_changes = parsed["agent"], parsed["screening"]
    campaign_changes, blocks = parsed["campaign"], parsed["blocks"]
    if agent_changes:
        new = _write_yaml(CONFIG_FILE, agent_changes)
        settings.clear()
        settings.update(new)  # new calls use it straight away, no restart needed
    if screening_changes or blocks:
        _write_yaml(SCREENING_FILE, screening_changes, blocks)
    if campaign_changes:
        _write_yaml(CAMPAIGN_FILE, campaign_changes)
    if "prompt" in body:
        pp = _prompt_path()
        if pp.exists():
            pp.with_suffix(pp.suffix + ".bak").write_text(pp.read_text(encoding="utf-8"), encoding="utf-8")
        pp.write_text(str(body["prompt"]).replace("\r\n", "\n"), encoding="utf-8")
    logger.info("Dashboard: settings saved (applies to the next call)")


@router.post("/api/config")
async def api_config_save(request: Request):
    _guard(request)
    body = await request.json()
    apply_config(body)
    from app import agents_store  # keep the active agent profile in step with the live settings

    agents_store.snapshot_active(await _config_snapshot())
    return {"ok": True}


async def _config_snapshot() -> dict:
    """The live agent settings in the same shape the Agents page edits."""
    cfg = await _config_dict()
    keep = agents_store_keys()
    return {k: cfg[k] for k in keep if k in cfg}


def agents_store_keys() -> list[str]:
    from app.agents_store import SETTING_KEYS

    return SETTING_KEYS


# ---------- Phone numbers ----------
@router.get("/api/phone-numbers")
async def api_phone_numbers(request: Request):
    _guard(request)
    pub_ok, url = await _public_ok()
    v_ok, v_detail = await _vobiz_ok()
    number = os.getenv("VOBIZ_FROM_NUMBER", "").strip()
    return {"from_number": number, "carrier": "Vobiz (SIP trunk)", "carrier_ok": v_ok, "carrier_detail": v_detail,
            "public_url": url, "public_ok": pub_ok,
            "answer_url": f"{url}/answer" if url else "", "hangup_url": f"{url}/hangup" if url else "",
            "stream_url": (url.replace("https://", "wss://") + "/ws/•••") if url else "",
            "ring_timeout_secs": (settings.get("phone", {}) or {}).get("ring_timeout_secs", 30),
            "max_call_seconds": (settings.get("call_limits", {}) or {}).get("max_call_seconds", 240)}


# ---------- manual call (dialer / contacts "Dial now") ----------
@router.post("/api/call")
async def api_call(request: Request):
    _guard(request)
    body = await request.json()
    name = str(body.get("name", "")).strip()[:60]
    role = str(body.get("role", "")).strip()[:80]
    try:
        to = vobiz.normalize_number(str(body.get("phone", "")))
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if cm.phone_key(to) in cm.load_dnc():
        raise HTTPException(400, "This number is on the do-not-call list (campaigns/do_not_call.txt)")
    limits = settings.get("call_limits", {}) or {}
    ring = int((settings.get("phone", {}) or {}).get("ring_timeout_secs", 30))
    try:
        result = await vobiz.make_call(to, time_limit_secs=int(limits.get("max_call_seconds", 240)) + 30,
                                       ring_timeout_secs=ring, candidate_name=name, role=role)
    except RuntimeError as e:
        raise HTTPException(502, str(e)[:300]) from None
    logger.info(f"Dashboard: call to {to} ({name or 'no name'})")
    return {"ok": True, "to": to, "vobiz": result}


# ---------- Batch call (campaigns) ----------
def _csv(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-. ]{1,80}\.csv", name or "") or ".." in name:
        raise HTTPException(400, "Choose a .csv file from the list")
    path = cm.CAMPAIGN_DIR / name
    if not path.exists():
        raise HTTPException(404, f"campaigns/{name} not found")
    return path


@router.get("/api/campaigns")
async def api_campaigns(request: Request):
    _guard(request)
    cm.CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(cm.CAMPAIGN_DIR.glob("*.csv")):
        try:
            state = cm.load_state(path)
            out.append({"file": path.name, "summary": cm.summary_line(state), "count": len(state["candidates"])})
        except Exception as e:  # noqa: BLE001
            out.append({"file": path.name, "summary": f"cannot read: {e}", "count": 0})
    return {"files": out, "config": cm.campaign_cfg(), "campaign": _campaign_info()}


@router.post("/api/campaigns/upload")
async def api_campaign_upload(request: Request, file: UploadFile = File(...)):
    _guard(request)
    name = re.sub(r"[^A-Za-z0-9_\-.]+", "_", Path(file.filename or "list.csv").name)
    if not name.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file (Excel: File > Save As > CSV)")
    data = await file.read(1_000_001)
    if len(data) > 1_000_000:
        raise HTTPException(400, "File larger than 1 MB")
    cm.CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    path = cm.CAMPAIGN_DIR / name
    if _campaign["file"] == name and _campaign["task"] and not _campaign["task"].done():
        raise HTTPException(409, "This list is running. Pause it first.")
    tmp = path.with_suffix(".upload")
    tmp.write_bytes(data)
    try:
        rows = cm.read_candidates(tmp)
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read the CSV: {e}") from None
    if not rows or not any(r["phone"] for r in rows):
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "The CSV needs a header row with 'name' and 'phone' columns (optional: 'role')")
    tmp.replace(path)
    return {"ok": True, "file": name, "rows": len(rows)}


@router.get("/api/campaign/state")
async def api_campaign_state(request: Request, file: str):
    _guard(request)
    path = _csv(file)
    runner = _campaign["runner"]
    state = runner.state if runner and _campaign["file"] == file else cm.load_state(path)
    rows = []
    for k, c in state["candidates"].items():
        rows.append({"key": k, "name": c.get("name"), "phone": c.get("phone"), "role": c.get("role", ""),
                     "status": DISPLAY_STATUS.get(c.get("status"), c.get("status")), "attempts": c.get("attempts", 0),
                     "next_at": c.get("next_at"), "note": c.get("note"), "last_result": c.get("last_result")})
    return {"file": file, "summary": cm.summary_line(state), "candidates": rows}


@router.post("/api/campaign/dry-run")
async def api_campaign_dry_run(request: Request):
    _guard(request)
    body = await request.json()
    path = _csv(body.get("file", ""))
    state = cm.load_state(path)
    now = cm.now_ist()
    due = cm.due_candidates(state, now, cm.load_dnc())
    return {"in_calling_hours": cm.in_calling_hours(now), "starts_at": cm.iso(cm.next_calling_time(now)),
            "due": [{"name": c["name"], "phone": c["phone"], "status": c["status"]} for _, c in due],
            "summary": cm.summary_line(state)}


@router.post("/api/campaign/start")
async def api_campaign_start(request: Request):
    _guard(request)
    body = await request.json()
    path = _csv(body.get("file", ""))
    task = _campaign["task"]
    if task and not task.done():
        raise HTTPException(409, f"Campaign {_campaign['file']} is already running. Pause it first.")
    runner = cm.CampaignRunner(path)

    async def run():
        try:
            await runner.run()
        except Exception as e:  # noqa: BLE001
            _campaign["error"] = str(e)[:300]
            logger.exception(f"Campaign crashed: {e}")

    _campaign.update(runner=runner, file=path.name, started=cm.iso(cm.now_ist()), error="",
                     task=asyncio.create_task(run()))
    return {"ok": True, "campaign": _campaign_info()}


@router.post("/api/campaign/pause")
async def api_campaign_pause(request: Request):
    _guard(request)
    runner, task = _campaign["runner"], _campaign["task"]
    if not runner or not task or task.done():
        raise HTTPException(409, "No campaign is running")
    runner.stop_reason = "paused from the dashboard"
    return {"ok": True, "message": "Pausing: no new calls; calls in progress finish normally."}


# ---------- Call history / chat history ----------
def _summary(path: Path, rec: dict) -> dict:
    out = rec.get("outcome") or {}
    lat = rec.get("latency") or {}
    data = out.get("data") or {}
    return {
        "file": path.name, "call_id": rec.get("call_id", ""), "candidate_name": rec.get("candidate_name", ""),
        "phone": rec.get("phone", ""), "role": rec.get("role", ""), "started_at": rec.get("started_at", ""),
        "duration_secs": rec.get("duration_secs"), "outcome": out.get("call_outcome", ""),
        "interested": data.get("interested"), "ended_by": out.get("ended_by") or "",
        "end_reason": rec.get("end_reason", ""), "summary": out.get("summary", ""), "channel": _channel(rec),
        "p50_ms": lat.get("p50_ms"), "turns": len(rec.get("transcript") or []), "simulated": _is_test(rec),
    }


@router.get("/api/logs")
async def api_logs(request: Request, limit: int = 200, include_tests: bool = False, kind: str = "calls"):
    _guard(request)
    calls = []
    for path, rec in _load_logs():
        is_text = _channel(rec) in TEXT_CHANNELS and _channel(rec) != "browser test"  # web calls are in Call History
        if (kind == "chats") != is_text:
            continue
        if _is_test(rec) and not include_tests:
            continue
        calls.append(_summary(path, rec))
        if len(calls) >= max(1, min(limit, 1000)):
            break
    return {"calls": calls}


@router.get("/api/logs/{name}")
async def api_log(request: Request, name: str):
    _guard(request)
    if not re.fullmatch(r"[\w.\-]{1,120}\.json", name):
        raise HTTPException(400, "bad name")
    path = CALL_LOG_DIR / name
    if not path.exists():
        raise HTTPException(404, "not found")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


# ---------- Contacts ----------
@router.get("/api/contacts")
async def api_contacts(request: Request):
    _guard(request)
    dnc = cm.load_dnc()
    people: dict[str, dict] = {}

    def person(key: str) -> dict:
        return people.setdefault(key, {"key": key, "name": "", "phone": "91" + key, "role": "", "status": "",
                                       "list": "", "attempts": 0, "next_at": None, "calls": 0,
                                       "last_call": "", "last_outcome": "", "dnc": key in dnc})

    for path in sorted(cm.CAMPAIGN_DIR.glob("*.csv")) if cm.CAMPAIGN_DIR.exists() else []:
        try:
            state = cm.load_state(path)
        except Exception:  # noqa: BLE001
            continue
        for key, c in state["candidates"].items():
            if not cm.phone_key(key):
                continue
            p = person(key)
            p.update(name=p["name"] or c.get("name", ""), role=p["role"] or c.get("role", ""),
                     status=DISPLAY_STATUS.get(c.get("status"), c.get("status")), list=path.name,
                     attempts=c.get("attempts", 0), next_at=c.get("next_at"))
    for path, rec in reversed(_load_logs()):  # oldest first, so the newest call wins
        key = cm.phone_key(rec.get("phone", ""))
        if not key or _is_test(rec) or _channel(rec) in TEXT_CHANNELS:
            continue
        p = person(key)
        p["calls"] += 1
        p["name"] = p["name"] or rec.get("candidate_name", "")
        p["last_call"] = rec.get("started_at", "")
        p["last_outcome"] = (rec.get("outcome") or {}).get("call_outcome", "")
    for key in dnc:
        person(key)["dnc"] = True
    rows = sorted(people.values(), key=lambda p: (p["last_call"] or "", p["name"]), reverse=True)
    return {"contacts": rows}


# ---------- Analytics ----------
@router.get("/api/analytics")
async def api_analytics(request: Request, days: int = 7):
    _guard(request)
    now = cm.now_ist()
    since = now - dt.timedelta(days=days) if days > 0 else None
    calls = []
    for path, rec in _load_logs():
        if _is_test(rec) or _channel(rec) in TEXT_CHANNELS:
            continue
        when = _when(rec, path)
        if since and when < since:
            continue
        calls.append((when, rec))

    events = [e for e in _events() if not since or float(e.get("t", 0)) >= since.timestamp()]
    by_call: dict[str, dict] = {}
    for e in events:
        by_call[e.get("call_uuid") or e.get("ref") or str(e.get("t"))] = e
    dialed = len(by_call)

    def answered_event(e: dict) -> bool:
        for k in ("bill_duration", "duration"):
            try:
                if float(e.get(k) or 0) > 0:
                    return True
            except ValueError:
                pass
        return False

    answered = sum(1 for e in by_call.values() if answered_event(e)) if dialed else len(calls)
    dialed = max(dialed, len(calls))
    outcomes = Counter((rec.get("outcome") or {}).get("call_outcome") or "unknown" for _, rec in calls)
    conversations = [rec for _, rec in calls if (rec.get("outcome") or {}).get("call_outcome") not in ("voicemail", "dropped")]
    qualified = [rec for rec in conversations
                 if (rec.get("outcome") or {}).get("call_outcome") == "completed"
                 and ((rec.get("outcome") or {}).get("data") or {}).get("interested") in ("yes", True)]
    durations = [float(rec.get("duration_secs") or 0) for _, rec in calls if rec.get("duration_secs")]
    p50s = [rec["latency"]["p50_ms"] for _, rec in calls if (rec.get("latency") or {}).get("p50_ms")]
    firsts = [rec["latency"]["first_audio_ms"] for _, rec in calls if (rec.get("latency") or {}).get("first_audio_ms")]
    llm_errors = sum(1 for _, rec in calls for a in ((rec.get("outcome") or {}).get("actions") or [])
                     if a.get("action") == "llm_error")
    rate_limits = sum(1 for _, rec in calls for a in ((rec.get("outcome") or {}).get("actions") or [])
                      if a.get("action") == "llm_error" and "rate limit" in str(a.get("problem", "")))

    span = days if days > 0 else 14
    per_day = []
    for i in range(span - 1, -1, -1):
        day = (now - dt.timedelta(days=i)).date()
        day_calls = [rec for when, rec in calls if when.date() == day]
        per_day.append({"date": day.isoformat(), "calls": len(day_calls),
                        "qualified": sum(1 for r in day_calls if r in qualified)})

    def pct(a, b):
        return round(100 * a / b, 1) if b else None

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    return {
        "days": days, "total_calls": len(calls), "dialed": dialed, "answered": answered,
        "pickup_rate": pct(answered, dialed), "qualified": len(qualified),
        "qualification_rate": pct(len(qualified), len(conversations)),
        "avg_duration_secs": round(sum(durations) / len(durations)) if durations else None,
        "total_minutes": round(sum(durations) / 60, 1),
        "median_reply_ms": median(p50s), "median_first_audio_ms": median(firsts),
        "llm_errors": llm_errors, "rate_limit_errors": rate_limits,
        "server_llm_errors": LLM_ERROR_STATS["total"], "last_llm_error": LLM_ERROR_STATS["last"],
        "outcomes": dict(outcomes.most_common()), "per_day": per_day,
        "pickup_source": "Vobiz hang-up reports" if events else "call logs only (no hang-up reports yet)",
    }


# ---------- Live monitoring ----------
@router.get("/api/live")
async def api_live(request: Request):
    _guard(request)
    return {"calls": live.snapshot()}


@router.post("/api/live/{call_id}/hangup")
async def api_live_hangup(request: Request, call_id: str):
    _guard(request)
    c = live.LIVE_CALLS.get(call_id)
    if not c:
        raise HTTPException(404, "That call has already ended")
    if c.get("simulated"):
        raise HTTPException(400, "Load-test calls end by themselves")
    await vobiz.hang_up(call_id)
    return {"ok": True}


# ---------- Chat with the agent by text (Chat History source) ----------
def _chat(sid: str) -> dict:
    chat = _chats.get(sid)
    if not chat:
        raise HTTPException(404, "Chat session expired; start a new one")
    chat["t"] = time.time()
    return chat


@router.post("/api/chat/start")
async def api_chat_start(request: Request):
    _guard(request)
    from app.text_sim import TextCall

    body = await request.json()
    for sid in [s for s, c in _chats.items() if time.time() - c["t"] > 3600]:
        _chats.pop(sid, None)
    if len(_chats) >= 20:
        raise HTTPException(429, "Too many open chats; end some first")
    variables = body.get("variables") if isinstance(body.get("variables"), dict) else {}
    name = str(body.get("name") or variables.get("candidate_name") or "").strip()[:60]
    role = str(body.get("role") or variables.get("role") or variables.get("job_title") or "").strip()[:80]
    extra = {k: v for k, v in variables.items() if k not in ("candidate_name", "role", "job_title")}
    draft = body.get("draft") if isinstance(body.get("draft"), dict) else None
    prompt_template, greeting = None, None
    if draft:  # Agent Studio "Test LLM": try unpublished changes without saving them
        if draft.get("prompt"):
            prompt_template = str(draft["prompt"])[:30000]
        greeting = {k: draft[k] for k in ("with_name", "without_name", "mode") if draft.get(k)}
        greeting = dict(_yaml(SCREENING_FILE).get("greeting") or {}, **greeting)
    call = TextCall(candidate_name=name, job_title=role, prompt_template=prompt_template,
                    greeting_cfg=greeting, variables=extra)
    try:
        opening = await call.start()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Gemini: {str(e)[:200]}") from None
    sid = secrets.token_urlsafe(12)
    _chats[sid] = {"call": call, "t": time.time(), "started": time.time(),
                   "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "role": role}
    return {"session": sid, "reply": opening, "ended": call.ended, "user_first": call.user_first}


@router.post("/api/chat/{sid}/send")
async def api_chat_send(request: Request, sid: str):
    _guard(request)
    chat = _chat(sid)
    text = _clean_str((await request.json()).get("text"), "Message", 1000)
    try:
        reply = await chat["call"].reply(text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Gemini: {str(e)[:200]}") from None
    return {"reply": reply, "ended": chat["call"].ended, "end_reason": chat["call"].state.end_reason}


@router.post("/api/chat/{sid}/end")
async def api_chat_end(request: Request, sid: str):
    _guard(request)
    from app.outcome import extract_outcome, fallback_outcome, reconcile

    chat = _chats.pop(sid, None)
    if not chat:
        raise HTTPException(404, "Chat session expired")
    call = chat["call"]
    if call.state.ended_by is None:
        call.state.ended_by = "candidate"
    try:
        outcome = reconcile(await extract_outcome(call.transcript, agent_name=settings["agent"].get("name", "Priya")),
                            call.state)
    except Exception as e:  # noqa: BLE001
        outcome = fallback_outcome(str(e), call.state)
    CALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_web-chat.json"
    record = {"call_id": f"chat-{sid[:8]}", "channel": "web chat", "candidate_name": chat["name"], "phone": "",
              "role": chat["role"], "started_at": chat["started_at"], "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "duration_secs": round(time.time() - chat["started"]), "outcome": outcome,
              "transcript": call.transcript}
    (CALL_LOG_DIR / name).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "file": name, "outcome": outcome}


# ---------- Docs ----------
@router.get("/api/docs/{which}")
async def api_docs(request: Request, which: str):
    _guard(request)
    files = {"guide": "PROJECT_GUIDE.md", "readme": "README.md", "deploy": "deploy/DEPLOY.md"}
    if which not in files:
        raise HTTPException(404)
    path = Path(ROOT) / files[which]
    return {"name": files[which], "text": path.read_text(encoding="utf-8") if path.exists() else ""}
