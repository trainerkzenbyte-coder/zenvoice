"""Knowledge bases: text snippets, FAQs and crawled web pages Priya can use on calls.

    data/knowledge/index.json            list of knowledge bases (name, enabled)
    data/knowledge/<kb_id>/sources.json  their sources (text, FAQ, URL with crawled pages)

Every ENABLED knowledge base is added to the call script (see app.knowledge.prompt_section),
up to MAX_PROMPT_CHARS in total: it is sent to Gemini on every turn, so it costs tokens and
counts towards the free-tier limits. Short, specific facts work best.
"""

import html
import ipaddress
import json
import re
import secrets
import socket
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import yaml

from app.config import ROOT

KB_DIR = Path(ROOT) / "data" / "knowledge"
INDEX = KB_DIR / "index.json"
LEGACY_FILE = Path(ROOT) / "config" / "knowledge.yaml"
MAX_PROMPT_CHARS = 12000        # default; Agents > Knowledge Base can change it (knowledge.max_chars)
MAX_PAGE_CHARS = 300000          # stored per page
MAX_DOWNLOAD = 2_000_000        # bytes per page
SKIP_EXT = re.compile(r"\.(png|jpe?g|gif|svg|webp|ico|pdf|zip|mp4|mp3|css|js|woff2?|ttf|xml|json)$", re.I)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(5)}"


# ---------- storage ----------
def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _migrate_legacy() -> None:
    """Console v2 kept snippets in config/knowledge.yaml: move them into a knowledge base once."""
    if INDEX.exists() or not LEGACY_FILE.exists():
        return
    try:
        entries = (yaml.safe_load(LEGACY_FILE.read_text(encoding="utf-8")) or {}).get("entries") or []
    except yaml.YAMLError:
        entries = []
    kb = create_kb("Zenbyte notes")
    for e in entries:
        add_text(kb["id"], e.get("title", "Note"), e.get("content", ""), kind=e.get("kind", "snippet"),
                 enabled=e.get("enabled", True))


def list_kbs() -> list[dict]:
    _migrate_legacy()
    kbs = _read(INDEX, {"kbs": []})["kbs"]
    for kb in kbs:
        sources = get_sources(kb["id"])
        kb["source_count"] = len(sources)
        kb["chars"] = sum(len(_source_text(s)) for s in sources if s.get("enabled", True))
    return kbs


def get_kb(kb_id: str) -> dict | None:
    return next((k for k in list_kbs() if k["id"] == kb_id), None)


def _save_kbs(kbs: list[dict]) -> None:
    _write(INDEX, {"kbs": [{k: v for k, v in kb.items() if k not in ("source_count", "chars")} for kb in kbs]})


def create_kb(name: str) -> dict:
    kbs = _read(INDEX, {"kbs": []})["kbs"]
    kb = {"id": _new_id("kb"), "name": name.strip()[:80] or "Untitled", "created": _now(), "updated": _now(),
          "enabled": True}
    kbs.append(kb)
    _save_kbs(kbs)
    _write(KB_DIR / kb["id"] / "sources.json", [])
    return kb


def update_kb(kb_id: str, **changes) -> dict:
    kbs = _read(INDEX, {"kbs": []})["kbs"]
    for kb in kbs:
        if kb["id"] == kb_id:
            kb.update({k: v for k, v in changes.items() if k in ("name", "enabled")}, updated=_now())
            _save_kbs(kbs)
            return kb
    raise KeyError(kb_id)


def delete_kb(kb_id: str) -> None:
    kbs = [k for k in _read(INDEX, {"kbs": []})["kbs"] if k["id"] != kb_id]
    _save_kbs(kbs)
    folder = KB_DIR / kb_id
    if folder.exists() and folder.parent == KB_DIR:
        for f in folder.glob("*"):
            f.unlink()
        folder.rmdir()


def get_sources(kb_id: str) -> list[dict]:
    if not re.fullmatch(r"kb_[0-9a-f]{10}", kb_id or ""):
        return []
    return _read(KB_DIR / kb_id / "sources.json", [])


def _save_sources(kb_id: str, sources: list[dict]) -> None:
    _write(KB_DIR / kb_id / "sources.json", sources)
    try:
        update_kb(kb_id)
    except KeyError:
        pass


def add_text(kb_id: str, title: str, content: str, kind: str = "snippet", enabled: bool = True) -> dict:
    src = {"id": _new_id("src"), "type": "faq" if kind == "faq" else "text", "title": title.strip()[:200],
           "text": content.strip()[:MAX_PAGE_CHARS], "added": _now(), "enabled": bool(enabled)}
    sources = get_sources(kb_id)
    sources.append(src)
    _save_sources(kb_id, sources)
    return src


def update_source(kb_id: str, src_id: str, **changes) -> dict:
    sources = get_sources(kb_id)
    for s in sources:
        if s["id"] == src_id:
            for k in ("title", "text", "enabled"):
                if k in changes:
                    s[k] = changes[k] if k == "enabled" else str(changes[k])[:MAX_PAGE_CHARS]
            _save_sources(kb_id, sources)
            return s
    raise KeyError(src_id)


def delete_source(kb_id: str, src_id: str) -> None:
    _save_sources(kb_id, [s for s in get_sources(kb_id) if s["id"] != src_id])


# ---------- web crawl ----------
class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe", "template"}
    BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "tr", "td", "th"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = html.unescape("".join(self.parts))
        lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in raw.split("\n")]
        out, blank = [], False
        for ln in lines:
            if ln:
                out.append(ln)
                blank = False
            elif not blank and out:
                out.append("")
                blank = True
        return "\n".join(out).strip()


def _public_host(host: str) -> bool:
    """Only crawl the public internet (never this computer or the office network)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return True


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        p = urlparse(newurl)
        if p.scheme not in ("http", "https") or not p.hostname or not _public_host(p.hostname):
            raise urllib.error.URLError(f"redirect to a non-public address blocked: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SafeRedirect())


def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("Enter a web address like www.zenbytech.org")
    return urldefrag(url)[0]


GOOGLE_DOC = re.compile(r"^https://docs\.google\.com/(spreadsheets|document)/d/([\w-]{20,})")


def google_export_url(url: str) -> str | None:
    """A shared Google Sheet / Doc link -> its plain-text export (sheet: CSV of that tab, doc: text).
    The file must be shared as 'Anyone with the link can view'."""
    m = GOOGLE_DOC.match(url)
    if not m:
        return None
    kind, doc_id = m.groups()
    if kind == "document":
        return f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    gid = re.search(r"[#&?]gid=(\d+)", url)
    return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv" + (f"&gid={gid.group(1)}" if gid else "")


def fetch_page(url: str, timeout: float = 15) -> tuple[str, str, list[str]]:
    """(title, text, links) of one page."""
    export = google_export_url(url)
    if export:
        return _fetch_google(url, export, timeout)
    p = urlparse(url)
    if not _public_host(p.hostname or ""):
        raise ValueError(f"{p.hostname} is not a public website address")
    req = urllib.request.Request(url, headers={"User-Agent": "ZenbyteVoiceKB/1.0 (+knowledge base sync)",
                                               "Accept": "text/html,text/plain;q=0.9"})
    with _OPENER.open(req, timeout=timeout) as r:
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "text/plain" not in ctype:
            raise ValueError(f"not a web page ({ctype or 'unknown type'})")
        body = r.read(MAX_DOWNLOAD + 1)[:MAX_DOWNLOAD]
        charset = r.headers.get_content_charset() or "utf-8"
        final = r.geturl()
    text = body.decode(charset, errors="replace")
    if "text/plain" in ctype:
        return final, text[:MAX_PAGE_CHARS], []
    parser = _TextExtractor()
    parser.feed(text)
    links = [urldefrag(urljoin(final, h))[0] for h in parser.links]
    return (parser.title.strip() or final)[:200], parser.text()[:MAX_PAGE_CHARS], links


def _fetch_google(url: str, export: str, timeout: float) -> tuple[str, str, list[str]]:
    req = urllib.request.Request(export, headers={"User-Agent": "ZenbyteVoiceKB/1.0 (+knowledge base sync)"})
    with _OPENER.open(req, timeout=timeout) as r:
        ctype = r.headers.get("content-type", "")
        body = r.read(MAX_DOWNLOAD + 1)[:MAX_DOWNLOAD]
    if "html" in ctype:
        raise ValueError("Google asked for a sign-in: share the file as 'Anyone with the link can view'")
    text = body.decode("utf-8-sig", errors="replace")
    if "/spreadsheets/" in url:  # CSV rows -> "Header: value; Header: value" lines (easier for the AI)
        import csv
        import io

        rows = list(csv.reader(io.StringIO(text)))
        if rows:
            head = [h.strip() for h in rows[0]]
            lines = ["; ".join(f"{h}: {v.strip()}" for h, v in zip(head, row) if h and v.strip()) for row in rows[1:]]
            text = "\n".join(line for line in lines if line)
    title = "Google Sheet" if "/spreadsheets/" in url else "Google Doc"
    return title, text[:MAX_PAGE_CHARS], []


def crawl(url: str, max_pages: int = 1) -> list[dict]:
    """Start page plus up to max_pages-1 more pages on the same site."""
    start = normalize_url(url)
    host = urlparse(start).hostname
    queue, seen, pages, errors = [start], {start}, [], []
    while queue and len(pages) < max_pages:
        page_url = queue.pop(0)
        try:
            title, text, links = fetch_page(page_url)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{page_url}: {str(e)[:120]}")
            if not pages and len(errors) == 1 and page_url == start:
                raise ValueError(f"Could not read {page_url}: {str(e)[:160]}") from None
            continue
        if text.strip():
            pages.append({"url": page_url, "title": title, "text": text, "chars": len(text)})
        for link in links:
            lp = urlparse(link)
            if lp.scheme in ("http", "https") and lp.hostname == host and link not in seen and not SKIP_EXT.search(lp.path):
                seen.add(link)
                queue.append(link)
    if not pages:
        raise ValueError("No readable text found on that page" + (f" ({errors[0]})" if errors else ""))
    return pages


def add_url(kb_id: str, url: str, max_pages: int = 1) -> dict:
    max_pages = max(1, min(int(max_pages), 10))
    pages = crawl(url, max_pages)
    host = urlparse(pages[0]["url"]).hostname
    src = {"id": _new_id("src"), "type": "url", "url": normalize_url(url), "title": f"{host}",
           "max_pages": max_pages, "pages": pages, "added": _now(), "synced": _now(), "enabled": True}
    sources = get_sources(kb_id)
    sources.append(src)
    _save_sources(kb_id, sources)
    return src


def sync_kb(kb_id: str) -> dict:
    """Crawl every URL source again. Returns {'synced': n, 'errors': [...]}."""
    sources = get_sources(kb_id)
    synced, errors = 0, []
    for s in sources:
        if s["type"] != "url":
            continue
        try:
            s["pages"] = crawl(s["url"], s.get("max_pages", 1))
            s["synced"] = _now()
            synced += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{s['url']}: {str(e)[:160]}")
    _save_sources(kb_id, sources)
    return {"synced": synced, "errors": errors}


# ---------- what Priya gets ----------
def _source_text(s: dict) -> str:
    if s["type"] == "url":
        return "\n\n".join(p["text"] for p in s.get("pages", []))
    if s["type"] == "faq":
        return f"Q: {s.get('title', '')}\nA: {s.get('text', '')}"
    return s.get("text", "")


DROP_DIRS = [Path(ROOT) / "data" / "knowledge_files", Path(ROOT) / "config" / "kb"]  # files copied here are read too
DROP_EXT = {".txt", ".md", ".csv", ".pdf"}


def prompt_limit() -> int:
    from app.config import settings

    try:
        return max(2000, min(40000, int((settings.get("knowledge") or {}).get("max_chars", MAX_PROMPT_CHARS))))
    except (TypeError, ValueError):
        return MAX_PROMPT_CHARS


def _drop_files() -> list[dict]:
    """.txt/.md/.csv/.pdf files placed in data/knowledge_files/ (or config/kb/) count as text sources."""
    out = []
    for folder in DROP_DIRS:
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in DROP_EXT or not path.is_file():
                continue
            try:
                if path.suffix.lower() == ".pdf":
                    from pypdf import PdfReader

                    text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages[:200])
                else:
                    text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001  (pypdf missing, unreadable file)
                continue
            if text.strip():
                out.append({"type": "text", "title": f"{path.name}", "text": text[:MAX_PAGE_CHARS], "enabled": True})
    return out


def compile_sources() -> list[dict]:
    """Every enabled source of every enabled knowledge base, most precise first:
    FAQs, then documents/text, then web pages (across ALL knowledge bases)."""
    items = []
    for kb in list_kbs():
        if not kb.get("enabled", True):
            continue
        for s in get_sources(kb["id"]):
            if s.get("enabled", True):
                items.append(s)
    items += _drop_files()
    order = {"faq": 0, "text": 1, "url": 2}
    items.sort(key=lambda s: order.get(s["type"], 3))
    return items


def prompt_text(limit: int | None = None) -> tuple[str, int, int]:
    """(text for the prompt, characters used, characters available).

    The budget is shared fairly: FAQs always fit first (they're short), then the rest of the
    space is split between the other sources, so one long PDF or website can't push
    everything else out. Each source that doesn't fit completely is cut at a paragraph."""
    limit = limit or prompt_limit()
    sources = [(s, _source_text(s).strip()) for s in compile_sources()]
    sources = [(s, body) for s, body in sources if body]
    total = sum(len(b) for _, b in sources)
    blocks, used = [], 0
    faqs = [(s, b) for s, b in sources if s["type"] == "faq"]
    rest = [(s, b) for s, b in sources if s["type"] != "faq"]
    if faqs:
        lines = "\n\n".join(b for _, b in faqs)
        blocks.append("### Frequently asked questions\n" + lines[: limit])
        used += len(blocks[-1])
    remaining = rest[:]
    while remaining and used < limit:
        share = max(400, (limit - used) // len(remaining))
        s, body = remaining.pop(0)
        label = s.get("url") if s["type"] == "url" else s.get("title", "Note")
        piece = body if len(body) <= share else _cut(body, share)
        if not piece:
            continue
        block = f"### {label}\n{piece}"
        blocks.append(block)
        used += len(block) + 2
    full = "\n\n".join(blocks)
    return full[:limit], min(len(full), limit), total


def _cut(text: str, n: int) -> str:
    cut = text[:n]
    para = cut.rfind("\n")
    return (cut[:para] if para > n * 0.6 else cut).rstrip() + " …"


def report() -> dict:
    """What the AI actually gets: shown in the console and logged at the start of each call."""
    body, used, total = prompt_text()
    srcs = compile_sources()
    return {"sources": len(srcs), "faqs": sum(1 for s in srcs if s["type"] == "faq"), "used_chars": used,
            "source_chars": total, "max_chars": prompt_limit(), "truncated": total > used,
            "files_folder": str(DROP_DIRS[0])}
