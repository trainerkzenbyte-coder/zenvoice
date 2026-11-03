"""Knowledge base search (RAG) so the WHOLE knowledge base is available without sending it every turn.

How it works on a call (knowledge.mode: "retrieve" in config/agent.yaml):

  - FAQs stay pinned in the system prompt (short, exact answers, always there).
  - Every document, PDF, Google Doc/Sheet and web page is cut into small passages ("chunks").
  - Each time the candidate speaks, the passages that best match what they said (and what they said
    just before) are looked up here (a few milliseconds, no API, no cost) and handed to the AI as
    "KNOWLEDGE FOR THIS TURN". The note is replaced every turn, so the conversation history stays small.
  - The AI can also call search_knowledge(query) itself for anything the automatic lookup missed.

Search is BM25 keyword ranking (the classic search-engine formula) with light word stemming and
fuzzy matching for speech-recognition slips, so "placemant", "placements" and "placement" match.

knowledge.mode: "full" keeps the old behaviour (whole knowledge base pinned in the prompt).
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass

from app.config import settings

DEFAULTS = {
    "mode": "retrieve",          # "retrieve" (search per turn) | "full" (whole KB in the prompt)
    "pinned_max_chars": 3000,    # FAQs pinned in the system prompt (retrieve mode)
    "chunk_chars": 700,          # size of one passage
    "top_k": 4,                  # passages added per turn
    "max_turn_chars": 2800,      # cap on the per-turn knowledge note
    "tool": True,                # also give the AI a search_knowledge tool
}

STOP = set("""a an and are as at be been but by can could did do does for from had has have how i if in into is it
its me my of on or our so than that the their them then there these they this to up us was we were what when where
which who why will with would you your yours yeah yes ok okay hmm hello hi please tell know about like just also
sir madam mam ma'am right sure one""".split())


def cfg() -> dict:
    out = {**DEFAULTS, **(settings.get("knowledge") or {})}
    out["mode"] = "full" if str(out.get("mode", "retrieve")).lower() == "full" else "retrieve"
    return out


def retrieve_mode() -> bool:
    return cfg()["mode"] == "retrieve"


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:[.+#][a-z0-9]+)*", str(text).lower())
    return [_stem(w) for w in words if w not in STOP and (len(w) > 1 or w.isdigit())]


@dataclass
class Chunk:
    title: str
    text: str
    kind: str          # faq | text | url
    toks: list[str]


def _paragraphs(text: str) -> list[str]:
    """Paragraphs, with a heading line (short, no full stop) kept together with the text under it."""
    raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    carry = ""
    for p in raw:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        # a block that is only a heading: attach it to the next block
        if len(lines) == 1 and len(lines[0]) < 80 and not lines[0].endswith((".", "?", "!", ":")) and len(raw) > 1:
            carry = (carry + " " + lines[0]).strip()
            continue
        out.append((carry + "\n" + p).strip() if carry else p)
        carry = ""
    if carry:
        out.append(carry)
    return out


def _sentences(p: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", p) if x.strip()]


def repair_text(text: str) -> str:
    """Fix text pulled out of PDFs made in design tools, which often comes out one or two words per
    line ("help\nyou\nEnd"). Lines are joined back into sentences; blank lines stay paragraph breaks."""
    lines = str(text).split("\n")
    words = [ln for ln in lines if ln.strip()]
    if len(words) < 8:
        return text
    short = sum(1 for ln in words if len(ln.split()) <= 3)
    if short / len(words) < 0.4:
        return text  # normal text: leave it alone
    paras, cur = [], []
    for ln in lines:
        if ln.strip():
            cur.append(ln.strip())
        elif cur:
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    # one-word "paragraphs" separated by blank lines: glue them too, breaking at sentence ends
    joined = " ".join(paras) if sum(1 for p in paras if len(p.split()) <= 3) / max(1, len(paras)) > 0.4 else "\n\n".join(paras)
    joined = re.sub(r"\s+([.,;:!?])", r"\1", joined)
    joined = re.sub(r"(?<=[.!?])\s+(?=(\d+[.)]|[A-Z][a-z]+ [a-z]))", "\n\n", joined)  # paragraph per sentence group
    return joined


def _split(text: str, size: int) -> list[str]:
    """Passages of about `size` characters. One topic (heading + its text) per passage where it fits;
    long sections are cut at sentence ends; each passage repeats the last sentence of the one before."""
    text = re.sub(r"[ \t]+", " ", repair_text(str(text))).strip()
    if not text:
        return []
    chunks: list[str] = []
    for para in _paragraphs(text):
        if len(para) <= size:
            chunks.append(para)
            continue
        head = para.split("\n", 1)[0] if "\n" in para and len(para.split("\n", 1)[0]) < 80 else ""
        cur = ""
        for sent in _sentences(para):
            sent = sent[:size]
            if cur and len(cur) + len(sent) + 1 > size:
                chunks.append(cur)
                last = _sentences(cur)[-1] if _sentences(cur) else ""
                cur = ((head + "\n") if head and not last.startswith(head) else "") + (last if len(last) < size // 2 else "")
            cur = (cur + " " + sent).strip()
        if cur:
            chunks.append(cur)
    # merge tiny neighbours (one-line notes) so passages aren't too small to be useful
    merged: list[str] = []
    for c in chunks:
        if merged and len(merged[-1]) < size // 4 and len(merged[-1]) + len(c) < size:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    return merged


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.n = len(chunks)
        self.avg = (sum(len(c.toks) for c in chunks) / self.n) if self.n else 1.0
        self.df: dict[str, int] = {}
        for c in chunks:
            for t in set(c.toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.vocab = set(self.df)

    def _expand(self, q: list[str]) -> dict[str, float]:
        """Query words, plus close spellings found in the knowledge base (speech-recognition slips)."""
        import difflib

        out: dict[str, float] = {}
        for t in q:
            if t in self.vocab:
                out[t] = max(out.get(t, 0), 1.0)
            elif len(t) >= 5 and not t.isdigit():
                for m in difflib.get_close_matches(t, self.vocab, n=2, cutoff=0.82):
                    out[m] = max(out.get(m, 0), 0.7)
        return out

    def search(self, query: str, k: int = 4) -> list[tuple[float, Chunk]]:
        q = self._expand(tokens(query))
        if not q or not self.n:
            return []
        k1, b = 1.4, 0.75
        scored = []
        for c in self.chunks:
            if not c.toks:
                continue
            tf: dict[str, int] = {}
            for t in c.toks:
                if t in q:
                    tf[t] = tf.get(t, 0) + 1
            if not tf:
                continue
            s = 0.0
            for t, f in tf.items():
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += q[t] * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(c.toks) / self.avg))
            title_hits = sum(1 for t in set(tokens(c.title)) if t in q)
            s *= 1 + 0.15 * title_hits
            scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


_cache: dict = {"key": "", "index": None, "t": 0.0}


def _sources() -> list[dict]:
    from app import kb_store

    return kb_store.compile_sources()


def index() -> Index:
    """Built from every enabled source; rebuilt only when the knowledge base changes (checked every 30 s)."""
    if _cache["index"] is not None and time.time() - _cache["t"] < 30:
        return _cache["index"]
    from app import kb_store

    srcs = _sources()
    texts = [(s, kb_store._source_text(s)) for s in srcs]
    key = hashlib.sha1("".join(f"{s.get('title')}|{s.get('url')}|{len(t)}|{t[:200]}|{t[-200:]}"
                               for s, t in texts).encode()).hexdigest()
    if key != _cache["key"] or _cache["index"] is None:
        size = int(cfg()["chunk_chars"])
        chunks: list[Chunk] = []
        for s, body in texts:
            title = str(s.get("title") or s.get("url") or "Note")
            if s.get("type") == "faq":
                chunks.append(Chunk(title, body, "faq", tokens(body)))
                continue
            if s.get("type") == "url":
                for p in s.get("pages") or []:
                    for piece in _split(p.get("text", ""), size):
                        chunks.append(Chunk(p.get("url") or title, piece, "url", tokens(piece + " " + title)))
                continue
            for piece in _split(body, size):
                chunks.append(Chunk(title, piece, s.get("type", "text"), tokens(piece + " " + title)))
        _cache.update(key=key, index=Index(chunks))
    _cache["t"] = time.time()
    return _cache["index"]


def search(query: str, k: int | None = None, *, include_faq: bool = True) -> list[tuple[float, Chunk]]:
    try:
        hits = index().search(query, k=(k or int(cfg()["top_k"])) * 2)
    except Exception:  # noqa: BLE001  a broken knowledge file must never stop a call
        return []
    if not include_faq:
        hits = [h for h in hits if h[1].kind != "faq"]
    if not hits:
        return []
    best = hits[0][0]
    hits = [h for h in hits if h[0] >= best * 0.35]  # drop weak matches
    return hits[: (k or int(cfg()["top_k"]))]


def format_hits(hits: list[tuple[float, Chunk]], limit: int | None = None) -> str:
    limit = limit or int(cfg()["max_turn_chars"])
    out, used = [], 0
    for _, c in hits:
        block = f"[{c.title}]\n{c.text.strip()}"
        if used + len(block) > limit:
            block = block[: max(0, limit - used)].rsplit(" ", 1)[0] + " …"
        if len(block) > 40:
            out.append(block)
            used += len(block) + 2
        if used >= limit:
            break
    return "\n\n".join(out)


def pinned_text() -> str:
    """Retrieve mode: what stays in the system prompt (FAQs only, within pinned_max_chars)."""
    from app import kb_store

    faqs = [kb_store._source_text(s).strip() for s in _sources() if s.get("type") == "faq"]
    body = "\n\n".join(f for f in faqs if f)
    limit = int(cfg()["pinned_max_chars"])
    return body if len(body) <= limit else body[:limit].rsplit("\n", 1)[0] + "\n…"


def report() -> dict:
    ix = index()
    return {"mode": cfg()["mode"], "chunks": ix.n, "faq_chunks": sum(1 for c in ix.chunks if c.kind == "faq"),
            "chars": sum(len(c.text) for c in ix.chunks), "pinned_chars": len(pinned_text())}


# ---------- live calls: add the matching passages to the conversation each turn ----------
NOTE_MARK = "KNOWLEDGE FOR THIS TURN"


def _text(m) -> str:
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return ""


def turn_query(messages: list) -> str:
    """What the candidate just said, plus what they said before and what the agent last asked,
    so follow-ups like 'how much is it?' still find the right passage."""
    users = [_text(m) for m in messages if isinstance(m, dict) and m.get("role") == "user"
             and not m.get("tool_call_id") and not _text(m).startswith(("(", NOTE_MARK))]
    bots = [_text(m) for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and _text(m)]
    last = users[-1] if users else ""
    prev = users[-2] if len(users) > 1 else ""
    asked = bots[-1] if bots else ""
    # The newest words count most: repeat them.
    return " ".join([last, last, prev, asked[-200:]])


def note_for(messages: list) -> str | None:
    hits = search(turn_query(messages), include_faq=False)  # FAQs are pinned already
    body = format_hits(hits)
    if not body:
        return None
    return (f"{NOTE_MARK} (looked up from the knowledge base for what the candidate just said; the candidate "
            "can't see this. Use it only if it answers them, in one or two short spoken sentences, in your own "
            "words. If it doesn't cover their question, use search_knowledge or say the team will confirm.)\n\n"
            + body)


def inject(context, role: str = "system") -> str | None:
    """Replace last turn's knowledge note with one for the newest candidate message. Returns the note."""
    msgs = list(context.get_messages())
    msgs = [m for m in msgs if not _text(m).startswith(NOTE_MARK)]
    if not msgs or not isinstance(msgs[-1], dict) or msgs[-1].get("role") != "user" or msgs[-1].get("tool_call_id"):
        context.set_messages(msgs)
        return None
    note = note_for(msgs)
    if note:
        msgs.insert(len(msgs) - 1, {"role": role, "content": note})
    context.set_messages(msgs)
    return note


def make_retriever(label: str = ""):
    from loguru import logger
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    from app.llm_choice import llm_provider

    # OpenAI-style APIs (Cerebras, Groq, OpenAI) take a system note mid-conversation; Gemini gets it as user text.
    role = "user" if llm_provider(settings.get("llm") or {})[0] == "google" else "system"

    class KnowledgeRetriever(FrameProcessor):
        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
                try:
                    t0 = time.perf_counter()
                    note = inject(frame.context, role=role)
                    if note:
                        titles = sorted(set(re.findall(r"^\[(.+?)\]$", note, re.M)))
                        logger.info(f"[{label}]   [knowledge] {len(note):,} chars from {', '.join(titles)[:120]} "
                                    f"({(time.perf_counter() - t0) * 1000:.0f} ms)")
                except Exception as e:  # noqa: BLE001  never block a reply because of the lookup
                    logger.warning(f"[{label}] Knowledge lookup failed: {e}")
            await self.push_frame(frame, direction)

    return KnowledgeRetriever()
