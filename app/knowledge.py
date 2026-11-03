"""Knowledge base text for the call script (Knowledge Base pages of the console).

Compiled by app.kb_store.prompt_text(): every enabled FAQ, document, Google Sheet/Doc, web page and
file in data/knowledge_files/, fairly shared within knowledge.max_chars (default 12,000 characters).
It is pinned in the system prompt for the whole call: no lookup during the call, so no extra delay.
"""

from app import kb_store

MAX_TOTAL_CHARS = kb_store.MAX_PROMPT_CHARS


def prompt_section() -> str:
    """Text appended to the system prompt ('' when nothing is enabled).

    knowledge.mode "retrieve" (default): only FAQs are pinned here; the rest of the knowledge base is
    searched every turn (app/kb_search.py). knowledge.mode "full": everything pinned, as before."""
    from app import kb_search

    if kb_search.retrieve_mode():
        try:
            faqs = kb_search.pinned_text()
            has_docs = kb_search.index().n > 0
        except Exception:  # noqa: BLE001
            return ""
        if not faqs and not has_docs:
            return ""
        out = ("\n\n# COMPANY & ROLE FACTS (your knowledge base: the ONLY source for facts about the company, "
               "the training, fees, process, locations and policies)\n"
               "Answer in one or two short spoken sentences, in your own words. Never read it out like a document.\n")
        if faqs:
            out += "\n## Frequently asked questions\n" + faqs + "\n"
        if has_docs:
            out += ("\n## Detailed documents\nThe full documents are searched for you on every turn: matching "
                    "passages appear as a KNOWLEDGE FOR THIS TURN note just before the candidate's words. If the "
                    "candidate asks something specific that neither the FAQs nor that note answer, call "
                    "search_knowledge with a short query before saying you don't know. Never guess facts.\n")
        return out + "# END OF FACTS"
    try:
        body, _, _ = kb_store.prompt_text()
    except Exception:  # noqa: BLE001  (a broken knowledge file must never stop a call)
        return ""
    if not body.strip():
        return ""
    return ("\n\n# COMPANY & ROLE FACTS (your knowledge base: the ONLY source for facts about the company, "
            "the role, fees, process, locations and policies)\n"
            "When the candidate asks something, look here first and answer in one or two short spoken sentences, "
            "in your own words. Never read it out like a document.\n\n" + body + "\n# END OF FACTS")
