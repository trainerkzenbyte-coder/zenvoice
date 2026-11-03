"""Which LLM answers on live calls (no Pipecat imports, so the console can use it too)."""

FAST_LLMS = {
    "groq": ("GROQ_API_KEY", "pipecat.services.groq.llm", "GroqLLMService"),
    "cerebras": ("CEREBRAS_API_KEY", "pipecat.services.cerebras.llm", "CerebrasLLMService"),
    "openai": ("OPENAI_API_KEY", "pipecat.services.openai.llm", "OpenAILLMService"),
}
REASONING_MODELS = ("gpt-oss", "qwen3", "gpt-5", "o3", "o4")

# Google Gemini model ID mapping
GEMINI_MODELS = {
    "gemini-3.6-flash": "gemini-3.6-flash",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3.8-flash": "gemini-3.8-flash",
}


def llm_provider(cfg: dict | None = None) -> tuple[str, str]:
    from app.config import settings

    cfg = cfg if cfg is not None else (settings.get("llm", {}) or {})
    raw_model = str(cfg.get("model", "gemini-3.6-flash")).strip()
    provider = str(cfg.get("provider", "google")).strip().lower()

    if ":" in raw_model:
        prefix, model_part = raw_model.split(":", 1)
        if prefix in FAST_LLMS or prefix == "anthropic":
            return prefix, model_part

    if provider == "anthropic" or raw_model.lower().startswith("claude"):
        return "anthropic", raw_model  # Claude (needs ANTHROPIC_API_KEY in .env)

    if provider in FAST_LLMS:
        return provider, raw_model

    model_id = GEMINI_MODELS.get(raw_model.lower(), raw_model)
    return "google", model_id