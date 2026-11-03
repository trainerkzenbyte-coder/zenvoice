"""Builds the speech, LLM, voice and turn-taking pieces from config/agent.yaml.

Shared by the laptop bot (Sprint 1) and the phone bot (Sprint 2), so latency
tuning done here applies to both.
"""

import inspect

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.turns.user_start import (
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from app import voice_config
from app.config import secret, settings
from app.prompts import load_prompt  # noqa: F401  (re-exported for older imports)


def turn_cfg() -> dict:
    return settings.get("turn", {}) or {}


# ---------- Deepgram region (Sprint 5) ----------
from app.voice_config import DEEPGRAM_HOSTS, deepgram_host  # noqa: E402,F401  (India endpoint by default)


def _url_kwarg(cls, url: str) -> dict:
    """Pass a custom address under whichever name this Pipecat version uses."""
    

    params = inspect.signature(cls.__init__).parameters
    for name in ("base_url", "url"):
        if name in params:
            return {name: url}
    return {}


# ---------- Speech-to-text ----------
def build_stt():
    """English calls: Deepgram Nova-3. Indian-language and auto calls: Sarvam Saaras v4.
    See app/voice_config.py for the routing table."""
    cfg = settings.get("stt", {})
    p = voice_config.plan(settings)
    if p["stt"] == "sarvam":
        return _sarvam_stt(p, cfg)
    if uses_flux():
        return _flux_stt(cfg)
    return DeepgramSTTService(
        api_key=secret("DEEPGRAM_API_KEY"),
        **_url_kwarg(DeepgramSTTService, deepgram_host()),
        settings=DeepgramSTTService.Settings(
            model=cfg.get("model", "nova-3"),
            language=cfg.get("language", "en-IN"),  # Indian English: best accuracy for English calls
            smart_format=True,
            punctuate=True,
            **_keyterms(cfg),
        ),
    )


def _sarvam_stt(p: dict, cfg: dict):
    try:
        from pipecat.services.sarvam.stt import SarvamRealtimeSTTService
    except Exception as e:  # noqa: BLE001  ModuleNotFoundError: sarvamai
        raise RuntimeError('Sarvam is not installed: pip install "pipecat-ai[sarvam]==1.12.0"') from e
    # Endpointing (stt.sarvam_endpointing):
    #   "vad" (default): Sarvam's own speech detection decides when the candidate starts and stops,
    #       keeps ~300 ms of audio from BEFORE speech starts, and transcribes whole utterances.
    #       Most accurate on phone lines.
    #   "manual": our Silero + smart-turn decide; Sarvam only hears audio between our start/stop
    #       signals, so the first word can be clipped and short pauses split sentences.
    # codemix: Tamil/Hindi words in their script, English words (Java, CTC, notice period) in English.
    prompt = ", ".join(str(t).strip() for t in (cfg.get("keyterms") or []) if str(t).strip())[:400]
    endpointing = sarvam_endpointing()
    extra: dict = {}
    vad_settings: dict = {}
    if endpointing == "vad":
        extra["prefix_padding_ms"] = int(cfg.get("sarvam_prefix_ms", 300))
        vad_settings["silence_duration_ms"] = int(cfg.get("sarvam_silence_ms", 700))
        if cfg.get("sarvam_vad_threshold") is not None:  # higher = needs clearer speech (ignores noise)
            vad_settings["threshold"] = float(cfg["sarvam_vad_threshold"])
        if cfg.get("sarvam_min_speech_ms") is not None:   # shorter sounds than this are ignored
            vad_settings["min_speech_duration_ms"] = int(cfg["sarvam_min_speech_ms"])
        interrupt_words = int(turn_cfg().get("interrupt_min_words", 1))
        extra["should_interrupt"] = True  # candidate talking over the agent stops it
        if interrupt_words > 1:
            from loguru import logger

            logger.debug("Sarvam endpointing: interrupt_min_words is not used (Sarvam decides turns)")
    return SarvamRealtimeSTTService(
        api_key=secret("SARVAM_API_KEY"),
        endpointing=endpointing,
        **extra,
        settings=SarvamRealtimeSTTService.Settings(
            model=cfg.get("sarvam_model", "saaras:v4"),
            language_code=p["stt_language"],
            mode=cfg.get("sarvam_mode", "codemix"),
            stream_type=cfg.get("sarvam_stream_type", "balanced"),
            **vad_settings,
            **({"prompt": f"Phone call about IT training and interviews. Words that may come up: {prompt}"}
               if prompt else {}),
        ),
    )


def sarvam_endpointing() -> str:
    cfg = settings.get("stt", {}) or {}
    return "manual" if str(cfg.get("sarvam_endpointing", "vad")).lower() == "manual" else "vad"


def sarvam_decides_turns() -> bool:
    """True when Sarvam speech-to-text is used with its own speech detection: then Sarvam decides
    when the candidate has finished, and our smart-turn strategies must not be installed."""
    return voice_config.plan(settings)["stt"] == "sarvam" and sarvam_endpointing() == "vad"


def uses_flux() -> bool:
    """stt.provider: deepgram_flux = Deepgram Flux decides when the candidate has finished
    (built for voice agents: replies start sooner). English calls only."""
    return (settings.get("stt", {}) or {}).get("provider") == "deepgram_flux" and voice_config.call_language(settings) == "en-IN"


def _flux_stt(cfg: dict):
    from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

    kw = {}
    if cfg.get("flux_region") == "in":  # only if Deepgram's India endpoint serves Flux for your account
        kw["url"] = f"wss://{deepgram_host()}/v2/listen"
    return DeepgramFluxSTTService(
        api_key=secret("DEEPGRAM_API_KEY"),
        enable_eager_end_of_turn=bool(cfg.get("flux_eager", True)),  # start the reply on Flux's early guess
        settings=DeepgramFluxSTTService.Settings(
            model=cfg.get("flux_model", "flux-general-en"),
            eot_threshold=float(cfg.get("flux_eot_threshold", 0.7)),
            eot_timeout_ms=int(cfg.get("flux_eot_timeout_ms", 3000)),
            **_keyterms(cfg),
        ),
        **kw,
    )


# Used when agent.yaml has no stt.keyterms. Deepgram Nova-3 listens harder for these words.
DEFAULT_KEYTERMS = ["Zenbyte", "MERN stack", "MEAN stack", "full stack", "Java", "Spring Boot", "Python",
                    "JavaScript", "React", "Node.js", "MongoDB", "Angular", "DevOps", "AWS", "CTC", "LPA",
                    "notice period", "fresher", "mock interview", "Naukri", "Chennai", "Coimbatore",
                    "Bangalore", "Hyderabad"]


def _keyterms(cfg: dict) -> dict:
    """Boosted keywords (Agents > Realtime transcription): names, skills, places Deepgram should
    hear correctly, e.g. Zenbyte, Velachery, Spring Boot. Nova-3 'keyterm prompting'."""
    terms = [str(t).strip() for t in (cfg.get("keyterms") or DEFAULT_KEYTERMS) if str(t).strip()][:50]
    return {"keyterm": terms} if terms else {}


# ---------- LLM ----------
def _thinking(cfg: dict) -> dict:
    """Gemini 2.5 models: thinking_budget 0 turns thinking OFF (fastest).
    Gemini 3.x models cannot turn it off; thinking_level 'minimal' is the lowest."""
    config_cls = getattr(GoogleLLMService, "GoogleThinkingConfig", None) or getattr(
        GoogleLLMService, "ThinkingConfig", None
    )
    if config_cls is None:
        return {}
    if cfg.get("thinking_budget") is not None:
        return {"thinking": config_cls(thinking_budget=int(cfg["thinking_budget"]))}
    level = str(cfg.get("thinking_level", "minimal")).lower()
    if level == "none":
        return {}
    model = str(cfg.get("model", ""))
    if level == "minimal" and any(m in model for m in ("gemini-3.8", "-pro")):
        level = "low"  # these models reject "minimal"; "low" is their fastest setting
    return {"thinking": config_cls(thinking_level=level)}


def _resilience(cfg: dict) -> dict:
    """A caller is waiting, so retry fast and give up fast:
    - rate limit (429) / overload (5xx): up to 2 quick retries (~0.5 s, ~1 s apart)
    - no first word within 4 s: send the request again once
    - stream stalls for 6 s: give up (Priya then asks the candidate to repeat,
      see app.call_actions.attach_llm_error_handler)"""
    kwargs: dict = {}
    params = inspect.signature(GoogleLLMService.__init__).parameters
    attempts = int(cfg.get("retry_attempts", 3))
    if attempts > 1 and "http_options" in params:
        try:
            from google.genai.types import HttpRetryOptions  # noqa: F401  (needs google-genai >= 1.2x)

            kwargs["http_options"] = {"retry_options": {
                "attempts": attempts, "initial_delay": 0.5, "max_delay": 2.0, "exp_base": 2.0,
                "jitter": 0.3, "http_status_codes": [429, 500, 502, 503, 504],
            }}
        except ImportError:  # older google-genai without retry options
            pass
    if "retry_on_timeout" in params:
        kwargs["retry_on_timeout"] = True
        kwargs["retry_timeout_secs"] = float(cfg.get("first_word_timeout_secs", 4.0))
    if "stream_idle_timeout_secs" in params:
        kwargs["stream_idle_timeout_secs"] = float(cfg.get("stall_timeout_secs", 6.0))
    return kwargs


from app.llm_choice import FAST_LLMS, REASONING_MODELS, llm_provider  # noqa: E402,F401


def build_llm(system_prompt: str):
    cfg = settings.get("llm", {})
    provider, model = llm_provider(cfg)
    if provider in FAST_LLMS:
        return _fast_llm(provider, model, cfg, system_prompt)
    return GoogleLLMService(
        api_key=secret("GOOGLE_API_KEY"),
        settings=GoogleLLMService.Settings(
            model=model,
            system_instruction=system_prompt,
            temperature=cfg.get("temperature", 0.4),
            max_tokens=cfg.get("max_output_tokens", 120),
            **_thinking(cfg),
        ),
        **_resilience(cfg),
    )


def _fast_llm(provider: str, model: str, cfg: dict, system_prompt: str):
    """Groq / Cerebras / OpenAI through Pipecat's built-in OpenAI-compatible services (streaming)."""
    import importlib

    key_name, module, cls_name = FAST_LLMS[provider]
    cls = getattr(importlib.import_module(module), cls_name)
    reasoning = any(r in model for r in REASONING_MODELS)
    effort = str(cfg.get("reasoning_effort", "low"))
    # Reasoning models spend tokens "thinking" before they answer: leave room for that, or the
    # reply comes back empty. The prompt keeps spoken replies short.
    ceiling = int(cfg.get("max_output_tokens", 120))
    ceiling = max(ceiling, 400) if reasoning else ceiling
    kw: dict = {"model": model, "system_instruction": system_prompt,
            "temperature": cfg.get("temperature", 0.4), "max_completion_tokens": ceiling}
    if reasoning and provider == "groq":
        kw["reasoning_effort"] = effort
    elif reasoning:
        kw["extra"] = {"reasoning_effort": effort}
    params = inspect.signature(cls.__init__).parameters
    extra_kwargs = {}
    if "retry_on_timeout" in params or "kwargs" in params:
        # A caller is waiting: if no first word within 3 s, ask again once.
        extra_kwargs = {"retry_on_timeout": True, "retry_timeout_secs": float(cfg.get("first_word_timeout_secs", 3.0))}
    return cls(api_key=secret(key_name), settings=cls.Settings(**kw), **extra_kwargs)


# ---------- Text-to-speech ----------
def _text_filters() -> dict:
    """Pronunciation fixes (Agents > Speech settings > Pronunciation)."""
    from app.pronunciation import PronunciationFilter, rules_from

    rules = rules_from(settings)
    return {"text_filters": [PronunciationFilter(rules)]} if rules else {}


def build_tts(sample_rate: int | None = None):
    """sample_rate: the pipeline's output rate (phone: 16000). Deepgram adapts by itself;
    Sarvam must be told, otherwise it sends 24 kHz audio."""
    cfg = settings.get("tts", {})
    p = voice_config.plan(settings)
    if p.get("note"):
        from loguru import logger
        logger.info(f"Voice: {p['note']}")

    # The chosen voice decides the provider (voice_config.plan): sarvam:<name> -> Sarvam preset,
    # clone:svc-... -> your cloned voice, aura-2-... -> Deepgram. tts.provider in agent.yaml is only
    # used to pick Deepgram Flux; it must NOT override a cloned voice, or the clone is never used.
    provider = p.get("tts") or "sarvam"
    
    # FORCE SARVAM: If agent.yaml explicitly asks for sarvam, override everything else
    if str(cfg.get("provider", "")).lower() == "sarvam":
        provider = "sarvam"

    if provider == "deepgram" and cfg.get("provider") == "deepgram_flux":
        provider = "deepgram_flux"

    if provider == "sarvam":
        return _sarvam_tts(p, cfg, sample_rate)
    if provider == "clone":
        from app.sarvam_clone_tts import SarvamCloneTTSService
        return SarvamCloneTTSService(
            api_key=secret("SARVAM_API_KEY"),
            voice_id=p.get("voice"),  # the svc-... id, without the "clone:" prefix
            language=p.get("tts_language", "en-IN"),
            pace=p.get("speed", 1.0),
            sample_rate=sample_rate or 16000,
            **_text_filters(),
        )
    if provider == "deepgram_flux":
        from pipecat.services.deepgram.flux.tts import DeepgramFluxTTSService
        return DeepgramFluxTTSService(
            api_key=secret("DEEPGRAM_API_KEY"),
            **_url_kwarg(DeepgramFluxTTSService, f"wss://{deepgram_host()}/v2/speak"),
            settings=DeepgramFluxTTSService.Settings(
                voice=cfg.get("flux_voice", "flux-heather-en"),
            ),
            **_text_filters(),
        )
    return DeepgramTTSService(
        api_key=secret("DEEPGRAM_API_KEY"),
        **_url_kwarg(DeepgramTTSService, f"wss://{deepgram_host()}"),
        settings=DeepgramTTSService.Settings(voice=cfg.get("voice") or p.get("voice")),
        **_text_filters(),
    )



def _sarvam_tts(p: dict, cfg: dict, sample_rate: int | None):
    try:
        from pipecat.services.sarvam.tts import SarvamTTSService
    except Exception as e:  # noqa: BLE001
        raise RuntimeError('Sarvam is not installed: pip install "pipecat-ai[sarvam]==1.12.0"') from e

    from app import sarvam_v4

    sarvam_v4.register()  # adds bulbul:v4-flash to Pipecat's model table (no site-packages edits)
    # voice_config.plan already picked the voice (and swapped in a fallback when the voice can't
    # speak the call language). Use it, unless tts.provider forces Sarvam with a non-Sarvam voice.
    if p.get("tts") == "sarvam":
        voice_name = str(p["voice"])
    else:
        raw = str(cfg.get("voice") or "").replace("sarvam:", "").strip().lower()
        voice_name = raw if raw in voice_config.SARVAM_ALL else voice_config.DEFAULT_SARVAM_VOICE

    return SarvamTTSService(
        api_key=secret("SARVAM_API_KEY"),
        sample_rate=sample_rate or 16000,
        settings=SarvamTTSService.Settings(
            # The voice decides the model: v4 persona voices -> bulbul:v4-flash, v3 names -> bulbul:v3.
            model=sarvam_v4.model_for(voice_name),
            voice=voice_name,
            language=p.get("tts_language", "en-IN"),
            pace=float(p.get("speed", 1.15)),
            min_buffer_size=int(cfg.get("sarvam_min_buffer", 30)),
            max_chunk_length=int(cfg.get("sarvam_max_chunk", 150)),
        ),
        **_text_filters(),
    )


def build_language_follower():
    """Auto-detect calls: when the candidate switches language (Sarvam STT reports it on each
    final transcript), switch the voice's language too. None for fixed-language calls."""
    p = voice_config.plan(settings)
    if not p["follow_language"] or p["tts"] != "sarvam":
        return None
    from app.language_follower import LanguageFollower

    return LanguageFollower(start=p["tts_language"])


# ---------- Turn-taking (when has the candidate finished talking?) ----------
def _stop_strategy():
    cfg = turn_cfg()
    mode = cfg.get("mode", "smart")
    if mode == "timeout":
        # Simple rule: reply after a fixed pause. Fast but may cut people off.
        return SpeechTimeoutUserTurnStopStrategy(
            user_speech_timeout=float(cfg.get("speech_timeout_secs", 0.6))
        )
    # Smart turn: a small on-device model judges if the sentence is finished.
    # Pipecat waits up to 3.0 s when it thinks you are NOT finished - that was
    # the main cause of the 2.8 s median. We cap that wait.
    analyzer = LocalSmartTurnAnalyzerV3(
        cpu_count=int(cfg.get("smart_turn_cpu_threads", 2)),
        params=SmartTurnParams(stop_secs=float(cfg.get("smart_turn_max_wait_secs", 2.0))),
    )
    return TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)


def _start_strategies():
    """When may the candidate take the turn (and interrupt the bot)?

    Pipecat's default lets ANY detected sound interrupt the bot: a fan, traffic,
    a cough or the bot's own voice leaking from the speaker. That makes the bot
    stop mid-sentence and wait for speech that never comes.
    With interrupt_min_words: 2 the bot only stops when you say 2+ real words.
    While the bot is silent, a single word still starts your turn.
    """
    min_words = int(turn_cfg().get("interrupt_min_words", 1))
    if min_words <= 1:
        return [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]
    return [MinWordsUserTurnStartStrategy(min_words=min_words, use_interim=True)]


def build_vad() -> SileroVADAnalyzer:
    cfg = turn_cfg()
    params = VADParams(
        confidence=float(cfg.get("vad_confidence", 0.7)),   # how sure it must be that it's speech
        start_secs=float(cfg.get("vad_start_secs", 0.2)),   # speech needed before it counts
        stop_secs=float(cfg.get("vad_stop_secs", 0.2)),     # silence needed before turn check
        min_volume=float(cfg.get("vad_min_volume", 0.6)),   # ignore quieter sounds
    )
    return SileroVADAnalyzer(params=params)


def build_context_aggregators(tools=None, silence_secs: float | None = None,
                              messages: list[dict] | None = None):
    """tools: ToolsSchema from app.call_actions.register_tools (Sprint 3b), or None.
    silence_secs: seconds of candidate silence (after Priya stops) before the
    'on_user_turn_idle' event fires; None/0 = off.
    messages: conversation history to start with (app.prompts.opening_messages)."""
    vad_stop_secs = float(turn_cfg().get("vad_stop_secs", 0.2))
    initial = list(messages or [])
    context = LLMContext(messages=initial, tools=tools) if tools is not None else LLMContext(messages=initial)
    if uses_flux():
        user_params = {}  # Deepgram Flux brings its own turn detection (its recommended strategies apply)
    elif sarvam_decides_turns():
        # Sarvam's speech detection decides turns (its recommended strategies apply). The local VAD
        # is still needed: it times latency and marks the real end of speech.
        user_params = dict(vad_analyzer=build_vad())
    else:
        user_params = dict(
            vad_analyzer=build_vad(),
            user_turn_strategies=UserTurnStrategies(start=_start_strategies(), stop=[_stop_strategy()]),
        )
    if silence_secs:
        user_params["user_idle_timeout"] = float(silence_secs)
    if bool(turn_cfg().get("protect_greeting", True)):
        # The opening line always plays in full: while the bot speaks its FIRST sentence, the
        # candidate's audio can't interrupt it (a "hello?" over the greeting is ignored). From then
        # on, interruptions work normally.
        from pipecat.turns.user_mute import FirstSpeechUserMuteStrategy

        user_params["user_mute_strategies"] = [FirstSpeechUserMuteStrategy()]
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(**user_params)
    )
    return user_agg, assistant_agg, vad_stop_secs


def build_context_window():
    """Rolling history for long calls (app/context_window.py). llm.max_history_messages: 0 = off."""
    n = int((settings.get("llm", {}) or {}).get("max_history_messages", 24))
    if n <= 0:
        return None
    from app.context_window import ContextWindow

    return ContextWindow(max_messages=n)
