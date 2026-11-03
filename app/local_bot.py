"""Sprint 1: talk to the agent through your laptop microphone and speaker.

Run from the project folder (virtual environment active):
    python -m app.local_bot

Use headphones, otherwise the bot hears its own voice and interrupts itself.
Press Ctrl+C to stop; a latency summary is printed at the end.

Pipeline:  mic -> Deepgram STT -> Gemini -> Deepgram TTS -> speaker
All settings come from config/agent.yaml (see app/services.py).
"""

import asyncio
import sys

from loguru import logger

from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from app.call_actions import (
    CallContext,
    CallState,
    attach_llm_error_handler,
    attach_silence_handler,
    make_call_state_observer,
    opening_frames,
    register_tools,
)
from app.config import settings
from app.latency import LatencyObserver
from app.prompts import greeting_line, load_prompt, opening_messages, test_candidate_name
try:  # knowledge base search (only on systems that have app/kb_search.py)
    from app.kb_search import build_injector
except ImportError:
    def build_injector():
        return None
from app.reply_guard import NoReplyWithoutNewWords, OneQuestionPerTurn
from app.services import (
    build_context_aggregators,
    build_llm,
    build_stt,
    build_language_follower,
    build_tts,
)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> {message}")


async def main():
    audio = settings.get("audio", {}) or {}
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # None = system default. Run scripts/mic_test.py to find the right numbers.
            input_device_index=audio.get("input_device_index"),
            output_device_index=audio.get("output_device_index"),
        )
    )
    # Sprint 3b: Priya asks for this name and can end the call / save a callback herself
    state = CallState(context=CallContext(candidate_name=test_candidate_name()))
    silence_secs = (settings.get("call_limits", {}) or {}).get("silence_hangup_seconds", 8)

    try:  # newer services.py: per-call vocabulary hints
        stt = build_stt(extra_terms=[state.context.candidate_name])
    except TypeError:  # this system's services.py
        stt = build_stt()
    llm = build_llm(load_prompt(candidate_name=state.context.candidate_name))
    tools = register_tools(llm, state)
    tts = build_tts()
    follower = build_language_follower()
    kb = build_injector()  # None when the knowledge base is empty or in "full" mode
    greeting = greeting_line(state.context.candidate_name)  # same instant opening line as calls
    user_agg, assistant_agg, vad_stop_secs = build_context_aggregators(
        tools=tools, silence_secs=silence_secs, messages=opening_messages(greeting)
    )

    pipeline = Pipeline(
        [
            transport.input(),   # microphone
            stt,                 # speech -> text
            *([follower] if follower else []),
            user_agg,            # turn detection + adds your words to the context
            *([kb] if kb else []),  # knowledge base: best-matching parts for this turn
            NoReplyWithoutNewWords(),  # never answer twice without new words from the candidate
            llm,                 # decides what to say
            OneQuestionPerTurn(),  # cut the reply after its first question
            tts,                 # text -> speech
            transport.output(),  # speaker
            assistant_agg,       # adds the bot's words to the context
        ]
    )

    verbose = bool(settings.get("logging", {}).get("verbose", True))
    latency = LatencyObserver(vad_stop_secs=vad_stop_secs, verbose=verbose)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[latency, make_call_state_observer(state)],
    )
    attach_silence_handler(user_agg, worker, state)

    async def _hang_up_after_goodbye():  # Priya's goodbye plays first, then the line closes
        await worker.queue_frames([EndFrame()])

    state.auto_end = _hang_up_after_goodbye
    attach_llm_error_handler(worker, llm, state)

    turn = settings.get("turn", {})
    logger.info(
        f"Model {settings['llm'].get('model')} | voice "
        f"{settings['tts'].get('provider', 'deepgram')} | turn mode {turn.get('mode', 'smart')} "
        f"(max wait {turn.get('smart_turn_max_wait_secs', 0.8)} s, interrupt after "
        f"{turn.get('interrupt_min_words', 2)} words) | mic device "
        f"{audio.get('input_device_index', 'default')}"
    )

    @worker.event_handler("on_pipeline_started")
    async def on_started(_worker, *args):
        logger.info(
            "Bot is live. Wait for the greeting, then talk. Priya hangs up by herself when done "
            "(or press Ctrl+C)."
        )
        await worker.queue_frames(opening_frames(greeting))  # bot speaks first, like a real call

    runner = WorkerRunner()
    await runner.add_workers(worker)

    try:
        await runner.run()
    finally:
        logger.info(latency.summary())
        if settings.get("logging", {}).get("extract_after_local_session", True):
            await _extract_and_save(latency.transcript, latency.stats(), state)


async def _extract_and_save(transcript: list, stats: dict, state: CallState):
    """Sprint 3: show the screening data Priya captured in this session."""
    import json
    import time

    from app.config import ROOT
    from app.outcome import describe, extract_outcome, reconcile

    if state.ended_by is None:
        state.ended_by = "candidate"  # you pressed Ctrl+C
    if not transcript:
        return
    logger.info("Extracting screening data from this conversation...")
    try:
        outcome = await extract_outcome(transcript, agent_name=settings["agent"].get("name", "Priya"))
        outcome = reconcile(outcome, state)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Extraction failed: {e}")
        return
    for line in describe(outcome).splitlines():
        logger.info(line)
    out_dir = ROOT / "logs" / "calls"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_local-voice.json"
    path.write_text(
        json.dumps({"call_id": "local-voice", "channel": "laptop",
                    "candidate_name": state.context.candidate_name, "outcome": outcome,
                    "latency": stats, "transcript": transcript},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())