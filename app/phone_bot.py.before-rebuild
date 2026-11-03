"""Sprint 2: Priya on a real phone call (one pipeline per call).

Same speech, LLM, voice and turn settings as the laptop bot (app/services.py);
only the audio transport changes: Vobiz streams the call over a WebSocket.
"""

import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import WebSocket
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from app import live, sheet_sync, vobiz
from app.call_actions import (
    CallContext,
    CallState,
    attach_llm_error_handler,
    attach_reply_watchdog,
    attach_silence_handler,
    make_call_state_observer,
    opening_frames,
    register_tools,
)
from app.config import ROOT, settings
from app.latency import LatencyObserver
from app.outcome import describe, extract_outcome, fallback_outcome, reconcile
from app.prompts import fallback_greeting, greeting_delay, greeting_line, greeting_mode, load_prompt, opening_messages
from app.services import (
    build_context_aggregators,
    build_context_window,
    build_llm,
    build_stt,
    build_language_follower,
    build_tts,
)

CALL_LOG_DIR = Path(ROOT) / "logs" / "calls"
PIPELINE_SAMPLE_RATE = 16000  # phone audio is 8 kHz; the connector resamples both ways


def _transport_params(serializer) -> FastAPIWebsocketParams:
    kwargs = dict(
        audio_in_enabled=True,
        audio_out_enabled=True,
        add_wav_header=False,
        serializer=serializer,
    )
    if "allowed_origins" in getattr(FastAPIWebsocketParams, "model_fields", {}):
        kwargs["allowed_origins"] = []
    return FastAPIWebsocketParams(**kwargs)


async def run_phone_bot(
    websocket: WebSocket, call_id: str, stream_id: str, candidate: CallContext | None = None
) -> None:
    label = call_id[-6:] if call_id else stream_id[-6:]
    browser_test = (call_id or "").startswith("browser-")
    simulated = (call_id or "").startswith("loadtest-") or browser_test
    limits = settings.get("call_limits", {}) or {}
    max_call_secs = int(limits.get("max_call_seconds", 240))
    started = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    state = CallState(context=candidate or CallContext())
    state.context.call_id = call_id or ""
    if state.context.candidate_name:
        logger.info(f"[{label}] Calling {state.context.candidate_name}")

    transport = FastAPIWebsocketTransport(
        websocket=websocket, params=_transport_params(vobiz.make_serializer(stream_id))
    )
    stt = build_stt()
    from app.transcript_fixer import TranscriptFixer, rules_from

    fixer = TranscriptFixer(rules_from(settings), log_label=label)  # "merger stack" -> "MERN stack"
    try:
        from app import kb_store

        kb = kb_store.report()
        logger.info(
            f"[{label}] Knowledge base: {kb['used_chars']:,} of {kb['source_chars']:,} characters from "
            f"{kb['sources']} source(s), {kb['faqs']} FAQ(s)"
            + (" (some text cut: raise the limit or shorten)" if kb["truncated"] else "")
        )
    except Exception:
        pass

    # ---- Call resumption ("two brains"): standard prompt, or the callback prompt with the last
    # call's summary and closing lines when this number's recent call ended early. ----
    from app import call_resume

    previous = None if browser_test else call_resume.find_previous_call(state.context.phone)
    callback_prompt = call_resume.callback_template(previous) if previous else None
    if previous and not callback_prompt:
        logger.warning(f"[{label}] Previous call found but config/prompts/"
                       f"{call_resume.cfg()['prompt_file']} is missing: using the standard prompt")
    if callback_prompt:
        logger.info(f"[{label}] Resuming: last call {previous.get('_file')} "
                    f"({(previous.get('outcome') or {}).get('call_outcome') or 'ended early'}) -> callback prompt")
        system_prompt = load_prompt(
            template=callback_prompt,
            candidate_name=state.context.candidate_name,
            job_title=state.context.role,
            phone="",  # history is already in the prompt; don't add contact memory a second time
        )
    else:
        system_prompt = load_prompt(
            candidate_name=state.context.candidate_name,
            job_title=state.context.role,
            phone=state.context.phone,
        )
    llm = build_llm(system_prompt)
    tools = register_tools(llm, state)
    tts = build_tts(sample_rate=PIPELINE_SAMPLE_RATE)
    from app.call_actions import make_reply_guard

    reply_guard = make_reply_guard(state, label)
    from app.call_actions import make_interruption_note

    interruption_note = make_interruption_note(state, label)
    follower = build_language_follower()
    greeting = greeting_line(state.context.candidate_name)
    if callback_prompt and greeting:  # "Hi Karthik, it's Suguna again. Sorry, our call got cut earlier..."
        greeting = call_resume.resume_greeting(state.context.candidate_name) or greeting
    window = build_context_window()
    from app import kb_search

    retriever = kb_search.make_retriever(label) if kb_search.retrieve_mode() else None
    if retriever:
        try:
            r = kb_search.report()
            logger.info(f"[{label}] Knowledge search: {r['chunks']} passages ({r['chars']:,} chars) searchable, "
                        f"{r['pinned_chars']:,} chars of FAQs pinned")
        except Exception:  # noqa: BLE001
            pass
    user_agg, assistant_agg, vad_stop_secs = build_context_aggregators(
        tools=tools,
        silence_secs=limits.get("silence_hangup_seconds", 8),
        messages=opening_messages(greeting),
    )

    recorder, recording_file = _recorder(call_id or stream_id, label)
    tap_ids = [ call_id, stream_id ]
    in_tap = live.LiveAudioTap(tap_ids, tap_input=True, tap_output=False)
    out_tap = live.LiveAudioTap(tap_ids, tap_input=False, tap_output=True)

    pipeline = Pipeline(
        [
            transport.input(),   # caller's audio from Vobiz
            in_tap,              # <-- broadcasts caller's voice to dashboard
            stt,
            fixer,               # corrects common mishearings before the AI reads them
            *([follower] if follower else []),
            user_agg,
            *([window] if window else []),
            *([retriever] if retriever else []),  # adds the matching knowledge passages for this turn
            interruption_note,   # cut off mid-sentence? drop the unfinished question
            llm,
            reply_guard,         # tool calls written as text are never spoken
            tts,
            out_tap,             # <-- broadcasts Suguna's voice to dashboard
            transport.output(),  # Suguna's audio back to Vobiz
            *([recorder] if recorder else []),
            assistant_agg,
        ]
    )

    verbose = bool(settings.get("logging", {}).get("verbose", True))
    observer = LatencyObserver(vad_stop_secs=vad_stop_secs, verbose=verbose, label=label)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
            audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[observer, make_call_state_observer(state)],
    )
    attach_silence_handler(user_agg, worker, state)
    attach_llm_error_handler(worker, llm, state, label)
    timer: asyncio.Task | None = None
    opener: asyncio.Task | None = None
    watchdogs: list[asyncio.Task] = []

    async def call_time_limit():
        await asyncio.sleep(max_call_secs)
        state.ended_by = state.ended_by or "time_limit"
        state.record("end_call", reason=f"max call length {max_call_secs} s")
        logger.info(f"[{label}] Max call length reached, wrapping up")
        await worker.queue_frames(
            [
                TTSSpeakFrame("I'm sorry, I have to end the call now. A recruiter will follow up. Thank you!"),
                EndFrame(),
            ]
        )

    delay = greeting_delay()
    user_first = greeting_mode() == "user_first"

    async def open_call():
        if delay:
            await asyncio.sleep(delay)
        if not user_first:
            await worker.queue_frames(opening_frames(greeting))
            return
        await asyncio.sleep(4)
        if not state.last_user_text_at and not state.bot_spoke_since_user():
            line = fallback_greeting(state.context.candidate_name)
            if line:
                logger.info(f"[{label}] Candidate silent after pick-up, Priya greets")
                await worker.queue_frames([TTSSpeakFrame(line)])

    @transport.event_handler("on_client_connected")
    async def on_connected(_transport, _client):
        nonlocal timer, opener
        logger.info(f"[{label}] Call connected" + (", waiting for candidate" if user_first else ", Priya greeting"))
        opener = asyncio.create_task(open_call())
        timer = asyncio.create_task(call_time_limit())
        watchdogs.append(attach_reply_watchdog(worker, state))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client):
        logger.info(f"[{label}] Caller disconnected")
        state.ended_by = state.ended_by or "candidate"
        try:
            await worker.cancel(reason="caller hung up")
        except TypeError:
            await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    live_id = call_id or stream_id
    live.register(
        live_id,
        name=state.context.candidate_name,
        phone=state.context.phone,
        role=state.context.role,
        transcript=observer.transcript,
        simulated=simulated,
    )
    if call_id and stream_id and call_id != stream_id:
        live.register(
            stream_id,
            name=state.context.candidate_name,
            phone=state.context.phone,
            role=state.context.role,
            transcript=observer.transcript,
            simulated=simulated,
        )

    try:
        await runner.run()
    finally:
        live.unregister(live_id)
        if stream_id:
            live.unregister(stream_id)
        if timer:
            timer.cancel()
        if opener:
            opener.cancel()
        for w in watchdogs:
            w.cancel()
        if not simulated and state.ended_by != "transfer":
            await vobiz.hang_up(call_id)
        if state.ended_by == "transfer":
            logger.info(f"[{label}] Call handed to the recruiter; not hanging up")

        duration = round(time.time() - started)
        stats = observer.stats()
        state.ended_by = state.ended_by or "candidate"
        reason = f"ended by {state.ended_by}" + (f" ({state.end_reason})" if state.end_reason else "")
        logger.info(f"[{label}] Call ended after {duration} s: {reason}")
        logger.info(observer.summary())
        outcome = await _extract(label, observer.transcript, state)
        record = _save_call_log(
            call_id or stream_id,
            duration,
            reason,
            stats,
            observer.transcript,
            outcome,
            state,
            started_at,
            recording_file if recording_file and recording_file.exists() else None,
        )
        if not simulated:
            await sheet_sync.push_call(record, label)


async def _extract(label: str, transcript: list, state: CallState) -> dict | None:
    try:
        outcome = await extract_outcome(transcript, agent_name=settings["agent"].get("name", "Priya"))
        outcome = reconcile(outcome, state)
    except Exception as e:
        logger.warning(f"[{label}] Extraction failed: {e}")
        try:
            outcome = fallback_outcome(str(e), state)
        except Exception:
            return None
    for line in describe(outcome).splitlines():
        logger.info(f"[{label}] {line}")
    return outcome


RECORDING_DIR = Path(ROOT) / "logs" / "recordings"


def _recorder(call_id: str, label: str):
    if not (settings.get("recording") or {}).get("enabled"):
        return None, None
    try:
        from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
    except ImportError:
        return None, None
    import re
    import wave

    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORDING_DIR / (re.sub(r"[^\w.-]", "_", call_id)[:80] + ".wav")
    rec = AudioBufferProcessor(sample_rate=8000, num_channels=1, auto_start_recording=True)

    @rec.event_handler("on_audio_data")
    async def on_audio(_proc, audio: bytes, sample_rate: int, num_channels: int):
        try:
            with wave.open(str(path), "wb") as w:
                w.setnchannels(num_channels)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(audio)
            logger.info(f"[{label}] Recording saved: {path.name} ({len(audio) // (2 * sample_rate)} s)")
        except Exception as e:
            logger.warning(f"[{label}] Could not save recording: {e}")

    return rec, path


def _save_call_log(
    call_id: str,
    duration: int,
    reason: str,
    stats: dict,
    transcript: list,
    outcome: dict | None,
    state: CallState,
    started_at: str = "",
    recording: Path | None = None,
) -> dict:
    record = {
        "call_id": call_id,
        "candidate_name": state.context.candidate_name,
        "phone": state.context.phone,
        "ref": state.context.ref,
        "role": state.context.role,
        "agent": settings.get("agent", {}).get("name", "Priya"),
        "from_number": os.getenv("VOBIZ_FROM_NUMBER", "").strip(),
        "direction": "web" if call_id.startswith("browser-") else "outbound",
        **({"recording": recording.name} if recording else {}),
        **({"channel": "browser test"} if call_id.startswith("browser-") else {}),
        "started_at": started_at,
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_secs": duration,
        "end_reason": reason,
        "outcome": outcome,
        "latency": stats,
        "transcript": transcript,
    }
    try:
        CALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = CALL_LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{call_id[-8:]}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Call log saved: {path.relative_to(ROOT)}")
    except Exception as e:
        logger.warning(f"Could not save call log: {e}")
    return record