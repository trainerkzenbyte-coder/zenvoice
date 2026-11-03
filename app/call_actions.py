"""Sprint 3b: actions Priya can take DURING a call (LLM function calling).



    end_call(reason)              Priya says goodbye, then hangs up herself.

    schedule_callback(time, ...)  Records when to call the candidate back.

    transfer_call(reason, summary) Hands the candidate to a human recruiter (Sprint 9, app/transfer.py).

                                  Only offered to the AI when Agents > Functions > transfer_call is on.



Plus silence handling: if the candidate goes quiet, Priya asks "are you still

there?"; after a second silence she closes politely and hangs up.



The rules (validation, what gets recorded) are plain Python in this file and are

shared by the live voice pipeline AND the text simulator / automated tests, so

what you test by text is exactly what runs on a call.

"""



import datetime as dt

import time

from dataclasses import dataclass, field

from zoneinfo import ZoneInfo



from loguru import logger



from app.config import settings



IST = ZoneInfo("Asia/Kolkata")



END_REASONS = [

    "completed",

    "not_interested",

    "callback_requested",

    "wrong_person",

    "language_barrier",

    "do_not_call",

    "voicemail",

    "other",

]



# One definition of each tool, used for Pipecat (live calls) and Gemini REST (text tests).

TOOL_SPECS = [

    {

        "name": "end_call",

        "description": (

            "End the phone call. Use it ONLY when one of these is clearly true: all screening questions are "

            "answered and you have said goodbye; the candidate clearly says they are not interested, says bye, "

            "or asks to end the call; it is the wrong person; they ask not to be called again; you reached "

            "voicemail or an automated message; a callback has been saved and you said goodbye. "

            "NEVER use it because the candidate asked a question, paused, or said something unclear: answer or "

            "ask again instead. Say your goodbye in the same reply."

        ),

        "properties": {

            "reason": {

                "type": "string",

                "enum": END_REASONS,

                "description": "Why the call is ending.",

            }

        },

        "required": ["reason"],

    },

    {

        "name": "schedule_callback",

        "description": (

            "Save a callback time. Use it ONLY when the candidate has asked to be called later AND has "

            "said a day or time (e.g. 'call me tomorrow at 6'). Never use it for questions or for anything "

            "else. Repeat the time back to them first. If the result says invalid, ask for another time."

        ),

        "properties": {

            "callback_time": {

                "type": "string",

                "description": "Callback time in India time as YYYY-MM-DDTHH:MM, resolved from what they said.",

            },

            "spoken_time": {

                "type": "string",

                "description": "The time exactly as the candidate said it, e.g. 'tomorrow after 6 pm'.",

            },

            "language": {

                "type": "string",

                "enum": ["english", "hindi", "tamil"],

                "description": "Language for the callback, if the candidate asked for one.",

            },

        },

        "required": ["callback_time", "spoken_time"],

    },

]





TRANSFER_REASONS = ["candidate_asked_for_human", "question_needs_recruiter", "interested_wants_details", "complaint"]

TRANSFER_SPEC = {

    "name": "transfer_call",

    "description": (

        "Transfer the call to a human recruiter. Use it only when the candidate asks to speak to a person, "

        "or wants details only a recruiter can give and is happy to be connected now. Do NOT say a goodbye: "

        "the system says the hand-off line for you. Never use it for voicemail or uninterested candidates."

    ),

    "properties": {

        "reason": {"type": "string", "enum": TRANSFER_REASONS, "description": "Why the call is being transferred."},

        "summary": {"type": "string", "description": "One short sentence for the recruiter: who the candidate is and what they want."},

    },

    "required": ["reason", "summary"],

}





SEARCH_SPEC = {
    "name": "search_knowledge",
    "description": (
        "Search the full knowledge base (documents, PDFs, policies, fees, process, locations). Use it when the "
        "candidate asks something specific that the FAQs and the KNOWLEDGE FOR THIS TURN note don't answer. "
        "Pass a short query with the key words, e.g. 'refund policy' or 'mock interview schedule online'. "
        "Say nothing before calling it; answer from the result in one or two short sentences."
    ),
    "properties": {
        "query": {"type": "string", "description": "A few key words for what the candidate asked."},
    },
    "required": ["query"],
}


def search_knowledge_result(query: str) -> dict:
    """What the AI sees after search_knowledge (shared by live calls and text tests)."""
    from app import kb_search

    query = " ".join(str(query or "").split())[:200]
    hits = kb_search.search(query, k=3)
    if not hits:
        return {"status": "no_match", "message": "Nothing in the knowledge base about this. Say the team will "
                "confirm the details (don't guess), then continue."}
    return {"status": "found", "passages": kb_search.format_hits(hits, limit=2200),
            "next": "Answer the candidate from these passages in one or two short spoken sentences."}


def active_tool_specs() -> list[dict]:

    """The tools the AI may use on this call: transfer_call only when it is switched on,
    search_knowledge when the knowledge base is searched per turn (knowledge.mode: retrieve)."""

    from app import kb_search, transfer_cfg

    search = []
    try:
        if kb_search.retrieve_mode() and kb_search.cfg()["tool"] and kb_search.index().n:
            search = [SEARCH_SPEC]
    except Exception:  # noqa: BLE001
        search = []
    return TOOL_SPECS + ([TRANSFER_SPEC] if transfer_cfg.available() else []) + search





@dataclass

class CallContext:

    """Who we are calling. Filled from make_call.py on phone calls, from config in local tests."""



    candidate_name: str = ""

    phone: str = ""

    ref: str = ""  # campaign reference (Sprint 6), empty for test calls

    role: str = ""  # job title for this call (dashboard dialer); empty = screening.yaml job.title

    call_id: str = ""  # Vobiz call UUID (needed to transfer the call)





@dataclass

class CallState:

    """Everything Priya decided during the call. Saved with the call log."""



    context: CallContext = field(default_factory=CallContext)

    actions: list[dict] = field(default_factory=list)

    end_reason: str | None = None

    ended_by: str | None = None  # "agent" | "candidate" | "time_limit" | "silence" | "error" | "transfer"

    callback_time: str | None = None

    callback_spoken: str | None = None

    callback_language: str | None = None

    silence_prompts: int = 0

    last_user_text_at: float = 0.0

    last_bot_text_at: float = 0.0

    llm_errors: int = 0          # Gemini requests that failed during this call

    llm_errors_in_row: int = 0   # reset when Gemini answers again

    bot_speaking: bool = False   # Priya's audio is playing (transfer waits for the hand-off line)

    user_speaking: bool = False

    user_turns: int = 0          # final transcripts from the candidate

    recent_user: list = field(default_factory=list)  # last few things the candidate said (tool guards)

    watchdog_fired_for: float = 0.0

    last_user_turn_at: float = 0.0  # candidate spoke while Priya was silent (a reply is expected)

    transferred_to: str = ""

    llm_busy: bool = False       # the AI is working on a reply (thinking, or waiting on a slow provider)

    bot_interrupted: bool = False  # the candidate spoke over the agent; the next reply drops its unfinished thought

    llm_started_at: float = 0.0



    def record(self, action: str, **details):

        self.actions.append({"t": time.strftime("%H:%M:%S"), "action": action, **details})



    def bot_spoke_since_user(self) -> bool:

        return self.last_bot_text_at > self.last_user_text_at





# ---------- the rules (pure Python, shared by voice and text) ----------

def _calling_hours() -> tuple[dt.time, dt.time]:

    start, end = (settings.get("call_limits", {}) or {}).get("calling_hours", ["09:00", "21:00"])

    return dt.time.fromisoformat(start), dt.time.fromisoformat(end)





def apply_schedule_callback(state: CallState, args: dict, now: dt.datetime | None = None) -> dict:

    """Validate and save a callback. The returned dict is what Priya 'sees' as the result."""

    now = now or dt.datetime.now(IST)

    raw = str(args.get("callback_time", "")).strip().replace(" ", "T")[:16]

    spoken = str(args.get("spoken_time", "")).strip()

    try:

        when = dt.datetime.strptime(raw, "%Y-%m-%dT%H:%M").replace(tzinfo=IST)

    except ValueError:

        return {"status": "invalid", "message": "Could not understand the time. Ask for a clear day and time."}

    if when < now + dt.timedelta(minutes=10):

        return {"status": "invalid", "message": "That time has already passed or is too soon. Ask for a later time."}

    if when > now + dt.timedelta(days=30):

        return {"status": "invalid", "message": "That is more than 30 days away. Ask for a time within the next few weeks."}

    start, end = _calling_hours()

    if not (start <= when.time() <= end):

        return {

            "status": "invalid",

            "message": (

                f"We can only call between {start.strftime('%I %p').lstrip('0').lower()} and "

                f"{end.strftime('%I %p').lstrip('0').lower()}. Politely ask for a time in that window."

            ),

        }

    state.callback_time = when.strftime("%Y-%m-%dT%H:%M")

    state.callback_spoken = spoken

    state.callback_language = args.get("language") or state.callback_language

    state.record("schedule_callback", callback_time=state.callback_time, spoken=spoken,

                 language=state.callback_language)

    return {"status": "saved", "callback_time": state.callback_time,

            "next": "Confirm the time in one short sentence, say goodbye, then call end_call."}





# ---------- tool guards: the AI must not hang up or book callbacks on a whim ----------

import re as _re



_BYE = _re.compile(r"\b(bye|good ?bye|cut (the )?call|hang ?up|end (the )?call|disconnect|call (me )?later|busy|driving|in a meeting|"

                   r"not (now|a good time)|stop calling|don'?t call|remove my number)\b", _re.I)

_NOT_INTERESTED = _re.compile(r"\b(no|not interested|no thanks?|not looking|don'?t want|no need|not required|nahi|nahin|vendam|venam|"

                              r"illa|beda|vaddu|already (placed|joined|got))\b", _re.I)

_CALLBACK = _re.compile(r"\b(call (me )?(back|later|again)|callback|later|tomorrow|tonight|evening|morning|afternoon|night|"

                        r"\d{1,2}(:\d{2})? ?(am|pm|o'?clock)?|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"

                        r"next week|weekend|after|baad mein|kal|naalai|saayangalam|busy|free)\b", _re.I)

ALWAYS_OK = {"voicemail", "wrong_person", "do_not_call", "language_barrier"}





def _recent(state: CallState, n: int = 3) -> str:

    return " | ".join(state.recent_user[-n:])





def guard_end_call(state: CallState, reason: str) -> str | None:

    """None = allowed. Otherwise the message the AI gets back (it then keeps talking)."""

    said = _recent(state)

    if reason in ALWAYS_OK or _BYE.search(said):

        return None

    if reason == "not_interested" and _NOT_INTERESTED.search(said):

        return None

    if reason == "callback_requested" and (state.callback_time or _CALLBACK.search(said)):

        return None

    if reason == "completed" and state.user_turns >= 4:

        return None

    if reason == "other" and state.user_turns >= 6:

        return None

    return ("Do not end the call: the candidate has not said goodbye, refused, or asked to end it. "

            "Reply to what they just said (answer their question if they asked one) and continue the conversation.")





def guard_callback(state: CallState) -> str | None:

    if _CALLBACK.search(_recent(state)):

        return None

    return ("The candidate did not ask to be called back. Do not schedule a callback. "

            "Reply to what they just said and continue the conversation.")





def apply_end_call(state: CallState, args: dict) -> str:

    """Record the end reason. Returns the reason actually used."""

    reason = str(args.get("reason", "other"))

    if reason not in END_REASONS:

        reason = "other"

    state.end_reason = reason

    state.ended_by = "agent"

    state.record("end_call", reason=reason)

    return reason





def apply_transfer(state: CallState, args: dict) -> dict:

    """Check a transfer request. Returns what the AI 'sees'; status 'transferring' means go ahead."""

    from app import transfer_cfg as transfer



    reason = str(args.get("reason", "candidate_asked_for_human"))

    reason = reason if reason in TRANSFER_REASONS else "candidate_asked_for_human"

    summary = " ".join(str(args.get("summary", "")).split())[:240]

    if not transfer.available():

        return {"status": "unavailable", "message": "Transfers are switched off. Tell the candidate a recruiter "

                "will call them back, then call end_call with reason callback_requested."}

    if not state.context.call_id or state.context.call_id.startswith(("browser-", "loadtest-")):

        return {"status": "unavailable", "message": "This is a test call (no phone line to transfer). Tell the "

                "candidate a recruiter will call them back, then call end_call with reason callback_requested."}

    state.record("transfer_call", reason=reason, summary=summary)

    return {"status": "transferring", "reason": reason, "summary": summary}





def closing_line(reason: str) -> str:

    """Fallback goodbye, used only if Priya ends the call without saying goodbye herself."""

    from app.prompts import screening_cfg



    lines = screening_cfg().get("closing_lines") or {}

    defaults = {

        "callback_requested": "Thank you, we'll call you back then. Have a good day!",

        "do_not_call": "I'm sorry for the trouble. We won't call you again. Goodbye.",

        "wrong_person": "Sorry for the trouble. Thank you, goodbye.",

        "voicemail": "",

    }

    return lines.get(reason) or defaults.get(reason, "Thank you for your time. Have a great day!")





# ---------- live voice pipeline (Pipecat) ----------

def opening_frames(greeting: str) -> list:

    """What to queue when the call connects.



    With a fixed greeting (app.prompts.greeting_line): speak it straight away. It is already

    in the conversation history (app.prompts.opening_messages), so it is not added twice.

    Without one: let the AI write the greeting (slower: waits for Gemini)."""

    from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame



    if not greeting:

        return [LLMRunFrame()]

    try:

        return [TTSSpeakFrame(greeting, append_to_context=False)]

    except TypeError:  # older Pipecat: spoken lines were never added to the history

        return [TTSSpeakFrame(greeting)]





def register_tools(llm, state: CallState):

    """Register the handlers on the Pipecat LLM and return the ToolsSchema for the context."""

    from pipecat.adapters.schemas.function_schema import FunctionSchema

    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    from pipecat.frames.frames import FunctionCallResultProperties, TTSSpeakFrame



    try:  # Pipecat >= 1.4

        from pipecat.frames.frames import EndWorkerFrame



        def end_frame_and_direction():

            return EndWorkerFrame(), None  # downstream: queued speech plays first

    except ImportError:  # older Pipecat

        from pipecat.frames.frames import EndTaskFrame

        from pipecat.processors.frame_processor import FrameDirection



        def end_frame_and_direction():

            return EndTaskFrame(), FrameDirection.UPSTREAM



    async def on_end_call(params):
        wanted = str((params.arguments or {}).get("reason", "other"))
        blocked = guard_end_call(state, wanted if wanted in END_REASONS else "other")
        if blocked:
            logger.info(f"  [guard] end_call({wanted}) refused: candidate didn't end the call")
            state.record("end_call_refused", reason=wanted)
            await params.result_callback("Call continuing.")  # <--- Clean string instead of JSON dict
            return
        reason = apply_end_call(state, dict(params.arguments or {}))
        logger.info(f"  [action] end_call ({reason})")
        await params.result_callback(
            "Call ended.", properties=FunctionCallResultProperties(run_llm=False)
        )
        # Say goodbye if the AI didn't, then hang up. The end frame goes out after the speech
        # already queued, so the goodbye plays completely before the line drops.
        line = closing_line(reason)
        if line and not state.bot_spoke_since_user():
            await params.llm.push_frame(TTSSpeakFrame(line))
        frame, direction = end_frame_and_direction()
        await (params.llm.push_frame(frame) if direction is None else params.llm.push_frame(frame, direction))



    async def on_schedule_callback(params):

        blocked = guard_callback(state)

        if blocked:

            logger.info("  [guard] schedule_callback refused: no callback request heard")

            state.record("callback_refused")

            await params.result_callback({"status": "rejected", "message": blocked})

            return

        result = apply_schedule_callback(state, dict(params.arguments or {}))

        logger.info(f"  [action] schedule_callback -> {result['status']} {result.get('callback_time', '')}")

        await params.result_callback(result)  # Priya continues: confirms, says bye, ends



    async def on_transfer_call(params):

        import asyncio



        from app import transfer, vobiz



        result = apply_transfer(state, dict(params.arguments or {}))

        logger.info(f"  [action] transfer_call -> {result['status']}")

        if result["status"] != "transferring":

            await params.result_callback(result)  # the AI says "a recruiter will call you back"

            return

        await params.result_callback({"status": "transferring"}, properties=FunctionCallResultProperties(run_llm=False))

        await params.llm.push_frame(TTSSpeakFrame(transfer.cfg()["announce"]))



        async def hand_over():

            # Let the hand-off line play completely, then swap the call over to the recruiter.

            waited = 0.0

            while not state.bot_speaking and waited < 3:  # wait for the line to start playing...

                await asyncio.sleep(0.1)

                waited += 0.1

            waited = 0.0

            while state.bot_speaking and waited < 15:  # ...and to finish

                await asyncio.sleep(0.1)

                waited += 0.1

            await asyncio.sleep(0.3)

            token = transfer.create(state.context.call_id, state.context.candidate_name, state.context.role,

                                    result["summary"], result["reason"])

            # Mark first: Vobiz closes our audio stream the moment the transfer starts, and the

            # call clean-up must then NOT hang up the (now transferred) call.

            state.ended_by, state.end_reason = "transfer", "transferred"

            state.transferred_to = str(transfer.cfg()["number"])

            try:

                await vobiz.transfer(state.context.call_id, f"{vobiz.public_base_url()}/transfer/{token}")

                logger.info("  [action] call handed to the recruiter")

            except Exception as e:  # noqa: BLE001  never leave the candidate in silence

                logger.warning(f"  [action] transfer failed: {e}")

                state.record("transfer_failed", error=str(e)[:200])

                state.end_reason, state.ended_by = "callback_requested", "agent"

                await params.llm.push_frame(TTSSpeakFrame(transfer.cfg()["unavailable"]))

                frame, direction = end_frame_and_direction()

                await (params.llm.push_frame(frame) if direction is None else params.llm.push_frame(frame, direction))



        asyncio.create_task(hand_over())



    async def on_search_knowledge(params):
        query = str((params.arguments or {}).get("query", ""))
        result = search_knowledge_result(query)
        logger.info(f"  [action] search_knowledge({query!r}) -> {result['status']}")
        state.record("search_knowledge", query=query[:120], status=result["status"])
        await params.result_callback(result)  # the AI then answers from it

    llm.register_function("search_knowledge", on_search_knowledge, cancel_on_interruption=True)

    llm.register_function("end_call", on_end_call, cancel_on_interruption=False)

    llm.register_function("schedule_callback", on_schedule_callback, cancel_on_interruption=False)

    llm.register_function("transfer_call", on_transfer_call, cancel_on_interruption=False)



    return ToolsSchema(

        standard_tools=[

            FunctionSchema(name=s["name"], description=s["description"],

                           properties=s["properties"], required=s["required"])

            for s in active_tool_specs()

        ]

    )





def reminder_cfg() -> tuple[int, str]:

    """Agents > Speech settings > Reminder: how many 'are you there?' prompts before hanging up."""

    limits = settings.get("call_limits", {}) or {}

    count = max(0, min(5, int(limits.get("reminder_count", 1))))

    return count, str(limits.get("reminder_line") or "Hello, are you still there?")





def attach_silence_handler(user_aggregator, worker, state: CallState):

    """Silence after Priya spoke: remind (reminder_count times), then close and hang up.

    The wait before each reminder is call_limits.silence_hangup_seconds."""

    from pipecat.frames.frames import EndFrame, TTSSpeakFrame



    @user_aggregator.event_handler("on_user_turn_idle")

    async def on_idle(*_args):

        if state.ended_by:

            return

        state.silence_prompts += 1

        count, line = reminder_cfg()

        if state.silence_prompts <= count:

            logger.info(f"  [silence] reminder {state.silence_prompts}/{count}")

            await worker.queue_frames([TTSSpeakFrame(line)])

        else:

            logger.info("  [silence] no response again, ending the call")

            state.end_reason = state.end_reason or "other"

            state.ended_by = "silence"

            state.record("end_call", reason="silence")

            await worker.queue_frames(

                [TTSSpeakFrame("It seems we got disconnected. We'll try you again later. Goodbye!"),

                 EndFrame()]

            )





def watchdog_secs() -> float:
    """call_limits.reply_watchdog_secs in config/agent.yaml (default 7 s)."""
    try:
        return max(3.0, min(20.0, float((settings.get("call_limits") or {}).get("reply_watchdog_secs", 7.0))))
    except (TypeError, ValueError):
        return 7.0


def attach_reply_watchdog(worker, state: CallState, timeout_secs: float | None = None):
    """Never leave the candidate in dead silence, without talking over a reply that is on its way.

    After the candidate finishes speaking and nothing has been said back:
      - while the AI is still working on the reply (thinking, slow provider), wait: up to 2x the timeout
      - after timeout_secs with no reply in progress: quietly ask the AI again (the candidate's words
        are already in the conversation, so they never have to repeat themselves)
      - after 2x the timeout, still nothing: one short recovery line, so the line is never silent."""
    import asyncio

    from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame

    timeout = float(timeout_secs or watchdog_secs())
    retried_for = 0.0

    async def loop():
        nonlocal retried_for
        while not state.ended_by:
            await asyncio.sleep(0.25)
            t = state.last_user_turn_at
            if (not t or t == state.watchdog_fired_for or state.user_speaking or state.bot_speaking
                    or state.last_bot_text_at >= t):
                continue
            now = time.monotonic()
            waited = now - t
            working = state.llm_busy and state.llm_started_at >= t and now - state.llm_started_at < timeout * 2
            if waited > timeout and retried_for != t and not working:
                retried_for = t
                state.record("silence_watchdog", step="retry")
                logger.warning(f"  [watchdog] no reply after {waited:.1f} s, asking the AI again")
                await worker.queue_frames([LLMRunFrame()])
            elif waited > timeout * 2 and retried_for == t:
                state.watchdog_fired_for = t
                state.record("silence_watchdog", step="spoken")
                logger.warning(f"  [watchdog] still no reply after {waited:.1f} s, speaking a recovery line")
                await worker.queue_frames([TTSSpeakFrame("Sorry, the line dropped for a second there. "
                                                         "Could you say that once more?")])

    return asyncio.create_task(loop())





def _leaked_call(text: str) -> tuple[str | None, dict]:
    """('end_call', {'reason': ...}) from a tool call the AI wrote as text, in any of the shapes
    models use: {"reason": ...}, {"type"/"name": "end_call", "args"/"arguments": {...}}, ```json ...```."""
    import json as _json
    import re as _r

    raw = text.strip().strip("`").strip()
    raw = raw[4:].strip() if raw.lower().startswith("json") else raw
    data: dict = {}
    try:
        val = _json.loads(raw)
        data = val if isinstance(val, dict) else {}
    except ValueError:
        m = _r.search(r'"reason"\s*:\s*"([a-z_]+)"', text)
        if m:
            data = {"reason": m.group(1)}
    name = str(data.get("type") or data.get("name") or data.get("tool") or "")
    args = data.get("args") or data.get("arguments") or data.get("parameters") or data
    if isinstance(args, str):
        try:
            args = _json.loads(args)
        except ValueError:
            args = {}
    args = args if isinstance(args, dict) else {}
    if not name and "reason" in args:
        name = "end_call"
    return (name or None), args


def make_reply_guard(state: CallState, label: str = ""):
    """Sits between the AI and the voice so a tool call the AI WRITES as text is never spoken.

    Models sometimes put the call in their reply instead of calling the tool, at the start
    ('{"reason": "completed"}') or after a goodbye ('Bye! {"type":"end_call","args":{...}}').
    From the first '{', '[' or ``` the text is held back:
      - it is an end_call with a valid reason and the candidate is ending the call: hang up
        (with a goodbye only if none was spoken)
      - other tool text: dropped silently (the reply watchdog asks the AI again if nothing was said)
      - it wasn't a tool call after all (a brace in a normal sentence): released and spoken.
    Text before the brace plays without delay."""
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        TTSSpeakFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    try:
        from pipecat.frames.frames import EndWorkerFrame as _End
    except ImportError:  # older Pipecat
        from pipecat.frames.frames import EndFrame as _End

    OPENERS = ("{", "[", "```")

    class ReplyGuard(FrameProcessor):
        def __init__(self):
            super().__init__()
            self._holding = False
            self._held = ""
            self._spoken = ""

        async def _say(self, text: str, direction):
            if text:
                self._spoken += text
                await self.push_frame(LLMTextFrame(text), direction)

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if direction != FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)
                return
            if isinstance(frame, LLMFullResponseStartFrame):
                self._holding, self._held, self._spoken = False, "", ""
                await self.push_frame(frame, direction)
            elif isinstance(frame, LLMTextFrame):
                text = frame.text or ""
                if self._holding:
                    self._held += text
                    return
                cut = min([i for i in (text.find(o) for o in OPENERS) if i >= 0], default=-1)
                if cut < 0:
                    self._spoken += text
                    await self.push_frame(frame, direction)
                    return
                await self._say(text[:cut], direction)  # e.g. "Thanks, goodbye! " plays at once
                self._holding, self._held = True, text[cut:]
            elif isinstance(frame, LLMFullResponseEndFrame):
                if self._holding:
                    await self._resolve(self._held, direction)
                self._holding, self._held = False, ""
                await self.push_frame(frame, direction)
            else:
                await self.push_frame(frame, direction)

        async def _resolve(self, held: str, direction):
            looks_like_call = '":' in held or held.lstrip().startswith("```")
            if not looks_like_call:
                await self._say(held, direction)  # "Okay {thanks}" - a normal sentence after all
                return
            state.record("tool_text_leak", text=held.strip()[:120])
            name, args = _leaked_call(held)
            reason = str(args.get("reason", "")) if name == "end_call" else ""
            if reason in END_REASONS and not state.ended_by and not guard_end_call(state, reason):
                reason = apply_end_call(state, {"reason": reason})
                logger.info(f"[{label}]   [action] end_call ({reason}) (written as text: hanging up, not spoken)")
                said_goodbye = bool(self._spoken.strip())
                line = closing_line(reason)
                if line and not said_goodbye:
                    await self.push_frame(TTSSpeakFrame(line))
                await self.push_frame(_End())
            else:
                logger.warning(f"[{label}]   [guard] AI wrote a tool call as text; not spoken: {held.strip()[:80]!r}")

    return ReplyGuard()


INTERRUPTED_MARK = "(You were cut off"


def make_interruption_note(state: CallState, label: str = ""):
    """When the candidate speaks over the agent, the agent's half-finished sentence is still in the
    conversation and the AI tends to answer the candidate AND repeat its unfinished question in one
    breath. This adds a one-turn note so the AI drops the unfinished thought, the way a person would."""
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    note = (f"{INTERRUPTED_MARK} mid-sentence: the candidate started talking while you were speaking. Drop "
            "whatever you were saying or asking. Reply ONLY to what they just said. If you couldn't make sense "
            "of it, say something like \"Sorry, we spoke at the same time, what were you saying?\" Bring back "
            "your earlier question later only if it still matters, in a later turn.)")

    class InterruptionNote(FrameProcessor):
        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
                try:
                    msgs = [m for m in frame.context.get_messages()
                            if not str((m or {}).get("content", "")).startswith(INTERRUPTED_MARK)]
                    if (state.bot_interrupted and msgs and msgs[-1].get("role") == "user"
                            and not msgs[-1].get("tool_call_id")):
                        msgs.insert(len(msgs) - 1, {"role": "system", "content": note})
                        state.bot_interrupted = False
                        logger.info(f"[{label}]   [turn] agent was cut off: dropping its unfinished question")
                    frame.context.set_messages(msgs)
                except Exception as e:  # noqa: BLE001  never block a reply
                    logger.warning(f"[{label}] interruption note failed: {e}")
            await self.push_frame(frame, direction)

    return InterruptionNote()


# Server-wide counters shown by GET /status (scripts/server_status.py, load_test.py).

LLM_ERROR_STATS = {"total": 0, "last": ""}



LLM_ERROR_LINES = {

    # First failure: the automatic retry has already been tried; asking to repeat gives

    # Gemini a fresh chance and sounds like a normal phone glitch.

    "repeat": "Sorry, I didn't catch that. Could you please say that again?",

    # Second failure in a row (or a failure that won't fix itself): close politely.

    "give_up": "I'm sorry, I'm having trouble with my line. A recruiter will call you back. Thank you!",

}





def attach_llm_error_handler(worker, llm, state: CallState, label: str = ""):

    """Priya never goes silent when Gemini fails (rate limit, outage, timeout).



    1st failure in a row: she asks the candidate to repeat (Gemini usually recovers).

    2nd failure in a row, or an error that will keep happening (bad key, bad request):

    she apologises, says a recruiter will call back, and hangs up. The call log and

    Sheet row then show "Ended By: error" so the candidate can be called again."""

    from pipecat.frames.frames import EndFrame, TTSSpeakFrame



    from app.gemini_rest import describe_error



    prefix = f"[{label}] " if label else ""



    @worker.event_handler("on_pipeline_error")

    async def on_error(_worker, frame):

        if getattr(frame, "processor", None) is not llm or state.ended_by:

            return

        state.llm_errors += 1

        state.llm_errors_in_row += 1

        problem = describe_error(str(getattr(frame, "error", "")))

        LLM_ERROR_STATS["total"] += 1

        LLM_ERROR_STATS["last"] = f"{time.strftime('%H:%M:%S')} {problem}"

        lasting = not getattr(llm, "is_usable", True)  # e.g. invalid request: no point retrying

        logger.warning(f"{prefix}AI ({type(llm).__name__}) failed ({state.llm_errors_in_row} in a row): {problem}")

        state.record("llm_error", problem=problem)

        if state.llm_errors_in_row == 1 and not lasting:

            await worker.queue_frames([TTSSpeakFrame(LLM_ERROR_LINES["repeat"])])

            return

        state.end_reason = state.end_reason or "other"

        state.ended_by = "error"

        state.record("end_call", reason=f"technical problem: {problem}")

        await worker.queue_frames([TTSSpeakFrame(LLM_ERROR_LINES["give_up"]), EndFrame()])





def make_call_state_observer(state: CallState):

    """Pipecat observer that tracks when the candidate and Priya last spoke

    (end_call uses it to decide whether a fallback goodbye is needed)."""

    from pipecat.frames.frames import (

        InterruptionFrame,

        LLMFullResponseEndFrame,

        LLMFullResponseStartFrame,

        BotStartedSpeakingFrame,

        BotStoppedSpeakingFrame,

        LLMTextFrame,

        TranscriptionFrame,

        TTSSpeakFrame,

        UserStartedSpeakingFrame,

        UserStoppedSpeakingFrame,

    )

    from pipecat.observers.base_observer import BaseObserver, FramePushed



    seen: set[int] = set()  # observers see each frame once per hop; count it once



    class _Observer(BaseObserver):

        async def on_push_frame(self, data: FramePushed):

            f = data.frame

            if isinstance(f, UserStartedSpeakingFrame):

                state.user_speaking = True

            elif isinstance(f, UserStoppedSpeakingFrame):

                state.user_speaking = False

            elif isinstance(f, InterruptionFrame):

                if state.bot_speaking:

                    state.bot_interrupted = True

            elif isinstance(f, BotStartedSpeakingFrame):

                state.bot_speaking = True

            elif isinstance(f, BotStoppedSpeakingFrame):

                state.bot_speaking = False

            elif isinstance(f, TranscriptionFrame):

                if id(f) not in seen and (f.text or "").strip():

                    seen.add(id(f))

                    if len(seen) > 500:

                        seen.clear()

                    state.user_turns += 1

                    state.recent_user = (state.recent_user + [f.text.strip()])[-6:]

                state.last_user_text_at = time.monotonic()

                if not state.bot_speaking:

                    state.last_user_turn_at = state.last_user_text_at

                state.silence_prompts = 0  # candidate is back

            elif isinstance(f, LLMFullResponseStartFrame):

                if not state.llm_busy:

                    state.llm_busy, state.llm_started_at = True, time.monotonic()

            elif isinstance(f, LLMFullResponseEndFrame):

                state.llm_busy = False

            elif isinstance(f, (LLMTextFrame, TTSSpeakFrame)):

                state.last_bot_text_at = time.monotonic()

                if isinstance(f, LLMTextFrame):

                    state.llm_errors_in_row = 0  # Gemini is answering again



    return _Observer() 