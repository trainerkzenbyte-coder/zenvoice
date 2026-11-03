"""A phone call by text: same script, same tools, same rules as a live call.

Used by scripts/chat_sim.py (you type as the candidate) and
scripts/test_call_flow.py (an AI plays the candidate).
"""

import datetime as dt
import time

from app import gemini_rest
from app.call_actions import (
    TOOL_SPECS,  # noqa: F401
    active_tool_specs,
    apply_transfer,
    guard_callback,
    guard_end_call,
    CallContext,
    CallState,
    apply_end_call,
    apply_schedule_callback,
    closing_line,
)
from app.config import settings
from app.prompts import CALL_CONNECTED_NOTE, greeting_line, greeting_mode, load_prompt

CALL_STARTED = "(The call has connected. The candidate has picked up. Greet them now.)"
MAX_TOOL_ROUNDS = 4


class TextCall:
    def __init__(self, candidate_name: str = "", now: dt.datetime | None = None,
                 model: str | None = None, job_title: str = "", prompt_template: str | None = None,
                 greeting_cfg: dict | None = None, variables: dict | None = None):
        """prompt_template / greeting_cfg: test unpublished changes from the console (default: the live files).
        variables: extra {{placeholders}} for the test (console "Dynamic Variables")."""
        self.now = now
        self.model = model
        self.state = CallState(context=CallContext(candidate_name=candidate_name, role=job_title))
        self.system = load_prompt(candidate_name=candidate_name, now=now, job_title=job_title,
                                  template=prompt_template, variables=variables, greeting_cfg=greeting_cfg)
        self.user_first = greeting_mode(greeting_cfg) == "user_first"
        # Same opening as a live call: the fixed greeting (config/screening.yaml), no AI wait.
        self.greeting = greeting_line(candidate_name, cfg=greeting_cfg, variables=variables)
        note = CALL_CONNECTED_NOTE if self.greeting else CALL_STARTED
        self.contents: list[dict] = [] if self.user_first else [{"role": "user", "parts": [{"text": note}]}]
        self.transcript: list[dict] = []
        self.ended = False

    async def start(self) -> str:
        if self.user_first:  # the candidate speaks first: nothing to say yet
            return ""
        if not self.greeting:  # greeting.enabled: false -> the AI writes it
            return await self._bot_turn()
        self.contents.append({"role": "model", "parts": [{"text": self.greeting}]})
        self.transcript.append({"role": "bot", "text": self.greeting, "t": time.time()})
        self.state.last_bot_text_at = time.monotonic()
        return self.greeting

    async def reply(self, user_text: str) -> str:
        if self.ended:
            return ""
        parts = [{"text": user_text}]
        from app import kb_search

        if kb_search.retrieve_mode():  # same per-turn lookup as a live call
            asked = next((t["text"] for t in reversed(self.transcript) if t.get("role") == "bot"), "")
            note = kb_search.note_for([{"role": "assistant", "content": asked}, {"role": "user", "content": user_text}])
            if note:
                parts = [{"text": note}, {"text": user_text}]
        self.contents.append({"role": "user", "parts": parts})
        self.transcript.append({"role": "user", "text": user_text, "t": time.time()})
        self.state.last_user_text_at = time.monotonic()
        self.state.user_turns += 1
        self.state.recent_user = (self.state.recent_user + [user_text])[-6:]
        return await self._bot_turn()

    async def _bot_turn(self) -> str:
        llm = settings.get("llm", {})
        spoken: list[str] = []
        for _ in range(MAX_TOOL_ROUNDS):
            parts = await gemini_rest.generate_parts(
                self.contents, system=self.system, model=self.model, tools=active_tool_specs(),
                temperature=llm.get("temperature", 0.4),
                max_tokens=max(200, int(llm.get("max_output_tokens", 120))),
            )
            self.contents.append({"role": "model", "parts": parts})
            text = gemini_rest.text_of(parts)
            if text:
                spoken.append(text)
                self.state.last_bot_text_at = time.monotonic()
            calls = gemini_rest.calls_of(parts)
            if not calls:
                break

            responses, end_reason = [], None
            for call in calls:
                if call["name"] == "schedule_callback" and guard_callback(self.state):
                    result = {"status": "rejected", "message": guard_callback(self.state)}
                elif call["name"] == "end_call" and guard_end_call(self.state, str(call["args"].get("reason", "other"))):
                    result = {"status": "not_ended", "message": guard_end_call(self.state, str(call["args"].get("reason", "other")))}
                elif call["name"] == "schedule_callback":
                    result = apply_schedule_callback(self.state, call["args"], now=self.now)
                elif call["name"] == "transfer_call":
                    result = apply_transfer(self.state, call["args"])
                    if result["status"] == "transferring":  # text test: show the hand-off, then stop
                        from app import transfer_cfg as transfer
                        self.state.ended_by, self.state.end_reason = "transfer", "transferred"
                        result = {"status": "transferring", "spoken": transfer.cfg()["announce"]}
                        end_reason = "transferred"
                elif call["name"] == "search_knowledge":
                    from app.call_actions import search_knowledge_result

                    result = search_knowledge_result(str(call["args"].get("query", "")))
                elif call["name"] == "end_call":
                    end_reason = apply_end_call(self.state, call["args"])
                    result = {"status": "ending call"}
                else:
                    result = {"status": "error", "message": f"unknown tool {call['name']}"}
                responses.append({"functionResponse": {"name": call["name"], "response": result}})
            self.contents.append({"role": "user", "parts": responses})

            if end_reason:  # end_call: no further model turn (same as run_llm=False on calls)
                line = closing_line(end_reason)
                if line and not spoken:
                    spoken.append(line)
                self.ended = True
                break

        bot_text = " ".join(spoken).strip()
        if bot_text:
            self.transcript.append({"role": "bot", "text": bot_text, "t": time.time()})
        return bot_text
