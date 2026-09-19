"""Listener Agent — the voice of the system.

It is the only agent whose output the user normally sees. Its job is
person-centred: reflect, validate, then (optionally) offer. Any technique
content is handed to it by the Planner as tool output, so the Listener never
invents clinical material on its own.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..schemas import RiskLevel
from .base import Agent

LISTENER_PERSONA = """You are the Listener unit — the voice the user hears.

Follow this shape every turn:
1. Reflect what you heard, in your own words, specific to what they actually said.
2. Validate it without flattery or cheerleading.
3. Only then, if it fits, offer ONE thing: a question, a technique passed to you
   in TOOL OUTPUT, or a gentle observation.

Never stack suggestions. Never start with 'I'm sorry you're going through this'.
Never say 'as an AI' unless asked what you are. If TOOL OUTPUT is provided, weave
it in naturally instead of pasting it. Keep to 90-160 words."""


class ListenerAgent(Agent):
    name = "ListenerAgent"
    role = "Empathic reflection and response generation"
    persona = LISTENER_PERSONA

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        triage = state["triage"]
        tool_results = state.get("tool_results", [])
        tool_summary = " ".join(r.display for r in tool_results if r.ok and r.display)

        messages: List[Dict[str, str]] = []
        if state.get("summary"):
            messages.append({
                "role": "user",
                "content": f"[Context from earlier in this conversation: {state['summary']}]",
            })
        for m in state.get("history", []):
            if m["role"] in {"user", "assistant"}:
                messages.append({"role": m["role"], "content": m["content"]})

        turn = state["user_text"]
        annotations = []
        if triage.emotions:
            annotations.append("detected feelings: " + ", ".join(triage.emotions))
        if tool_summary:
            annotations.append("TOOL OUTPUT you may use: " + tool_summary)
        if triage.risk_level == RiskLevel.ELEVATED:
            annotations.append(
                "RISK NOTE: elevated distress. Be warmer, slower, and mention that "
                "talking to a person — counsellor, friend, family — is worth doing. "
                "Do not sound alarmed."
            )
        if annotations:
            turn = turn + "\n\n[" + " | ".join(annotations) + "]"
        messages.append({"role": "user", "content": turn})

        reply = self.ask(messages, offline_context={
            "user_text": state["user_text"],
            "emotions": triage.emotions,
            "intent": triage.intent.value,
            "tool_summary": tool_summary,
            "recall": state.get("recall_line", ""),
        })
        state["draft_reply"] = reply.strip()
        self.trace("draft", f"{len(reply.split())} words")
        return state
