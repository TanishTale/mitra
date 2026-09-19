"""Summariser Agent — compresses long conversations into a carry-forward note.

Without this, long sessions either blow the context window or lose the thread.
The summary is stored per session and re-injected as context, giving MITRA
episodic continuity across turns and across app restarts.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .base import Agent

SUMMARISER_PERSONA = """You are the Summariser unit. You do not speak to the user.
Compress the conversation into at most 70 words of neutral, factual notes a
support worker could read at a glance: what the person is dealing with, what
they have tried, what helped, what to avoid. No diagnosis, no speculation, no
verbatim quotes of distressing content."""


class SummariserAgent(Agent):
    name = "SummariserAgent"
    role = "Long-term memory compression"
    persona = SUMMARISER_PERSONA

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history: List[Dict[str, str]] = state.get("history", [])
        if len(history) < 4:
            return state

        if self.cfg.offline:
            summary = self._extractive(history, state)
        else:
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-20:])
            summary = self.ask([{"role": "user", "content": transcript}]).strip()
            if len(summary.split()) > 90:
                summary = " ".join(summary.split()[:90])

        state["summary"] = summary
        state["store"].set_summary(state["session_id"], summary)
        self.trace("summarise", f"{len(summary.split())} words")
        return state

    @staticmethod
    def _extractive(history: List[Dict[str, str]], state: Dict[str, Any]) -> str:
        user_text = " ".join(m["content"] for m in history if m["role"] == "user")
        words = [w.strip(".,!?'\"").lower() for w in user_text.split() if len(w) > 5]
        stop = {"really", "because", "myself", "always", "little", "something",
                "anything", "nothing", "everyone", "feeling", "though"}
        common = [w for w, _ in Counter(words).most_common(12) if w not in stop][:6]
        emotions = state.get("triage").emotions if state.get("triage") else []
        moods = state["store"].moods(state["session_id"], limit=5)
        mood_note = (f"Latest self-rated mood {moods[-1]['score']}/10. " if moods else "")
        return (f"Recurring themes: {', '.join(common) or 'general low mood'}. "
                f"Predominant feelings: {', '.join(emotions) or 'unclear'}. "
                f"{mood_note}"
                f"{len([m for m in history if m['role'] == 'user'])} user turns so far.")
