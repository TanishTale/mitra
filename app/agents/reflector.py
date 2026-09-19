"""Reflector Agent — self-critique before delivery.

This closes the agentic loop: generate, evaluate against an explicit rubric,
repair. It runs a deterministic rubric first (cheap, always available) and, when
a model backend is configured, a second model-based pass. A reply that fails the
hard safety screen is never shown; it is replaced, not patched.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .. import safety
from ..schemas import Critique, RiskLevel
from .base import Agent

REFLECTOR_PERSONA = """You are the Reflector unit. You do not speak to the user.
You receive a draft reply and must audit it against this rubric:
- no diagnosis, no medication or dosage advice
- no method or means information of any kind
- does not discourage professional or family support
- reflects the user's feeling before advising
- at most one question, at most one suggestion
- 60-200 words, warm, plain language, not preachy
Return ONLY JSON: {"approved":true|false,"violations":["..."],"revised_reply":"..."}"""

CLICHES = [
    "i'm sorry you're going through this",
    "i'm sorry to hear that",
    "everything happens for a reason",
    "just stay positive",
    "others have it worse",
    "you should be grateful",
    "calm down",
    "don't worry",
    "it could be worse",
    "cheer up",
]


class ReflectorAgent(Agent):
    name = "ReflectorAgent"
    role = "Self-critique and safety repair"
    persona = REFLECTOR_PERSONA

    def rubric(self, reply: str, state: Dict[str, Any]) -> Critique:
        violations: List[str] = []

        ok, unsafe = safety.screen_reply(reply)
        violations.extend(unsafe)

        words = len(reply.split())
        if words > 230:
            violations.append("too long")
        if words < 25:
            violations.append("too short to be supportive")
        if reply.count("?") > 2:
            violations.append("more than one question")
        low = reply.lower()
        for phrase in CLICHES:
            if phrase in low:
                violations.append(f"dismissive cliché: '{phrase}'")
                break
        if re.search(r"^\s*(1\.|- |\* )", reply, re.M) and \
                state["triage"].intent.value not in {"skill", "track", "information"}:
            violations.append("unsolicited list formatting")

        score = max(0.0, 1.0 - 0.2 * len(violations))
        return Critique(approved=not violations, violations=violations, score=score)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        draft = state.get("draft_reply", "")
        critique = self.rubric(draft, state)
        self.trace("rubric", f"violations={critique.violations or 'none'}")

        hard = [v for v in critique.violations
                if v in {"implied diagnosis", "medication guidance",
                         "discourages professional help", "false identity claim",
                         "unfounded reassurance"}]
        if hard:
            # Hard failure: never attempt a cosmetic patch.
            state["final_reply"] = (
                "Let me take that again. I can't speak to anything medical, and I "
                "don't want to pretend otherwise. What I can do is stay with what "
                "you're feeling right now — tell me a little more about how today "
                "has actually been?"
            )
            state["critique"] = Critique(approved=False, violations=hard, score=0.0)
            self.trace("hard_block", ", ".join(hard))
            return state

        if critique.violations and self.cfg.reflection_enabled and not self.cfg.offline:
            revised = self._model_revise(draft, critique.violations)
            if revised:
                ok, unsafe = safety.screen_reply(revised)
                if ok:
                    state["final_reply"] = revised
                    state["critique"] = Critique(approved=True,
                                                 violations=critique.violations,
                                                 revised_reply=revised,
                                                 score=critique.score)
                    self.trace("revised", "model repair applied")
                    return state

        if critique.violations and self.cfg.offline:
            state["final_reply"] = self._local_repair(draft, critique.violations)
            state["critique"] = critique
            self.trace("revised", "local repair applied")
            return state

        state["final_reply"] = draft
        state["critique"] = critique
        return state

    # -- repair paths -----------------------------------------------------
    def _model_revise(self, draft: str, violations: List[str]) -> str | None:
        try:
            out = self.ask([{
                "role": "user",
                "content": f"DRAFT:\n{draft}\n\nVIOLATIONS: {violations}\n"
                           f"Rewrite the draft so it passes. Reply with the rewritten "
                           f"message only, no JSON, no commentary.",
            }])
            return out.strip() or None
        except Exception:
            return None

    @staticmethod
    def _local_repair(draft: str, violations: List[str]) -> str:
        text = draft
        for phrase in CLICHES:
            text = re.sub(re.escape(phrase) + r"[.,!]?\s*", "", text, flags=re.I)
        if "more than one question" in violations:
            parts = text.split("?")
            if len(parts) > 2:
                text = parts[0] + "?" + " ".join(p for p in parts[1:-1] if False) + \
                       parts[-1]
        if "too long" in violations:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            text = " ".join(sentences[:8])
        return text.strip()
