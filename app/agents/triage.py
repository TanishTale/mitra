"""Triage Agent — risk assessment, emotion tagging and intent classification.

Runs first on every turn and has veto power over the rest of the graph. It
combines a deterministic safety scorer (`app.safety`) with lightweight
lexical emotion/intent detection. When an LLM backend is configured the agent
may *raise* the risk band on the model's advice, but it can never lower the
deterministic score — an important asymmetry, because a jailbroken or
hallucinating model must not be able to talk the system out of escalating.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .. import safety
from ..llm import strip_json
from ..schemas import Intent, RiskLevel, TriageResult
from .base import Agent

EMOTION_LEXICON: Dict[str, List[str]] = {
    "anxiety": ["anxious", "nervous", "panic", "panicking", "worry", "worried",
                "scared", "afraid", "tense", "on edge", "dread", "restless",
                "overthinking", "heart is racing", "racing", "shaking"],
    "sadness": ["sad", "down", "low", "depressed", "unhappy", "crying", "cry", "tears",
                "miserable", "empty", "blue"],
    "anger": ["angry", "furious", "irritated", "annoyed", "frustrated", "rage",
               "unfair", "resent"],
    "loneliness": ["lonely", "alone", "isolated", "no one", "nobody", "left out",
                    "ignored", "invisible"],
    "overwhelm": ["overwhelmed", "too much", "a lot", "drowning", "pressure",
                   "deadline", "deadlines", "submissions", "workload",
                   "can't keep up", "piling up", "juggling"],
    "shame": ["ashamed", "embarrassed", "guilty", "failure", "worthless", "useless",
               "stupid", "hate myself", "not good enough", "disappointment"],
    "grief": ["died", "passed away", "loss", "lost him", "lost her", "funeral",
               "miss them", "grief"],
    "fatigue": ["tired", "exhausted", "drained", "no energy", "burnt out",
                 "burned out", "can't sleep", "cant sleep", "insomnia",
                 "haven't slept", "havent slept", "not slept", "sleep properly"],
}

INTENT_CUES: Dict[Intent, List[str]] = {
    Intent.TRACK: ["log my mood", "track", "rate my", "out of 10", "journal",
                    "write it down", "note this", "how have i been"],
    Intent.SKILL: ["what can i do", "help me calm", "calm down", "technique",
                    "exercise", "breathing", "ground me", "how do i stop", "tips",
                    "cope", "settle down"],
    Intent.REFRAME: ["i always", "i never", "i am a failure", "i'm a failure",
                      "everyone thinks", "my fault", "i should have", "i'm useless"],
    Intent.SCREENING: ["questionnaire", "self check", "self-check", "assessment",
                        "phq", "gad", "am i depressed", "test me"],
    Intent.INFORMATION: ["what is", "why do i", "how does", "explain", "is it normal"],
    Intent.SMALLTALK: ["hello", "good morning", "good evening", "how are you",
                        "thanks", "thank you", "goodbye"],
    Intent.OUT_OF_SCOPE: ["write my code", "homework", "essay for", "legal advice",
                           "loan", "recipe", "cricket score"],
}

TRIAGE_PERSONA = """You are the Triage unit. You do not speak to the user.
Read the latest user turn and return ONLY a JSON object:
{"risk_level":"supportive|elevated|crisis","risk_score":0.0-1.0,
 "intent":"vent|reframe|skill|track|information|screening|smalltalk|out_of_scope",
 "emotions":["..."],"rationale":"one short sentence"}
Be conservative: when in doubt, choose the higher risk band. No prose, no fences."""


class TriageAgent(Agent):
    name = "TriageAgent"
    role = "Risk assessment and routing"
    persona = TRIAGE_PERSONA

    # -- lexical passes ---------------------------------------------------
    @staticmethod
    def _cue_hits(text: str, cues: List[str]) -> int:
        """Count cue matches on word boundaries.

        Naive substring matching is a real bug here, not a nicety: "down" is
        inside "download", "hi" is inside "this", and an emotion mislabelled at
        this stage propagates through every downstream agent.
        """
        return sum(1 for c in cues
                   if re.search(r"\b" + re.escape(c) + r"\b", text, re.I))

    @staticmethod
    def detect_emotions(text: str) -> List[str]:
        low = text.lower()
        scored = []
        for emotion, cues in EMOTION_LEXICON.items():
            hits = TriageAgent._cue_hits(low, cues)
            if hits:
                scored.append((hits, emotion))
        scored.sort(reverse=True)
        return [e for _, e in scored[:3]]

    @staticmethod
    def detect_intent(text: str, emotions: List[str]) -> Intent:
        low = text.lower().strip()
        if len(low.split()) <= 4 and any(
            low.startswith(c) for c in INTENT_CUES[Intent.SMALLTALK]
        ):
            return Intent.SMALLTALK
        best, best_hits = Intent.VENT, 0
        for intent, cues in INTENT_CUES.items():
            hits = TriageAgent._cue_hits(low, cues)
            if hits > best_hits:
                best, best_hits = intent, hits
        if best_hits == 0:
            return Intent.VENT if emotions else Intent.SMALLTALK
        return best

    # -- main -------------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text: str = state["user_text"]
        history: List[str] = state.get("prior_user_turns", [])

        assessment = safety.assess(
            text, history,
            crisis_threshold=self.cfg.crisis_threshold,
            elevated_threshold=self.cfg.elevated_threshold,
        )
        emotions = self.detect_emotions(text)
        intent = self.detect_intent(text, emotions)
        self.trace("rule_triage",
                   f"score={assessment.score} level={assessment.level.value} "
                   f"intent={intent.value}")

        # Optional model-assisted second opinion (escalate-only).
        if not self.cfg.offline:
            raised = self._model_second_opinion(text, assessment)
            if raised:
                assessment = raised

        needs_tool = intent in {Intent.TRACK, Intent.SKILL, Intent.REFRAME,
                                Intent.SCREENING} or assessment.level == RiskLevel.CRISIS

        result = TriageResult(
            risk_level=assessment.level,
            risk_score=assessment.score,
            intent=intent,
            emotions=emotions or ["neutral"],
            signals=assessment.signals,
            needs_tool=needs_tool,
            rationale=assessment.blocked_request or "lexical + contextual scoring",
        )
        state["triage"] = result
        state["blocked_request"] = assessment.blocked_request
        return state

    def _model_second_opinion(self, text: str, current) -> Any:
        try:
            raw = self.ask([{"role": "user", "content": text}])
            data = json.loads(strip_json(raw))
            model_score = float(data.get("risk_score", 0))
            if model_score > current.score:
                level = RiskLevel(data.get("risk_level", current.level.value))
                self.trace("model_escalation",
                           f"{current.score} -> {model_score}")
                return safety.RiskAssessment(
                    level=level if level != RiskLevel.SUPPORTIVE else current.level,
                    score=model_score,
                    signals=current.signals + ["model-assisted escalation"],
                    blocked_request=current.blocked_request,
                )
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            self.trace("model_second_opinion", "unparseable — rule score retained")
        return None


def extract_number(text: str) -> int | None:
    m = re.search(r"\b([1-9]|10)\b", text)
    return int(m.group(1)) if m else None
