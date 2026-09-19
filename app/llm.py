"""Model access layer.

MITRA is model-agnostic. `LLMClient.complete()` takes a system prompt plus a
message list and returns text, regardless of whether the backend is Anthropic,
OpenAI, Gemini, or the built-in offline engine.

The offline engine matters more than it looks: an evaluator must be able to
clone the repository, run it with no API key and no internet, and still get a
coherent, safe conversation. It is a compositional template engine driven by the
same triage features the LLM path uses, so the agent graph is exercised
identically in both modes.
"""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request
from typing import Dict, List

from .config import settings


# --------------------------------------------------------------------------
# Offline engine
# --------------------------------------------------------------------------

REFLECTIONS = {
    "anxiety": [
        "It sounds like your mind has been running ahead of you, sketching out every "
        "way this could go wrong.",
        "That constant bracing for something bad is exhausting in a way people "
        "underestimate.",
    ],
    "sadness": [
        "There's a heaviness in how you're describing this, and it makes sense that "
        "it's hard to carry.",
        "It sounds like something in you has gone quiet, and that's a lonely place to be.",
    ],
    "anger": [
        "There's real frustration underneath this — it sounds like something felt unfair.",
        "That irritation usually shows up when something you care about got stepped on.",
    ],
    "loneliness": [
        "Being surrounded by people and still feeling unseen is its own kind of tiring.",
        "It sounds like you've been holding a lot of this on your own.",
    ],
    "overwhelm": [
        "That's a lot to be holding at once, and it makes sense that it's blurring "
        "together.",
        "When everything is urgent, nothing feels finishable — that's not a discipline "
        "problem.",
    ],
    "shame": [
        "You're speaking about yourself very harshly, and I notice that.",
        "It sounds like you're measuring yourself against a standard you'd never hold "
        "a friend to.",
    ],
    "grief": [
        "Loss doesn't move in a straight line, and it sounds like today pulled you back.",
    ],
    "fatigue": [
        "Tired in the body is one thing; this sounds more like tired in the self.",
    ],
    "neutral": [
        "Thank you for putting that into words — that's not always easy.",
        "I'm listening. Take it at whatever pace feels right.",
    ],
}

VALIDATIONS = [
    "What you're feeling is a reasonable response to what you've described.",
    "You're not overreacting, and you're not being dramatic about this.",
    "Anyone carrying this for as long as you have would be worn down by it.",
    "It matters that you noticed this in yourself and said it out loud.",
]

INVITATIONS = {
    "vent": [
        "What's the part of it that's been sitting heaviest today?",
        "If you want to keep going, I'm here — what happened just before this hit?",
    ],
    "reframe": [
        "If a close friend said this exact sentence to you, what would you say back?",
        "What's one piece of evidence that doesn't fit that story about yourself?",
    ],
    "skill": [
        "Would it help to try a two-minute grounding exercise together?",
        "Want to try something small that can take the edge off right now?",
    ],
    "track": [
        "Where would you put today on a scale of one to ten?",
        "Want to note this down so we can look at the pattern later?",
    ],
    "information": [
        "Would it help if I explained what usually helps with this?",
    ],
    "screening": [
        "Would you like to try a short self-check questionnaire together?",
    ],
    "smalltalk": [
        "How has your day actually been, under the surface?",
    ],
    "out_of_scope": [
        "Is there something underneath that question that's been bothering you?",
    ],
}

NORMALISERS = {
    "anxiety": "Anxiety is the body preparing for a threat it can't locate — which is "
               "why reasoning with it rarely works, but slowing the breath often does.",
    "overwhelm": "When the load is genuinely too big, the useful move is usually to "
                 "shrink the next step, not to try harder at all of it.",
    "shame": "Self-criticism feels like accountability, but research consistently finds "
             "it lowers follow-through rather than raising it.",
    "loneliness": "Loneliness responds less to the number of people around you and more "
                  "to one interaction where you feel known.",
    "sadness": "Low mood narrows attention to what's missing; that narrowing is a "
               "symptom, not an accurate read of your life.",
}


class OfflineEngine:
    """Deterministic-by-seed, template-composed responder."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def respond(self, *, user_text: str, emotions: List[str], intent: str,
                context: Dict) -> str:
        primary = emotions[0] if emotions else "neutral"
        parts: List[str] = []

        parts.append(self.rng.choice(REFLECTIONS.get(primary, REFLECTIONS["neutral"])))
        parts.append(self.rng.choice(VALIDATIONS))

        if primary in NORMALISERS and intent in {"vent", "reframe", "skill", "information"}:
            parts.append(NORMALISERS[primary])

        if context.get("tool_summary"):
            parts.append(context["tool_summary"])

        if context.get("recall"):
            parts.append(context["recall"])

        parts.append(self.rng.choice(INVITATIONS.get(intent, INVITATIONS["vent"])))
        return " ".join(parts)


# --------------------------------------------------------------------------
# Remote providers
# --------------------------------------------------------------------------

def _post(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class LLMClient:
    def __init__(self, cfg=settings):
        self.cfg = cfg
        self.offline_engine = OfflineEngine()
        self.last_backend = "offline"

    # -- public ----------------------------------------------------------
    def complete(self, system: str, messages: List[Dict[str, str]],
                 *, offline_context: Dict | None = None) -> str:
        """Return assistant text. Falls back to the offline engine on any error."""
        if self.cfg.offline:
            self.last_backend = "offline"
            return self._offline(offline_context or {})
        try:
            if self.cfg.provider == "anthropic":
                text = self._anthropic(system, messages)
            elif self.cfg.provider == "openai":
                text = self._openai(system, messages)
            elif self.cfg.provider == "gemini":
                text = self._gemini(system, messages)
            else:
                raise ValueError(f"unknown provider {self.cfg.provider}")
            self.last_backend = self.cfg.provider
            return text.strip()
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError, OSError):
            # Graceful degradation is a safety property here: a network blip must
            # never leave a distressed user without a response.
            self.last_backend = "offline (fallback)"
            return self._offline(offline_context or {})

    # -- backends ---------------------------------------------------------
    def _offline(self, ctx: Dict) -> str:
        return self.offline_engine.respond(
            user_text=ctx.get("user_text", ""),
            emotions=ctx.get("emotions", []),
            intent=ctx.get("intent", "vent"),
            context=ctx,
        )

    def _anthropic(self, system: str, messages: List[Dict[str, str]]) -> str:
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.cfg.model,
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "system": system,
                "messages": messages,
            },
            {
                "content-type": "application/json",
                "x-api-key": self.cfg.api_key,
                "anthropic-version": "2023-06-01",
            },
            self.cfg.request_timeout,
        )
        return "".join(b.get("text", "") for b in data.get("content", []))

    def _openai(self, system: str, messages: List[Dict[str, str]]) -> str:
        data = _post(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": self.cfg.model,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
                "messages": [{"role": "system", "content": system}] + messages,
            },
            {"content-type": "application/json",
             "authorization": f"Bearer {self.cfg.api_key}"},
            self.cfg.request_timeout,
        )
        return data["choices"][0]["message"]["content"]

    def _gemini(self, system: str, messages: List[Dict[str, str]]) -> str:
        contents = [
            {"role": "user" if m["role"] == "user" else "model",
             "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.cfg.model}:generateContent?key={self.cfg.api_key}",
            {"systemInstruction": {"parts": [{"text": system}]},
             "contents": contents,
             "generationConfig": {"temperature": self.cfg.temperature,
                                  "maxOutputTokens": self.cfg.max_tokens}},
            {"content-type": "application/json"},
            self.cfg.request_timeout,
        )
        return data["candidates"][0]["content"]["parts"][0]["text"]


def strip_json(text: str) -> str:
    """Remove markdown fences so a JSON-only reply can be parsed."""
    return re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
