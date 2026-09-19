"""Mood logging and trend analysis."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any

from ..schemas import ToolResult

MOOD_WORDS = {
    10: ("great", "amazing", "wonderful"), 9: ("really good", "excellent"),
    8: ("good", "happy", "positive"), 7: ("okay-ish", "fine"),
    6: ("alright", "so-so"), 5: ("neutral", "flat"),
    4: ("low", "down"), 3: ("bad", "rough"),
    2: ("very low", "awful"), 1: ("terrible", "at my worst"),
}


def parse_mood(text: str) -> int | None:
    """Extract a 1-10 self-rating from free text ('about a 4 today')."""
    m = re.search(r"\b(?:a |at |around |about )?([1-9]|10)\s*(?:/|out of|on)?\s*(?:10)?\b", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 10:
            return val
    for score, words in MOOD_WORDS.items():
        if any(w in text.lower() for w in words):
            return score
    return None


def label_for(score: int) -> str:
    return MOOD_WORDS.get(max(1, min(10, score)), ("unspecified",))[0]


def log_mood(store: Any, session_id: str, score: int, label: str = "",
             note: str = "") -> ToolResult:
    score = int(max(1, min(10, score)))
    label = label or label_for(score)
    store.log_mood(session_id, score, label, note)
    return ToolResult(
        name="log_mood", ok=True,
        output={"score": score, "label": label},
        display=f"Logged today's mood as {score}/10 ({label}).",
    )


def mood_trend(store: Any, session_id: str) -> ToolResult:
    entries = store.moods(session_id, limit=30)
    if not entries:
        return ToolResult(name="mood_trend", ok=True, output={"count": 0},
                          display="No mood entries logged yet.")
    scores = [e["score"] for e in entries]
    avg = round(mean(scores), 1)
    if len(scores) >= 3:
        first, last = mean(scores[: max(1, len(scores) // 2)]), mean(scores[len(scores) // 2:])
        delta = last - first
        direction = "improving" if delta > 0.6 else "declining" if delta < -0.6 else "steady"
    else:
        direction = "too early to call"
    lowest, highest = min(scores), max(scores)
    plural = "check-in" if len(scores) == 1 else "check-ins"
    display = (f"Across {len(scores)} {plural} your average is {avg}/10 "
               f"(low {lowest}, high {highest}) and the trend looks {direction}.")
    return ToolResult(
        name="mood_trend", ok=True,
        output={"count": len(scores), "average": avg, "direction": direction,
                "lowest": lowest, "highest": highest, "series": scores},
        display=display,
    )
