"""Brief wellbeing self-check.

Important framing, repeated in the UI and the report: these two-item scales are
*conversation starters*, not diagnostic instruments. MITRA never states or
implies a diagnosis from them; it reports the score and, above the standard
cut-off, encourages a conversation with a qualified professional.
"""

from __future__ import annotations

from typing import List

from ..schemas import ToolResult

ITEMS = {
    "phq2": [
        "Over the last two weeks, how often have you had little interest or "
        "pleasure in doing things?",
        "Over the last two weeks, how often have you felt down, hopeless, or "
        "that things were pointless?",
    ],
    "gad2": [
        "Over the last two weeks, how often have you felt nervous, anxious or "
        "on edge?",
        "Over the last two weeks, how often have you been unable to stop or "
        "control worrying?",
    ],
}

SCALE = {0: "not at all", 1: "several days", 2: "more than half the days",
         3: "nearly every day"}

CUTOFF = 3  # the widely used screening cut-off for both PHQ-2 and GAD-2


def self_check(answers: List[int] | None = None, instrument: str = "phq2") -> ToolResult:
    instrument = instrument if instrument in ITEMS else "phq2"
    if not answers:
        return ToolResult(
            name="self_check", ok=True,
            output={"instrument": instrument, "items": ITEMS[instrument], "scale": SCALE},
            display="Two quick questions, each answered 0-3: " +
                    " | ".join(ITEMS[instrument]),
        )

    answers = [max(0, min(3, int(a))) for a in answers][: len(ITEMS[instrument])]
    total = sum(answers)
    above = total >= CUTOFF
    interpretation = (
        "That's above the usual screening cut-off, which simply means it's worth "
        "talking to a counsellor or doctor — it is not a diagnosis and it does not "
        "mean anything is wrong with you."
        if above else
        "That sits below the usual screening cut-off. It doesn't rule anything in or "
        "out; what you feel still counts, whatever a number says."
    )
    return ToolResult(
        name="self_check", ok=True,
        output={"instrument": instrument, "answers": answers, "total": total,
                "cutoff": CUTOFF, "above_cutoff": above, "diagnostic": False},
        display=f"Your {instrument.upper()} self-check total is {total} out of 6. "
                f"{interpretation}",
    )
