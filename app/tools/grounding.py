"""Coping-skill tools: paced breathing, sensory grounding, CBT reframing.

All content here is psychoeducational self-help material of the kind published
by public health bodies. None of it is treatment, and none of it is presented
as a substitute for care.
"""

from __future__ import annotations

import re
from typing import Dict, List

from ..schemas import ToolResult

BREATHING = {
    "box": [
        "Sit with both feet on the floor and let your shoulders drop.",
        "Breathe in through your nose for a count of four.",
        "Hold gently for four.",
        "Breathe out through your mouth for four.",
        "Hold empty for four. That's one box. Do four boxes.",
    ],
    "478": [
        "Rest the tip of your tongue behind your top teeth.",
        "Breathe in quietly through the nose for four.",
        "Hold for seven.",
        "Breathe out through the mouth for eight, slowly.",
        "Repeat three more times. The long out-breath is the part that calms you.",
    ],
    "physiological_sigh": [
        "Take one normal breath in through your nose.",
        "On top of it, sneak in a second short sip of air.",
        "Let it all out slowly through your mouth, longer than the in-breath.",
        "Two or three of these can drop the edge off panic within a minute.",
    ],
}

GROUNDING = {
    "54321": [
        "Name five things you can see right now.",
        "Four things you can physically feel — chair, fabric, floor, air.",
        "Three things you can hear.",
        "Two things you can smell.",
        "One thing you can taste, or one slow breath instead.",
    ],
    "temperature": [
        "Hold something cool — a bottle, a steel tumbler, your wrists under a tap.",
        "Notice exactly where the cold starts and stops on your skin.",
        "Stay with that for thirty seconds before you go back to the thought.",
    ],
    "orienting": [
        "Say today's date and where you are, out loud if you can.",
        "Turn your head slowly and let your eyes land on three separate objects.",
        "Remind yourself: this feeling is intense, and it is also temporary.",
    ],
}

# Beck / Burns cognitive distortion patterns.
DISTORTIONS: Dict[str, Dict[str, object]] = {
    "all_or_nothing": {
        "patterns": [r"\balways\b", r"\bnever\b", r"\bevery ?time\b", r"\bcompletely\b",
                     r"\btotal (failure|disaster)\b"],
        "label": "all-or-nothing thinking",
        "question": "Is there a middle version of this that's also true?",
    },
    "catastrophising": {
        "patterns": [r"\bruined\b", r"\bdisaster\b", r"\bover for me\b", r"\bend of\b",
                     r"\bworst\b"],
        "label": "catastrophising",
        "question": "What is the most likely outcome, as opposed to the worst one?",
    },
    "mind_reading": {
        "patterns": [r"\bthey (think|must think)\b", r"\beveryone (thinks|knows)\b",
                     r"\bhe (thinks|hates)\b", r"\bshe (thinks|hates)\b"],
        "label": "mind reading",
        "question": "What would you actually need to hear from them to know this?",
    },
    "labelling": {
        "patterns": [r"\bi(?:'m| am) (a )?(failure|loser|idiot|useless|worthless|stupid)\b"],
        "label": "labelling",
        "question": "Is that a description of one event, or a verdict on a whole person?",
    },
    "should_statements": {
        "patterns": [r"\bi should(?:'ve| have)?\b", r"\bi must\b", r"\bi have to be\b"],
        "label": "'should' statements",
        "question": "Whose standard is that, and would you apply it to a friend?",
    },
    "overgeneralisation": {
        "patterns": [r"\bnothing ever\b", r"\bno ?one ever\b", r"\bit's always like this\b"],
        "label": "overgeneralisation",
        "question": "Can you think of one occasion that didn't go this way?",
    },
    "personalisation": {
        "patterns": [r"\bmy fault\b", r"\bbecause of me\b", r"\bi ruined\b"],
        "label": "personalisation",
        "question": "What else contributed that had nothing to do with you?",
    },
}


def breathing_exercise(style: str = "box") -> ToolResult:
    style = style if style in BREATHING else "box"
    steps: List[str] = BREATHING[style]
    return ToolResult(
        name="breathing_exercise", ok=True,
        output={"style": style, "steps": steps},
        display="Paced breathing (" + style + "): " + " ".join(steps),
    )


def grounding_exercise(technique: str = "54321") -> ToolResult:
    technique = technique if technique in GROUNDING else "54321"
    steps = GROUNDING[technique]
    return ToolResult(
        name="grounding_exercise", ok=True,
        output={"technique": technique, "steps": steps},
        display="Grounding (" + technique + "): " + " ".join(steps),
    )


def detect_distortions(thought: str) -> List[Dict[str, str]]:
    found = []
    for key, spec in DISTORTIONS.items():
        for pattern in spec["patterns"]:                      # type: ignore[index]
            if re.search(pattern, thought, re.I):
                found.append({"key": key, "label": spec["label"],      # type: ignore[index]
                              "question": spec["question"]})           # type: ignore[index]
                break
    return found


def _balanced_alternative(thought: str, labels: List[str]) -> str:
    if "labelling" in labels:
        return ("One outcome went badly. That is a thing that happened, not a "
                "description of who you are.")
    if "all-or-nothing thinking" in labels:
        return ("Some of this went wrong and some of it didn't. Both parts are "
                "allowed to be true at once.")
    if "catastrophising" in labels:
        return ("This is a setback with real consequences, and it is also survivable "
                "and probably recoverable.")
    if "mind reading" in labels:
        return ("You are guessing at what someone else thinks. Until they tell you, "
                "it's a hypothesis, not a fact.")
    return ("A fairer version might be: this is hard right now, and that doesn't "
            "settle anything permanent about you.")


def cbt_reframe(thought: str) -> ToolResult:
    found = detect_distortions(thought or "")
    labels = [f["label"] for f in found]
    questions = [f["question"] for f in found][:2] or [
        "What evidence do you have for that thought, and what evidence sits against it?"
    ]
    alternative = _balanced_alternative(thought, labels)
    display = (
        ("I noticed a pattern here: " + ", ".join(labels) + ". " if labels else "")
        + " ".join(questions) + " " + alternative
    )
    return ToolResult(
        name="cbt_reframe", ok=True,
        output={"distortions": found, "questions": questions, "alternative": alternative},
        display=display,
    )
