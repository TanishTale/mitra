"""Tool registry.

Tools are plain callables with a JSON-serialisable signature. Keeping them
side-effect-scoped (they receive the store and the session id explicitly) means
a tool can be unit-tested without spinning up the agent graph — which is how
`tests/test_tools.py` works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from ..schemas import ToolResult
from . import grounding, journal, mood, resources, screening


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]
    fn: Callable[..., ToolResult]
    safe_in_crisis: bool = False    # may this run while risk == CRISIS?


REGISTRY: Dict[str, Tool] = {}


def register(tool: Tool) -> None:
    REGISTRY[tool.name] = tool


def specs() -> List[Dict[str, Any]]:
    """Machine-readable catalogue handed to the planner agent."""
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in REGISTRY.values()
    ]


def call(name: str, **kwargs) -> ToolResult:
    tool = REGISTRY.get(name)
    if not tool:
        return ToolResult(name=name, ok=False, error="unknown tool",
                          display="(unknown tool requested)")
    try:
        return tool.fn(**kwargs)
    except TypeError as exc:
        return ToolResult(name=name, ok=False, error=f"bad arguments: {exc}",
                          display="(tool call skipped)")
    except Exception as exc:                      # defensive: a tool must never
        return ToolResult(name=name, ok=False,    # crash the conversation
                          error=f"{type(exc).__name__}: {exc}",
                          display="(tool unavailable)")


# --- registrations --------------------------------------------------------

register(Tool(
    name="log_mood",
    description="Record the user's self-rated mood (1-10) with an optional note.",
    parameters={"store": "MemoryStore", "session_id": "str", "score": "int 1-10",
                "label": "str", "note": "str"},
    fn=mood.log_mood,
))

register(Tool(
    name="mood_trend",
    description="Summarise the user's recent mood entries and the direction of travel.",
    parameters={"store": "MemoryStore", "session_id": "str"},
    fn=mood.mood_trend,
))

register(Tool(
    name="breathing_exercise",
    description="Return a guided paced-breathing script (box or 4-7-8).",
    parameters={"style": "'box' | '478' | 'physiological_sigh'"},
    fn=grounding.breathing_exercise,
    safe_in_crisis=True,
))

register(Tool(
    name="grounding_exercise",
    description="Return a sensory grounding technique for panic or dissociation.",
    parameters={"technique": "'54321' | 'temperature' | 'orienting'"},
    fn=grounding.grounding_exercise,
    safe_in_crisis=True,
))

register(Tool(
    name="cbt_reframe",
    description="Detect cognitive distortions in a thought and return Socratic "
                "questions plus a balanced alternative thought.",
    parameters={"thought": "str"},
    fn=grounding.cbt_reframe,
))

register(Tool(
    name="save_journal",
    description="Save a reflective journal entry for the user.",
    parameters={"store": "MemoryStore", "session_id": "str", "entry": "str",
                "prompt": "str"},
    fn=journal.save_journal,
))

register(Tool(
    name="recall_journal",
    description="Search the user's earlier journal entries for a keyword.",
    parameters={"store": "MemoryStore", "session_id": "str", "query": "str"},
    fn=journal.recall_journal,
))

register(Tool(
    name="journal_prompt",
    description="Suggest a reflective writing prompt matched to an emotion.",
    parameters={"emotion": "str"},
    fn=journal.journal_prompt,
))

register(Tool(
    name="self_check",
    description="Run a brief, non-diagnostic wellbeing self-check (PHQ-2 / GAD-2 "
                "style) and interpret it cautiously.",
    parameters={"answers": "list[int] of 0-3", "instrument": "'phq2' | 'gad2'"},
    fn=screening.self_check,
))

register(Tool(
    name="find_support",
    description="Return verified helplines and support services for a region.",
    parameters={"region": "ISO country code, default IN", "kind": "'crisis' | 'campus' | 'all'"},
    fn=resources.find_support,
    safe_in_crisis=True,
))

register(Tool(
    name="sleep_hygiene",
    description="Return evidence-based, non-medical sleep or routine suggestions.",
    parameters={"issue": "'sleep' | 'focus' | 'routine'"},
    fn=resources.wellbeing_tips,
))

__all__ = ["Tool", "REGISTRY", "register", "specs", "call"]
