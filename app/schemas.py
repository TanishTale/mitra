"""Typed contracts exchanged between the orchestrator, agents and tools."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Triage bands. The orchestrator routes on this value alone."""

    SUPPORTIVE = "supportive"   # ordinary low mood, stress, venting
    ELEVATED = "elevated"       # distress, hopelessness, burnout signals
    CRISIS = "crisis"           # self-harm / suicidal ideation / harm to others


class Intent(str, Enum):
    VENT = "vent"                     # wants to be heard
    REFRAME = "reframe"               # stuck in a negative thought loop
    SKILL = "skill"                   # wants a coping technique
    TRACK = "track"                   # log / review mood or journal
    INFORMATION = "information"       # asking about wellbeing topics
    SCREENING = "screening"           # wants a self-check questionnaire
    SMALLTALK = "smalltalk"
    OUT_OF_SCOPE = "out_of_scope"     # medical, legal, homework, etc.


class Message(BaseModel):
    role: str                          # "user" | "assistant" | "system"
    content: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TriageResult(BaseModel):
    """Output of the Triage Agent — the gate every turn must pass through."""

    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    intent: Intent
    emotions: List[str] = []
    signals: List[str] = []            # human-readable reasons for the score
    needs_tool: bool = False
    rationale: str = ""


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}


class ToolResult(BaseModel):
    name: str
    ok: bool = True
    output: Any = None
    error: Optional[str] = None
    display: str = ""                  # short line shown in the transcript


class Critique(BaseModel):
    """Output of the Reflector Agent (self-review before the reply is sent)."""

    approved: bool = True
    violations: List[str] = []
    revised_reply: Optional[str] = None
    score: float = 1.0


class AgentTrace(BaseModel):
    """Everything the UI/report needs to explain *why* a reply looked like it did."""

    agent: str
    action: str
    detail: str = ""
    ms: int = 0


class AgentResponse(BaseModel):
    session_id: str
    reply: str
    risk_level: RiskLevel
    risk_score: float
    intent: Intent
    emotions: List[str] = []
    tools_used: List[ToolResult] = []
    trace: List[AgentTrace] = []
    escalated: bool = False
    resources: List[Dict[str, str]] = []
    disclaimer: str = (
        "MITRA is a student-built support companion, not a medical service. "
        "It cannot diagnose or treat any condition."
    )


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    consent: bool = True               # explicit consent to store notes locally


class MoodEntry(BaseModel):
    session_id: str
    score: int = Field(ge=1, le=10)
    label: str = ""
    note: str = ""
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
