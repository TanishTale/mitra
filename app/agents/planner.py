"""Planner Agent — decides which tools to invoke for this turn, and executes
them under a hard iteration cap.

Autonomy here is deliberately *bounded*. The planner may chain at most
`max_tool_iterations` calls, may only use tools flagged `safe_in_crisis` when
the triage band is CRISIS, and may never call a tool that writes to memory when
the user has withheld consent.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .. import tools as toolkit
from ..schemas import Intent, RiskLevel, ToolCall, ToolResult
from ..tools.mood import parse_mood
from .base import Agent

PLANNER_PERSONA = """You are the Planner unit. You do not speak to the user.
Given the user's turn and the tool catalogue, reply with ONLY a JSON list of
tool calls: [{"name":"tool_name","arguments":{...}}]. Use an empty list when no
tool is needed. Never invent tool names."""


class PlannerAgent(Agent):
    name = "PlannerAgent"
    role = "Tool selection and execution"
    persona = PLANNER_PERSONA

    def plan(self, state: Dict[str, Any]) -> List[ToolCall]:
        """Rule-based planning — readable, testable and deterministic.

        A model-driven planner is available through `_model_plan`, but the rule
        planner is the default so the graph behaves identically offline.
        """
        triage = state["triage"]
        text: str = state["user_text"]
        calls: List[ToolCall] = []

        if triage.risk_level == RiskLevel.CRISIS:
            return [ToolCall(name="find_support",
                             arguments={"region": self.cfg.region, "kind": "crisis"})]

        if triage.intent == Intent.TRACK:
            score = parse_mood(text)
            if score is not None:
                calls.append(ToolCall(name="log_mood", arguments={"score": score,
                                                                  "note": text[:200]}))
            calls.append(ToolCall(name="mood_trend", arguments={}))
            if "journal" in text.lower() or "write" in text.lower():
                calls.append(ToolCall(name="journal_prompt",
                                      arguments={"emotion": triage.emotions[0]}))

        elif triage.intent == Intent.REFRAME:
            calls.append(ToolCall(name="cbt_reframe", arguments={"thought": text}))

        elif triage.intent == Intent.SKILL:
            primary = triage.emotions[0]
            if primary in {"anxiety", "overwhelm"}:
                calls.append(ToolCall(name="breathing_exercise",
                                      arguments={"style": "478"}))
            elif primary in {"fatigue"}:
                calls.append(ToolCall(name="sleep_hygiene", arguments={"issue": "sleep"}))
            else:
                calls.append(ToolCall(name="grounding_exercise",
                                      arguments={"technique": "54321"}))

        elif triage.intent == Intent.SCREENING:
            calls.append(ToolCall(name="self_check", arguments={"instrument": "phq2"}))

        elif triage.risk_level == RiskLevel.ELEVATED:
            calls.append(ToolCall(name="find_support",
                                  arguments={"region": self.cfg.region, "kind": "campus"}))

        # Continuity: if the user references the past, look it up rather than guess.
        if any(k in text.lower() for k in
               ("last time", "earlier you", "we talked", "remember", "i wrote",
                "what i said")):
            calls.append(ToolCall(name="recall_journal",
                                  arguments={"query": triage.emotions[0]}))

        return calls[: self.cfg.max_tool_iterations]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        triage = state["triage"]
        calls = self.plan(state)
        results: List[ToolResult] = []

        for i, call in enumerate(calls):
            if i >= self.cfg.max_tool_iterations:
                self.trace("cap_reached", f"stopped after {i} tool calls")
                break
            tool = toolkit.REGISTRY.get(call.name)
            if not tool:
                continue
            if triage.risk_level == RiskLevel.CRISIS and not tool.safe_in_crisis:
                self.trace("tool_blocked", f"{call.name} not permitted in crisis band")
                continue
            args = dict(call.arguments)
            if "store" in tool.parameters:
                if not state.get("consent", True):
                    self.trace("tool_blocked", f"{call.name} needs consent to store")
                    continue
                args["store"] = state["store"]
                args["session_id"] = state["session_id"]
            result = toolkit.call(call.name, **args)
            self.trace("tool_call", f"{call.name} -> ok={result.ok}")
            results.append(result)

        state["tool_results"] = results
        return state
