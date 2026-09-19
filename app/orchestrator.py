"""Orchestrator — the finite-state agent graph that makes MITRA agentic.

    user turn
        |
        v
  [TriageAgent]  ---- crisis ----> [Crisis pathway]  (deterministic, no LLM)
        |                                 |
   supportive / elevated                  v
        |                          log escalation + helplines
        v
  [PlannerAgent] -- bounded tool loop --> tool results
        |
        v
  [ListenerAgent] -- draft reply
        |
        v
  [ReflectorAgent] -- rubric + safety screen -- fail --> repair / replace
        |
        v
  [SummariserAgent] (every N turns)
        |
        v
    final reply + trace

Two properties are worth naming for the viva:
* **Deterministic override.** The crisis pathway is pure Python. No model call
  can prevent it from firing or alter its wording.
* **Observability.** Every node appends to `trace`, so the UI can show exactly
  which agents ran, which tools fired and why the reply looks the way it does.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from . import safety
from .agents import (ListenerAgent, PlannerAgent, ReflectorAgent,
                     SummariserAgent, TriageAgent)
from .config import settings
from .llm import LLMClient
from .memory import MemoryStore
from .schemas import AgentResponse, AgentTrace, RiskLevel, ToolResult


class Orchestrator:
    def __init__(self, cfg=settings, store: MemoryStore | None = None):
        self.cfg = cfg
        self.store = store or MemoryStore(cfg.db_path)
        self.llm = LLMClient(cfg)
        self.triage = TriageAgent(self.llm, cfg)
        self.planner = PlannerAgent(self.llm, cfg)
        self.listener = ListenerAgent(self.llm, cfg)
        self.reflector = ReflectorAgent(self.llm, cfg)
        self.summariser = SummariserAgent(self.llm, cfg)

    # ---------------------------------------------------------------
    def handle(self, message: str, session_id: str | None = None,
               consent: bool = True) -> AgentResponse:
        started = time.time()
        sid = self.store.ensure_session(session_id, consent)
        for agent in (self.triage, self.planner, self.listener,
                      self.reflector, self.summariser):
            agent.traces.clear()

        info = self.store.session_info(sid)
        state: Dict[str, Any] = {
            "session_id": sid,
            "user_text": message.strip(),
            "consent": consent,
            "store": self.store,
            "history": self.store.history(sid, self.cfg.memory_window),
            "prior_user_turns": self.store.user_turns(sid, 5),
            "summary": info.get("summary", ""),
        }

        if consent:
            self.store.add_message(sid, "user", state["user_text"])

        # --- node 1: triage -----------------------------------------
        state = self.triage.run(state)
        triage = state["triage"]

        # --- node 2a: crisis pathway (deterministic) ------------------
        if triage.risk_level == RiskLevel.CRISIS:
            return self._crisis(state, started)

        # --- node 2b: blocked request ---------------------------------
        if state.get("blocked_request"):
            return self._refusal(state, started)

        # --- node 3: planner ------------------------------------------
        state = self.planner.run(state)

        # --- node 4: listener -----------------------------------------
        state = self.listener.run(state)

        # --- node 5: reflector ----------------------------------------
        state = self.reflector.run(state)
        reply = state["final_reply"]

        # --- node 6: summariser (periodic) ----------------------------
        turns = self.store.bump_turns(sid)
        if turns and turns % self.cfg.summarise_after == 0:
            state = self.summariser.run(state)

        if consent:
            self.store.add_message(sid, "assistant", reply,
                                   risk_level=triage.risk_level.value,
                                   risk_score=triage.risk_score,
                                   intent=triage.intent.value)

        resources: List[Dict[str, str]] = []
        if triage.risk_level == RiskLevel.ELEVATED:
            resources = safety.helplines(self.cfg.region)[:3]

        return AgentResponse(
            session_id=sid,
            reply=reply,
            risk_level=triage.risk_level,
            risk_score=triage.risk_score,
            intent=triage.intent,
            emotions=triage.emotions,
            tools_used=state.get("tool_results", []),
            trace=self._trace(started),
            escalated=False,
            resources=resources,
        )

    # ---------------------------------------------------------------
    def _crisis(self, state: Dict[str, Any], started: float) -> AgentResponse:
        sid = state["session_id"]
        triage = state["triage"]
        reply, resources = safety.crisis_reply(self.cfg.region)
        self.store.log_escalation(sid, triage.risk_score, triage.signals)
        if state["consent"]:
            self.store.add_message(sid, "assistant", reply,
                                   risk_level="crisis",
                                   risk_score=triage.risk_score,
                                   intent=triage.intent.value)
        self.triage.trace("crisis_pathway",
                          "generative agents bypassed; deterministic response served")
        return AgentResponse(
            session_id=sid, reply=reply,
            risk_level=RiskLevel.CRISIS, risk_score=triage.risk_score,
            intent=triage.intent, emotions=triage.emotions,
            tools_used=[ToolResult(name="find_support", ok=True,
                                   output=resources,
                                   display="Surfaced verified crisis helplines.")],
            trace=self._trace(started), escalated=True, resources=resources,
        )

    def _refusal(self, state: Dict[str, Any], started: float) -> AgentResponse:
        sid = state["session_id"]
        triage = state["triage"]
        label = state["blocked_request"]
        reply = safety.REFUSAL_TEMPLATES.get(
            label,
            "That's outside what I can safely help with, but I'm still here for "
            "what's underneath it.",
        )
        if state["consent"]:
            self.store.add_message(sid, "assistant", reply,
                                   risk_level=triage.risk_level.value,
                                   risk_score=triage.risk_score,
                                   intent=triage.intent.value)
        self.triage.trace("guardrail_refusal", label)
        return AgentResponse(
            session_id=sid, reply=reply,
            risk_level=triage.risk_level, risk_score=triage.risk_score,
            intent=triage.intent, emotions=triage.emotions,
            trace=self._trace(started),
            escalated=triage.risk_level == RiskLevel.CRISIS,
            resources=safety.helplines(self.cfg.region)[:2],
        )

    def _trace(self, started: float) -> List[AgentTrace]:
        traces: List[AgentTrace] = []
        for agent in (self.triage, self.planner, self.listener,
                      self.reflector, self.summariser):
            traces.extend(agent.traces)
        traces.append(AgentTrace(agent="Orchestrator", action="complete",
                                 detail=f"backend={self.llm.last_backend}",
                                 ms=int((time.time() - started) * 1000)))
        return traces

    # ---------------------------------------------------------------
    def session_report(self, sid: str) -> Dict[str, Any]:
        """Data for the dashboard / viva demo."""
        return {
            "session": self.store.session_info(sid),
            "moods": self.store.moods(sid),
            "journal": self.store.journal_entries(sid),
            "escalations": self.store.escalations(sid),
            "messages": self.store.history(sid, 100),
        }
