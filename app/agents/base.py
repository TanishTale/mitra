"""Common scaffolding for every agent in the graph."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..config import SYSTEM_CONTRACT, settings
from ..llm import LLMClient
from ..schemas import AgentTrace


class Agent(ABC):
    """An agent is a named, single-responsibility unit with its own persona.

    Splitting responsibilities this way is what makes the system *agentic*
    rather than a single prompt: each unit has a narrow objective, its own
    instructions, and the orchestrator decides which ones run for a given turn.
    """

    name: str = "agent"
    role: str = ""
    persona: str = ""

    def __init__(self, llm: LLMClient | None = None, cfg=settings):
        self.llm = llm or LLMClient(cfg)
        self.cfg = cfg
        self.traces: List[AgentTrace] = []

    # -- helpers ---------------------------------------------------------
    def system_prompt(self) -> str:
        return f"{SYSTEM_CONTRACT}\n\n--- Your role in the system ---\n{self.persona}"

    def trace(self, action: str, detail: str = "", ms: int = 0) -> AgentTrace:
        t = AgentTrace(agent=self.name, action=action, detail=detail, ms=ms)
        self.traces.append(t)
        return t

    def ask(self, messages: List[Dict[str, str]], offline_context: Dict | None = None) -> str:
        start = time.time()
        out = self.llm.complete(self.system_prompt(), messages,
                                offline_context=offline_context)
        self.trace("llm_call", f"backend={self.llm.last_backend}",
                   int((time.time() - start) * 1000))
        return out

    # -- contract --------------------------------------------------------
    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Read from and write to the shared conversation state."""
        raise NotImplementedError
