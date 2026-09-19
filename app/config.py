"""Central configuration for MITRA.

Every tunable lives here so that the viva demo can be reconfigured without
touching agent code. Values are read from environment variables (see
`.env.example`) and fall back to safe, offline-friendly defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MITRA_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime settings for the agent stack."""

    # ---- Identity -------------------------------------------------------
    app_name: str = "MITRA"
    tagline: str = "Mental-wellness Intelligent Triage & Reflective Agent"
    version: str = "1.0.0"

    # ---- LLM backend ----------------------------------------------------
    # provider: "anthropic" | "openai" | "gemini" | "offline"
    provider: str = field(default_factory=lambda: os.getenv("MITRA_PROVIDER", "offline"))
    model: str = field(default_factory=lambda: os.getenv("MITRA_MODEL", "claude-sonnet-4-5"))
    api_key: str = field(default_factory=lambda: os.getenv("MITRA_API_KEY", ""))
    temperature: float = field(default_factory=lambda: float(os.getenv("MITRA_TEMPERATURE", "0.4")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MITRA_MAX_TOKENS", "700")))
    request_timeout: int = 45

    # ---- Storage --------------------------------------------------------
    db_path: Path = field(default_factory=lambda: DATA_DIR / "mitra.db")

    # ---- Agent loop -----------------------------------------------------
    max_tool_iterations: int = 4          # bounded autonomy — never an open loop
    reflection_enabled: bool = field(default_factory=lambda: _env_bool("MITRA_REFLECTION", True))
    memory_window: int = 12               # turns replayed into the prompt
    summarise_after: int = 14             # turns before long-term summarisation

    # ---- Safety ---------------------------------------------------------
    # Any risk score at or above this value forces the crisis pathway and
    # disables every other agent. This is deliberately conservative.
    crisis_threshold: float = 0.55
    elevated_threshold: float = 0.30
    region: str = field(default_factory=lambda: os.getenv("MITRA_REGION", "IN"))

    @property
    def offline(self) -> bool:
        return self.provider == "offline" or not self.api_key

    def as_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "version": self.version,
            "provider": self.provider,
            "model": self.model if not self.offline else "rule-based-fallback",
            "offline": self.offline,
            "reflection_enabled": self.reflection_enabled,
            "region": self.region,
        }


settings = Settings()

# Non-negotiable behavioural contract injected into every LLM call.
SYSTEM_CONTRACT = """You are MITRA, a mental wellness support companion built for a
college project. You are NOT a doctor, therapist, or diagnostic tool.

Hard rules you never break:
1. Never diagnose a condition, never name or adjust medication, never give dosages.
2. Never discourage a person from seeking professional or family support.
3. Never provide, confirm, or discuss methods of self-harm, suicide, or disordered
   eating - not even to refuse them in detail.
4. Always answer in warm, plain, non-clinical language; short paragraphs.
5. Reflect feelings before offering any suggestion. Ask at most one question per turn.
6. If a person appears to be in danger, everything else stops and you surface
   human help immediately.
7. You never claim to be human, never claim to remember beyond this app's stored
   notes, and you say plainly that you are an AI when asked.

Style: 90-160 words, second person, no bullet lists unless the user asks for steps,
no emojis unless the user uses them first."""
