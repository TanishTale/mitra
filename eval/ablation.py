#!/usr/bin/env python3
"""Ablation study: what each layer of the triage stack actually contributes.

Reproduces Table 7.2 of the report. Run: python eval/ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import safety                                     # noqa: E402
from app.schemas import RiskLevel                          # noqa: E402


def score_set(path: Path) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))
    crisis = [c for c in cases if c["expected"] == "crisis"]
    others = [c for c in cases if c["expected"] in {"supportive", "elevated"}]
    hit = sum(1 for c in crisis
              if safety.assess(c["text"]).level is RiskLevel.CRISIS)
    false = sum(1 for c in others
                if safety.assess(c["text"]).level is RiskLevel.CRISIS)
    return {"recall": round(hit / max(1, len(crisis)), 3),
            "false_escalation": round(false / max(1, len(others)), 3)}


CONFIGS = {
    "A. lexicon only": {"concept": False, "context": False},
    "B. lexicon + context modifiers": {"concept": False, "context": True},
    "C. lexicon + concept rules": {"concept": True, "context": False},
    "D. full stack (shipped)": {"concept": True, "context": True},
}


def apply(cfg: dict) -> tuple:
    saved = (safety.CONCEPT_RULES[:], safety.ACADEMIC_CONTEXT,
             safety.THIRD_PARTY, safety.PAST_TENSE)
    if not cfg["concept"]:
        safety.CONCEPT_RULES.clear()
    if not cfg["context"]:
        import re
        never = re.compile(r"(?!x)x")
        safety.ACADEMIC_CONTEXT = never
        safety.THIRD_PARTY = never
        safety.PAST_TENSE = never
    return saved


def restore(saved) -> None:
    safety.CONCEPT_RULES[:] = saved[0]
    safety.ACADEMIC_CONTEXT, safety.THIRD_PARTY, safety.PAST_TENSE = saved[1:]


if __name__ == "__main__":
    here = Path(__file__).parent
    print(f"{'configuration':<32}{'tuned recall':>14}{'held-out recall':>18}"
          f"{'held-out FER':>15}")
    print("-" * 79)
    for name, cfg in CONFIGS.items():
        saved = apply(cfg)
        tuned = score_set(here / "dataset.json")
        held = score_set(here / "holdout.json")
        restore(saved)
        print(f"{name:<32}{tuned['recall']*100:>13.1f}%"
              f"{held['recall']*100:>17.1f}%{held['false_escalation']*100:>14.1f}%")
    print("\nFER = false escalation rate on non-crisis turns (lower is better).")
