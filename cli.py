#!/usr/bin/env python3
"""Terminal client for MITRA — useful for the viva when a browser is awkward.

    python cli.py             # chat
    python cli.py --trace     # chat, printing the agent trace each turn
"""

from __future__ import annotations

import argparse
import sys

from app.config import settings
from app.orchestrator import Orchestrator

BANNER = """
  MITRA  ·  mental wellness support conversation agent
  {tag}
  backend: {backend}   region: {region}
  Not a medical service. In danger? Tele-MANAS 14416 (24x7) or 112.
  Type 'exit' to quit, 'report' for a session summary.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="MITRA command-line client")
    ap.add_argument("--trace", action="store_true", help="print the agent trace")
    ap.add_argument("--session", default=None, help="resume an existing session id")
    args = ap.parse_args()

    orch = Orchestrator()
    sid = orch.store.ensure_session(args.session)
    print(BANNER.format(
        tag=settings.tagline,
        backend="offline rule engine" if settings.offline else
                f"{settings.provider}/{settings.model}",
        region=settings.region,
    ))
    print(f"  session: {sid}\n")

    while True:
        try:
            text = input("you » ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTake care of yourself.")
            return 0
        if not text:
            continue
        if text.lower() in {"exit", "quit", "bye"}:
            print("mitra » Thanks for talking. Come back whenever you need to.")
            return 0
        if text.lower() == "report":
            rep = orch.session_report(sid)
            print(f"  turns: {rep['session'].get('turns', 0)} | "
                  f"moods logged: {len(rep['moods'])} | "
                  f"escalations: {len(rep['escalations'])}")
            print(f"  summary: {rep['session'].get('summary') or '(none yet)'}\n")
            continue

        res = orch.handle(text, sid)
        sid = res.session_id
        print(f"\nmitra » {res.reply}\n")
        if res.escalated:
            print("  ── escalation logged; helplines shown above ──")
        if args.trace:
            print(f"  [risk={res.risk_level.value} {res.risk_score:.2f} "
                  f"intent={res.intent.value} feelings={','.join(res.emotions)}]")
            for t in res.trace:
                print(f"    {t.agent:<18}{t.action:<20}{t.detail}")
            for tool in res.tools_used:
                print(f"    tool: {tool.name} -> {tool.display[:80]}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
