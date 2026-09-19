"""Reflective journalling tools (expressive-writing style)."""

from __future__ import annotations

from typing import Any

from ..schemas import ToolResult

PROMPTS = {
    "anxiety": "Write down the worry exactly as it sounds in your head, then write "
               "what you would need to know to be 10% less afraid of it.",
    "sadness": "Describe one small thing that used to matter to you. Not why it "
               "stopped — just what it was like when it did.",
    "anger": "Who or what crossed a line, and what line was it? Name the value, not "
             "just the incident.",
    "shame": "Write the sentence you keep saying about yourself. Then write it again "
             "as if a friend had done the same thing.",
    "loneliness": "Who is one person who has seen a real version of you? What would "
                  "you say to them if replying were easy?",
    "overwhelm": "List everything on your plate without ranking it. Then circle the "
                 "one item that, if it moved, would unblock two others.",
    "grief": "Write one memory you don't want to lose. Detail over meaning.",
    "neutral": "What did today actually feel like, before you explain or justify it?",
}


def journal_prompt(emotion: str = "neutral") -> ToolResult:
    prompt = PROMPTS.get((emotion or "neutral").lower(), PROMPTS["neutral"])
    return ToolResult(name="journal_prompt", ok=True, output={"prompt": prompt},
                      display=f"A prompt, if you want it: {prompt}")


def save_journal(store: Any, session_id: str, entry: str, prompt: str = "") -> ToolResult:
    if not (entry or "").strip():
        return ToolResult(name="save_journal", ok=False, error="empty entry",
                          display="Nothing to save yet.")
    rid = store.add_journal(session_id, entry.strip(), prompt)
    return ToolResult(name="save_journal", ok=True, output={"id": rid},
                      display="Saved that to your journal — it stays on this device.")


def recall_journal(store: Any, session_id: str, query: str) -> ToolResult:
    rows = store.search_journal(session_id, query or "")
    if not rows:
        return ToolResult(name="recall_journal", ok=True, output={"matches": []},
                          display="I couldn't find an earlier entry about that.")
    snippet = rows[0]["entry"]
    snippet = snippet[:160] + ("..." if len(snippet) > 160 else "")
    return ToolResult(
        name="recall_journal", ok=True, output={"matches": rows},
        display=f"Earlier you wrote: \u201c{snippet}\u201d",
    )
