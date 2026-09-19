#!/usr/bin/env python3
"""Generate the figures used in the report and the presentation.

    python docs/figures.py

Writes PNGs into docs/figures/. Matplotlib is used rather than an external
drawing tool so the diagrams regenerate from source alongside the code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

PINE = "#2D5F53"
SOFT = "#DCE8E3"
INK = "#15211E"
AMBER = "#9A6510"
ALERT = "#8E3B3B"
DUSK = "#5A6E8C"
LINE = "#B9C7C2"


def box(ax, x, y, w, h, text, fc=SOFT, ec=PINE, tc=INK, fs=9.5, bold=False, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.006,rounding_size={r}",
                                fc=fc, ec=ec, lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, linespacing=1.45,
            fontweight="bold" if bold else "normal")


def arrow(ax, p1, p2, color=PINE, style="-|>", ls="-", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w=11, h=7.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


# ---------------------------------------------------------------- figure 1
def fig_architecture() -> None:
    fig, ax = canvas(11.5, 7.6)
    ax.text(50, 96, "MITRA — System Architecture", ha="center", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(50, 92, "agentic orchestration with a deterministic safety kernel",
            ha="center", fontsize=9.5, color="#5C6B66", style="italic")

    box(ax, 4, 79, 22, 8, "Presentation layer\nweb chat UI  ·  CLI  ·  REST",
        fc="white", ec=DUSK)
    box(ax, 30, 79, 26, 8, "FastAPI service\n/chat  /mood  /resources  /session",
        fc="white", ec=DUSK)
    box(ax, 60, 79, 36, 8,
        "Agent trace inspector\nper-turn explainability for every decision",
        fc="white", ec=DUSK)

    box(ax, 4, 62, 92, 11, "", fc="#F4F8F6", ec=LINE)
    ax.text(6, 70.4, "ORCHESTRATOR — finite-state agent graph", fontsize=9.5,
            fontweight="bold", color=PINE)
    ax.text(6, 66.4,
            "bounded autonomy: max 4 tool calls per turn · deterministic crisis "
            "override · every node emits a trace record",
            fontsize=8.6, color="#5C6B66")

    agents = [
        (4.5, "Triage", "risk · emotion\nintent"),
        (23.0, "Planner", "tool selection\n+ execution"),
        (41.5, "Listener", "reflection &\nresponse"),
        (60.0, "Reflector", "self-critique\n+ repair"),
        (78.5, "Summariser", "memory\ncompression"),
    ]
    for x, name, sub in agents:
        box(ax, x, 43, 17, 14, f"{name}\nAgent\n\n{sub}", fc=SOFT, ec=PINE, fs=8.4)
    for i in range(len(agents) - 1):
        arrow(ax, (agents[i][0] + 17, 50), (agents[i + 1][0], 50))

    box(ax, 4.5, 27, 34, 12,
        "SAFETY KERNEL\n(pure Python — no model in the loop)\n"
        "Stage A lexicon · Stage B concept rules\ncontext modifiers · outbound filter\n"
        "crisis pathway · refusal templates",
        fc="#F7ECEC", ec=ALERT, fs=7.8)
    box(ax, 41.5, 27, 26, 12,
        "TOOL LAYER  (11 tools)\nmood · CBT reframe · breathing\n"
        "grounding · journal · self-check\nhelpline lookup",
        fc="white", ec=AMBER, fs=8.2)
    box(ax, 70, 27, 26, 12,
        "MODEL LAYER\nAnthropic / OpenAI / Gemini\nor offline rule engine\n"
        "(graceful degradation)",
        fc="white", ec=DUSK, fs=8.6)

    box(ax, 5.5, 11, 42, 10,
        "MEMORY  (local SQLite — nothing leaves the device)\n"
        "episodic turns · rolling summary · mood series\njournal · escalation audit log",
        fc="white", ec=PINE, fs=8.6)
    box(ax, 52, 11, 44, 10,
        "GOVERNANCE\nper-session consent · right to erasure\n"
        "audited escalations · 41 tests + 2 evaluation sets",
        fc="white", ec=PINE, fs=8.6)

    arrow(ax, (50, 79), (50, 73.2))
    arrow(ax, (13, 43), (13, 39.2), color=ALERT)
    arrow(ax, (31.5, 43), (54.5, 39.2), color=AMBER, rad=-0.12)
    arrow(ax, (68.5, 43), (83, 39.2), color=DUSK, rad=0.12)
    arrow(ax, (86, 27), (86, 21.2), color=LINE)
    arrow(ax, (26, 27), (26, 21.2), color=LINE)

    fig.savefig(OUT / "fig1_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_flow() -> None:
    fig, ax = canvas(9.5, 8.4)
    ax.text(50, 97, "Per-turn control flow", ha="center", fontsize=14,
            fontweight="bold", color=INK)

    box(ax, 33, 87, 34, 7, "user turn", fc="white", ec=DUSK)
    box(ax, 30, 74, 40, 8, "TRIAGE\nStage A lexicon + Stage B concepts\n+ context modifiers",
        fc=SOFT, ec=PINE, fs=8.6)

    ax.add_patch(plt.Polygon([[50, 70], [72, 60], [50, 50], [28, 60]],
                             fc="#F7ECEC", ec=ALERT, lw=1.3, zorder=2))
    ax.text(50, 60, "risk band?", ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=ALERT, zorder=3)

    box(ax, 74, 54, 24, 12,
        "CRISIS PATHWAY\ndeterministic reply\nverified helplines\nescalation audited",
        fc="#F7ECEC", ec=ALERT, fs=8.4)
    box(ax, 2, 54, 22, 12,
        "BLOCKED REQUEST\nrefusal template\n(no diagnosis,\nno medication)",
        fc="#FDF4E8", ec=AMBER, fs=8.4)

    box(ax, 31, 38, 38, 7, "PLANNER — select and run tools (cap 4)", fc=SOFT, ec=PINE, fs=9)
    box(ax, 31, 28, 38, 7, "LISTENER — draft the reply", fc=SOFT, ec=PINE, fs=9)
    box(ax, 31, 18, 38, 7, "REFLECTOR — rubric + outbound filter", fc=SOFT, ec=PINE, fs=9)
    box(ax, 31, 4, 38, 7, "reply + trace to the user", fc="white", ec=DUSK, fs=9)

    arrow(ax, (50, 87), (50, 82.2))
    arrow(ax, (50, 74), (50, 70.2))
    arrow(ax, (72, 60), (74, 60), color=ALERT)
    arrow(ax, (28, 60), (24, 60), color=AMBER)
    arrow(ax, (50, 50), (50, 45.2))
    arrow(ax, (50, 38), (50, 35.2))
    arrow(ax, (50, 28), (50, 25.2))
    arrow(ax, (50, 18), (50, 11.2))
    arrow(ax, (86, 54), (70, 8), color=ALERT, ls=(0, (4, 3)), rad=0.16)
    arrow(ax, (13, 54), (31, 8), color=AMBER, ls=(0, (4, 3)), rad=-0.16)
    arrow(ax, (31, 21.5), (24, 31.5), color="#8E3B3B", ls=(0, (3, 3)), rad=0.3)
    ax.text(18, 26, "fail →\nrepair", fontsize=7.8, color="#8E3B3B", ha="center")

    ax.text(72, 67.5, "crisis", fontsize=8.4, color=ALERT, style="italic")
    ax.text(26, 67.5, "blocked", fontsize=8.4, color=AMBER, style="italic", ha="right")
    ax.text(52, 47.5, "supportive / elevated", fontsize=8.4, color=PINE, style="italic")

    fig.savefig(OUT / "fig2_flow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_results() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    fig.patch.set_facecolor("white")

    configs = ["A\nlexicon\nonly", "B\n+ context\nmodifiers",
               "C\n+ concept\nrules", "D\nfull stack\n(shipped)"]
    recall = [20.0, 20.0, 100.0, 100.0]
    fer = [13.3, 0.0, 13.3, 0.0]

    x = range(len(configs))
    axes[0].bar([i - 0.19 for i in x], recall, width=0.38, color=PINE,
                label="crisis recall")
    axes[0].bar([i + 0.19 for i in x], fer, width=0.38, color=ALERT,
                label="false escalation")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(configs, fontsize=8.4)
    axes[0].set_ylabel("% (held-out set)", fontsize=9)
    axes[0].set_title("Ablation: contribution of each triage layer",
                      fontsize=10.5, fontweight="bold", color=INK)
    axes[0].set_ylim(0, 112)
    axes[0].legend(fontsize=8.4, frameon=False, loc="upper left")
    for i, v in enumerate(recall):
        axes[0].text(i - 0.19, v + 2.5, f"{v:.0f}", ha="center", fontsize=8, color=PINE)
    for i, v in enumerate(fer):
        axes[0].text(i + 0.19, v + 2.5, f"{v:.0f}", ha="center", fontsize=8, color=ALERT)

    metrics = ["crisis\nrecall", "band\naccuracy", "guardrail\npass", "false\nescalation"]
    tuned = [100, 100, 100, 0]
    held = [100, 95.5, 100, 0]
    x2 = range(len(metrics))
    axes[1].bar([i - 0.19 for i in x2], tuned, width=0.38, color=PINE,
                label="tuned set (n=51)")
    axes[1].bar([i + 0.19 for i in x2], held, width=0.38, color=DUSK,
                label="held-out set (n=22)")
    axes[1].set_xticks(list(x2)); axes[1].set_xticklabels(metrics, fontsize=8.4)
    axes[1].set_ylim(0, 112); axes[1].set_ylabel("%", fontsize=9)
    axes[1].set_title("Shipped configuration, both evaluation sets",
                      fontsize=10.5, fontweight="bold", color=INK)
    axes[1].legend(fontsize=8.4, frameon=False, loc="upper center")

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8.4)
        a.grid(axis="y", color="#E3EAE7", lw=0.8)
        a.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_risk_bands() -> None:
    fig, ax = canvas(10, 3.0)
    ax.set_ylim(0, 40)
    bands = [(0, 30, "SUPPORTIVE", PINE, "listen, reflect, offer a skill"),
             (30, 55, "ELEVATED", AMBER, "warmer tone, surface human support"),
             (55, 100, "CRISIS", ALERT, "generative agents bypassed entirely")]
    for lo, hi, name, col, desc in bands:
        ax.add_patch(FancyBboxPatch((lo, 16), hi - lo, 10,
                                    boxstyle="round,pad=0.004,rounding_size=0.01",
                                    fc=col, ec="none", alpha=0.16))
        ax.text((lo + hi) / 2, 22.5, name, ha="center", fontsize=10.5,
                fontweight="bold", color=col)
        ax.text((lo + hi) / 2, 18.5, desc, ha="center", fontsize=8, color="#5C6B66")
    for v in (30, 55):
        ax.plot([v, v], [13, 27], color=INK, lw=1.1)
        ax.text(v, 10, f"{v/100:.2f}", ha="center", fontsize=8.6, color=INK)
    ax.text(0, 10, "0.00", ha="center", fontsize=8.6, color=INK)
    ax.text(100, 10, "1.00", ha="center", fontsize=8.6, color=INK)
    ax.text(50, 33, "Risk score → routing band", ha="center", fontsize=12,
            fontweight="bold", color=INK)
    ax.text(50, 4, "thresholds are configuration, not magic numbers: "
                   "they are tuned against the evaluation sets and are the single "
                   "place the safety/utility trade-off is made",
            ha="center", fontsize=7.8, color="#5C6B66", style="italic")
    fig.savefig(OUT / "fig4_bands.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_flow()
    fig_results()
    fig_risk_bands()
    for f in sorted(OUT.glob("*.png")):
        print("wrote", f)
