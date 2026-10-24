"""Render the fixed data and arithmetic examples in Gemma Part 1.

No network access, agent runs, or new measurements. The task counts are the
September 25 inventory; budget and paired outcomes are explicitly hypothetical.
Both desktop and narrow layouts use exactly the same values.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[1] / "assets/img/posts/gemma-4-series"
BG, INK, MUTED, GRID = "#fbfbf8", "#183044", "#5c6d78", "#d7dfe1"
BLUE, TEAL, ORANGE, NEUTRAL = "#3b6fa1", "#25806f", "#bc6537", "#a2adb4"
COUNTS = [("FastAPI", 67), ("Rich", 48), ("Requests", 13), ("HTTPX", 1)]

plt.rcParams.update({"font.family": "Arial", "font.size": 14,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "svg.fonttype": "none"})


def save(fig, name, mobile):
    stem = name + ("-mobile" if mobile else "")
    fig.savefig(OUT / (stem + ".png"), dpi=200)
    fig.savefig(OUT / (stem + ".svg"))
    plt.close(fig)


def title(fig, headline, detail, mobile):
    fig.text(.07, .96, headline, fontsize=19 if mobile else 21,
             fontweight="bold", ha="left", va="top")
    fig.text(.07, .865 if mobile else .855, detail,
             fontsize=13.5 if mobile else 14, color=MUTED, va="top",
             linespacing=1.45)


def composition(mobile):
    fig = plt.figure(figsize=(4.6, 6.7) if mobile else (10, 5.2))
    title(fig, "Development tasks\nby repository" if mobile else "Development tasks by repository",
          "129 tasks · 4 Python projects\nInventory: 25 September 2026" if mobile else
          "129 tasks · 4 Python projects · Inventory: 25 September 2026", mobile)
    ax = fig.add_axes([.09, .16, .82, .57] if mobile else [.10, .18, .82, .54])
    colors = [TEAL, BLUE, NEUTRAL, NEUTRAL]
    for i, ((name, count), color) in enumerate(zip(COUNTS, colors)):
        y = 6 - 2 * i
        ax.text(0, y + .55, name, ha="left", va="center", fontsize=14)
        ax.text(1, y + .55, f"{count} · {count/129:.1%}",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=14)
        ax.barh(y, count, height=.45, color=color, zorder=3)
    ax.set(xlim=(0, 75), ylim=(-.65, 7.3), yticks=[], xticks=[0, 25, 50, 75], xlabel="Tasks")
    ax.grid(axis="x", color=GRID, linewidth=.7, zorder=0)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(.07, .035, "FastAPI + Rich: 115 / 129 tasks (89.1%)", fontsize=13, color=MUTED)
    save(fig, "part1-fig3-task-composition", mobile)


def budget(mobile):
    fig = plt.figure(figsize=(4.6, 7.2) if mobile else (10, 5.6))
    title(fig, "Sharing the\n12-hour budget" if mobile else "Agent work and setup share the 12-hour budget",
          "120 sequential tasks\nEach session uses its full cap" if mobile else
          "120 sequential tasks · Each session uses its full cap", mobile)
    ax = fig.add_axes([.25, .28, .66, .44] if mobile else [.18, .24, .75, .45])
    for i, minutes in enumerate(range(1, 7)):
        used, left, y = 2 * minutes, 12 - 2 * minutes, 5 - i
        ax.barh(y, used, color=BLUE, height=.55)
        ax.barh(y, left, left=used, color="#e6ecec", height=.55)
        ax.text(used / 2, y, f"{used} h", color="white", fontsize=14,
                ha="center", va="center", fontweight="bold")
        if left:
            ax.text(used + left / 2, y, f"{left} h", fontsize=14, ha="center", va="center")
    ax.set(xlim=(0, 12), ylim=(-.6, 5.6), yticks=range(6),
           yticklabels=[f"{n} min" for n in range(6, 0, -1)], xticks=[0, 6, 12], xlabel="Total hours")
    ax.tick_params(length=0, pad=7)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(.07, .16 if mobile else .13, "Blue: agent sessions\nGray: time left for setup + overhead" if mobile else
             "Blue: agent sessions     Gray: time left for setup + other overhead",
             fontsize=13, color=MUTED, linespacing=1.5)
    fig.text(.07, .035, "3 min/task leaves 6 h for setup.\nVerification is outside this budget." if mobile else
             "3 min/task leaves 6 h for setup. Verification is outside this budget.",
             fontsize=13, color=MUTED, linespacing=1.5)
    save(fig, "part1-fig4-generation-budget", mobile)


def paired(mobile):
    fig = plt.figure(figsize=(4.6, 7.3) if mobile else (10, 6))
    title(fig, "One net gain,\nfive changed results" if mobile else "One net gain, five changed outcomes",
          "Same 12 tasks\nPair results on each issue" if mobile else
          "Same 12 tasks · Pair each candidate result with its baseline result", mobile)
    ax = fig.add_axes([.25, .24, .68, .44] if mobile else [.25, .20, .62, .48])
    cells = [(0, 1, 2, "Both\nsolve", "#e6ecec", INK),
             (1, 1, 2, "Only\nbaseline", "#f2e3da", ORANGE),
             (0, 0, 3, "Only\ncandidate", "#dbe8f1", BLUE),
             (1, 0, 5, "Neither\nsolves", "#e6ecec", INK)]
    for x, y, n, label, fill, color in cells:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fill, edgecolor=BG, linewidth=3))
        ax.text(x + .5, y + .66, str(n), ha="center", va="center",
                fontsize=29 if mobile else 34, fontweight="bold", color=color)
        ax.text(x + .5, y + .25, label, ha="center", va="center", fontsize=13.5)
    ax.set(xlim=(0, 2), ylim=(0, 2), xticks=[.5, 1.5], yticks=[.5, 1.5],
           xticklabels=["Candidate\nsolved", "Candidate\nunresolved"],
           yticklabels=["Baseline\nunresolved", "Baseline\nsolved"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=10, labelsize=13.5)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(.07, .16 if mobile else .10,
             "Baseline: 4/12 → Candidate: 5/12\n3 gains − 2 losses = +1 resolved task",
             fontsize=14, linespacing=1.6, color=INK)
    fig.text(.07, .035, "Inspect all 5 changed outcomes.\nHypothetical paired results." if mobile else
             "Inspect all 5 changed outcomes. Hypothetical paired results.", fontsize=13, color=MUTED,
             linespacing=1.5)
    save(fig, "part1-fig5-paired-outcomes", mobile)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for mobile in (False, True):
        composition(mobile)
        budget(mobile)
        paired(mobile)
    (OUT / "part1-data-figures.json").write_text(json.dumps({
        "inventory_date": "2026-09-25", "repository_task_counts": dict(COUNTS),
        "budget_example": {"tasks": 120, "limit_hours": 12, "session_caps_minutes": list(range(1, 7)),
                           "hypothetical": True, "verification_in_budget": False},
        "paired_example": {"both_solve": 2, "baseline_only": 2, "candidate_only": 3,
                           "neither_solves": 5, "hypothetical": True},
        "source": "Values retained from the published Part 1. No new measurement.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
