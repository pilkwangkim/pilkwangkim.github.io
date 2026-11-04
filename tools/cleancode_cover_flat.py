"""Restrained, code-native flat cover for the Clean Code / TDD article."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from matplotlib.patches import Rectangle


def cover():
    # Local imports keep the cover reusable by the shared exporter and checker.
    from cleancode_figures import Canvas, INK, MUTED, GRID, TEAL, RED, RP, TP

    c = Canvas(False, 941, 1672)
    c.text(836, 135, "Clean Code & TDD", 116, INK, ha="center")
    c.line((96, 235), (1576, 235), GRID, 1.4)
    c.text(836, 310, "Capacity 1 · A confirmed · B waiting", 84,
           MUTED, ha="center")
    c.text(836, 420, "Actual confirmed", 84, INK, ha="center")

    # Outputs from one fixed cancellation test, not three different inputs.
    # Refactoring preserves the already-correct result and expectation.
    for x, label, value, color, fill in [
        (330, "Fail", "[A]", RED, RP),
        (836, "Pass", "[B]", TEAL, TP),
        (1342, "Refactor", "[B]", TEAL, TP),
    ]:
        c.text(x, 525, label, 84, color, ha="center")
        c.ax.add_patch(Rectangle((x-120, 575), 240, 170,
                                 facecolor=fill, edgecolor=color, linewidth=4))
        c.text(x, 660, value, 112, color, ha="center", mono=True)
    c.arrow((478, 660), (688, 660), MUTED, 4.5)
    c.arrow((984, 660), (1194, 660), TEAL, 4.5)
    c.text(836, 855, "cancel(A) → expect [B]", 84, INK,
           ha="center", mono=True)
    c.fig._semantics = {
        "kind": "conceptual-cover",
        "title": "Clean Code & TDD",
        "feedback_labels": ["Fail", "Pass", "Refactor"],
        "input": {"capacity": 1, "confirmed": ["A"], "waiting": ["B"]},
        "operation": "cancel(A)",
        "displayed_actual_confirmed": [["A"], ["B"], ["B"]],
        "expected_confirmed": ["B"],
        "rendering": "flat vector; one fixed test across implementation stages",
    }
    return c.fig
