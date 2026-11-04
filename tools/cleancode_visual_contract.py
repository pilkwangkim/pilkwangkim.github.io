"""Native visual explanations of exact test values and file-test boundaries.

The diagrams show objects, routes, and mismatched identities rather than
repeating explanatory paragraphs in large colored boxes.  Imports of the shared
figure canvas stay local so the parent generator can import these makers.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from matplotlib.patches import Circle, Polygon, Rectangle

INK = "#29343D"
MUTED = "#6C7780"
GRID = "#D9DFE3"
TEAL = "#536F7A"
RED = "#916865"
GOLD = "#847660"
TP = "#F0F4F5"
RP = "#F7F3F2"


def _cell(c, x, y, value, color=INK, fill="white", size=68, index=None):
    c.ax.add_patch(Rectangle((x-size/2, y-size/2), size, size,
                             facecolor=fill, edgecolor=color, linewidth=2))
    c.text(x, y-2, value, 34, color, ha="center", weight="bold")
    if index is not None:
        c.text(x, y-56, str(index), 28, MUTED, ha="center", mono=True)


def _output(c, x, y, width, heading, promoted, waiting, faulty=False):
    color = RED if faulty else TEAL
    c.text(x+width/2, y, heading, 29, color, ha="center", weight="bold")
    c.line((x+15, y+28), (x+width-15, y+28), color, 1.8)
    c.text(x+width/2, y+68, "Confirmed", 28, MUTED, ha="center")
    _cell(c, x+width/2-34, y+154, "B", index=0)
    _cell(c, x+width/2+34, y+154, promoted, color,
          RP if faulty else TP, index=1)
    c.text(x+width/2, y+216, "Waiting", 28, MUTED, ha="center")
    _cell(c, x+width/2, y+273, waiting, color,
          RP if faulty else TP)
    c.text(x+width/2, y+334, "2 confirmed", 28, MUTED, ha="center")


def fig02(mobile):
    from cleancode_figures import Canvas, cancel, state

    c = Canvas(mobile, 1060 if mobile else 810)
    if mobile:
        c.text(24, 35, "A count is not enough", 30, weight="bold")
        c.text(24, 86, 'cancel(state, "A")', 29, mono=True)
        c.text(24, 145, "Before · capacity 2", 28, MUTED)
        c.text(109, 205, "Confirmed", 28, MUTED, ha="center")
        _cell(c, 75, 270, "A")
        _cell(c, 143, 270, "B")
        c.text(354, 205, "Waiting", 28, MUTED, ha="center")
        _cell(c, 320, 270, "C", TEAL, TP)
        _cell(c, 388, 270, "D", GOLD)
        c.line((24, 330), (456, 330), GRID, 1.5)
        c.text(120, 359, "FIFO rule", 28, TEAL, ha="center")
        c.text(360, 359, "faulty code", 28, RED, ha="center")
        c.arrow((120, 390), (120, 420), TEAL)
        c.arrow((360, 390), (360, 420), RED)
        _output(c, 12, 455, 216, "Expected", "C", "D")
        _output(c, 252, 455, 216, "Faulty actual", "D", "C", True)
        c.line((240, 442), (240, 808), GRID)
        c.text(240, 870, "C  ≠  D", 45, RED, ha="center", weight="bold")
        c.line((132, 905), (348, 905), RED, 2)
        c.text(240, 948, "assert actual == expected", 28, RED,
               ha="center", mono=True)
        c.text(240, 1005, "FAIL · identities differ", 29, RED,
               ha="center", weight="bold")
    else:
        c.text(36, 37, "Same count. Different people.", 30,
               weight="bold")
        c.text(36, 98, 'cancel(state, "A")', 29, mono=True)
        c.text(36, 151, "Confirmed · capacity 2", 28, MUTED)
        _cell(c, 105, 215, "A")
        _cell(c, 173, 215, "B")
        c.text(489, 151, "Waiting · FIFO", 28, MUTED)
        _cell(c, 566, 215, "C", TEAL, TP)
        _cell(c, 634, 215, "D", GOLD)
        c.text(566, 263, "head", 28, TEAL, ha="center")
        c.line((36, 276), (924, 276), GRID, 1.5)
        c.text(249, 314, "From the FIFO rule", 28, TEAL, ha="center")
        c.text(711, 314, "Faulty implementation", 28, RED, ha="center")
        c.arrow((249, 340), (249, 366), TEAL)
        c.arrow((711, 340), (711, 366), RED)
        _output(c, 43, 403, 412, "Expected", "C", "D")
        _output(c, 505, 403, 412, "Faulty actual", "D", "C", True)
        c.line((480, 385), (480, 724), GRID)
        c.text(480, 774, "C ≠ D  ·  assert actual == expected  →  FAIL", 29,
               RED, ha="center", mono=True)
    initial = state(2, ["A", "B"], ["C", "D"])
    expected = state(2, ["B", "C"], ["D"])
    faulty_actual = cancel(initial, "A", "lifo-promotion")
    assert cancel(initial, "A") == expected
    assert faulty_actual == state(2, ["B", "D"], ["C"])
    c.fig._semantics = {
        "input": initial,
        "operation": "cancel(A)",
        "expected": expected,
        "faulty_actual": faulty_actual,
        "bug": "lifo-promotion",
    }
    return c.fig


def _record(c, x, y, width=110, height=85, color=INK):
    """A small structured value, deliberately not a filesystem icon."""
    c.ax.add_patch(Rectangle((x-width/2, y-height/2), width, height,
                             facecolor="white", edgecolor=color, linewidth=1.8))
    c.line((x-width*.32, y-height*.24), (x+width*.32, y-height*.24), color, 1.7)
    c.line((x-width*.32, y), (x+width*.1, y), color, 1.7)
    c.line((x-width*.32, y+height*.24), (x+width*.23, y+height*.24), color, 1.7)


def _file(c, x, y, width=104, height=126, color=GOLD):
    fold = 28
    left, top = x-width/2, y-height/2
    points = [(left, top), (left+width-fold, top), (left+width, top+fold),
              (left+width, top+height), (left, top+height)]
    c.ax.add_patch(Polygon(points, closed=True, facecolor="#F6F4F0",
                           edgecolor=color, linewidth=1.8))
    c.line((left+width-fold, top), (left+width-fold, top+fold), color, 1.6)
    c.line((left+width-fold, top+fold), (left+width, top+fold), color, 1.6)
    c.text(x, y+10, "JSON", 28, color, ha="center", mono=True)


def _compute(c, x, y, width=300, label="register / cancel", color=TEAL):
    h = 91
    c.ax.add_patch(Polygon([(x-width/2+22, y-h/2), (x+width/2-22, y-h/2),
                           (x+width/2, y), (x+width/2-22, y+h/2),
                           (x-width/2+22, y+h/2), (x-width/2, y)],
                          facecolor=TP, edgecolor=color, linewidth=1.8))
    c.text(x, y, label, 28, color, ha="center", mono=True)


def _column_heading(c, x, y, number, title, color=INK):
    c.ax.add_patch(Circle((x+14, y), 17, facecolor=color, edgecolor="none"))
    c.text(x+14, y, str(number), 26, "white", ha="center", weight="bold")
    c.text(x+49, y, title, 30, color, weight="bold")


def fig05(mobile):
    from cleancode_figures import Canvas

    c = Canvas(mobile, 1250 if mobile else 850)
    if mobile:
        _column_heading(c, 24, 37, 1, "Pure logic", TEAL)
        _record(c, 88, 125, 96, 76)
        c.text(88, 192, "state", 28, MUTED, ha="center")
        c.arrow((151, 125), (198, 125), TEAL)
        _compute(c, 323, 125, 228, "register", TEAL)
        c.text(323, 192, "or cancel", 28, TEAL, ha="center", mono=True)
        c.arrow((323, 220), (323, 247), TEAL)
        _record(c, 323, 297, 96, 76, TEAL)
        c.text(142, 296, "new state", 28, TEAL, ha="center")
        c.line((24, 365), (456, 365), GRID, 1.5)

        _column_heading(c, 24, 412, 2, "Real file test", GOLD)
        _record(c, 82, 508, 94, 76)
        c.text(82, 575, "expected", 28, INK, ha="center")
        c.arrow((147, 508), (264, 508), GOLD)
        c.text(208, 475, "save", 28, GOLD, ha="center", mono=True)
        _file(c, 354, 508, 110, 126)
        c.arrow((354, 594), (354, 641), GOLD)
        c.text(284, 622, "load", 28, GOLD, ha="center", mono=True)
        _record(c, 354, 709, 94, 76, GOLD)
        c.text(199, 708, "actual", 28, GOLD, ha="center")
        c.text(240, 791, "actual == expected", 28, INK, ha="center", mono=True)
        c.line((24, 837), (456, 837), GRID, 1.5)

        _column_heading(c, 24, 884, 3, "Stub + spy", INK)
        c.text(24, 951, "load stub", 28, TEAL, mono=True)
        _record(c, 379, 951, 104, 66, TEAL)
        c.arrow((215, 951), (304, 951), TEAL)
        c.arrow((379, 994), (240, 1015), TEAL)
        _compute(c, 240, 1068, 300, "register")
        c.arrow((240, 1125), (240, 1153), GOLD)
        c.text(240, 1188, "save spy · records state", 28, GOLD,
               ha="center", mono=True)
    else:
        _column_heading(c, 36, 39, 1, "Pure logic", TEAL)
        _record(c, 137, 148, color=INK)
        c.text(137, 221, "state", 28, MUTED, ha="center")
        _compute(c, 480, 148, 338)
        _record(c, 824, 148, color=TEAL)
        c.text(824, 221, "new state", 28, TEAL, ha="center")
        c.arrow((213, 148), (291, 148), TEAL)
        c.arrow((669, 148), (749, 148), TEAL)
        c.line((36, 267), (924, 267), GRID, 1.5)

        _column_heading(c, 36, 311, 2, "Real file test", GOLD)
        _record(c, 137, 422)
        c.text(137, 499, "expected", 28, INK, ha="center")
        _file(c, 480, 422)
        _record(c, 824, 422, color=GOLD)
        c.text(824, 499, "actual", 28, GOLD, ha="center")
        c.arrow((213, 422), (400, 422), GOLD)
        c.text(306, 378, "save", 28, GOLD, ha="center", mono=True)
        c.arrow((560, 422), (749, 422), GOLD)
        c.text(654, 378, "load", 28, GOLD, ha="center", mono=True)
        c.line((137, 535), (824, 535), GRID, 1.5)
        c.line((137, 520), (137, 535), GRID)
        c.line((824, 520), (824, 535), GRID)
        c.text(480, 566, "actual == expected", 28, INK, ha="center", mono=True)
        c.line((36, 610), (924, 610), GRID, 1.5)

        _column_heading(c, 36, 655, 3, "Stub + spy", INK)
        c.text(143, 731, "load stub", 28, TEAL, ha="center", mono=True)
        c.text(480, 731, "register", 28, INK,
               ha="center", mono=True)
        c.text(825, 731, "save spy", 28, GOLD, ha="center", mono=True)
        c.arrow((230, 731), (292, 731), TEAL)
        c.arrow((669, 731), (732, 731), GOLD)
        c.text(143, 791, "supplies state", 28, TEAL, ha="center")
        c.text(825, 791, "records state", 28, GOLD, ha="center")
    c.fig._semantics = {
        "pure_call": "state -> register/cancel -> new state; no I/O",
        "real_file_test": ["save_workshop(expected)", "temporary JSON file",
                           "load_workshop()", "actual == expected"],
        "test_double": {"load_stub": "supply known state",
                        "wrapper": "register_with_storage",
                        "save_spy": "record new state", "real_file_io": False},
    }
    return c.fig


def main():
    import matplotlib.pyplot as plt
    from cleancode_figures import configure, validate

    configure()
    destination = Path(__file__).resolve().parents[1] / "_draft/_checks/cleancode-redesign-preview"
    destination.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, maker in [("fig-02-test", fig02), ("fig-05-boundaries", fig05)]:
        for mobile in (False, True):
            stem = name + ("-mobile" if mobile else "")
            fig = maker(mobile)
            report = validate(fig, stem)
            reports.append({"stem": stem, "width": fig._width,
                            "height": fig._height, "layout": report,
                            "semantics": fig._semantics})
            fig.savefig(destination / (stem+".png"), dpi=100)
            fig.savefig(destination / (stem+".svg"), metadata={"Date": None})
            plt.close(fig)
    (destination / "contract-figure-checks.json").write_text(json.dumps(reports, indent=2)+"\n")
    print(json.dumps([{key: row[key] for key in ("stem", "width", "height")} for row in reports], indent=2))


if __name__ == "__main__":
    main()
