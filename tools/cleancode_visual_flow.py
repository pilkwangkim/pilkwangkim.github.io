"""State trajectories and code-extraction plates for the Clean Code article.

These three renderers are intentionally separate from export/publishing.  Every
arrow has a specific role: movement of an ID, progression of a TDD cycle, or a
call/extraction dependency.  Array positions are not seats assigned to people.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path


def _api():
    # Local imports let the export module select these makers without a cycle.
    from cleancode_figures import Canvas, domain_examples, state
    return Canvas, domain_examples, state


INK = "#29343D"
MUTED = "#6C7780"
FAINT = "#D9DFE3"
TEAL = "#536F7A"
RED = "#916865"
GOLD = "#847660"
SOFT = "#F5F7F8"
TP = "#F0F4F5"
GP = "#F6F4F0"
RP = "#F7F3F2"


def _route(c, points, color=TEAL, width=2.8, dash=None, arrow=True):
    codes = [Path.MOVETO] + [Path.LINETO] * (len(points)-1)
    patch = FancyArrowPatch(path=Path(points, codes), arrowstyle="-|>" if arrow else "-",
                            mutation_scale=18, linewidth=width, color=color,
                            linestyle=dash or "-", joinstyle="round", zorder=1)
    c.ax.add_patch(patch)


def _cell(c, x, y, value, color=INK, fill="white", crossed=False, size=76):
    c.box(x-size/2, y-size/2, size, size, fill, color, 8, 1.8)
    c.text(x, y, value, 40, color, ha="center", weight="bold", mono=True)
    if crossed:
        c.line((x-size*.41, y+size*.34), (x+size*.41, y-size*.34), color, 2.6)


def _array(c, x_positions, y, values, label, label_x, crossed=None,
           promotion=None, label_y=None):
    c.text(label_x, label_y or y-64, label, 28, MUTED, mono=True)
    for i, (x, value) in enumerate(zip(x_positions, values)):
        color = RED if value == crossed else TEAL if value == promotion else INK
        fill = RP if value == crossed else TP if value == promotion else "white"
        c.box(x-38,y-56,76,112,fill,color,8,1.8)
        c.text(x,y-17,value,40,color,ha="center",weight="bold",mono=True)
        c.line((x-29,y+11),(x+29,y+11),FAINT,1.1)
        c.text(x,y+35,str(i),28,MUTED,ha="center",mono=True)
        if value == crossed:
            c.line((x-29,y+4),(x+29,y-39),RED,2.6)


def _curve(c, vertices, color=TEAL, width=2.8):
    """A cubic route whose control points are given in source-pixel coordinates."""
    codes=[Path.MOVETO]+[Path.CURVE4]*(len(vertices)-1)
    c.ax.add_patch(FancyArrowPatch(path=Path(vertices,codes),arrowstyle="-|>",
                                   mutation_scale=18,linewidth=width,color=color,
                                   capstyle="round",joinstyle="round",zorder=1))


def fig01(mobile):
    """Cancellation routes the first waiting ID into the new confirmed list."""
    Canvas, domain_examples, _ = _api()
    c = Canvas(mobile, 1000 if mobile else 670)
    if mobile:
        c.text(24, 35, "One cancellation", 32, weight="bold")
        c.text(24, 80, "capacity = 2 · cancel(A)", 28, MUTED, mono=True)
        c.text(24, 125, "Input snapshot", 30, weight="bold")
        _array(c, [130, 290], 250, ["A", "B"], "confirmed", 24,
               crossed="A", label_y=170)
        _array(c, [130, 290], 420, ["C", "D"], "waiting · FIFO", 24,
               promotion="C", label_y=340)
        # The shortest clear mobile route runs between the two state snapshots.
        _route(c, [(130,476), (130,524), (290,524), (290,644)])
        c.text(220, 494, "FIFO", 28, TEAL, ha="center", weight="bold")
        c.text(24, 575, "Returned state", 30, weight="bold")
        _array(c, [130, 290], 700, ["B", "C"], "confirmed", 24,
               promotion="C", label_y=620)
        _array(c, [130, 290], 880, ["D"], "waiting", 24, label_y=800)
        c.text(24, 970, "Input unchanged · list indices", 28, MUTED)
    else:
        c.text(32, 36, "Cancel A; promote the first waiting ID", 32, weight="bold")
        c.text(32, 85, "capacity = 2", 28, MUTED, mono=True)
        c.text(54, 140, "Input snapshot", 30, weight="bold")
        c.text(602, 140, "Returned state", 30, weight="bold")
        _array(c, [130, 270], 263, ["A", "B"], "confirmed", 54,
               crossed="A", label_y=184)
        _array(c, [660, 800], 263, ["B", "C"], "confirmed", 602,
               promotion="C", label_y=184)
        _array(c, [130, 270], 486, ["C", "D"], "waiting · FIFO", 54,
               promotion="C", label_y=575)
        _array(c, [660, 800], 486, ["D"], "waiting", 602, label_y=575)
        # Surviving IDs keep their relative order, not their old array indices.
        _route(c, [(310,263), (620,263)], FAINT, 2.2)
        _route(c, [(310,486), (620,486)], FAINT, 2.2)
        c.text(466, 237, "B: 1 → 0", 28, MUTED, ha="center", mono=True)
        c.text(466, 460, "D: 1 → 0", 28, MUTED, ha="center", mono=True)
        _route(c, [(130,430), (130,373), (800,373), (800,319)])
        c.text(480, 348, "C: waiting[0] → confirmed[1]", 28, TEAL,
               ha="center", mono=True)
        c.text(54, 635, "Input stays unchanged", 28, MUTED)
        c.text(602, 635, "New dict + new lists", 28, TEAL)
    c.fig._semantics = domain_examples()["cancellation"]
    c.fig._semantics["indices"] = {"B": [1,0], "D": [1,0], "C": ["waiting[0]", "confirmed[1]"]}
    c.fig._semantics["input_mutated"] = False
    return c.fig


def _outcome(c, x, y, value, color, stage, count):
    c.ax.add_patch(Rectangle((x-45,y-45),90,90,facecolor="white",edgecolor=color,
                            linewidth=2.2, zorder=3))
    c.text(x, y, "["+value+"]", 40, color, ha="center", weight="bold", mono=True)
    c.text(x, y-87, stage, 30, color, ha="center", weight="bold")
    c.text(x, y+91, count, 28, MUTED, ha="center")


def fig03(mobile):
    """A closed TDD loop with the recorded cancellation outputs and counts."""
    Canvas, _, state = _api()
    c = Canvas(mobile, 960 if mobile else 730)
    if mobile:
        c.text(24, 36, "One behavior, one cycle", 32, weight="bold")
        c.text(240,88,"Implement",28,RED,ha="center")
        for x,value,name,color,passed,failed in [
            (105,"A","RED",RED,4,1),(375,"B","GREEN",TEAL,5,0)]:
            c.text(x,125,name,30,color,ha="center",weight="bold")
            _cell(c,x,210,"["+value+"]",RED if failed else TEAL,
                  RP if failed else TP,size=90)
            c.text(x,285,f"{passed} pass",28,MUTED,ha="center")
            c.text(x,324,f"{failed} fail",28,RED if failed else MUTED,ha="center")
        _curve(c,[(158,210),(195,135),(285,135),(322,210)],RED,2.7)
        # Match the two cubic tangents while clearing the REFACTOR label.
        _curve(c,[(428,210),(475,250),(470,490),(425,550),
                  (380,610),(370,730),(293,730)],TEAL,2.7)
        _curve(c,[(187,730),(15,650),(15,440),(35,380),
                  (55,320),(35,210),(52,210)],MUTED,2.1)
        c.text(240,412,"Same test",30,INK,ha="center",weight="bold")
        c.text(240,456,"cancel(A)",28,INK,ha="center",mono=True)
        c.text(240,500,"expect [B]",28,TEAL,ha="center",mono=True)
        c.text(165,550,"Next behavior",28,MUTED,ha="center")
        c.text(260,600,"Restructure",28,TEAL,ha="center")
        c.text(240,652,"REFACTOR",30,GOLD,ha="center",weight="bold")
        _cell(c,240,730,"[B]",TEAL,TP,size=90)
        c.text(240,820,"copy_state()",28,GOLD,ha="center",mono=True)
        c.text(240,868,"5 pass",28,MUTED,ha="center")
        c.text(240,909,"0 fail · same tests",28,MUTED,ha="center")
    else:
        c.text(32,36,"One behavior, three checkpoints",32,weight="bold")
        _curve(c,[(243,235),(330,105),(630,105),(717,235)],RED,2.8)
        c.text(480,90,"Implement",28,RED,ha="center")
        _outcome(c,190,235,"A",RED,"RED","4 pass · 1 fail")
        _outcome(c,770,235,"B",TEAL,"GREEN","5 pass · 0 fail")
        # A test defines the reference value around which the structure changes.
        c.text(480,340,"Same test",30,INK,ha="center",weight="bold")
        c.text(480,385,"cancel(A)",30,INK,ha="center",mono=True)
        c.text(480,430,"expect [B]",30,TEAL,ha="center",mono=True)
        _curve(c,[(823,235),(950,235),(950,400),(850,450),
                  (750,500),(655,580),(533,580)],TEAL,2.8)
        c.text(785,400,"Restructure",28,TEAL,ha="center")
        c.text(480,510,"REFACTOR",30,GOLD,ha="center",weight="bold")
        _cell(c,480,580,"[B]",TEAL,TP,size=90)
        c.text(480,655,"copy_state()",28,GOLD,ha="center",mono=True)
        c.text(480,700,"5 pass · 0 fail · same tests",28,MUTED,ha="center")
        _curve(c,[(427,580),(135,590),(30,415),(30,295),
                  (30,235),(90,235),(137,235)],MUTED,2.0)
        c.text(190,440,"Next behavior",28,MUTED,ha="center")
    c.fig._semantics={"fixture":state(1,["A"],["B"]),"operation":"cancel(A)",
                      "red_actual_confirmed":["A"],"expected_confirmed":["B"],
                      "green_change":"promote waiting[0]","refactor":"extract copy_state()",
                      "result_after_green_and_refactor":"same tests pass",
                      "test_counts":{"red":{"passed":4,"failed":1},
                                     "green":{"passed":5,"failed":0},
                                     "refactor":{"passed":5,"failed":0}},
                      "source_snapshot_ids":["cancel-red","cancel-green","copy-refactor"],
                      "return_arrow":"choose a new test for the next behavior"}
    return c.fig


def _prep_fragment(c,x,y,width):
    """A selected preparation region: literal validator plus two list copies."""
    patch=c.box(x,y,width,176,GP,GOLD,3,1.7)
    patch.set_linestyle((0,(4,3)))
    c.text(x+15,y+34,"validate_state(state)",28,INK,mono=True)
    c.text(x+15,y+87,"Copy confirmed list",28,GOLD)
    c.text(x+15,y+140,"Copy waiting list",28,GOLD)


def _helper(c,x,y,width):
    c.box(x,y,width,191,TP,TEAL,5,1.8)
    c.text(x+width/2,y+37,"copy_state(state)",30,TEAL,ha="center",mono=True,weight="bold")
    c.line((x+20,y+69),(x+width-20,y+69),FAINT)
    c.text(x+width/2,y+107,"validate_state(state)",28,INK,ha="center",mono=True)
    c.text(x+width/2,y+157,"new dict + new lists",28,MUTED,ha="center")


def _caller(c,x,y,name):
    c.text(x,y,name+"(...)",30,INK,mono=True,weight="bold")
    c.text(x,y+48,"copy_state(state)",28,TEAL,mono=True)
    c.line((x,y+66),(x+287,y+66),TEAL,1.8)


def fig04(mobile):
    """Lift repeated preparation out of two operations into one real helper."""
    Canvas,_,_= _api()
    c=Canvas(mobile,1300 if mobile else 810)
    if mobile:
        c.text(24,35,"One helper, two callers",31,weight="bold")
        c.text(24,80,"Keep the public behavior",28,MUTED)
        c.text(24,143,"Before",30,MUTED,weight="bold")
        c.text(39,198,"register(...)",30,INK,mono=True,weight="bold")
        _prep_fragment(c,24,228,432)
        c.text(39,462,"cancel(...)",30,INK,mono=True,weight="bold")
        _prep_fragment(c,24,492,432)
        _route(c,[(457,315),(468,315),(468,714),(408,738)],GOLD,2.3)
        _route(c,[(240,684),(240,734)],GOLD,2.3)
        _helper(c,24,752,432)
        c.text(24,1003,"After · calls one helper",30,TEAL,weight="bold")
        _caller(c,24,1060,"register")
        _caller(c,24,1184,"cancel")
        _route(c,[(337,1108),(421,1108),(421,943)],TEAL,2.2)
        _route(c,[(337,1232),(449,1232),(449,943)],TEAL,2.2)
    else:
        c.text(32,36,"One helper, two callers",32,weight="bold")
        c.text(32,100,"Before",30,MUTED,weight="bold")
        c.text(47,151,"register(...)",30,INK,mono=True,weight="bold")
        c.text(522,151,"cancel(...)",30,INK,mono=True,weight="bold")
        _prep_fragment(c,32,184,423)
        _prep_fragment(c,507,184,421)
        _route(c,[(250,376),(330,424)],GOLD,2.5)
        _route(c,[(715,376),(634,424)],GOLD,2.5)
        _helper(c,220,440,520)
        c.text(32,653,"After",30,TEAL,weight="bold")
        _caller(c,55,705,"register")
        _caller(c,620,705,"cancel")
        _route(c,[(201,682),(330,631)],TEAL,2.5)
        _route(c,[(766,682),(634,631)],TEAL,2.5)
    c.fig._semantics={"public_operations":["register(state, participant_id)","cancel(state, participant_id)"],
                      "shared_boundary":"copy_state(state)",
                      "shared_policy":"validate_state(state); copy confirmed and waiting lists",
                      "operation_rules":{"register":"duplicate identity and capacity",
                                         "cancel":"removal and FIFO promotion"},
                      "equivalence_condition":"same state validation, copying, and operation rules",
                      "highlighted_regions":"exact validator invocation plus plain-language labels for two list-copy operations",
                      "gold_arrows":"extraction of duplicated preparation",
                      "teal_arrows":"calls to the extracted helper, not execution order",
                      "quality_score":None}
    return c.fig
