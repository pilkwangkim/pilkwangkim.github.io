#!/usr/bin/env python3
"""Generate exact, art-directed SVG diagrams for the ECC introduction.

All diagrams use the article's column-vector convention, Hc = 0 over F2.
The figures contain no sampled or externally measured data.
"""

from __future__ import annotations

from html import escape
from itertools import combinations
from pathlib import Path
import argparse
import shutil
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/2026-11-05-error-correcting-codes"
INK = "#203548"
MUTED = "#617282"
TEAL = "#087f83"
TEAL_PALE = "#e0f1ef"
RED = "#b84c46"
RED_PALE = "#f8e8e5"
GOLD = "#a46c28"
GOLD_PALE = "#f5ecd9"
GRID = "#d8e0e3"
PAPER = "#fcfbf8"
WHITE = "#ffffff"


class Figure:
    def __init__(self, width, height, title, desc):
        self.width, self.height = width, height
        self.nodes = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{escape(title)}</title>",
            f"<desc id=\"desc\">{escape(desc)}</desc>",
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9 Z" fill="context-stroke"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        ]

    def text(self, x, y, value, size=26, color=INK, anchor="start", weight="400", mono=False):
        font = "DejaVu Sans Mono, Menlo, monospace" if mono else "DejaVu Sans, Arial, sans-serif"
        self.nodes.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=2, arrow=False, dash=False):
        more = ' marker-end="url(#arrow)"' if arrow else ""
        more += ' stroke-dasharray="6 6"' if dash else ""
        self.nodes.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{more}/>')

    def rect(self, x, y, w, h, fill=WHITE, stroke=GRID, radius=7, width=1.5):
        self.nodes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def circle(self, x, y, r, fill=TEAL, stroke="none", width=1.5):
        self.nodes.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def bitrow(self, bits, x, y, cell=58, highlights=(), roles=False, numbered=False, color=RED, fill=RED_PALE):
        """Draw bits with aligned position and role labels; return right edge."""
        role_names = ("p1", "p2", "d1", "p4", "d2", "d3", "d4")
        for i, bit in enumerate(bits):
            cx = x + i * cell
            active = i + 1 in highlights
            self.rect(cx, y, cell - 7, 55, fill if active else WHITE,
                      color if active else GRID, radius=6, width=2 if active else 1.3)
            self.text(cx + (cell - 7) / 2, y + 38, bit, 32, color if active else INK, "middle", "500", mono=True)
            if numbered:
                self.text(cx + (cell - 7) / 2, y - 14, i + 1, 22, MUTED, "middle", mono=True)
            if roles:
                self.text(cx + (cell - 7) / 2, y + 85, role_names[i], 22, GOLD if i + 1 in (1, 2, 4) else TEAL, "middle", mono=True)
        return x + len(bits) * cell - 7

    def save(self, name):
        source = "\n".join(self.nodes + ["</svg>"]) + "\n"
        # Parse every output before saving, including accessible descriptions.
        ET.fromstring(source)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(source, encoding="utf-8")
        print(f"{name}: {self.width}x{self.height}")


def cover(mobile=False):
    w, h = (480, 650) if mobile else (960, 600)
    f = Figure(w, h, "Error-correcting codes", "A seven-bit Hamming word 0110011 suffers an error in position 5. Parity checks 1 and 4 fail, producing syndrome (1,0,1), whose weighted index is 5. The flipped bit is restored under the single-error assumption.")
    if mobile:
        f.text(30, 52, "Error-correcting", 35, weight="600")
        f.text(30, 96, "codes", 35, weight="600")
        f.text(30, 140, "1011 → 0110011", 24, MUTED, mono=True)
        x, step, y = 33, 60, 228
        f.bitrow("0110111", x, y, step, (5,), numbered=True)
        f.text(33, 187, "Received", 23, MUTED)
        # Mobile uses a labelled three-check fan-out rather than shrinking the
        # seven-column matrix to an unreadable or misaligned miniature.
        f.line(299, 288, 90, 367, RED, 1.8)
        f.line(299, 288, 390, 367, RED, 1.8)
        for row, (cx, title) in enumerate(zip((90, 240, 390), ("s1", "s2", "s4"))):
            col = RED if row in (0, 2) else TEAL
            f.circle(cx, 392, 25, RED_PALE if row in (0, 2) else TEAL_PALE, col)
            f.text(cx, 402, 1 if row in (0, 2) else 0, 29, col, "middle", mono=True)
            f.text(cx, 445, title, 24, col, "middle", mono=True)
        f.line(30, 480, 450, 480)
        f.text(30, 527, "Syndrome", 23, MUTED)
        f.text(231, 527, "1 0 1", 30, RED, mono=True)
        f.text(30, 569, "1 + 2·0 + 4·1 = 5", 25, INK, mono=True)
        f.text(30, 615, "Correct bit 5  →  0110011", 23, TEAL, mono=True)
    else:
        f.text(47, 70, "Error-correcting codes", 47, weight="600")
        f.text(49, 110, "Hamming(7,4): 1011 → 0110011", 26, MUTED, mono=True)
        f.text(49, 173, "One flipped bit", 23, MUTED)
        x, step, y = 323, 79, 191
        f.text(48, 226, "Received", 26, MUTED)
        f.bitrow("0110111", x, y, step, (5,), numbered=True)
        for row, (mask, cy, label) in enumerate(zip(("1010101", "0110011", "0001111"), (312, 369, 426), ("Check 1", "Check 2", "Check 4"))):
            col = RED if row in (0, 2) else TEAL
            f.text(48, cy + 8, label, 25, col)
            f.line(237, cy, 877, cy, GRID, 1)
            for j, v in enumerate(mask):
                f.circle(x + j * step + (step - 7) / 2, cy, 9 if v == "1" else 3.3, col if v == "1" else GRID)
            f.text(894, cy + 9, 1 if row in (0, 2) else 0, 28, col, "middle", mono=True)
        f.line(49, 472, 912, 472)
        f.text(49, 519, "Syndrome (s1, s2, s4) = (1, 0, 1)", 25, INK, mono=True)
        f.text(49, 560, "1 + 2·0 + 4·1 = 5", 26, RED, mono=True)
        f.text(913, 560, "Corrected: 0110011", 27, TEAL, "end", "500", mono=True)
    f.save("cover-mobile.svg" if mobile else "cover.svg")


def channel(mobile=False):
    w, h = (480, 820) if mobile else (960, 530)
    f = Figure(w, h, "Encoding makes some received words invalid", "Message 1011 encodes to the Hamming word 0110011. A bit flip at position 5 changes it to 0110111. The receiver does not know the original word, but failed parity checks locate position 5 when at most one bit has flipped. Correction restores 0110011 and extracts message 1011.")
    f.text(28 if mobile else 38, 45, "One message, one flipped bit", 27 if mobile else 31, weight="500")
    if mobile:
        x, step = 32, 60
        f.text(32, 91, "Message", 22, MUTED)
        f.text(289, 91, "1011", 29, TEAL, mono=True)
        f.text(32, 145, "Encode", 23, MUTED)
        f.bitrow("0110011", x, 162, step, (1, 2, 4), roles=True, color=GOLD, fill=GOLD_PALE)
        f.line(299, 260, 299, 302, RED, 2.3, arrow=True)
        f.text(170, 291, "flip 5", 23, RED)
        f.text(32, 338, "Receive", 23, MUTED)
        f.bitrow("0110111", x, 355, step, (5,))
        f.text(32, 456, "Check", 23, MUTED)
        f.text(32, 498, "(s1, s2, s4) = (1, 0, 1)", 24, RED, mono=True)
        f.text(32, 541, "1 + 2·0 + 4·1 = 5", 24, INK, mono=True)
        f.line(299, 563, 299, 608, TEAL, 2.3, arrow=True)
        f.text(170, 595, "flip 5", 23, TEAL)
        f.text(32, 641, "Correct", 23, MUTED)
        f.bitrow("0110011", x, 658, step, (5,), color=TEAL, fill=TEAL_PALE)
        f.text(32, 767, "Extract data positions 3, 5, 6, 7", 22, MUTED)
        f.text(448, 804, "1011", 29, TEAL, "end", mono=True)
    else:
        f.text(38, 98, "Message  1011", 27, TEAL, mono=True)
        f.text(922, 98, "Data positions: 3, 5, 6, 7", 24, MUTED, "end")
        f.text(38, 176, "Transmit", 26, MUTED)
        f.bitrow("0110011", 221, 137, 73, (1, 2, 4), roles=True, color=GOLD, fill=GOLD_PALE)
        f.text(922, 174, "checks pass", 24, TEAL, "end")
        f.line(546, 238, 546, 280, RED, 2.3, arrow=True)
        f.text(576, 266, "channel flips position 5", 24, RED)
        f.text(38, 335, "Receive", 26, MUTED)
        f.bitrow("0110111", 221, 296, 73, (5,))
        f.text(922, 332, "checks fail", 24, RED, "end")
        f.text(38, 412, "Syndrome", 25, MUTED)
        f.text(222, 412, "(1, 0, 1)  →  position 5", 26, RED, mono=True)
        f.line(38, 446, 921, 446)
        f.text(38, 492, "Correct: 0110011", 27, TEAL, mono=True)
        f.text(921, 492, "Decode: 1011", 27, TEAL, "end", mono=True)
    f.save("fig-01-channel-mobile.svg" if mobile else "fig-01-channel.svg")


def distance(mobile=False):
    w, h = (480, 595) if mobile else (960, 650)
    f = Figure(w, h, "Disjoint one-error neighborhoods of the repetition code", "The eight vertices are all three-bit words. Each cube edge changes exactly one bit. Teal vertices 000, 001, 010, 100 are within Hamming distance one of 000. Gold vertices 111, 110, 101, 011 are within distance one of 111. The neighborhoods share no word. Geometric edge lengths on the page do not measure Hamming distance.")
    f.text(27 if mobile else 38, 44, "One error still leaves a unique answer", 24 if mobile else 31, weight="500")
    if mobile:
        f.text(27, 83, "One edge = one bit flip", 22, MUTED)
        origin = (78, 407)
        a, b, c = (227, 0), (0, -189), (85, -90)
    else:
        f.text(38, 89, "3-bit repetition code: 000 and 111", 26, MUTED)
        origin = (235, 464)
        a, b, c = (270, 0), (0, -210), (150, -105)
    points = {}
    for i in range(8):
        bits = f"{i:03b}"
        xyz = tuple(map(int, bits))
        points[bits] = (origin[0] + xyz[0] * a[0] + xyz[1] * b[0] + xyz[2] * c[0], origin[1] + xyz[0] * a[1] + xyz[1] * b[1] + xyz[2] * c[1])
    for first, second in combinations(points, 2):
        if sum(x != y for x, y in zip(first, second)) == 1:
            f.line(*points[first], *points[second], GRID, 3)
    for bits, (x, y) in points.items():
        teal = bits.count("1") <= 1
        color, pale = (TEAL, TEAL_PALE) if teal else (GOLD, GOLD_PALE)
        center = bits in ("000", "111")
        f.rect(x - 35, y - 22, 70, 44, color if center else pale, color, 8, 2.3 if center else 1.3)
        f.text(x, y + 9, bits, 25, WHITE if center else color, "middle", "500", mono=True)
    ly = 492 if mobile else 552
    f.circle(39 if mobile else 51, ly - 8, 7, TEAL)
    f.text(56 if mobile else 70, ly, "d(x, 000) ≤ 1", 23 if mobile else 26, TEAL, mono=True)
    f.circle(39 if mobile else 511, ly + 36 if mobile else ly - 8, 7, GOLD)
    f.text(56 if mobile else 530, ly + 44 if mobile else ly, "d(x, 111) ≤ 1", 23 if mobile else 26, GOLD, mono=True)
    f.text(27 if mobile else 38, 576 if mobile else 616, "Every received word belongs to exactly one set.", 20 if mobile else 25, MUTED)
    f.save("fig-02-distance-mobile.svg" if mobile else "fig-02-distance.svg")


def parity(mobile=False):
    w, h = (480, 725) if mobile else (960, 585)
    f = Figure(w, h, "Parity-check memberships identify an error in position five", "Columns are positions 1 through 7, whose roles are p1, p2, d1, p4, d2, d3, d4. Row s1 includes 1,3,5,7; row s2 includes 2,3,6,7; row s4 includes 4,5,6,7. Each included entry is one. The highlighted fifth column is (1,0,1), so an error at position 5 fails precisely checks 1 and 4.")
    f.text(25 if mobile else 38, 44, "Each position has a unique check pattern", 23 if mobile else 30, weight="500")
    x0, step, top = (90, 49, 133) if mobile else (295, 76, 155)
    role = ("p1", "p2", "d1", "p4", "d2", "d3", "d4")
    f.text(25 if mobile else 38, 91, "Position 5 belongs to checks 1 and 4.", 22 if mobile else 26, MUTED)
    # The highlighted column stays aligned across position, role, and matrix.
    box_top, box_h = top - 32, 292 if mobile else 300
    f.rect(x0 + 4 * step - 22, box_top, 44, box_h, RED_PALE, "none", 5)
    for j in range(7):
        x = x0 + j * step
        f.text(x, top, j + 1, 25, RED if j == 4 else INK, "middle", "500", mono=True)
        f.text(x, top + 45, role[j], 21 if mobile else 23, GOLD if j in (0, 1, 3) else TEAL, "middle", mono=True)
    for i, (mask, label) in enumerate(zip(("1010101", "0110011", "0001111"), ("s1", "s2", "s4"))):
        y = top + 112 + i * 67
        f.text(25 if mobile else 56, y + 8, label, 25 if mobile else 28, RED if i in (0, 2) else TEAL, mono=True)
        if not mobile:
            f.text(135, y + 8, ("1,3,5,7", "2,3,6,7", "4,5,6,7")[i], 23, MUTED, mono=True)
        for j, value in enumerate(mask):
            x = x0 + j * step
            if value == "1":
                f.circle(x, y, 17, RED if j == 4 else TEAL)
                f.text(x, y + 8, "1", 22, WHITE, "middle", mono=True)
            else:
                f.circle(x, y, 16, PAPER, GRID)
                f.text(x, y + 8, "0", 22, MUTED, "middle", mono=True)
    bottom = top + 292
    f.line(25 if mobile else 38, bottom, w - 25 if mobile else w - 38, bottom)
    if mobile:
        f.text(25, 475, "A single error at position 5", 23, MUTED)
        f.text(25, 519, "s = He5 = column 5 of H", 23, INK, mono=True)
        f.text(25, 566, "(s1, s2, s4) = (1, 0, 1)", 23, RED, mono=True)
        f.text(25, 622, "j = s1 + 2s2 + 4s4", 25, INK, mono=True)
        f.text(25, 670, "  = 1 + 0 + 4 = 5", 25, RED, mono=True)
        f.text(25, 708, "All arithmetic in Hc uses modulo 2.", 21, MUTED)
    else:
        f.text(38, 510, "s = He5 = column 5 of H = (1, 0, 1)", 26, INK, mono=True)
        f.text(38, 555, "j = s1 + 2s2 + 4s4 = 1 + 0 + 4 = 5", 26, RED, mono=True)
    f.save("fig-03-parity-checks-mobile.svg" if mobile else "fig-03-parity-checks.svg")


def miscorrection(mobile=False):
    w, h = (480, 890) if mobile else (960, 585)
    f = Figure(w, h, "Two errors can trigger a wrong correction", "The transmitted Hamming word is 0110011. Channel errors at positions 3 and 5 produce 0100111, with syndrome index 6. A single-error decoder flips position 6 and returns a different valid word, 0100101, whose extracted message is 0101 rather than 1011. The original and returned words have Hamming distance three; the channel and the decoder together traverse the three-edge path.")
    f.text(26 if mobile else 38, 45, "A valid word is not proof of recovery", 24 if mobile else 31, weight="500")
    if mobile:
        f.text(26, 87, "Two channel errors; one mistaken correction", 21, MUTED)
        x, step = 32, 60
        rows = [("0110011", 153, "Transmitted", TEAL, (), TEAL_PALE), ("0100011", 335, "After channel flip 3", RED, (3,), RED_PALE), ("0100111", 517, "After channel flip 5", RED, (3, 5), RED_PALE), ("0100101", 728, "Decoder flips 6", GOLD, (3, 5, 6), GOLD_PALE)]
        for idx, (bits, y, title, col, hi, pale) in enumerate(rows):
            f.text(32, y - 22, title, 23, col)
            f.bitrow(bits, x, y, step, hi, color=col, fill=pale)
            if idx < 2:
                f.line(241, y + 82, 241, y + 124, RED, 2.3, arrow=True)
        f.text(32, 622, "s = (0, 1, 1)  →  j = 6", 24, RED, mono=True)
        f.line(241, 645, 241, 685, GOLD, 2.3, arrow=True)
        f.text(32, 830, "Hc′ = 0, but data = 0101", 24, GOLD, mono=True)
        f.text(32, 872, "The original data was 1011.", 22, MUTED)
    else:
        f.text(38, 88, "The decoder observes the received word, not the channel's history.", 25, MUTED)
        # An explicit path through the Hamming cube, not a Euclidean distance plot.
        xs = (135, 365, 595, 825)
        y = 214
        for a, b in zip(xs, xs[1:]):
            f.line(a + 99, y, b - 99, y, RED if a < xs[2] else GOLD, 2.5, arrow=True)
        words = ("0110011", "0100011", "0100111", "0100101")
        for i, (x, bits) in enumerate(zip(xs, words)):
            color = TEAL if i == 0 else GOLD if i == 3 else RED
            f.rect(x - 96, y - 32, 192, 64, TEAL_PALE if i == 0 else GOLD_PALE if i == 3 else WHITE, color, 7, 1.8)
            f.text(x, y + 10, bits, 30, color, "middle", mono=True)
        for x, label in zip(xs, ("Transmitted", "After flip 3", "Received", "Returned")):
            f.text(x, 160, label, 24, MUTED, "middle")
        for x, label, col in ((250, "flip 3", RED), (480, "flip 5", RED), (710, "flip 6", GOLD)):
            f.text(x, 294, label, 24, col, "middle")
        f.line(39, 330, 691, 330, RED, 2)
        f.text(365, 368, "Channel: two bit flips", 25, RED, "middle")
        f.line(729, 330, 920, 330, GOLD, 2)
        f.text(825, 368, "Decoder", 25, GOLD, "middle")
        f.text(38, 438, "Received syndrome: (0, 1, 1)  →  position 6", 26, INK, mono=True)
        f.line(38, 471, 921, 471)
        f.text(38, 519, "Hc′ = 0", 29, GOLD, mono=True)
        f.text(921, 519, "Data: 1011 → 0101", 29, GOLD, "end", mono=True)
        f.text(38, 560, "Three flips separate the two valid codewords.", 25, MUTED)
    f.save("fig-04-miscorrection-mobile.svg" if mobile else "fig-04-miscorrection.svg")


def verify_examples():
    h = ((1, 0, 1, 0, 1, 0, 1), (0, 1, 1, 0, 0, 1, 1), (0, 0, 0, 1, 1, 1, 1))
    syndrome = lambda word: tuple(sum(int(b) * a for a, b in zip(row, word)) % 2 for row in h)
    assert syndrome("0110011") == (0, 0, 0)
    assert syndrome("0110111") == (1, 0, 1)
    assert syndrome("0100111") == (0, 1, 1)
    assert syndrome("0100101") == (0, 0, 0)
    assert sum(a != b for a, b in zip("0110011", "0100101")) == 3
    assert "".join("0100101"[i - 1] for i in (3, 5, 6, 7)) == "0101"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="Also render PNG covers with rsvg-convert.")
    args = parser.parse_args()
    verify_examples()
    for renderer in (cover, channel, distance, parity, miscorrection):
        for compact in (False, True):
            renderer(compact)
    if args.png:
        rasterizer = shutil.which("rsvg-convert")
        if not rasterizer:
            raise SystemExit("--png requires rsvg-convert (librsvg). SVG outputs were generated.")
        for stem, width, height in (("cover", 1600, 1000), ("cover-mobile", 960, 1300)):
            subprocess.run([rasterizer, "-w", str(width), "-h", str(height), str(OUT / f"{stem}.svg"), "-o", str(OUT / f"{stem}.png")], check=True)
            print(f"{stem}.png: {width}x{height}")
