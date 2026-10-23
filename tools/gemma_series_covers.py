"""Covers for the Gemma 4 Developer Agent working notes.

One technical-note template for every part; only the text block changes.
Run from the repository root:  python tools/gemma_series_covers.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/gemma-4-series"
ART = ROOT / "assets/img/gemma4-developer-agent/hero.png"
FONTS = Path("/System/Library/Fonts/Supplemental")
REGULAR = FONTS / "Arial.ttf"
BOLD = FONTS / "Arial Bold.ttf"

W, H = 2200, 1240
BG = (251, 251, 249)
INK = (17, 17, 17)
MUTED = (96, 104, 112)
RULE = (205, 210, 214)
ACCENT = (36, 103, 214)

PARTS = [
    {
        "number": 1,
        "kind": "INTRODUCTION",
        "title": "FROM ISSUE TO VERIFIED PATCH",
        "subtitle": "What we submit, how one attempt is graded, and where a first experiment should begin.",
        "date": "25 SEP 2026",
        "columns": [
            ("01 / TASK", "Repair, not prediction",
             "The agent must find the faulty code before it can change it."),
            ("02 / SCORE", "Only verified patches count",
             "A task counts when fresh tests pass on the extracted patch."),
            ("03 / BUDGET", "One twelve-hour budget",
             "Setup and agent time share a single limit across about 120 tasks."),
        ],
    },
    {
        "number": 2,
        "kind": "TECHNICAL NOTE",
        "palette": {
            "background": "#fbfbf8", "ink": "#183044", "muted": "#5c6d78",
            "rule": "#d7dfe1", "accent": "#25806f",
        },
        "artwork_credit": False,
        "title": "MEASURING AN AGENT WHILE ITS EVALUATOR MOVED",
        "subtitle": "Repeated calls, sparse hosted feedback, and the cost of missing a twelve-hour deadline.",
        "date": "23 OCT 2026",
        "columns": [
            ("01 / REPEAT", "Repeated calls cluster",
             "After three identical calls, observed continuation pooled to 0.97."),
            ("02 / SCORE", "One score, one observation",
             "The same bundle scored 0.10 and 0.15 within one host epoch."),
            ("03 / TIME", "Time changes the outcome",
             "A twelve-hour overrun leaves the submission without a score."),
        ],
    },
]


def font(path, size):
    return ImageFont.truetype(str(path), size)


def wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def emblem(art):
    """The four-point star emblem from the official artwork, on a light square."""
    w, h = art.size
    box = (int(w * 0.268), int(h * 0.352), int(w * 0.362), int(h * 0.537))
    star = art.crop(box).convert("RGB").resize((96, 96), Image.LANCZOS)
    return star


def render(part):
    palette = part.get("palette", {})
    background, ink = palette.get("background", BG), palette.get("ink", INK)
    muted, rule = palette.get("muted", MUTED), palette.get("rule", RULE)
    accent = palette.get("accent", ACCENT)
    img = Image.new("RGB", (W, H), background)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 12], fill=ink)
    d.rectangle([0, 12, W, 18], fill=accent)

    art = Image.open(ART).convert("RGB")
    img.paste(emblem(art), (110, 92))
    d.text((232, 98), "GEMMA 4 DEVELOPER AGENT", font=font(BOLD, 30), fill=ink)
    d.text((232, 140), "OPEN-MODEL CODING AGENTS", font=font(BOLD, 21), fill=muted)

    d.text((110, 236), f"PART {part['number']}  /  {part['kind']}", font=font(BOLD, 24), fill=accent)
    title_font = font(BOLD, 74)
    y = 292
    for line in wrap(d, part["title"], title_font, 1130):
        d.text((106, y), line, font=title_font, fill=ink)
        y += 86
    sub_font = font(REGULAR, 31)
    y += 22
    for line in wrap(d, part["subtitle"], sub_font, 1300):
        d.text((110, y), line, font=sub_font, fill=muted)
        y += 42

    # Official artwork panel
    panel_w, panel_h = 560, 280
    aw, ah = art.size
    scale = max(panel_w / aw, panel_h / ah)
    resized = art.resize((int(aw * scale), int(ah * scale)), Image.LANCZOS)
    left = (resized.width - panel_w) // 2
    top = (resized.height - panel_h) // 2
    panel = resized.crop((left, top, left + panel_w, top + panel_h))
    px, py = 1530, 92
    img.paste(panel, (px, py))
    d.rectangle([px - 2, py - 2, px + panel_w + 1, py + panel_h + 1], outline=ink, width=2)
    if part.get("artwork_credit", True):
        d.text((px, py + panel_h + 22), "OFFICIAL COMPETITION ARTWORK, ADAPTED", font=font(BOLD, 18), fill=muted)

    d.line([110, 680, W - 110, 680], fill=ink, width=2)
    col_x = [110, 730, 1350]
    col_w = 560
    label_size, head_size, body_size, body_step = (24, 34, 30, 40) if part["number"] == 2 else (20, 30, 24, 34)
    for i, (label, head, body) in enumerate(part["columns"]):
        x = col_x[i]
        label_color = accent if part["number"] == 2 or i == 2 else muted
        d.text((x, 748), label, font=font(BOLD, label_size), fill=label_color)
        d.text((x, 806), head, font=font(BOLD, head_size), fill=ink)
        by = 872
        for line in wrap(d, body, font(REGULAR, body_size), col_w - 30):
            d.text((x, by), line, font=font(REGULAR, body_size), fill=muted)
            by += body_step
        if i:
            d.line([x - 30, 730, x - 30, 1010], fill=rule, width=2)

    d.line([110, 1068, W - 110, 1068], fill=rule, width=2)
    foot = font(BOLD, 20)
    d.text((110, 1110), f"PILKWANG KIM  /  {part['date']}", font=foot, fill=muted)
    label = f"SERIES {part['number']}"
    d.text((W - 110 - d.textlength(label, font=foot), 1110), label, font=foot, fill=muted)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"cover-{part['number']}"
    img.save(stem.with_suffix(".png"), optimize=True)
    img.save(stem.with_suffix(".webp"), quality=92, method=6)
    return stem


if __name__ == "__main__":
    for part in PARTS:
        print(render(part))
