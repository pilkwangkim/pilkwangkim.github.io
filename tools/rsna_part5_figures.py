"""Draw two conceptual English figures for the unpublished RSNA Part 5.

These diagrams state evidence boundaries and a conditional comparison. They do
not read research artifacts, fit a combination, or assert numerical performance.
Only the two part5 PNG/SVG pairs are written.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


BLOG = Path(__file__).resolve().parents[1]
OUT = BLOG / "assets/img/posts/rsna-knee-series"
C = dict(ink="#183344", muted="#526572", teal="#277F85", blue="#426E9C",
         grid="#DCE4E7", light="#F1F5F5", white="#FFFFFF")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15,
    "text.color": C["ink"], "svg.fonttype": "none",
    "savefig.facecolor": "white", "svg.hashsalt": "rsna-part5",
})


def frame(title):
    fig = plt.figure(figsize=(12, 7.2), facecolor="white")
    fig.text(.045, .945, "RSNA KNEE  /  PART 5", fontsize=12,
             weight="bold", color=C["teal"])
    fig.text(.045, .885, title, fontsize=21, weight="bold")
    ax = fig.add_axes([.045, .15, .91, .65])
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    return fig, ax, []


def text(ax, x, y, value, size=15, color="ink", weight="normal", **kw):
    return ax.text(x, y, value, ha=kw.pop("ha", "center"),
                   va=kw.pop("va", "center"), fontsize=size,
                   color=C[color], weight=weight, linespacing=1.45, **kw)


def box(ax, checks, x, y, w, h, title, body=None, color="teal", size=15):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.20,rounding_size=1.0",
                          facecolor=C["light"], edgecolor=C[color], linewidth=1.25)
    ax.add_patch(patch)
    if body:
        heading = text(ax, x+w/2, y+h*.71, title, size=size, weight="bold")
        content = text(ax, x+w/2, y+h*.31, body, size=size-1, color="muted")
        checks.extend([(heading, patch), (content, patch)])
    else:
        checks.append((text(ax, x+w/2, y+h/2, title,
                            size=size, weight="bold"), patch))
    return patch


def arrow(ax, a, b, color="muted", bend=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                                connectionstyle=f"arc3,rad={bend}",
                                mutation_scale=14, linewidth=1.35,
                                color=C[color]))


def save(fig, name, checks):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures = []
    for item, patch in checks:
        t = item.get_window_extent(renderer)
        p = patch.get_window_extent(renderer)
        if t.x0 < p.x0+3 or t.x1 > p.x1-3 or t.y0 < p.y0+3 or t.y1 > p.y1-3:
            failures.append(item.get_text())
    for item in fig.findobj(match=matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        t = item.get_window_extent(renderer)
        f = fig.bbox
        if t.x0 < f.x0 or t.x1 > f.x1 or t.y0 < f.y0 or t.y1 > f.y1:
            failures.append("Outside figure: " + item.get_text())
    if failures:
        raise RuntimeError(f"Text containment failed in {name}: {failures}")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=160)
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def oof_scope():
    fig, ax, checks = frame("OOF evidence has a defined scope")
    text(ax, .2, 96, "WITHIN THE CLEAN OWN BANK", size=12,
         weight="bold", color="teal", ha="left")
    box(ax, checks, .5, 67, 27, 22, "Clean OOF bank", "Complete own families")
    box(ax, checks, 36.5, 67, 27, 22, "Matched predictions", "Same held-out studies")
    box(ax, checks, 72.5, 67, 27, 22, "Cross-fitted choice", "Selection within the bank")
    arrow(ax, (28.3, 78), (35.8, 78))
    arrow(ax, (64.3, 78), (71.8, 78))

    ax.plot([0, 100], [56, 56], color=C["grid"], linewidth=1)
    text(ax, .2, 47, "ADDING A SHARED FINAL-ONLY PREDICTOR", size=12,
         weight="bold", color="blue", ha="left")
    box(ax, checks, .5, 13, 27, 25, "Shared public parent", "Full-data or\nunknown ancestry",
        color="blue", size=15)
    box(ax, checks, 36.5, 13, 27, 25, "Combined package", "Own components\n+ public parent",
        color="blue", size=15)
    box(ax, checks, 72.5, 13, 27, 25, "Public / Private", "Transfer of the\ncomplete package",
        color="blue", size=15)
    arrow(ax, (28.3, 25.5), (35.8, 25.5), color="blue")
    arrow(ax, (64.3, 25.5), (71.8, 25.5), color="blue")
    text(ax, 50, 2.5, "No compatible joint held-out evidence is implied.",
         size=14, color="muted")
    fig.text(.045, .082,
             "Clean own components retain their OOF identity. The complete combination remains final-only.",
             fontsize=12.5, color=C["muted"])
    fig.text(.045, .039,
             "Conceptual evidence map. Adding clean predictions does not make a full-data or unknown-ancestry parent clean OOF.",
             fontsize=11.2, color=C["muted"])
    save(fig, "part5-oof-scope", checks)


def shared_baseline():
    fig, ax, checks = frame("A shared baseline changes the decision question")
    box(ax, checks, .5, 73, 27, 21, "Own predictor  u", "Reusable model evidence")
    box(ax, checks, .5, 44, 27, 21, "Shared baseline  b", "Available to many teams", color="blue")
    box(ax, checks, 40, 58.5, 25, 21, "Declared operator", r"$p=C(u,b)$")
    box(ax, checks, 78, 58.5, 21.5, 21, "Final predictor  p", "Complete package")
    arrow(ax, (28.2, 83.5), (39.3, 73), color="teal")
    arrow(ax, (28.2, 54.5), (39.3, 65), color="blue")
    arrow(ax, (65.8, 69), (77.3, 69))

    box(ax, checks, .5, 1, 47.5, 30, "Reusable evidence",
        "Held-out quality and identity of u\nremain useful beyond this package.", size=15)
    box(ax, checks, 52, 1, 47.5, 30, "Conditional predictive gain",
        r"$\Delta M=M\!\left(C(u,b)\right)-M(b)$" "\nGain over the shared baseline", size=15)
    fig.text(.045, .082,
             "The comparison needs matched, valid evaluation. Standalone quality and training effort do not determine the gain.",
             fontsize=12, color=C["muted"])
    fig.text(.045, .039,
             "Conceptual only. M is the same metric on the same evaluation set; C is a declared operator, with no weights or gains asserted.",
             fontsize=10.8, color=C["muted"])
    save(fig, "part5-shared-baseline", checks)


if __name__ == "__main__":
    oof_scope()
    shared_baseline()
    rows = []
    for stem in ("part5-oof-scope", "part5-shared-baseline"):
        for suffix in ("png", "svg"):
            p = OUT / f"{stem}.{suffix}"
            row = {"path": str(p.relative_to(BLOG)),
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                   "bytes": p.stat().st_size}
            if suffix == "png":
                with Image.open(p) as im:
                    row["dimensions"] = list(im.size)
            rows.append(row)
    print(json.dumps({"text_containment": "pass", "assets": rows}, indent=2))
