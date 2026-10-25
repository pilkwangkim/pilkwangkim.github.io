"""Result-to-decision diagram for the RSNA technical learning section.

Rebuild: /opt/anaconda3/bin/python tools/rsna_solution_learning_figure.py
Writes only result-to-decision{,-mobile}.{png,svg} and learning-figure-sources.json.
This is an editorial decision diagram, not a scientific experiment or score plot.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/img/posts/rsna-knee-solutions'
# Match the existing English analysis figures; do not execute their generator.
C = {'ink': '#183344', 'muted': '#526572', 'teal': '#277F85',
     'teal_light': '#DCECEE', 'blue': '#426E9C', 'blue_light': '#EBF1F7',
     'ochre': '#AB7440', 'ochre_light': '#F7EEE3', 'grid': '#DCE4E7',
     'light': '#F3F6F7', 'white': '#FFFFFF', 'gray': '#8C9BA3'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': C['ink'],
                     'svg.fonttype': 'none', 'savefig.facecolor': 'white'})

COLUMNS = ['Observed result', 'Defensible conclusion', 'Next action']
ROWS = [
    ['Invalid input or output', 'The hypothesis was not tested', 'Repair and resume'],
    ['Weak packet or early checkpoint', 'This configuration underperformed',
     'Resolve a viable bottleneck, or stop'],
    ['Local/transfer conflict', 'Evidence channels disagree',
     'Check target, ancestry, and operator'],
    ['Valid decision-facing contrast', 'Evidence about the declared choice',
     'Scale, retain, or redirect'],
]
DESKTOP_ROWS = [
    ['Invalid input\nor output', 'The hypothesis\nwas not tested', 'Repair and resume'],
    ['Weak packet or\nearly checkpoint', 'This configuration\nunderperformed',
     'Resolve a viable\nbottleneck, or stop'],
    ['Local/transfer\nconflict', 'Evidence channels\ndisagree',
     'Check target,\nancestry, and operator'],
    ['Valid decision-facing\ncontrast', 'Evidence about the\ndeclared choice',
     'Scale, retain,\nor redirect'],
]


def text(ax, x, top, value, fs, *, ha='left', va='top', color='ink', bold=False):
    return ax.text(x, ax.get_ylim()[1] - top, value, fontsize=fs,
                   ha=ha, va=va, color=C[color], weight='bold' if bold else 'normal',
                   linespacing=1.28)


def card(ax, x, top, width, height, fill):
    p = FancyBboxPatch((x, ax.get_ylim()[1] - top - height), width, height,
                      boxstyle='round,pad=0,rounding_size=12',
                      facecolor=C[fill], edgecolor=C['grid'], linewidth=1.2)
    ax.add_patch(p)
    return p


def arrow(ax, x0, top0, x1, top1):
    h = ax.get_ylim()[1]
    patch = FancyArrowPatch((x0, h - top0), (x1, h - top1),
                            arrowstyle='-|>', mutation_scale=14,
                            linewidth=1.6, color=C['gray'], shrinkA=0, shrinkB=0)
    ax.add_patch(patch)


def check_bounds(fig, constrained):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text()]
    errors = []
    for t in texts:
        b = t.get_window_extent(renderer)
        if b.x0 < 2 or b.y0 < 2 or b.x1 > fig.bbox.width - 2 or b.y1 > fig.bbox.height - 2:
            errors.append('figure edge: ' + t.get_text())
    for t, p in constrained:
        b, pb = t.get_window_extent(renderer), p.get_window_extent(renderer)
        if b.x0 < pb.x0 + 8 or b.x1 > pb.x1 - 8 or b.y0 < pb.y0 + 8 or b.y1 > pb.y1 - 8:
            errors.append('card edge: ' + t.get_text())
    # Category labels and body lines must also have their own space.
    boxes = [(t, t.get_window_extent(renderer)) for t in texts]
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1:]:
            if a.overlaps(b):
                errors.append('text overlap: ' + ta.get_text() + ' / ' + tb.get_text())
    if errors:
        raise ValueError(errors)
    return {'figure_edges': 'passed', 'card_text_bounds': 'passed',
            'text_overlap_check': 'passed'}


def draw(mobile=False):
    width, height, dpi = (760, 2100, 100) if mobile else (1800, 1250, 150)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis('off')
    constrained = []
    if mobile:
        text(ax, 42, 75, 'RSNA KNEE  /  SOLUTION ANALYSIS', 12, bold=True, color='teal')
        text(ax, 42, 150, 'What a result\ncan justify', 26, bold=True)
        text(ax, 42, 285, 'The scope of the evidence determines\nthe scope of the next investment', 16, color='muted')
        for r, row in enumerate(ROWS):
            outer_top = 420 + r * 392
            card(ax, 40, outer_top, 680, 364, 'white')
            for c, value in enumerate(row):
                top = outer_top + 18 + c * 118
                p = card(ax, 58, top, 644, 92, ['light', 'blue_light', 'teal_light'][c])
                label = text(ax, 80, top + 13, COLUMNS[c], 12, bold=True,
                             color=['muted', 'blue', 'teal'][c])
                body = text(ax, 80, top + 48, value, 17.5, va='center')
                constrained.extend([(label, p), (body, p)])
                if c < 2:
                    arrow(ax, 380, top + 94, 380, top + 116)
        text(ax, 42, 2010, 'Four independent cases, not a sequence.', 13, color='muted')
    else:
        text(ax, 100, 50, 'RSNA KNEE  /  SOLUTION ANALYSIS', 12, bold=True, color='teal')
        text(ax, 100, 112, 'What a result can justify', 26, bold=True)
        text(ax, 100, 205, 'The scope of the evidence determines the scope of the next investment',
             15.5, color='muted')
        xs = [90, 660, 1230]
        for c, label in enumerate(COLUMNS):
            text(ax, xs[c] + 240, 313, label, 17, ha='center', bold=True,
                 color=['muted', 'blue', 'teal'][c])
        for r, row in enumerate(DESKTOP_ROWS):
            top = 380 + r * 187
            for c, value in enumerate(row):
                p = card(ax, xs[c], top, 480, 143, ['light', 'blue_light', 'teal_light'][c])
                t = text(ax, xs[c] + 240, top + 71.5, value, 17,
                         ha='center', va='center', bold=c == 2)
                constrained.append((t, p))
                if c < 2:
                    arrow(ax, xs[c] + 500, top + 71.5, xs[c + 1] - 20, top + 71.5)
        text(ax, 100, 1158, 'Four independent cases, not a sequence.', 14, color='muted')
    checks = check_bounds(fig, constrained)
    stem = 'result-to-decision' + ('-mobile' if mobile else '')
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=dpi)
    with Image.open(OUT / f'{stem}.png') as image:
        assert image.size == (width, height)
    plt.close(fig)
    return {'stem': stem, 'width': width, 'height': height, **checks}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dimensions = [draw(False), draw(True)]
    files = []
    for row in dimensions:
        for ext in ['png', 'svg']:
            path = OUT / f"{row['stem']}.{ext}"
            files.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        'figure': 'result-to-decision',
        'scope': 'Conceptual research-decision diagram. Four independent evidence situations, not a sequential pipeline or a promise that every negative justifies another experiment.',
        'subtitle': 'The scope of the evidence determines the scope of the next investment',
        'columns': COLUMNS, 'rows': ROWS,
        'source': 'Author synthesis for the RSNA technical learning section; no new empirical measurements.',
        'style_reference': 'tools/rsna_solution_analysis_deep_figures.py',
        'dimensions': dimensions, 'files': files,
    }
    (OUT / 'learning-figure-sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'dimensions': dimensions, 'files_written': len(files)}, indent=2))


if __name__ == '__main__':
    main()
