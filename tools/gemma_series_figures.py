"""Rebuild the data figures of Gemma 4 Developer Agent, Part 2.

Reads frozen project records only (no network, no experiments):
  the frozen submission snapshot, the 2026-10-22 leaderboard snapshot, and the trace archive of
  benchmark B7f79efa (E0005 thinking off vs E0050 reasoning 512, 22 dev tasks).
Run with a Python environment containing matplotlib:  python tools/gemma_series_figures.py
"""
import csv
import json
import math
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

BLOG = Path(__file__).resolve().parents[1]
GEMMA = Path("/Volumes/PKKim_990/Kaggle/2026/26_Gemma4")
OUT = BLOG / "assets/img/posts/gemma-4-series"
SUBMISSIONS = BLOG / "_draft/_checks/gemma-part2-review-20261023/submissions-asof.csv"
LEADERBOARD = GEMMA / "data/refs/refresh-20261022T0615Z/lb/gemma-4-developer-agent-publicleaderboard-2026-10-22T06:15:56.csv"
TRACES = GEMMA / ("data/runs/workers/runpod-ada/55050a6a6963d6d17762/B7f79efa3e8a74e4eb6c1/"
                  "evidence-20261004T075430374663-25df1a66.zip")

BG, INK, MUTED, GRID, NEUTRAL = "#fbfbf8", "#183044", "#5c6d78", "#d7dfe1", "#a2adb4"
# Fixed agent identities across all figures. Hypothetical scenarios use grays.
BLUE, ORANGE, AQUA = "#3b6fa1", "#bc6537", "#25806f"
OFF_LABEL = "E0005 · thinking off"
ON_LABEL = "E0050 · reasoning 512"
REFERENCE_LABEL = "E0052 · Budget-Fit reference"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 12, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": BG,
    "figure.facecolor": BG, "savefig.facecolor": BG, "svg.fonttype": "path",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def heading(fig, title, subtitle, x=0.085):
    """One left-aligned title hierarchy for the technical-note figures."""
    fig.text(x, 0.96, title, fontsize=15, fontweight="bold", va="top", color=INK)
    fig.text(x, 0.905, subtitle, fontsize=10.5, va="top", color=MUTED)


def recessive(ax, grid_axis="y"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_leaderboard():
    rows = list(csv.DictReader(open(LEADERBOARD, encoding="utf-8-sig")))
    counts = Counter(round(float(r["Score"]), 2) for r in rows)
    scores = sorted(counts)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.79, bottom=0.19)
    recessive(ax)
    xs = range(len(scores))
    ax.bar(xs, [counts[s] for s in scores], width=0.78, edgecolor=BG, linewidth=2,
           color=[AQUA if abs(s - 0.18) < 1e-9 else NEUTRAL for s in scores])
    for x, s in zip(xs, scores):
        ax.text(x, counts[s] + 8, f"{counts[s]}", ha="center", va="bottom", fontsize=10.5, color=MUTED)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{s:.2f}\n{round(s * 58 + 0.49)}" for s in scores], fontsize=10.5)
    i18, i24 = scores.index(0.18), scores.index(0.24)
    ax.annotate("Reference: 0.18\nrank 398", (i18, counts[0.18] + 12),
                xytext=(i18 + 0.6, counts[0.18] + 108), fontsize=11, color=AQUA,
                arrowprops=dict(arrowstyle="-", color=AQUA, linewidth=1))
    ax.annotate("Top score\n4 teams", (i24, counts[0.24] + 18), xytext=(i24, 105), fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))
    ax.set_ylim(0, max(counts.values()) * 1.20)
    ax.set_ylabel("Teams")
    ax.set_xlabel("Displayed Public score / inferred solved tasks (second row, assuming N = 58)", labelpad=12)
    heading(fig, "Public leaderboard: team-best scores",
            f"22 Oct 2026 · 06:15 UTC · {len(rows):,} scored teams · teal marks the reference's score group")
    save(fig, "part2-fig1-leaderboard")


def fig_hosted_scores():
    rows = list(csv.DictReader(open(SUBMISSIONS)))
    groups = {
        "E0005": (OFF_LABEL, BLUE, "o"),
        "E0050": (ON_LABEL, ORANGE, "s"),
        "E0052": (REFERENCE_LABEL, AQUA, "D"),
    }
    fig = plt.figure(figsize=(10.5, 7.3))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.7, 1.25], left=0.085, right=0.985,
                           top=0.79, bottom=0.12, hspace=0.12)
    ax = fig.add_subplot(grid[0])
    status = fig.add_subplot(grid[1], sharex=ax)
    recessive(ax)
    events = [("2026-09-25 18:36", "data reissued"), ("2026-09-27 10:37", "v23–25"),
              ("2026-09-30 19:08", "v26–28"), ("2026-10-08 19:18", "v29–30"), ("2026-10-12 19:51", "v31")]
    for i, (when, label) in enumerate(events):
        t = datetime.strptime(when, "%Y-%m-%d %H:%M")
        ax.axvline(t, color=GRID, linewidth=1.2, linestyle=(0, (4, 3)), zorder=0)
        ax.text(t, 0.278 if i == 0 else 0.263, label, ha="center", va="bottom", fontsize=10.5, color=MUTED)
        status.axvline(t, color=GRID, linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
    ax.hlines(0.24, datetime(2026, 10, 2), datetime(2026, 10, 23, 9), color=MUTED,
              linewidth=1.0, linestyles=(0, (1, 2)), zorder=0)
    ax.text(datetime(2026, 10, 2, 10), 0.244, "Observed leaderboard best: 0.24 since Oct 2",
            fontsize=10.5, color=MUTED, va="bottom")
    series = {k: ([], []) for k in groups}
    other_x, other_y, errors, pending, same_zip = [], [], [], [], []
    for r in rows:
        t = datetime.strptime(r["date"][:16], "%Y-%m-%dT%H:%M")
        if r["status"] == "PENDING":
            pending.append((t, r["experiment_id"]))
        elif not r["public_lb"]:
            errors.append((t, r["experiment_id"]))
        elif r["experiment_id"] in series:
            series[r["experiment_id"]][0].append(t)
            series[r["experiment_id"]][1].append(float(r["public_lb"]))
        else:
            other_x.append(t)
            other_y.append(float(r["public_lb"]))
            if r["experiment_id"] == "E0053":
                same_zip.append((t, float(r["public_lb"])))
    ax.scatter(other_x, other_y, s=64, marker="^", color=NEUTRAL, edgecolor=BG, linewidth=1.5, zorder=3,
               label="Other variants")
    for key, (label, color, marker) in groups.items():
        ax.scatter(*series[key], s=78, marker=marker, color=color, edgecolor=BG, linewidth=2, zorder=4, label=label)
    for t, score in same_zip:
        ax.annotate("E0053 · same ZIP", (t, score), xytext=(-10, -23), textcoords="offset points",
                    ha="right", va="top", fontsize=10.5, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=NEUTRAL, linewidth=0.8))
    # A separate categorical axis ensures unavailable scores never look like zeros.
    status.set_facecolor("#f0f3f3")
    for t, eid in errors:
        y = 2 if eid == "E0057" else 1
        status.scatter(t, y, s=55, marker="x", color=INK, linewidth=1.8, zorder=3)
        status.annotate(eid, (t, y), xytext=(-9, 0), textcoords="offset points",
                        va="center", ha="right", fontsize=10.5)
    for t, eid in pending:
        status.scatter(t, 0, s=65, marker="o", facecolors="none", edgecolors=INK,
                       linewidth=1.5, zorder=3)
        status.annotate(f"{eid} · E4B", (t, 0), xytext=(-9, 0), textcoords="offset points",
                        va="center", ha="right", fontsize=10.5)
    status.set_ylim(-0.55, 2.55)
    status.set_yticks([0, 1, 2], ["Pending", "Error", "12 h timeout"], fontsize=10.5)
    status.tick_params(axis="both", length=0)
    for spine in status.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelbottom=False)
    ax.set_ylim(0, 0.295)
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_ylabel("Public score")
    status.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    status.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    status.set_xlabel("Submission date (UTC)", labelpad=10)
    ax.set_xlim(datetime(2026, 9, 24), datetime(2026, 10, 23, 9))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="lower left",
               bbox_to_anchor=(0.075, 0.815), ncol=4, frameon=False, fontsize=11,
               handletextpad=0.25, columnspacing=1.0)
    heading(fig, "Hosted scores across public artifact updates",
            f"{len(rows)} frozen submission records · dashed lines mark public artifact publications")
    save(fig, "part2-fig2-hosted-scores")


def call_sequences(eid):
    runs = []
    with zipfile.ZipFile(TRACES) as z:
        for name in sorted(z.namelist()):
            if f"/artifacts/{eid}/runs/" in name and "/raw/traces/trace_" in name:
                trace = json.loads(z.read(name))
                seq = [c["function_name"] + json.dumps(c.get("arguments"), sort_keys=True)
                       for s in trace.get("steps", []) for c in (s.get("tool_calls") or [])]
                runs.append(seq)
    return runs


BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 9, "5–9"), (10, 19, "10–19"), (20, 10**9, "20+")]


def hazard(runs):
    num, den = Counter(), Counter()
    for r in runs:
        k = 1
        for a, b in zip(r, r[1:]):
            bucket = next(i for i, (lo, hi, _) in enumerate(BUCKETS) if lo <= k <= hi)
            den[bucket] += 1
            if a == b:
                num[bucket] += 1
                k += 1
            else:
                k = 1
    return {i: (num[i] / den[i], den[i]) for i in den}


def fig_repeat_hazard():
    off, on = call_sequences("E0005"), call_sequences("E0050")
    h_off, h_on = hazard(off), hazard(on)
    pairs = sum(len(r) - 1 for r in off if r)
    q_off = sum(sum(a == b for a, b in zip(r, r[1:])) for r in off) / pairs
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.79, bottom=0.28)
    recessive(ax)
    xs = sorted(h_off)
    ax.plot(xs, [h_off[i][0] for i in xs], color=BLUE, linewidth=2, marker="o", markersize=8,
            markeredgecolor=BG, markeredgewidth=2, label=OFF_LABEL)
    keep = [i for i in sorted(h_on) if h_on[i][1] >= 5]
    ax.plot(keep, [h_on[i][0] for i in keep], color=ORANGE, linewidth=2, marker="s", markersize=8,
            markeredgecolor=BG, markeredgewidth=2, label=ON_LABEL)
    # Keep sample sizes in aligned rows rather than on top of the observed curves.
    transform = ax.get_xaxis_transform()
    for y, label, observed, color in ((-0.27, "n (off)", h_off, BLUE),
                                       (-0.35, "n (512)", h_on, ORANGE)):
        ax.text(-0.38, y, label, transform=transform, ha="right", va="center", fontsize=10.5, color=color)
        for i in range(len(BUCKETS)):
            count = observed[i][1] if i in observed and observed[i][1] >= 5 else "—"
            ax.text(i, y, str(count), transform=transform, ha="center", va="center", fontsize=10.5, color=MUTED)
    ax.axhline(q_off, color=BLUE, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(BUCKETS) - 1 + 0.15, q_off + 0.035,
            f"Constant-rate comparator\nmatched to E0005: q = {q_off:.3f}",
            fontsize=10.5, color=BLUE, ha="right", va="bottom")
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels([b[2] for b in BUCKETS])
    ax.set_xlim(-0.3, len(BUCKETS) - 0.7)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Current streak length, k (identical calls)", labelpad=10)
    ax.set_ylabel("Probability the next call is identical")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    heading(fig, "Observed continuation probability by streak length",
            "22 tasks per bundle · n counts adjacent tool-call pairs · buckets with n < 5 omitted")
    save(fig, "part2-fig3-repeat-hazard")
    return q_off, h_off, h_on


def fig_reasoning():
    panels = [("Resolved tasks (of 22)", 5, 7), ("Runs with 20+ identical calls", 11, 0),
              ("Malformed calls (delimiter slips)", 223, 24), ("Calls missing required arguments", 80, 17)]
    fig, grid = plt.subplots(2, 2, figsize=(10.5, 5.6))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.10, wspace=0.46, hspace=0.58)
    axes = grid.flatten()
    for ax, (title, off, on) in zip(axes, panels):
        recessive(ax, grid_axis="x")
        ax.spines["bottom"].set_visible(False)
        ax.barh([1, 0], [off, on], height=0.48, color=[BLUE, ORANGE], edgecolor=BG, linewidth=2)
        top = max(off, on)
        for y, v in ((1, off), (0, on)):
            ax.text(v + top * 0.03, y, f"{v}", va="center", fontsize=11, color=INK)
        ax.set_xlim(0, top * 1.3)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["Thinking off", "Reasoning 512"], fontsize=10)
        ax.set_xticks([])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=12)
    heading(fig, "Reasoning and execution outcomes on the same 22 tasks",
            f"{OFF_LABEL} (blue) · {ON_LABEL} (orange) · same runtime and sampling settings")
    save(fig, "part2-fig4-reasoning")


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def p_finish(mu, sigma, n=120, startup=20.0, limit=720.0):
    return phi((limit - startup - n * mu) / (sigma * math.sqrt(n)))


def fig_time_cliff():
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.79, bottom=0.15)
    recessive(ax)
    ratios = [0.8 + i * 0.0025 for i in range(241)]
    cases = [(4.5, "#9ca8b0", "4.5 min per task", (0, (6, 2))),
             (5.0, INK, "5.0 min per task", "-"),
             (5.5, "#687c87", "5.5 min per task", (0, (2, 2)))]
    for mu, color, label, style in cases:
        ys = [p_finish(mu * r, 2.5 * r) for r in ratios]
        ax.plot(ratios, ys, color=color, linewidth=2.2, linestyle=style, label=f"Baseline mean: {label}")
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
    ax.axvline(1.18, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(1.188, 0.91, "×1.18", fontsize=11, color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    ax.set_xlim(0.8, 1.4)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Multiplier applied to per-task time (setup + agent session)", labelpad=10)
    ax.set_ylabel("Probability of completing within 12 h")
    heading(fig, "Hypothetical runtime multiplier and completion probability",
            "Illustration only · 120 independent tasks · 20 min startup · baseline SD 2.5 min; mean and SD scale together")
    save(fig, "part2-fig5-time-cliff")
    return {mu: (p_finish(mu, 2.5), p_finish(mu * 1.18, 2.5 * 1.18)) for mu, *_ in cases}


if __name__ == "__main__":
    fig_leaderboard()
    fig_hosted_scores()
    q, h_off, h_on = fig_repeat_hazard()
    fig_reasoning()
    cliff = fig_time_cliff()
    print("memoryless average repeat probability (thinking off):", round(q, 3))
    print("hazard thinking off:", {BUCKETS[i][2]: (round(p, 3), n) for i, (p, n) in sorted(h_off.items())})
    print("hazard reasoning:", {BUCKETS[i][2]: (round(p, 3), n) for i, (p, n) in sorted(h_on.items())})
    print("P(finish) at x1.00 and x1.18:", {k: tuple(round(x, 3) for x in v) for k, v in cliff.items()})
