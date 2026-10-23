"""Plot frozen official results for the unpublished RSNA Part 5.

No remote calls, model execution, training or new performance estimation.
Only part5-results-* PNG/SVG assets and the owned figure-results QA directory
are written. Team-level standings are not same-submission transfer estimates.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

BLOG = Path(__file__).resolve().parents[1]
DATA = BLOG / "_draft/_checks/rsna-series-20261023-results-revision"
QA = DATA / "figure-results"
OUT = BLOG / "assets/img/posts/rsna-knee-series"
QA.mkdir(parents=True, exist_ok=True)
COL = dict(ink="#183344", muted="#526572", teal="#277F85", blue="#426E9C",
           copper="#AA6341", grid="#DCE4E7", light="#EFF6F5", gray="#A6B3BB")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "text.color": COL["ink"], "axes.labelcolor": COL["muted"],
                     "xtick.color": COL["muted"], "ytick.color": COL["muted"],
                     "axes.edgecolor": COL["grid"], "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none",
                     "svg.hashsalt": "rsna-results-part5", "savefig.facecolor": "white"})
SOURCE_FILES = [DATA / p for p in ("own-submissions-sdk.json", "selected-submissions.json",
                                  "leaderboard-paired.csv", "leaderboard-analysis.json")]
SOURCE_HASHES = {str(p.relative_to(BLOG)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in SOURCE_FILES}
BENCHMARK_SOURCE = Path("/Volumes/PKKim_990/Kaggle/2026/26_RSNA/analysis/kernel_versions.json")
BENCHMARK_REGISTRY = json.loads(BENCHMARK_SOURCE.read_text())
BENCHMARK_RECEIPT = next(r for r in BENCHMARK_REGISTRY["submissions"]
                         if str(r["submission"]) == "56339713")
if BENCHMARK_RECEIPT.get("execution_failed") != "deployment failure: benchmark file scored, excluded from offset/slope":
    raise ValueError("The known benchmark execution error needs its exact registry receipt")
SOURCE_HASHES[str(BENCHMARK_SOURCE)] = hashlib.sha256(BENCHMARK_SOURCE.read_bytes()).hexdigest()
RAW = json.loads(SOURCE_FILES[0].read_text())
SELECTED_RAW = json.loads(SOURCE_FILES[1].read_text())
SUMMARY = json.loads(SOURCE_FILES[3].read_text())
SELECTED = {str(r["ref"]) for r in SELECTED_RAW["rows"]}
BENCHMARK_ERROR_REF = "56339713"
ROWS = []
for raw in RAW["rows"]:
    try:
        public, private = float(raw["public"]), float(raw["private"])
    except (ValueError, TypeError):
        continue
    if not (math.isfinite(public) and math.isfinite(private) and
            0 <= public <= 1 and 0 <= private <= 1):
        raise ValueError("Unexpected numeric score")
    if raw["status"] != "SubmissionStatus.COMPLETE":
        raise ValueError("Numeric pair from a non-COMPLETE submission")
    match = re.search(r"\bv(\d+)\b", raw["description"])
    ROWS.append(dict(ref=str(raw["ref"]),
                     date=datetime.fromisoformat(raw["date"]).replace(tzinfo=timezone.utc),
                     public=public, private=private,
                     version=int(match[1]) if match else None,
                     selected=str(raw["ref"]) in SELECTED,
                     description=raw["description"]))
ROWS.sort(key=lambda row: (row["date"], row["ref"]))
if len({r["ref"] for r in ROWS}) != len(ROWS):
    raise ValueError("Duplicate submission refs")
for row in SELECTED_RAW["rows"]:
    same = next(r for r in ROWS if r["ref"] == str(row["ref"]))
    if same["public"] != float(row["public"]) or same["private"] != float(row["private"]):
        raise ValueError("Selected receipt and all-submissions receipt disagree")
with SOURCE_FILES[2].open() as stream:
    TEAMS = list(csv.DictReader(stream))
for row in TEAMS:
    for key in ("team_id", "public_rank", "private_rank", "rank_gain"):
        row[key] = int(row[key])
    for key in ("public", "private", "private_minus_public"):
        row[key] = float(row[key])
if len(TEAMS) != SUMMARY["team_count"] or len({r["team_id"] for r in TEAMS}) != len(TEAMS):
    raise ValueError("Leaderboard coverage mismatch")
OUR = next(r for r in TEAMS if r["team_id"] == SUMMARY["our_team"]["team_id"])
CANDIDATE_LABELS = {
    213: "S6F", 219: "H", 220: "D2c", 226: "K-eq", 227: "C2 · own 50%",
    229: "C2 · own 25%", 231: "K-W", 233: "C1U", 234: "N2", 235: "S6FX", 236: "K-W2",
}
CANDIDATES = []
for version, label in CANDIDATE_LABELS.items():
    found = [r for r in ROWS if r["version"] == version]
    if len(found) != 1:
        raise ValueError(f"Need one exact numeric receipt for v{version}: {len(found)}")
    CANDIDATES.append(dict(found[0], label=label))
LAYOUTS = []
OUTPUTS = []


def new_figure(title, subtitle, mobile=False, height=8.6):
    fig = plt.figure(figsize=(7.6 if mobile else 12, height), dpi=100 if mobile else 160,
                     facecolor="white")
    fig.text(.045, .958, "RSNA KNEE  /  PART 5", fontsize=13 if mobile else 12,
             weight="bold", color=COL["teal"])
    fig.text(.045, .905, title, fontsize=23 if mobile else 21, weight="bold", va="top")
    if mobile:
        subtitle = "\n".join(textwrap.fill(line, width=66) for line in subtitle.splitlines())
    fig.text(.045, .836 if mobile else .847, subtitle, fontsize=13.5 if mobile else 12.2,
             color=COL["muted"], va="top", linespacing=1.4)
    return fig


def style(ax, mobile=False):
    ax.grid(color=COL["grid"], linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=15 if mobile else 11.3, length=0, pad=7)
    ax.xaxis.label.set_fontsize(16 if mobile else 12)
    ax.yaxis.label.set_fontsize(16 if mobile else 12)


def save(fig, stem, mobile=False):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    errors, text_bounds = [], []
    # Check every visible label against the complete saved canvas. Direct visual
    # review separately checks connectors, labels and explanatory whitespace.
    for item in fig.findobj(match=matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        bb = item.get_window_extent(renderer)
        text_bounds.append({"text": item.get_text(), "bounds_px": list(bb.bounds)})
        if bb.x0 < -1 or bb.x1 > fig.bbox.x1+1 or bb.y0 < -1 or bb.y1 > fig.bbox.y1+1:
            errors.append(item.get_text())
    intersections=[]
    for i, first in enumerate(text_bounds):
        ax0,ay0,aw,ah=first["bounds_px"]
        for second in text_bounds[i+1:]:
            bx0,by0,bw,bh=second["bounds_px"]
            if min(ax0+aw,bx0+bw)-max(ax0,bx0)>2 and min(ay0+ah,by0+bh)-max(ay0,by0)>2:
                intersections.append([first["text"],second["text"]])
    if errors:
        raise ValueError(f"Out-of-canvas text in {stem}: {errors}")
    if intersections:
        raise ValueError(f"Text overlap in {stem}: {intersections}")
    dimensions = [round(fig.get_figwidth()*fig.dpi), round(fig.get_figheight()*fig.dpi)]
    for ext in ("png", "svg"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=fig.dpi, metadata={"Date": None} if ext == "svg" else None)
        OUTPUTS.append({"path": str(path.relative_to(BLOG)), "dimensions": dimensions,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size})
    LAYOUTS.append({"figure": stem, "dimensions": dimensions,
                    "out_of_canvas_text": [], "text_intersections": [], "text_bounds": text_bounds})
    plt.close(fig)


def journey(mobile=False):
    title = "Public progress and\nPrivate transfer" if mobile else "Public progress and Private transfer"
    subtitle = (f"{len(ROWS)} numeric pairs, including one known execution error.\n"
                "Private scores were revealed after the competition.")
    fig = new_figure(title, subtitle, mobile, 13.8 if mobile else 8.8)
    rects = [[.15, .49, .79, .22], [.15, .15, .79, .23]] if mobile else [
        [.10, .48, .85, .235], [.10, .15, .85, .245]]
    dates = [r["date"] for r in ROWS]
    public = np.array([r["public"] for r in ROWS])
    private = np.array([r["private"] for r in ROWS])
    end = max(datetime.fromisoformat(r["date"]).replace(tzinfo=timezone.utc)
              for r in RAW["rows"]).replace(hour=23, minute=59, second=59)
    start_late = end.replace(hour=0, minute=0, second=0) - timedelta(days=14)
    for index, rect in enumerate(rects):
        ax = fig.add_axes(rect)
        mask = np.array([r["ref"] != BENCHMARK_ERROR_REF and
                         (index == 0 or r["date"] >= start_late) for r in ROWS])
        # Each faint vertical segment joins the same submission, never two models.
        ax.vlines(np.array(dates)[mask], private[mask], public[mask],
                  color=COL["gray"], linewidth=.6, alpha=.28, zorder=1)
        ax.scatter(np.array(dates)[mask], public[mask], s=18 if not mobile else 27,
                   color=COL["blue"], alpha=.7, zorder=3)
        ax.scatter(np.array(dates)[mask], private[mask], s=18 if not mobile else 27,
                   color=COL["copper"], alpha=.7, zorder=3)
        ax.step(dates+[end], list(np.maximum.accumulate(public))+[float(public.max())],
                where="post", color=COL["ink"], linewidth=1.55, zorder=2)
        for row in ROWS:
            if row["selected"]:
                for key in ("public", "private"):
                    ax.scatter([row["date"]], [row[key]], s=100 if not mobile else 120,
                               facecolor="none", edgecolor=COL["teal"], linewidth=1.8, zorder=5)
        ax.set_xlim((dates[0]-timedelta(days=1)) if index == 0 else start_late, end+timedelta(hours=8))
        ax.set_ylim((.48, .98) if index == 0 else (.882, .966))
        ax.set_ylabel("Macro AUC")
        if index == 0:
            ax.set_title("All paired scored records", loc="left", fontsize=17 if mobile else 13,
                         color=COL["ink"], weight="bold", pad=12)
            ticks = [datetime(2026, m, 1, tzinfo=timezone.utc) for m in (9, 10)]
            ticks = [dates[0].replace(hour=0,minute=0,second=0)] + ticks + [end.replace(hour=0,minute=0,second=0)]
            ax.set_yticks([.5,.7,.9])
            error=next(r for r in ROWS if r["ref"]==BENCHMARK_ERROR_REF)
            ax.scatter([error["date"]],[error["public"]],marker="x",s=55,
                       linewidth=1.7,color=COL["muted"],zorder=4)
            ax.annotate("Benchmark-file error\n0.500 / 0.500",xy=(error["date"],.5),
                        xytext=(datetime(2026,8,28,tzinfo=timezone.utc),.57),
                        color=COL["muted"],fontsize=13 if mobile else 11,
                        arrowprops=dict(arrowstyle="-",color=COL["muted"],lw=.9),
                        ha="left",va="center")
        else:
            ax.set_title("Final two weeks in the returned history", loc="left",
                         fontsize=17 if mobile else 13, color=COL["ink"], weight="bold", pad=12)
            ticks = [start_late, start_late+timedelta(days=7), end.replace(hour=0,minute=0,second=0)]
            ax.set_yticks([.89, .92, .95])
            ax.set_xlabel("Submission date (UTC)", labelpad=10)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=timezone.utc))
        style(ax, mobile)
    handles = [Line2D([], [], marker="o", linestyle="none", color=COL["blue"], label="Public"),
               Line2D([], [], marker="o", linestyle="none", color=COL["copper"], label="Private"),
               Line2D([], [], color=COL["ink"], label="Public best so far"),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                      markeredgecolor=COL["teal"], label="Selected: v229, v236")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.045, .745),
               ncol=2 if mobile else 4, fontsize=13 if mobile else 11.6,
               frameon=False, columnspacing=1.4)
    footer = ("Connectors pair one submission. The error × is not model quality.\n"
              f"{len(RAW['rows'])-len(ROWS)} returned records lack a numeric pair; no scores are imputed.")
    fig.text(.045, .033, footer, fontsize=12 if mobile else 11.3,
             color=COL["muted"], linespacing=1.5)
    save(fig, "part5-results-journey"+("-mobile" if mobile else ""), mobile)


def candidates(mobile=False):
    fig = new_figure("The final candidates,\npaired submission by submission" if mobile else
                     "The final candidates, paired submission by submission",
                     "Displayed scores from exact completed submissions. Teal rows mark the final two choices.",
                     mobile, 17.8 if mobile else 9.4)
    if mobile:
        # Each candidate has a separate heading and a repeated pair of score
        # columns; no tiny desktop table is squeezed into a phone-width image.
        fig.text(.76, .786, "PUBLIC", fontsize=12.2, color=COL["blue"], ha="center", weight="bold")
        fig.text(.92, .786, "PRIVATE", fontsize=12.2, color=COL["copper"], ha="center", weight="bold")
        top, step = .748, .0605
        for i, row in enumerate(CANDIDATES):
            y = top-i*step
            if row["selected"]:
                fig.patches.append(plt.Rectangle((.035, y-.036), .93, .055, transform=fig.transFigure,
                                                  color=COL["light"], zorder=-1))
            fig.text(.055, y+.003, f"v{row['version']} · {row['label']}", fontsize=18,
                     weight="bold" if row["selected"] else "normal", color=COL["ink"])
            ax = fig.add_axes([.075, y-.025, .56, .018])
            ax.plot([row["private"],row["public"]],[0,0],color=COL["gray"],lw=1.5)
            ax.scatter([row["public"]],[0],s=65,color=COL["blue"],zorder=3)
            ax.scatter([row["private"]],[0],s=65,color=COL["copper"],zorder=3)
            ax.set(xlim=(.93,.96),ylim=(-1,1));ax.axis("off")
            for x,key,col in [(.76,"public","blue"),(.92,"private","copper")]:
                fig.text(x,y-.018,f"{row[key]:.3f}",fontsize=19,color=COL[col],ha="center")
        fig.text(.075,.085,"Dumbbell axis: 0.930–0.960 AUC",fontsize=13,color=COL["muted"])
        fig.text(.045,.040,"One row = one submission, not a family estimate.\n"
                 "No rank or causal contribution is inferred for a model.",fontsize=12.5,
                 color=COL["muted"],linespacing=1.5)
    else:
        ax = fig.add_axes([.35,.145,.40,.61])
        n=len(CANDIDATES)
        for i,row in enumerate(CANDIDATES):
            y=n-1-i
            if row["selected"]:
                ax.axhspan(y-.38,y+.38,color=COL["light"],zorder=0)
            ax.plot([row["private"],row["public"]],[y,y],c=COL["gray"],lw=1.5,zorder=1)
            ax.scatter([row["public"]],[y],s=55,color=COL["blue"],zorder=3)
            ax.scatter([row["private"]],[y],s=55,color=COL["copper"],zorder=3)
            yy=.145+.61*(y+.6)/(n+.2)
            fig.text(.045,yy,f"v{row['version']}",fontsize=14,va="center",
                     weight="bold" if row["selected"] else "normal",
                     color=COL["teal"] if row["selected"] else COL["ink"])
            fig.text(.12,yy,row["label"],fontsize=13.4,va="center")
            for xx,key,col in [(.83,"public","blue"),(.94,"private","copper")]:
                fig.text(xx,yy,f"{row[key]:.3f}",fontsize=14,va="center",ha="center",
                         color=COL[col],weight="bold" if row["selected"] else "normal")
        ax.set(xlim=(.93,.96),ylim=(-.6,n-.4),yticks=[],xticks=[.93,.94,.95,.96],xlabel="Macro AUC")
        style(ax)
        ax.spines["left"].set_visible(False)
        fig.text(.045,.79,"VERSION / PACKAGE",fontsize=11.2,color=COL["muted"],weight="bold")
        fig.text(.83,.79,"PUBLIC",fontsize=11.2,color=COL["blue"],ha="center",weight="bold")
        fig.text(.94,.79,"PRIVATE",fontsize=11.2,color=COL["copper"],ha="center",weight="bold")
        fig.text(.045,.055,"Selected: v229 and v236. Each connector joins the Public and Private scores of the same submission.",
                 fontsize=11.4,color=COL["muted"])
        fig.text(.045,.025,"These are coupled package comparisons, not isolated model effects or model-level Private ranks.",
                 fontsize=11.2,color=COL["muted"])
    save(fig,"part5-results-candidates"+("-mobile" if mobile else ""),mobile)


def shakeup(mobile=False):
    fig = new_figure("How the final\nstandings changed" if mobile else "How the final standings changed",
                     f"{len(TEAMS):,} matched teams. Team-level Public and Private can come from different submissions.",
                     mobile,13.0 if mobile else 8.0)
    rects = [[.15,.48,.77,.255],[.15,.15,.77,.245]] if mobile else [
        [.09,.25,.37,.47],[.60,.25,.35,.47]]
    a=fig.add_axes(rects[0]);b=fig.add_axes(rects[1])
    a.scatter([r["public_rank"] for r in TEAMS],[r["private_rank"] for r in TEAMS],
              s=7 if not mobile else 9,c=COL["blue"],alpha=.32,edgecolors="none")
    a.plot([1,len(TEAMS)],[1,len(TEAMS)],ls="--",c=COL["muted"],lw=1.1,zorder=2)
    a.set(xscale="log",yscale="log",xlim=(.8,8000),ylim=(8000,.8),
          xlabel="Public rank",ylabel="Private position")
    a.set_title("All matched teams",loc="left",fontsize=17 if mobile else 13.5,weight="bold",pad=13)
    ticks=[1,10,100,1000,6425]
    a.set_xticks(ticks,labels=["1","10","100","1k","6,425"])
    a.set_yticks(ticks,labels=["1","10","100","1k","6,425"])
    a.minorticks_off()
    a.scatter([OUR["public_rank"]],[OUR["private_rank"]],s=110,
              facecolor=COL["teal"],edgecolor="white",lw=1.2,zorder=5)
    a.annotate("Me\n1,785 → 560",xy=(OUR["public_rank"],OUR["private_rank"]),
               xytext=(7,2100),fontsize=14 if mobile else 11.8,color=COL["teal"],
               ha="left",va="center",bbox=dict(fc="white",ec="none",alpha=.95,pad=3),
               arrowprops=dict(arrowstyle="-",color=COL["teal"],lw=1.1))
    cohort=[r for r in TEAMS if r["public_rank"]<=500]
    counts=Counter((r["public"],r["private"]) for r in cohort)
    b.scatter([p[0] for p in counts],[p[1] for p in counts],
              s=[5*n for n in counts.values()],c=COL["blue"],alpha=.65,
              edgecolors="white",linewidth=.45,zorder=3)
    b.plot([.957,.971],[.957,.971],ls="--",c=COL["muted"],lw=1.1,zorder=2)
    b.set(xlim=(.957,.971),ylim=(.938,.972),xlabel="Public AUC",ylabel="Private AUC",
          xticks=[.958,.962,.966,.970],yticks=[.940,.950,.960,.970])
    b.set_title("Public top 500: paired team scores",loc="left",
                fontsize=16 if mobile else 13.1,weight="bold",pad=13)
    for n in (1,10,100):
        b.scatter([],[],s=5*n,color=COL["blue"],alpha=.65,label=f"{n}")
    b.legend(title="Teams per displayed score pair",loc="upper left",frameon=False,
             fontsize=11.5 if mobile else 10.3,title_fontsize=11.5 if mobile else 10.4,
             ncol=3,handletextpad=.35,columnspacing=.9,borderpad=.1)
    style(a,mobile);style(b,mobile)
    retained=[SUMMARY["retention"][str(k)]["same_top_k"] for k in (10,100,500)]
    if mobile:
        fig.text(.045,.077,f"Top-k retained: {retained[0]}/10 · {retained[1]}/100 · {retained[2]}/500",
                 fontsize=13.6,color=COL["ink"],weight="bold")
        fig.text(.045,.025,"Private position follows the returned API order.\n"
                 "Dashed lines: no rank change / no displayed score gap.\n"
                 "Rounded scores do not reveal the tie-breaking rule.",
                 fontsize=11.8,color=COL["muted"],linespacing=1.45)
    else:
        fig.text(.09,.135,f"Public top-k retained in Private:  {retained[0]}/10     {retained[1]}/100     {retained[2]}/500",
                 fontsize=13,color=COL["ink"],weight="bold")
        fig.text(.045,.065,"Private position uses the returned API order. Dashed lines: no rank change (left); no displayed score gap (right).",
                 fontsize=11.2,color=COL["muted"])
        fig.text(.045,.033,"Marker area on the right counts coincident displayed score pairs. Rounded scores do not identify tie-breaking or error causes.",
                 fontsize=11.0,color=COL["muted"])
    save(fig,"part5-results-shakeup"+("-mobile" if mobile else ""),mobile)


if __name__ == "__main__":
    for is_mobile in (False,True):
        journey(is_mobile);candidates(is_mobile);shakeup(is_mobile)
    # Refuse a silently changed input even if another process updated a receipt
    # while this bounded editorial plotting job was running.
    for path in SOURCE_FILES:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_HASHES[str(path.relative_to(BLOG))]:
            raise RuntimeError(f"Input changed during plotting: {path}")
    if hashlib.sha256(BENCHMARK_SOURCE.read_bytes()).hexdigest()!=SOURCE_HASHES[str(BENCHMARK_SOURCE)]:
        raise RuntimeError("Benchmark classification source changed during plotting")
    (QA/"benchmark-execution-error-source.json").write_text(json.dumps({
        "source":str(BENCHMARK_SOURCE),"sha256":SOURCE_HASHES[str(BENCHMARK_SOURCE)],
        "receipt":BENCHMARK_RECEIPT},indent=2)+"\n")
    plot_rows=[dict(r,date=r["date"].isoformat()) for r in ROWS]
    (QA/"own-plotted-rows.json").write_text(json.dumps(plot_rows,indent=2)+"\n")
    previews=[]
    (QA/"previews").mkdir(exist_ok=True)
    for row in OUTPUTS:
        if not row["path"].endswith(".png"):
            continue
        path=BLOG/row["path"]
        width=360 if "-mobile" in path.stem else 760
        with Image.open(path) as image:
            scaled=image.resize((width,round(image.height*width/image.width)),Image.Resampling.LANCZOS)
            target=QA/"previews"/f"{path.stem}-{width}px.png"
            scaled.save(target)
            previews.append({"path":str(target.relative_to(BLOG)),"dimensions":list(scaled.size),
                             "sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest={"purpose":"Retrospective descriptive plots from frozen official aggregate results",
              "source_sha256":SOURCE_HASHES,
              "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "numeric_pairs":len(ROWS),"returned_submission_records":len(RAW["rows"]),
              "non_numeric_pairs_excluded":len(RAW["rows"])-len(ROWS),
              "known_execution_error_ref":BENCHMARK_ERROR_REF,
              "model_numeric_pairs":len(ROWS)-1,
              "selected_refs":sorted(SELECTED),"matched_teams":len(TEAMS),
              "candidate_refs":{str(r["version"]):r["ref"] for r in CANDIDATES},
              "timestamp_convention":"Official submission date strings are interpreted as UTC.",
              "scope_limits":["Private scores were unknown at submission time.",
                "No missing score is imputed and no scoreless run is assigned a value.",
                "Team-level scores may use different submissions, unlike the personal paired plots.",
                "Private positions are API sequence positions; no tie-break mechanism is inferred.",
                "Aggregate results cannot identify model effects, statistical significance or error causes.",
                "No per-model Private rank or unrun-alternative performance is asserted."],
              "outputs":OUTPUTS,"article_scale_previews":previews,"layout_checks":LAYOUTS}
    (QA/"figure-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("numeric_pairs","matched_teams","selected_refs","outputs")},indent=2))
