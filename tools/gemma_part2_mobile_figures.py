"""Portrait derivatives of the frozen Gemma Part 2 figures.

Uses the exact fixed records, trace functions, labels, and colors from
``gemma_series_figures.py``. No network or evaluation is performed. Desktop
figures are never rewritten. Python with matplotlib is required.
Run: python tools/gemma_part2_mobile_figures.py
"""
import ast
import csv
import inspect
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import gemma_series_figures as base
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = base.OUT
W = 5.0
plt.rcParams.update({
    'font.family':'Arial', 'font.size':15, 'axes.labelsize':15,
    'xtick.labelsize':15,'ytick.labelsize':15,'svg.fonttype':'none',
    'figure.facecolor':base.BG,'axes.facecolor':base.BG,
    'savefig.facecolor':base.BG,
})


def literal_assignment(function, name):
    """Read a frozen literal from the canonical desktop generator."""
    for node in ast.walk(ast.parse(inspect.getsource(function))):
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            class ResolveConstant(ast.NodeTransformer):
                def visit_Name(self, value):
                    constant = getattr(base, value.id, None)
                    if not isinstance(constant, (str, int, float)):
                        raise ValueError(f'Unknown plot constant: {value.id}')
                    return ast.Constant(constant)
            return ast.literal_eval(ResolveConstant().visit(node.value))
    raise ValueError(f'{function.__name__}: fixed {name} was not found')


def save(fig, name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+'-mobile.svg'))
    fig.savefig(OUT/(name+'-mobile.png'),dpi=156)
    plt.close(fig)


def heading(fig, title, lines):
    fig.text(.065,.975,title,va='top',fontsize=19,fontweight='bold',linespacing=1.22)
    y=.975-.043*len(title.split('\n'))
    for line in lines:
        fig.text(.065,y,line,va='top',fontsize=15,color=base.MUTED)
        y-=.033


def plain(ax, axis='x'):
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis=axis,color=base.GRID,lw=.8)
    ax.set_axisbelow(True); ax.tick_params(length=0,pad=7)


def leaderboard():
    rows=list(csv.DictReader(open(base.LEADERBOARD,encoding='utf-8-sig')))
    counts=Counter(round(float(r['Score']),2) for r in rows)
    scores=sorted(counts,reverse=True)
    fig=plt.figure(figsize=(W,12.8))
    heading(fig,'Public leaderboard\nTeam-best scores',[
        '22 Oct 2026 · 06:15 UTC', f'{len(rows):,} scored teams'])
    ax=fig.add_axes([.29,.185,.63,.57]); plain(ax)
    yy=range(len(scores))
    ax.barh(yy,[counts[v] for v in scores],height=.70,
            color=[base.AQUA if v==.18 else base.NEUTRAL for v in scores])
    ax.set_yticks(list(yy),[f'{v:.2f} / {round(v*58+.49)}' for v in scores])
    ax.invert_yaxis(); ax.set_ylim(len(scores)-.4,-.6)
    ax.set_xlim(0,650); ax.set_xticks([0,200,400,600]); ax.set_xlabel('Teams',labelpad=10)
    for y,v in zip(yy,scores):
        ax.text(counts[v]+12,y,str(counts[v]),va='center',fontsize=15,color=base.MUTED)
    fig.text(.065,.785,'Score / inferred solves*',fontsize=15,fontweight='bold')
    fig.text(.065,.09,'Teal: E0052 reference\n0.18 · rank 398',fontsize=15,color=base.AQUA,linespacing=1.5)
    fig.text(.065,.043,'*Assuming 58 Public tasks.\nTop score: 0.24 (4 teams).',fontsize=15,color=base.MUTED,linespacing=1.5)
    save(fig,'part2-fig1-leaderboard')
    return {'teams':len(rows),'score_counts':dict(sorted(counts.items()))}


def hosted():
    rows=list(csv.DictReader(open(base.SUBMISSIONS)))
    events=literal_assignment(base.fig_hosted_scores,'events')
    groups={
        'E0005':(base.OFF_LABEL,base.BLUE,'o'),
        'E0050':(base.ON_LABEL,base.ORANGE,'s'),
        'E0052':(base.REFERENCE_LABEL,base.AQUA,'D'),
    }
    fig=plt.figure(figsize=(W,16.2))
    # Use physical line spacing for a long portrait with two complete time panels.
    def t(inches,text,**kw):
        kw.setdefault('fontsize',15)
        fig.text(.065,1-inches/16.2,text,va='top',**kw)
    t(.28,'Hosted scores across\npublic artifact updates',fontweight='bold',fontsize=19,linespacing=1.25)
    t(1.05,f'{len(rows)} frozen submission records')
    for i,(label,color,marker) in enumerate(list(groups.values())+[('Other variants',base.NEUTRAL,'^')]):
        y=1-(1.56+i*.30)/16.2
        fig.add_artist(Line2D([.085],[y],transform=fig.transFigure,marker=marker,color=color,
                              linestyle='none',markersize=8))
        fig.text(.13,y,label,fontsize=15,va='center',color=color)
    panels=[(datetime(2026,9,24),datetime(2026,10,5),.590,'Sep 24 – Oct 04',
             [datetime(2026,9,25),datetime(2026,9,28),datetime(2026,10,1),datetime(2026,10,4)]),
            (datetime(2026,10,5),datetime(2026,10,23,9),.335,'Oct 05 – Oct 23',
             [datetime(2026,10,5),datetime(2026,10,11),datetime(2026,10,17),datetime(2026,10,23)])]
    for start,end,bottom,title,ticks in panels:
        ax=fig.add_axes([.205,bottom,.735,.192]); plain(ax,'y')
        ax.set_ylim(0,.295);ax.set_xlim(start,end)
        ax.set_yticks([0,.1,.2]);ax.set_ylabel('Public score',labelpad=6)
        ax.set_xticks(ticks);ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.set_title(title,loc='left',fontsize=16,fontweight='bold',pad=22)
        for idx,(when,label) in enumerate(events):
            dt=datetime.strptime(when,'%Y-%m-%d %H:%M')
            if start<=dt<end:
                ax.axvline(dt,color=base.GRID,lw=1.4,ls=(0,(4,3)),zorder=0)
                ax.text(dt,.277,chr(65+idx),ha='center',va='center',fontsize=15,color=base.MUTED)
        best_start=max(start,datetime(2026,10,2))
        ax.hlines(.24,best_start,end,color=base.MUTED,lw=1,ls=(0,(1,2)))
        for r in rows:
            dt=datetime.strptime(r['date'][:16],'%Y-%m-%dT%H:%M')
            if not (start<=dt<end) or not r['public_lb']: continue
            score=float(r['public_lb'])
            _,color,marker=groups.get(r['experiment_id'],('',base.NEUTRAL,'^'))
            ax.scatter(dt,score,s=73,marker=marker,color=color,edgecolor=base.BG,lw=1.5,zorder=3)
            if r['experiment_id']=='E0053':
                ax.annotate('E0053',(dt,score),xytext=(0,-23),textcoords='offset points',
                            ha='center',va='top',fontsize=15,color=base.MUTED)
        if start.month == 10:
            ax.set_xlabel('Submission date (UTC)',labelpad=8)
    t(11.70,'Unavailable scores (not zeros)',fontweight='bold')
    unavailable=[]
    for r in rows:
        if r['public_lb']:continue
        dt=datetime.strptime(r['date'][:16],'%Y-%m-%dT%H:%M')
        state='pending · E4B' if r['status']=='PENDING' else ('12 h timeout' if r['experiment_id']=='E0057' else 'error')
        unavailable.append(f"{r['experiment_id']} · {dt:%b %d}: {state}")
    for i,line in enumerate(unavailable):t(12.04+i*.30,line)
    t(13.13,'Dashed: public artifact updates',fontweight='bold')
    for idx,(when,label) in enumerate(events):
        dt=datetime.strptime(when,'%Y-%m-%d %H:%M')
        t(13.46+idx*.30,f'{chr(65+idx)} · {dt:%b %d}: {label}',color=base.MUTED)
    t(15.12,'Dotted: observed best 0.24\nsince Oct 2. E0053 used the same ZIP\non Oct 16 (0.10) and Oct 20 (0.15).',color=base.MUTED,linespacing=1.5)
    save(fig,'part2-fig2-hosted-scores')
    return {'records':len(rows),'scored':sum(bool(r['public_lb']) for r in rows),'unavailable':unavailable,'events':events}


def repeat_hazard():
    off,on=base.call_sequences('E0005'),base.call_sequences('E0050')
    h_off,h_on=base.hazard(off),base.hazard(on)
    pairs=sum(len(r)-1 for r in off if r)
    q=sum(sum(a==b for a,b in zip(r,r[1:])) for r in off)/pairs
    fig=plt.figure(figsize=(W,12.7))
    heading(fig,'Repeat continuation\nby streak length',[
        '22 tasks per bundle', 'Adjacent call pairs; n < 5 omitted'])
    ax=fig.add_axes([.23,.49,.70,.285]);plain(ax,'y')
    for hh,color,marker in [(h_off,base.BLUE,'o'),(h_on,base.ORANGE,'s')]:
        keep=[i for i in sorted(hh) if hh[i][1]>=5]
        ax.plot(keep,[hh[i][0] for i in keep],color=color,marker=marker,markersize=7,lw=2)
    ax.axhline(q,color=base.BLUE,lw=1.3,ls=(0,(4,3)))
    ax.set_xticks(range(7),[x[2] for x in base.BUCKETS])
    ax.set_xlim(-.35,6.35);ax.set_ylim(0,1.12);ax.set_yticks([0,.25,.5,.75,1])
    ax.set_xlabel('Identical-call streak, k',labelpad=10)
    ax.set_ylabel('Next-call repeat probability',labelpad=9)
    fig.text(.065,.418,f'Dashed: off mean q = {q:.3f}',fontsize=15,color=base.BLUE)
    fig.text(.065,.377,base.OFF_LABEL,fontsize=15,color=base.BLUE)
    fig.text(.065,.339,base.ON_LABEL,fontsize=15,color=base.ORANGE)
    x=[.08,.45,.74]
    for xx,txt in zip(x,['Streak k','n (off)','n (512)']):fig.text(xx,.278,txt,fontsize=15,fontweight='bold')
    for i,bucket in enumerate(base.BUCKETS):
        y=.241-i*.032
        vals=[bucket[2]]+[str(hh[i][1]) if i in hh and hh[i][1]>=5 else '—' for hh in [h_off,h_on]]
        for xx,val in zip(x,vals):fig.text(xx,y,val,fontsize=15,color=base.MUTED)
    save(fig,'part2-fig3-repeat-hazard')
    return {'q':q,'off':h_off,'reasoning_512':h_on}


def reasoning():
    panels=literal_assignment(base.fig_reasoning,'panels')
    fig=plt.figure(figsize=(W,13.3))
    heading(fig,'Reasoning and\nexecution outcomes',[
        'Same 22 tasks, runtime, sampling',base.OFF_LABEL,base.ON_LABEL])
    titles=['Resolved tasks (of 22)','Runs with 20+\nidentical calls',
            'Malformed calls\n(delimiter slips)','Calls missing\nrequired arguments']
    for i,((_,off,on),title) in enumerate(zip(panels,titles)):
        ax=fig.add_axes([.23,.617-i*.182,.70,.105]);plain(ax)
        ax.barh([1,0],[off,on],height=.5,color=[base.BLUE,base.ORANGE])
        top=max(off,on);ax.set_xlim(0,top*1.25)
        ax.set_yticks([1,0],['Off','512']);ax.set_xticks([]);ax.set_ylim(-.6,1.6)
        for yy,v in [(1,off),(0,on)]:ax.text(v+top*.035,yy,str(v),va='center',fontsize=16)
        ax.set_title(title,loc='left',fontsize=16,fontweight='bold',pad=12,linespacing=1.25)
    fig.text(.065,.035,'Blue: thinking off · Orange: reasoning 512',fontsize=15,color=base.MUTED)
    save(fig,'part2-fig4-reasoning')
    return {'panels':panels}


def time_cliff():
    fig=plt.figure(figsize=(W,8.8))
    heading(fig,'Hypothetical runtime\nand completion',[
        '120 independent tasks · 20 min startup',
        'Per-task time: setup + agent session',
        'Baseline SD 2.5 min; mean & SD scale'])
    ax=fig.add_axes([.23,.32,.70,.365]);plain(ax,'y')
    ratios=[.8+i*.0025 for i in range(241)]
    cases=literal_assignment(base.fig_time_cliff,'cases')
    for mu,color,label,style in cases:
        ax.plot(ratios,[base.p_finish(mu*r,2.5*r) for r in ratios],color=color,lw=2.2,ls=style,label=f'Baseline mean: {mu:.1f} min')
    ax.axvline(1,color=base.MUTED,lw=1,ls=(0,(1,2)))
    ax.axvline(1.18,color=base.INK,lw=1.2,ls=(0,(4,3)))
    ax.text(1.19,.91,'×1.18',fontsize=15)
    ax.set_xlim(.8,1.4);ax.set_ylim(-.02,1.05)
    ax.set_xticks([.8,1,1.2,1.4]);ax.set_yticks([0,.25,.5,.75,1])
    ax.set_xlabel('Runtime multiplier',labelpad=10)
    ax.set_ylabel('P(finish within 12 h)',labelpad=9)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.065,.035),frameon=False,fontsize=15,labelspacing=.6,handlelength=2)
    save(fig,'part2-fig5-time-cliff')
    return {'finish_probability_at_1.00_and_1.18':{mu:[base.p_finish(mu,2.5),base.p_finish(mu*1.18,2.5*1.18)] for mu,*_ in cases}}

if __name__=='__main__':
    checks={name:fn() for name,fn in [('leaderboard',leaderboard),('hosted',hosted),('hazard',repeat_hazard),('reasoning',reasoning),('time_cliff',time_cliff)]}
    print(json.dumps(checks,indent=2))
