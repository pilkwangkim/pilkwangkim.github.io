"""Rebuild the original English data figures shared by unpublished RSNA parts 0–5.

Uses only the frozen local research records; no experiments or network requests.
Editorial covers are separately authored assets: verify them, never redraw them.
The two Part 5 conceptual figures have their own rsna_part5_figures.py generator.
Run with a Python environment containing matplotlib, numpy and Pillow.
"""
from pathlib import Path
import hashlib
import json
import textwrap
import re
import csv
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image, ImageDraw

BLOG = Path(__file__).resolve().parents[1]
RSNA = Path('/Volumes/PKKim_990/Kaggle/2026/26_RSNA')
OUT = BLOG / 'assets/img/posts/rsna-knee-series'
CHECKS = BLOG / '_draft/_checks/rsna-series-20261022'
OUT.mkdir(parents=True, exist_ok=True)
CHECKS.mkdir(parents=True, exist_ok=True)
C = dict(ink='#183344', muted='#526572', teal='#277F85', blue='#426E9C',
         orange='#AA6341', grid='#DCE4E7', light='#F1F5F5', white='#FFFFFF', navy='#102B3B')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': C['muted'], 'xtick.color': C['muted'], 'ytick.color': C['ink'],
    'text.color': C['ink'], 'axes.edgecolor': C['grid'], 'axes.titleweight': 'bold',
    'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
SOURCES = {}
TEXT_CHECKS = []

def load(rel):
    path = RSNA / rel
    SOURCES[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def frame(part, title, h=6.5):
    fig = plt.figure(figsize=(12, h), facecolor='white')
    fig.text(.045, .945, f'RSNA KNEE  /  PART {part}', fontsize=12, weight='bold', color=C['teal'])
    fig.text(.045, .885, title, fontsize=21, weight='bold')
    return fig

def diagram(part, title, h=6.5):
    fig = frame(part, title, h)
    ax = fig.add_axes([.035, .07, .93, .72])
    ax.set(xlim=(0, 100), ylim=(-5, 100))
    ax.axis('off')
    return fig, ax

def txt(ax, x, y, text, size=16, weight='normal', color=None, **kw):
    return ax.text(x, y, text, fontsize=size, fontweight=weight,
                   color=color or C['ink'], ha=kw.pop('ha', 'center'),
                   va=kw.pop('va', 'center'), linespacing=1.5, **kw)

def box(ax, x, y, w, h, title, body=None, color='teal', size=16):
    patch = FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.25,rounding_size=1.4',
                          fc=C['light'], ec=C[color], lw=1.3)
    ax.add_patch(patch)
    if body:
        a = txt(ax, x+w/2,y+h*.71,title,size=size,weight='bold')
        b = txt(ax, x+w/2,y+h*.34,body,size=size-1,color=C['muted'])
        TEXT_CHECKS.extend([(a,patch),(b,patch)])
    else:
        a = txt(ax,x+w/2,y+h/2,title,size=size,weight='bold')
        TEXT_CHECKS.append((a,patch))
    return patch

def arrow(ax, a, b, color='muted', dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>', mutation_scale=17,
                 lw=1.6,color=C[color],linestyle='--' if dashed else '-'))

def note(fig, text):
    fig.text(.045,.025,text,fontsize=11.5,color=C['muted'],va='bottom')

def save(fig, name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    issues=[]
    for t,p in TEXT_CHECKS:
        if t.figure is not fig: continue
        tb=t.get_window_extent(renderer); pb=p.get_window_extent(renderer)
        if tb.x0 < pb.x0+3 or tb.x1 > pb.x1-3 or tb.y0 < pb.y0+3 or tb.y1 > pb.y1-3:
            issues.append(t.get_text())
    if issues: raise RuntimeError(f'Text outside box in {name}: {issues}')
    fig.savefig(OUT/f'{name}.png',dpi=160)
    fig.savefig(OUT/f'{name}.svg')
    plt.close(fig)

def covers():
    """Protect the image-generated cover masters and their web encodings."""
    record=json.loads((CHECKS/'cover-redesign.json').read_text())
    if {row['part'] for row in record['covers']} != set(range(6)):
        raise RuntimeError('Expected six separately authored editorial covers.')
    for row in record['covers']:
        for path_key, hash_key in [('asset','sha256'),('web_asset','web_sha256')]:
            path=BLOG/row[path_key]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row[hash_key]:
                raise RuntimeError(f'Cover differs from its design receipt: {path}')
            with Image.open(path) as im:
                if list(im.size) != row['dimensions']:
                    raise RuntimeError(f'Cover dimensions differ: {path}')

def part0():
    fig,ax=diagram(0,'One study enters; twelve finding scores leave')
    box(ax,1,58,22,26,'MRI study','Multiple series\nand slices')
    box(ax,36,58,27,26,'Image model','Selection → encoder\n→ aggregation')
    box(ax,76,58,23,26,'12 scores','One probability\nper finding')
    arrow(ax,(24,71),(35,71));arrow(ax,(64,71),(75,71))
    box(ax,1,6,30,27,'Training reports','4,407 report-linked\nstudies')
    box(ax,42,6,31,27,'Derived targets','Definitions + extraction\nwith uncertain labels')
    arrow(ax,(32,20),(41,20));arrow(ax,(57,34),(52,56),dashed=True)
    txt(ax,85,22,'Reports support\ntraining only.\nNo test-time reports.',size=13)
    note(fig,'Task overview. This is a conceptual pipeline, not a measured architecture or a clinical workflow.')
    save(fig,'part0-task')

    fig,ax=diagram(0,'Physical geometry comes before the image tensor')
    # Left spatial ordering: three schematic planes with different origins.
    for k in range(5):
        ax.add_patch(Rectangle((5+k*2.8,39+k*4),18,27,fill=False,ec=C['blue'],lw=1.4))
    txt(ax,20,84,'Order by physical position',size=16,weight='bold')
    txt(ax,20,18,'Slice index follows geometry,\nnot the file name.',size=15)
    arrow(ax,(35,56),(44,56))
    ax.add_patch(Rectangle((48,35),27,38,ec=C['teal'],fc=C['light'],lw=1.3))
    for q in np.linspace(48,75,7): ax.plot([q,q],[35,73],c=C['grid'],lw=.6)
    for q in np.linspace(35,73,7): ax.plot([48,75],[q,q],c=C['grid'],lw=.6)
    txt(ax,61.5,84,'Physical crop → fixed grid',size=16,weight='bold')
    txt(ax,61.5,18,'Field of view L, output width P\nEffective pitch ≈ L / P',size=15)
    box(ax,82,36,17,36,'Slots','Plane / sequence\n+ missing mask',size=14)
    arrow(ax,(76,56),(81,56))
    note(fig,'Schematic. Spatial pitch motivates sampling choices; it is not a diagnostic resolution guarantee.')
    save(fig,'part0-sampling')

    fig,ax=diagram(0,'An unmentioned finding has more than one possible image label')
    txt(ax,14,94,'IMAGE FINDING',size=13,weight='bold',color=C['teal'])
    txt(ax,55,94,'REPORT STATEMENT',size=13,weight='bold',color=C['teal'])
    box(ax,1,65,26,21,'Present',size=18)
    box(ax,1,8,26,21,'Absent',size=18)
    box(ax,42,65,27,21,'Stated positive',size=16)
    box(ax,42,36,27,21,'Not mentioned',size=16,color='orange')
    box(ax,42,8,27,21,'Stated negative',size=16)
    arrow(ax,(28,75.5),(41,75.5))
    arrow(ax,(28,18.5),(41,18.5))
    arrow(ax,(28,69),(41,52),color='orange')
    arrow(ax,(28,25),(41,41),color='orange')
    box(ax,78,28,21,36,'Extractor target','Negative?\nUnknown?\nSoft probability?',size=14,color='orange')
    arrow(ax,(70,46.5),(77,46.5),color='orange')
    note(fig,'Selective-reporting schematic: either image state can lead to silence, even before report or extraction errors.')
    save(fig,'part0-evidence')

    source=RSNA/'_draft/2026-W32_0730-0809.md'
    SOURCES['_draft/2026-W32_0730-0809.md']=hashlib.sha256(source.read_bytes()).hexdigest()
    fig=frame(0,'How many windows should one model read?',h=6.5)
    a=fig.add_axes([.10,.24,.37,.52]);b=fig.add_axes([.59,.24,.34,.52])
    windows=[1,4,10];values=[.8218,.8397,.8428]
    a.plot(windows,values,'o-',c=C['teal'],lw=2.4,ms=8)
    for x,y in zip(windows,values):
        a.annotate(f'{y:.4f}',(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=13)
    a.set(xlim=(.2,10.8),ylim=(.817,.848),xticks=windows,xlabel='Windows per model',ylabel='Recorded local macro AUC')
    a.grid(axis='y',color=C['grid']);a.set_title('Early local window sweep',fontsize=15,pad=15)
    b.bar([0,1,2],windows,width=.55,color=[C['blue'],C['teal'],C['blue']])
    for i,w in enumerate(windows):b.text(i,w+.25,f'{w}×',ha='center',fontsize=14)
    b.set(xticks=[0,1,2],xticklabels=['1','4','10'],ylim=(0,11.7),xlabel='Windows per model',ylabel='Nominal relative model work')
    b.set_title('Fixed models and input recipe',fontsize=15,pad=15)
    b.grid(axis='y',color=C['grid']);b.set_axisbelow(True)
    note(fig,'August 8 local record. Nominal work is not measured runtime; the sweep does not isolate coverage from averaging.')
    save(fig,'part0-windows')

def part1():
    fig,ax=diagram(1,'Training targets, image predictions, and validation references')
    box(ax,1,65,26,23,'Training reports',size=16)
    box(ax,37,65,26,23,'Soft target q',size=17)
    box(ax,73,65,26,23,'Fit image model',size=16)
    arrow(ax,(28,76.5),(36,76.5));arrow(ax,(64,76.5),(72,76.5))
    box(ax,1,10,26,24,'Held-out MRI',size=17)
    box(ax,37,10,26,24,'OOF prediction p',size=16)
    box(ax,73,10,26,24,'Reference AUC',size=16)
    arrow(ax,(28,22),(36,22));arrow(ax,(64,22),(72,22))
    arrow(ax,(80,64),(57,35),dashed=True)
    txt(ax,84,48,'Binary reference y\nfrom report interpretation',size=12)
    arrow(ax,(86,40),(86,35))
    note(fig,'Holding an MRI out of training does not make the report-derived validation reference independent.')
    save(fig,'part1-evaluation')

    d=load('analysis/ruler_audit.json')
    before=d['home_advantage']['shipped_per_ruler'];delta=d['known_candidates']['v1chase_KNOWN_TILT']['delta_per_ruler']
    fig=frame(1,'The “worst of three” objective kept choosing v1')
    ax=fig.add_axes([.13,.20,.46,.58])
    for y,(key,name,col) in enumerate([('v1','v1','orange'),('opus','Opus','teal'),('gpt56sol','GPT','blue')]):
        x=before[key];dx=delta[key]
        ax.plot([x,x+dx],[y,y],color=C[col],lw=3)
        ax.plot(x,y,'o',mfc='white',mec=C[col],ms=9,mew=2)
        ax.plot(x+dx,y,'o',color=C[col],ms=9)
        ax.text(x+dx+(.002 if dx>=0 else -.002),y,f'{dx:+.4f}',
                va='center',ha='left' if dx>=0 else 'right',fontsize=13)
    ax.set(yticks=[0,1,2],yticklabels=['v1','Opus','GPT'],xlim=(.812,.902),ylim=(-.7,2.7),xlabel='Macro AUC on each reference-label table')
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x',color=C['grid']);ax.tick_params(axis='y',length=0)
    fig.text(.67,.66,'200 / 200',fontsize=32,color=C['orange'],weight='bold')
    fig.text(.67,.58,'sampled packages had\nv1 as the minimum.\n\nThis describes that pool;\nit is not a theorem\nabout every future model.',fontsize=15,linespacing=1.6,va='top')
    note(fig,'Open → filled: one historically v1-favoring candidate. Source: the 2026-08-24 reference-label audit.')
    save(fig,'part1-minimum')

    fig=frame(1,'Changing a target can reverse the training update',h=6.5)
    ax=fig.add_axes([.10,.23,.49,.53])
    q=np.linspace(0,1,201)
    ax.plot(q,.20-q,color=C['blue'],lw=2.7)
    ax.axhline(0,color=C['muted'],lw=1,linestyle='--')
    ax.axvline(.20,color=C['grid'],lw=1,linestyle=':')
    ax.scatter([.02,.60],[.18,-.40],s=90,c=[C['teal'],C['orange']],zorder=4)
    ax.annotate('q = 0.02\ngradient = +0.18',xy=(.02,.18),xytext=(.13,.29),
        fontsize=13,ha='left',va='top',color=C['teal'],
        arrowprops=dict(arrowstyle='-',color=C['teal'],lw=1.1))
    ax.annotate('q = 0.60\ngradient = −0.40',xy=(.60,-.40),xytext=(.34,-.64),
        fontsize=13,ha='left',va='top',color=C['orange'],
        arrowprops=dict(arrowstyle='-',color=C['orange'],lw=1.1))
    ax.set(xlim=(-.025,1.025),ylim=(-.85,.35),xlabel='Training target q',
        ylabel='Logit gradient  p − q',xticks=[0,.2,.4,.6,.8,1],yticks=[-.8,-.4,0,.2])
    ax.grid(color=C['grid'],alpha=.65)
    ax.set_axisbelow(True)
    fig.text(.66,.72,'FIXED PREDICTION',fontsize=12,color=C['muted'],weight='bold')
    fig.text(.66,.64,'p = 0.20',fontsize=27,weight='bold')
    fig.text(.66,.48,'Positive gradient',fontsize=16,color=C['teal'],weight='bold')
    fig.text(.66,.42,'Descent lowers the logit.',fontsize=14)
    fig.text(.66,.31,'Negative gradient',fontsize=16,color=C['orange'],weight='bold')
    fig.text(.66,.25,'Descent raises the logit.',fontsize=14)
    note(fig,'Illustrative BCE example, not an experimental result. A percentile rank is not a finding probability.')
    save(fig,'part1-target-scale')

def part2():
    fig,ax=diagram(2,'Test the whole recipe when its components interact')
    box(ax,1,48,44,39,'Parts-only transfer','64-slice representation\n+ existing head / fine-tuning\nFirst-fold AUC: 0.8322',size=16)
    box(ax,55,48,44,39,'Coupled reproduction','Same 64-slice representation\n+ attention head / broader fine-tuning\nFirst-fold AUC: 0.8701',size=16)
    arrow(ax,(46,67),(54,67))
    txt(ax,50,26,'A separate complete-OOF comparison recorded +0.0303.\nThe first-fold values and the full-OOF difference are different measurements.',size=16)
    note(fig,'Both use 64-slice inputs. Head and fine-tuning changed together; the difference is not a pure head effect.')
    save(fig,'part2-recipe')

    d=load('analysis/deployed_composition_20260922.json')
    added=d['never_voting_runs'];runs=[r for r in d['package_runs'] if r not in added]+added
    targets=list(d['per_target']);fig=frame(2,'Computed predictions were not always included in the score',h=7.3)
    labels=[f'DINOv2-S variant {i+1}' for i in range(7)]+['CoAtNet-384','ConvNeXt-L']
    for k,title in enumerate(['OOF comparison: equal mean','Scored composition: fixed target lists']):
        ax=fig.add_axes([.19+k*.41,.27,.37,.48])
        mat=np.array([[1 if k==0 else r in d['per_target'][t]['voting_runs'] for t in targets] for r in runs])
        for i in range(9):
            for j in range(12):
                ax.add_patch(Rectangle((j+.1,8-i+.1),.8,.8,fc=C['teal'] if mat[i,j] else C['white'],ec=C['grid'],lw=.8))
        ax.set(xlim=(0,12),ylim=(0,9),xticks=np.arange(12)+.5,xticklabels=[str(i) for i in range(1,13)])
        ax.set_yticks(np.arange(9)+.5)
        ax.set_yticklabels(labels[::-1] if k==0 else ['']*9,fontsize=12)
        ax.set_aspect('equal');ax.tick_params(length=0);ax.spines[:].set_visible(False)
        ax.set_title(title,fontsize=14,pad=16)
        ax.set_xlabel('Finding index in the fixed target order',fontsize=12)
        ax.plot([0,12],[2,2],color=C['orange'],lw=1.3)
    fig.text(.19,.14,'Filled = participates in that finding. The two lower runs were read, but never voted.',fontsize=14)
    note(fig,'Exact per-target membership replay: September 22, 4,339 shared studies. Membership is not estimated value.')
    save(fig,'part2-members')

    fig,ax=diagram(2,'Evaluate the selection procedure outside its selection folds')
    colors=[C['teal'],C['teal'],C['teal'],C['teal'],C['orange']]
    for j in range(5):
        y=82-j*14
        txt(ax,7,y+3,f'Round {j+1}',size=14)
        for k in range(5):
            selected=(k==j)
            ax.add_patch(Rectangle((19+k*8,y-2),6,10,fc=C['orange'] if selected else C['teal'],ec='white'))
        txt(ax,62,y+3,'Select on 4 → evaluate on 1',size=14,ha='left')
    box(ax,20,0,37,15,'Pool held-out evaluation',size=15)
    box(ax,64,0,34,15,'Refit a final roster',size=15)
    arrow(ax,(37,23),(37,16))
    arrow(ax,(58,7),(63,7),dashed=True)
    txt(ax,81,20,'Separate full-data step',size=12,color=C['muted'])
    note(fig,'Orange: held out by the selector. Evaluation is conditional on the existing OOF bank; final refitting is separate.')
    save(fig,'part2-crossfit')

    L=load('analysis/calibration.json')['points']
    public={p['version']:p['public'] for p in L if p.get('version') is not None and p.get('public') is not None}
    fig=frame(2,'More planned inference did not win these Public comparisons',h=7)
    ax=fig.add_axes([.13,.29,.80,.49])
    rows=[(84,207.7,'10 members · rank mean','blue'),
          (101,207.7,'same 10 · probability mean','teal'),
          (94,539.8,'60-member package','blue'),
          (100,567.8,'65-member package','blue')]
    offsets={84:(19,-.00012),101:(19,-.00006),94:(-182,-.00018),100:(-180,.00005)}
    for version,minutes,label,col in rows:
        y=public[version];ax.plot(minutes,y,'o',color=C[col],ms=10)
        dx,dy=offsets[version]
        ax.annotate(f'v{version} · {y:.3f}\n{label}',xy=(minutes,y),
                    xytext=(minutes+dx,y+dy),textcoords='data',fontsize=12,va='center',
                    arrowprops=dict(arrowstyle='-',color=C['muted'],lw=.8,
                                    shrinkA=5,shrinkB=7))
    ax.set(xlim=(160,635),ylim=(.9338,.942),xlabel='Historical component-based planning cost (minutes)',ylabel='Displayed Public macro AUC')
    ax.grid(color=C['grid'],lw=.7);ax.spines[['top','right']].set_visible(False)
    fig.text(.13,.13,'No numeric score: v96 · 65 members · 599.8 planning min · TIMEOUT',fontsize=14,color=C['orange'])
    note(fig,'Contents, operators and read routes differ. These comparisons do not isolate ensemble size or define a runtime boundary.')
    save(fig,'part2-runtime')

def part3():
    L=load('analysis/calibration.json')['points']
    pub={p['version']:p['public'] for p in L if p.get('version') is not None and p.get('public') is not None}
    rows=[('Raptor · 64-slice fusion',60,81,-.0054),('Raptor · 80-slice gate',79,89,-.0157),
          ('Raptor · 2D windows',54,112,-.0204),('Raptor · ConvNeXt-B gate',83,97,-.0222),
          ('Raptor · 80-slice fusion',58,115,-.0181),('Definition · 2D windows',54,111,.0031),
          ('Definition · 64-slice fusion',60,116,.0109),('Definition · 80-slice gate',79,127,-.0012)]
    fig=frame(3,'A label swap could move OOF and Public in opposite directions',h=7.5)
    ax=fig.add_axes([.34,.19,.60,.60])
    for i,(label,c,t,dx) in enumerate(rows):
        y=7-i;dy=round(pub[t]-pub[c],3)
        for off,val,col in [(.13,dx,'teal'),(-.13,dy,'orange')]:
            ax.plot([0,val],[y+off,y+off],color=C[col],lw=2)
            ax.plot(val,y+off,'o',color=C[col],ms=7)
            ax.text(val+(.00065 if val>=0 else -.00065),y+off,f'{val:+.4f}' if col=='teal' else f'{val:+.3f}',
                    ha='left' if val>=0 else 'right',va='center',fontsize=11.5)
    ax.axvline(0,c=C['muted'],lw=1)
    ax.set(yticks=range(8),yticklabels=[r[0] for r in rows][::-1],xlim=(-.032,.023),ylim=(-.6,7.6),xlabel='Candidate minus the same setup trained on GPT labels')
    ax.spines[['left','top','right']].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=12)
    fig.text(.34,.81,'● Reference-label OOF primary',color=C['teal'],fontsize=13)
    fig.text(.68,.81,'● Displayed Public',color=C['orange'],fontsize=13)
    note(fig,'Paired model setups; historical primary reader. No fitted transfer line, universal noise band, or independence claim.')
    save(fig,'part3-labels')

    d=load('analysis/h1c/pilot_read_20261006.json')
    targets=['Effusion','Synovitis','Contusion']
    fig=frame(3,'Image sensitivity varied across findings in the reader pilot',h=7)
    ax=fig.add_axes([.13,.24,.81,.52]);x=np.arange(3);w=.24
    series=[('Own images added','teal',lambda p:p['A_minus_B']['mean_abs']),
            ('Identical-input repeat','blue',lambda p:p['repeat_mean_abs_dev']),
            ('Other-study images','orange',lambda p:p['D']['mean_abs_D_minus_A'])]
    for k,(label,col,value) in enumerate(series):
        vals=[value(d['per_target'][t]) for t in targets]
        xs=x+(k-1)*w;ax.bar(xs,vals,color=C[col],width=.21,label=label,zorder=3)
        for xi,v in zip(xs,vals):ax.text(xi,v+.002,f'{v:.3f}',ha='center',fontsize=12,weight='bold')
    ax.set(xticks=x,xticklabels=targets,ylim=(0,.12),ylabel='Mean absolute probability change')
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.20),fontsize=12,handlelength=1.2,columnspacing=1.5)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color=C['grid']);ax.set_axisbelow(True)
    note(fig,'Own-image comparison: 100 non-gold studies; repeat and other-study controls: 25 each. Input sensitivity is not clinical accuracy.')
    save(fig,'part3-multimodal')

    fig,ax=diagram(3,'Teacher ancestry determines whether student OOF is clean',h=7)
    for x,title,color in [(1,'Clean double exclusion','teal'),(53,'T* teacher mixture','orange')]:
        box(ax,x,65,46,25,title,size=18,color=color)
    box(ax,1,25,46,26,'Exclude both folds','Student evaluation fold j\n+ teacher-label source fold k',size=16)
    box(ax,53,25,46,26,'Exclude k only in one source','Teacher may have seen fold j;\npublic / package sources also enter',size=16,color='orange')
    arrow(ax,(24,64),(24,52));arrow(ax,(76,64),(76,52),color='orange')
    txt(ax,24,9,'Eligible clean OOF procedure',size=16,weight='bold',color=C['teal'])
    txt(ax,76,9,'All descendants: FINAL-ONLY',size=16,weight='bold',color=C['orange'])
    note(fig,'Diagram of evaluation ancestry, not a quality ranking. A student does not remove information already seen by its teacher.')
    save(fig,'part3-teachers')

def part4():
    L=load('analysis/calibration.json')['points']
    p={x['version']:x['public'] for x in L if x.get('version') is not None and x.get('public') is not None}
    rows=[(219,'H · public-only reference'),(227,'C2 · own share 0.50'),(229,'C2 · own share 0.25'),
          (231,'K-W'),(234,'N2'),(235,'S6FX · final-only'),(236,'K-W2')]
    fig=frame(4,'The selected pair was not the two highest Public scores',h=7)
    ax=fig.add_axes([.34,.17,.59,.63])
    for i,(v,label) in enumerate(rows):
        y=len(rows)-1-i;selected=v in [229,236]
        ax.plot(p[v],y,'D' if selected else 'o',ms=10 if selected else 8,
                color=C['teal'] if selected else C['blue'])
        ax.text(p[v]+.0003,y,f'{p[v]:.3f}'+('  SELECTED' if selected else ''),va='center',fontsize=13)
    ax.set(xlim=(.951,.961),ylim=(-.6,6.6),yticks=range(7),yticklabels=[f'v{v}  {label}' for v,label in rows][::-1],xlabel='Displayed Public macro AUC')
    ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',color=C['grid']);ax.tick_params(axis='y',length=0,labelsize=12)
    note(fig,'Selection snapshot: October 21. C2 v229 + C1 v236 were selected; Private outcomes were not known.')
    save(fig,'part4-candidates')

    fig=frame(4,'K-W2 versus K-W: two kinds of evidence disagreed')
    ax=fig.add_axes([.27,.34,.65,.42])
    ax.axvline(0,color=C['muted'],lw=1)
    ax.plot([.00055,.00136],[1,1],c=C['teal'],lw=3)
    ax.plot(.00095,1,'o',c=C['teal'],ms=10)
    ax.plot(-.001,0,'D',c=C['orange'],ms=10)
    ax.text(.00148,1,'+0.00095\n5 / 5 folds positive',fontsize=13,va='center')
    ax.text(-.001,0.25,'−0.001',fontsize=14,ha='center',color=C['orange'])
    ax.set(xlim=(-.0018,.0029),ylim=(-.5,1.6),yticks=[1,0],yticklabels=['Clean OOF primary','Displayed Public'],xlabel='K-W2 minus K-W (macro AUC)')
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.text(.10,.14,'OOF interval: [+0.00055, +0.00136], conditional on this reader and cohort.\nIt is not a Public noise interval or a Private probability.',fontsize=14,linespacing=1.6)
    note(fig,'AUC gains are not guaranteed by seed averaging. Retaining K-W2 is a choice under conflicting evidence.')
    save(fig,'part4-evidence')

    fig,ax=diagram(4,'What the final S6F/N comparisons actually changed',h=7.5)
    columns=[(2,'Candidate'),(36,'S6F fold\nmodels'),(49,'N fold\nmodels'),(62,'V3 fold\nmodels'),(75,'Full-data\nX'),(89,'Public result')]
    for x,label in columns:txt(ax,x,95,label,size=13,weight='bold',ha='left' if x==2 else 'center')
    rows=[(231,'K-W',[5,5,5,0],f'{p[231]:.3f}'),(234,'N2',[5,10,5,0],f'{p[234]:.3f}'),
          (235,'S6FX',[5,5,5,1],f'{p[235]:.3f}'),(236,'K-W2',[10,5,5,0],f'{p[236]:.3f}'),
          (237,'N2W2',[10,10,5,0],'TIMEOUT'),(242,'K-W2+X',[10,5,5,1],'TIMEOUT'),
          (243,'N2W2+X',[10,10,5,1],'TIMEOUT'),(244,'K-W3 r1',[15,5,5,0],'TIMEOUT')]
    for i,(v,name,counts,result) in enumerate(rows):
        y=81-i*10.3
        if v==236:ax.add_patch(Rectangle((0,y-4.7),100,9.4,fc='#E4F0EF',ec='none'))
        ax.plot([0,100],[y-5,y-5],c=C['grid'],lw=.8)
        txt(ax,2,y,f'v{v} · {name}',ha='left',size=14,weight='bold' if v==236 else 'normal')
        for x,count in zip([36,49,62,75],counts):
            txt(ax,x,y,str(count) if count else '—',size=16,color=C['orange'] if x==75 and count else C['ink'])
        txt(ax,89,y,result,size=13,weight='bold' if v==236 else 'normal',color=C['orange'] if result=='TIMEOUT' else C['teal'])
    note(fig,'Counts describe hidden-read contents, not equal probability weights. X makes a row final-only; C2-N3 and R7+O use other routes.')
    save(fig,'part4-budget')

    rel='leaderboard_snapshots/rsna-knee-abnormality-detection-publicleaderboard-2026-10-21T09:00:41.csv'
    source=RSNA/rel;SOURCES[rel]=hashlib.sha256(source.read_bytes()).hexdigest()
    rows=list(csv.DictReader(source.read_text(encoding='utf-8-sig').splitlines()))
    counts=Counter(r['Score'] for r in rows)
    scores=sorted([v for v in counts if float(v)>=.953],key=float,reverse=True)
    fig=frame(4,'1,471 teams displayed 0.958 on October 21',h=8.8)
    ax=fig.add_axes([.12,.17,.79,.65])
    yy=np.arange(len(scores));xx=[counts[v] for v in scores]
    ax.barh(yy,xx,height=.62,color=[C['teal'] if v=='0.958' else C['blue'] for v in scores])
    for i,n in enumerate(xx):ax.text(n+13,i,f'{n:,}',va='center',fontsize=12)
    ax.set(yticks=yy,yticklabels=scores,xlim=(0,1660),xlabel='Teams with this displayed team-best Public score')
    ax.invert_yaxis();ax.grid(axis='x',color=C['grid']);ax.set_axisbelow(True)
    ax.tick_params(axis='y',length=0,labelsize=12);ax.spines[['top','right','left']].set_visible(False)
    fig.text(.54,.79,'6,307 teams in the snapshot\n228 displayed more than 0.958',fontsize=14,linespacing=1.5,va='top')
    note(fig,'Snapshot: October 21 09:00:41 UTC. Equal displayed scores do not prove identical predictions or a Private outcome.')
    save(fig,'part4-public-density')

def editorial_graph_data(name):
    for evidence in sorted(CHECKS.glob('depth-*-evidence.md')):
        for block in re.findall(r'```json\s*\n(.*?)\n```',evidence.read_text(),flags=re.S):
            obj=json.loads(block)
            for d in obj.get('figures',[obj]):
                if d.get('asset',d.get('output',Path(d.get('figure_path','')).name)) != name+'.png':continue
                sources=list(d.get('sources',[]))
                for key,hashkey in [('source','source_sha256'),('score_source','score_source_sha256')]:
                    if key in d:sources.append({'path':d[key],'sha256':d.get(hashkey)})
                for source in sources:
                    raw=(RSNA/source['path']).read_bytes();sha=hashlib.sha256(raw).hexdigest()
                    if source.get('sha256') and sha != source['sha256']:raise ValueError(f"Changed graph input: {source['path']}")
                    SOURCES[source['path']]=sha
                return d
    raise ValueError(f'No frozen graph data for {name}')

def depth_figures():
    # Frozen existing comparisons; no fitted slopes or new uncertainty estimates.
    d=editorial_graph_data('part1-omission-mask')
    fig=frame(1,'An omitted finding was different from an explicit denial',h=7.5)
    a=fig.add_axes([.10,.35,.32,.40]);b=fig.add_axes([.62,.35,.31,.40])
    rows=d['expert_panel_cells']
    a.bar([0,1],[r['image_annotation_positive_fraction'] for r in rows],width=.48,color=[C['blue'],C['orange']])
    for i,r in enumerate(rows):
        a.text(i,r['image_annotation_positive_fraction']+.009,f"{100*r['image_annotation_positive_fraction']:.1f}%\n{r['cells']} cells",ha='center',fontsize=13,linespacing=1.5)
    a.set(xticks=[0,1],xticklabels=['Explicit denial','Not mentioned'],ylim=(0,.21),ylabel='Image-annotation positive fraction')
    a.set_yticks([0,.05,.10,.15,.20],['0%','5%','10%','15%','20%']);a.set_title('Small expert panel',fontsize=15,pad=17)
    rows=d['wide_vs_baseline_target_groups']
    for i,r in enumerate(rows):
        b.barh(1-i,r['mean_auc_delta'],height=.4,color=[C['teal'],C['orange']][i])
        b.text(r['mean_auc_delta'],1-i+.28,f"{r['mean_auc_delta']:+.4f}",ha='center',fontsize=13)
    b.axvline(0,c=C['muted'],lw=1)
    b.set(yticks=[1,0],yticklabels=['Usually mentioned\n5 findings','Often unmentioned\n6 findings'],xlim=(-.043,.018),ylim=(-.6,1.6),xlabel='Wide recipe minus baseline: mean AUC')
    b.set_title('Finding groups in the local comparison',fontsize=14,pad=17)
    b.tick_params(axis='y',length=0,labelsize=11);b.tick_params(axis='x',labelsize=11)
    fig.text(.09,.16,'A separate masking comparison reported +0.0083 macro AUC, with interval [0.0055, 0.0112].\nIt is a different contrast, not an increment to add to either group above.',fontsize=13,linespacing=1.5)
    note(fig,'August record. Expert-panel cells are not independent patients. One finding is outside the two omission groups.')
    save(fig,'part1-omission-mask')

    d=editorial_graph_data('part1-crop-scale')
    fig=frame(1,'A tighter crop and more pixels bought overlapping detail',h=7.4)
    a=fig.add_axes([.23,.32,.34,.43]);b=fig.add_axes([.70,.32,.23,.43])
    for i,r in enumerate(d['points']):
        y=3-i;col=C['muted'] if i==0 else C['teal'] if i<3 else C['orange']
        if r.get('reported_ci95'):
            lo,hi=r['reported_ci95'];a.plot([lo,hi],[y,y],c=col,lw=3)
        a.plot(r['reported_delta'],y,'o',c=col,ms=8)
        a.text(r['reported_delta'],y+.23,r['display_delta'],ha='center',fontsize=12,color=col)
    a.axvline(0,c=C['muted'],lw=1)
    a.set(yticks=[3,2,1,0],yticklabels=['336 px / 130 mm','448 px / 130 mm','336 px / 100 mm','448 px / 100 mm'],xlim=(-.001,.009),ylim=(-.5,3.6),xlabel='Reported paired macro AUC difference')
    a.set_title('Two ways to increase output detail',fontsize=14,pad=18);a.tick_params(axis='y',length=0,labelsize=12);a.tick_params(axis='x',labelsize=11)
    curve=sorted(d['separate_crop_curve_at_336px'],key=lambda r:r['crop_mm'])
    b.axhline(0,c=C['muted'],lw=1);b.plot([r['crop_mm'] for r in curve],[r['reported_delta'] for r in curve],'-o',c=C['blue'],ms=7)
    b.set(xlim=(72,168),ylim=(-.003,.007),xticks=[80,100,130,160],xlabel='Physical crop width (mm)',ylabel='Reported AUC difference')
    b.set_title('Separate crop scan at 336 px',fontsize=14,pad=18);b.tick_params(labelsize=11)
    fig.text(.09,.15,'The combined +0.0053 was close to the crop-only +0.0051. Smaller crops also lost context.\nIntervals are reported where available; no interval was inferred for the combined configuration.',fontsize=13,linespacing=1.5)
    note(fig,'August historical paired comparisons. Output pixel pitch is not native scanner resolution or a clinical detection threshold.')
    save(fig,'part1-crop-scale')

    d=editorial_graph_data('part2-read-interaction')
    fig=frame(2,'Learning rate and trainable depth interacted strongly',h=7.6)
    a=fig.add_axes([.10,.34,.32,.42]);b=fig.add_axes([.67,.34,.26,.42])
    s=d['rate_depth_primary_auc']
    a.plot([0,1],[s['A'],s['R']],'-o',c=C['blue'],lw=2.5,label='Partial depth')
    a.plot([0,1],[s['D'],s['RD']],'-s',c=C['orange'],lw=2.5,label='Full depth')
    for x,k,dy in [(0,'A',.012),(0,'D',-.024),(1,'R',.012),(1,'RD',.012)]:
        a.text(x,s[k]+dy,f"{k}: {s[k]:.4f}",ha='center',fontsize=12)
    a.set(xticks=[0,1],xticklabels=['Original rate\n8 × 10⁻⁶','Raised rate\n2 × 10⁻⁵'],xlim=(-.22,1.22),ylim=(.70,.915),ylabel='Local primary AUC')
    a.set_title('Same DINOv2-small fold and seed',fontsize=14,pad=17);a.legend(frameon=False,fontsize=11,loc='lower left')
    keys=['CORNER','CORR','GEOM'];labels=['Full dense recipe','Rate restored','Geometry only']
    for i,k in enumerate(keys):
        r=d['full_corner_effects'][k];y=2-i;lo,hi=r['ci95'];col=C['orange'] if i==0 else C['teal'] if i==1 else C['blue']
        b.plot([lo,hi],[y,y],c=col,lw=3);b.plot(r['point'],y,'o',ms=8,c=col)
        b.text(r['point'],y+.24,f"{r['point']:+.4f}",ha='center',fontsize=12,color=col)
    b.axvline(0,c=C['muted'],lw=1)
    b.set(yticks=[2,1,0],yticklabels=labels,xlim=(-.16,.015),ylim=(-.5,2.6),xlabel='Difference from stored baseline A')
    b.set_title('Separate full-recipe repair',fontsize=14,pad=17);b.tick_params(axis='y',length=0,labelsize=11);b.tick_params(axis='x',labelsize=11)
    fig.text(.09,.15,'The rate × depth interaction was −0.1040 AUC [−0.1154, −0.0926].\nRestoring the rate recovered much of the full recipe loss; the geometry-related loss remained.',fontsize=13,linespacing=1.5)
    note(fig,'881 studies, one training realization. Saved study-resampling intervals do not measure variability across independent retraining.')
    save(fig,'part2-read-interaction')

    d=editorial_graph_data('part2-budget-sensitivity')
    fig=frame(2,'Planning-budget sensitivity depended on the selection method',h=7.2)
    a=fig.add_axes([.12,.30,.79,.46])
    for method,col,label in [('flat',C['blue'],'Flat family mean'),('per_target',C['teal'],'Per-target selection')]:
        rows=sorted([r for r in d['rows'] if r['method']==method],key=lambda r:r['ceiling_min'])
        a.plot([r['ceiling_min'] for r in rows],[r['primary_gain'] for r in rows],'-o',c=col,label=label,lw=2.5,ms=7)
    r=next(r for r in d['rows'] if r['method']=='per_target' and r['ceiling_min']==300)
    a.plot(300,r['primary_gain'],'o',mfc='white',mec=C['orange'],mew=2.5,ms=12)
    a.annotate('Two outer folds fell back to the comparator',xy=(300,r['primary_gain']),xytext=(335,.0047),fontsize=12,arrowprops={'arrowstyle':'->','color':C['muted']})
    a.set(xlim=(225,555),ylim=(.0035,.011),xticks=[240,300,360,420,480,540],xlabel='Planning ceiling (minutes)',ylabel='Cross-fitted pooled primary gain')
    a.grid(axis='y',color=C['grid']);a.legend(frameon=False,fontsize=12,loc='upper left')
    fig.text(.10,.14,'Same recorded 83-run candidate pool. The final flat roster used 25 models from five families,\nplanned at 233.8 minutes. Cap sensitivity is distinct from statistical selection uncertainty.',fontsize=13,linespacing=1.5)
    note(fig,'September 23 saved selection results. These points are not a measured hidden-inference runtime curve.')
    save(fig,'part2-budget-sensitivity')

    for rel,expected in [
      ('analysis/mprob_fulloof_20261006.json','af0fef0bdc9f658e3d93c4e0e488cf33fc99b764d070858a7b8000f06be9de36'),
      ('analysis/g25defn_fulloof_20261008.json','db1d2f4785718d467e3698f7fce815decd3c938a82657c89b623c71bdb582290'),
      ('analysis/p1sb_fulloof_20261006.json','b50abd39e028156fe0501dd93dd9821f5a0a5530208c575cfd530b001b3aebf1'),
      ('analysis/p1r_fulloof_20261006.json','73a450792688f1aec55eff2425273c2f90f9b52d4defec01681ecdebf1b252ec'),
      ('analysis/p1sa_fulloof_20261006.json','455d585394e8d7f934e4686b0b1d80cc138d7c78a52fdb044b770494ed363d83')]:
        raw=(RSNA/rel).read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if sha!=expected:raise ValueError(f'Changed depth-graph input: {rel}')
        SOURCES[rel]=sha
    d=editorial_graph_data('part3-transfer-contrasts')
    fig=frame(3,'Objective and recipe changes gave different transfer results',h=6.8)
    ax=fig.add_axes([.27,.30,.66,.43]);ax.axvline(0,c=C['muted'],lw=1)
    for i,r in enumerate(d['rows']):
        y=1-i;x=r['local_primary_delta'];lo,hi=r['local_ci95']
        ax.plot([lo,hi],[y+.11,y+.11],c=C['teal'],lw=3)
        ax.plot(x,y+.11,'o',ms=9,c=C['teal'],label='Local primary + saved interval' if i==0 else None)
        ax.plot(r['displayed_public_delta'],y-.11,'D',ms=9,c=C['orange'],label='Displayed Public difference' if i==0 else None)
        ax.text(x,y+.28,f'{x:+.4f}',color=C['teal'],ha='center',fontsize=13)
        ax.text(r['displayed_public_delta'],y-.30,f"{r['displayed_public_delta']:+.3f}",color=C['orange'],ha='center',fontsize=13)
    ax.set(yticks=[1,0],yticklabels=['Two-window objective\n(probability BCE − mean BCE)','Image recipe\n(G25 − V3; same labels)'],
        ylim=(-.65,1.65),xlim=(-.015,.017),xlabel='Candidate minus control: macro AUC')
    ax.tick_params(axis='y',length=0,labelsize=12);ax.grid(axis='x',color=C['grid']);ax.spines[['top','right','left']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.45,1.15),frameon=False,fontsize=12,ncol=1)
    fig.text(.09,.12,'These are different experimental contrasts. The second changes the complete image recipe.\nPublic points have no estimated error bars; the two rows do not define a transfer regression.',fontsize=13,linespacing=1.5)
    note(fig,'October 6–10 recorded outcomes. Local intervals are conditional OOF comparisons, not calibrated Public uncertainty.')
    save(fig,'part3-transfer-contrasts')

    d=editorial_graph_data('part3-teacher-controls')
    fig=frame(3,'Teacher content, prior shrinkage, and the Public result',h=7.6)
    a=fig.add_axes([.26,.29,.31,.48]);b=fig.add_axes([.68,.29,.25,.48])
    labels=['Definition control','All-target teacher','Prior mixture','Four-target teacher']
    colors=[C['muted'],C['teal'],C['orange'],C['blue']]
    for i,(r,col) in enumerate(zip(d['rows'],colors)):
        y=3-i
        if r['local_ci95']:
            lo,hi=r['local_ci95'];a.plot([lo,hi],[y,y],c=col,lw=2.7)
        a.plot(r['primary_delta'],y,'o',c=col,ms=8)
        a.text(r['primary_delta'],y+.26,f"{r['primary_delta']:+.4f}",ha='center',color=col,fontsize=12)
        b.plot(r['public'],y,'D',c=col,ms=9)
        b.text(r['public'],y+.26,f"{r['public']:.3f}",ha='center',color=col,fontsize=13)
    a.axvline(0,c=C['muted'],lw=1)
    a.set(yticks=range(4),yticklabels=labels[::-1],ylim=(-.6,3.7),xlim=(-.016,.015),xlabel='Local primary difference vs control')
    b.set(yticks=range(4),yticklabels=['']*4,ylim=(-.6,3.7),xlim=(.902,.917),xlabel='Displayed Public AUC')
    for ax in [a,b]:
        ax.grid(axis='x',color=C['grid']);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=12);ax.tick_params(axis='x',labelsize=11)
    a.set_title('Same small student recipe',fontsize=15,pad=16);b.set_title('Completed fixed comparisons',fontsize=15,pad=16)
    fig.text(.08,.14,'Teacher and prior mixtures both use a 0.5 coefficient, but do not match target entropy.\nThe four-target mask was informed by exposed gold58; it is not an untouched confirmation set.',fontsize=13,linespacing=1.5)
    note(fig,'Stored October 6–8 comparisons. Teacher versus prior is a specific control contrast, not a pure causal information decomposition.')
    save(fig,'part3-teacher-controls')

    d=editorial_graph_data('part4-e32-decision')
    fig=frame(4,'Longer training had different standalone and ensemble effects',h=6.7)
    ax=fig.add_axes([.28,.30,.62,.45]);ax.axvline(0,color=C['muted'],lw=1)
    for i,r in enumerate(d['rows']):
        y=1-i;col=C['orange'] if i==0 else C['teal']
        ax.plot([r['low'],r['high']],[y,y],c=col,lw=3)
        ax.plot(r['delta_primary'],y,'o',c=col,ms=10)
        ax.text(r['delta_primary'],y+.20,f"{r['delta_primary']:+.5f}",ha='center',fontsize=14,color=col)
    ax.set(yticks=[1,0],yticklabels=['S6F standalone','Replacement in fixed K-W'],
        xlim=(-.013,.005),ylim=(-.5,1.6),xlabel='e32 minus e16: local primary AUC')
    ax.grid(axis='x',color=C['grid']);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=12)
    fig.text(.10,.15,'One fold: 870 matched non-gold studies. Four folds remained untrained.\nFinishing them was planned at ~14.8 training GPU-hours, before export and test inference.',fontsize=13,linespacing=1.5)
    note(fig,'Stored report-group intervals, conditional on this pilot. These are not full-family or Public measurements.')
    save(fig,'part4-e32-decision')

    d=editorial_graph_data('part4-search-frontier')
    fig=frame(4,'Selection estimates and fitted-roster costs are different objects',h=8.1)
    a=fig.add_axes([.24,.30,.29,.48]);b=fig.add_axes([.67,.30,.27,.48])
    labels=['Fixed K-W','Select fourth family','P540 procedure','PUNC procedure','Fixed C1U after scan']
    for i,r in enumerate(d['score_panel']):
        hollow=r['type']!='selection cross-fit';y=4-i
        a.plot(r['within_primary'],y,'o',ms=9,mfc='white' if hollow else C['teal'],mec=C['blue'] if hollow else C['teal'],mew=2)
        a.text(r['within_primary'],y+.27,f"{r['within_primary']:.6f}",fontsize=11,ha='center')
    a.set(yticks=range(5),yticklabels=labels[::-1],xlim=(.9010,.9038),ylim=(-.6,4.7),xlabel='Mean within-fold primary AUC')
    a.tick_params(axis='y',length=0,labelsize=12);a.tick_params(axis='x',labelsize=11)
    a.set_xticks([.901,.902,.903]);a.grid(axis='x',color=C['grid']);a.spines[['top','right','left']].set_visible(False)
    a.set_title('Conditional OOF comparison',fontsize=15,pad=16)
    costs=d['separate_fitted_proposal_cost_panel']
    b.barh([1,0],[r['planning_minutes'] for r in costs],height=.42,color=[C['teal'],C['blue']])
    for i,r in enumerate(costs):
        b.text(r['planning_minutes']-20,1-i,f"{r['planning_minutes']:.1f}",fontsize=13,ha='right',va='center',color='white',weight='bold')
    b.axvline(540,color=C['orange'],lw=1.4,linestyle='--')
    b.set(yticks=[1,0],yticklabels=['P540 fit','PUNC fit'],xlim=(0,1120),ylim=(-.6,1.6),xlabel='Planning minutes')
    b.set_title('Separate fitted proposals',fontsize=15,pad=16);b.tick_params(axis='y',length=0,labelsize=12);b.spines[['top','right','left']].set_visible(False)
    b.text(558,1.42,'540',fontsize=11,color=C['orange'])
    fig.text(.08,.16,'Filled: selection cross-fit. Open: fixed prediction, including post-selection C1U.\nP540 used one S6F seed; PUNC used a two-seed input. Cost bars do not inherit the left-hand AUCs.',fontsize=13,linespacing=1.5)
    note(fig,'October 19 saved comparisons. This is not an AUC–runtime frontier or a matched intervention on the time cap.')
    save(fig,'part4-search-frontier')

def contact_sheet():
    paths=sorted(OUT.glob('*.png'))
    sheet=Image.new('RGB',(1440, int(np.ceil(len(paths)/3))*300),'#e5eaec')
    draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(paths):
        im=Image.open(p).convert('RGB');im.thumbnail((460,265))
        x=(i%3)*480+(480-im.width)//2;y=(i//3)*300+25
        sheet.paste(im,(x,y));draw.text(((i%3)*480+12,(i//3)*300+6),p.name,fill='#183344')
    sheet.save(CHECKS/'figure-contact-sheet.png')

if __name__=='__main__':
    covers();part0();part1();part2();part3();part4();depth_figures();contact_sheet()
    manifest={'purpose':'Unpublished bilingual RSNA series figures', 'source_sha256':SOURCES,
              'assets':[{ 'file':str(p.relative_to(BLOG)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                          'dimensions':Image.open(p).size } for p in sorted(OUT.glob('*.png'))],
              'cover_web_assets':[{ 'file':str(p.relative_to(BLOG)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                          'dimensions':Image.open(p).size } for p in sorted(OUT.glob('cover-*.webp'))],
              'checks':'Programmatic diagram text boxes checked for containment. Six editorial covers are independently authored conceptual artwork, inspected visually and verified against cover-redesign.json. Part 5 diagrams are maintained by their separate generator; browser QA is recorded separately.'}
    (CHECKS/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Verified 6 editorial covers; rebuilt the original 26 body figures and SVG sources; indexed {len(manifest["assets"])} PNG masters and {len(manifest["cover_web_assets"])} web covers.')
