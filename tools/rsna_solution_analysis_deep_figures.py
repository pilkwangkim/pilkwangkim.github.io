"""Additional English figures for the expanded RSNA solution analysis.

Rebuild with /opt/anaconda3/bin/python tools/rsna_solution_analysis_deep_figures.py.
Only the four stems declared below and their provenance manifest are written.
No model execution or original-series asset mutation is performed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/img/posts/rsna-knee-solutions'
OUT.mkdir(parents=True, exist_ok=True)
C={'ink':'#183344','muted':'#526572','teal':'#277F85','teal_light':'#DCECEE',
   'blue':'#426E9C','blue_light':'#EBF1F7','ochre':'#AB7440','ochre_light':'#F7EEE3',
   'grid':'#DCE4E7','light':'#F3F6F7','white':'#FFFFFF','gray':'#8C9BA3'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'text.color':C['ink'],
 'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':C['muted'],
 'xtick.color':C['muted'],'ytick.color':C['ink'],'axes.edgecolor':C['grid'],
 'svg.fonttype':'none','savefig.facecolor':'white'})
RECORD=[]
BOXCHECK=[]


def base(title, subtitle, width=1800, height=1100, mobile=False):
 dpi=100 if mobile else 150
 fig=plt.figure(figsize=(width/dpi,height/dpi),dpi=dpi,facecolor='white')
 fig.text(.055,.96,'RSNA KNEE  /  SOLUTION ANALYSIS',fontsize=12,weight='bold',color=C['teal'],va='top')
 fig.text(.055,.91,title,fontsize=23 if mobile else 25,weight='bold',va='top',linespacing=1.22)
 fig.text(.055,.82 if mobile else .835,subtitle,fontsize=16 if mobile else 15,color=C['muted'],va='top',linespacing=1.35)
 return fig


def rect(ax, x, y, w, h, fc='light', ec='grid', radius=.02):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.005,rounding_size={radius}',
  facecolor=C.get(fc,fc),edgecolor=C.get(ec,ec),lw=1.3)
 ax.add_patch(p)
 return p


def txt(ax,x,y,s,fs=17,ha='center',va='center',weight='normal',color='ink',box=None):
 t=ax.text(x,y,s,fontsize=fs,ha=ha,va=va,fontweight=weight,color=C.get(color,color),linespacing=1.35)
 if box is not None:BOXCHECK.append((t,box))
 return t


def save(fig, name):
 fig.canvas.draw();ren=fig.canvas.get_renderer();errors=[]
 for t,p in BOXCHECK:
  if t.figure is not fig:continue
  tb=t.get_window_extent(ren);pb=p.get_window_extent(ren)
  if tb.x0<pb.x0+1 or tb.x1>pb.x1-1 or tb.y0<pb.y0+1 or tb.y1>pb.y1-1:errors.append('box:'+t.get_text())
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_visible() or not t.get_text():continue
  b=t.get_window_extent(ren)
  if b.x0<-1 or b.y0<-1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1:errors.append('figure:'+t.get_text())
 if errors:raise ValueError((name,errors))
 for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=fig.dpi)
 with Image.open(OUT/f'{name}.png') as im:dims=list(im.size)
 RECORD.append({'stem':name,'width':dims[0],'height':dims[1],'text_bounds':'passed'})
 plt.close(fig)


def credit(scores):
 p=np.array(scores[:2]);n=np.array(scores[2:]);return (p[:,None]>n[None,:]).astype(float)+.5*(p[:,None]==n[None,:])


def average_rank(scores):
 # Unique scores in each example; mean rank retains ordinary average-tie semantics.
 s=np.array(scores);return np.array([1+np.sum(s<x)+.5*(np.sum(s==x)-1) for x in s])/len(s)


TOY=[{'case':'A','label':'Complementary errors','base':[.9,.4,.6,.1],'candidate':[.5,.9,.1,.6]},
     {'case':'B','label':'A shared hard negative','base':[.9,.4,.8,.1],'candidate':[.4,.9,.8,.1]}]
for row in TOY:
 row['mean_scores']=list((np.array(row['base'])+row['candidate'])/2)
 row['mean_ranks']=list((average_rank(row['base'])+average_rank(row['candidate']))/2)
 row['aucs']={k:float(credit(row[k]).mean()) for k in ['base','candidate','mean_scores','mean_ranks']}
assert TOY[0]['aucs']=={'base':.75,'candidate':.75,'mean_scores':1.0,'mean_ranks':1.0}
assert TOY[1]['aucs']=={'base':.75,'candidate':.75,'mean_scores':.5,'mean_ranks':.75}


def pair_card(ax,x,y,w,h,title,scores,mobile=False):
 p=rect(ax,x,y,w,h)
 txt(ax,x+w/2,y+h*.91,title,fs=17 if mobile else 16,weight='bold',box=p)
 # One cell is a positive-negative pair, not an individual study.
 vals=credit(scores);gw=w*.55;gh=h*.36;gx=x+w*.29;gy=y+h*.43
 for i in range(2):
  for j in range(2):
   value=vals[i,j];color='teal_light' if value==1 else 'ochre_light' if value==0 else 'blue_light'
   ax.add_patch(Rectangle((gx+j*gw/2,gy+(1-i)*gh/2),gw/2-.008,gh/2-.009,facecolor=C[color],edgecolor=C['white'],lw=1))
   txt(ax,gx+(j+.5)*gw/2-.004,gy+(1.5-i)*gh/2-.004,f'{value:g}',fs=18 if mobile else 17,weight='bold',color='teal' if value==1 else 'ochre' if value==0 else 'blue')
 for i in range(2):txt(ax,gx-.025,gy+(1.5-i)*gh/2,f'P{i+1}',fs=14 if mobile else 13,ha='right')
 for j in range(2):txt(ax,gx+(j+.5)*gw/2,gy-.045*h,f'N{j+1}',fs=14 if mobile else 13,va='top')
 txt(ax,x+w/2,y+h*.12,f'AUC {vals.mean():.2f}',fs=19 if mobile else 18,weight='bold',color='teal' if vals.mean()>.75 else 'ochre' if vals.mean()<.75 else 'ink',box=p)


def pairwise(mobile=False):
 fig=base('The same component AUC,\ndifferent ensemble outcomes' if mobile else 'The same component AUC, different ensemble outcomes',
  'Assumed four-study examples · two positives, two negatives',width=760 if mobile else 1800,height=2100 if mobile else 1200,mobile=mobile)
 ax=fig.add_axes([.045,.045,.91,.76 if mobile else .67]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 titles=[('base','Base'),('candidate','Candidate'),('mean_scores','Score average'),('mean_ranks','Rank average')]
 if mobile:
  for k,row in enumerate(TOY):
   top=.97-k*.49
   txt(ax,.01,top,f"{row['case']} · {row['label']}",fs=21,ha='left',weight='bold')
   for j,(key,title) in enumerate(titles):
    x=.01+(j%2)*.50;y=top-.225-(j//2)*.215
    pair_card(ax,x,y,.47,.195,title,row[key],True)
 else:
  for k,row in enumerate(TOY):
   top=.98-k*.51
   txt(ax,.005,top,f"{row['case']} · {row['label']}",fs=20,ha='left',weight='bold')
   for j,(key,title) in enumerate(titles):pair_card(ax,.005+j*.25,top-.42,.232,.34,title,row[key])
 fig.text(.055,.018,'Pair credit: 1 = correct order   ·   0.5 = tie   ·   0 = reversed',fontsize=13 if mobile else 14,color=C['muted'])
 save(fig,'pairwise-complementarity'+('-mobile' if mobile else ''))


TARGETS=[('Synovitis',.80),('PF OA',.60),("Baker’s cyst",.35),('Lateral OA',.20),('Other findings',.50)]


def composition(mobile=False):
 fig=base('The insertion point changes\nthe coefficient' if mobile else 'The insertion point changes the coefficient',
  ('Reduced two-branch example\ninternal β = 0.30; outside δ = 0.20' if mobile else 'Reduced two-branch example · internal β = 0.30; outside δ = 0.20'),width=760 if mobile else 1800,height=1500 if mobile else 1080,mobile=mobile)
 if mobile:
  ax=fig.add_axes([.055,.60,.89,.115]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');p=rect(ax,0,0,1,1,fc='light')
  txt(ax,.5,.73,'H = wA + (1 − w)B',20,weight='bold',box=p)
  txt(ax,.5,.3,'Internal: β(1 − w)     Outside: δ',18,box=p)
  chart=fig.add_axes([.29,.16,.63,.385])
 else:
  ax=fig.add_axes([.055,.28,.35,.40]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');p=rect(ax,.01,.03,.98,.94,fc='light')
  txt(ax,.5,.78,'H = wA + (1 − w)B',23,weight='bold',box=p)
  txt(ax,.5,.53,'S6 inserted inside B',19,color='teal',box=p)
  txt(ax,.5,.40,'Coefficient: β(1 − w)',20,color='teal',weight='bold',box=p)
  txt(ax,.5,.2,'S6 outside H: coefficient δ',19,color='blue',box=p)
  chart=fig.add_axes([.59,.25,.34,.45])
 ys=np.arange(len(TARGETS))[::-1];h=.42
 inner=np.array([.3*(1-w) for _,w in TARGETS]);outside=np.repeat(.2,len(TARGETS))
 chart.barh(ys+h/2,inner,height=h,color=C['teal']);chart.barh(ys-h/2,outside,height=h,color=C['blue'])
 for y,vi,vo in zip(ys,inner,outside):
  chart.text(vi+.008,y+h/2,f'{vi:.3f}',ha='left',va='center',fontsize=17 if mobile else 15,color=C['teal'],weight='bold')
  chart.text(vo+.008,y-h/2,f'{vo:.3f}',ha='left',va='center',fontsize=17 if mobile else 15,color=C['blue'])
 chart.set_yticks(ys,[f'{n}\nw = {w:.2f}' for n,w in TARGETS]);chart.tick_params(axis='y',length=0,pad=13,labelsize=16 if mobile else 15)
 chart.set_xlim(0,.305);chart.set_xticks([0,.1,.2,.3]);chart.set_ylim(-.7,4.7);chart.spines['left'].set_visible(False);chart.set_axisbelow(True);chart.grid(axis='x',color=C['grid'])
 chart.set_xlabel('Pre-rank coefficient',fontsize=17 if mobile else 16,labelpad=12)
 fig.text(.055,.08,'■ Internal S6 in B',fontsize=16,color=C['teal']);fig.text(.055 if mobile else .37,.052 if mobile else .08,'■ Outside S6',fontsize=16,color=C['blue'])
 fig.text(.055,.021,'Reader arm omitted; coefficients are not final AUC influence.',fontsize=12 if mobile else 14,color=C['muted'])
 save(fig,'composition-placement'+('-mobile' if mobile else ''))


def anchor(mobile=False):
 fig=base('Retaining the original\ntarget explicitly' if mobile else 'Retaining the original target explicitly',
  'Direct S term only · teacher ancestry is not shown',width=760 if mobile else 1800,height=1350 if mobile else 1050,mobile=mobile)
 if mobile:
  ax=fig.add_axes([.055,.58,.89,.16]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');p=rect(ax,0,0,1,1)
  txt(ax,.5,.74,'Fixed: Lₜ = 0.6S + 0.4Tₜ',19,color='teal',weight='bold',box=p)
  txt(ax,.5,.28,'Recursive: Lₜ = 0.6Lₜ₋₁ + 0.4Tₜ',18,color='ochre',box=p)
  chart=fig.add_axes([.16,.15,.77,.35])
 else:
  ax=fig.add_axes([.055,.30,.37,.35]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');p=rect(ax,0,0,1,1)
  txt(ax,.5,.77,'Fixed original anchor',20,color='teal',weight='bold',box=p)
  txt(ax,.5,.60,'Lₜ = 0.6S + 0.4Tₜ',21,color='teal',box=p)
  txt(ax,.5,.34,'Recursive mixture',20,color='ochre',weight='bold',box=p)
  txt(ax,.5,.17,'Lₜ = 0.6Lₜ₋₁ + 0.4Tₜ',20,color='ochre',box=p)
  chart=fig.add_axes([.55,.23,.38,.48])
 t=np.arange(1,6);fixed=np.repeat(.6,5);rec=.6**t
 chart.plot(t,fixed,'o-',color=C['teal'],lw=3,markersize=8,label='Fixed S: 0.6')
 chart.plot(t,rec,'o-',color=C['ochre'],lw=3,markersize=8,label='Recursive: 0.6ᵗ')
 for x,y in zip(t,rec):chart.annotate(f'{y:.3f}',(x,y),xytext=(0,15 if y<.15 else -27),textcoords='offset points',ha='center',fontsize=15 if mobile else 14,color=C['ochre'])
 chart.text(5,.635,'0.600',ha='center',va='bottom',fontsize=16 if mobile else 15,color=C['teal'],weight='bold')
 chart.set(xlim=(.7,5.3),ylim=(0,.75),xticks=t);chart.grid(color=C['grid']);chart.set_axisbelow(True)
 chart.set_xlabel('Mixing round t',fontsize=17 if mobile else 16,labelpad=12);chart.set_ylabel('Direct coefficient on original S',fontsize=15 if mobile else 16,labelpad=10)
 fig.text(.055,.042,'L₀ = S; the teacher term may contain earlier targets.',fontsize=13 if mobile else 14,color=C['muted'])
 save(fig,'teacher-anchor'+('-mobile' if mobile else ''))


SAMPLING=[{'slices':44,'span':'Original span','score':.926},
          {'slices':44,'span':'2–98% span','score':.917},
          {'slices':64,'span':'2–98% span','score':.928}]


def sampling(mobile=False):
 fig=base('Wider coverage also needs\nenough sampling positions' if mobile else 'Wider coverage also needs enough sampling positions',
  '977th-place account · reported sampling comparison',width=760 if mobile else 1800,height=1350 if mobile else 1050,mobile=mobile)
 ax=fig.add_axes([.15,.30,.78,.41]) if mobile else fig.add_axes([.14,.36,.73,.36])
 xs=np.arange(3);ys=[r['score'] for r in SAMPLING]
 ax.plot(xs,ys,color=C['grid'],lw=3,zorder=1)
 for x,y,col in zip(xs,ys,['blue','ochre','teal']):
  ax.scatter([x],[y],s=170,color=C[col],zorder=2)
  ax.annotate(f'{y:.3f}',(x,y),xytext=(0,18),textcoords='offset points',ha='center',fontsize=22 if mobile else 20,color=C[col],weight='bold')
 ax.set_xticks(xs,[f"{r['slices']} slices\n{r['span']}" for r in SAMPLING]);ax.tick_params(axis='x',length=0,pad=16,labelsize=17 if mobile else 18)
 ax.set_ylim(.912,.934);ax.set_xlim(-.35,2.35);ax.set_yticks([.915,.920,.925,.930]);ax.set_axisbelow(True);ax.grid(axis='y',color=C['grid'])
 ax.spines['left'].set_visible(False);ax.spines['bottom'].set_visible(False);ax.set_ylabel('Reported score',fontsize=18 if mobile else 17,labelpad=12)
 fig.text(.055,.15 if mobile else .18,'Same 44 positions, wider span: −0.009',fontsize=18 if mobile else 18,color=C['ochre'],weight='bold')
 fig.text(.055,.105 if mobile else .125,'64 positions at the wider span: +0.011',fontsize=18 if mobile else 18,color=C['teal'],weight='bold')
 fig.text(.055,.045,'Original span is not specified in the opening.\nThe comparison is not labeled as Private AUC.',fontsize=13 if mobile else 14,color=C['muted'],linespacing=1.4)
 save(fig,'sampling-density'+('-mobile' if mobile else ''))


def main():
 for mobile in [False,True]:
  pairwise(mobile);composition(mobile);anchor(mobile);sampling(mobile)
 data={'pairwise-complementarity':{'scope':'Assumed four-study binary examples, not competition results. P1/P2 positive, N1/N2 negative. Average ranks use average tie handling.','examples':TOY},
  'composition-placement':{'scope':'Reduced two-branch pre-rank coefficient example. Excludes reader arm. Coefficient is not causal influence on final prediction or AUC.','internal_beta':.3,'outside_delta':.2,'targets':[{'target':n,'w':w,'internal':.3*(1-w),'outside':.2} for n,w in TARGETS]},
  'teacher-anchor':{'scope':'Direct explicit coefficient on original S outside the teacher; does not quantify total recursive supervision ancestry.','rounds':list(range(1,6)),'fixed':[.6]*5,'recursive':[float(.6**t) for t in range(1,6)],'related_source':'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/8th-place-solution'},
  'sampling-density':{'scope':'Author-reported comparison; opening does not state original span or identify metric as Private AUC.','rows':SAMPLING,'source_url':'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752231'}}
 files=[]
 for r in RECORD:
  for ext in ['png','svg']:
   p=OUT/f"{r['stem']}.{ext}";files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 (OUT/'deep-figure-sources.json').write_text(json.dumps({'figures':data,'dimensions':RECORD,'files':files},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'figures':len(RECORD),'dimensions':RECORD},ensure_ascii=False))


if __name__=='__main__':main()
