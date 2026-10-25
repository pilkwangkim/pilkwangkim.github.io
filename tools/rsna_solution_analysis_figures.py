"""English figures for the standalone RSNA solution analysis.

Static source inspection and author-reported ablations; no model execution.
Rebuild: python3 tools/rsna_solution_analysis_figures.py [--source-archive PATH]
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/img/posts/rsna-knee-solutions'
OUT.mkdir(parents=True,exist_ok=True)
C={'ink':'#183344','muted':'#526572','teal':'#277F85','teal_light':'#DCECEE',
   'blue':'#426E9C','blue_light':'#EBF1F7','ochre':'#AB7440','ochre_light':'#F7EEE3',
   'grid':'#DCE4E7','light':'#F3F6F7','white':'#FFFFFF','gray':'#8C9BA3'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':C['muted'],'xtick.color':C['muted'],
 'ytick.color':C['ink'],'text.color':C['ink'],'axes.edgecolor':C['grid'],
 'svg.fonttype':'none','savefig.facecolor':'white'})
SOURCES={
 'label-transfer':{'url':'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752125',
  'scope':'Author-reported matched five-fold models at fixed epoch 16. OOF evaluated on 832 predefined random studies against VLM references; gold-58 is a reused expert development set.',
  'rows':[{'model':'ConvNeXt-Tiny','report_oof':.94109,'vlm_oof':.95248,'report_gold':.90922,'vlm_gold':.93528},
          {'model':'ConvNeXt-Small','report_oof':.93862,'vlm_oof':.95957,'report_gold':.91472,'vlm_gold':.94282}]},
 'site-ablation':{'url':'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752111',
  'scope':'Author-reported local diagnostic paragraph, and separately the exact 288px competition ablation. Do not substitute later teacher-retrained label scores.',
  'local':[{'metric':'Pooled report OOF','before':.912,'after':.906}, {'metric':'Within-site OOF','before':.906,'after':.907}],
  'competition_288':[{'metric':'Public','before':.947,'after':.953}, {'metric':'Private','before':.932,'after':.939}]},
 'spatial-supervision':{'url':'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752474',
  'scope':'Sakaku training-only annotation branch. Anatomy is structure support, lesion is abnormal evidence. Inference omits annotation losses; not a statement about the detector-based tascj branch.'},
 'distillation-folds':{'url':'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752125',
  'scope':'21st: teacher and student train on the same outer-train four folds; teacher predictions on A are in-sample. Outer validation B excluded. Original VLM q_B is the validation reference.'},
 'target-site':{'scope':'Conceptual information paths, not an identified causal model. Site contains scanner appearance and reporting practice; also may carry real case-mix.'}}

BOXCHECK=[]
RECORD=[]
def figbase(title,sub,width=1800,height=1050,mobile=False):
 if mobile:
  title={
   'The training target and the test target differ':'The training target and\nthe test target differ',
   'A label change, with the learner held fixed':'A label change, with\nthe learner held fixed',
   'Removing a report shortcut can lower report OOF':'Removing a report shortcut\ncan lower report OOF',
   'Where to evaluate, and where the lesion is':'Where to evaluate,\nand where the lesion is',
   'In-sample teacher targets, held-out student evaluation':'In-sample teacher targets,\nheld-out student evaluation'}[title]
  sub=sub.replace('A held-out row can still reward the wrong reporting convention.','A held-out row can still reward\nthe wrong reporting convention.').replace('21st place · A is outer training; B is outer validation.','21st place · A is outer training;\nB is outer validation.')
 dpi=100 if mobile else 150
 fig=plt.figure(figsize=(width/dpi,height/dpi),dpi=dpi,facecolor='white')
 fig.text(.055,.955,'RSNA KNEE  /  SOLUTION ANALYSIS',fontsize=12 if not mobile else 12,weight='bold',color=C['teal'],va='top')
 fig.text(.055,.902,title,fontsize=25 if not mobile else 22,weight='bold',va='top',linespacing=1.22)
 fig.text(.055,.835 if not mobile else .82,sub,fontsize=15 if not mobile else 16,color=C['muted'],va='top',linespacing=1.35)
 return fig

def axdiag(fig,top=.75,bottom=.06):
 ax=fig.add_axes([.04,bottom,.92,top-bottom]);ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off');return ax

def text(ax,x,y,s,fs=17,color='ink',weight='normal',ha='center',va='center'):
 return ax.text(x,y,s,fontsize=fs,color=C.get(color,color),fontweight=weight,ha=ha,va=va,linespacing=1.35)

def box(ax,x,y,w,h,title,body=None,color='teal',fs=17,fill=None):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=1.5',lw=1.5,
  edgecolor=C[color],facecolor=C[fill] if fill else C['light']);ax.add_patch(p)
 if body:
  a=text(ax,x+w/2,y+h*.7,title,fs,weight='bold');b=text(ax,x+w/2,y+h*.3,body,fs-1,color='muted');BOXCHECK.extend([(a,p),(b,p)])
 else:
  a=text(ax,x+w/2,y+h/2,title,fs,weight='bold');BOXCHECK.append((a,p))
 return p

def arr(ax,a,b,color='muted',style='-',lw=1.7):
 p=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=18,lw=lw,color=C[color],linestyle=style,shrinkA=0,shrinkB=0);ax.add_patch(p);return p

def elbow(ax,pts,color='muted',dashed=False):
 for a,b in zip(pts[:-2],pts[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=C[color],lw=1.7,ls='--' if dashed else '-')
 arr(ax,pts[-2],pts[-1],color,'--' if dashed else '-')

def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();issues=[]
 for t,p in BOXCHECK:
  if t.figure is not fig:continue
  tb=t.get_window_extent(renderer);pb=p.get_window_extent(renderer)
  if tb.x0<pb.x0+2 or tb.x1>pb.x1-2 or tb.y0<pb.y0+2 or tb.y1>pb.y1-2:issues.append(t.get_text())
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_text() or not t.get_visible():continue
  b=t.get_window_extent(renderer)
  if b.x0<-1 or b.y0<-1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1:issues.append('figure boundary: '+t.get_text())
 if issues:raise ValueError((name,issues))
 fig.savefig(OUT/f'{name}.png',dpi=fig.dpi)
 fig.savefig(OUT/f'{name}.svg')
 with Image.open(OUT/f'{name}.png') as im:sz=list(im.size)
 RECORD.append({'name':name,'width':sz[0],'height':sz[1], 'text_bounds':'passed'})
 plt.close(fig)

def target_site(mobile=False):
 fig=figbase('The training target and the test target differ',
  'A held-out row can still reward the wrong reporting convention.',height=1120 if mobile else 1050,width=760 if mobile else 1800,mobile=mobile)
 ax=axdiag(fig,top=.72 if mobile else .74)
 if mobile:
  fs=20
  box(ax,3,72,40,22,'Site S','Scanner + reporting\npractice',color='ochre',fs=fs,fill='ochre_light')
  box(ax,57,72,40,22,'Report target Q','Extracted finding\nprobability',color='ochre',fs=fs,fill='ochre_light')
  box(ax,3,39,40,22,'MRI X','Disease evidence\n+ site appearance',fs=fs,fill='teal_light')
  box(ax,57,39,40,22,'Image model','Fit to target Q',fs=fs,fill='teal_light')
  box(ax,3,6,40,22,'Expert label Y','Image-based\ndiagnostic target',color='blue',fs=fs,fill='blue_light')
  box(ax,57,6,40,22,'Test score','Compare p(X)\nwith Y',color='blue',fs=fs,fill='blue_light')
  arr(ax,(44,83),(56,83),'ochre');arr(ax,(23,71),(23,62),'ochre','--');arr(ax,(77,71),(77,62),'ochre')
  # Q also contains clinical image information, not only site/reporting effects.
  arr(ax,(44,62),(60,71),'teal')
  text(ax,43,66,'Image\nfindings',16,color='teal',ha='right')
  arr(ax,(44,50),(56,50),'teal');arr(ax,(77,38),(77,29),'teal');arr(ax,(44,17),(56,17),'blue')
 else:
  box(ax,2,65,25,25,'Site S','Scanner + reporting\npractice',color='ochre',fill='ochre_light')
  box(ax,37.5,65,25,25,'Report target Q','Extracted finding\nprobability',color='ochre',fill='ochre_light')
  box(ax,73,65,25,25,'Expert label Y','Image-based\ndiagnostic target',color='blue',fill='blue_light')
  box(ax,2,13,25,25,'MRI X','Disease evidence\n+ site appearance',fill='teal_light')
  box(ax,37.5,13,25,25,'Image model','Fit to target Q',fill='teal_light')
  box(ax,73,13,25,25,'Test score','Compare p(X)\nwith Y',color='blue',fill='blue_light')
  arr(ax,(28,77.5),(36.5,77.5),'ochre');arr(ax,(14.5,64),(14.5,39),'ochre','--')
  text(ax,12,51,'Site\nappearance',14,color='ochre',ha='right')
  arr(ax,(28,39),(42,64),'teal')
  text(ax,31,59,'Image\nfindings',14,color='teal',ha='right')
  arr(ax,(50,64),(50,39),'ochre');text(ax,53,51,'Supervision',14,color='ochre',ha='left')
  arr(ax,(85.5,64),(85.5,39),'blue');arr(ax,(28,25.5),(36.5,25.5),'teal');arr(ax,(63.5,25.5),(72,25.5),'teal')
 save(fig,'target-site'+('-mobile' if mobile else ''))

def label_transfer(mobile=False):
 fig=figbase('A label change, with the learner held fixed',
  '21st place · matched five-fold models at epoch 16',width=760 if mobile else 1800,height=1280 if mobile else 1050,mobile=mobile)
 panels=[(.94109,.95248,.93862,.95957),(.90922,.93528,.91472,.94282)]
 for j,vals in enumerate(panels):
  if mobile:ax=fig.add_axes([.24,.485-j*.37,.68,.245])
  else:ax=fig.add_axes([.12+j*.485,.23,.34,.46])
  r1,v1,r2,v2=vals
  for y,r,v in [(1,r1,v1),(0,r2,v2)]:
   ax.plot([r,v],[y,y],color=C['grid'],lw=4,zorder=1)
   ax.scatter([r],[y],s=90,c=C['blue'],zorder=3);ax.scatter([v],[y],s=90,c=C['teal'],zorder=3)
   ax.annotate(f'{r:.5f}',(r,y),xytext=(0,-25),textcoords='offset points',ha='center',fontsize=15 if not mobile else 17,color=C['blue'])
   ax.annotate(f'{v:.5f}',(v,y),xytext=(0,17),textcoords='offset points',ha='center',fontsize=15 if not mobile else 17,color=C['teal'])
  ax.set_yticks([0,1],['Small','Tiny']);ax.tick_params(axis='y',length=0,pad=12,labelsize=16 if not mobile else 18)
  ax.set_ylim(-.65,1.65);ax.set_axisbelow(True);ax.grid(axis='x',color=C['grid'],lw=.8)
  ax.spines['left'].set_visible(False)
  if j==0:
   ax.set_xlim(.934,.965);ax.set_xticks([.935,.945,.955,.965]);title='OOF · 832-study fixed subset';subtitle='VLM-derived reference'
  else:
   ax.set_xlim(.902,.95);ax.set_xticks([.91,.92,.93,.94]);title='Gold · 58 expert-labeled studies';subtitle='Reused development set'
  ax.set_xlabel('Macro AUC',fontsize=16 if not mobile else 18,labelpad=12)
  ax.set_title(title,fontsize=17 if not mobile else 18,pad=38,fontweight='bold')
  ax.text(.5,1.07,subtitle,transform=ax.transAxes,ha='center',fontsize=14 if not mobile else 16,color=C['muted'])
 fig.text(.13,.085 if not mobile else .035,'●  Report-consensus labels',color=C['blue'],fontsize=16 if not mobile else 17)
 fig.text(.56 if not mobile else .13,.085 if not mobile else .013,'●  Image-assisted VLM labels',color=C['teal'],fontsize=16 if not mobile else 17)
 save(fig,'label-transfer'+('-mobile' if mobile else ''))

def site_ablation(mobile=False):
 fig=figbase('Removing a report shortcut can lower report OOF',
  '72nd place · training-only site bias',width=760 if mobile else 1800,height=1280 if mobile else 990,mobile=mobile)
 groups=[('Local diagnostic',SOURCES['site-ablation']['local']),('288 px competition ablation',SOURCES['site-ablation']['competition_288'])]
 for j,(title,rows) in enumerate(groups):
  ax=fig.add_axes([.34,.51-j*.365,.55,.205]) if mobile else fig.add_axes([.22+j*.485,.27,.23,.39])
  for y,d in zip([1,0],rows):
   delta=d['after']-d['before'];col=C['teal'] if delta>=0 else C['ochre']
   ax.barh(y,delta,color=col,height=.3,zorder=2)
   ax.text(.001 if delta<0 else -.001,y,f'{delta:+.3f}',ha='left' if delta<0 else 'right',va='center',fontsize=17 if mobile else 16,color=col,weight='bold')
   ax.text(0,y-.33,f"{d['before']:.3f} → {d['after']:.3f}",ha='center',va='center',fontsize=17 if mobile else 15,color=C['muted'])
  ax.set_ylim(-.62,1.62);ax.set_xlim(-.008,.009);ax.set_yticks([1,0],[d['metric'].replace(' ','\n',1) if mobile else d['metric'] for d in rows]);ax.tick_params(axis='y',length=0,labelsize=17 if mobile else 15,pad=13)
  ax.set_xticks([-.006,0,.006],['−0.006','0','+0.006']);ax.axvline(0,color=C['muted'],lw=1)
  ax.grid(axis='x',color=C['grid'],zorder=0);ax.spines['left'].set_visible(False);ax.set_axisbelow(True)
  ax.set_xlabel('Change in AUC',fontsize=18 if mobile else 16,labelpad=12)
  ax.set_title(title,fontsize=19 if mobile else 17,pad=30,weight='bold')
 save(fig,'site-ablation'+('-mobile' if mobile else ''))

def featuregrid(ax,x,y,w,h,kind):
 # A feature-space schematic, not a fabricated medical image.
 for i in range(5):
  for j in range(5):
   val=.2+.5*((i*3+j*2)%7)/6
   ax.add_patch(Rectangle((x+i*w/5,y+j*h/5),w/5-.25,h/5-.25,ec='none',fc=plt.cm.Greys(val*.45)))
 if kind=='anatomy':
  ax.add_patch(Rectangle((x+.18*w,y+.17*h),.65*w,.66*h,fc=C['teal'],alpha=.25,ec='none'))
  ax.add_patch(Rectangle((x+.18*w,y+.17*h),.65*w,.66*h,fill=False,ec=C['teal'],lw=2.3))
 else:
  for u,v,a,b in [(.3,.4,.22,.18),(.6,.65,.17,.15)]:
   ax.add_patch(Rectangle((x+u*w,y+v*h),a*w,b*h,fc=C['ochre'],alpha=.5,ec=C['ochre'],lw=2))
 ax.add_patch(Rectangle((x,y),w,h,fill=False,ec=C['gray'],lw=1))

def spatial(mobile=False):
 fig=figbase('Where to evaluate, and where the lesion is',
  'Two spatial targets supervise the same image features.',width=760 if mobile else 1800,height=1480 if mobile else 1200,mobile=mobile)
 ax=axdiag(fig,top=.75)
 if mobile:
  cards=[(3,65,94,29,'Anatomy','anatomy'),(3,29,94,29,'Lesion','lesion')]
  for x,y,w,h,title,kind in cards:
   p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.2,rounding_size=1.2',fc=C['light'],ec=C['grid'],lw=1.5);ax.add_patch(p)
   featuregrid(ax,x+4,y+5,26,20,kind)
   text(ax,x+35,y+22,title,23,ha='left',weight='bold',color='teal' if kind=='anatomy' else 'ochre')
   body='Structure support\nNormal or abnormal' if kind=='anatomy' else 'Positive evidence\nOnly when annotated'
   text(ax,x+35,y+11,body,20,ha='left')
  box(ax,3,4,26,14,'Images',fs=20,fill='blue_light')
  box(ax,37,4,26,14,'Shared\nfeatures',fs=20,fill='teal_light')
  box(ax,71,4,26,14,'Finding\nhead',fs=20,fill='blue_light')
  arr(ax,(30,11),(36,11),'blue');arr(ax,(64,11),(70,11),'blue')
  # Plain dashed annotation joins, then one downward supervision arrow.
  ax.plot([3,1,1,50],[79.5,79.5,26.5,26.5],color=C['ochre'],lw=1.7,ls='--')
  ax.plot([3,1],[43.5,43.5],color=C['ochre'],lw=1.7,ls='--')
  arr(ax,(50,26.5),(50,19),'ochre','--')
  text(ax,24,22.5,'Spatial losses\ntraining only',17,ha='center',color='ochre')
 else:
  for x,title,kind in [(2,'Anatomy','anatomy'),(53,'Lesion','lesion')]:
   p=FancyBboxPatch((x,44),45,49,boxstyle='round,pad=.2,rounding_size=1.2',fc=C['light'],ec=C['grid'],lw=1.5);ax.add_patch(p)
   text(ax,x+22.5,86,title,22,weight='bold',color='teal' if kind=='anatomy' else 'ochre')
   featuregrid(ax,x+4,54,17,23,kind)
   body='Structure support\nNormal or abnormal' if kind=='anatomy' else 'Positive evidence\nOnly when annotated'
   text(ax,x+24,67,body,17,ha='left')
  box(ax,2,4,25,18,'Images',fs=19,fill='blue_light');box(ax,37.5,4,25,18,'Shared features',fs=19,fill='teal_light');box(ax,73,4,25,18,'Finding head',fs=19,fill='blue_light')
  arr(ax,(28,13),(36.5,13),'blue');arr(ax,(63.5,13),(72,13),'blue')
  # Merge both annotation paths without a directional horizontal edge.
  ax.plot([24.5,24.5,75.5,75.5],[43,33,33,43],color=C['ochre'],lw=1.7,ls='--')
  arr(ax,(50,33),(50,23),'ochre','--')
  text(ax,50,38,'Spatial losses: training only',16,color='ochre')
 save(fig,'spatial-supervision'+('-mobile' if mobile else ''))

def distillation(mobile=False):
 fig=figbase('In-sample teacher targets, held-out student evaluation',
  '21st place · A is outer training; B is outer validation.',width=760 if mobile else 1800,height=1600 if mobile else 1050,mobile=mobile)
 ax=axdiag(fig,top=.75)
 if mobile:
  outer=FancyBboxPatch((3,43),94,54,boxstyle='round,pad=.2,rounding_size=1.5',fc=C['blue_light'],ec=C['blue'],lw=1.5);ax.add_patch(outer)
  text(ax,50,93,'Outer training A · four folds',20,weight='bold',color='blue')
  box(ax,16,77,68,11,'Teacher trained on A',fs=20)
  box(ax,16,61,68,11,'Target on A: 0.75q + 0.25t(A)',fs=19)
  box(ax,16,45,68,11,'Student trained on A',fs=20)
  arr(ax,(50,76),(50,73),'blue');arr(ax,(50,60),(50,57),'blue')
  held=FancyBboxPatch((3,1),94,34,boxstyle='round,pad=.2,rounding_size=1.5',fc=C['teal_light'],ec=C['teal'],lw=1.5);ax.add_patch(held)
  text(ax,50,31,'Held-out B · one fold',20,weight='bold',color='teal')
  box(ax,16,17,68,10,'Apply student to MRI in B',fs=20)
  box(ax,16,3,68,10,'Score against original\nVLM q(B)',fs=19)
  arr(ax,(50,16),(50,14),'teal');arr(ax,(50,44),(50,36),'blue');text(ax,52,39.5,'Fitted student',17,ha='left',color='blue')
 else:
  outer=FancyBboxPatch((2,54),96,42,boxstyle='round,pad=.2,rounding_size=1.5',fc=C['blue_light'],ec=C['blue'],lw=1.5);ax.add_patch(outer)
  text(ax,50,89,'Outer training A · four folds',19,weight='bold',color='blue')
  box(ax,5,62,25,17,'Teacher','Train on A',fs=17)
  box(ax,37.5,62,25,17,'Targets on A','0.75q + 0.25t(A)',fs=16)
  box(ax,70,62,25,17,'Student','Train on A',fs=17)
  arr(ax,(31,70.5),(36.5,70.5),'blue');arr(ax,(63.5,70.5),(69,70.5),'blue')
  held=FancyBboxPatch((2,3),96,32,boxstyle='round,pad=.2,rounding_size=1.5',fc=C['teal_light'],ec=C['teal'],lw=1.5);ax.add_patch(held)
  text(ax,50,29,'Held-out B · excluded from teacher and student training',18,weight='bold',color='teal')
  box(ax,5,7,25,14,'MRI in B',fs=18)
  box(ax,37.5,7,25,14,'Apply student',fs=18)
  box(ax,70,7,25,14,'Score vs q(B)',fs=18)
  arr(ax,(31,14),(36.5,14),'teal');arr(ax,(63.5,14),(69,14),'teal')
  elbow(ax,[(82.5,61),(82.5,43),(50,43),(50,36)],'blue');text(ax,53,47,'Fitted student',15,ha='left',color='blue')
 save(fig,'distillation-folds'+('-mobile' if mobile else ''))

def cover():
 fig=plt.figure(figsize=(16.6,9.48),dpi=100,facecolor='#FAFCFC')
 ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1660),ylim=(0,948));ax.axis('off')
 # Familiar editorial rules and an abstract supervision motif, not MRI iconography.
 ax.add_patch(Rectangle((26,916),1608,8,fc=C['teal'],ec='none'))
 text(ax,95,784,'RSNA KNEE  /  SOLUTION ANALYSIS',19,color='teal',weight='bold',ha='left')
 text(ax,95,583,'LEARNING THE\nRIGHT TARGET',49,weight='bold',ha='left')
 text(ax,95,385,'Supervision, anatomy,\nand the reporting process',24,color='muted',ha='left')
 ax.plot([95,649],[306,306],color=C['grid'],lw=1.3)
 # Three clean information grids merge into a shared representation and output.
 for x,y,col in [(835,690,'teal'),(835,455,'ochre'),(835,220,'blue')]:
  for i in range(3):
   for j in range(3):
    fill=C[col] if (i+j)%3==0 else C['grid']
    ax.add_patch(FancyBboxPatch((x+i*45,y+j*39),37,31,boxstyle='round,pad=0,rounding_size=3',ec='none',fc=fill,alpha=.8))
 ax.plot([1000,1082,1082],[748,748,516],color=C['teal'],lw=2)
 ax.plot([1000,1082],[513,513],color=C['ochre'],lw=2)
 ax.plot([1000,1082,1082],[278,278,510],color=C['blue'],lw=2)
 for i in range(5):
  for j in range(5):
   col=C['teal'] if i in (1,2) and j in (1,2,3) else C['grid']
   ax.add_patch(FancyBboxPatch((1130+i*47,395+j*47),38,38,boxstyle='round,pad=0,rounding_size=4',ec='none',fc=col))
 arr(ax,(1084,513),(1115,513),'ink',lw=2)
 arr(ax,(1380,513),(1445,513),'ink',lw=2)
 for j,val in enumerate([.4,.72,.87,.53]):
  ax.add_patch(FancyBboxPatch((1480,409+j*48),100*val,22,boxstyle='round,pad=0,rounding_size=5',ec='none',fc=C['teal'] if j in (1,2) else C['blue'],alpha=.85))
 ax.plot([95,1565],[129,129],color=C['grid'],lw=1.3)
 text(ax,95,89,'LABELS  /  FEATURES  /  VALIDATION',15,color='muted',ha='left')
 fig.savefig(OUT/'cover.png',dpi=100);fig.savefig(OUT/'cover.svg');plt.close(fig)
 with Image.open(OUT/'cover.png') as im:im.convert('RGB').save(OUT/'cover.webp',quality=94,method=6)
 RECORD.append({'name':'cover','width':1660,'height':948})

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source-archive',type=Path,
  help='Optional directory of messages-ID.json responses to fingerprint in the manifest.')
 args=parser.parse_args()
 cover()
 for fun in [target_site,label_transfer,site_ablation,spatial,distillation]:
  fun(False);fun(True)
 raw=args.source_archive
 for name,tid in [('label-transfer',752125),('site-ablation',752111),('spatial-supervision',752474),('distillation-folds',752125)]:
  source=raw/f'messages-{tid}.json' if raw else None
  if source is not None and source.exists():
   SOURCES[name]['archived_response_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
 manifest={'sources':SOURCES,'figures':RECORD}

 for item in manifest['figures']:
  f=OUT/(item['name']+'.png');item['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
 (OUT/'figure-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'figures':len(RECORD),'out':str(OUT),'text_box_bounds':'passed'}))
