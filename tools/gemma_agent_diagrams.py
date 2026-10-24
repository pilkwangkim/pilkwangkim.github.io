"""Editable SVG architecture figures for Gemma Developer Agent Part 1.

Only standard Python and librsvg's rsvg-convert are needed. The small limit
example is a behavioral illustration, not competition task or benchmark data.
Rebuild: python3 tools/gemma_agent_diagrams.py
"""
from html import escape
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parents[1] / 'assets/img/posts/gemma-4-series'
BG, INK, MUTED = '#fbfbf8', '#183044', '#5c6d78'
TEAL, TL, AMBER, AL = '#167d82', '#e8f4f2', '#ad6825', '#fcf0dc'
LINE, WHITE, RED, RL = '#cbd8dd', '#ffffff', '#a24d3e', '#f9ebe6'

class SVG:
    def __init__(self, w, h, title, desc):
        self.w, self.h, self.a = w, h, []
        self.a.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">')
        self.a.append(f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>')
        self.a.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="context-stroke" stroke-width="1.6"/></marker></defs>')
        self.rect(0,0,w,h,BG,stroke='none',r=0)
    def rect(self,x,y,w,h,fill=WHITE,stroke=LINE,r=12,sw=1.5):
        self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(self,x,y,s,size=20,fill=INK,weight=400,anchor='start',mono=False):
        family='Menlo, Consolas, monospace' if mono else 'Arial, Helvetica, sans-serif'
        self.a.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" xml:space="preserve">{escape(str(s))}</text>')
    def path(self,d,color=INK,sw=2.2,arrow=False,dash=False):
        extra=(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="5 5"' if dash else '')
        self.a.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"{extra}/>')
    def line(self,x1,y1,x2,y2,color=LINE,sw=1.5,arrow=False,dash=False):
        self.path(f'M{x1},{y1} L{x2},{y2}',color,sw,arrow,dash)
    def circle(self,x,y,r,fill=WHITE,stroke=LINE,sw=1.5):
        self.a.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def check(self,x,y,size=18):
        self.circle(x,y,size/2,TL,TEAL,1.3)
        self.path(f'M{x-size*.23},{y} l{size*.16},{size*.18} l{size*.34},{-size*.38}',TEAL,1.8)
    def step(self,x,y,n,title,size=24):
        self.circle(x+13,y-8,13,INK,INK)
        self.text(x+13,y-2,n,16,WHITE,700,'middle')
        self.text(x+36,y,title,size,INK,700)
    def terminal(self,x,y,w,h,label,lines,size=19):
        self.rect(x,y,w,h,WHITE,INK,r=10)
        self.rect(x,y,w,30,INK,INK,r=10)
        self.rect(x,y+17,w,13,INK,INK,r=0)
        if len(label) <= 15:
            for i in range(3): self.circle(x+14+i*12,y+15,2.5,'#9ab3be','none')
        self.text(x+w-12,y+21,label,16 if w<350 else 18,WHITE,400,'end')
        for i,(s,c) in enumerate(lines): self.text(x+16,y+58+i*(size+11),s,size,c,mono=True)
    def folder(self,x,y,w,h,label,size=20):
        self.path(f'M{x},{y+15} v{-12} h{w*.38} l12,12 h{w*.62-12} v{h-15} H{x} Z',INK,1.8)
        self.path(f'M{x},{y+15} h{w}',INK,1.4)
        self.text(x+14,y+45,label,size,INK,700)
    def file(self,x,y,w,h,fill=WHITE,stroke=INK):
        self.a.append(f'<path d="M{x},{y} H{x+w-17} L{x+w},{y+17} V{y+h} H{x} Z" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        self.path(f'M{x+w-17},{y} v17 h17',stroke,1.4)
    def chip(self,x,y,w,h):
        for dx in range(18,int(w)-8,21):
            self.line(x+dx,y-8,x+dx,y,INK,2)
            self.line(x+dx,y+h,x+dx,y+h+8,INK,2)
        for dy in range(18,int(h)-8,21):
            self.line(x-8,y+dy,x,y+dy,INK,2)
            self.line(x+w,y+dy,x+w+8,y+dy,INK,2)
        self.rect(x,y,w,h,INK,INK,9)
        self.rect(x+9,y+9,w-18,h-18,INK,'#6e8b9c',6)
        self.text(x+w/2,y+h/2-2,'Gemma',24,WHITE,700,'middle')
        self.text(x+w/2,y+h/2+25,'GPU model server',16,'#c9dde5',400,'middle')
    def shield(self,x,y,w,h):
        self.a.append(f'<path d="M{x+w/2},{y} Q{x+w*.8},{y+12} {x+w},{y+13} V{y+h*.5} Q{x+w},{y+h*.8} {x+w/2},{y+h} Q{x},{y+h*.8} {x},{y+h*.5} V{y+13} Q{x+w*.2},{y+12} {x+w/2},{y} Z" fill="{TL}" stroke="{TEAL}" stroke-width="2"/>')
        self.path(f'M{x+w*.25},{y+h*.49} l{w*.18},{h*.18} l{w*.34},{-h*.34}',TEAL,4)
    def save(self,name):
        OUT.mkdir(parents=True,exist_ok=True)
        svg=OUT/(name+'.svg'); svg.write_text('\n'.join(self.a)+ '\n</svg>\n')
        subprocess.run(['rsvg-convert','-w',str(self.w*2),'-h',str(self.h*2),str(svg),'-o',str(OUT/(name+'.png'))],check=True)


def item_array(s,x,y,scale=1):
    s.path(f'M{x+13*scale},{y-3*scale} H{x+5*scale} V{y+51*scale} H{x+13*scale}',AMBER,2)
    for i in range(10):
        col,row=i%5,i//5
        s.rect(x+22*scale+col*24*scale,y+row*28*scale,18*scale,20*scale,AL,AMBER,3*scale)
        s.line(x+26*scale+col*24*scale,y+7*scale+row*28*scale,x+36*scale+col*24*scale,y+7*scale+row*28*scale,AMBER,1)
    s.path(f'M{x+150*scale},{y-3*scale} H{x+158*scale} V{y+51*scale} H{x+150*scale}',AMBER,2)


def issue_figure(mobile=False):
    w,h=(390,1510) if mobile else (1120,825)
    s=SVG(w,h,'Follow an explicit zero from request to repair',
          'A toy API uses limit as the maximum returned item count. An explicit zero becomes ten through value or default. A None-only fallback preserves None to ten, zero to zero, and five to five.')
    if mobile:
        s.text(24,35,'Follow the value to the bug',23,weight=700)
        s.text(24,64,'limit = maximum items returned',17,fill=MUTED)
        # Reproduce the behavior.
        s.step(24,111,1,'Reproduce the mismatch',21)
        s.terminal(25,132,340,85,'API request',[('GET /items?limit=0',INK)],17)
        s.line(195,220,195,245,INK,2,True)
        s.text(35,274,'Expected',17,TEAL,700)
        s.text(35,316,'[]',36,TEAL,mono=True)
        s.text(175,274,'Current response',17,AMBER,700)
        item_array(s,174,290,.9)
        s.text(35,363,'0 items',17,TEAL)
        s.text(175,363,'10 items',17,AMBER)
        s.line(195,386,195,420,INK,2,True)
        # Locate shared conversion.
        s.step(24,454,2,'Locate the conversion',21)
        s.folder(30,480,330,139,'query/',19)
        s.line(48,537,48,583,LINE,2)
        s.line(48,548,66,548,LINE,2)
        s.line(48,583,66,583,TEAL,2)
        s.text(77,554,'options.py',18,fill=MUTED,mono=True)
        s.rect(67,565,258,32,TL,'none',5)
        s.text(77,588,'limits.py',18,TEAL,700,mono=True)
        s.terminal(25,639,340,95,'normalize_limit', [('return value or default',INK)],17)
        s.circle(112,771,21,AL,AMBER)
        s.text(112,778,'0',22,AMBER,700,'middle',True)
        s.line(143,771,216,771,AMBER,2.4,True)
        s.circle(247,771,25,AL,AMBER)
        s.text(247,778,'10',22,AMBER,700,'middle',True)
        s.text(195,819,'0 is falsy; the fallback is 10',17,MUTED,anchor='middle')
        s.line(195,840,195,867,INK,2,True)
        s.step(24,904,3,'Make the fallback explicit',21)
        s.terminal(25,929,340,170,'query/limits.py',[
            ('if value is None:',TEAL),('    return default',TEAL),('return value',TEAL)],18)
        s.line(195,1110,195,1140,INK,2,True)
        s.step(24,1177,4,'Check all three behaviors',21)
        table(s,30,1205,330,fontsize=18,row=56)
        s.text(195,1471,'The omitted limit still uses 10.',17,MUTED,anchor='middle')
    else:
        s.text(35,45,'Follow an explicit zero from request to repair',29,weight=700)
        s.text(35,79,'limit = the maximum number of items returned by the API',21,fill=MUTED)
        s.step(35,131,1,'Reproduce the mismatch',24)
        s.terminal(35,152,430,88,'API request',[('GET /items?limit=0',INK)],22)
        s.line(250,245,250,274,INK,2,True)
        s.text(60,305,'Expected',21,TEAL,700)
        s.text(60,352,'[]',40,TEAL,mono=True)
        s.text(245,305,'Current response',21,AMBER,700)
        item_array(s,244,323,1.0)
        s.text(60,400,'0 items',21,TEAL)
        s.text(245,400,'10 items',21,AMBER)
        s.line(479,284,535,284,INK,2.4,True)
        s.step(553,131,2,'Locate the shared conversion',24)
        s.folder(558,166,492,92,'query/',21)
        s.text(575,236,'options.py',20,MUTED,mono=True)
        s.line(740,230,794,230,TEAL,2.2,True)
        s.rect(810,209,215,34,TL,'none',6)
        s.text(824,234,'limits.py',20,TEAL,700,mono=True)
        s.terminal(558,278,492,92,'normalize_limit',[('return value or default',INK)],22)
        s.text(558,407,'0',23,AMBER,700,mono=True)
        s.line(588,400,638,400,AMBER,2.2,True)
        s.text(652,407,'10',23,AMBER,700,mono=True)
        s.text(711,407,'0 is falsy; default is 10',20,MUTED)
        s.path('M805,424 V455 H342 V480',INK,2.3,True)
        s.step(35,520,3,'Make the fallback explicit',24)
        s.terminal(35,545,600,195,'query/limits.py',[
            ('if value is None:',TEAL),('    return default',TEAL),('return value',TEAL)],23)
        s.line(651,637,706,637,INK,2.3,True)
        s.step(725,520,4,'Verify the contract',24)
        table(s,725,545,360,fontsize=21,row=48)
        s.text(35,788,'Use the default only when limit is omitted; preserve explicit values.',21,MUTED)
    s.save('part1-fig1-issue-to-fix'+('-mobile' if mobile else ''))


def table(s,x,y,w,fontsize=20,row=48):
    s.rect(x,y,w,row*4,WHITE,LINE,9)
    s.rect(x,y,w,row,INK,INK,9)
    s.rect(x,y+row-10,w,10,INK,INK,0)
    pos=[x+18,x+w*.48,x+w*.77]
    for xx,txt in zip(pos,['Input','Before','After']): s.text(xx,y+row*.66,txt,fontsize-1,WHITE,700,anchor='start' if txt=='Input' else 'middle')
    for i,(a,b,c) in enumerate([('None',10,10),('0',10,0),('5',5,5)]):
        yy=y+row*(i+1)
        if i==1:s.rect(x+1,yy,w-2,row,TL,'none',0)
        s.line(x,yy,x+w,yy,LINE,1)
        s.text(pos[0],yy+row*.65,a,fontsize,mono=True)
        s.text(pos[1],yy+row*.65,b,fontsize,AMBER if i==1 else MUTED,anchor='middle',mono=True)
        s.text(pos[2],yy+row*.65,c,fontsize,TEAL,700,'middle',True)
        s.check(x+w-17,yy+row*.53,16)


def tree(s,x,y,w,h,label,size=19):
    s.folder(x,y,w,h,label,size)
    for i,txt in enumerate(['src/', 'tests/']):
        yy=y+73+i*30
        s.line(x+18,yy-15,x+18,yy-3,LINE,1.8)
        s.line(x+18,yy-3,x+31,yy-3,LINE,1.8)
        s.text(x+40,yy,txt,size,mono=True)
    # A small sheet and edited source lines are part of the workspace.
    s.file(x+w-75,y+52,50,64,TL,TEAL)
    for i in range(3): s.line(x+w-65,y+78+i*10,x+w-36,y+78+i*10,TEAL,2)


def architecture(mobile=False):
    w,h=(390,1590) if mobile else (1120,885)
    s=SVG(w,h,'Separate the repair workspace from fresh verification',
          'The GPU model server requests tools through the harness and agent loop. Two independent sandboxes start from the same repository snapshot. Only the extracted Git diff is transferred from the agent workspace to fresh verification; evaluator-held tests stay with the verifier.')
    if mobile:
        s.text(24,35,'Two checkouts, one patch',23,weight=700)
        s.text(24,65,'Agent work and verification are isolated.',17,MUTED)
        s.folder(62,100,292,75,'Repository @ base commit',18)
        s.text(62,200,'Two independent checkouts',17,MUTED)
        s.rect(30,235,340,620,TL,'#b9d4d4',14)
        s.text(48,267,'PATCH GENERATION',18,TEAL,700)
        s.chip(112,302,175,108)
        s.line(180,427,180,472,INK,2,True)
        s.line(217,472,217,427,TEAL,2,True)
        s.text(42,444,'tool call',16,INK)
        s.text(237,465,'observation',16,TEAL)
        s.terminal(60,490,280,109,'Harness / agent loop', [('read · edit · test',INK)],17)
        s.line(180,606,180,661,INK,2,True)
        s.line(217,661,217,606,TEAL,2,True)
        s.text(47,643,'tools',16,INK)
        s.text(237,643,'result',16,TEAL)
        tree(s,60,682,280,160,'Agent sandbox',17)
        s.text(74,830,'/workspace',16,MUTED,mono=True)
        s.file(99,879,192,110,WHITE,TEAL)
        s.text(115,910,'Git diff',22,TEAL,700)
        s.text(115,939,'− old line',17,RED,mono=True)
        s.text(115,968,'+ new line',17,TEAL,mono=True)
        s.line(195,860,195,871,TEAL,2.3,True)
        s.text(195,1030,'Only the diff crosses over',17,TEAL,700,'middle')
        s.rect(30,1067,340,495,WHITE,LINE,14)
        s.text(48,1098,'VERIFICATION SANDBOX',18,INK,700)
        tree(s,60,1123,280,132,'Fresh checkout + diff',17)
        s.line(254,1263,254,1321,TEAL,2.2,True)
        s.file(61,1312,111,118,WHITE,INK)
        # Lock shape represents evaluator-only tests, not agent source tests.
        s.path('M94,1352 v-8 a13,13 0 0 1 26,0 v8',INK,2)
        s.rect(91,1352,32,25,INK,INK,4)
        s.text(116,1401,'Held-out',16,INK,anchor='middle')
        s.text(116,1423,'tests',16,INK,anchor='middle')
        s.line(180,1370,214,1370,INK,2,True)
        s.shield(225,1326,77,92)
        s.text(265,1454,'Verifier',20,INK,700,'middle')
        s.text(200,1519,'Resolved / unresolved',18,TEAL,700,'middle')
        s.line(264,1466,264,1490,TEAL,2,True)
        s.path('M62,139 H17 V1199 H57',MUTED,2,True)
        s.path('M17,759 H57',MUTED,2,True)
        s.line(195,994,195,1005,TEAL,2.3)
        s.path('M195,1048 V1052 H378 V1185 H349',TEAL,2.3,True)
    else:
        s.text(35,44,'The agent edits one checkout. The verifier tests a fresh one.',28,weight=700)
        s.text(35,79,'Model requests become tool actions; only the resulting Git diff crosses the isolation boundary.',20,MUTED)
        s.rect(265,139,820,300,TL,'#b9d4d4',16)
        s.rect(265,575,820,268,WHITE,LINE,16)
        s.text(286,174,'PATCH GENERATION',20,TEAL,700)
        s.text(286,609,'VERIFICATION SANDBOX',20,INK,700)
        # One baseline, two independent filesystem copies.
        s.folder(28,309,170,102,'Repository',20)
        s.text(44,383,'@ base commit',17,MUTED,mono=True)
        s.path('M198,344 H235 V207 H947 V242',MUTED,2,True)
        s.text(590,196,'independent checkout',17,MUTED)
        s.path('M198,377 H235 V691 H295',MUTED,2,True)
        s.text(32,452,'Same base',19,INK,700)
        s.text(32,479,'Separate copies',17,MUTED)
        # The model does not act directly on disk.
        s.chip(305,266,158,116)
        s.terminal(583,257,200,116,'Harness / agent loop', [('read / edit',INK),('test',INK)],18)
        s.line(479,292,567,292,INK,2.3,True)
        s.line(567,345,479,345,TEAL,2.3,True)
        s.text(523,280,'tool call',16,INK,anchor='middle')
        s.text(523,371,'observation',16,TEAL,anchor='middle')
        tree(s,862,258,193,151,'/workspace',19)
        s.text(862,431,'Agent sandbox',18,TEAL,700)
        s.line(794,294,846,294,INK,2.3,True)
        s.line(846,345,794,345,TEAL,2.3,True)
        s.text(819,279,'tools',16,INK,anchor='middle')
        s.text(818,374,'results',16,TEAL,anchor='middle')
        # This is a data boundary, not an API available to the agent.
        s.path('M1055,333 H1068 V509 H650',TEAL,2.4,True)
        s.file(490,465,160,98,WHITE,TEAL)
        s.text(507,496,'Git diff',22,TEAL,700)
        s.text(507,525,'− old line',17,RED,mono=True)
        s.text(507,549,'+ new line',17,TEAL,mono=True)
        s.path('M570,563 V662 H533',TEAL,2.4,True)
        s.text(708,492,'Extract patch',19,TEAL)
        s.text(708,552,'Transfer only the diff',19,TEAL,700)
        tree(s,300,640,223,154,'Fresh checkout',20)
        s.text(317,824,'Apply the diff',19,TEAL,700)
        s.line(536,701,611,701,TEAL,2.3,True)
        s.text(574,679,'verify',17,TEAL,anchor='middle')
        s.shield(633,650,95,110)
        s.text(681,794,'Verifier',21,INK,700,'middle')
        s.text(862,827,'Resolved / unresolved',19,TEAL,700,'middle')
        s.path('M681,801 V827 H743',TEAL,2,True)
        s.file(869,636,167,144,WHITE,INK)
        s.path('M931,675 v-13 a17,17 0 0 1 34,0 v13',INK,2)
        s.rect(927,675,42,29,INK,INK,5)
        s.text(952,734,'Held-out tests',20,INK,700,'middle')
        s.text(952,762,'Evaluator only',17,MUTED,anchor='middle')
        s.line(852,701,748,701,INK,2.3,True)
    s.save('part1-fig2-agent-verifier'+('-mobile' if mobile else ''))

if __name__=='__main__':
    issue_figure(); issue_figure(True)
    architecture(); architecture(True)
