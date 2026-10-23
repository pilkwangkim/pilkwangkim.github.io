"""Generate the English RL tutorial figures; no games or model training.

Run with a Python interpreter providing Pillow. Each figure has an editable
SVG and a matching 2x PNG, in desktop and mobile layouts. Geometry is shared
by both renderers. Text bounds and text collisions are checked automatically;
arrow routing and visual meaning are also reviewed on the rendered images.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/img/posts/kaggriculture-rl-series"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD_FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
BG, INK, MUTED, GRID = "#fcfcfa", "#182d3b", "#526575", "#d8e0e5"
COLORS = {
    "policy": ("#edf4fc", "#3269a4"),
    "teacher": ("#f4eff9", "#77509e"),
    "learn": ("#ecf6f2", "#22765f"),
    "eval": ("#fbf3e7", "#a56822"),
    "data": ("#f0f3f5", "#526575"),
}
SCALE = 2
RECEIPTS = []


def font(size, bold=False):
    return ImageFont.truetype(BOLD_FONT if bold else FONT, round(size))


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    def port(self, side):
        return {
            "top": (self.x + self.w / 2, self.y - 5),
            "bottom": (self.x + self.w / 2, self.y + self.h + 5),
            "left": (self.x - 5, self.y + self.h / 2),
            "right": (self.x + self.w + 5, self.y + self.h / 2),
        }[side]


class Canvas:
    def __init__(self, name, title, mobile=False):
        self.name = name + ("-mobile" if mobile else "")
        self.mobile = mobile
        self.w = 480 if mobile else 900
        self.parts, self.ops, self.texts, self.nodes, self.arrows = [], [], [], {}, []
        self.bottom = 0
        self.text(26 if mobile else 32, 32, title, 23 if mobile else 28, bold=True, max_width=self.w-52)

    def text(self, x, y, text, size=21, color=INK, anchor="left", bold=False,
             max_width=None, node=None):
        text = text.replace("−", "-")
        assert not any("가" <= c <= "힣" for c in text), (self.name, "non-English label", text)
        if max_width:
            while font(size, bold).getlength(text) > max_width and size > 18:
                size -= 1
            assert font(size, bold).getlength(text) <= max_width + 0.5, (self.name, text)
        width = font(size, bold).getlength(text)
        height = size * 1.1
        left = x - width / 2 if anchor == "center" else x
        assert left >= 2 and left + width <= self.w - 2, (self.name, text, left, width)
        self.bottom = max(self.bottom, y + height / 2)
        self.parts.append(
            f'<text x="{x}" y="{y}" dominant-baseline="central" '
            f'text-anchor="{"middle" if anchor == "center" else "start"}" '
            f'font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" '
            f'fill="{color}">{escape(text)}</text>'
        )
        self.ops.append(("text", x, y, text, size, color, anchor, bold))
        self.texts.append((left, y - height / 2, left + width, y + height / 2, text, node))

    def rect(self, x, y, w, h, fill=BG, stroke=None, radius=12):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="{radius}" fill="{fill}" stroke="{stroke or "none"}" '
                          f'stroke-width="1.4"/>')
        self.ops.append(("rect", x, y, w, h, fill, stroke, radius))
        self.bottom = max(self.bottom, y + h)

    def node(self, name, x, y, w, h, title, lines=(), tone="data"):
        self.nodes[name] = Box(x, y, w, h)
        fill, stroke = COLORS[tone]
        self.rect(x, y, w, h, fill, GRID)
        n = len(lines)
        if n:
            title_y = y + 27
            start = title_y + 33
            assert start + (n - 1) * 27 + 15 <= y + h, (self.name, title)
            self.text(x + 18, title_y, title, 22, stroke, bold=True,
                      max_width=w - 36, node=name)
            for i, line in enumerate(lines):
                self.text(x + 18, start + i * 27, line, 20,
                          max_width=w - 36, node=name)
        else:
            self.text(x + w / 2, y + h / 2, title, 22, stroke,
                      anchor="center", bold=True, max_width=w - 32, node=name)
        return self.nodes[name]

    def line(self, points, color=MUTED, width=2, dashed=False):
        dash_attribute = 'stroke-dasharray="6 5"' if dashed else ""
        self.parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" '
                          f'fill="none" stroke="{color}" stroke-width="{width}" '
                          f'{dash_attribute}/>' )
        self.ops.append(("line", points, color, width, dashed))
        self.bottom = max(self.bottom, max(p[1] for p in points))

    def arrow(self, source, target, source_side="right", target_side="left", via=()):
        start = self.nodes[source].port(source_side)
        end = self.nodes[target].port(target_side)
        points = [start, *via, end]
        self.line(points, width=2.2)
        dx, dy = end[0] - points[-2][0], end[1] - points[-2][1]
        norm = math.hypot(dx, dy)
        assert norm > 12
        ux, uy = dx / norm, dy / norm
        arrow = [end, (end[0] - 11 * ux - 5 * uy, end[1] - 11 * uy + 5 * ux),
                 (end[0] - 11 * ux + 5 * uy, end[1] - 11 * uy - 5 * ux)]
        self.parts.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in arrow)}" fill="{MUTED}"/>')
        self.ops.append(("polygon", arrow, MUTED))
        self.arrows.append((source, target, points))

    def save(self):
        # Check node bounds, text containment and unrelated text intersections.
        for name, box in self.nodes.items():
            assert 0 <= box.x < box.x + box.w <= self.w
            for other, second in self.nodes.items():
                if name >= other:
                    continue
                overlap = (min(box.x + box.w, second.x + second.w) - max(box.x, second.x),
                           min(box.y + box.h, second.y + second.h) - max(box.y, second.y))
                assert min(overlap) <= 0, (self.name, "node collision", name, other)
        for x1, y1, x2, y2, text, owner in self.texts:
            if owner:
                b = self.nodes[owner]
                assert b.x + 8 <= x1 and x2 <= b.x + b.w - 8
                assert b.y + 4 <= y1 and y2 <= b.y + b.h - 4
        for i, a in enumerate(self.texts):
            for b in self.texts[i + 1:]:
                assert min(a[2], b[2]) - max(a[0], b[0]) < 1 or min(a[3], b[3]) - max(a[1], b[1]) < 1, (self.name, "text collision", a[4], b[4])
        # Sample shaft geometry so an arrow cannot pass through a third card.
        for source, target, points in self.arrows:
            for a, b in zip(points, points[1:]):
                for k in range(101):
                    x, y = a[0] + (b[0] - a[0]) * k / 100, a[1] + (b[1] - a[1]) * k / 100
                    for name, box in self.nodes.items():
                        if name in (source, target):
                            continue
                        assert not (box.x < x < box.x + box.w and box.y < y < box.y + box.h), (self.name, "arrow crosses node", source, target, name)
        height = math.ceil(self.bottom + 25)
        OUT.mkdir(parents=True, exist_ok=True)
        header = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{height}" '
                  f'viewBox="0 0 {self.w} {height}" role="img" aria-labelledby="title">'
                  f'<title id="title">{escape(self.name)}</title>'
                  f'<rect width="{self.w}" height="{height}" fill="{BG}"/>')
        (OUT / f"{self.name}.svg").write_text(header + "\n" + "\n".join(self.parts) + "\n</svg>\n")
        image = Image.new("RGB", (self.w * SCALE, height * SCALE), BG)
        draw = ImageDraw.Draw(image)
        for op, *v in self.ops:
            if op == "text":
                x, y, text, size, color, anchor, bold = v
                draw.text((x * SCALE, y * SCALE), text, font=font(size * SCALE, bold),
                          fill=color, anchor="mm" if anchor == "center" else "lm")
            elif op == "rect":
                x, y, w, h, fill, stroke, radius = v
                draw.rounded_rectangle((round(x*SCALE), round(y*SCALE), round((x+w)*SCALE), round((y+h)*SCALE)),
                                       radius=radius*SCALE, fill=fill, outline=stroke, width=round(1.4*SCALE))
            elif op == "line":
                points, color, width, dashed = v
                coords = [(round(x*SCALE), round(y*SCALE)) for x, y in points]
                # Dashed graph references are rendered as short segments.
                if dashed:
                    for a, b in zip(coords, coords[1:]):
                        length = math.dist(a,b)
                        for j in range(0, math.ceil(length), 11*SCALE):
                            p, q = j/length, min(j+6*SCALE,length)/length
                            draw.line([(a[0]+(b[0]-a[0])*p,a[1]+(b[1]-a[1])*p),
                                       (a[0]+(b[0]-a[0])*q,a[1]+(b[1]-a[1])*q)], fill=color, width=round(width*SCALE))
                else:
                    draw.line(coords, fill=color, width=round(width*SCALE), joint="curve")
            elif op == "circle":
                x, y, radius, fill, stroke, width = v
                draw.ellipse(((x-radius)*SCALE,(y-radius)*SCALE,(x+radius)*SCALE,(y+radius)*SCALE),fill=fill,outline=stroke,width=max(1,round(width*SCALE)))
            elif op == "polygon":
                points, color = v
                draw.polygon([(round(x*SCALE), round(y*SCALE)) for x,y in points], fill=color)
        image.save(OUT / f"{self.name}.png", optimize=True)
        RECEIPTS.append({"name":self.name,"width":self.w,"height":height,
                         "text_items":len(self.texts),"nodes":len(self.nodes),"arrows":len(self.arrows),
                         "text_bounds_and_collisions":"passed",
                         "arrow_routes":"manual visual review required"})


def circle(d,x,y,r,fill=BG,stroke=None,width=1.5):
    d.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>')
    d.ops.append(('circle',x,y,r,fill,stroke,width));d.bottom=max(d.bottom,y+r)


def poly(d,points,fill):
    d.parts.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in points)}" fill="{fill}"/>')
    d.ops.append(('polygon',points,fill));d.bottom=max(d.bottom,max(y for x,y in points))


def arrow(d,points,color=MUTED,width=2.2):
    d.line(points,color,width)
    a,b=points[-2:];dx,dy=b[0]-a[0],b[1]-a[1];n=math.hypot(dx,dy)
    assert n>12,(d.name,'short arrow')
    ux,uy=dx/n,dy/n
    poly(d,[b,(b[0]-10*ux-4*uy,b[1]-10*uy+4*ux),(b[0]-10*ux+4*uy,b[1]-10*uy-4*ux)],color)
    d.arrows.append(('manual','manual',points))


def worker(d,x,y,s=1,color='#356fa6'):
    circle(d,x,y-8*s,5*s,'#e1b48b')
    d.rect(x-8*s,y-13*s,16*s,3*s,'#d8aa45',radius=0)
    d.line([(x,y-2*s),(x,y+10*s)],color,5*s)
    d.line([(x-7*s,y+2*s),(x,y+1*s),(x+7*s,y+5*s)],color,2.5*s)
    d.line([(x-6*s,y+18*s),(x,y+9*s),(x+6*s,y+18*s)],color,2.5*s)


def plant(d,x,y,size=1,kind='growing'):
    if kind=='tomato':
        d.line([(x,y+8*size),(x,y-6*size)],'#3b7954',2*size)
        circle(d,x-4*size,y,5*size,'#d65946');circle(d,x+4*size,y+3*size,5*size,'#e26d50')
        poly(d,[(x-7*size,y-5*size),(x,y-11*size),(x+7*size,y-5*size),(x,y-6*size)],'#3f8157')
    elif kind=='strawberry':
        poly(d,[(x-8*size,y-5*size),(x+8*size,y-5*size),(x,y+10*size)],'#cf5272')
        poly(d,[(x-9*size,y-6*size),(x-1*size,y-12*size),(x+9*size,y-6*size),(x,y-7*size)],'#3f8157')
        for xx,yy in [(-3,0),(3,0),(0,5)]:circle(d,x+xx*size,y+yy*size,0.9*size,'#f6d99c')
    elif kind=='wheat':
        d.line([(x,y+12*size),(x,y-11*size)],'#ac8430',2*size)
        for yy in [-6,0,6]:
            d.line([(x,y+yy*size),(x-6*size,y+(yy-5)*size)],'#ceaa50',3*size)
            d.line([(x,y+yy*size),(x+6*size,y+(yy-5)*size)],'#ceaa50',3*size)
    else:
        d.line([(x,y+10*size),(x,y-3*size)],'#3d865d',2*size)
        poly(d,[(x,y+1*size),(x-10*size,y-6*size),(x-5*size,y-10*size),(x,y-4*size)],'#67a66a')
        poly(d,[(x,y),(x+10*size,y-8*size),(x+5*size,y-12*size),(x,y-4*size)],'#498d5e')


def farm(d,x,y,cell=38,cols=4,rows=3,crops=None,extension=False,path=False):
    crops=crops or {}
    for row in range(rows):
        for col in range(cols):
            xx,yy=x+col*cell,y+row*cell
            d.rect(xx,yy,cell-2,cell-2,'#e8d9bd' if row else '#e4eadc',radius=2)
            if row:
                for off in [cell*.28,cell*.55,cell*.8]:d.line([(xx+5,yy+off),(xx+cell-7,yy+off)],'#cfbb95',1)
            if (col,row) in crops:plant(d,xx+cell/2-1,yy+cell/2-1,cell/38*.8,crops[col,row])
    # A storehouse and a walking path make worker placement explicit.
    xx=x+cell*.15;yy=y+cell*.12
    d.rect(xx,yy+cell*.22,cell*.64,cell*.56,'#cad9e0','#718b9b',radius=1)
    poly(d,[(xx-3,yy+cell*.25),(xx+cell*.32,yy-2),(xx+cell*.67,yy+cell*.25)],'#526d82')
    d.rect(xx+cell*.24,yy+cell*.43,cell*.18,cell*.35,'#5c788c',radius=0)
    if extension:
        d.line([(x+2*cell-1,y+cell-2),(x+cols*cell-1,y+cell-2),(x+cols*cell-1,y+rows*cell-1),(x+2*cell-1,y+rows*cell-1),(x+2*cell-1,y+cell-2)],'#b98428',2.5,dashed=True)
    if path:
        d.line([(x+cell*.5,y+cell*.5),(x+cell*1.5,y+cell*.5),(x+cell*1.5,y+cell*1.5),(x+cell*3.5,y+cell*1.5),(x+cell*3.5,y+cell*2.5)],'#3b71aa',2.5,dashed=True)
    worker(d,x+cell*1.5,y+cell*.4,cell/46)


def net(d,x,y,w=150,h=104,color='#356da6'):
    columns=[3,4,2]
    pos=[]
    for k,n in enumerate(columns):pos.append([(x+w*k/2,y+h*(j+1)/(n+1)) for j in range(n)])
    for a,b in zip(pos,pos[1:]):
        for aa in a:
            for bb in b:d.line([aa,bb],'#c9d8e6',1.2)
    for col in pos:
        for xx,yy in col:circle(d,xx,yy,6,BG,color,2)


def lock(d,x,y,s=1):
    d.line([(x-6*s,y),(x-6*s,y-7*s),(x,y-11*s),(x+6*s,y-7*s),(x+6*s,y)],'#647b8e',2*s)
    d.rect(x-9*s,y,18*s,14*s,'#dce6ec','#647b8e',radius=2)
    circle(d,x,y+6*s,1.5*s,'#647b8e')


def replay_stack(d,x,y,w=124,h=92):
    for dx,dy in [(10,-10),(5,-5),(0,0)]:d.rect(x+dx,y+dy,w,h,'#f3f6fa','#8ca0b0',radius=3)
    for row in range(4):
        yy=y+18+row*17
        circle(d,x+14,yy,3,'#3c75ad')
        for j in range(5):d.rect(x+27+j*15,yy-3,10,6,['#b9cbdc','#7da3c8','#d9e4ee'][(row+j)%3],radius=0)


def environment(mobile):
    d=Canvas('fig-01-agent-environment','Observation, action, next state',mobile)
    if mobile:
        farm(d,42,108,42,3,2)
        d.text(240,114,'Observed now',22,bold=True)
        d.text(240,150,'Cash: 30',21);d.text(240,184,'Crop: empty',21)
        d.text(42,230,'Input: (30, "empty")',20,color=MUTED)
        arrow(d,[(230,260),(230,299)])
        circle(d,230,350,46,'#e9f2fb','#4b7ca9',2)
        d.text(230,347,'{  }',31,'#356da6',anchor='center',bold=True)
        d.text(230,414,'Policy program',22,anchor='center',bold=True)
        arrow(d,[(230,440),(230,487)])
        d.text(230,518,'PREPARE',25,'#356da6',anchor='center',bold=True)
        d.text(230,553,'Action: spend 10, start growing',19,anchor='center')
        arrow(d,[(230,578),(230,615)])
        farm(d,42,655,42,3,2,{(0,1):'growing'})
        d.text(240,662,'Next observation',21,bold=True)
        d.text(240,699,'Cash: 20',21);d.text(240,734,'Crop: growing',20)
        d.text(230,788,'Reward = -10',23,'#9d6830',anchor='center',bold=True)
    else:
        farm(d,42,124,44,3,2)
        d.text(42,86,'Observation',23,bold=True)
        d.text(42,245,'Cash 30  |  Empty plot',20)
        arrow(d,[(204,172),(302,172)])
        circle(d,371,172,48,'#e9f2fb','#4b7ca9',2)
        d.text(371,169,'{  }',34,'#356da6',anchor='center',bold=True)
        d.text(371,251,'Policy program',22,anchor='center',bold=True)
        arrow(d,[(432,172),(519,172)])
        d.text(566,116,'PREPARE',24,'#356da6',anchor='center',bold=True)
        d.text(566,151,'spend 10',19,anchor='center')
        arrow(d,[(607,172),(647,172)])
        farm(d,692,124,44,3,2,{(0,1):'growing'})
        d.text(692,86,'Next observation',23,bold=True)
        d.text(692,245,'Cash 20',20)
        d.text(692,275,'Crop growing',20)
        d.text(435,332,'Reward = -10',23,'#9d6830',anchor='center',bold=True)
    d.save()


def bc(mobile):
    d=Canvas('fig-04-bc-update','One BC update shifts action probabilities',mobile)
    p=[.1,.2,.7];z=[math.log(x)-.5*(x-(i==0)) for i,x in enumerate(p)]
    e=[math.exp(x) for x in z];after=[x/sum(e) for x in e]
    d.text(30,84,'Teacher action: SELL',22,'#77509e',bold=True)
    plant(d,290,84,1.0,'wheat')
    x0=145 if mobile else 165;length=270 if mobile else 320;y0=168
    d.text(30,128,'Action probability',21,bold=True)
    for tick in [0,.25,.5,.75]:
        xx=x0+tick/.8*length;d.line([(xx,y0-12),(xx,y0+170)],GRID,1)
        d.text(xx,y0+200,f'{tick:g}',17,anchor='center',color=MUTED)
    for i,name in enumerate(['SELL','BUY','WAIT']):
        y=y0+i*59;d.text(30,y+13,name,20,bold=i==0)
        d.rect(x0,y-5,p[i]/.8*length,13,'#b8cadd',radius=0)
        d.rect(x0,y+13,after[i]/.8*length,13,'#28765f',radius=0)
        d.text(x0+after[i]/.8*length+8,y+20,f'{after[i]:.3f}',17)
    d.rect(35,y0+230,16,10,'#b8cadd',radius=0);d.text(60,y0+235,'Before',18)
    d.rect(175,y0+230,16,10,'#28765f',radius=0);d.text(200,y0+235,'After one update',18)
    gx=235 if mobile else 655;gy=505 if mobile else 168;gw=150 if mobile else 145
    if mobile:d.text(30,458,'Logit change: -0.5 x (p - y)',21,bold=True)
    else:d.text(560,128,'Logit change',21,bold=True)
    d.line([(gx,gy-12),(gx,gy+170)],MUTED,1.5)
    for i,(name,change) in enumerate(zip(['SELL','BUY','WAIT'],[.45,-.1,-.35])):
        yy=gy+i*59;end=gx+change/.5*gw
        d.rect(min(gx,end),yy,abs(end-gx),20,'#28765f' if change>0 else '#cc8c57',radius=0)
        if mobile:d.text(30,yy+10,name,19)
        d.text(max(gx,end)+8,yy+10,f'{change:+.2f}',18)
    if not mobile:
        d.text(560,385,'Step size: 0.5',19,color=MUTED)
        d.text(32,458,'Cross-entropy loss: 2.303  ->  1.668',22,bold=True)
    else:d.text(30,727,'Loss: 2.303  ->  1.668',23,bold=True)
    d.save()


def rollout(mobile):
    d=Canvas('fig-05-ppo-data-contract','PPO: collect, freeze, update',mobile)
    if mobile:
        net(d,45,102,130,93);farm(d,287,104,31,3,2)
        d.text(30,222,'Old policy',21,bold=True);d.text(280,222,'New episodes',20,bold=True)
        arrow(d,[(193,148),(267,148)],'#356da6')
        arrow(d,[(332,248),(332,277),(57,277),(57,380),(70,380)],'#356da6')
        replay_stack(d,80,336,292,110);lock(d,396,351)
        d.text(226,313,'Rollout buffer',22,anchor='center',bold=True)
        d.text(226,432,'obs / action / mask',18,anchor='center')
        d.text(226,479,'Old log p, value, reward: fixed',20,anchor='center')
        # The stored values remain outside the gradient path.
        arrow(d,[(180,495),(180,533)],'#356da6')
        net(d,107,553,150,100,'#28765f');d.text(180,688,'Current policy',21,anchor='center',bold=True)
        arrow(d,[(288,605),(407,605),(407,745)],'#356da6')
        circle(d,230,790,43,'#fff2df','#b47832',2);d.text(230,789,'Loss',23,anchor='center',bold=True)
        d.text(230,859,'Clipped ratio-weighted advantage',19,anchor='center')
        arrow(d,[(184,790),(45,790),(45,605),(88,605)],'#bb7540')
        d.text(30,923,'Orange: gradient updates current weights',19,color='#9e6033')
        arrow(d,[(378,418),(443,418),(443,790),(289,790)],'#748a9a',1.6)
    else:
        net(d,44,110,137,100);d.text(115,246,'Old policy',22,anchor='center',bold=True)
        arrow(d,[(205,160),(302,160)],'#356da6')
        farm(d,335,107,38,3,2);d.text(392,246,'New episodes',22,anchor='center',bold=True)
        arrow(d,[(479,160),(592,160)],'#356da6')
        replay_stack(d,631,111,195,100);lock(d,849,130)
        d.text(728,246,'Rollout buffer',22,anchor='center',bold=True)
        d.text(728,204,'obs / action / mask',17,anchor='center')
        d.text(728,285,'Old log p, value, reward',18,anchor='center')
        arrow(d,[(710,315),(710,356),(481,356),(481,389)],'#356da6')
        net(d,411,409,142,96,'#28765f');d.text(481,538,'Current policy',22,anchor='center',bold=True)
        arrow(d,[(389,457),(296,457)],'#356da6')
        circle(d,224,457,45,'#fff2df','#b47832',2);d.text(224,456,'Loss',23,anchor='center',bold=True)
        arrow(d,[(728,315),(855,315),(855,585),(224,585),(224,515)],'#748a9a',1.6)
        d.text(224,373,'Clipped ratio-weighted',18,anchor='center');d.text(224,400,'advantage',18,anchor='center')
        arrow(d,[(268,483),(327,536),(389,483)],'#bb7540')
        d.text(32,639,'Blue: forward computation     Orange: gradient     Lock: stored reference values',19,color=MUTED)
    d.save()


def clipping(mobile):
    d=Canvas('fig-06-ppo-clip','What PPO clipping changes',mobile)
    origins=[(72,138),(512,138)] if not mobile else [(75,134),(75,484)]
    pw=320 if not mobile else 335;ph=190
    for index,(x,y) in enumerate(origins):
        positive=index==0;amin,amax=(.4,1.6) if positive else (-1.6,-.4)
        X=lambda r:x+(r-.5)*pw;Y=lambda v:y+ph-(v-amin)/(amax-amin)*ph
        d.text(x,y-48,'Positive advantage: A = +1' if positive else 'Negative advantage: A = -1',21,bold=True)
        d.text(x,y-20,'Objective value',17,color=MUTED)
        d.rect(X(.8),y,X(1.2)-X(.8),ph,'#eef3f7',radius=0)
        for r in [.5,.8,1,1.2,1.5]:
            d.line([(X(r),y),(X(r),y+ph)],GRID,1);d.text(X(r),y+ph+23,str(r),17,anchor='center')
        d.line([(x,y),(x,y+ph),(x+pw,y+ph)],MUTED,1.5)
        for value in ([.5,1,1.5] if positive else [-1.5,-1,-.5]):d.text(x-27,Y(value),f'{value:g}',17,anchor='center')
        raw=[];clipped=[]
        for k in range(101):
            r=.5+k/100;A=1 if positive else -1
            raw.append((X(r),Y(r*A)));clipped.append((X(r),Y(min(r*A,min(1.2,max(.8,r))*A))))
        d.line(raw,'#8b9ca8',2,dashed=True);d.line(clipped,'#b47726',3)
        d.text(x+pw/2,y+ph+61,'Probability ratio: new / old',18,anchor='center')
    by=origins[-1][1]+ph+116
    d.line([(35,by),(74,by)],'#b47726',3);d.text(87,by,'Clipped objective',19)
    d.line([(35,by+35),(74,by+35)],'#8b9ca8',2,dashed=True);d.text(87,by+35,'Unclipped objective',19)
    d.text(30,by+80,'Clipping limits the incentive, not the probability itself.',18,color=MUTED)
    d.save()


def investment(mobile):
    d=Canvas('fig-07-plan-and-validation','Investment is a bundle',mobile)
    c=38 if mobile else 48
    xs=[38,266] if mobile else [75,520]
    # The new plot contains six productive tiles, not isolated parameter edits.
    for i,(x,label,n) in enumerate(zip(xs,['A: keep existing farm','B: hire + land + crops'],[4,6])):
        if mobile:d.text(x,88,'Plan A' if i==0 else 'Plan B',22,bold=True)
        else:d.text(x,88,label,23,bold=True)
        crops={(0,1):'wheat',(1,1):'wheat'}
        if i:crops.update({(2,1):'tomato',(3,1):'tomato',(4,1):'tomato',(2,2):'strawberry',(3,2):'strawberry',(4,2):'strawberry'})
        farm(d,x,119,c,5 if i else 2,3,crops,extension=bool(i),path=bool(i))
        yy=119+3*c+28
        for j in range(n):worker(d,x+12+j*(24 if mobile else 31),yy,.7 if mobile else .9)
        d.text(x,yy+44,f'Total workforce: {n}',18,bold=True)
    y=355 if mobile else 390
    d.text(30,y,'Added cost of B',22,bold=True)
    bx=30;by=y+41;bw=420 if mobile else 810
    for value,color in [(1000,'#c59848'),(450,'#c65d63'),(8,'#356fa6')]:
        w=bw*value/1458;d.rect(bx,by,w,31,color,radius=0);bx+=w
    d.text(30,by+65,'Land 1,000  +  seeds 450  +  wages 8',19)
    d.text(30,by+104,'Total: 1,458',23,bold=True)
    y2=by+167
    d.text(30,y2,'Toy net value after cost (coins)',22,bold=True)
    for i,(name,value) in enumerate(zip(['A','B','C','D'],[0,525.252972,146.056,461.333])):
        yy=y2+48+i*42;d.text(30,yy,name,21,bold=name=='B')
        d.rect(80,yy-11,max(2,(value/550)*(340 if mobile else 640)),22,'#28765f' if name=='B' else '#b8c9d7',radius=0)
        end=80+(value/550)*(340 if mobile else 640)
        d.text(min(end+9,d.w-63),yy,f'{value:.0f}',18,bold=name=='B')
    d.save()


def alignment(mobile):
    d=Canvas('fig-08-replay-alignment','Replay rows are shifted',mobile)
    if mobile:
        for i,(cash,phase,action) in enumerate([(30,'empty','none'),(20,'growing','PREPARE'),(20,'ripe','WATER')]):
            y=112+i*181
            d.text(30,y-28,f'Record {i}',22,bold=True)
            farm(d,30,y,31,3,2,{(0,1):'wheat' if i==2 else 'growing'} if i else {})
            d.text(155,y+10,f'Cash {cash}, crop {phase}',19)
            d.text(155,y+48,f'Stored action: {action}',18,'#77509e')
        arrow(d,[(80,194),(80,239),(132,239),(132,341),(146,341)],'#356da6')
        arrow(d,[(80,375),(80,420),(132,420),(132,522),(146,522)],'#356da6')
        d.text(30,686,'Input from row 0  ->  action from row 1',19,bold=True)
        d.text(30,722,'Input from row 1  ->  action from row 2',19,bold=True)
    else:
        for i,(cash,phase,action) in enumerate([(30,'empty','none'),(20,'growing','PREPARE'),(20,'ripe','WATER')]):
            x=35+i*295;d.text(x,88,f'Record {i}',22,bold=True)
            farm(d,x,122,39,3,2,{(0,1):'wheat' if i==2 else 'growing'} if i else {})
            d.text(x,235,f'Cash {cash}, crop {phase}',19)
            d.text(x,385,f'Action: {action}',21,'#77509e',bold=True)
        arrow(d,[(166,250),(166,330),(447,330),(447,357)],'#356da6')
        arrow(d,[(461,250),(461,350),(742,350),(742,367)],'#356da6')
        d.text(35,445,'The action stored in the next row was chosen from the previous observation.',19,color=MUTED)
    d.save()


def paired(mobile):
    from random import Random
    d=Canvas('fig-09-paired-world','Pair by world, then resample',mobile)
    x0=65;pw=365 if mobile else 760;y0=132;ph=166
    d.text(30,86,'Paired score change by opponent',21,bold=True)
    X=lambda i:x0+i/9*pw;Y=lambda v:y0+ph/2-v/.3*(ph/2)
    d.rect(X(0)-14,y0,28,ph,'#e8f0f8',radius=0)
    for v in [-.25,0,.25]:
        d.line([(x0-8,Y(v)),(x0+pw+8,Y(v))],GRID,1);d.text(x0-32,Y(v),f'{v:+.2f}',16,anchor='center')
    for i in range(10):
        e=.25 if i<4 else 0;h=.25 if i<2 else (-.25 if i<4 else 0)
        xx=X(i);circle(d,xx-5,Y(e),5,'#356da6')
        poly(d,[(xx+5,Y(h)-6),(xx-1,Y(h)+4),(xx+11,Y(h)+4)],'#b27934')
        d.text(xx,y0+ph+24,str(i+1),16,anchor='center')
    d.text(x0+pw/2,y0+ph+58,'World ID',18,anchor='center')
    circle(d,40,y0+ph+98,5,'#356da6');d.text(55,y0+ph+98,'E: weight 0.7',18)
    poly(d,[(245,y0+ph+92),(239,y0+ph+102),(251,y0+ph+102)],'#b27934');d.text(263,y0+ph+98,'H: weight 0.3',18)
    yy=y0+ph+162;d.text(30,yy,'Weighted change for each world',21,bold=True)
    by=yy+43;bh=140
    for i,v in enumerate([.25,.25,.1,.1,0,0,0,0,0,0]):
        xx=X(i);h=v/.25*bh;d.rect(xx-10,by+bh-h,20,max(2,h),'#70a18c',radius=0)
    mean_y=by+bh-.07/.25*bh;d.line([(x0-10,mean_y),(x0+pw+8,mean_y)],'#b47832',1.7,dashed=True)
    d.text(30,by+bh+40,'Mean change: +0.07',22,bold=True)
    d.text(30,by+bh+79,'One world = 8 games kept together.',19,color=MUTED)
    yy2=by+bh+132;d.text(30,yy2,'Bootstrap means: 1,000 resamples',21,bold=True)
    rng=Random(2026);values=[.25,.25,.1,.1,0,0,0,0,0,0]
    samples=[sum(rng.choices(values,k=10))/10 for _ in range(1000)]
    bins=[0]*20
    for v in samples:bins[min(19,int(v/.2*20))]+=1
    hy=yy2+34;hh=111
    for i,n in enumerate(bins):
        width=pw/20;d.rect(x0+i*width,hy+hh-n/max(bins)*hh,width-1,n/max(bins)*hh,'#9bb8d0',radius=0)
    for v in [0,.05,.1,.15,.2]:d.text(x0+v/.2*pw,hy+hh+27,f'{v:.2f}',17,anchor='center')
    d.text(30,hy+hh+69,'95% percentile interval: [0.010, 0.135]',19,bold=True)
    d.save()


def timing(mobile):
    d=Canvas('fig-10-end-to-end-time','Measure the whole training loop',mobile)
    x=40 if mobile else 180;pw=380 if mobile else 620;y0=160
    for i,(label,values) in enumerate([('Baseline',[120,30,10]),('2x faster update',[120,15,10])]):
        y=y0+i*126;d.text(30,y-29,label,20,bold=True)
        start=x
        for j,(v,c) in enumerate(zip(values,['#356da6','#28765f','#94a5b1'])):
            w=pw*v/160;d.rect(start,y,w,43,c,radius=0)
            if j==0:d.text(start+w/2,y+21,'120 s',20,'#ffffff',anchor='center',bold=True)
            elif i==1 and j==1:
                d.line([(start+w/2,y-3),(start+w/2,y-13)],c,1.5)
                d.text(start+w/2,y-28,f'{v} s',17,anchor='center')
            else:
                d.line([(start+w/2,y+45),(start+w/2,y+58)],c,1.5)
                d.text(start+w/2,y+78,f'{v} s',17,anchor='center')
            start+=w
        if not mobile:d.text(829,y+21,f'{sum(values)} s',20,bold=True)
    by=y0+126+115
    for i,(name,color) in enumerate([('Rollout','#356da6'),('Update','#28765f'),('Save','#94a5b1')]):
        xx=35+i*(145 if mobile else 255);d.rect(xx,by-7,15,15,color,radius=0);d.text(xx+24,by,name,18)
    d.text(30,by+49,'160 s  ->  145 s total',23,bold=True)
    d.text(30,by+85,'Sequential execution; compilation and evaluation excluded.',16 if mobile else 19,color=MUTED)
    d.save()


def main():
    for mobile in [False,True]:
        for make in [environment,bc,rollout,clipping,investment,alignment,paired,timing]:make(mobile)
    (OUT/'figures.json').write_text(json.dumps(RECEIPTS,indent=2)+'\n')
    print(f'Generated {len(RECEIPTS)} English figure layouts.')


if __name__=='__main__':main()
