"""Check RSNA source parity; use --published when preparing the series for publication."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import hashlib
import json
import re
import sys
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--published', action='store_true', help='Require published: true instead of draft flags.')
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '_draft/_checks/rsna-series-20261022'
OUT.mkdir(parents=True, exist_ok=True)
errors, pairs, inventory, math = [], {}, [], []
paths = sorted((ROOT/'_posts').rglob('*RSNA-Knee*.md'))
expected_dates = {0:'2026-08-09',1:'2026-09-20',2:'2026-10-04',3:'2026-10-15',4:'2026-10-22',5:'2026-10-23',6:'2026-10-25'}

def formulas(body):
    result = []
    def display(m):
        result.append({'tex':m.group(1).strip(), 'display':True})
        return ''
    rest = re.sub(r'^\$\$\s*\n(.*?)\n\$\$\s*$',display,body,flags=re.S|re.M)
    for m in re.finditer(r'(?<!\\)\$([^$\n]+?)(?<!\\)\$',rest):
        result.append({'tex':m.group(1), 'display':False})
    return result

for p in paths:
    raw=p.read_text();_,fm,body=raw.split('---',2);data=yaml.safe_load(fm)
    if data.get('series') != 'rsna-knee':continue
    order,lang=data['series_order'],data['lang']
    date=str(data['date'])[:10]
    if date != expected_dates[order]:errors.append(f'{p.name}: incorrect chronology date {date}')
    if order == 6 and datetime.fromisoformat(str(data['date'])).isoformat() != '2026-10-25T15:00:00+09:00':
        errors.append(f'{p.name}: Part 6 must be published at 15:00 KST')
    if not p.name.startswith(date+'-'):errors.append(f'{p.name}: filename and date differ')
    if body.count('<details markdown="1">') != body.count('</details>'):errors.append(f'{p.name}: unbalanced explanations')
    if re.search(r'<summary>\s*</summary>',body):errors.append(f'{p.name}: empty explanation summary')
    if data.get('published') is not args.published:errors.append(f'{p.name}: published must be {str(args.published).lower()}')
    if not re.search(rf'\bPart {order}:', data['title']):errors.append(f'{p.name}: title and series_order differ')
    if re.search(r'^# ',body,re.M):errors.append(f'{p.name}: duplicate body H1')
    narrative=re.sub(r'^\[\^[^\]]+\]:.*$', '', body, flags=re.M)
    if re.search(r'\b(Claude|Codex|agents?|we|our)\b|우리',narrative,re.I):errors.append(f'{p.name}: non-author narrative')
    if re.search(r'가능성이 닫|가망이 닫',body):errors.append(f'{p.name}: unnatural Korean')
    images=[url for url in re.findall(r'/assets/img/[^)\s"<>]+',body)
            if Path(url).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.svg','.avif'}]
    images.append(data['image']['path'])
    for image in images:
        if not (ROOT/image.lstrip('/')).is_file():errors.append(f'{p.name}: missing image {image}')
    exprs=formulas(body)
    for i,expr in enumerate(exprs):math.append(dict(post=p.name,index=i,**expr))
    slug=re.sub(r'^\d{4}-\d{2}-\d{2}-','',p.stem)
    entry={'path':str(p.relative_to(ROOT)), 'url':f'/posts/{slug}/', 'order':order,'lang':lang,'date':str(data['date']),
           'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),'images':images,
           'explanations':body.count('<details markdown="1">'),
           'footnotes':len(re.findall(r'^\[\^[^\]]+\]:',body,re.M)),
           'display_math':sum(x['display'] for x in exprs),'inline_math':sum(not x['display'] for x in exprs)}
    inventory.append(entry);pairs.setdefault(order,{})[lang]=(data,body,exprs,images)

numeric_differences={}
for n in expected_dates:
    pair=pairs.get(n,{})
    if set(pair) != {'en','ko'}:
        errors.append(f'Part{n}: missing language');continue
    a,b=pair['en'],pair['ko']
    for key in ['date','tags','series_order','translation_key','published']:
        if a[0].get(key)!=b[0].get(key):errors.append(f'Part{n}: metadata mismatch {key}')
    if a[3]!=b[3]:errors.append(f'Part{n}: image path mismatch')
    if a[1].count('<details markdown="1">') != b[1].count('<details markdown="1">'):errors.append(f'Part{n}: explanation count mismatch')
    if Counter((x['tex'],x['display']) for x in a[2])!=Counter((x['tex'],x['display']) for x in b[2]):
        errors.append(f'Part{n}: mathematical expression mismatch')
    numbers=lambda s:Counter(re.findall(r'(?<![A-Za-z0-9.])\d+\.\d+(?![A-Za-z0-9]|\.\d)',s))
    ca,cb=numbers(a[1]),numbers(b[1])
    if ca!=cb:numeric_differences[n]={'en_only':dict(ca-cb),'ko_only':dict(cb-ca)}

if len(inventory)!=2*len(expected_dates):errors.append(f'Expected {2*len(expected_dates)} posts, got {len(inventory)}')
report={'status':'passed' if not errors else 'failed','posts':inventory,'expected_published':args.published,
        'errors':errors,'decimal_differences_for_editor_review':numeric_differences,
        'limits':'Source/structure checks; not a browser or clinical validation.'}
(OUT/'source-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
(OUT/'math-inputs.json').write_text(json.dumps(math,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'posts':len(inventory),'formulas':len(math),
                  'errors':errors,'decimal_differences':numeric_differences},ensure_ascii=False,indent=2))
sys.exit(bool(errors))
