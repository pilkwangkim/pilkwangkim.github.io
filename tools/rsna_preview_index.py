"""Add an RSNA series landing page to a *generated temporary* Jekyll site."""
from pathlib import Path
import html
import re
import sys

site = Path(sys.argv[1]).resolve()
source = Path(__file__).resolve().parents[1]
if site == source or source in site.parents:
    raise SystemExit('Preview output must be outside the source checkout.')

posts = []
for p in sorted((source / '_posts').rglob('*RSNA-Knee*.md')):
    raw = p.read_text()
    if not re.search(r'^series: rsna-knee$', raw, re.M):
        continue
    fm = raw.split('---', 2)[1]
    def field(key):
        m = re.search(r'^' + key + r':\s*(.+)$', fm, re.M)
        return m.group(1).strip().strip('"\'') if m else ''
    if field('published') not in {'true', 'false'}:
        raise SystemExit(f'Invalid publication flag: {p.name}')
    slug = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', p.stem)
    url = f'/posts/{slug}/'
    if not (site / url.strip('/') / 'index.html').is_file():
        raise SystemExit(f'Missing built post: {url}')
    image_block = re.search(r'^image:\s*\n((?:[ \t]+[^\n]*\n?)+)', fm, re.M)
    if not image_block:
        raise SystemExit(f'Missing cover metadata: {p.name}')
    def image_field(key):
        m = re.search(r'^\s+' + key + r':\s*(.+)$', image_block.group(1), re.M)
        return m.group(1).strip().strip('"\'') if m else ''
    posts.append(dict(order=int(field('series_order')), lang=field('lang'), date=field('date')[:10],
                      title=field('title'), url=url, published=field('published'), cover=image_field('path'),
                      cover_alt=image_field('alt'), width=int(image_field('width')),
                      height=int(image_field('height'))))

if len(posts) != 12:
    raise SystemExit(f'Expected 12 bilingual posts, got {len(posts)}')
rows = []
publication_flags = {p['published'] for p in posts}
publication_notice = ('모든 원고는 <code>published: ' + next(iter(publication_flags)) + '</code>입니다.'
                      if len(publication_flags) == 1 else '원고별 발행 설정이 다릅니다.')
for n in range(6):
    pair = {p['lang']: p for p in posts if p['order'] == n}
    en, ko = pair['en'], pair['ko']
    rows.append(f'''<section><img src="{html.escape(en['cover'], quote=True)}" alt="{html.escape(ko['cover_alt'], quote=True)}" width="{en['width']}" height="{en['height']}">
    <div><small>PART {n} · <time datetime="{en['date']}">{en['date']}</time></small><h2>{html.escape(en['title'])}</h2>
    <p>{html.escape(ko['title'])}</p><nav><a href="{en['url']}">Read in English</a>
    <a href="{ko['url']}" lang="ko">한국어로 읽기</a></nav></div></section>''')
page = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>RSNA Knee — Local Editorial Preview</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f3;color:#183344;font:17px/1.65 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:45px 24px}h1{font-size:clamp(27px,5vw,43px);margin:0 0 12px}
.notice{border-left:4px solid #277f85;padding:12px 18px;background:white;margin:25px 0 35px}
section{display:grid;grid-template-columns:36% 1fr;background:#fff;margin:20px 0;border-radius:12px;overflow:hidden}
section img{width:100%;height:100%;object-fit:contain;background:#f3f0e9}section>div{padding:24px}
small{color:#277f85;letter-spacing:.12em;font-weight:700}h2{font-size:22px;line-height:1.4;margin:8px 0}
p{margin:8px 0 18px}nav{display:flex;gap:12px;flex-wrap:wrap}a{display:inline-block;padding:8px 14px;border:1px solid #c6d4d9;border-radius:6px;color:#176a75;text-decoration:none}
@media(max-width:650px){section{grid-template-columns:1fr}section img{max-height:240px}section>div{padding:20px}}
</style><main><h1>RSNA Knee · 0–5편 로컬 미리보기</h1>
<div class="notice">''' + publication_notice + ''' 이 페이지는 로컬 임시 build 안에만 있습니다. 이 미리보기 도구는 commit·push·배포를 수행하지 않습니다.<br>
English 원고를 먼저 완성한 뒤 한국어판을 작성했습니다. 두 언어는 같은 영문 figure를 사용합니다.</div>
''' + '\n'.join(rows) + '''<p>Part 5는 대회 종료 후의 회고입니다. 외부 write-up 모음은 별도로 준비할 예정입니다.</p></main></html>'''
dest = site / 'rsna-preview'
dest.mkdir(exist_ok=True)
(dest / 'index.html').write_text(page)
print(f'Generated local-only index with {len(posts)} posts.')
