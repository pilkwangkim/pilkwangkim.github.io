import html
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
qa = Path(__file__).resolve().parent
manifest = json.loads((qa / "cover-prompts.json").read_text())
cards = []
short_titles = {
    "arc-agi-3-introduction": "ARC-AGI-3 Introduction",
    "arc-agi-3-research-r1": "ARC-AGI-3: Program-as-Policy",
    "birdclef-2026-eos9-pcen": "BirdCLEF: OOF-Gated PCEN",
    "maze-crawler-structured-baseline": "Maze Crawler",
    "mmlm-2026-baseline": "NCAA Basketball Baseline",
    "orbit-wars-structured-baseline": "Orbit Wars",
    "playground-s6e2-eda": "S6E2: Heart Disease EDA",
    "playground-s6e2-prediction": "S6E2: Boosting Pipeline",
    "playground-s6e5-driver-features": "S6E5: Driver Feature Engineering",
    "playground-s6e6-stellar-classification": "S6E6: Redshift and Color Geometry",
    "pokemon-tcg-working-note-1": "Pokémon TCG: Legal-Option Ranking",
    "quantum-transport-negf": "NEGF: Exact Lead Elimination",
    "rogii-error-anatomy": "ROGII: Error Anatomy",
    "rogii-tvt-alignment": "ROGII: Stratigraphic Alignment",
}
for row in manifest["covers"]:
    mode = "Existing body figure" if row["mode"] == "reused-body-image" else "Generated cover"
    cards.append(f'<figure><div><img src="{html.escape(row["path"])}" alt="{html.escape(row["alt_en"])}"></div><figcaption>{html.escape(short_titles[row["translation_key"]])}<small>{mode}</small></figcaption></figure>')
page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Technical cover review</title><style>
*{box-sizing:border-box}body{margin:0;padding:20px 25px;background:#f2f4f4;color:#243947;font:13px system-ui,sans-serif}
main{max-width:1050px;margin:auto}h1{font-size:22px;margin:0 0 4px}p{margin:0 0 18px;color:#526b74}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
figure{margin:0;background:white;border:1px solid #d9e2e4;border-radius:8px;overflow:hidden}figure>div{aspect-ratio:40/21;display:flex;align-items:center;justify-content:center;background:white}img{max-width:100%;max-height:100%;object-fit:contain}
figcaption{padding:6px 9px;font-size:11px;line-height:1.3;height:44px}small{display:block;color:#617d83;margin-top:3px;font-size:10px}
@media(max-width:650px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><main><h1>Kaggle &amp; Physics covers</h1><p>14 articles · 8 existing body figures · 6 generated technical illustrations</p><div class="grid">"""
page += "".join(cards) + "</div></main></html>"
Path("/tmp/pilkwang-technical-covers-20261007-site/cover-review.html").write_text(page)
print("Created local-only cover contact sheet")
