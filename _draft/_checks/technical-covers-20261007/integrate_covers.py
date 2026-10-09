import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[3]
qa = root / "_draft/_checks/technical-covers-20261007"
baseline = json.loads((qa / "baseline-posts.json").read_text())
missing = [row for row in baseline
           if row["front_matter"].get("topic") not in ("essays", "reference")
           and not row["front_matter"].get("hidden")
           and row["front_matter"].get("published") is not False
           and not row["front_matter"].get("image")]
groups = {}
for row in missing:
    groups.setdefault(row["front_matter"]["translation_key"], []).append(row)
assert len(groups) == 14 and len(missing) == 26
receipts = []
for key, members in sorted(groups.items()):
    receipt = json.loads((qa / f"{key}-cover.json").read_text())
    assert receipt["translation_key"] == key
    assert sorted(receipt["source_posts"]) == sorted(row["path"] for row in members)
    assert receipt["mode"] in ("reused-body-image", "built-in imagegen")
    asset = root / receipt["path"].lstrip("/")
    assert asset.is_file(), str(asset)
    if isinstance(receipt["dimensions"], dict):
        receipt["dimensions"] = [receipt["dimensions"]["width"], receipt["dimensions"]["height"]]
    assert len(receipt["dimensions"]) == 2 and min(receipt["dimensions"]) > 0
    for row in members:
        raw = (root / row["path"]).read_bytes()
        if receipt["mode"] == "reused-body-image":
            assert receipt["path"].encode() in raw, (key, row["path"])
        assert receipt["alt_en" if row["front_matter"]["lang"] == "en" else "alt_ko"].strip()
    receipt["bytes"] = asset.stat().st_size
    receipt["sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
    receipt["title"] = next(row["front_matter"]["title"] for row in members
                            if row["front_matter"]["lang"] == "en")
    receipts.append(receipt)

# Add only image metadata; preserve every original byte of the article body.
for receipt in receipts:
    for row in groups[receipt["translation_key"]]:
        path = root / row["path"]
        raw = path.read_bytes()
        closing = re.search(rb"\n---[ \t]*\r?\n", raw)
        assert closing and raw.startswith(b"---\n"), str(path)
        alt = receipt["alt_en" if row["front_matter"]["lang"] == "en" else "alt_ko"]
        block = ("image:\n"
                 f"  path: {receipt['path']}\n"
                 f"  alt: {json.dumps(alt, ensure_ascii=False)}\n"
                 "  hide_caption: true\n"
                 + (f"  fit: contain\n  width: {receipt['dimensions'][0]}\n  height: {receipt['dimensions'][1]}\n"
                    if receipt["mode"] == "reused-body-image" else "")).encode()
        position = closing.start() + 1
        path.write_bytes(raw[:position] + block + raw[position:])

manifest = {
    "date": "2026-10-07",
    "scope": "Previously coverless visible Kaggle and Physics articles",
    "logical_articles": len(receipts),
    "source_pages": len(missing),
    "reused_body_figures": sum(r["mode"] == "reused-body-image" for r in receipts),
    "generated_covers": sum(r["mode"] == "built-in imagegen" for r in receipts),
    "generation_method": "Built-in image_gen imagegen tool; no CLI fallback",
    "policy": "Existing body figures reused unchanged where suitable. Generated covers depict conceptual workflows, not measured results. EN and KO share one asset per article. Article bodies and existing cover assets preserved.",
    "covers": receipts,
}
(qa / "cover-prompts.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in manifest.items() if k != "covers"}, ensure_ascii=False, indent=2))
