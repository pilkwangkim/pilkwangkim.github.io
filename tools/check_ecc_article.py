#!/usr/bin/env python3
"""Check bilingual ECC structure, formula/code parity, local assets and notebooks."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-11-05-Error-Correcting-Codes-From-Scratch"
LABS = [f"ecc-{name}" for name in ("repetition", "parity", "distance", "hamming", "quiz")]


def extract(text):
    return {
        "code": re.findall(r"^```python\n(.*?)^```", text, re.M | re.S),
        "math": re.findall(r"\$\$.*?\$\$", text, re.S),
        "labs": re.findall(r'include interactive/article-lab\.html id="([^"]+)"', text),
        "images": re.findall(r'<img src="([^"]+)"', text),
        "sources": re.findall(r"src='([^']+)'", text),
        "refs": re.findall(r"^\[[^\]]+\]: (\S+)$", text, re.M),
    }


def main():
    texts = [(ROOT / f"_posts/2026-4Q/{STEM}{suffix}.md").read_text(encoding="utf-8")
             for suffix in ("", "-KR")]
    values = [extract(text) for text in texts]
    assert values[0] == values[1], "Bilingual equations/code/labs/assets/references drifted"
    data = values[0]
    assert data["labs"] == LABS
    assert len(data["images"]) == 4 and len(data["sources"]) == 4
    tags = re.findall(r"\\tag\{(\d+)\}", texts[0])
    assert tags == [str(n) for n in range(1, 29)], tags
    assert len(data["code"]) == 9
    for snippet in data["code"]:
        ast.parse(snippet)
    for text in texts:
        assert text.count('<details markdown="1">') == text.count("</details>")
        assert text.count("$$") == 56
        assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text)
        assert len(re.findall(r"^## \d+\.", text, re.M)) == 16
        assert "NotImplementedError" in text
        # Explicit lbrace/rbrace survive Markdown's backslash processing.
        assert "\\{" not in text and "\\}" not in text
    for image in data["images"] + data["sources"]:
        path = ROOT / image.lstrip("/")
        assert path.is_file(), path
        svg = ET.parse(path).getroot()
        assert svg.find("{http://www.w3.org/2000/svg}title") is not None
        assert svg.find("{http://www.w3.org/2000/svg}desc") is not None
    for lang in ("en", "ko"):
        path = ROOT / f"assets/notebooks/error-correcting-codes-{lang}.ipynb"
        notebook = json.loads(path.read_text())
        code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert len(code) == 9 and all(c["execution_count"] for c in code)
        assert all(o["output_type"] != "error" for c in code for o in c["outputs"])
        assert sum(len(c.get("attachments", {})) for c in notebook["cells"]) == 4
        assert len([c for c in code if "exercise" in c["metadata"].get("tags", [])]) == 1
        assert not any("{%" in c["source"] or "{{" in c["source"] for c in notebook["cells"])
        assert not any("](#lab-ecc-" in c["source"] for c in notebook["cells"]), "Notebook activity links must target the article"
        assert not any(re.search(r"^\{:", c["source"], re.M) for c in notebook["cells"]), "Jekyll block attributes must not leak into notebooks"
    print(json.dumps({"paired_articles": 2, "sections": 16, "equations": 28,
                      "python_examples": 9, "labs": 5, "body_figure_pairs": 4,
                      "executed_notebooks": 2, "status": "PASS"}))


if __name__ == "__main__":
    main()
