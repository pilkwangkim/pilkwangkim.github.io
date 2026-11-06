#!/usr/bin/env python3
"""Check paired MS-BTE source, formulas, runnable code, and notebook artifacts."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-11-06-Multi-Subband-Boltzmann-Transport-From-Scratch"


def structure(source):
    return {
        "math": re.findall(r"\$\$.*?\$\$", source, re.S),
        "code": re.findall(r"^```python\n(.*?)^```", source, re.M | re.S),
        "labs": re.findall(r'include interactive/article-lab\.html id="([^"]+)"', source),
        "images": re.findall(r'<img src="([^"]+)"', source),
        "mobile": re.findall(r"src='([^']+)'", source),
        "references": re.findall(r"^\[([^\]]+)\]: (\S+)$", source, re.M),
    }


def main():
    texts = [(ROOT / f"_posts/2026-4Q/{STEM}{suffix}.md").read_text(encoding="utf-8")
             for suffix in ("", "-KR")]
    pair = [structure(text) for text in texts]
    assert pair[0] == pair[1], "English/Korean math, code, labs, images, or references diverged"
    data = pair[0]
    assert data["labs"] == [f"msbte-{name}" for name in ("basics", "confinement", "injection", "scattering", "quiz")]
    assert len(data["images"]) == 7 and len(data["images"]) == len(data["mobile"])
    assert [re.search(r"fig-(\d+)-", path)[1] for path in data["images"]] == ["06", "01", "07", "05", "02", "03", "04"]
    assert {key for key, _ in data["references"]} >= {"bulk", "limit", "kinetics", "modespace", "folding"}
    tags = re.findall(r"\\tag\{(\d+)\}", texts[0])
    assert tags == [str(i) for i in range(1, 43)], tags
    for snippet in data["code"]:
        ast.parse(snippet, feature_version=(3, 9))
    for text in texts:
        assert text.count('<details markdown="1">') == text.count("</details>")
        assert text.count("$$") == 98  # 42 numbered equations + 7 worked derivations.
        assert re.findall(r"^## (\d+)\.", text, re.M) == [str(i) for i in range(1, 20)]
        assert text.index('id="msbte-basics"') < text.index('id="msbte-confinement"')
        assert [int(number) for number in re.findall(r'<figcaption[^>]*>(?:Figure|그림) (\d+)\.', text)] == list(range(1, 7))
        assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text)
        assert "\\{" not in text and "\\}" not in text
        assert "C_n[f],dk" not in text
        assert r"f_b=1/[1+\exp(1)]" in text
        assert r"W_{b\to a}/[2(1+\exp(1))]" in text
        assert r"\exp(8)" not in text, "Equation renumbering must not rewrite numeric math arguments"
        assert "$m^*$" not in text, "Repeated literal stars can become Markdown emphasis; use m^{\\ast}"
        assert "date: 2026-11-06 09:00:00 +0900" in text
        assert "topic: quantum-transport" in text
        assert not re.search(r"^\$\|[^\n$]*\|\$", text, re.M)
        # Kramdown may interpret ANY raw pipe in prose math as a table cell,
        # even when the pipe is not at the start of the paragraph.
        prose = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
        assert not any("|" in line and not line.startswith("|") for line in prose.splitlines()), "Use lvert/rvert or vert in math, never raw Markdown pipes"
    for image in data["images"] + data["mobile"]:
        path = ROOT / image.lstrip("/")
        svg = ET.parse(path).getroot()
        assert svg.find("{http://www.w3.org/2000/svg}title") is not None
        assert svg.find("{http://www.w3.org/2000/svg}desc") is not None
    for language in ("en", "ko"):
        notebook = json.loads((ROOT / f"assets/notebooks/multi-subband-bte-{language}.ipynb").read_text())
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert [cell["source"].strip() for cell in code] == [snippet.strip() for snippet in data["code"]], "Notebook code is stale"
        assert [cell["execution_count"] for cell in code] == list(range(1, len(data["code"]) + 1)), "Notebook must be executed in source order"
        assert not any(o["output_type"] == "error" for cell in code for o in cell["outputs"])
        assert sum(len(cell.get("attachments", {})) for cell in notebook["cells"]) == len(data["images"])
        assert sum("exercise" in cell["metadata"].get("tags", []) for cell in code) == 1
        assert not any("{%" in cell["source"] or "{{" in cell["source"] for cell in notebook["cells"])
    assert len(data["code"]) == 12
    print(json.dumps({"status": "PASS", "paired_articles": 2, "sections": 19,
                      "numbered_equations": len(tags), "display_math_blocks": len(data["math"]), "python_examples": len(data["code"]),
                      "labs": len(data["labs"]), "responsive_body_figures": len(data["images"]),
                      "executed_notebooks": 2}))


if __name__ == "__main__":
    main()
