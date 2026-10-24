#!/usr/bin/env python3
"""Verify bilingual sorting sources, Python code, mathematical fixtures and notebooks."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import runpy
import xml.etree.ElementTree as ET
import build_sorting_notebooks as notebook_builder

ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / "_posts/2026-4Q"
STEM = "2026-10-24-Sorting-From-Scratch"
ASSETS = ROOT / "assets/img/posts/2026-10-24-sorting-from-scratch"


def parts(source):
    body = source.split("---", 2)[2]
    code = re.findall(r"^```python\n(.*?)^```", body, re.M | re.S)
    prose = re.sub(r"^```.*?^```", "", body, flags=re.M | re.S)
    return {
        "code": code,
        "display": re.findall(r"\$\$.*?\$\$", prose, re.S),
        "math": re.findall(r"\$\$.*?\$\$|(?<!\$)\$(?!\$)[^\n$]+\$", prose, re.S),
        "figures": re.findall(r'<img src="([^"]+)"[^>]*>', body),
        "labs": re.findall(r'include interactive/article-lab\.html id="([^"]+)"', body),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = [(POSTS / f"{STEM}{suffix}.md").read_text(encoding="utf-8")
               for suffix in ("", "-KR")]
    en, ko = map(parts, sources)
    for field in ("code", "display", "figures", "labs"):
        assert en[field] == ko[field], f"Bilingual mismatch: {field}"
    assert len(en["labs"]) == 8 and len(set(en["labs"])) == 8
    assert set(en["labs"]) == {f"sorting-{name}" for name in (
        "playground", "random", "inversions", "merge", "partition", "distribution", "quiz", "code")}
    assert len(en["figures"]) == 16, "Eight desktop/mobile figure pairs expected"
    for source in sources:
        assert source.count('<details markdown="1">') == source.count("</details>")
        assert "categories: [\"Data Structure & Algorithm\"]" in source
        assert "date: 2026-10-17 09:00:00 +0900" in source
        assert "translation_key: sorting-from-scratch" in source
        assert "{{ site.baseurl }}" not in re.sub(r'{{ site.baseurl }}(?:/[^\s\"\)]+)', "", source), "Unexpected Liquid"
        for expression in parts(source)["math"]:
            assert "^*" not in expression, "Use \\ast, not a Markdown-sensitive superscript star"
            assert "|" not in expression, "Use \\lvert and \\rvert in Markdown math"
        assert len(re.findall(r"(?m)^```", source)) % 2 == 0, "Unbalanced code fences"
    for url in en["figures"]:
        path = ROOT / url.replace("{{ site.baseurl }}", "").lstrip("/")
        assert path.is_file(), path
        svg = ET.parse(path)
        assert not svg.findall(".//{http://www.w3.org/2000/svg}text"), "SVG text must be outlined"
        assert path.with_suffix(".png").is_file()
    namespace = {"__name__": "__sorting_check__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for index, snippet in enumerate(en["code"], 1):
            exec(compile(snippet, f"sorting-article-snippet-{index}", "exec"), namespace)
    reference_source = ('#!/usr/bin/env python3\n'
                        '"""Executed Python snippets from the sorting tutorial; standard library only."""\n\n'
                        + "\n\n".join(en["code"]) + "\n")
    assert (ROOT / "assets/code/sorting-reference.py").read_text(encoding="utf-8") == reference_source, "Downloadable reference drifted from article code"
    challenge = runpy.run_path(str(ROOT / "assets/code/sorting-code-test.py"))
    reference = challenge["reference_sort_colors"]
    assert all(row["passed"] for row in challenge["run_tests"](reference, False))
    starter_results = challenge["run_tests"](challenge["sort_colors"], False)
    assert not all(row["passed"] for row in starter_results), "The TODO starter must reveal an incorrect decision"
    assert [row["id"] for row in starter_results if row["passed"]] == ["empty", "sorted", "all-twos"]
    code_cases = 0
    for n in range(9):
        for values in itertools.product(range(3), repeat=n):
            actual, stats = list(values), {}
            reference(actual, stats=stats)
            assert actual == sorted(values) and stats["iterations"] == n
            assert stats["writes"] <= 2 * n
            code_cases += 1
    # Count inversion expectations by enumeration, independently of all trace engines.
    inversion_permutations = 0
    for n in range(1, 8):
        total = 0
        permutations = list(itertools.permutations(range(n)))
        for a in permutations:
            total += sum(a[i] > a[j] for i in range(n) for j in range(i + 1, n))
        assert 4 * total == len(permutations) * n * (n - 1)
        inversion_permutations += len(permutations)
    notebooks = []
    code_sequences = []
    browser_content = notebook_builder.browser_content()
    for language in ("en", "ko"):
        path = ROOT / f"assets/notebooks/sorting-from-scratch-{language}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        _, current_cells = notebook_builder.article_cells(language, browser_content)
        notebook_builder.execute(current_cells)
        assert notebook["cells"] == current_cells, f"Notebook source or executed output drift: {language}"
        assert notebook["metadata"]["article_language"] == language
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert [cell["execution_count"] for cell in code] == list(range(1, len(code) + 1))
        assert not any(out["output_type"] == "error" for cell in code for out in cell["outputs"])
        code_sequences.append([cell["source"] for cell in code])
        figures = 0
        for cell in notebook["cells"]:
            text = "".join(cell["source"])
            assert "{% include" not in text and "{{ site.baseurl }}" not in text
            assert '<div class="sorting-figure-' not in text and "{: .sorting-" not in text
            assert not re.search(r"(?m)^\{:\s*#", text), "Blog heading anchors must not leak into notebooks"
            assert "sorting-section-anchor" not in text
            for name, attachment in cell.get("attachments", {}).items():
                assert f"attachment:{name}" in text
                assert base64.b64decode(attachment["image/png"]) == (ASSETS / name).read_bytes()
                figures += 1
        assert figures == 8, f"Notebook must embed all eight figures: {figures}"
        assert any("exercise-starter" in c["metadata"].get("tags", []) for c in code)
        assert any("exercise-solution" in c["metadata"].get("tags", []) for c in code)
        notebooks.append({"language": language, "cells": len(notebook["cells"]),
                          "executed_code_cells": len(code), "embedded_figures": figures})
    assert code_sequences[0] == code_sequences[1], "Notebook code must match between languages"
    report = {"status": "passed", "article_python_snippets": len(en["code"]),
              "display_equations": len(en["display"]), "math_fragments": len(en["math"]),
              "figure_pairs": 8, "activities": len(en["labs"]),
              "reference_matches_article": True, "notebooks_freshly_executed": True,
              "sort_colors_exhaustive_cases": code_cases,
              "inversion_permutations": inversion_permutations, "notebooks": notebooks,
              "source_sha256": [hashlib.sha256(source.encode()).hexdigest() for source in sources]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
