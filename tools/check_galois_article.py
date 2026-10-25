#!/usr/bin/env python3
"""Check paired Galois article sources, figures, executable examples, and notebooks."""
from __future__ import annotations
import argparse
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import build_galois_notebooks as builder

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-10-25-Galois-Theory-From-Scratch"
POSTS = ROOT / "_posts/2026-4Q"
ASSETS = ROOT / "assets/img/posts/2026-10-25-galois-from-scratch"
LABS = [f"galois-{s}" for s in
        ("quadratic", "permutations", "correspondence", "derived", "quintic", "quiz")]

def parts(source):
    body = source.split("---", 2)[2]
    return {
        "code": re.findall(r"^```python\n(.*?)^```", body, re.M | re.S),
        "display": re.findall(r"\$\$.*?\$\$", body, re.S),
        "figures": re.findall(r'<img src="([^"]+)"', body),
        "labs": re.findall(r'include interactive/article-lab\.html id="([^"]+)"', body),
        "tags": re.findall(r"\\tag\{(\d+)\}", body),
    }

def learning_structure(source):
    """Optional calculations must not hide section entries or browser labs."""
    depth = 0
    section = None
    folds = {}
    in_code = False
    for line in source.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        heading = re.match(r"## (\d+)\.", line)
        if heading:
            assert depth == 0, "A main section was accidentally folded away"
            section = int(heading[1])
            folds[section] = 0
        if line.startswith("<details"):
            assert depth == 0, "Use sibling examples rather than nested foldouts"
            assert section is not None
            depth += 1
            folds[section] += 1
        if line.startswith("</details>"):
            depth -= 1
            assert depth == 0, "Unmatched details closing tag"
        if 'include interactive/article-lab.html' in line:
            assert depth == 0, "Keep each interactive activity in the visible explanation"
    assert depth == 0, "Unclosed optional example"
    return folds

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = [(POSTS / f"{STEM}{suffix}.md").read_text(encoding="utf-8")
               for suffix in ("", "-KR")]
    en, ko = map(parts, sources)
    for key in ("code", "display", "figures", "labs", "tags"):
        assert en[key] == ko[key], f"Bilingual {key} mismatch"
    assert en["labs"] == LABS and len(set(en["labs"])) == 6
    assert len(en["code"]) == 6
    assert len(en["figures"]) == 20
    assert en["tags"] == list(map(str, range(1, 63))), "Equation tags must be sequential"
    assert len(en["display"]) == 62
    fold_layouts = [learning_structure(source) for source in sources]
    assert fold_layouts[0] == fold_layouts[1], "Bilingual foldout placement drifted"
    for index, source in enumerate(sources):
        assert source.count('<details markdown="1">') == source.count("</details>")
        assert source.count("```") % 2 == 0
        assert "date: 2026-09-25 09:00:00 +0900" in source
        assert "topic: algebra" in source
        assert "categories: [Algebra]" in source
        assert "translation_key: galois-theory-from-scratch" in source
        assert f"lang: {['en','ko'][index]}" in source
        assert len(re.findall(r"(?m)^## \d+\.", source)) == 18
        captions = re.findall(r"(?m)^[*_]Figure \d+\..*$", source)
        assert len(captions) == 10 and all(c.startswith("*Figure") and c.endswith("*")
                                         for c in captions), \
            "Use asterisk caption emphasis so TeX subscripts cannot close it"
        assert not re.search(r"\[(?:Milne|Conrad)[^\]]*`", source), "Broken citation delimiter"
        assert "\t" not in source, "Tabs can corrupt TeX commands and Markdown"
        assert not any(ord(char) < 32 and char != "\n" for char in source), \
            "An escaped string introduced a control character into the source"
        prose = re.sub(r"\$\$.*?\$\$", "", source, flags=re.S)
        prose = re.sub(r"^```.*?^```", "", prose, flags=re.M | re.S)
        for expression in re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", prose):
            assert "|" not in expression, \
                "Use TeX delimiters in inline math; literal pipes can create Markdown tables"
            assert not re.search(r"(?<!\\)\bsqrt(?=[{(0-9])", expression), \
                "Missing TeX backslash in an inline square root"
            assert not re.search(r"\\[{}]", expression), \
                "Use lbrace/rbrace in inline math; Markdown consumes escaped braces"
        assert not re.search(r"(?m)^\+[^+]", prose), "Patch artifact in post"
        for expression in parts(source)["display"]:
            assert "|" not in expression, "Use TeX delimiters rather than Markdown pipes"
            assert "^*" not in expression
            assert expression.count("{") == expression.count("}")
    for url in en["figures"]:
        path = ROOT / url.replace("{{ site.baseurl }}", "").lstrip("/")
        assert path.is_file() and path.with_suffix(".png").is_file()
        svg = ET.parse(path)
        assert not svg.findall(".//{http://www.w3.org/2000/svg}text"), "Outline scientific labels"
    manifest = json.loads((ASSETS / "figure-data.json").read_text())
    by_stem = {row["stem"]: row for row in manifest["assets"]}
    for source in sources:
        tags = re.findall(r'<img src="([^"]+)" alt="[^"]*" width="(\d+)" height="(\d+)">', source)
        assert len(tags) == 20
        for url, width, height in tags:
            row = by_stem[Path(url).stem]
            assert (int(width), int(height)) == (row["source_width"], row["source_height"]), \
                "HTML image dimensions must match the responsive figure"
    figure_rows = [row for row in manifest["assets"] if row["stem"].startswith("fig-")]
    assert len(figure_rows) == 20
    for row in figure_rows:
        assert row["layout"]["text_collisions"] == 0
        assert row["layout"]["minimum_css_font_px"] >= 16
    namespace = {"__name__": "galois_article_source_check"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        for number, code in enumerate(en["code"], 1):
            exec(compile(code, f"galois-source-{number}", "exec"), namespace)
    assert namespace["test_solvability"](namespace["reference_generated_solvable"]) == 7
    # A Run All must not silently replace the user's editable starter.
    try:
        namespace["generated_solvable"]([(1, 0)], 2)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("The editable TODO starter was overwritten")
    questions = builder.browser_quiz()
    notebook_stats = []
    for lang in ("en", "ko"):
        article = builder.read_article(lang)
        cells = builder.article_cells(lang, article, questions)
        builder.execute(cells)
        notebook = json.loads((ROOT / f"assets/notebooks/galois-from-scratch-{lang}.ipynb").read_text())
        assert notebook["cells"] == cells, f"Notebook drift: {lang}"
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        assert len(code_cells) == 6
        assert [cell["execution_count"] for cell in code_cells] == list(range(1, 7))
        assert not any(output["output_type"] == "error"
                       for cell in code_cells for output in cell["outputs"])
        figures = 0
        for cell in cells:
            for filename, payload in cell.get("attachments", {}).items():
                assert base64.b64decode(payload["image/png"]) == (ASSETS / filename).read_bytes()
                figures += 1
        assert figures == 10
        notebook_stats.append({"lang": lang, "cells": len(cells),
                               "executed_python_cells": len(code_cells), "embedded_figures": figures})
    example_source = (ROOT / "assets/code/galois-article-examples.py").read_text()
    for code in en["code"]:
        assert code.strip() in example_source, "Downloadable examples drifted"
    report = {
        "status": "passed", "sections": 18, "display_equations": len(en["display"]),
        "python_examples": len(en["code"]), "coding_test_cases": 7,
        "figure_pairs": 10, "activities": len(en["labs"]), "illustrated_quiz_questions": 5,
        "optional_blocks_by_section": fold_layouts[0],
        "notebooks": notebook_stats,
        "source_sha256": [hashlib.sha256(s.encode()).hexdigest() for s in sources],
    }
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
