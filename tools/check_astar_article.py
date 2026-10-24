#!/usr/bin/env python3
"""Validate A* translation parity, snippets, figures, and source math safety."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import contextlib
import hashlib
import json
import io
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from build_astar_notebooks import article_cells, exercise_code

ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / "_posts/2026-4Q"
STEM = "2026-10-24-A-Star-From-Scratch"
FIGURES = ROOT / "assets/img/posts/2026-10-24-astar-from-scratch"


def protected(source):
    body = source.split("---", 2)[2]
    code = re.findall(r"^```python\n(.*?)^```", body, re.M | re.S)
    without_code = re.sub(r"^```.*?^```", "", body, flags=re.M | re.S)
    math = re.findall(r"\$\$.*?\$\$|(?<!\$)\$(?!\$)[^\n$]+\$", without_code, re.S)
    tags = re.findall(r"\\tag\{(\d+)\}", without_code)
    figures = re.findall(r'<img src="([^"]+)"[^>]*width="(\d+)" height="(\d+)"', body)
    labs = re.findall(r'include interactive/article-lab\.html id="([^"]+)"', body)
    return {"code": code, "math": math, "tags": tags, "figures": figures, "labs": labs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    en_path, ko_path = (POSTS / f"{STEM}{suffix}.md" for suffix in ("", "-KR"))
    en_source, ko_source = en_path.read_text(), ko_path.read_text()
    en, ko = protected(en_source), protected(ko_source)
    for key in ("code", "tags", "figures", "labs"):
        assert en[key] == ko[key], f"Translation mismatch: {key}"
    en_display = [m for m in en["math"] if m.startswith("$$")]
    ko_display = [m for m in ko["math"] if m.startswith("$$")]
    assert en_display == ko_display, "Display-equation mismatch"
    # Natural Korean word order may reorder inline mentions. Every English
    # formula occurrence is retained; only extra single-symbol mentions are OK.
    missing = Counter(en["math"]) - Counter(ko["math"])
    extra = Counter(ko["math"]) - Counter(en["math"])
    assert not missing, f"Missing Korean formula occurrences: {missing}"
    assert all(re.fullmatch(r"\$[A-Za-z]\$", value) for value in extra), extra
    assert en["tags"] == [str(n) for n in range(1, 28)]
    assert len(en["figures"]) == 16 and len(en["labs"]) == 13
    assert len(set(en["labs"])) == 13
    assert en["labs"][-2:] == ["astar-quiz", "astar-code-test"]
    exercises = json.loads((ROOT / "assets/data/astar-exercises.json").read_text())
    assert len(exercises["questions"]) == 5
    assert len({q["id"] for q in exercises["questions"]}) == 5
    assert len(exercises["coding"]["cases"]) == 8
    exercise_images = [q["image"] for q in exercises["questions"]] + [exercises["coding"]["image"]]
    for question in exercises["questions"]:
        assert 0 <= question["answer"] < len(question["options"])
        for key in ("title", "prompt", "hint", "explanation", "followup"):
            assert all(question[key].get(language) for language in ("en", "ko")), key
    for spec in exercise_images:
        for variant in ("desktop", "mobile", "png"):
            path = ROOT / spec[variant].lstrip("/")
            assert path.is_file(), path
            if variant != "png":
                assert not ET.parse(path).findall(".//{http://www.w3.org/2000/svg}text")
    for expression in en["math"]:
        assert "^*" not in expression, "Use \\ast to protect Markdown inline math"
        assert "|" not in expression, "Use \\lvert / \\rvert, not Markdown pipe delimiters"
    namespace = {"__name__": "__main__"}
    for index, snippet in enumerate(en["code"], 1):
        exec(compile(snippet, f"article-python-{index}", "exec"), namespace)
    # The intentionally unfinished starter must remain a real exercise. The
    # separately named reference is executed against the same eight fixtures.
    challenge = exercise_code(exercises)
    assert all(token in challenge["starter"] for token in
               ("TODO_PRIORITY", "TODO_RELAXATION", "TODO_GOAL_CHECK"))
    exercise_namespace = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(challenge["starter"], exercise_namespace)
        exec(challenge["tests"], exercise_namespace)
        unfinished = exercise_namespace["run_astar_code_tests"](exercise_namespace["astar_cost"], verbose=False)
        assert not any(row["passed"] for row in unfinished), "Untouched starter unexpectedly solved tests"
        exec(challenge["reference"], exercise_namespace)
    assert all(row["passed"] for row in exercise_namespace["reference_results"])
    assert (ROOT / "assets/code/astar-code-test.py").is_file()
    for source, width, height in en["figures"]:
        path = ROOT / source.replace("{{ site.baseurl }}", "").lstrip("/")
        svg = ET.parse(path)
        assert not svg.findall(".//{http://www.w3.org/2000/svg}text")
        assert int(width) > 0 and int(height) > 0
    for language in ("en", "ko"):
        notebook = json.loads((ROOT / f"assets/notebooks/astar-from-scratch-{language}.ipynb").read_text())
        article_source = en_source if language == "en" else ko_source
        expected_title = json.loads(re.search(r'^title: (".*")$', article_source, re.M).group(1))
        assert notebook["metadata"]["title"] == expected_title, "Notebook title differs from article"
        assert all(not re.search(r'(?m)^\{:[ \t]*id="[^"\n]+"[ \t]*\}', "".join(c["source"]))
                   for c in notebook["cells"]), "Blog-only heading IDs leaked into notebook Markdown"
        assert all("{: .astar-worked-table }" not in "".join(c["source"])
                   for c in notebook["cells"]), "Blog-only table attributes leaked into notebook Markdown"
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        expected_code = ["".join(c["source"]).rstrip() for c in article_cells(language)
                         if c["cell_type"] == "code"]
        assert ["".join(c["source"]).rstrip() for c in code_cells] == expected_code
        assert [c["execution_count"] for c in code_cells] == list(range(1, len(code_cells) + 1))
        assert all(o.get("output_type") != "error" for c in code_cells for o in c["outputs"])
        assert sum(len(c.get("attachments", {})) for c in notebook["cells"]) == 14
        tags = [tag for c in code_cells for tag in c.get("metadata", {}).get("tags", [])]
        assert all(tag in tags for tag in ("exercise-starter", "exercise-tests", "exercise-solution"))
        markdown = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for question in exercises["questions"]:
            assert question["prompt"][language] in markdown
        for cell in notebook["cells"]:
            for name, mime_data in cell.get("attachments", {}).items():
                assert base64.b64decode(mime_data["image/png"]) == (FIGURES / name).read_bytes(), name
    report = {"status": "passed", "python_blocks_per_language": len(en["code"]),
              "math_fragments": {"en": len(en["math"]), "ko": len(ko["math"])},
              "matching_display_equations": len(en_display), "equations": len(en["tags"]),
              "responsive_figure_pairs": len(en["figures"]) // 2, "interactive_labs": en["labs"],
              "executed_notebooks": 2,
              "illustrated_questions": 5, "exercise_figure_pairs": 6,
              "coding_cases": len(exercises["coding"]["cases"]),
              "coding_reference": "8/8 pass; untouched TODO starter intentionally fails",
              "notebook_embedded_figures": 14,
              "parity": "identical code, display math, figure sources/dimensions, lab IDs; inline formula multiset retained",
              "korean_extra_symbol_mentions": dict(extra),
              "english_sha256": hashlib.sha256(en_source.encode()).hexdigest(),
              "korean_sha256": hashlib.sha256(ko_source.encode()).hexdigest()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
