#!/usr/bin/env python3
"""Check paired MS-BTE source, formulas, runnable code, and notebook artifacts."""
from __future__ import annotations

import ast
import base64
from collections import Counter
import contextlib
import io
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-11-06-Multi-Subband-Boltzmann-Transport-From-Scratch"
OUTPUT_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
OUTPUT_INTEGER = re.compile(r"[-+]?\d+\Z")
OUTPUT_NONFINITE = re.compile(r"(?<!\w)[+-]?(?:nan|inf(?:inity)?)(?!\w)", re.I)


def inline_math(source):
    prose = re.sub(r"^```[^\n]*\n.*?^```[ \t]*$", "", source, flags=re.M | re.S)
    prose = re.sub(r"\$\$.*?\$\$", "", prose, flags=re.S)
    return Counter(re.findall(r"(?<![\\$])\$(?!\$)(.+?)(?<!\\)\$(?!\$)", prose, flags=re.S))


def structure(source):
    return {
        "math": re.findall(r"\$\$.*?\$\$", source, re.S),
        "code": re.findall(r"^```python\n(.*?)^```", source, re.M | re.S),
        "labs": re.findall(r'include interactive/article-lab\.html id="([^"]+)"', source),
        "images": re.findall(r'<img src="([^"]+)"', source),
        "mobile": re.findall(r"src='([^']+)'", source),
        "references": re.findall(r"^\[([^\]]+)\]: (\S+)$", source, re.M),
        "inline_math": inline_math(source),
    }


def notebook_outputs_match(stored, replayed):
    """Compare stream structure/text exactly, with 1e-12 float roundoff only."""
    if not isinstance(stored, list) or not isinstance(replayed, list) or len(stored) != len(replayed):
        return False
    stream_keys = {"output_type", "name", "text"}
    for left, right in zip(stored, replayed):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if set(left) != stream_keys or set(right) != stream_keys:
            return False
        if left["output_type"] != "stream" or right["output_type"] != "stream" or left["name"] != right["name"]:
            return False
        a, b = left["text"], right["text"]
        if not isinstance(a, str) or not isinstance(b, str):
            return False
        if OUTPUT_NONFINITE.search(a) or OUTPUT_NONFINITE.search(b):
            return False
        # Compare the actual intervening text, not a replacement sentinel that
        # could also appear literally in the output.
        if OUTPUT_NUMBER.split(a) != OUTPUT_NUMBER.split(b):
            return False
        a_numbers, b_numbers = OUTPUT_NUMBER.findall(a), OUTPUT_NUMBER.findall(b)
        if len(a_numbers) != len(b_numbers):
            return False
        for x, y in zip(a_numbers, b_numbers):
            # Counts and dimensions printed as integers must remain exact,
            # including values too large for an exact float representation.
            if OUTPUT_INTEGER.fullmatch(x) and OUTPUT_INTEGER.fullmatch(y):
                if int(x) != int(y):
                    return False
                continue
            p, q = float(x), float(y)
            if not math.isfinite(p) or not math.isfinite(q):
                return False
            if not math.isclose(p, q, rel_tol=1e-12, abs_tol=1e-12):
                return False
    return True


def check_executed_notebook(notebook, language, images):
    # Reconstruct Markdown with the documented builder conversion, without
    # invoking its executor or accepting a previously generated text snapshot.
    from build_msbte_notebooks import notebook_cells
    expected = notebook_cells(language, render_figures=False)
    actual = notebook["cells"]
    assert [(c["cell_type"], c["source"], c["id"]) for c in actual] == [
        (c["cell_type"], c["source"], c["id"]) for c in expected
    ], f"Notebook {language}: stale Markdown/code or cell IDs"
    namespace = {"__name__": "__main__"}
    attachments = {}
    code_count = 0
    for cell in actual:
        if cell["cell_type"] == "code":
            code_count += 1
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(cell["source"], f"msbte-{language}:cell-{code_count}", "exec"), namespace)
            outputs = [] if not stdout.getvalue() else [{"output_type": "stream", "name": "stdout", "text": stdout.getvalue()}]
            assert notebook_outputs_match(cell["outputs"], outputs), f"Notebook {language}: stale stored output in cell {code_count}"
            assert not stderr.getvalue(), f"Notebook {language}: unexpected stderr"
        else:
            names = re.findall(r"attachment:([^\s)]+)", cell["source"])
            assert set(names) == set(cell.get("attachments", {}))
            for name, formats in cell.get("attachments", {}).items():
                assert name not in attachments and set(formats) == {"image/png"}
                attachments[name] = base64.b64decode(formats["image/png"], validate=True)
    expected_pngs = {Path(path).stem + ".png": (ROOT / path.lstrip("/")).with_suffix(".png").read_bytes() for path in images}
    assert attachments == expected_pngs, f"Notebook {language}: embedded figures differ from checked-in PNGs"
    assert namespace["check_collision"](namespace["elastic_collision_reference"]) == "3 cases passed"
    try:
        namespace["elastic_collision"](None, None, None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Run All overwrote the exercise starter")
    return code_count


def main():
    texts = [(ROOT / f"_posts/2026-4Q/{STEM}{suffix}.md").read_text(encoding="utf-8")
             for suffix in ("", "-KR")]
    pair = [structure(text) for text in texts]
    # Inline expressions can be split differently by the two languages (for
    # example a quantity and its unit in one span versus two). Their rendered
    # TeX is checked in the browser; numbered/display equations must be exact.
    assert {k: v for k, v in pair[0].items() if k != "inline_math"} == {
        k: v for k, v in pair[1].items() if k != "inline_math"
    }, "English/Korean display math, code, labs, images, or references diverged"
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
    replayed = 0
    for language in ("en", "ko"):
        notebook = json.loads((ROOT / f"assets/notebooks/multi-subband-bte-{language}.ipynb").read_text())
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert [cell["source"].strip() for cell in code] == [snippet.strip() for snippet in data["code"]], "Notebook code is stale"
        assert [cell["execution_count"] for cell in code] == list(range(1, len(data["code"]) + 1)), "Notebook must be executed in source order"
        assert not any(o["output_type"] == "error" for cell in code for o in cell["outputs"])
        assert sum(len(cell.get("attachments", {})) for cell in notebook["cells"]) == len(data["images"])
        assert sum("exercise" in cell["metadata"].get("tags", []) for cell in code) == 1
        assert not any("{%" in cell["source"] or "{{" in cell["source"] for cell in notebook["cells"])
        replayed += check_executed_notebook(notebook, language, data["images"])
    assert len(data["code"]) == 12
    print(json.dumps({"status": "PASS", "paired_articles": 2, "sections": 19,
                      "numbered_equations": len(tags), "display_math_blocks": len(data["math"]), "python_examples": len(data["code"]),
                      "labs": len(data["labs"]), "responsive_body_figures": len(data["images"]),
                      "executed_notebooks": 2, "replayed_notebook_cells": replayed,
                      "inline_math_occurrences": {lang: sum(p["inline_math"].values()) for lang, p in zip(("en", "ko"), pair)},
                      "byte_exact_embedded_figures": 2 * len(data["images"])}))


if __name__ == "__main__":
    main()
