#!/usr/bin/env python3
"""Paired-source and executable-artifact checks for the Drude tutorial."""
from __future__ import annotations

import ast
import base64
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-11-07-Drude-Model-From-Scratch"
COMMON_ANCHORS = [f"drude-{i:02d}" for i in range(1, 21)] + ["drude-references"]
SITE = "https://pilkwangkim.github.io"
FIGURE_STEMS = (
    "fig-sec01-charge-conservation", "fig-01-fields", "fig-sec03-mean-drift", "fig-sec04-mobility",
    "fig-02-drude", "fig-03-ac", "fig-sec07-bloch-states", "fig-sec08-wave-packet", "fig-04-band",
    "fig-sec10-occupations", "fig-sec11-golden-rule", "fig-05-matrix-element", "fig-sec13-born-cross-section",
    "fig-sec14-angular-relaxation", "fig-sec15-short-range", "fig-06-scattering", "fig-sec17-acoustic-phonons",
    "fig-sec18-energy-average", "fig-07-charge", "fig-sec20-exercise-input",
)
LAB_IDS = ["drude-" + name for name in (
    "step", "ac", "bloch", "packet", "band", "occupation", "transition",
    "gaussian", "angle", "impurity", "scattering", "phonon", "average", "charge", "quiz",
)]


def inline_math(text):
    """Count inline formulas outside fenced code and display-math blocks."""
    prose = re.sub(r"^```[^\n]*\n.*?^```[ \t]*$", "", text, flags=re.M | re.S)
    prose = re.sub(r"\$\$.*?\$\$", "", prose, flags=re.S)
    return Counter(re.findall(r"(?<![\\$])\$(?!\$)(.+?)(?<!\\)\$(?!\$)", prose, flags=re.S))


def cell_text(source):
    # Both strings and arrays of lines are valid notebook JSON sources/text.
    assert isinstance(source, (str, list))
    assert not isinstance(source, list) or all(isinstance(line, str) for line in source)
    return source if isinstance(source, str) else "".join(source)


def check_notebook_markdown(notebook, article, language):
    """Compare every cell with the article after explicit notebook conversions.

    This independently applies the documented link/figure/fold conversions;
    it does not import the builder or trust a notebook's previous content.
    """
    _, front, body = article.split("---", 2)
    title = json.loads(re.search(r'^title: (".*")$', front, re.M)[1])
    definitions = dict(re.findall(r"^\[([^\]]+)\]: (\S+)\s*$", body, re.M))
    body = re.sub(r"\[([^\]]+)\]\[([^\]]+)\]", lambda match: f"[{match[1]}]({definitions[match[2]]})", body)
    body = re.sub(r"^\[[^\]]+\]: \S+\s*$", "", body, flags=re.M)
    body = body.replace("{{ site.baseurl }}", SITE)
    body = re.sub(r"^\{:[^\n}]*\}[ \t]*$", "", body, flags=re.M)
    body = re.sub(r" · \[[^\]\n]+\]\([^)\n]*/assets/notebooks/drude-model-(?:en|ko)\.ipynb\)", "", body)

    def embedded_figure(match):
        fragment = match[0]
        image = re.search(r'<img src="([^"]+)"', fragment)[1]
        alt = re.search(r'<img[^>]* alt="([^"]+)"', fragment)[1]
        caption = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', fragment, re.S)[1]
        return f"\n![{alt}](attachment:{Path(image).stem}.png)\n\n*{caption}*\n"

    body = re.sub(r'<figure class="drude-figure">.*?</figure>', embedded_figure, body, flags=re.S)
    suffix = "-KR" if language == "ko" else ""
    lab_label = "Interactive example 열기" if language == "ko" else "Open interactive example"
    body = re.sub(r'\{% include interactive/article-lab\.html id="([^"]+)" %\}',
                  lambda match: f"\n[{lab_label}]({SITE}/posts/Drude-Model-From-Scratch{suffix}/#lab-{match[1]})\n", body)
    body = body.replace('<details markdown="1">', "").replace("</details>", "")
    body = re.sub(r"<summary>(.*?)</summary>", r"\n**\1**\n", body)
    introduction = ("Python 3.9+와 NumPy가 필요하다. Cell을 위에서부터 실행하면 된다. 그림은 notebook에 포함되어 있고, "
                    "interactive example 링크는 글의 해당 예제로 연결된다. 연습문제 starter는 reference solution과 이름이 다르다."
                    if language == "ko" else
                    "Requires Python 3.9+ and NumPy. Run cells in order. Figures are embedded; interactive links open the article labs. "
                    "The coding-exercise starter and reference solution have different names.")
    body = f"# {title}\n\n{introduction}\n\n" + body.lstrip()
    expected = [("code" if index % 2 else "markdown", source.strip() + "\n")
                for index, source in enumerate(re.split(r"^```python\n(.*?)^```[ \t]*$", body, flags=re.M | re.S))
                if source.strip()]
    actual = [(cell["cell_type"], cell_text(cell["source"])) for cell in notebook["cells"]]
    assert actual == expected, f"Notebook {language}: article Markdown/code content is stale"
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids)), f"Notebook {language}: duplicate cell IDs"
    return sum(kind == "markdown" for kind, source in expected)


def check_notebook_attachments(notebook, image_paths):
    expected = {Path(path).stem + ".png": ROOT / Path(path.lstrip("/")).with_suffix(".png")
                for path in image_paths}
    assert len(expected) == 20, "Each section needs a uniquely named figure attachment"
    seen = Counter()
    for cell in notebook["cells"]:
        source = cell_text(cell["source"])
        references = set(re.findall(r"attachment:([^\s)]+)", source))
        attached = cell.get("attachments", {})
        assert references == set(attached), "Notebook has an unresolved or unused attachment"
        assert not attached or cell["cell_type"] == "markdown", "Only Markdown cells may own attachments"
        for name, formats in attached.items():
            assert name in expected, f"Unexpected notebook attachment: {name}"
            assert set(formats) == {"image/png"}, f"Unexpected attachment MIME types: {name}"
            data = base64.b64decode(formats["image/png"], validate=True)
            png = expected[name].read_bytes()
            assert data == png, (f"Notebook attachment differs from checked-in PNG: {name}; "
                                 f"attachment SHA256={hashlib.sha256(data).hexdigest()}, "
                                 f"PNG SHA256={hashlib.sha256(png).hexdigest()}")
            seen[name] += 1
    assert seen == Counter({name: 1 for name in expected}), "Each figure must be embedded exactly once"


def replay_notebook(notebook, language):
    """Independent Run All: never call or import the builder's execute()."""
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(cell_text(cell["source"]), f"drude-{language}:cell-{count}", "exec"), namespace)
        expected = [] if not stdout.getvalue() else [{"output_type": "stream", "name": "stdout", "text": stdout.getvalue()}]
        outputs = []
        for output in cell["outputs"]:
            normalized = dict(output)
            if "text" in normalized:
                normalized["text"] = cell_text(normalized["text"])
            outputs.append(normalized)
        assert outputs == expected, f"Notebook {language} cell {count}: stored stdout is stale"
        assert not stderr.getvalue(), f"Notebook {language} cell {count}: unexpected stderr"
    assert namespace["check_angular_rates"](namespace["angular_relaxation_rates_reference"]) == "4 cases passed"
    assert namespace["angular_relaxation_rates"] is not namespace["angular_relaxation_rates_reference"]
    try:
        namespace["angular_relaxation_rates"](None, None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError(f"Notebook {language}: Run All overwrote the exercise starter")
    return count


def structure(text):
    return {
        "math": re.findall(r"\$\$.*?\$\$", text, re.S),
        "inline_math": inline_math(text),
        "code": re.findall(r"^```python\n(.*?)^```", text, re.M | re.S),
        "labs": re.findall(r'include interactive/article-lab\.html id="([^"]+)"', text),
        "images": re.findall(r'<img src="([^"]+)"', text),
        "mobile": re.findall(r"src='([^']+)'", text),
        "refs": re.findall(r"^\[([^\]]+)\]: (\S+)$", text, re.M),
        "ids": re.findall(r'\{:\s*id="([^"]+)"\s*\}', text),
        "h2_ids": re.findall(r'^## [^\n]+\n\s*\{:\s*id="([^"]+)"\s*\}', text, re.M),
    }


def main():
    texts = [(ROOT / f"_posts/2026-4Q/{STEM}{suffix}.md").read_text()
             for suffix in ("", "-KR")]
    en, ko = [structure(t) for t in texts]
    assert en["inline_math"] == ko["inline_math"], ("Paired inline math diverged: "
        f"EN-only {en['inline_math'] - ko['inline_math']}; KR-only {ko['inline_math'] - en['inline_math']}")
    assert en == ko, "Paired equations, code, labs, images, references or anchors diverged"
    assert en["ids"] == en["h2_ids"] == COMMON_ANCHORS, "All 21 common anchors must appear exactly once on their H2 headings"
    assert len(en["math"]) == 92 and len(en["code"]) == 16
    assert all("'" not in formula for formula in en["inline_math"]), "Use explicit TeX prime commands; smart quotes can alter inline math"
    assert all(r"\begin{pmatrix}" not in formula for formula in en["inline_math"]), "Put row-separated matrices in display math; Markdown can consume backslashes"
    assert en["labs"] == LAB_IDS
    assert len(en["images"]) == len(en["mobile"]) == 20
    prefix = "/assets/img/posts/2026-11-07-drude-model/"
    assert en["images"] == [prefix + stem + ".svg" for stem in FIGURE_STEMS]
    assert en["mobile"] == [prefix + stem + "-mobile.svg" for stem in FIGURE_STEMS]
    assert len(en["refs"]) == 19
    assert re.findall(r"\\tag\{(\d+)\}", texts[0]) == [str(i) for i in range(1, 66)]
    for snippet in en["code"]:
        ast.parse(snippet, feature_version=(3, 9))
    for text in texts:
        assert re.findall(r"^## (\d+)\.", text, re.M) == [str(i) for i in range(1, 21)]
        assert text.count('<details markdown="1">') == text.count("</details>") == 23
        assert text.count("$$") == 184
        assert "date: 2026-11-07 18:00:00 +0900" in text
        assert "translation_key: drude-model-from-scratch" in text
        assert "topic: quantum-transport" in text
        assert "interactive_labs: [drude, drude-foundations, drude-quantum, drude-microscopic]" in text
        assert "{{" not in text.replace("{{ site.baseurl }}", ""), "Unexpected Liquid opening braces; check nested math grouping"
        assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text)
        assert "$m^*$" not in text, "Use m^{\\ast} to avoid Markdown emphasis"
        assert "\\{" not in text and "\\}" not in text
        prose = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
        assert not any("|" in line and not line.startswith("|") for line in prose.splitlines()), "Avoid raw pipes in math prose"
        assert re.findall(r'<figcaption[^>]*>(?:Figure|그림) (\d+)\.', text) == [str(i) for i in range(1, 21)]
        sections = list(re.finditer(r'^## (\d+)\.[^\n]+$', text, re.M))
        for index, heading in enumerate(sections):
            stop = sections[index+1].start() if index+1 < len(sections) else text.index('\n## ', heading.end())
            section = text[heading.end():stop]
            assert section.count('<figure class="drude-figure">') == 1, f"Section {index+1}: expected exactly one explanatory figure"
            before_figure = section.split('<figure class="drude-figure">', 1)[0]
            assert before_figure.count('<details markdown="1">') == before_figure.count('</details>'), \
                f"Section {index+1}: essential figure must not be hidden in a fold"
        for tag in re.findall(r'<(?:img|source)\b[^>]+>', text):
            reference = re.search(r'/assets/img/posts/2026-11-07-drude-model/[^\s\x27\x22]+\.svg', tag)
            if reference:
                svg = ET.parse(ROOT / reference[0].lstrip('/')).getroot()
                assert all(f'{key}="{svg.attrib[key]}"' in tag for key in ('width', 'height')), \
                    f"Missing/mismatched intrinsic figure dimensions: {tag}"
        assert r"\sum_{b\ne a}" in text and r"qL_{\rm cell}=2\pi" in text
    for image in list((ROOT / "assets/img/posts/2026-11-07-drude-model").glob("*.svg")):
        svg = ET.parse(image).getroot()
        ns = "{http://www.w3.org/2000/svg}"
        assert svg.find(ns + "title") is not None and svg.find(ns + "desc") is not None
        ids = [e.attrib["id"] for e in svg.iter() if "id" in e.attrib]
        assert len(ids) == len(set(ids))
        for element in svg.iter():
            for key, value in element.attrib.items():
                if key.endswith("href") and value.startswith("#"):
                    assert value[1:] in ids
        png = image.with_suffix(".png").read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", png[16:24]) == (int(svg.attrib["width"]), int(svg.attrib["height"]))
    replayed, markdown_cells = 0, 0
    for language, text in zip(("en", "ko"), texts):
        n = json.loads((ROOT / f"assets/notebooks/drude-model-{language}.ipynb").read_text())
        markdown_cells += check_notebook_markdown(n, text, language)
        code = [c for c in n["cells"] if c["cell_type"] == "code"]
        assert [cell_text(c["source"]).strip() for c in code] == [c.strip() for c in en["code"]]
        assert [c["execution_count"] for c in code] == list(range(1, len(code)+1))
        assert not any(o["output_type"] == "error" for c in code for o in c["outputs"])
        check_notebook_attachments(n, en["images"])
        assert sum("exercise" in c["metadata"].get("tags", []) for c in code) == 1
        assert all("{%" not in cell_text(c["source"]) and "{{" not in cell_text(c["source"]) for c in n["cells"])
        replayed += replay_notebook(n, language)
    print(json.dumps({"status": "PASS", "articles": 2, "sections": 20,
                      "numbered_equations": 65, "display_math": len(en["math"]), "python_cells": len(en["code"]),
                      "folds": 23,
                      "inline_math_occurrences": sum(en["inline_math"].values()), "common_anchors": 21,
                      "labs": len(LAB_IDS), "figure_pairs": 20, "visually_supported_sections": 20, "references": 19, "notebooks": 2,
                      "exact_markdown_cells": markdown_cells,
                      "replayed_code_cells": replayed, "byte_exact_notebook_attachments": 40}))


if __name__ == "__main__":
    main()
