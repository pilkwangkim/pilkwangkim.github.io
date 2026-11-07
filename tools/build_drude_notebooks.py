#!/usr/bin/env python3
"""Build executed standalone Drude notebooks from the paired post snippets.

Requires Python 3.9+, NumPy and rsvg-convert for embedded figure generation.
The resulting notebooks need only Python and NumPy; browser labs stay links.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-11-07-Drude-Model-From-Scratch"
ROUTE = "Drude-Model-From-Scratch"
SITE = "https://pilkwangkim.github.io"


def notebook_cells(language):
    suffix = "-KR" if language == "ko" else ""
    raw = (ROOT / f"_posts/2026-4Q/{STEM}{suffix}.md").read_text(encoding="utf-8")
    _, front, body = raw.split("---", 2)
    title = json.loads(re.search(r'^title: (".*")$', front, re.M)[1])
    references = dict(re.findall(r"^\[([^\]]+)\]: (\S+)\s*$", body, re.M))
    body = re.sub(r"\[([^\]]+)\]\[([^\]]+)\]",
                  lambda m: f"[{m[1]}]({references[m[2]]})", body)
    body = re.sub(r"^\[[^\]]+\]: \S+\s*$", "", body, flags=re.M)
    body = body.replace("{{ site.baseurl }}", SITE)
    body = re.sub(r"^\{:[^\n}]*\}[ \t]*$", "", body, flags=re.M)
    body = re.sub(r" · \[[^\]\n]+\]\([^)\n]*/assets/notebooks/drude-model-(?:en|ko)\.ipynb\)", "", body)
    attachments = {}

    def figure(match):
        fragment = match[0]
        src = re.search(r'<img src="([^"]+)"', fragment)[1]
        alt = re.search(r'<img[^>]* alt="([^"]+)"', fragment)[1]
        caption = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', fragment, re.S)[1]
        path = ROOT / src.lstrip("/")
        png = subprocess.check_output(["rsvg-convert", "--format", "png", str(path)])
        name = path.stem + ".png"
        attachments[name] = {"image/png": base64.b64encode(png).decode("ascii")}
        return f"\n![{alt}](attachment:{name})\n\n*{caption}*\n"

    body = re.sub(r'<figure class="drude-figure">.*?</figure>', figure, body, flags=re.S)
    body = re.sub(r'\{% include interactive/article-lab\.html id="([^"]+)" %\}',
                  lambda m: f"\n[{'Interactive example 열기' if language == 'ko' else 'Open interactive example'}]"
                  f"({SITE}/posts/{ROUTE}{suffix}/#lab-{m[1]})\n", body)
    body = body.replace('<details markdown="1">', "").replace("</details>", "")
    body = re.sub(r"<summary>(.*?)</summary>", r"\n**\1**\n", body)
    note = ("Python 3.9+와 NumPy가 필요하다. Cell을 위에서부터 실행하면 된다. 그림은 notebook에 포함되어 있고, "
            "interactive example 링크는 글의 해당 예제로 연결된다. 연습문제 starter는 reference solution과 이름이 다르다."
            if language == "ko" else
            "Requires Python 3.9+ and NumPy. Run cells in order. Figures are embedded; interactive links open the article labs. "
            "The coding-exercise starter and reference solution have different names.")
    body = f"# {title}\n\n{note}\n\n" + body.lstrip()
    parts = re.split(r"^```python\n(.*?)^```[ \t]*$", body, flags=re.M | re.S)
    cells = []
    for index, source in enumerate(parts):
        if not source.strip():
            continue
        code = index % 2 == 1
        cell = {"cell_type": "code" if code else "markdown",
                "id": hashlib.sha256(f"{language}:{index}:{source}".encode()).hexdigest()[:12],
                "metadata": {}, "source": source.strip() + "\n"}
        if code:
            cell.update(execution_count=None, outputs=[])
            if "def angular_relaxation_rates(rate_weights, cos_theta):" in source:
                cell["metadata"]["tags"] = ["exercise"]
        else:
            local = {name: data for name, data in attachments.items() if f"attachment:{name}" in source}
            if local:
                cell["attachments"] = local
        cells.append(cell)
    return cells


def execute(cells):
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(cell["source"], "<drude-notebook>", "exec"), namespace)
        count += 1
        cell["execution_count"] = count
        if stream.getvalue():
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": stream.getvalue()}]
    assert namespace["check_angular_rates"](namespace["angular_relaxation_rates_reference"]) == "4 cases passed"

    def mutates_input(index):
        def candidate(rates, cosines):
            (rates, cosines)[index][:] *= 0.5
            return namespace["angular_relaxation_rates_reference"](rates, cosines)
        return candidate

    for index in range(2):
        try:
            namespace["check_angular_rates"](mutates_input(index))
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Exercise checker accepted mutation of input {index}")
    try:
        namespace["angular_relaxation_rates"](None, None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Exercise starter was overwritten")
    return count


def main():
    import numpy
    for language in ("en", "ko"):
        cells = notebook_cells(language)
        executed = execute(cells)
        notebook = {"cells": cells, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
            "requirements": f"Python 3.9+; NumPy (verified with {numpy.__version__}); embedded figures."},
            "nbformat": 4, "nbformat_minor": 5}
        destination = ROOT / f"assets/notebooks/drude-model-{language}.ipynb"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"language": language, "executed_cells": executed,
                          "embedded_figures": sum(len(c.get("attachments", {})) for c in cells)}))


if __name__ == "__main__":
    main()
