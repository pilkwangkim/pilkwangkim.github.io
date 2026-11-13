#!/usr/bin/env python3
"""Build executed standalone EN/KR notebooks from the actual MS-BTE snippets.

Use a Python 3.9+ interpreter with NumPy. Figures are PNG attachments; no network
is required to execute the notebook. Browser-only labs link to the paired post.
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
STEM = "2026-11-06-Multi-Subband-Boltzmann-Transport-From-Scratch"
ROUTE = "Multi-Subband-Boltzmann-Transport-From-Scratch"
SITE = "https://pilkwangkim.github.io"


def notebook_cells(language, *, render_figures=True):
    """Build cells, optionally reusing checked-in PNGs for read-only checks."""
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
    body = re.sub(r" · \[(?:Executable notebook|실행용 notebook)\].*?(?=\n|$)", "", body)
    attachments = {}

    def figure(match):
        fragment = match[0]
        src = re.search(r'<img src="([^"]+)"', fragment)[1]
        alt = re.search(r'<img[^>]* alt="([^"]+)"', fragment)[1]
        caption = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', fragment, re.S)[1]
        path = ROOT / src.lstrip("/")
        png = (subprocess.check_output(["rsvg-convert", "--format", "png", str(path)])
               if render_figures else path.with_suffix(".png").read_bytes())
        name = path.stem + ".png"
        attachments[name] = {"image/png": base64.b64encode(png).decode("ascii")}
        return f"\n![{alt}](attachment:{name})\n\n*{caption}*\n"

    body = re.sub(r'<figure class="msbte-figure">.*?</figure>', figure, body, flags=re.S)
    body = re.sub(r'\{% include interactive/article-lab\.html id="([^"]+)" %\}',
                  lambda m: f"\n[{'Interactive example 열기' if language == 'ko' else 'Open interactive example'}]"
                  f"({SITE}/posts/{ROUTE}{suffix}/#lab-{m[1]})\n", body)
    body = body.replace('<details markdown="1">', "").replace("</details>", "")
    body = re.sub(r"<summary>(.*?)</summary>", r"\n**\1**\n", body)
    note = ("Python 3.9+와 NumPy가 필요하다. Cell을 위에서부터 실행하면 된다. 그림은 notebook에 들어 있으며, "
            "browser interactive example은 글의 해당 위치로 연결된다. 연습문제 starter와 reference solution의 함수 이름은 다르다."
            if language == "ko" else
            "Requires Python 3.9+ and NumPy. Run cells in order. Figures are embedded; browser labs link to the article. "
            "The exercise starter and reference solution have different names, so Run All does not overwrite your answer.")
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
            if "def elastic_collision(f, v, K):" in source:
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
            exec(compile(cell["source"], "<msbte-notebook>", "exec"), namespace)
        count += 1
        cell["execution_count"] = count
        if stream.getvalue():
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": stream.getvalue()}]
    assert namespace["check_collision"](namespace["elastic_collision_reference"]) == "3 cases passed"
    # The exercise contract forbids mutating *any* input. Expectations must
    # not be recomputed from a candidate's already-mutated K or v.
    def mutates_input(index):
        def candidate(f, v, K):
            (f, v, K)[index][:] *= 2
            return namespace["elastic_collision_reference"](f, v, K)
        return candidate
    for index in range(3):
        try:
            namespace["check_collision"](mutates_input(index))
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Exercise checker accepted mutation of input {index}")
    try:
        namespace["elastic_collision"](None, None, None)
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
        destination = ROOT / f"assets/notebooks/multi-subband-bte-{language}.ipynb"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"language": language, "executed_cells": executed,
                          "embedded_figures": sum(len(c.get("attachments", {})) for c in cells)}))


if __name__ == "__main__":
    main()
