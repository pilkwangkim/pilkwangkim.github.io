#!/usr/bin/env python3
"""Build executed, self-contained ECC notebooks from the paired Jekyll posts."""
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
STEM = "2026-11-05-Error-Correcting-Codes-From-Scratch"
POSTS = ROOT / "_posts/2026-4Q"
SITE = "https://pilkwangkim.github.io"
FENCE = re.compile(r"^```python\n(.*?)^```[ \t]*$", re.M | re.S)


def article(lang):
    suffix = "-KR" if lang == "ko" else ""
    source = (POSTS / f"{STEM}{suffix}.md").read_text(encoding="utf-8")
    _, front, body = source.split("---", 2)
    title = json.loads(re.search(r'^title: (".*")$', front, re.M)[1])
    return title, body, suffix


def cells_for(lang):
    title, text, suffix = article(lang)
    # Markdown references are resolved per notebook cell, not across the document.
    refs = dict(re.findall(r"^\[([^\]]+)\]: (\S+)\s*$", text, re.M))
    text = re.sub(r"\[([^\]]+)\]\[([^\]]+)\]", lambda m: f"[{m[1]}]({refs[m[2]]})", text)
    text = re.sub(r"^\[[^\]]+\]: \S+\s*$", "", text, flags=re.M)
    text = text.replace("{{ site.baseurl }}", SITE)
    # Jekyll block attributes (heading IDs and table classes) are not notebook
    # Markdown. Keep the associated heading/table, not the literal attribute.
    text = re.sub(r'^\{:[^\n}]*\}[ \t]*$', "", text, flags=re.M)
    # Browser lab anchors do not exist inside the standalone notebook.
    text = re.sub(r"\]\(#(lab-ecc-[a-z]+)\)",
                  lambda m: f"]({SITE}/posts/Error-Correcting-Codes-From-Scratch{suffix}/#{m[1]})", text)
    text = re.sub(r" · \[(?:Executable notebook|실행용 notebook)\].*?(?=\n|$)", "", text)
    pictures = {}

    def picture(match):
        fragment = match[0]
        path = re.search(r'<img src="([^"]+)"', fragment)[1]
        alt = re.search(r'<img[^>]* alt="([^"]+)"', fragment)[1]
        caption = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', fragment, re.S)[1]
        asset = ROOT / path.lstrip("/")
        png = subprocess.check_output(["rsvg-convert", "--format", "png", str(asset)])
        name = asset.stem + ".png"
        pictures[name] = {"image/png": base64.b64encode(png).decode("ascii")}
        return f"\n![{alt}](attachment:{name})\n\n*{caption}*\n"

    text = re.sub(r'<figure class="ecc-figure">.*?</figure>', picture, text, flags=re.S)
    text = re.sub(r'\{% include interactive/article-lab\.html id="([^"]+)" %\}',
                  lambda m: f"\n[{'Interactive example 열기' if lang == 'ko' else 'Open interactive example'}]"
                  f"({SITE}/posts/Error-Correcting-Codes-From-Scratch{suffix}/#lab-{m[1]})\n", text)
    # HTML details cannot span independently rendered notebook cells. Retain all
    # optional material as labelled subsections, rather than losing code in folds.
    text = text.replace('<details markdown="1">', "").replace("</details>", "")
    text = re.sub(r"<summary>(.*?)</summary>", r"\n**\1**\n", text)
    note = ("예제는 위에서부터 차례로 실행하면 된다. Python standard library만 사용하고, 그림 네 개도 notebook에 들어 있다. "
            "Coding exercise의 starter와 reference solution은 함수 이름이 달라서 Run All을 실행해도 작성 중인 풀이를 덮어쓰지 않는다."
            if lang == "ko" else
            "Run cells in order. The examples use only Python's standard library, and all four figures are embedded. "
            "The exercise starter and reference solution use different names, so Run All does not overwrite your answer.")
    text = f"# {title}\n\n{note}\n\n" + text.lstrip()
    parts = FENCE.split(text)
    cells = []
    for index, part in enumerate(parts):
        if not part.strip():
            continue
        is_code = index % 2 == 1
        cell = {"cell_type": "code" if is_code else "markdown",
                "id": hashlib.sha256(f"{lang}:{index}:{part}".encode()).hexdigest()[:12],
                "metadata": {}, "source": part.strip() + "\n"}
        if is_code:
            cell.update(execution_count=None, outputs=[])
            if "def decode_single(received)" in part:
                cell["metadata"]["tags"] = ["exercise"]
        else:
            attachments = {name: value for name, value in pictures.items() if f"attachment:{name}" in part}
            if attachments:
                cell["attachments"] = attachments
        cells.append(cell)
    assert sum(len(c.get("attachments", {})) for c in cells) == 4
    return cells


def execute(cells):
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(cell["source"], "<ecc-notebook>", "exec"), namespace)
        count += 1
        cell["execution_count"] = count
        if output.getvalue():
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output.getvalue()}]
    assert namespace["check_decoder"](namespace["decode_single_reference"]) == 128
    try:
        namespace["decode_single"]((0,) * 7)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Run All must not overwrite the learner's starter")
    return count


def main():
    for lang in ("en", "ko"):
        cells = cells_for(lang)
        count = execute(cells)
        notebook = {"cells": cells, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
            "requirements": "Python 3.9+; standard library only; all figures embedded."},
            "nbformat": 4, "nbformat_minor": 5}
        destination = ROOT / f"assets/notebooks/error-correcting-codes-{lang}.ipynb"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"language": lang, "cells": len(cells), "executed": count, "embedded_figures": 4}))


if __name__ == "__main__":
    main()
