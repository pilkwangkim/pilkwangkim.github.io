#!/usr/bin/env python3
"""Build executed, self-contained sorting notebooks from the paired articles.

Source edits belong in the articles. This generator embeds the desktop PNGs,
executes all Python snippets, and adds the same five questions and eight coding
fixtures as the browser bundle. Browser steppers remain links, not fake widgets.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import hashlib
import html
import io
import json
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-10-24-Sorting-From-Scratch"
POSTS = ROOT / "_posts/2026-4Q"
ASSETS = ROOT / "assets/img/posts/2026-10-24-sorting-from-scratch"
CODE_PATH = ROOT / "assets/code/sorting-code-test.py"
LABS = {
    "sorting-playground": ("Sorting workbench", "정렬 과정 직접 비교하기"),
    "sorting-random": ("Bounded bogo experiment", "Bogo sort 실험"),
    "sorting-inversions": ("Inversion challenge", "Inversion 없애기"),
    "sorting-merge": ("Stable merge challenge", "Stable merge 직접 해 보기"),
    "sorting-partition": ("Three-way partition", "3-way partition 실습"),
    "sorting-distribution": ("Counting and radix scatter", "Counting·radix 배치 과정"),
    "sorting-quiz": ("Five illustrated questions", "그림으로 풀어 보는 다섯 문제"),
    "sorting-code": ("Sort Colors coding exercise", "Sort Colors 코드 연습"),
}


def browser_content():
    code = ("const m=require('./assets/js/article-labs-sorting.js');"
            "process.stdout.write(JSON.stringify({quiz:m.QUIZ,cases:m.CODE_CASES}));")
    return json.loads(subprocess.check_output(["node", "-e", code], cwd=ROOT, text=True))


def code_sections():
    source = CODE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    found = {}
    for item in tree.body:
        if isinstance(item, ast.FunctionDef):
            found[item.name] = "".join(lines[item.lineno - 1:item.end_lineno])
        elif isinstance(item, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in ("CASES", "CASE_PURPOSES")
                for target in item.targets):
            name = next(target.id for target in item.targets if isinstance(target, ast.Name))
            found[name] = "".join(lines[item.lineno - 1:item.end_lineno])
    return found


def question_markdown(question, lang, number):
    def field(*names):
        for name in names:
            flat_name = name + ("Ko" if lang == "ko" else "En")
            if flat_name in question:
                return question[flat_name]
            if name in question:
                value = question[name]
                return value.get(lang, value.get("en", "")) if isinstance(value, dict) else value
        return ""
    options = question.get("options", question.get("choices", []))
    choices = []
    for index, value in enumerate(options):
        if isinstance(value, dict):
            value = value.get(lang, value.get("en", str(value)))
        elif isinstance(value, list):
            value = value[lang == "ko"]
        choices.append(f"{index + 1}. {value}")
    answer = question.get("answer", question.get("correct", 0))
    title = field('title') or question.get(lang, question.get("en", ""))
    input_values = question.get("input", [])
    labels = question.get("labels", [])
    diagram = ""
    if input_values:
        values = labels or input_values
        diagram = "`[ " + " | ".join(str(x) for x in values) + " ]`\n\n"
    option_reasons = question.get("optionReasons", [])
    reasoning = "\n".join(f"- {i + 1}. {reason[lang == 'ko']}"
                          for i, reason in enumerate(option_reasons))
    return (f"\n#### {number}. {title}\n\n"
            + ("**확인할 개념:** " if lang == "ko" else "**Concept:** ")
            + str(field("skill")) + f"\n\n{field('prompt', 'question')}\n\n" + diagram
            + "\n".join(choices) + "\n\n"
            + ("**힌트:** " if lang == "ko" else "**Hint:** ") + str(field("hint"))
            + "\n\n" + ("**정답·해설:** " if lang == "ko" else "**Answer and reasoning:** ")
            + f"{answer + 1}. {field('explanation', 'explain', 'reason')}\n\n"
            + reasoning + "\n")


def article_cells(lang, content):
    suffix = "-KR" if lang == "ko" else ""
    raw = (POSTS / f"{STEM}{suffix}.md").read_text(encoding="utf-8")
    title = json.loads(re.search(r'^title: (".*")$', raw, re.M).group(1))
    text = raw.split("---", 2)[2].lstrip()
    notebook_note = (
        "Python 3의 standard library만 있으면 실행할 수 있다. 그림도 notebook 안에 포함돼 있다. "
        "예제 cell을 위에서부터 차례로 실행하면 된다. 마지막 coding exercise의 starter에는 "
        "일부러 오류를 넣었다. 세 TODO를 모두 확인하고 잘못된 선택을 고친 뒤 테스트를 다시 실행해 보자. "
        "모든 테스트를 통과하는 reference solution은 별도 cell에 있다."
        if lang == "ko" else
        "Python 3 standard library only; all figures are embedded. Run the example cells "
        "in order. The final coding starter runs but deliberately fails some cases; "
        "edit its three TODOs. The separately labelled reference solution passes all tests."
    )
    text = f"# {title}\n\n{notebook_note}\n\n" + text
    # Keep the reciprocal-language link, but not this notebook's download link.
    text = "\n".join(line.split(" · ")[0] if line.startswith("[Read in ") else line
                     for line in text.splitlines())
    text = text.replace("{{ site.baseurl }}", "https://pilkwangkim.github.io")
    text = re.sub(r'<div class="sorting-figure-mobile">.*?</div>\s*', "", text, flags=re.S)
    text = re.sub(r'(?m)^\{:[^\n]*\}\s*$', "", text)
    text = re.sub(r' <span class="sorting-section-anchor" id="[^"]+"></span>', "", text)
    attachments = {}

    def picture(name, alt):
        filename = f"{name}.png"
        attachments[filename] = {"image/png": base64.b64encode(
            (ASSETS / filename).read_bytes()).decode("ascii")}
        return f"\n![{alt}](attachment:{filename})\n"

    def figure(match):
        tag = match.group(0)
        src = re.search(r'src="([^"]+)"', tag).group(1)
        alt = html.unescape(re.search(r'alt="([^"]*)"', tag).group(1))
        return picture(Path(src).stem, alt)

    text = re.sub(r'<div class="sorting-figure-desktop">.*?</div>', figure, text, flags=re.S)
    sections = code_sections()

    def lab(match):
        lab_id = match.group(1)
        if lab_id == "sorting-quiz":
            return "\n".join(question_markdown(question, lang, i + 1)
                             for i, question in enumerate(content["quiz"]))
        if lab_id == "sorting-code":
            instruction = (
                "아래 starter에는 세 가지 선택을 확인하는 TODO가 있으며, 일부는 일부러 틀리게 해 두었다. "
                "그대로 실행하면 일부 테스트에서 실패한다. TODO 1–3을 모두 확인하고 잘못된 선택을 고친 뒤 테스트 cell을 다시 실행해 보자. "
                "Counter와 iteration 제한은 이미 들어 있다. 원래 list를 직접 정렬하고 같은 list를 "
                "반환해야 한다. 테스트마다 실행 결과와 정답, 실패 이유를 확인할 수 있다. "
                "정답 코드는 별도의 reference solution cell에 있다."
                if lang == "ko" else
                "The starter is runnable, but some of its three decisions are wrong. "
                "Edit TODOs 1–3, then rerun the test cell. Counters and a finite iteration "
                "guard are already included. Mutate and return the original list. Each case "
                "prints actual/expected results and a diagnosis. A separate reference cell follows."
            )
            return (f"\n{instruction}\n\n```python\n{sections['sort_colors']}\n```\n\n"
                    f"```python\n{sections['CASES']}\n\n{sections['CASE_PURPOSES']}\n\n{sections['run_tests']}\n\n"
                    "starter_results = run_tests(sort_colors)\n```\n\n"
                    "#### Reference solution\n\n```python\n"
                    + sections["reference_sort_colors"] + "\n\n"
                    + "reference_results = run_tests(reference_sort_colors)\n"
                    + "assert all(row['passed'] for row in reference_results)\n```\n")
        label = LABS[lab_id][lang == "ko"]
        url = f"https://pilkwangkim.github.io/posts/Sorting-From-Scratch{suffix}/#lab-{lab_id}"
        return (f"\n**{label}:** ["
                + ("블로그에서 직접 조작하기" if lang == "ko" else "Open the browser activity")
                + f"]({url}).\n")

    text = re.sub(r'{% include interactive/article-lab\.html id="([^"]+)" %}', lab, text)
    text = re.sub(r'<details markdown="1">\s*<summary>(.*?)</summary>',
                  r"\n#### \1\n", text, flags=re.S).replace("</details>", "")
    parts = re.split(r"^```python\n(.*?)^```\s*$", text, flags=re.M | re.S)
    cells = []
    for index, source in enumerate(parts):
        source = source.strip() + "\n"
        if not source.strip():
            continue
        if index % 2 and source.startswith("def sort_colors(a):"):
            # Keep the later compact reference from overwriting the editable
            # instrumented starter when a reader runs all cells, then retries it.
            source = ("# This reference does not replace the editable sort_colors starter.\n"
                      + re.sub(r"\bsort_colors\(", "article_sort_colors(", source))
        cell = {"cell_type": "code" if index % 2 else "markdown",
                "id": hashlib.sha256(f"{lang}:{index}:{source}".encode()).hexdigest()[:12],
                "metadata": {}, "source": source.splitlines(keepends=True)}
        if index % 2:
            cell.update(execution_count=None, outputs=[])
            if "TODO 1:" in source:
                cell["metadata"]["tags"] = ["exercise-starter"]
            elif "starter_results =" in source:
                cell["metadata"]["tags"] = ["exercise-tests"]
            elif "def reference_sort_colors" in source:
                cell["metadata"] = {"tags": ["exercise-solution"],
                                    "jupyter": {"source_hidden": True}}
        else:
            selected = {name: image for name, image in attachments.items()
                        if f"attachment:{name}" in source}
            if selected:
                cell["attachments"] = selected
        cells.append(cell)
    return title, cells


def execute(cells):
    namespace = {"__name__": "__sorting_notebook__"}
    count = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        count += 1
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(compile("".join(cell["source"]), f"sorting-cell-{count}", "exec"), namespace)
        cell["execution_count"] = count
        if captured.getvalue():
            cell["outputs"] = [{"output_type": "stream", "name": "stdout",
                                "text": captured.getvalue().splitlines(keepends=True)}]
    # The exercise remains usable after Run All: retrying its tests must still
    # call the starter, rather than a later uninstrumented reference definition.
    retried = namespace["run_tests"](namespace["sort_colors"], False)
    assert [row["passed"] for row in retried] == [
        row["passed"] for row in namespace["starter_results"]]
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["en", "ko", "all"], default="all")
    args = parser.parse_args()
    content = browser_content()
    sections = code_sections()
    cases_namespace = {}
    exec(sections["CASES"], cases_namespace)
    # Accept either tuple fixtures or exported JS objects with id/input fields.
    js_cases = content["cases"]
    normalized = [(case.get("id", case.get("name")),
                   case.get("input", case.get("values", case.get("array"))))
                  if isinstance(case, dict) else tuple(case) for case in js_cases]
    assert normalized == cases_namespace["CASES"], "Python/browser coding fixtures drifted"
    for lang in (["en", "ko"] if args.language == "all" else [args.language]):
        title, cells = article_cells(lang, content)
        count = execute(cells)
        notebook = {"cells": cells, "metadata": {
            "title": title, "article_language": lang,
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
            "requirements": "Python 3.9+; standard library only; figures embedded."},
            "nbformat": 4, "nbformat_minor": 5}
        path = ROOT / f"assets/notebooks/sorting-from-scratch-{lang}.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"notebook": str(path.relative_to(ROOT)), "cells": len(cells),
                          "executed_python_cells": count, "embedded_figures": sum(
                              len(c.get("attachments", {})) for c in cells),
                          "status": "passed"}))
    # Full reference module has the exact article snippets, not another implementation.
    source = (POSTS / f"{STEM}.md").read_text(encoding="utf-8")
    snippets = re.findall(r"^```python\n(.*?)^```\s*$", source, flags=re.M | re.S)
    reference = ROOT / "assets/code/sorting-reference.py"
    reference.write_text('#!/usr/bin/env python3\n'
                         '"""Executed Python snippets from the sorting tutorial; standard library only."""\n\n'
                         + "\n\n".join(snippets) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
