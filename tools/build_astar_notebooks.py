#!/usr/bin/env python3
"""Build self-contained, executed EN/KO notebooks from the paired A* articles.

The article is the prose source of truth. Python fences become executable cells;
desktop figures become embedded PNG attachments. Browser-only labs are replaced
by a pointer to the matching section, never by a claim that the notebook has a
JavaScript runtime. Every Python cell executes in one fresh shared namespace.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import html
import io
import json
import platform
import pprint
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / "_posts/2026-4Q"
STEM = "2026-10-24-A-Star-From-Scratch"
ASSETS = ROOT / "assets/img/posts/2026-10-24-astar-from-scratch"
EXERCISES = ROOT / "assets/data/astar-exercises.json"
LABS = {
    "astar-traversal": ("DFS/BFS traversal lab", "DFS/BFS 탐색 실습"),
    "astar-weighted": ("Editable weighted-map lab", "Weighted map 수정 실습"),
    "astar-priority-lab": ("Priority-expression exercise", "Priority 식 작성 실습"),
    "astar-quiz": ("Interactive knowledge check", "개념 확인 퀴즈"),
    "astar-code-test": ("Complete the A* core", "A*의 핵심 채우기"),
    "search-dfs-maze": ("DFS maze builder", "DFS 미로 만들기"),
    "search-dfs-queens": ("N-Queens backtracking", "N-Queens backtracking"),
    "search-bfs-words": ("Word-ladder challenge", "Word ladder 퍼즐"),
    "search-bfs-jugs": ("Water-jug challenge", "물통 퍼즐"),
    "search-dijkstra-delivery": ("Delivery-route challenge", "배달 경로 퍼즐"),
    "search-dijkstra-dispatch": ("Multi-source dispatch", "여러 station에서 출동하기"),
    "search-astar-detective": ("Heuristic detective", "Heuristic 탐정"),
    "search-astar-tiles": ("Sliding-tile puzzle", "Sliding tile puzzle"),
}


def exercise_code(data: dict) -> dict[str, str]:
    """One Python scaffold and the exact same fixtures as the browser judge."""
    starter = "\n".join(data["coding"]["starter"]) + "\n"
    cases = [{key: value for key, value in case.items()
              if key in ("id", "graph", "expected", "maxExpanded")}
             for case in data["coding"]["cases"]]
    fixtures = "CODE_CASES = " + pprint.pformat(cases, width=88, sort_dicts=False) + "\n"
    harness = '''
def materialize_code_case(case):
    spec = case["graph"]
    if "grid" in spec:
        width, height = spec["grid"]["width"], spec["grid"]["height"]
        graph = {}
        for u in range(width * height):
            x, y = u % width, u // width
            graph[u] = [(ny * width + nx, 1)
                        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1))
                        for nx, ny in [(x + dx, y + dy)]
                        if 0 <= nx < width and 0 <= ny < height]
        tx, ty = spec["goal"] % width, spec["goal"] // width
        h = {u: abs(u % width - tx) + abs(u // width - ty) for u in graph}
    else:
        graph = {u: [tuple(edge) for edge in edges]
                 for u, edges in enumerate(spec["edges"])}
        h = dict(enumerate(spec["h"]))
    return graph, spec["start"], spec["goal"], h


def run_astar_code_tests(solver, verbose=True):
    # Optional stats are maintained by the supplied scaffold, not a timer.
    results = []
    for case in CODE_CASES:
        graph, start, goal, h = materialize_code_case(case)
        stats = {}
        cost, error = None, None
        try:
            cost = solver(graph, start, goal, h, stats=stats)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        cost_ok = error is None and cost == case["expected"]
        effort_ok = error is None and ("maxExpanded" not in case or
                     stats.get("expanded", float("inf")) <= case["maxExpanded"])
        result = {"id": case["id"], "cost": cost, "expected": case["expected"],
                  "cost_passed": cost_ok, "effort_passed": effort_ok,
                  "passed": cost_ok and effort_ok, "error": error, **stats}
        results.append(result)
        if verbose:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{case['id']}: {status}; cost={cost!r}; "
                  f"expected={case['expected']!r}; "
                  f"expanded={stats.get('expanded', '?')}" +
                  (f"; budget={case['maxExpanded']}" if "maxExpanded" in case else "") +
                  (f"; {error}" if error else ""))
    return results
'''.lstrip()
    reference = (starter.replace("priority(", "reference_priority(")
                 .replace("def astar_cost(", "def reference_astar_cost(")
                 .replace("TODO_PRIORITY", "g + h")
                 .replace("TODO_RELAXATION", "candidate < old_g")
                 .replace("TODO_GOAL_CHECK", "pop"))
    reference += "\nreference_results = run_astar_code_tests(reference_astar_cost)\n"
    reference += "assert all(row['passed'] for row in reference_results)\n"
    return {"starter": starter, "tests": fixtures + "\n" + harness,
            "reference": reference}


def exercise_markdown(data: dict, language: str, attachments: dict,
                      coding=False) -> str:
    def picture(spec):
        path = ROOT / spec["png"].lstrip("/")
        attachments[path.name] = {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}
        return (f"![{spec['alt'][language]}](attachment:{path.name})\n\n"
                f"*{spec['caption'][language]}*\n")

    if not coding:
        parts = []
        for question in data["questions"]:
            parts.extend([f"#### {question['title'][language]}",
                          question["prompt"][language], picture(question["image"])])
            parts.append("\n".join(f"{i + 1}. {option[language]}"
                                   for i, option in enumerate(question["options"])))
            hint, answer = ("힌트", "정답·해설") if language == "ko" else ("Hint", "Answer and reasoning")
            parts.extend([f"**{hint}.** {question['hint'][language]}",
                          f"**{answer}: {question['answer'] + 1}.** {question['explanation'][language]}",
                          question["followup"][language]])
        return "\n\n".join(parts) + "\n"

    code = exercise_code(data)
    if language == "ko":
        instruction = ("아래 함수의 TODO_PRIORITY, TODO_RELAXATION, TODO_GOAL_CHECK를 채운 뒤 "
                       "다음 cell을 실행하자. Stats에는 expansion과 heap pop 횟수 등을 기록한다. "
                       "TODO가 남아 있으면 테스트가 FAIL로 표시되는 것이 정상이다. "
                       "코드 전체를 직접 수정할 수 있으며, reference solution은 마지막 별도 cell에 있다.")
        reference_label = "Reference solution — 먼저 직접 풀어 본 뒤 확인"
    else:
        instruction = ("Fill TODO_PRIORITY, TODO_RELAXATION and TODO_GOAL_CHECK, then run the next cell. "
                       "The scaffold records stats for the judge. An untouched TODO starter is expected "
                       "to report FAIL, not a solved exercise. You can edit the entire function; "
                       "a separate reference cell follows the tests.")
        reference_label = "Reference solution — try the exercise before opening"
    return (picture(data["coding"]["image"]) + "\n" + instruction + "\n\n"
            "```python\n" + code["starter"] + "```\n\n"
            "```python\n" + code["tests"] + "\n"
            "exercise_results = run_astar_code_tests(astar_cost)\n```\n\n"
            f"<details markdown=\"1\"><summary>{reference_label}</summary>\n\n"
            "```python\n" + code["reference"] + "```\n\n</details>\n")


def article_cells(language: str) -> list[dict]:
    suffix = "-KR" if language == "ko" else ""
    text = (POSTS / f"{STEM}{suffix}.md").read_text(encoding="utf-8")
    text = text.split("---", 2)[2].lstrip()
    # The downloadable notebook is already open; its reciprocal language link
    # remains useful, but the self-link is not an instruction to download again.
    text = "\n".join(line.split(" · ")[0] if line.startswith("[Read in ") else line
                     for line in text.splitlines())
    current_notebook = f"{{{{ site.baseurl }}}}/assets/notebooks/astar-from-scratch-{language}.ipynb"
    text = re.sub(r"\[[^\]]+\]\(" + re.escape(current_notebook) + r"\)",
                  "이 notebook" if language == "ko" else "this notebook", text)
    text = text.replace("{{ site.baseurl }}", "https://pilkwangkim.github.io")
    # Kramdown's table-style attribute is for the blog, not notebook Markdown.
    text = re.sub(r"(?m)^\{:[ \t]*\.astar-worked-table[ \t]*\}[ \t]*$", "", text)
    # Preserve existing blog anchors when headings are renamed, but keep the
    # Kramdown-only explicit-ID syntax out of Jupyter Markdown.
    text = re.sub(r'(?m)^\{:[ \t]*id="[^"\n]+"[ \t]*\}[ \t]*$', "", text)
    text = re.sub(r'(?m)^<span id="[^"\n]+"></span>\s*$', "", text)
    text = re.sub(r'<div class="astar-figure-mobile">.*?</div>\s*', "", text)
    attachments = {}
    exercises = json.loads(EXERCISES.read_text(encoding="utf-8"))

    def figure(match):
        tag = match.group(0)
        source = re.search(r'src="([^"]+)"', tag).group(1)
        alt = html.unescape(re.search(r'alt="([^"]*)"', tag).group(1))
        filename = Path(source).stem + ".png"
        attachments[filename] = {"image/png": base64.b64encode(
            (ASSETS / filename).read_bytes()).decode("ascii")}
        return f"\n![{alt}](attachment:{filename})\n"

    text = re.sub(r'<div class="astar-figure-desktop">.*?</div>', figure, text)

    def lab(match):
        lab_id = match.group(1)
        if lab_id in ("astar-quiz", "astar-code-test"):
            return exercise_markdown(exercises, language, attachments,
                                     coding=lab_id == "astar-code-test")
        label = LABS[lab_id][language == "ko"]
        url = f"https://pilkwangkim.github.io/posts/A-Star-From-Scratch{suffix}/#lab-{lab_id}"
        if language == "ko":
            return (f"\n**{label}.** 직접 조작하고 탐색을 한 단계씩 진행하는 실습은 "
                    f"[블로그의 해당 실습]({url})에서 실행합니다. "
                    "아래 Python cell은 notebook 안에서 직접 수정하고 실행할 수 있습니다.\n")
        return (f"\n**{label}.** The step-by-step editable browser activity is "
                f"available in the [matching article section]({url}). "
                "The Python cells in this notebook can be edited and executed directly.\n")

    text = re.sub(r'{% include interactive/article-lab\.html id="([^"]+)" %}', lab, text)
    # Details cannot span different Jupyter cells: turn the optional material
    # into ordinary subheadings rather than leave a broken cross-cell fold.
    optional = r"\n#### 참고: \1\n" if language == "ko" else r"\n#### Optional — \1\n"
    text = re.sub(r'<details markdown="1">\s*<summary>(.*?)</summary>',
                  optional, text, flags=re.S)
    text = text.replace("</details>", "")
    parts = re.split(r"^```python\n(.*?)^```\s*$", text, flags=re.M | re.S)
    cells = []
    for index, source in enumerate(parts):
        source = source.strip() + "\n"
        if not source.strip():
            continue
        cell_id = hashlib.sha256(f"{language}:{index}:{source}".encode()).hexdigest()[:12]
        if index % 2:
            cells.append({"cell_type": "code", "id": cell_id, "metadata": {},
                          "source": source.splitlines(keepends=True),
                          "execution_count": None, "outputs": []})
            if "TODO_PRIORITY" in source:
                cells[-1]["metadata"]["tags"] = ["exercise-starter"]
            elif "def run_astar_code_tests(" in source:
                cells[-1]["metadata"]["tags"] = ["exercise-tests"]
            elif "def reference_astar_cost(" in source:
                cells[-1]["metadata"] = {"tags": ["exercise-solution"],
                                          "jupyter": {"source_hidden": True}}
        else:
            cell = {"cell_type": "markdown", "id": cell_id, "metadata": {},
                    "source": source.splitlines(keepends=True)}
            selected = {name: data for name, data in attachments.items()
                        if f"attachment:{name}" in source}
            if selected:
                cell["attachments"] = selected
            cells.append(cell)
    return cells


def execute(cells: list[dict]) -> int:
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        count += 1
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile("".join(cell["source"]), f"astar-cell-{count}", "exec"), namespace)
        cell["execution_count"] = count
        if stdout.getvalue():
            cell["outputs"] = [{"output_type": "stream", "name": "stdout",
                                "text": stdout.getvalue().splitlines(keepends=True)}]
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["en", "ko", "all"], default="all")
    args = parser.parse_args()
    output = ROOT / "assets/notebooks"
    output.mkdir(parents=True, exist_ok=True)
    code = exercise_code(json.loads(EXERCISES.read_text(encoding="utf-8")))
    challenge = ROOT / "assets/code/astar-code-test.py"
    challenge.parent.mkdir(parents=True, exist_ok=True)
    challenge.write_text(
        '#!/usr/bin/env python3\n'
        '"""A* guided challenge. Fill the three TODOs; run with Python 3.9+.\n\n'
        'Inputs: finite directed graph, nonnegative edge costs, admissible h,\n'
        'h(goal)=0. Return minimum cost or None; stats belong to the judge.\n'
        'The untouched starter deliberately does not pass.\n"""\n\n'
        + code["starter"] + "\n" + code["tests"]
        + '\nif __name__ == "__main__":\n'
        + '    results = run_astar_code_tests(astar_cost)\n'
        + '    raise SystemExit(0 if all(row["passed"] for row in results) else 1)\n',
        encoding="utf-8")
    for language in (["en", "ko"] if args.language == "all" else [args.language]):
        article_path = POSTS / f"{STEM}{'-KR' if language == 'ko' else ''}.md"
        article_source = article_path.read_text(encoding="utf-8")
        article_title = json.loads(re.search(r'^title: (".*")$', article_source, re.M).group(1))
        cells = article_cells(language)
        count = execute(cells)
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": platform.python_version()},
                "title": article_title,
                "article_language": language,
                "requirements": "Python 3.9+; standard library only. All figures are embedded.",
            },
            "nbformat": 4, "nbformat_minor": 5,
        }
        path = output / f"astar-from-scratch-{language}.ipynb"
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"notebook": str(path.relative_to(ROOT)), "cells": len(cells),
                          "executed_python_cells": count, "embedded_figures": sum(
                              len(c.get("attachments", {})) for c in cells),
                          "status": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
