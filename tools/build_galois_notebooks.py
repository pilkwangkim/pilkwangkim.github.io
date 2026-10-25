#!/usr/bin/env python3
"""Build executed, self-contained Galois notebooks from the bilingual articles.

Run with a Python environment containing SymPy to execute the optional symbolic
cell as well as the standard-library examples. Source prose and Python snippets
belong in the articles, and quiz content belongs in the browser module.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import hashlib
import html
import importlib.util
import io
import json
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-10-25-Galois-Theory-From-Scratch"
POSTS = ROOT / "_posts/2026-4Q"
ASSETS = ROOT / "assets/img/posts/2026-10-25-galois-from-scratch"
SITE = "https://pilkwangkim.github.io"
LABS = {
    "galois-quadratic": ("Exact arithmetic in Q(√2)", "Q(√2)에서 정확히 계산하기"),
    "galois-permutations": ("Allowed root permutations", "가능한 근의 permutation 확인하기"),
    "galois-correspondence": ("Subgroups and fixed fields", "Subgroup과 fixed field 찾기"),
    "galois-derived": ("Derived-series workbench", "Derived series 직접 계산하기"),
    "galois-quintic": ("Numerical roots and radical formulas", "근삿값과 radical 공식 비교하기"),
    "galois-quiz": ("Five illustrated questions", "그림과 함께 푸는 다섯 문제"),
}
CODE_FENCE = re.compile(r"^```python\n(.*?)^```[ \t]*$", re.M | re.S)


def browser_quiz():
    """Read the actual browser questions without executing any DOM factory."""
    command = ("const m=require('./assets/js/article-labs-galois.js');"
               "process.stdout.write(JSON.stringify(m.QUIZ));")
    questions = json.loads(subprocess.check_output(["node", "-e", command], cwd=ROOT, text=True))
    assert len(questions) == 5, "The article promises five quiz questions"
    for question in questions:
        assert len(question["options"]) == len(question["optionReasons"]) == 3
        assert 0 <= question["answer"] < 3
        assert question["diagram"], "Every quiz input needs a diagram description"
        for key in ("promptEn", "promptKo", "hintEn", "hintKo", "en", "ko"):
            assert question[key]
        assert all(len(pair) == 2 and all(pair) for pair in question["options"] + question["optionReasons"])
    return questions


def localized(value, lang):
    if isinstance(value, dict):
        return value[lang]
    if isinstance(value, list):
        return value[lang == "ko"]
    return value


def question_markdown(question, lang, number):
    """Export input, all choices, hint, answer and every option's rationale."""
    index = lang == "ko"
    prompt = question["promptKo" if index else "promptEn"]
    hint = question["hintKo" if index else "hintEn"]
    diagram_heading = "그림에 주어진 조건" if index else "Diagram input"
    diagram_lines = "\n".join(f"- `{localized(line, lang)}`" for line in question["diagram"])
    choices = "\n".join(f"{i + 1}. {localized(option, lang)}"
                        for i, option in enumerate(question["options"]))
    answer = question["answer"]
    answer_label = "정답" if index else "Answer"
    reason_label = "보기별 해설" if index else "Reasoning for every choice"
    reasons = "\n".join(f"- **{i + 1}.** {localized(reason, lang)}"
                        for i, reason in enumerate(question["optionReasons"]))
    return (f"\n#### {number}. {question[lang]}\n\n{prompt}\n\n"
            f"**{diagram_heading}:**\n\n{diagram_lines}\n\n{choices}\n\n"
            + ("**힌트:** " if index else "**Hint:** ") + hint + "\n\n"
            + f"**{answer_label}: {answer + 1}.** {localized(question['optionReasons'][answer], lang)}\n\n"
            + f"**{reason_label}:**\n\n{reasons}\n")


def read_article(lang):
    suffix = "-KR" if lang == "ko" else ""
    path = POSTS / f"{STEM}{suffix}.md"
    source = path.read_text(encoding="utf-8")
    pieces = source.split("---", 2)
    assert len(pieces) == 3 and not pieces[0].strip(), "Missing YAML front matter"
    title_match = re.search(r'^title: (".*")$', pieces[1], re.M)
    assert title_match, "Use a quoted article title"
    title = json.loads(title_match.group(1))
    snippets = [match.group(1).rstrip() + "\n" for match in CODE_FENCE.finditer(pieces[2])]
    assert snippets, "No Python examples found"
    for number, snippet in enumerate(snippets, 1):
        ast.parse(snippet, filename=f"{lang}-galois-example-{number}")
    return {"source": source, "body": pieces[2].lstrip(), "title": title,
            "snippets": snippets, "suffix": suffix}


def article_cells(lang, article, questions):
    is_ko = lang == "ko"
    note = (
        "예제 code cell을 위에서부터 차례로 실행하면 된다. 그림 10개는 notebook 안에 포함돼 있다. "
        "핵심 계산에는 Python standard library만 사용한다. 별도로 표시한 symbolic 계산에는 SymPy가 필요하다. "
        "Interactive 조작은 각 절의 블로그 링크에서 할 수 있다. Coding exercise의 starter는 직접 완성한 뒤 "
        "주석 처리된 테스트 호출을 실행해 보자. Reference solution은 다른 이름으로 정의하므로 Run All을 해도 starter가 바뀌지 않는다."
        if is_ko else
        "Run the example code cells in order. All ten figures are embedded. Core calculations use "
        "Python's standard library; the separately labelled symbolic check requires SymPy. "
        "Each browser activity has a link beside its explanation. Complete the coding starter, "
        "then uncomment its test call. The reference solution uses a different name, so Run All "
        "does not overwrite the editable starter."
    )
    text = f"# {article['title']}\n\n{note}\n\n" + article["body"]
    # Download links point back to the article's own notebook; keep the language
    # link and separate implementations rather than a circular download link.
    text = "\n".join(" · ".join(part for part in line.split(" · ")
                                  if "/assets/notebooks/galois-from-scratch-" not in part)
                     if line.startswith("[Read in ") else line for line in text.splitlines())
    text = text.replace("{{ site.baseurl }}", SITE)
    text = re.sub(r'<div class="galois-figure-mobile">.*?</div>\s*', "", text, flags=re.S)
    text = re.sub(r'(?m)^\{:[^\n]*\}[ \t]*$', "", text)
    text = re.sub(r'\s*<span class="galois-section-anchor" id="[^"]+"></span>', "", text)
    attachments = {}

    def figure(match):
        tag = match.group(0)
        source_match = re.search(r'src="([^"]+)"', tag)
        alt_match = re.search(r'alt="([^"]*)"', tag)
        assert source_match and alt_match, "Figure needs src and alt"
        stem = Path(source_match.group(1)).stem
        assert not stem.endswith("-mobile")
        filename = stem + ".png"
        path = ASSETS / filename
        assert path.is_file(), f"Missing embedded figure: {path}"
        attachments[filename] = {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}
        return f"\n![{html.unescape(alt_match.group(1))}](attachment:{filename})\n"

    text = re.sub(r'<div class="galois-figure-desktop">.*?</div>', figure, text, flags=re.S)
    assert len(attachments) == 10, f"Expected ten figure attachments, found {len(attachments)}"
    labs_seen = []

    def lab(match):
        lab_id = match.group(1)
        assert lab_id in LABS, f"Unknown activity: {lab_id}"
        labs_seen.append(lab_id)
        label = LABS[lab_id][is_ko]
        url = f"{SITE}/posts/Galois-Theory-From-Scratch{article['suffix']}/#lab-{lab_id}"
        link = (f"\n**{label}:** [" + ("블로그에서 직접 조작하기" if is_ko else "Open the browser activity")
                + f"]({url}).\n")
        if lab_id == "galois-quiz":
            link += "\n" + "\n".join(question_markdown(question, lang, i + 1)
                                      for i, question in enumerate(questions))
        return link

    text = re.sub(r'{% include interactive/article-lab\.html id="([^"]+)" %}', lab, text)
    assert sorted(labs_seen) == sorted(LABS), "Each of the six activities must occur once"
    # Optional proofs remain full notebook text, without blog-only HTML wrappers.
    text = re.sub(r'<details(?: markdown="1")?>\s*<summary>(.*?)</summary>',
                  r"\n#### \1\n", text, flags=re.S).replace("</details>", "")
    assert not re.search(r"\{[{%]|galois-figure-(?:desktop|mobile)|<details|<summary|<img|</?div", text), \
        "Unconverted Liquid or responsive HTML in notebook"
    parts = CODE_FENCE.split(text)
    cells = []
    for index, source in enumerate(parts):
        source = source.strip() + "\n"
        if not source.strip():
            continue
        is_code = bool(index % 2)
        cell = {"cell_type": "code" if is_code else "markdown",
                "id": hashlib.sha256(f"galois:{lang}:{index}:{source}".encode()).hexdigest()[:12],
                "metadata": {}, "source": source.splitlines(keepends=True)}
        if is_code:
            cell.update(execution_count=None, outputs=[])
            if "NotImplementedError" in source:
                cell["metadata"]["tags"] = ["exercise-starter"]
            elif "def reference_" in source:
                cell["metadata"] = {"tags": ["exercise-solution"], "jupyter": {"source_hidden": True}}
            if re.search(r"(?:import sympy|from sympy)", source):
                cell["metadata"].setdefault("tags", []).append("optional-sympy")
        else:
            selected = {name: image for name, image in attachments.items() if f"attachment:{name}" in source}
            if selected:
                cell["attachments"] = selected
        cells.append(cell)
    actual_snippets = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    assert actual_snippets == [source.strip() + "\n" for source in article["snippets"]], \
        "Notebook code must be byte-identical to the article snippets"
    assert sum(len(cell.get("attachments", {})) for cell in cells) == 10
    return cells


def starter_functions(source):
    """Find deliberately incomplete functions without ever calling them."""
    names = []
    for item in ast.parse(source).body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(isinstance(node, ast.Name) and node.id == "NotImplementedError" for node in ast.walk(item)):
                names.append(item.name)
    return names


def execute(cells):
    namespace = {"__name__": "__galois_notebook__"}
    starters = {}
    count = 0
    optional_sympy = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        count += 1
        source = "".join(cell["source"])
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(compile(source, f"galois-cell-{count}", "exec"), namespace)
        cell["execution_count"] = count
        cell["outputs"] = ([] if not captured.getvalue() else [{
            "output_type": "stream", "name": "stdout", "text": captured.getvalue().splitlines(keepends=True)}])
        for name in starter_functions(source):
            starters[name] = namespace[name]
        optional_sympy += "optional-sympy" in cell["metadata"].get("tags", [])
    assert starters, "Keep a genuinely editable, incomplete coding starter"
    for name, original in starters.items():
        assert namespace[name] is original, f"Run All overwrote the editable starter {name}"
    assert optional_sympy >= 1, "Include and execute the optional SymPy example"
    return {"executed_code_cells": count, "optional_sympy_cells": optional_sympy,
            "editable_starter_functions": sorted(starters)}


def export_examples(articles):
    """One download contains exactly the executable article fences, in order."""
    snippets = articles["en"]["snippets"]
    if "ko" in articles:
        assert snippets == articles["ko"]["snippets"], "English/Korean Python examples differ"
    source = ('#!/usr/bin/env python3\n'
              '"""Python examples from the Galois tutorial, in article order.\n\n'
              'Core examples use the standard library; the optional symbolic cell uses SymPy.\n'
              'The incomplete coding starter is not called; its reference is separately named.\n'
              '"""\n\n'
              + "\n\n".join(snippet.rstrip() for snippet in snippets) + "\n")
    ast.parse(source)
    path = ROOT / "assets/code/galois-article-examples.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "snippets": len(snippets),
            "sha256": hashlib.sha256(source.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "ko", "all"), default="all")
    args = parser.parse_args()
    if importlib.util.find_spec("sympy") is None:
        raise SystemExit("Build with SymPy installed, e.g. /opt/anaconda3/bin/python tools/build_galois_notebooks.py")
    languages = ("en", "ko") if args.language == "all" else (args.language,)
    articles = {lang: read_article(lang) for lang in languages}
    # Always check the paired code if both articles exist, even for a one-language
    # rebuild. Failing before writing avoids leaving mismatched artifacts.
    for lang in ("en", "ko"):
        if lang not in articles and (POSTS / f"{STEM}{'-KR' if lang == 'ko' else ''}.md").exists():
            articles[lang] = read_article(lang)
    if len(articles) == 2:
        assert articles["en"]["snippets"] == articles["ko"]["snippets"], "Paired Python code drift"
    questions = browser_quiz()
    prepared = {}
    for lang in languages:
        cells = article_cells(lang, articles[lang], questions)
        checked = execute(cells)
        notebook = {"cells": cells, "metadata": {
            "title": articles[lang]["title"], "article_language": lang,
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
            "requirements": "Python 3.9+; standard library for core examples; SymPy for the optional symbolic check; figures embedded.",
            "source_article": f"{SITE}/posts/Galois-Theory-From-Scratch{articles[lang]['suffix']}/",
            "quiz_source": "assets/js/article-labs-galois.js:QUIZ",
            "quiz_questions": len(questions)}, "nbformat": 4, "nbformat_minor": 5}
        prepared[lang] = (notebook, checked)
    for lang, (notebook, checked) in prepared.items():
        path = ROOT / f"assets/notebooks/galois-from-scratch-{lang}.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        source = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        path.write_text(source, encoding="utf-8")
        print(json.dumps({"notebook": str(path.relative_to(ROOT)), "cells": len(notebook["cells"]),
                          "embedded_figures": 10, "quiz_questions": len(questions), **checked,
                          "sha256": hashlib.sha256(source.encode()).hexdigest(), "status": "passed"}))
    if "en" not in articles:
        articles["en"] = read_article("en")
    print(json.dumps({"article_examples": export_examples(articles), "status": "passed"}))


if __name__ == "__main__":
    main()
