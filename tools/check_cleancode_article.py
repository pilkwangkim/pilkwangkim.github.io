#!/usr/bin/env python3
"""Check the bilingual Clean Code article's source structure and executable code.

This verifies local links, source/asset contracts, and the displayed Python
examples. It does not verify external site availability, authenticate quotations,
or replace a production build and actual desktop/mobile browser review. The
deliberately assertion-free example is run as displayed, not counted as evidence
that a broken function is correct. Finite fixture agreement is not a proof.
"""

import argparse
import ast
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
POSTS = {
    "en": ROOT / "_posts/2026-4Q/2026-11-04-Clean-Code-and-TDD.md",
    "ko": ROOT / "_posts/2026-4Q/2026-11-04-Clean-Code-and-TDD-KR.md",
}
PROJECT = ROOT / "assets/code/clean-code-tdd"
FIGURES = ROOT / "assets/img/posts/2026-11-04-clean-code-tdd"
FIGURE_NAMES = ["fig-01-state", "fig-02-test", "fig-03-cycle", "fig-04-refactor",
                "fig-05-boundaries"]
COVER_STEM = "cover-flat"
COVER_SIZE = (1672, 941)
MATRIX_TESTS = ["boundary", "duplicate", "fifo", "nonmutation"]
LABS = ["cleancode-state", "cleancode-tdd", "cleancode-refactor", "cleancode-tests", "cleancode-quiz"]
LAB_SECTIONS = [2, 5, 9, 13, 15]
FOLD_SECTIONS = [2, 4, 6, 7, 12, 13, 14, 15]
QUOTES = ["Readability counts.", "Duplication is a hint, not a command.",
          "without changing its observable behavior."]
REQUIRED_REFERENCES = [
    "https://newsletter.kentbeck.com/p/canon-tdd",
    "https://martinfowler.com/bliki/TestDrivenDevelopment.html",
    "https://refactoring.com/",
    "https://refactoring.com/catalog/extractFunction.html",
    "https://refactoring.com/catalog/replaceNestedConditionalWithGuardClauses.html",
    "https://peps.python.org/pep-0020/",
    "https://peps.python.org/pep-0008/",
    "https://google.github.io/eng-practices/review/reviewer/looking-for.html",
    "https://martinfowler.com/bliki/TestCoverage.html",
    "https://martinfowler.com/bliki/TestDouble.html",
    "https://docs.pytest.org/en/stable/getting-started.html",
    "https://docs.pytest.org/en/stable/how-to/fixtures.html",
    "https://docs.pytest.org/en/stable/how-to/parametrize.html",
    "https://docs.pytest.org/en/stable/how-to/tmp_path.html",
    "https://github.com/pytest-dev/pytest",
    "https://docs.python.org/3/library/copy.html",
    "https://docs.python.org/3/library/json.html",
    "https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement",
    "https://martinfowler.com/bliki/UnitTest.html",
    "https://martinfowler.com/bliki/IntegrationTest.html",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def split_post(path):
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.S)
    require(match, f"{path.name}: missing YAML front matter")
    header, body = match.groups()
    # Only the known top-level scalar/list fields are needed here; the repository
    # metadata checker remains responsible for its complete YAML metadata model.
    fields = dict(re.findall(r"^([A-Za-z_]+):[ \t]*(.*)$", header, re.M))
    return {"path": path, "header": header, "fields": fields, "body": body}


def fenced_blocks(body):
    return [(match.group(1), match.group(2)) for match in re.finditer(
        r"^```([^\n]*)\n(.*?)^```[ \t]*$", body, re.M | re.S)]


def section_at(body, offset):
    sections = list(re.finditer(r"^## (\d+)\. ", body[:offset], re.M))
    require(sections, "element before first numbered section")
    return int(sections[-1].group(1))


class MatrixTableParser(HTMLParser):
    """Read the authored table without relying on its translated cell labels."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.column_headers = 0
        self.caption = []
        self.in_caption = False
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "caption":
            self.in_caption = True
        elif tag == "tr" and "data-test" in attrs:
            require(self.row is None, "test matrix: nested rows")
            self.row = {"test": attrs["data-test"], "row_headers": 0, "cells": []}
            self.rows.append(self.row)
        elif tag == "th":
            if self.row is None:
                require(attrs.get("scope") == "col", "test matrix: column header needs scope=col")
                self.column_headers += 1
            else:
                require(attrs.get("scope") == "row", "test matrix: row header needs scope=row")
                self.row["row_headers"] += 1
        elif tag == "td":
            require(self.row is not None, "test matrix: data cell outside a named test row")
            require(attrs.get("data-detected") in {"true", "false"},
                    "test matrix: every data cell needs explicit data-detected=true/false")
            self.cell = {"detected": attrs["data-detected"] == "true", "text": []}
            self.row["cells"].append(self.cell)

    def handle_endtag(self, tag):
        if tag == "caption":
            self.in_caption = False
        elif tag == "tr":
            self.row = None
        elif tag == "td":
            self.cell = None

    def handle_data(self, data):
        if self.in_caption:
            self.caption.append(data)
        if self.cell is not None:
            self.cell["text"].append(data)


def check_test_matrix(body, name, folds):
    matches = list(re.finditer(
        r'<table\b(?=[^>]*\bid=["\']cleancode-test-matrix["\'])[^>]*>.*?</table>',
        body, re.S))
    require(len(matches) == 1, f"{name}: expected one native test matrix table")
    table = matches[0]
    require(section_at(body, table.start()) == 13, f"{name}: test matrix belongs in section 13")
    require(not any(start < table.start() < end for start, end, _ in folds),
            f"{name}: the test matrix must not be folded away")
    parsed = MatrixTableParser()
    parsed.feed(table.group(0))
    parsed.close()
    require("".join(parsed.caption).strip(), f"{name}: table needs a native nonempty caption")
    require(parsed.column_headers == 5, f"{name}: table needs five scoped column headers")
    require([row["test"] for row in parsed.rows] == MATRIX_TESTS,
            f"{name}: test matrix row IDs/order differ")
    for row in parsed.rows:
        require(row["row_headers"] == 1, f"{name}/{row['test']}: expected one scoped row header")
        require(len(row["cells"]) == 4, f"{name}/{row['test']}: expected four fault cells")
        require(all("".join(cell["text"]).strip() for cell in row["cells"]),
                f"{name}/{row['test']}: data cell must have readable text or a symbol")
    expected = json.loads((FIGURES / "figure-data.json").read_text(encoding="utf-8"))[
        "domain_examples"]["mutation_test_matrix"]["detected"]
    require([[cell["detected"] for cell in row["cells"]] for row in parsed.rows] == expected,
            f"{name}: native table differs from executed test/fault matrix")


def check_structure(post, language):
    name, fields, body = post["path"].name, post["fields"], post["body"]
    for key, expected in {
        "date": "2026-11-03 21:00:00 +0900", "categories": "[Software Engineering]",
        "lang": language, "translation_key": "clean-code-and-tdd",
        "topic": "software-engineering", "interactive_labs": "[cleancode]",
    }.items():
        require(fields.get(key) == expected, f"{name}: {key} must be {expected!r}")
    require(bool(fields.get("title")) and bool(fields.get("description")), f"{name}: empty title/description")
    require(not fields.get("series") and not fields.get("series_order"), f"{name}: standalone article must not invent a series")
    require(re.search(r"^  path: /assets/img/posts/2026-11-04-clean-code-tdd/cover-flat.png$", post["header"], re.M), f"{name}: unexpected cover")
    require(re.search(r"^  width: 1672$", post["header"], re.M) and re.search(r"^  height: 941$", post["header"], re.M), f"{name}: cover dimensions")
    numbers = [int(value) for value in re.findall(r"^## (\d+)\. ", body, re.M)]
    require(numbers == list(range(1, 17)), f"{name}: expected sixteen numbered sections, got {numbers}")

    folds, stack = [], []
    for match in re.finditer(r"<details\b[^>]*>|</details>", body):
        if match.group(0).startswith("<details"):
            require(not stack, f"{name}: nested details obscure the beginner path")
            require('markdown="1"' in match.group(0), f"{name}: details needs markdown=1")
            stack.append(match)
        else:
            require(bool(stack), f"{name}: unmatched closing details")
            opening = stack.pop()
            content = body[opening.end():match.start()]
            require(re.search(r"<summary>[^<]+</summary>", content), f"{name}: details summary missing")
            require(not re.search(r"^## ", content, re.M), f"{name}: a main section is folded away")
            require("interactive/article-lab.html" not in content, f"{name}: interactive lab is folded away")
            folds.append((opening.start(), match.end(), section_at(body, opening.start())))
    require(not stack, f"{name}: unclosed details")
    require([section for _, _, section in folds] == FOLD_SECTIONS, f"{name}: unexpected optional-fold placement")

    includes = list(re.finditer(r'{% include interactive/article-lab\.html id="([^"]+)" %}', body))
    require([match.group(1) for match in includes] == LABS, f"{name}: lab order/IDs differ")
    require([section_at(body, match.start()) for match in includes] == LAB_SECTIONS, f"{name}: lab section placement differs")
    require(body.count("interactive/article-lab.html") == len(LABS), f"{name}: malformed or extra lab include")

    figures = list(re.finditer(r'<div class="cleancode-figure-(desktop|mobile)">\s*<img\s+([^>]+)>\s*</div>', body))
    require(len(figures) == 10, f"{name}: expected five responsive figure pairs")
    figure_records = {row["stem"]: row for row in json.loads(
        (FIGURES / "figure-data.json").read_text(encoding="utf-8"))["assets"]}
    for match, (stem, layout) in zip(figures, [(stem, layout) for stem in FIGURE_NAMES for layout in ["desktop", "mobile"]]):
        attrs = dict(re.findall(r'([\w:-]+)="([^"]*)"', match.group(2)))
        filename = stem + ("-mobile" if layout == "mobile" else "") + ".svg"
        require(match.group(1) == layout, f"{name}: figure layout order")
        require(attrs.get("src") == "{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/" + filename, f"{name}: wrong figure source")
        require(attrs.get("width") == ("480" if layout == "mobile" else "960"), f"{name}: wrong source figure width")
        require(attrs.get("height") == str(figure_records[filename[:-4]]["height"]),
                f"{name}: figure height differs from generated manifest")
        require(bool(attrs.get("alt")), f"{name}: missing image alt")
        require(not any(start < match.start() < end for start, end, _ in folds), f"{name}: main figure is folded away")
    require("fig-06-test-strength" not in body, f"{name}: obsolete rasterized matrix is still embedded")
    check_test_matrix(body, name, folds)
    captions = re.findall(r"^\*(?:Figure [1-5]|Table 1)\..*$", body, re.M)
    require(len(captions) == 6 and all(line.endswith("*") for line in captions),
            f"{name}: missing or unbalanced figure/table caption emphasis")
    quotes = re.findall(r"^> (.+)$", body, re.M)
    require(quotes == QUOTES, f"{name}: exact English source quotations changed")
    for quote, source in zip(QUOTES, [REQUIRED_REFERENCES[5], REQUIRED_REFERENCES[0], REQUIRED_REFERENCES[2]]):
        offset = body.index("> " + quote)
        require(source in body[offset:offset + 600], f"{name}: quote attribution not adjacent")
    urls = set(re.findall(r"\]\((https?://[^)\s]+)\)", body))
    require(set(REQUIRED_REFERENCES) <= urls, f"{name}: canonical references missing: {set(REQUIRED_REFERENCES) - urls}")
    # The common article guide renders the translation link from metadata.
    # Do not require a second, duplicate translation link in the article body.
    partner = split_post(POSTS["ko" if language == "en" else "en"])
    require(partner["fields"].get("translation_key") == fields["translation_key"],
            f"{name}: translation metadata differs")
    local_assets = set(re.findall(r"/assets/[A-Za-z0-9/_.-]+", post["header"] + body))
    for asset in local_assets:
        require((ROOT / asset.lstrip("/")).is_file(), f"{name}: missing local asset {asset}")
    require(len(fenced_blocks(body)) > 0, f"{name}: no code examples")
    print(f"{language}: 16 sections, 8 optional folds, 5 ordered labs, 5 responsive pairs, one native test matrix, {len(urls)} source URLs, {len(local_assets)} local assets")


def png_dimensions(path):
    data = path.read_bytes()
    require(data[:8] == b"\x89PNG\r\n\x1a\n", f"not PNG: {path.name}")
    return struct.unpack(">II", data[16:24])


def check_figures():
    manifest = json.loads((FIGURES / "figure-data.json").read_text(encoding="utf-8"))
    expected_stems = set(FIGURE_NAMES + [name + "-mobile" for name in FIGURE_NAMES] + [COVER_STEM])
    require({record["stem"] for record in manifest["assets"]} == expected_stems, "figure manifest stem set differs")
    require(len(manifest["assets"]) == 11, "figure manifest duplicates")
    for record in manifest["assets"]:
        stem, width, height = record["stem"], record["width"], record["height"]
        require(png_dimensions(FIGURES / (stem + ".png")) == (width, height), f"{stem}: PNG dimensions differ")
        if stem == COVER_STEM:
            require((width, height) == COVER_SIZE, "flat cover dimensions differ")
            require(record.get("kind") == "code-native-flat", "flat cover kind differs")
            require(record["layout"]["review_width_css_px"] == 328,
                    "flat cover must audit its typography at 328 CSS pixels")
            digest = hashlib.sha256((FIGURES / (stem + ".png")).read_bytes()).hexdigest()
            require(record.get("sha256") == digest, "flat cover hash differs")
            require(record.get("provenance") == "cover-provenance.json", "flat cover provenance path differs")
            provenance = json.loads((FIGURES / record["provenance"]).read_text(encoding="utf-8"))
            require(provenance.get("sha256") == digest, "cover provenance hash differs")
            require((provenance.get("width"), provenance.get("height")) == COVER_SIZE,
                    "cover provenance dimensions differ")
            require(provenance.get("tool") == "matplotlib" and provenance.get("mode") == "code-native-svg",
                    "cover provenance must identify its actual code-native renderer")
            require(provenance.get("asset") == COVER_STEM + ".png"
                    and provenance.get("svg") == COVER_STEM + ".svg"
                    and provenance.get("renderer") == "tools/cleancode_cover_flat.py",
                    "cover provenance asset or renderer path differs")
        svg = ET.parse(FIGURES / (stem + ".svg")).getroot()
        require(svg.tag == "{http://www.w3.org/2000/svg}svg", f"{stem}: no SVG root")
        for attribute, expected in [("width", width), ("height", height)]:
            value = svg.attrib[attribute]
            require(value.endswith("pt") and abs(float(value[:-2]) * 100 / 72 - expected) < .01, f"{stem}: SVG {attribute} differs")
        require(not list(svg.iter("{http://www.w3.org/2000/svg}text")), f"{stem}: SVG fonts are not outlined")
        audit = record["layout"]
        require(audit["text_collisions"] == 0 and audit["text_out_of_bounds"] == 0, f"{stem}: layout audit failed")
        require(audit["minimum_css_font_px"] >= 16, f"{stem}: small labels")
        for label in audit["labels"]:
            x0, y0, x1, y1 = label["bbox_source_px"]
            require(x0 >= 7.99 and y0 >= 7.99 and x1 <= width - 7.99 and y1 <= height - 7.99, f"{stem}: recorded label out of bounds")
            expected_size = label["font_source_px"] * audit["review_width_css_px"] / width
            require(abs(label["font_css_px"] - expected_size) <= .02, f"{stem}: inconsistent recorded CSS font")
    sys.path.insert(0, str(PROJECT))
    from workshop import cancel, register
    examples = manifest["domain_examples"]
    cancellation = examples["cancellation"]
    before = deepcopy(cancellation["initial"])
    require(cancel(cancellation["initial"], "A") == cancellation["result"], "figure cancellation differs from package")
    require(cancellation["initial"] == before, "figure input mutated")
    independent = examples["independent_expectation"]
    require(register(independent["input"], "A") == independent["expected"] == independent["actual"], "figure assertion example differs")
    counterexample = examples["assertion_counterexample"]
    require(counterexample == {
        "input": {"capacity": 2, "confirmed": ["A", "B"], "waiting": ["C", "D"]},
        "operation": "cancel(A)",
        "expected": {"capacity": 2, "confirmed": ["B", "C"], "waiting": ["D"]},
        "faulty_actual": {"capacity": 2, "confirmed": ["B", "D"], "waiting": ["C"]},
        "bug": "lifo-promotion",
    }, "figure assertion counterexample changed")
    original = deepcopy(counterexample["input"])
    require(cancel(counterexample["input"], "A") == counterexample["expected"] != counterexample["faulty_actual"],
            "correct cancellation must match the independent expectation, not the deliberate LIFO fault")
    require(counterexample["input"] == original, "assertion-counterexample input mutated")
    for stem in ["fig-02-test", "fig-02-test-mobile"]:
        record = next(row for row in manifest["assets"] if row["stem"] == stem)
        require(record["semantics"] == counterexample, f"{stem}: drawn counterexample differs")
    cycle_ids = ["cancel-red", "cancel-green", "copy-refactor"]
    saved_runs = {row["id"]: row for row in json.loads(
        (ROOT / "assets/data/cleancode/tdd-runs.json").read_text(encoding="utf-8"))["snapshots"]}
    cycle_counts = {stage: {key: saved_runs[snapshot_id][key] for key in ["passed", "failed"]}
                    for stage, snapshot_id in zip(["red", "green", "refactor"], cycle_ids)}
    require(cycle_counts == {"red": {"passed": 4, "failed": 1},
                             "green": {"passed": 5, "failed": 0},
                             "refactor": {"passed": 5, "failed": 0}},
            "recorded TDD cycle counts changed; review the drawn counts")
    for stem in ["fig-03-cycle", "fig-03-cycle-mobile"]:
        semantics = next(row["semantics"] for row in manifest["assets"] if row["stem"] == stem)
        require(semantics.get("source_snapshot_ids") == cycle_ids
                and semantics.get("test_counts") == cycle_counts,
                f"{stem}: drawn test counts differ from the exact saved pytest records")
    print("Figure manifest: 11 outlined SVG/PNG dimension pairs and label audits; flat cover renderer provenance and PNG hash; exact state/counterexample fixtures")


EXTRA_TEST = '''

def test_article_copy_state_matches_project_and_keeps_lists_independent():
    import json
    from copy import deepcopy
    from pathlib import Path
    from workshop import copy_state as canonical_copy_state
    fixtures = json.loads(Path(FIXTURE_PATH).read_text(encoding="utf-8"))
    states = [case["initial"] for case in fixtures["cases"]]
    states.extend(step["expected"] for case in fixtures["cases"]
                  for step in case["steps"])
    assert states
    for fixture in states:
        state = deepcopy(fixture)
        original = deepcopy(state)
        actual = copy_state(state)
        assert state == original
        assert actual == canonical_copy_state(state)
        assert actual is not state
        assert actual["confirmed"] is not state["confirmed"]
        assert actual["waiting"] is not state["waiting"]
        actual["confirmed"].append("unshared result")
        actual["waiting"].append("unshared result")
        assert state == original
    candidates = [None, {}, []]
    candidates.extend(case["initial"] for case in fixtures["invalid_cases"]
                      if "initial" in case)
    rejected = 0
    for fixture in candidates:
        # Some invalid operation fixtures have valid states but invalid IDs.
        # Only states rejected by the canonical state copier belong here.
        try:
            canonical_copy_state(deepcopy(fixture))
        except ValueError:
            state = deepcopy(fixture)
            original = deepcopy(state)
            with pytest.raises(ValueError):
                copy_state(state)
            assert state == original
            rejected += 1
    assert rejected > 0


def test_article_worked_registration_matches_handwritten_fixtures():
    import json
    from copy import deepcopy
    from pathlib import Path
    from workshop import register
    fixtures = json.loads(Path(FIXTURE_PATH).read_text(encoding="utf-8"))
    checked = 0
    for case in fixtures["cases"]:
        state = deepcopy(case["initial"])
        for step in case["steps"]:
            if step["operation"] == "register":
                original = deepcopy(state)
                actual = register_worked(state, step["participant_id"])
                assert actual == step["expected"]
                assert actual == register(state, step["participant_id"])
                assert state == original
                assert actual is not state
                assert actual["confirmed"] is not state["confirmed"]
                assert actual["waiting"] is not state["waiting"]
                checked += 1
            state = deepcopy(step["expected"])
    assert checked > 0
    for case in fixtures["invalid_cases"]:
        if case["operation"] == "register":
            with pytest.raises(ValueError):
                register_worked(case["initial"], case["participant_id"])
'''


def check_python(post, language):
    blocks = [source for tag, source in fenced_blocks(post["body"]) if tag == "python"]
    source = "\n\n".join(blocks)
    tree = ast.parse(source)
    function_names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    require(len(function_names) == len(set(function_names)), f"{language}: concatenated code would overwrite a function")
    weak = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "test_executes_but_checks_nothing")
    require(not any(isinstance(node, ast.Assert) for node in ast.walk(weak)), f"{language}: weak-test example no longer matches explanation")
    require("register_worked" in function_names, f"{language}: worked registration missing")
    require("copy_state" in function_names, f"{language}: displayed copy-state helper missing")
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    environment["PYTHONPATH"] = str(PROJECT)
    with tempfile.TemporaryDirectory(prefix="clean-code-article-") as directory:
        module = Path(directory) / "test_article.py"
        fixture_path = str(ROOT / "assets/data/cleancode/workshop-fixtures.json")
        module.write_text("FIXTURE_PATH = " + repr(fixture_path) + "\n\n" + source + EXTRA_TEST, encoding="utf-8")
        process = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short", "--color=no"],
                                 cwd=directory, env=environment, capture_output=True, text=True, check=False)
    output = process.stdout + process.stderr
    require(process.returncode == 0, f"{language}: article Python failed\n{output}")
    summary = next(line for line in reversed(output.splitlines()) if re.search(r"\d+ passed", line))
    print(f"{language}: all {len(blocks)} displayed Python fences concatenated into one real pytest module; {summary}")
    print(f"{language}: scope includes the deliberately assertion-free demonstration; its PASS is not correctness evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing-ko", action="store_true", help="Preview EN checks before the translated file exists; final gate requires both")
    arguments = parser.parse_args()
    posts = {}
    for language, path in POSTS.items():
        if not path.exists() and language == "ko" and arguments.allow_missing_ko:
            print("Preview only: Korean article is missing; bilingual parity not checked.")
            continue
        require(path.is_file(), f"Missing article {path.relative_to(ROOT)}")
        posts[language] = split_post(path)
        check_structure(posts[language], language)
    if "ko" in posts:
        require(fenced_blocks(posts["en"]["body"]) == fenced_blocks(posts["ko"]["body"]), "EN/KO fenced code language or bytes differ")
        for key in ["date", "categories", "tags", "topic", "interactive_labs", "translation_key"]:
            require(posts["en"]["fields"].get(key) == posts["ko"]["fields"].get(key), f"EN/KO metadata {key} differs")
        print("EN/KO: all fenced code bytes/languages and shared metadata match")
    check_figures()
    for language, post in posts.items():
        check_python(post, language)
    print("Clean Code article source/asset/Python checks passed; external provenance, build and browser checks remain separate.")


if __name__ == "__main__":
    main()
