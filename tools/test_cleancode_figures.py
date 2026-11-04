#!/usr/bin/env python3
"""Recompute layout, executable state examples, and browser-lab matrix parity."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
from PIL import Image

from cleancode_figures import (BUGS, MAKERS, NAMES, OUT, ROOT, TESTS, cancel,
                               configure, domain_examples, register,
                               run_case, state, validate)
from cleancode_cover_flat import cover


BODY_STEMS = ["fig-01-state", "fig-02-test", "fig-03-cycle", "fig-04-refactor",
              "fig-05-boundaries"]
COVER_STEM = "cover-flat"
COVER_SIZE = (1672, 941)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    manifest = json.loads((OUT / "figure-data.json").read_text())
    records = {row["stem"]: row for row in manifest["assets"]}
    expected_stems = set(BODY_STEMS + [name + "-mobile" for name in BODY_STEMS] + [COVER_STEM])
    check(NAMES == BODY_STEMS and len(MAKERS) == len(NAMES), "Exactly five authored body diagram makers")
    check(set(records) == expected_stems and len(manifest["assets"]) == 11,
          "Five desktop/mobile figure pairs and one code-native flat cover")
    configure()
    for mobile in (False, True):
        for maker, name in zip(MAKERS, NAMES):
            stem = name+("-mobile" if mobile else "")
            fig = maker(mobile)
            live = validate(fig, stem)
            row = records[stem]
            check(live == row["layout"], stem+": actual text bounds match exported manifest")
            check(fig._semantics == row["semantics"], stem+": regenerated semantics match exported manifest")
            check(live["minimum_css_font_px"] >= 16, stem+": minimum 16 CSS px at article width")
            check(live["text_collisions"] == live["text_out_of_bounds"] == 0,
                  stem+": no text collision or clipping")
            check(fig._width == (480 if mobile else 960), stem+": correct responsive source width")
            with Image.open(OUT / (stem+".png")) as image:
                check(image.size == (row["width"], row["height"]), stem+": PNG size matches source geometry")
            svg = ET.parse(OUT / (stem+".svg")).getroot()
            check(not list(svg.iter("{http://www.w3.org/2000/svg}text"))
                  and len(list(svg.iter("{http://www.w3.org/2000/svg}path"))) > 20,
                  stem+": SVG glyphs outlined for portable rendering")
            plt.close(fig)
    cover_record = records[COVER_STEM]
    cover_path = OUT / (COVER_STEM + ".png")
    fig = cover()
    live = validate(fig, COVER_STEM)
    check(live == cover_record["layout"], "Flat cover: actual text bounds match exported manifest")
    check(fig._semantics == cover_record["semantics"],
          "Flat cover: regenerated semantics match exported manifest")
    cover_fixture = state(1, ["A"], ["B"])
    cover_expected = state(1, ["B"], [])
    cover_semantics = {
        "input": cover_fixture,
        "operation": "cancel(A)",
        "feedback_labels": ["Fail", "Pass", "Refactor"],
        "displayed_actual_confirmed": [["A"], ["B"], ["B"]],
        "expected_confirmed": ["B"],
    }
    check({key: cover_record["semantics"].get(key) for key in cover_semantics}
          == cover_semantics,
          "Flat cover uses one fixed capacity-one cancellation fixture and the exact staged outputs")
    check(cancel(deepcopy(cover_fixture), "A") == cover_expected,
          "Independent state model matches the flat cover's unchanged FIFO expectation")
    check((fig._width, fig._height) == COVER_SIZE,
          "Flat cover renderer uses the exact source geometry")
    check(live["review_width_css_px"] == 328 and live["minimum_css_font_px"] >= 16,
          "Flat cover typography remains at least 16 CSS px at 328 CSS px width")
    check(live["text_collisions"] == live["text_out_of_bounds"] == 0,
          "Flat cover: no actual text collision or clipping")
    with Image.open(cover_path) as image:
        check(image.format == "PNG" and image.size == COVER_SIZE,
              "Flat cover is a PNG with exact 1672×941 geometry")
    check((cover_record["width"], cover_record["height"]) == COVER_SIZE
          and cover_record.get("kind") == "code-native-flat",
          "Flat cover manifest identifies its dimensions and code-native kind")
    svg = ET.parse(OUT / (COVER_STEM + ".svg")).getroot()
    check(svg.tag == "{http://www.w3.org/2000/svg}svg"
          and not list(svg.iter("{http://www.w3.org/2000/svg}text"))
          and len(list(svg.iter("{http://www.w3.org/2000/svg}path"))) > 20,
          "Flat cover SVG glyphs are outlined for portable rendering")
    for attribute, expected in zip(["width", "height"], COVER_SIZE):
        value = svg.attrib[attribute]
        check(value.endswith("pt") and abs(float(value[:-2]) * 100 / 72 - expected) < .01,
              "Flat cover SVG " + attribute + " matches its source geometry")
    plt.close(fig)
    digest = hashlib.sha256(cover_path.read_bytes()).hexdigest()
    check(cover_record.get("sha256") == digest, "Flat cover bytes match the manifest hash")
    check(cover_record.get("provenance") == "cover-provenance.json",
          "Flat cover has its dedicated provenance record")
    provenance = json.loads((OUT / cover_record["provenance"]).read_text())
    check(provenance.get("sha256") == digest
          and (provenance.get("width"), provenance.get("height")) == COVER_SIZE,
          "Flat cover provenance matches the actual bytes and geometry")
    check(provenance.get("tool") == "matplotlib" and provenance.get("mode") == "code-native-svg",
          "Flat cover records its actual code-native rendering tool and mode")
    check(provenance.get("asset") == COVER_STEM + ".png"
          and provenance.get("svg") == COVER_STEM + ".svg"
          and provenance.get("renderer") == "tools/cleancode_cover_flat.py",
          "Flat cover provenance identifies the rendered PNG, SVG and source module")

    examples = domain_examples()
    check(examples["cancellation"]["result"] == state(2, ["B", "C"], ["D"]),
          "Capacity-two cancellation promotes C and retains D in waiting")
    check(examples["independent_expectation"]["expected"] == state(2, ["A"]),
          "Expected first-registration state is independent literal data")
    counterexample = examples["assertion_counterexample"]
    check(counterexample == {
        "input": state(2, ["A", "B"], ["C", "D"]), "operation": "cancel(A)",
        "expected": state(2, ["B", "C"], ["D"]),
        "faulty_actual": state(2, ["B", "D"], ["C"]), "bug": "lifo-promotion",
    }, "Assertion counterexample is the exact independently specified FIFO/LIFO fixture")
    check(all(records[stem]["semantics"] == counterexample
              for stem in ["fig-02-test", "fig-02-test-mobile"]),
          "Both assertion diagrams retain that exact counterexample")
    check(cancel(counterexample["input"], "A") == counterexample["expected"]
          and cancel(counterexample["input"], "A", "lifo-promotion") == counterexample["faulty_actual"]
          and counterexample["expected"] != counterexample["faulty_actual"],
          "Deliberate LIFO mutant, not the correct implementation, produces the drawn mismatch")
    matrix = examples["mutation_test_matrix"]["detected"]
    check(matrix == [[i == j for j in range(4)] for i in range(4)],
          "Four focused assertions detect their own isolated mutant")
    check(all(run_case(test) for test in TESTS), "All four assertions pass the correct implementation")

    # Compare the independent figure model with the actual downloadable project,
    # including all short valid orderings rather than only the drawn example.
    project = ROOT / "assets/code/clean-code-tdd/workshop.py"
    spec = importlib.util.spec_from_file_location("downloadable_workshop", project)
    workshop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workshop)
    cover_input = deepcopy(cover_fixture)
    check(workshop.cancel(cover_input, "A") == cover_expected
          and cover_input == cover_fixture,
          "Downloadable cancellation matches the flat cover and preserves its input")
    counterexample_input = deepcopy(counterexample["input"])
    check(workshop.cancel(counterexample_input, "A") == counterexample["expected"]
          and counterexample_input == counterexample["input"],
          "Downloadable cancellation matches FIFO expected state and preserves the counterexample input")
    state_count, operation_count = 0, 0
    for capacity in (1, 2, 3):
        for count in range(5):
            for order in itertools.permutations("ABCD", count):
                initial = state(capacity, order[:capacity], order[capacity:])
                state_count += 1
                workshop.validate_state(initial)
                for user in "ABCDE":
                    for reference, canonical in ((register, workshop.register), (cancel, workshop.cancel)):
                        source, snapshot = deepcopy(initial), deepcopy(initial)
                        expected = reference(initial, user)
                        actual = canonical(source, user)
                        if actual != expected or source != snapshot:
                            raise AssertionError((capacity, order, reference.__name__, user, expected, actual))
                        if actual is source or actual["confirmed"] is source["confirmed"] or actual["waiting"] is source["waiting"]:
                            raise AssertionError("canonical operation did not return independent lists")
                        workshop.validate_state(actual)
                        operation_count += 1
    check(operation_count > 1000, "Independent state model matches downloadable code and preserves source across 1,000+ operations")

    snapshot_spec = importlib.util.spec_from_file_location("tdd_snapshots", project.with_name("tdd_snapshots.py"))
    snapshots = importlib.util.module_from_spec(snapshot_spec)
    snapshot_spec.loader.exec_module(snapshots)
    stages = {row["id"]: row for row in snapshots.SNAPSHOTS}
    cycle_ids = ["cancel-red", "cancel-green", "copy-refactor"]
    saved_runs = {row["id"]: row for row in json.loads(
        (ROOT / "assets/data/cleancode/tdd-runs.json").read_text())["snapshots"]}
    check(all(saved_runs[snapshot_id][key] == stages[snapshot_id][key]
              for snapshot_id in cycle_ids for key in ["code", "test"]),
          "The cycle's saved pytest records refer to the exact downloadable snapshot code and tests")
    cycle_counts = {stage: {key: saved_runs[snapshot_id][key] for key in ["passed", "failed"]}
                    for stage, snapshot_id in zip(["red", "green", "refactor"], cycle_ids)}
    check(cycle_counts == {"red": {"passed": 4, "failed": 1},
                           "green": {"passed": 5, "failed": 0},
                           "refactor": {"passed": 5, "failed": 0}},
          "Saved cycle runs contain the depicted 4/1, 5/0, 5/0 passed/failed counts")
    check(all(records[stem]["semantics"].get("source_snapshot_ids") == cycle_ids
              and records[stem]["semantics"].get("test_counts") == cycle_counts
              for stem in ["fig-03-cycle", "fig-03-cycle-mobile"]),
          "Both TDD cycle diagrams match the counts in those exact saved pytest records")
    cycle_outputs = []
    for snapshot_id in cycle_ids:
        namespace = {}
        exec(stages[snapshot_id]["code"], namespace)
        cycle_outputs.append(namespace["cancel"](state(1, ["A"], ["B"]), "A")["confirmed"])
    check(cycle_outputs == [["A"], ["B"], ["B"]],
          "Drawn RED/GREEN/REFACTOR outputs match executed downloadable TDD snapshots")

    # This is the same JS module served by the article: every matrix entry comes
    # from actually executing the selected assertion against an isolated bug.
    node_program = """
const m = require('./assets/js/article-labs-cleancode.js');
const tests = ['at-capacity','duplicate-waiting','fifo','nonmutation'];
process.stdout.write(JSON.stringify({tests, bugs:m.bugIds,
  detected: tests.map(t=>m.bugIds.map(b=>!m.runTest(t,b).passed))}));
"""
    browser_matrix = json.loads(subprocess.check_output(["node", "-e", node_program], cwd=ROOT, text=True))
    check(browser_matrix["bugs"] == ["boundary", "duplicate", "lifo", "mutation"], "JS matrix uses the same four bug roles")
    check(browser_matrix["detected"] == matrix, "All sixteen native-table cells match browser-lab test execution")
    report = {"status": "passed", "checks": checks, "assertions": len(checks),
              "fresh_layouts": len(BODY_STEMS) * 2 + 1, "code_native_cover": {
                  "stem": COVER_STEM, "sha256": digest,
                  "validation": "Fresh renderer text bounds, outlined SVG glyphs, PNG geometry, content hash and code-native provenance"},
              "canonical_valid_states": state_count,
              "canonical_operation_comparisons": operation_count,
              "browser_matrix": browser_matrix,
              "minimum_css_font_px": min(row["layout"]["minimum_css_font_px"]
                                         for row in records.values() if "layout" in row)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
