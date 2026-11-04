#!/usr/bin/env python3
"""Check fixtures, finite-state properties, recorded runs, and the actual ZIP.

This is a bounded executable check, not a proof of arbitrary program behavior.
No files outside temporary directories are changed.
"""

from copy import deepcopy
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "assets/code/clean-code-tdd"
DATA = ROOT / "assets/data/cleancode"
ARCHIVE = ROOT / "assets/code/clean-code-tdd-workshop.zip"
sys.path.insert(0, str(PROJECT))

from build_cleancode_assets import ARCHIVE_MEMBERS  # noqa: E402
from workshop import cancel, new_workshop, register, validate_state  # noqa: E402


def assert_independent(before, original, after):
    assert before == original, "input was mutated"
    assert after is not before
    for name in ["confirmed", "waiting"]:
        assert after[name] is not before[name]
    validate_state(after)


def verify_fixtures():
    data = json.loads((DATA / "workshop-fixtures.json").read_text(encoding="utf-8"))
    transition_count = 0
    for case in data["cases"]:
        current = deepcopy(case["initial"])
        validate_state(current)
        for step in case["steps"]:
            original = deepcopy(current)
            operation = {"register": register, "cancel": cancel}[step["operation"]]
            updated = operation(current, step["participant_id"])
            assert updated == step["expected"], (case["id"], step)
            assert_independent(current, original, updated)
            current = updated
            transition_count += 1
    for case in data["invalid_cases"]:
        before = deepcopy(case.get("initial"))
        try:
            if case["operation"] == "new_workshop":
                new_workshop(case["capacity"])
            else:
                {"register": register, "cancel": cancel}[case["operation"]](
                    case["initial"], case["participant_id"])
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid fixture unexpectedly accepted: {case['id']}")
        assert case.get("initial") == before
    print(f"Python fixtures: {len(data['cases'])} sequences, {transition_count} valid transitions, {len(data['invalid_cases'])} rejected inputs")


def valid_small_states():
    ids = ["A", "B", "C"]
    for capacity in range(1, 4):
        for confirmed_count in range(capacity + 1):
            for confirmed in itertools.permutations(ids, confirmed_count):
                available = [participant_id for participant_id in ids if participant_id not in confirmed]
                waiting_counts = range(len(available) + 1) if confirmed_count == capacity else [0]
                for waiting_count in waiting_counts:
                    for waiting in itertools.permutations(available, waiting_count):
                        yield {"capacity": capacity, "confirmed": list(confirmed), "waiting": list(waiting)}


def verify_bounded_properties():
    state_count = 0
    transition_count = 0
    for state in valid_small_states():
        validate_state(state)
        state_count += 1
        for participant_id in ["A", "B", "C", "unknown"]:
            for operation in [register, cancel]:
                original = deepcopy(state)
                result = operation(state, participant_id)
                assert_independent(state, original, result)
                before_ids = state["confirmed"] + state["waiting"]
                after_ids = result["confirmed"] + result["waiting"]
                assert result["capacity"] == state["capacity"]
                if operation is register:
                    assert set(after_ids) == set(before_ids + [participant_id])
                    if participant_id in before_ids:
                        assert result == state
                    elif len(state["confirmed"]) < state["capacity"]:
                        assert result["confirmed"] == state["confirmed"] + [participant_id]
                        assert result["waiting"] == state["waiting"]
                    else:
                        assert result["confirmed"] == state["confirmed"]
                        assert result["waiting"] == state["waiting"] + [participant_id]
                else:
                    assert set(after_ids) == set(before_ids) - {participant_id}
                    if participant_id in state["confirmed"]:
                        remaining = [item for item in state["confirmed"] if item != participant_id]
                        assert result["confirmed"] == remaining + state["waiting"][:1]
                        assert result["waiting"] == state["waiting"][1:]
                    elif participant_id in state["waiting"]:
                        assert result["confirmed"] == state["confirmed"]
                        assert result["waiting"] == [item for item in state["waiting"] if item != participant_id]
                    else:
                        assert result == state
                transition_count += 1
    print(f"Bounded check: capacities 1..3, IDs A/B/C, {state_count} valid ordered states, {transition_count} transitions (including unknown ID)")


def verify_recordings():
    data = json.loads((DATA / "tdd-runs.json").read_text(encoding="utf-8"))
    assert len(data["snapshots"]) == 11
    previous_tests = {}
    previous_record = None
    red_record = None
    for record in data["snapshots"]:
        current_tests = {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(record["test"]).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        }
        assert record["code_sha256"] == hashlib.sha256(record["code"].encode()).hexdigest()
        assert record["test_sha256"] == hashlib.sha256(record["test"].encode()).hexdigest()
        if record["stage"] == "red":
            assert record["exit_code"] == 1 and record["failed"] == 1
            assert "AssertionError" in record["stdout"]
            # One new test per Red step; do not silently alter earlier tests.
            assert set(previous_tests) <= set(current_tests), record["id"]
            added_tests = set(current_tests) - set(previous_tests)
            assert len(added_tests) == 1, record["id"]
            assert all(current_tests[name] == source for name, source in previous_tests.items()), record["id"]
            failed_tests = re.findall(r"^FAILED \S+::(test_\w+)(?:\[[^\]]*\])?(?:\s|$)",
                                      record["stdout"], re.M)
            assert failed_tests == list(added_tests), record["id"]
            red_record = record
        else:
            assert record["exit_code"] == 0 and record["failed"] == 0
            if record["stage"] == "green":
                assert red_record is not None and previous_record is red_record
                assert record["test"] == red_record["test"], record["id"]
                red_record = None
            else:
                assert record["stage"] == "refactor"
                assert previous_record is not None and previous_record["stage"] == "green"
                assert record["test"] == previous_record["test"], record["id"]
        assert record["summary"] in record["stdout"]
        assert len(current_tests) == record["passed"] + record["failed"], record["id"]
        previous_tests, previous_record = current_tests, record
    assert red_record is None
    assert data["snapshots"][-2]["test"] == data["snapshots"][-1]["test"]
    process = subprocess.run([sys.executable, str(ROOT / "tools/build_cleancode_assets.py"), "--check"],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    print(f"Recorded snapshots: {len(data['snapshots'])} source/hash/result matches; rerun subprocesses agree")
    print("TDD sequence: each Red adds one test, each Green preserves it, and Refactor preserves the same suite")


def run_suite(directory, arguments, solution=False, expected_exit=0):
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("WORKSHOP_EXERCISE_SOLUTION", None)
    if solution:
        environment["WORKSHOP_EXERCISE_SOLUTION"] = "1"
    process = subprocess.run([sys.executable, "-m", "pytest", *arguments, "--color=no"],
                             cwd=directory, env=environment, capture_output=True, text=True, check=False)
    output = process.stdout + process.stderr
    assert process.returncode == expected_exit, output
    assert "ERROR collecting" not in output and "ModuleNotFoundError" not in output, output
    summary = next(line for line in reversed(output.splitlines()) if re.search(r"\d+ (passed|failed)", line))
    if expected_exit:
        assert "NotImplementedError" in output, output
    print(f"ZIP {'solution' if solution else 'starter' if expected_exit else 'default'}: exit={process.returncode}; {summary}")


def verify_download():
    with zipfile.ZipFile(ARCHIVE) as archive:
        expected = [f"clean-code-tdd/{name}" for name in ARCHIVE_MEMBERS]
        assert archive.namelist() == expected, "ZIP file list differs from workshop members"
        for name, entry in zip(ARCHIVE_MEMBERS, expected):
            assert archive.read(entry) == (PROJECT / name).read_bytes(), entry
        with tempfile.TemporaryDirectory(prefix="clean-code-download-") as directory:
            archive.extractall(directory)
            project = Path(directory) / "clean-code-tdd"
            run_suite(project, ["-q"])
            run_suite(project, ["exercise_tests", "-q", "--tb=short"], expected_exit=1)
            run_suite(project, ["exercise_tests", "-q"], solution=True)
    print(f"ZIP: {len(expected)} files match source byte-for-byte")


def main():
    verify_fixtures()
    verify_bounded_properties()
    verify_recordings()
    verify_download()
    print("Clean Code Python/download checks passed.")


if __name__ == "__main__":
    main()
