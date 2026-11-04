#!/usr/bin/env python3
"""Execute the teaching snapshots and generate the data/download assets.

Run from any directory. JSON contains real subprocess output; elapsed times in
that output can differ between runs. The ZIP contains only the workshop files
listed in ARCHIVE_MEMBERS, with fixed entry metadata.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys
import tempfile
import zipfile


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "assets/code/clean-code-tdd"
DATA = ROOT / "assets/data/cleancode"
ARCHIVE = ROOT / "assets/code/clean-code-tdd-workshop.zip"
ARCHIVE_MEMBERS = (
    "README.md",
    "exercise.py",
    "exercise_solution.py",
    "exercise_tests/test_increase_capacity.py",
    "pytest.ini",
    "requirements.txt",
    "storage.py",
    "tdd_snapshots.py",
    "tests/test_exercise_solution.py",
    "tests/test_storage.py",
    "tests/test_workshop.py",
    "workshop.py",
)


def state(capacity, confirmed=(), waiting=()):
    return {"capacity": capacity, "confirmed": list(confirmed), "waiting": list(waiting)}


def fixture_data():
    """Hand-written expected outputs, not outputs calculated by workshop.py."""
    return {
        "schema_version": 1,
        "contract": {
            "keys": ["capacity", "confirmed", "waiting"],
            "capacity": "positive integer; bool is not accepted",
            "participant_id": "exact nonempty string; case and whitespace preserved",
            "invariants": ["IDs unique across both lists", "confirmed <= capacity",
                           "waiting nonempty implies confirmed is full"],
            "copy_policy": "input unchanged; output dict and both lists are independent, even on a value no-op",
            "execution": "single-process sequential; no concurrent-update guarantees",
        },
        "cases": [
            {"id": "free-seat-boundary", "initial": state(2), "steps": [
                {"operation": "register", "participant_id": "Ada", "expected": state(2, ["Ada"])},
                {"operation": "register", "participant_id": "Bo", "expected": state(2, ["Ada", "Bo"])},
                {"operation": "register", "participant_id": "Cy", "expected": state(2, ["Ada", "Bo"], ["Cy"])},
            ]},
            {"id": "fifo-promotion", "initial": state(1, ["Ada"], ["Bo", "Cy", "Di"]), "steps": [
                {"operation": "cancel", "participant_id": "Ada", "expected": state(1, ["Bo"], ["Cy", "Di"])},
                {"operation": "cancel", "participant_id": "Bo", "expected": state(1, ["Cy"], ["Di"])},
            ]},
            {"id": "duplicate-confirmed", "initial": state(1, ["Ada"], ["Bo"]), "steps": [
                {"operation": "register", "participant_id": "Ada", "expected": state(1, ["Ada"], ["Bo"])},
            ]},
            {"id": "duplicate-waiting", "initial": state(1, ["Ada"], ["Bo", "Cy"]), "steps": [
                {"operation": "register", "participant_id": "Bo", "expected": state(1, ["Ada"], ["Bo", "Cy"])},
            ]},
            {"id": "cancel-middle-waiting", "initial": state(2, ["Ada", "Bo"], ["Cy", "Di", "Ez"]), "steps": [
                {"operation": "cancel", "participant_id": "Di", "expected": state(2, ["Ada", "Bo"], ["Cy", "Ez"])},
            ]},
            {"id": "cancel-no-waiting", "initial": state(2, ["Ada", "Bo"]), "steps": [
                {"operation": "cancel", "participant_id": "Ada", "expected": state(2, ["Bo"])},
                {"operation": "register", "participant_id": "Cy", "expected": state(2, ["Bo", "Cy"])},
            ]},
            {"id": "unknown-cancel", "initial": state(1, ["Ada"], ["Bo"]), "steps": [
                {"operation": "cancel", "participant_id": "missing", "expected": state(1, ["Ada"], ["Bo"])},
            ]},
            {"id": "empty-cancel", "initial": state(1), "steps": [
                {"operation": "cancel", "participant_id": "Ada", "expected": state(1)},
            ]},
            {"id": "exact-id-case-whitespace", "initial": state(3), "steps": [
                {"operation": "register", "participant_id": "Ada", "expected": state(3, ["Ada"])},
                {"operation": "register", "participant_id": "ada", "expected": state(3, ["Ada", "ada"])},
                {"operation": "register", "participant_id": " ", "expected": state(3, ["Ada", "ada", " "])},
                {"operation": "cancel", "participant_id": "ADA", "expected": state(3, ["Ada", "ada", " "])},
            ]},
            {"id": "unicode-id", "initial": state(1), "steps": [
                {"operation": "register", "participant_id": "아다", "expected": state(1, ["아다"])},
                {"operation": "register", "participant_id": "보", "expected": state(1, ["아다"], ["보"])},
                {"operation": "cancel", "participant_id": "아다", "expected": state(1, ["보"])},
            ]},
        ],
        "invalid_cases": [
            {"id": "zero-capacity", "operation": "new_workshop", "capacity": 0, "error": "ValueError"},
            {"id": "negative-capacity", "operation": "new_workshop", "capacity": -1, "error": "ValueError"},
            {"id": "boolean-capacity", "operation": "new_workshop", "capacity": True, "error": "ValueError"},
            {"id": "string-capacity", "operation": "new_workshop", "capacity": "2", "error": "ValueError"},
            {"id": "empty-id", "operation": "register", "initial": state(1), "participant_id": "", "error": "ValueError"},
            {"id": "nonstring-id", "operation": "cancel", "initial": state(1), "participant_id": 1, "error": "ValueError"},
            {"id": "null-id", "operation": "register", "initial": state(1), "participant_id": None, "error": "ValueError"},
            {"id": "duplicate-in-confirmed", "operation": "register", "initial": state(2, ["Ada", "Ada"]), "participant_id": "Cy", "error": "ValueError"},
            {"id": "duplicate-across-lists", "operation": "cancel", "initial": state(1, ["Ada"], ["Ada"]), "participant_id": "Ada", "error": "ValueError"},
            {"id": "duplicate-in-waiting", "operation": "register", "initial": state(1, ["Ada"], ["Bo", "Bo"]), "participant_id": "Cy", "error": "ValueError"},
            {"id": "over-capacity", "operation": "cancel", "initial": state(1, ["Ada", "Bo"]), "participant_id": "Bo", "error": "ValueError"},
            {"id": "waiting-with-free-seat", "operation": "register", "initial": state(2, ["Ada"], ["Bo"]), "participant_id": "Cy", "error": "ValueError"},
            {"id": "missing-key", "operation": "register", "initial": {"capacity": 1, "confirmed": []}, "participant_id": "Ada", "error": "ValueError"},
            {"id": "extra-key", "operation": "cancel", "initial": {**state(1), "extra": 1}, "participant_id": "Ada", "error": "ValueError"},
            {"id": "lists-required", "operation": "register", "initial": {"capacity": 1, "confirmed": "Ada", "waiting": []}, "participant_id": "Bo", "error": "ValueError"},
            {"id": "invalid-stored-id", "operation": "cancel", "initial": state(1, [""]), "participant_id": "Ada", "error": "ValueError"},
        ],
    }


def run_snapshot(source):
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--color=no"]
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryDirectory(prefix="clean-code-tdd-") as directory:
        directory = Path(directory)
        (directory / "workshop.py").write_text(source["code"], encoding="utf-8")
        (directory / "test_workshop.py").write_text(source["test"], encoding="utf-8")
        process = subprocess.run(command, cwd=directory, env=environment, text=True,
                                 capture_output=True, check=False)
    stdout = process.stdout + process.stderr
    counts = {name: 0 for name in ["passed", "failed", "error"]}
    for number, name in re.findall(r"(\d+) (passed|failed|error)\b", stdout):
        counts[name] = int(number)
    expected_code = 1 if source["expected_failed"] else 0
    if (process.returncode != expected_code or counts["error"] or
            counts["passed"] != source["expected_passed"] or
            counts["failed"] != source["expected_failed"] or
            (expected_code and "AssertionError" not in stdout)):
        raise RuntimeError(f"Unexpected snapshot result: {source['id']}\n{stdout}")
    summary = next(line for line in reversed(stdout.splitlines()) if re.search(r"\d+ (passed|failed)", line))
    record = {key: value for key, value in source.items() if not key.startswith("expected_")}
    record.update({
        "command": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --tb=short --color=no",
        "outcome": "failed" if counts["failed"] else "passed",
        "exit_code": process.returncode,
        "passed": counts["passed"], "failed": counts["failed"],
        "summary": summary,
        "stdout": stdout,
        "output_excerpt": stdout,
        "code_sha256": hashlib.sha256(source["code"].encode()).hexdigest(),
        "test_sha256": hashlib.sha256(source["test"].encode()).hexdigest(),
    })
    return record


def build_recordings():
    import pytest
    sources = runpy.run_path(str(PROJECT / "tdd_snapshots.py"))["SNAPSHOTS"]
    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "pytest_version": pytest.__version__,
        "scope": {"en": "Saved subprocess runs, not Python executed by the browser. Early snapshots cover valid inputs; workshop.py adds full validation.",
                  "ko": "브라우저의 Python 실행이 아니라 저장된 subprocess 실행 기록이다. 초기 snapshot은 유효한 입력을 다루며 workshop.py에서 전체 검증을 추가한다."},
        "snapshots": [run_snapshot(source) for source in sources],
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_archive():
    members = []
    for name in ARCHIVE_MEMBERS:
        path = PROJECT / name
        if not path.resolve().is_relative_to(PROJECT):
            raise ValueError(f"Workshop archive member is outside the project: {name}")
        members.append((name, path.read_bytes()))

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members:
            entry = zipfile.ZipInfo(f"clean-code-tdd/{name}", date_time=(2026, 11, 4, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Rerun snapshots and compare semantic results/sources without rewriting assets")
    arguments = parser.parse_args()
    recordings = build_recordings()
    fixtures = fixture_data()
    if arguments.check:
        stored = json.loads((DATA / "tdd-runs.json").read_text(encoding="utf-8"))
        assert stored["schema_version"] == recordings["schema_version"]
        assert len(stored["snapshots"]) == len(recordings["snapshots"])
        for old, new in zip(stored["snapshots"], recordings["snapshots"]):
            for key in ["id", "stage", "title", "code", "test", "failure_reason", "exit_code", "passed", "failed", "outcome", "code_sha256", "test_sha256"]:
                assert old[key] == new[key], (old["id"], key)
        assert json.loads((DATA / "workshop-fixtures.json").read_text(encoding="utf-8")) == fixtures
        print("Snapshot source/results and hand-written fixtures match saved assets.")
    else:
        write_json(DATA / "tdd-runs.json", recordings)
        write_json(DATA / "workshop-fixtures.json", fixtures)
        build_archive()
        print(f"Wrote {len(recordings['snapshots'])} executed snapshots, {len(fixtures['cases'])} fixture cases, and {ARCHIVE.relative_to(ROOT)}")
    for record in recordings["snapshots"]:
        print(f"{record['id']}: exit={record['exit_code']} passed={record['passed']} failed={record['failed']}")


if __name__ == "__main__":
    main()
