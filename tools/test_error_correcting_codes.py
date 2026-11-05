#!/usr/bin/env python3
"""Exhaustively audit the article's small binary codes using only the stdlib.

Run from any directory: python3 tools/test_error_correcting_codes.py
Optional --articles also executes the completed articles' Python fences; any
exercise starter containing NotImplementedError is parsed but not executed.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import contextlib
import importlib.util
import io
import itertools
import json
from math import comb, isclose
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "assets/code/error-correcting-codes.py"
SPEC = importlib.util.spec_from_file_location("ecc_reference", REFERENCE)
ecc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ecc  # dataclasses resolves annotations by module name.
SPEC.loader.exec_module(ecc)


class Audit:
    def __init__(self):
        self.checks = Counter()

    def check(self, condition, message, group):
        self.checks[group] += 1
        if not condition:
            raise AssertionError(message)

    def rejects(self, fn, *args):
        try:
            fn(*args)
        except ValueError:
            self.check(True, "", "validation")
        else:
            raise AssertionError(f"Expected ValueError from {fn.__name__}{args!r}")


def rank_mod2(rows):
    """Independent Gaussian elimination, for tests rather than decoding."""
    work = [list(row) for row in rows]
    rank = 0
    for column in range(len(work[0])):
        pivot = next((r for r in range(rank, len(work)) if work[r][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for row in range(len(work)):
            if row != rank and work[row][column]:
                work[row] = [a ^ b for a, b in zip(work[row], work[rank])]
        rank += 1
    return rank


def audit_codes(audit):
    messages = list(itertools.product((0, 1), repeat=4))
    words = [ecc.encode_hamming(message) for message in messages]
    extended_words = [ecc.encode_secded(message) for message in messages]
    audit.check(len(set(words)) == 16, "Encoding must be injective", "linear_algebra")
    audit.check(rank_mod2(ecc.G) == 4, "G must have rank 4", "linear_algebra")
    audit.check(rank_mod2(ecc.H) == 3, "H must have rank 3", "linear_algebra")
    columns = list(zip(*ecc.G))
    for column in columns:
        audit.check(ecc.matvec(ecc.H, column) == (0, 0, 0), "H G != 0", "linear_algebra")
    all_words = list(itertools.product((0, 1), repeat=7))
    nullspace = {word for word in all_words if ecc.syndrome(word) == (0, 0, 0)}
    audit.check(nullspace == set(words), "Image(G) must equal nullspace(H)", "linear_algebra")

    distances = [ecc.hamming_distance(a, b) for a, b in itertools.combinations(words, 2)]
    extended_distances = [ecc.hamming_distance(a, b)
                          for a, b in itertools.combinations(extended_words, 2)]
    audit.check(min(distances) == 3, "Hamming distance must be exactly 3", "distance")
    audit.check(min(extended_distances) == 4, "Extended distance must be exactly 4", "distance")
    # Also check linearity independently for every pair of messages.
    for m1, m2 in itertools.product(messages, repeat=2):
        audit.check(ecc.encode_hamming(ecc.xor_vectors(m1, m2))
                    == ecc.xor_vectors(ecc.encode_hamming(m1), ecc.encode_hamming(m2)),
                    "Encoding is not linear", "linearity")

    clean_single = double_misdecoded = secded_cases = 0
    covered_words = Counter()
    for message, sent in zip(messages, words):
        for positions in [()] + [(index,) for index in range(1, 8)]:
            received = ecc.flip(sent, *positions)
            covered_words[received] += 1
            result = ecc.decode_hamming(received)
            audit.check(result.word == sent and result.message == message,
                        "Hamming clean/single recovery failed", "hamming_clean_single")
            audit.check(result.flipped_position == (positions[0] if positions else None),
                        "Wrong correction position", "hamming_clean_single")
            audit.check(ecc.decode_single(received) == message,
                        "Coding exercise solution failed", "coding_exercise")
            clean_single += 1

        for positions in itertools.combinations(range(1, 8), 2):
            received = ecc.flip(sent, *positions)
            result = ecc.decode_hamming(received)
            expected_position = positions[0] ^ positions[1]
            audit.check(result.flipped_position == expected_position,
                        "Double-error syndrome must XOR the two columns", "hamming_double")
            audit.check(result.word != sent and result.message != message,
                        "Double errors should miscorrect to a different codeword", "hamming_double")
            audit.check(ecc.syndrome(result.word) == (0, 0, 0),
                        "Miscorrected word must still satisfy parity", "hamming_double")
            double_misdecoded += 1

        extended_sent = ecc.encode_secded(message)
        for errors in range(3):
            for positions in itertools.combinations(range(1, 9), errors):
                result = ecc.decode_secded(ecc.flip(extended_sent, *positions))
                if errors == 2:
                    audit.check(result.status == "detected_double_error"
                                and result.word is None and result.message is None,
                                "SECDED must refuse every double error", "secded")
                else:
                    expected = ("no_error_detected" if not positions
                                else "corrected_overall_parity" if positions == (8,)
                                else "corrected_code_bit")
                    audit.check(result.word == extended_sent and result.message == message
                                and result.status == expected,
                                "SECDED clean/single recovery failed", "secded")
                secded_cases += 1
    audit.check(len(covered_words) == 128 and set(covered_words.values()) == {1},
                "The 16 radius-one Hamming balls must partition all 128 words", "perfect_code")

    # The same zero-syndrome triple error exists around every codeword.
    for message, sent in zip(messages, words):
        received = ecc.flip(sent, 3, 5, 6)
        result = ecc.decode_hamming(received)
        audit.check(result.syndrome == (0, 0, 0) and result.word != sent
                    and result.status == "no_error_detected",
                    "Triple error must demonstrate zero-syndrome ambiguity", "limits")
        extended = ecc.encode_secded(message)
        triple_result = ecc.decode_secded(ecc.flip(extended, 3, 5, 6))
        audit.check(triple_result.status == "corrected_overall_parity"
                    and triple_result.word != extended,
                    "Three flips can fool SECDED into a false single-parity correction", "limits")
        four_result = ecc.decode_secded(ecc.flip(extended, 3, 5, 6, 8))
        audit.check(four_result.status == "no_error_detected" and four_result.word != extended,
                    "Four flips can produce another valid extended codeword", "limits")
    return {"hamming_clean_single_cases": clean_single,
            "hamming_double_miscorrection_cases": double_misdecoded,
            "secded_zero_one_two_error_cases": secded_cases,
            "hamming_codewords": len(words), "minimum_distance": min(distances),
            "extended_minimum_distance": min(extended_distances)}


def audit_examples(audit):
    sent = ecc.encode_hamming("1011")
    audit.check(ecc.bitstring(sent) == "0110011", "Article codeword mismatch", "examples")
    received = ecc.flip(sent, 5)
    audit.check(ecc.bitstring(received) == "0110111", "Article received word mismatch", "examples")
    audit.check(ecc.syndrome(received) == (1, 0, 1)
                and ecc.syndrome_position(received) == 5, "Article syndrome mismatch", "examples")
    double = ecc.decode_hamming(ecc.flip(sent, 3, 5))
    audit.check(ecc.bitstring(double.word) == "0100101"
                and ecc.bitstring(double.message) == "0101", "Article double error mismatch", "examples")
    audit.check(ecc.bitstring(ecc.encode_secded("1011")) == "01100110",
                "Article extended word mismatch", "examples")
    source = [0, 1, 1, 0, 1, 1, 1]
    snapshot = source.copy()
    ecc.decode_hamming(source)
    audit.check(source == snapshot, "Decoder mutated caller input", "validation")
    audit.check(ecc.flip("1011", 2, 2) == ecc.bits("1011"),
                "Repeated flips must cancel", "validation")
    audit.check(ecc.xor_vectors("1011", "0110") == ecc.bits("1101"),
                "XOR example mismatch", "examples")

    for invalid in ["1 01", "1021", [1, 0, 2, 1], [1, 0, 1.0, 1], [True, 0, 1, 0], 1011]:
        audit.rejects(ecc.bits, invalid)
    for invalid in ["", "101", "10110"]:
        audit.rejects(ecc.encode_hamming, invalid)
    audit.rejects(ecc.syndrome, "01100110")
    audit.rejects(ecc.decode_hamming, "011001")
    audit.rejects(ecc.decode_secded, "0110011")
    audit.rejects(ecc.xor_vectors, "10", "101")
    audit.rejects(ecc.matvec, [(1, 0), (1,)], (1, 0))
    for position in [0, 8, 1.5, True]:
        audit.rejects(ecc.flip, "0110011", position)
    for repeats in [0, -1, 2, 2.0, True]:
        audit.rejects(ecc.repetition_failure_probability, 0.1, repeats)
    for p in [-0.1, 1.1, float("inf"), float("nan"), True, "0.1"]:
        audit.rejects(ecc.repetition_failure_probability, p)

    for i in range(101):
        p = i / 100
        expected = 3 * p * p - 2 * p**3
        audit.check(isclose(ecc.repetition_failure_probability(p), expected,
                            rel_tol=1e-12, abs_tol=1e-12),
                    "Threefold repetition polynomial mismatch", "probability")
        audit.check(isclose(ecc.repetition_failure_probability(p, 1), p),
                    "One repetition should have error probability p", "probability")
    audit.check(isclose(ecc.repetition_failure_probability(0.1), 0.028),
                "BSC example p=.1 must give .028", "probability")


def audit_articles(audit):
    """Execute actual prose snippets after the parent has finished both posts."""
    records = []
    for path in sorted((ROOT / "_posts").glob("**/*Error-Correcting-Codes*.md")):
        text = path.read_text(encoding="utf-8")
        blocks = re.findall(r"^```python\s*\n(.*?)^```\s*$", text, re.M | re.S)
        namespace = {"__name__": "ecc_article_snippets"}
        executed = skipped = 0
        for index, block in enumerate(blocks, start=1):
            parsed = ast.parse(block, filename=f"{path.name}:python-{index}")
            if any(isinstance(node, ast.Name) and node.id == "NotImplementedError"
                   for node in ast.walk(parsed)):
                skipped += 1
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(parsed, f"{path.name}:python-{index}", "exec"), namespace)
            audit.check(True, "", "article_python_fences")
            executed += 1
        if not blocks:
            raise AssertionError(f"No Python code blocks in {path}")
        required = ("encode_hamming", "syndrome", "decode_assuming_one_error",
                    "encode_extended", "decode_extended", "flip_positions",
                    "check_decoder", "decode_single_reference", "repetition_failure")
        for name in required:
            audit.check(callable(namespace.get(name)),
                        f"Missing authored function {name} in {path.name}",
                        "article_dependencies")
        audit.check(namespace["H"] == ecc.H,
                    f"Authored parity-check matrix differs in {path.name}",
                    "article_dependencies")

        # Check the actual authored implementations, not just separate reference
        # functions or the few assert statements printed in the article.
        authored_cases = Counter()
        status_names = {
            "no_error_detected": "no error detected",
            "corrected_code_bit": "corrected code bit",
            "corrected_overall_parity": "corrected overall parity",
            "detected_double_error": "detected double error",
        }
        for message in itertools.product((0, 1), repeat=4):
            sent = ecc.encode_hamming(message)
            audit.check(namespace["encode_hamming"](message) == sent,
                        "Authored encoder differs from reference", "article_reference_parity")
            extended = ecc.encode_secded(message)
            audit.check(namespace["encode_extended"](message) == extended,
                        "Authored extended encoder differs from reference",
                        "article_reference_parity")
            for errors in range(3):
                for positions in itertools.combinations(range(1, 8), errors):
                    received = ecc.flip(sent, *positions)
                    audit.check(namespace["flip_positions"](sent, *positions) == received,
                                "Authored flip helper differs from reference",
                                "article_reference_parity")
                    audit.check(namespace["syndrome"](received) == ecc.syndrome(received),
                                "Authored syndrome differs from reference",
                                "article_reference_parity")
                    audit.check(namespace["decode_assuming_one_error"](received)
                                == ecc.decode_hamming(received).message,
                                "Authored Hamming decoder differs from reference",
                                "article_reference_parity")
                    authored_cases["hamming_zero_one_two_errors"] += 1
                for positions in itertools.combinations(range(1, 9), errors):
                    received = ecc.flip(extended, *positions)
                    reference = ecc.decode_secded(received)
                    actual = namespace["decode_extended"](received)
                    audit.check(actual == (reference.message, status_names[reference.status]),
                                "Authored SECDED decoder differs from reference",
                                "article_reference_parity")
                    authored_cases["secded_zero_one_two_errors"] += 1

        # Audit the checker as well as the sample solution. Because Hamming is
        # perfect, these 128 cases cover every possible seven-bit received word
        # exactly once when paired with the promised clean/single-error source.
        checker_inputs = []

        def checked_solution(received):
            checker_inputs.append(tuple(received))
            return namespace["decode_single_reference"](received)

        audit.check(namespace["check_decoder"](checked_solution) == 128,
                    "Authored exercise checker did not run all 128 cases",
                    "article_exercise_harness")
        audit.check(len(checker_inputs) == 128 and len(set(checker_inputs)) == 128,
                    "Authored exercise checker skipped or repeated a received word",
                    "article_exercise_harness")
        for wrong in (
            lambda received: (0, 0, 0, 0),
            lambda received: tuple(received[index] for index in ecc.DATA_INDICES),
            lambda received: list(namespace["decode_single_reference"](received)),
        ):
            try:
                namespace["check_decoder"](wrong)
            except AssertionError:
                audit.check(True, "", "article_exercise_harness")
            else:
                raise AssertionError("Authored exercise checker accepted a wrong solution")
        for p in (0, 0.01, 0.1, 0.5, 0.9, 1):
            for repeats in (1, 3, 5, 7):
                audit.check(isclose(namespace["repetition_failure"](repeats, p),
                                    ecc.repetition_failure_probability(p, repeats),
                                    rel_tol=1e-12, abs_tol=1e-12),
                            "Authored repetition probability differs from reference",
                            "article_reference_parity")
        records.append({"article": str(path.relative_to(ROOT)),
                        "python_fences_executed": executed, "exercise_starters_parsed": skipped,
                        "authored_cases": dict(authored_cases)})
    if not records:
        raise AssertionError("--articles requested, but no ECC articles found")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", action="store_true")
    args = parser.parse_args()
    audit = Audit()
    summary = audit_codes(audit)
    audit_examples(audit)
    if args.articles:
        summary["articles"] = audit_articles(audit)
    summary["checks"] = dict(sorted(audit.checks.items()))
    summary["total_assertions"] = sum(audit.checks.values())
    print(json.dumps(summary, indent=2))
    print("PASS: exact reference calculations, bounded-error guarantees, and failure examples")


if __name__ == "__main__":
    main()
