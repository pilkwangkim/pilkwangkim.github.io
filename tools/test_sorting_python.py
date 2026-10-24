#!/usr/bin/env python3
"""Execute article snippets and audit the sorting implementations themselves.

No external packages are needed. This checks correctness, record identity,
stability where promised, key-domain validation, seeded reproducibility, and
the distinction between binary-search probes and array relocation. It is not
a wall-clock benchmark or a proof of an asymptotic claim.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import contextlib
import copy
import hashlib
import io
import itertools
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "_posts/2026-4Q/2026-10-24-Sorting-From-Scratch.md"
INPLACE = ["selection_sort", "bubble_sort", "insertion_sort",
           "binary_insertion_sort", "quick_sort_3way", "heap_sort", "shell_sort_halving"]
STABLE = ["bubble_sort", "insertion_sort", "binary_insertion_sort", "merge_sort"]


class Audit:
    def __init__(self):
        self.assertions = 0
        self.groups = Counter()

    def check(self, condition, message, group):
        self.assertions += 1
        self.groups[group] += 1
        if not condition:
            raise AssertionError(message)

    def raises(self, exceptions, fn, *args, group, **kwargs):
        try:
            fn(*args, **kwargs)
        except exceptions:
            self.check(True, "", group)
        else:
            self.check(False, f"{fn.__name__}{args!r} did not reject an invalid input", group)


def load_article(path):
    source = path.read_text(encoding="utf-8")
    blocks = re.findall(r"^```python\s*\n(.*?)^```\s*$", source, flags=re.M | re.S)
    if not blocks:
        raise AssertionError("No Python fences found")
    namespace = {"__name__": "sorting_article_audit"}
    output = io.StringIO()
    for index, code in enumerate(blocks, 1):
        parsed = ast.parse(code, filename=f"{path.name}:python-fence-{index}")
        for node in ast.walk(parsed):
            modules = ([n.name for n in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in modules:
                if name and name.partition(".")[0] not in sys.stdlib_module_names:
                    raise AssertionError(f"Non-stdlib import in fence {index}: {name}")
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(compile(parsed, f"{path.name}:python-fence-{index}", "exec"), namespace)
        if "def quick_sort_3way(" in code:
            # This fence declares its own random import when copied separately.
            standalone = {"__name__": "standalone_quicksort_example"}
            exec(compile(code, "standalone-quicksort-fence", "exec"), standalone)
            assert standalone["quick_sort_3way"]([3, 1, 3, 2]) == [1, 2, 3, 3]
    required = INPLACE + ["is_sorted", "merge_sort", "counting_sort", "radix_sort_nonnegative",
                          "bucket_sort_unit", "sort_colors", "bogo_sort_bounded", "permutation_sort",
                          "bitonic_four"]
    missing = [name for name in required if not callable(namespace.get(name))]
    if missing:
        raise AssertionError(f"Article incomplete or missing implementations: {missing}")
    namespace["__article_python__"] = "\n\n".join(blocks)
    return namespace, {"python_fences_executed": len(blocks),
                       "stdout_characters": len(output.getvalue()),
                       "article_sha256": hashlib.sha256(source.encode()).hexdigest()}


def core_arrays(ns, audit):
    # Add a counter to the current article's actual while loop, rather than
    # assuming that a separately retyped Dutch-flag reference is equivalent.
    tree = ast.parse(ns["__article_python__"])
    color_fn = copy.deepcopy(next(node for node in tree.body
                                 if isinstance(node, ast.FunctionDef) and node.name == "sort_colors"))
    color_fn.name = "_sort_colors_instrumented"
    color_fn.body.insert(0, ast.parse("_classifications = 0").body[0])
    loops = [node for node in ast.walk(color_fn) if isinstance(node, ast.While)]
    if len(loops) != 1:
        raise AssertionError("The one-pass exercise must contain one classification while loop")
    loops[0].body.insert(0, ast.parse("_classifications += 1").body[0])
    for node in ast.walk(color_fn):
        if isinstance(node, ast.Return):
            node.value = ast.Tuple(elts=[node.value, ast.Name(id="_classifications", ctx=ast.Load())], ctx=ast.Load())
    module = ast.fix_missing_locations(ast.Module(body=[color_fn], type_ignores=[]))
    exec(compile(module, "instrumented-current-article-colors", "exec"), ns)
    checked = 0
    for n in range(8):
        for values in itertools.product(range(3), repeat=n):
            expected = sorted(values)
            for name in INPLACE:
                a = list(values)
                result = ns[name](a)
                audit.check(a == expected, f"{name}: {values} -> {a}", "exhaustive_core")
                audit.check(result is None or result is a, f"{name} did not preserve in-place return identity", "exhaustive_core")
            for name in ["merge_sort", "radix_sort_nonnegative"]:
                a = list(values)
                result = ns[name](a)
                audit.check(result == expected, f"{name}: {values} -> {result}", "exhaustive_core")
                audit.check(a == list(values), f"{name} mutated its input", "exhaustive_core")
            a = list(values)
            result = ns["counting_sort"](a, 3)
            audit.check(result == expected and a == list(values), f"counting_sort failed: {values}", "exhaustive_core")
            a = list(values)
            ns["sort_colors"](a)
            audit.check(a == expected, f"sort_colors failed: {values}", "exhaustive_core")
            got, iterations = ns["_sort_colors_instrumented"](list(values))
            audit.check(got == expected and iterations == n,
                        f"sort_colors did not make exactly n classifications: {values} -> {iterations}", "colors_classifications")
            unit = [value/3 for value in values]
            for bucket_count in [None, 1, 3]:
                audit.check(ns["bucket_sort_unit"](unit, bucket_count=bucket_count) == sorted(unit),
                            f"bucket_sort_unit exhaustive case failed: {unit}", "bucket_exhaustive")
            checked += 1
    return checked


def stable_records(ns, audit):
    count = 0
    for n in range(7):
        for keys in itertools.product(range(3), repeat=n):
            records = [(key, index) for index, key in enumerate(keys)]
            expected = sorted(records, key=lambda item: item[0])
            for name in STABLE:
                a = records.copy()
                result = ns[name](a, key=lambda item: item[0])
                got = result if name == "merge_sort" else a
                audit.check(got == expected, f"{name} unstable or wrong: {records} -> {got}", "stability")
                audit.check(Counter(got) == Counter(records), f"{name} lost record identities", "stability")
            got = ns["counting_sort"](records, 3, key=lambda item: item[0])
            audit.check(got == expected, f"counting_sort unstable: {records} -> {got}", "stability")
            # Non-stable algorithms must still sort by the supplied key without
            # comparing identities as a hidden secondary key.
            for name in ["selection_sort", "quick_sort_3way", "heap_sort"]:
                a = records.copy()
                ns[name](a, key=lambda item: item[0])
                audit.check([p[0] for p in a] == sorted(keys) and Counter(a) == Counter(records),
                            f"{name} key contract failed", "key_contract")
            count += 1
    records = [{"score": key, "identity": i} for i, key in enumerate([2, 1, 2, 0, 1])]
    for name in INPLACE + ["merge_sort"]:
        a = records.copy()
        result = ns[name](a, key=lambda record: -record["score"])
        got = result if name == "merge_sort" else a
        audit.check([p["score"] for p in got] == [2, 2, 1, 1, 0]
                    and {id(p) for p in got} == {id(p) for p in records},
                    f"{name} failed a non-orderable record / descending-key contract", "key_contract")
        if name in STABLE:
            audit.check(got == sorted(records, key=lambda record: -record["score"]),
                        f"{name} failed descending-key stability", "stability")
    return count


def randomized(ns, audit):
    rng = random.Random(20261024)
    for index in range(350):
        n = rng.randrange(101)
        values = [rng.randrange(-100, 101) for _ in range(n)]
        expected = sorted(values)
        for name in INPLACE + ["merge_sort"]:
            a = values.copy()
            result = ns[name](a)
            got = result if name == "merge_sort" else a
            audit.check(got == expected, f"{name} random case {index} failed", "random_signed")
        positive = [abs(x) for x in values]
        for base in range(2, 17):
            got = ns["radix_sort_nonnegative"](positive, base=base)
            audit.check(got == sorted(positive), f"radix base={base} failed case {index}", "radix_bases")
        unit = [rng.random() for _ in range(n)]
        if n > 2:
            unit[:3] = [0.0, .999999999, .5]
        for bucket_count in [None, 1, 2, 7, 32]:
            got = ns["bucket_sort_unit"](unit, bucket_count=bucket_count)
            audit.check(got == sorted(unit), f"bucket count={bucket_count} failed case {index}", "bucket_random")
        for seed in [0, 1, 42]:
            a, b = values.copy(), values.copy()
            ns["quick_sort_3way"](a, seed=seed)
            ns["quick_sort_3way"](b, seed=seed)
            audit.check(a == b == expected, f"quicksort seeded repeatability failed", "quick_seed")
    for base in range(2, 17):
        for values in [[], [0], [0, 0, 0], [base-1, base, base+1, base**4, 0],
                       [10**40, 10**25+7, 0, 10**40]]:
            audit.check(ns["radix_sort_nonnegative"](values, base=base) == sorted(values),
                        f"radix boundary/base {base} failed", "radix_boundaries")


def validation(ns, audit):
    for values, K in [([], 0), ([], 1), ([0], 1), ([0, 0], 1), ([3, 0, 3, 1], 4)]:
        audit.check(ns["counting_sort"](values, K) == sorted(values),
                    f"counting boundary failed: {values}, K={K}", "counting_boundaries")
    for values, K in [([1], 1), ([-1], 3), ([0], 0), ([], -1)]:
        audit.raises((ValueError, TypeError), ns["counting_sort"], values, K, group="invalid_counting")
    for values in [[-1], [0, -5, 3]]:
        audit.raises((ValueError, TypeError), ns["radix_sort_nonnegative"], values, group="invalid_radix")
    for base in [0, 1, -2]:
        audit.raises((ValueError, TypeError), ns["radix_sort_nonnegative"], [1], base=base, group="invalid_radix")
    for values in [[-0.01], [1.0], [2.0], [float("nan")], [float("inf")]]:
        audit.raises((ValueError, TypeError), ns["bucket_sort_unit"], values, group="invalid_bucket")
    for count in [0, -1]:
        audit.raises((ValueError, TypeError), ns["bucket_sort_unit"], [.2], bucket_count=count, group="invalid_bucket")
    for colors in [[-1], [3], [0, 3, 1]]:
        audit.raises(ValueError, ns["sort_colors"], colors, group="invalid_colors")
    audit.raises(ValueError, ns["bogo_sort_bounded"], [2, 1], max_shuffles=-1,
                 group="invalid_bogo_budget")


def advanced_examples(ns, audit):
    # The teaching network has exactly four wires, not a general-length API.
    for values in itertools.chain(itertools.product(range(3), repeat=4),
                                  itertools.permutations(range(4))):
        a = list(values)
        got = ns["bitonic_four"](a)
        audit.check(got == sorted(values) and a == list(values),
                    f"four-wire network failed or mutated input: {values}", "bitonic_network")
    for n in [0, 1, 2, 3, 5, 8]:
        audit.raises(ValueError, ns["bitonic_four"], list(range(n)), group="bitonic_domain")
    # Each stage's two independent comparators execute once in this sequential
    # implementation; depth three must not be confused with six operations.
    CountedKey.comparisons = 0
    result = ns["bitonic_four"]([CountedKey(x) for x in [3, 1, 4, 2]])
    audit.check([x.value for x in result] == [1, 2, 3, 4]
                and CountedKey.comparisons == 6,
                "four-wire network work/depth example disagrees with code", "bitonic_work")
    records = [(2, "a"), (2, "b"), (1, "a"), (3, "a")]
    result = ns["shell_sort_halving"](records, key=lambda record: record[0])
    audit.check(result is records and result == [(1, "a"), (2, "b"), (2, "a"), (3, "a")],
                "Shell instability example disagrees with code", "shell_instability")


class CountedKey:
    comparisons = 0

    def __init__(self, value):
        self.value = value

    def _cmp(self, other, op):
        CountedKey.comparisons += 1
        return op(self.value, other.value)

    def __lt__(self, other):
        return self._cmp(other, lambda x, y: x < y)

    def __le__(self, other):
        return self._cmp(other, lambda x, y: x <= y)

    def __gt__(self, other):
        return self._cmp(other, lambda x, y: x > y)


class CountedList(list):
    def __init__(self, values):
        super().__init__(values)
        self.writes = 0

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            value = list(value)
            self.writes += len(value)
        else:
            self.writes += 1
        super().__setitem__(index, value)


def work_counters(ns, audit):
    rows = []
    n = 64
    I = n*(n-1)//2
    for pattern, values in [("reversed", list(range(n-1, -1, -1))), ("sorted", list(range(n)))]:
        for name in ["selection_sort", "bubble_sort", "insertion_sort", "binary_insertion_sort"]:
            a = CountedList(values)
            CountedKey.comparisons = 0
            ns[name](a, key=CountedKey)
            comparisons = CountedKey.comparisons
            audit.check(a == list(range(n)), f"instrumented {name} failed", "work_counters")
            rows.append({"algorithm": name, "pattern": pattern,
                         "n": n, "inversions": I if pattern == "reversed" else 0,
                         "key_comparisons": comparisons, "record_slot_writes": a.writes})
            if name == "selection_sort":
                audit.check(comparisons == I, "selection comparison formula disagrees with code", "work_counters")
            elif name == "bubble_sort" and pattern == "sorted":
                audit.check(comparisons == n-1 and a.writes == 0,
                            "bubble early-exit best case disagrees with code", "work_counters")
            elif name == "insertion_sort" and pattern == "reversed":
                audit.check(a.writes == I+n-1 and comparisons == I,
                            "insertion exact shifts/writes disagrees with code", "work_counters")
            elif name == "binary_insertion_sort" and pattern == "reversed":
                audit.check(a.writes == I+n-1 and comparisons <= n*(n.bit_length()),
                            "binary insertion did not retain shifts / reduce probes", "work_counters")
    return rows


def brute_force(ns, audit):
    for n in range(8):
        for values in itertools.product(range(3), repeat=n):
            a = list(values)
            got = ns["permutation_sort"](a)
            audit.check(got == sorted(values) and a == list(values),
                        f"permutation sort failed: {values}", "permutation_exhaustive")
    for n in range(8):
        for values in itertools.product(range(3), repeat=n):
            for budget in [0, 3, 128]:
                result, used = ns["bogo_sort_bounded"](values, max_shuffles=budget, seed=7)
                audit.check(0 <= used <= budget, "bogo exceeded its budget", "bounded_bogo")
                audit.check(result is None or result == sorted(values), "bogo returned unsorted output", "bounded_bogo")
                audit.check((result, used) == ns["bogo_sort_bounded"](values, max_shuffles=budget, seed=7),
                            "bogo seed not reproducible", "bounded_bogo")
                if list(values) == sorted(values):
                    audit.check(result == list(values) and used == 0, "bogo missed initially sorted case", "bounded_bogo")
    audit.check(ns["bogo_sort_bounded"]([2, 1], max_shuffles=0) == (None, 0),
                "zero bogo budget must leave unsorted input unresolved", "bounded_bogo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article", type=Path, default=ARTICLE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ns, source_info = load_article(args.article)
    audit = Audit()
    arrays = core_arrays(ns, audit)
    records = stable_records(ns, audit)
    randomized(ns, audit)
    validation(ns, audit)
    advanced_examples(ns, audit)
    counters = work_counters(ns, audit)
    brute_force(ns, audit)
    result = {"status": "PASS", **source_info, "assertions": audit.assertions,
              "exhaustive_arrays": arrays, "tagged_record_arrays": records,
              "random_arrays": 350, "groups": dict(audit.groups), "work_counters": counters,
              "scope": "Current article Python source, not browser trace parity or measured runtime"}
    serialized = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
