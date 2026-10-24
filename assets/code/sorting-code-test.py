#!/usr/bin/env python3
"""Sort Colors: a one-pass, in-place partitioning exercise (Python 3.9+).

Edit the three TODO decisions in sort_colors, then run this file. The starter
is executable but deliberately wrong: the tests explain which requirement it
misses. A correct solution changes and returns the supplied list, and uses
constant workspace. Counters and a safety stop are already wired up.
"""

CASES = [
    ("empty", []),
    ("singleton", [2]),
    ("sorted", [0, 0, 1, 1, 2, 2]),
    ("reverse", [2, 2, 1, 1, 0, 0]),
    ("recheck", [1, 2, 0]),
    ("all-twos", [2, 2, 2, 2]),
    ("mixed", [2, 0, 2, 1, 1, 0]),
    ("long-cycle", [2, 0, 1] * 8),
]

CASE_PURPOSES = {
    "empty": "An empty unknown interval needs no iteration.",
    "singleton": "The last unknown record must still be classified.",
    "sorted": "Already sorted keys still have to satisfy the partition contract.",
    "reverse": "Swaps must preserve all three classified regions.",
    "recheck": "A value arriving from high is not classified yet.",
    "all-twos": "Moving high alone must empty an all-2 unknown interval.",
    "mixed": "All three branches must cooperate without skipping a record.",
    "long-cycle": "The same rules must stay within n iterations and 2n writes.",
}


def sort_colors(nums, *, stats=None):
    """Fill the loop condition and the two mid updates below.

    Invariant (half-open intervals except for the unknown region):
      nums[:low] == 0; nums[low:mid] == 1;
      nums[mid:high+1] is unclassified; nums[high+1:] == 2.
    The judge's counters are not required workspace for the algorithm.
    """
    if any(type(x) is not int or x not in (0, 1, 2) for x in nums):
        raise ValueError("Each item must be 0, 1, or 2")
    low, mid, high = 0, 0, len(nums) - 1
    iterations = writes = 0
    stop = None
    while mid < high:  # TODO 1: Is equality still a nonempty unknown interval?
        # Safety checks diagnose wrong choices instead of hanging or indexing out
        # of range. They are not additional steps needed by the correct algorithm.
        if iterations >= len(nums):
            stop = "n-iteration safety budget reached"
            break
        if not (0 <= low <= mid <= high < len(nums)):
            stop = "pointer invariant 0 <= low <= mid <= high < n failed"
            break
        iterations += 1
        if nums[mid] == 0:
            if low != mid:
                nums[low], nums[mid] = nums[mid], nums[low]
                writes += 2
            low += 1
            mid += 1  # TODO 2: Keep this update, or leave mid unchanged?
        elif nums[mid] == 1:
            mid += 1
        else:
            if mid != high:
                nums[mid], nums[high] = nums[high], nums[mid]
                writes += 2
            high -= 1
            mid += 1  # TODO 3: Keep this update, or recheck the incoming value?
    if stats is not None:
        stats.update(iterations=iterations, writes=writes, done=mid > high,
                     low=low, mid=mid, high=high, stop=stop)
    return nums


def reference_sort_colors(nums, *, stats=None):
    """Reference solution. Read after trying the exercise."""
    if any(type(x) is not int or x not in (0, 1, 2) for x in nums):
        raise ValueError("Each item must be 0, 1, or 2")
    low, mid, high = 0, 0, len(nums) - 1
    iterations = writes = 0
    while mid <= high:
        iterations += 1
        if nums[mid] == 0:
            if low != mid:
                nums[low], nums[mid] = nums[mid], nums[low]
                writes += 2
            low += 1
            mid += 1  # The moved-in value was in the already-classified 1 region.
        elif nums[mid] == 1:
            mid += 1
        else:
            if mid != high:
                nums[mid], nums[high] = nums[high], nums[mid]
                writes += 2
            high -= 1
            # The moved-in value came from the unknown region. Inspect it next.
    if stats is not None:
        stats.update(iterations=iterations, writes=writes, done=mid > high,
                     low=low, mid=mid, high=high, stop=None)
    return nums


def run_tests(solver, verbose=True):
    """The counter checks supplement examples; they are not a workspace proof."""
    results = []
    for name, original in CASES:
        values, stats = original.copy(), {}
        identity = id(values)
        error = None
        try:
            returned = solver(values, stats=stats)
            correct = values == sorted(original)
            same_list = returned is values and id(values) == identity
            iterations_ok = stats.get("iterations", float("inf")) <= len(values)
            writes_ok = stats.get("writes", float("inf")) <= 2 * len(values)
            passed = (correct and same_list and iterations_ok and writes_ok
                      and stats.get("done") is True and not stats.get("stop"))
            if stats.get("stop"):
                reason = stats["stop"]
            elif not stats.get("done"):
                reason = (f"unknown interval [{stats.get('mid')}, "
                          f"{stats.get('high')}] is not empty")
            elif not correct:
                reason = "output/return contract failed; inspect the incoming value after a 2-swap"
            elif not same_list:
                reason = "return the original list object, not a replacement or None"
            elif not iterations_ok or not writes_ok:
                reason = "linear-work budget exceeded"
            else:
                reason = "sorted in place; unknown interval empty; work within budget"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            reason = error
            passed = False
        result = dict(id=name, passed=passed, actual=values,
                      expected=sorted(original), error=error, reason=reason, **stats)
        results.append(result)
        if verbose:
            print(f"{name}: {'PASS' if passed else 'FAIL'}; "
                  f"result={values}; iterations={stats.get('iterations', '?')}" +
                  f"; writes={stats.get('writes', '?')}; {reason}")
            print(f"  Checks: {CASE_PURPOSES[name]} Expected: {sorted(original)}")
    return results


if __name__ == "__main__":
    results = run_tests(sort_colors)
    raise SystemExit(0 if all(row["passed"] for row in results) else 1)
