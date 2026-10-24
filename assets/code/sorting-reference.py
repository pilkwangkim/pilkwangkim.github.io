#!/usr/bin/env python3
"""Executed Python snippets from the sorting tutorial; standard library only."""

def is_sorted(a, key=lambda x: x):
    """Check adjacent keys without changing the input."""
    return all(key(a[i]) <= key(a[i + 1])
               for i in range(len(a) - 1))

assert is_sorted([1, 2, 2, 7])
assert not is_sorted([1, 7, 2])
assert is_sorted([])


# Requires is_sorted from Section 1.
import random

def bogo_sort_bounded(a, max_shuffles=2000, seed=0):
    """Return (sorted copy or None, shuffles used); never change a."""
    if max_shuffles < 0:
        raise ValueError("max_shuffles must be nonnegative")
    a = list(a)  # Keep the caller's input intact.
    rng = random.Random(seed)
    for attempts in range(max_shuffles + 1):
        if is_sorted(a):
            return a, attempts
        if attempts < max_shuffles:
            rng.shuffle(a)
    return None, max_shuffles

result, attempts = bogo_sort_bounded([3, 1, 2])
assert result is None or result == [1, 2, 3]


# Requires is_sorted from Section 1.
from itertools import permutations

def permutation_sort(a):
    # Only use tiny inputs: every candidate is a complete ordering.
    for candidate in permutations(a):
        if is_sorted(candidate):
            return list(candidate)

assert permutation_sort([3, 1, 2]) == [1, 2, 3]


def selection_sort(a, key=lambda x: x):
    """Sort a in place and return the same list; not stable."""
    for left in range(len(a) - 1):
        smallest = left
        for j in range(left + 1, len(a)):
            if key(a[j]) < key(a[smallest]):
                smallest = j
        # Everything before left is already final.
        if smallest != left:
            a[left], a[smallest] = a[smallest], a[left]
    return a

values = [4, 1, 3, 2]
result = selection_sort(values)
assert values == [1, 2, 3, 4]
assert result is values  # The caller sees the same modified list.


def bubble_sort(a, key=lambda x: x):
    """Stably sort a in place and return the same list."""
    for end in range(len(a) - 1, 0, -1):
        changed = False
        for j in range(end):
            if key(a[j]) > key(a[j + 1]):
                a[j], a[j + 1] = a[j + 1], a[j]
                changed = True
        # A pass with no swaps proves all remaining neighbors are ordered.
        if not changed:
            break
    return a

assert bubble_sort([4, 1, 3, 2]) == [1, 2, 3, 4]


def insertion_sort(a, key=lambda x: x):
    """Stably sort a in place and return the same list."""
    for i in range(1, len(a)):
        item = a[i]
        item_key = key(item)
        j = i - 1
        while j >= 0 and key(a[j]) > item_key:
            a[j + 1] = a[j]  # Shift right; the hole moves left.
            j -= 1
        a[j + 1] = item
    return a

assert insertion_sort([4, 1, 3, 2]) == [1, 2, 3, 4]


def binary_insertion_sort(a, key=lambda x: x):
    """Stably sort a in place using upper-bound insertion positions."""
    for i in range(1, len(a)):
        item = a[i]
        item_key = key(item)
        lo, hi = 0, i  # Search the half-open prefix [0, i).
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) <= item_key:
                lo = mid + 1  # Equal records stay before the new record.
            else:
                hi = mid
        # Finding lo did not create an empty slot: records must move.
        for j in range(i, lo, -1):
            a[j] = a[j - 1]
        a[lo] = item
    return a

records = [(2, "a"), (1, "a"), (2, "b")]
assert binary_insertion_sort(records, key=lambda x: x[0]) == [
    (1, "a"), (2, "a"), (2, "b")
]


def merge_sort(a, key=lambda x: x):
    """Return a stable sorted copy; leave a unchanged."""
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    left = merge_sort(a[:mid], key)
    right = merge_sort(a[mid:], key)
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            out.append(left[i])  # Left wins ties: preserve stability.
            i += 1
        else:
            out.append(right[j])
            j += 1
    # Only one tail can be nonempty; its records are already sorted.
    out.extend(left[i:])
    out.extend(right[j:])
    return out

records = [(7, "a"), (2, "a"), (5, "a"), (2, "b")]
ordered = merge_sort(records, key=lambda record: record[0])
assert ordered == [(2, "a"), (2, "b"), (5, "a"), (7, "a")]
assert records == [(7, "a"), (2, "a"), (5, "a"), (2, "b")]
assert ordered is not records  # A new list; the input is unchanged.


import random


def quick_sort_3way(a, key=lambda x: x, seed=0):
    """Sort a in place with random-pivot partitioning; return a."""
    rng = random.Random(seed)

    def sort(lo, hi):
        if lo >= hi:
            return
        # Choose a pivot from the inclusive interval [lo, hi].
        pivot = key(a[rng.randrange(lo, hi + 1)])
        lt, i, gt = lo, lo, hi
        while i <= gt:
            current = key(a[i])
            if current < pivot:
                a[lt], a[i] = a[i], a[lt]
                lt += 1
                i += 1
            elif current > pivot:
                a[i], a[gt] = a[gt], a[i]
                gt -= 1  # Recheck the incoming record at i.
            else:
                i += 1
        # The equal region is final. Only the two sides need sorting.
        sort(lo, lt - 1)
        sort(gt + 1, hi)

    sort(0, len(a) - 1)
    return a

assert quick_sort_3way([4, 2, 4, 1, 4, 3]) == [1, 2, 3, 4, 4, 4]


def heap_sort(a, key=lambda x: x):
    """Sort a in place using a max-heap and return a; not stable."""
    def sift_down(root, end):
        # [0, end) is the active heap. Children are already heaps.
        while 2 * root + 1 < end:
            child = 2 * root + 1
            if child + 1 < end and key(a[child]) < key(a[child + 1]):
                child += 1
            if key(a[root]) >= key(a[child]):
                break
            a[root], a[child] = a[child], a[root]
            root = child

    n = len(a)
    # Leaves need no repair. Build valid subheaps from bottom to top.
    for root in range(n // 2 - 1, -1, -1):
        sift_down(root, n)
    for end in range(n - 1, 0, -1):
        # Put the maximum into its final slot, outside the smaller heap.
        a[0], a[end] = a[end], a[0]
        sift_down(0, end)
    return a

assert heap_sort([3, 1, 6, 5, 2, 4, 7]) == [1, 2, 3, 4, 5, 6, 7]


def counting_sort(a, K, key=lambda x: x):
    """Return a stable sorted copy for integer keys in [0, K)."""
    if K < 0:
        raise ValueError("K must be nonnegative")
    count = [0] * K
    for item in a:
        value = key(item)
        if not isinstance(value, int) or not 0 <= value < K:
            raise ValueError("keys must be integers in [0, K)")
        count[value] += 1

    # Exclusive prefix sums: each key begins after all smaller keys.
    next_slot = [0] * K
    for value in range(1, K):
        next_slot[value] = next_slot[value - 1] + count[value - 1]
    out = [None] * len(a)
    for item in a:
        value = key(item)
        out[next_slot[value]] = item
        next_slot[value] += 1  # Advance this key's next free slot.
    return out

records = [(2, "a"), (1, "a"), (2, "b"), (0, "a"), (1, "b")]
assert counting_sort(records, 3, key=lambda record: record[0]) == [
    (0, "a"), (1, "a"), (1, "b"), (2, "a"), (2, "b")
]


# Requires the stable counting_sort from Section 13.
def radix_sort_nonnegative(a, base=10):
    """Return a sorted copy of nonnegative integers, one digit per pass."""
    if base < 2:
        raise ValueError("base must be at least 2")
    if any(not isinstance(x, int) or x < 0 for x in a):
        raise ValueError("expected nonnegative integers")
    out = list(a)
    if not out:
        return out
    maximum = max(out)
    place = 1
    while place <= maximum:
        out = counting_sort(
            out, base, key=lambda x: (x // place) % base
        )
        place *= base
    return out

assert radix_sort_nonnegative([170, 45, 75, 90, 802, 24, 2, 66]) == [
    2, 24, 45, 66, 75, 90, 170, 802
]


# Requires insertion_sort from Section 7 for each local bucket.
def bucket_sort_unit(a, bucket_count=None):
    """Return a sorted copy of values in [0, 1)."""
    n = len(a)
    if bucket_count is None:
        bucket_count = max(1, n)
    if bucket_count < 1:
        raise ValueError("bucket_count must be positive")
    buckets = [[] for _ in range(bucket_count)]
    for x in a:
        if not 0 <= x < 1:
            raise ValueError("expected keys in [0, 1)")
        buckets[int(bucket_count * x)].append(x)
    out = []
    for bucket in buckets:
        insertion_sort(bucket)
        out.extend(bucket)
    return out

assert bucket_sort_unit([0.78, 0.17, 0.39, 0.26, 0.72], 4) == [
    0.17, 0.26, 0.39, 0.72, 0.78
]


students = [
    {"name": "Mina", "score": 90},
    {"name": "Joon", "score": 75},
    {"name": "Ara", "score": 90},
]
ordered = sorted(students, key=lambda student: student["score"])
assert [student["name"] for student in ordered] == ["Joon", "Mina", "Ara"]


def bitonic_four(values):
    """Return a sorted copy using the fixed six-comparator network."""
    if len(values) != 4:
        raise ValueError("this network has exactly four wires")
    a = list(values)
    stages = [
        [(0, 1, True), (2, 3, False)],  # Build opposite-order pairs.
        [(0, 2, True), (1, 3, True)],   # Split the bitonic block.
        [(0, 1, True), (2, 3, True)],   # Finish each half.
    ]
    for stage in stages:
        for i, j, ascending in stage:
            if (ascending and a[i] > a[j]) or (not ascending and a[i] < a[j]):
                a[i], a[j] = a[j], a[i]
    return a

assert bitonic_four([3, 1, 4, 2]) == [1, 2, 3, 4]
assert bitonic_four([2, 1, 2, 1]) == [1, 1, 2, 2]


def shell_sort_halving(a, key=lambda x: x):
    """Sort a in place with halving gaps and return a; not stable."""
    gap = len(a) // 2
    while gap > 0:
        for i in range(gap, len(a)):
            item = a[i]
            item_key = key(item)
            j = i
            # Insertion sort, but neighbors are gap positions apart.
            while j >= gap and key(a[j - gap]) > item_key:
                a[j] = a[j - gap]
                j -= gap
            a[j] = item
        gap //= 2
    return a

assert shell_sort_halving([8, 7, 6, 5, 4, 3, 2, 1]) == list(range(1, 9))
assert shell_sort_halving([]) == []
tagged = [(2, "a"), (2, "b"), (1, "a"), (3, "a")]
assert shell_sort_halving(tagged, key=lambda record: record[0]) == [
    (1, "a"), (2, "b"), (2, "a"), (3, "a")
]


def sort_colors(a):
    """Sort 0/1/2 in place in one classification pass; return a."""
    low = mid = 0
    high = len(a) - 1
    while mid <= high:
        if a[mid] == 0:
            a[low], a[mid] = a[mid], a[low]
            low += 1
            mid += 1
        elif a[mid] == 1:
            mid += 1
        elif a[mid] == 2:
            a[mid], a[high] = a[high], a[mid]
            high -= 1  # The incoming value at mid is still unknown.
        else:
            raise ValueError("colors must be 0, 1, or 2")
    return a

assert sort_colors([2, 0, 2, 1, 1, 0]) == [0, 0, 1, 1, 2, 2]
assert sort_colors([]) == []
assert sort_colors([2, 2, 0]) == [0, 2, 2]

