#!/usr/bin/env python3
"""A* guided challenge. Fill the three TODOs; run with Python 3.9+.

Inputs: finite directed graph, nonnegative edge costs, admissible h,
h(goal)=0. Return minimum cost or None; stats belong to the judge.
The untouched starter deliberately does not pass.
"""

from heapq import heappush, heappop
from itertools import count

def priority(g, h):
    return TODO_PRIORITY

def astar_cost(graph, start, goal, h, *, stats=None):
    goal_check = 'TODO_GOAL_CHECK'
    if stats is not None:
        stats.update(expanded=0, pops=0, stale=0, reopened=0)
    serial = count()
    best = {start: 0}
    expanded_before = set()
    pops = 0
    heap = []
    heappush(heap, (priority(0, h[start]), h[start],
                    next(serial), 0, start))
    while heap:
        if pops >= 5000:
            raise RuntimeError('5000-pop teaching limit')
        _, _, _, g, u = heappop(heap)
        pops += 1
        if stats is not None:
            stats['pops'] = pops
        if g != best.get(u):
            if stats is not None:
                stats['stale'] += 1
            continue  # obsolete snapshot
        if goal_check == 'pop' and u == goal:
            return g
        expanded_before.add(u)
        if stats is not None:
            stats['expanded'] += 1
        for v, cost in graph.get(u, []):
            candidate = g + cost
            old_g = best.get(v, float('inf'))
            if TODO_RELAXATION:
                if v in expanded_before and stats is not None:
                    stats['reopened'] += 1
                best[v] = candidate
                if goal_check == 'discovery' and v == goal:
                    return candidate
                heappush(heap, (priority(candidate, h[v]),
                                h[v], next(serial), candidate, v))
    return None

CODE_CASES = [{'id': 'illustrated',
  'graph': {'edges': [[[1, 1], [2, 4]], [[3, 20]], [[3, 1]], []],
            'start': 0,
            'goal': 3,
            'h': [5, 0, 1, 0]},
  'expected': 5},
 {'id': 'first-goal',
  'graph': {'edges': [[[2, 10], [1, 1]], [[2, 1]], []],
            'start': 0,
            'goal': 2,
            'h': [0, 0, 0]},
  'expected': 2},
 {'id': 'reopening',
  'graph': {'edges': [[[1, 3], [2, 1]], [[3, 3]], [[1, 1], [3, 100]], []],
            'start': 0,
            'goal': 3,
            'h': [5, 0, 4, 0]},
  'expected': 5},
 {'id': 'start-goal',
  'graph': {'edges': [[]], 'start': 0, 'goal': 0, 'h': [0]},
  'expected': 0},
 {'id': 'unreachable',
  'graph': {'edges': [[[1, 2]], [], []], 'start': 0, 'goal': 2, 'h': [0, 0, 0]},
  'expected': None},
 {'id': 'zero-cycle',
  'graph': {'edges': [[[1, 0]], [[0, 0], [2, 2]], []],
            'start': 0,
            'goal': 2,
            'h': [0, 0, 0]},
  'expected': 2},
 {'id': 'stale',
  'graph': {'edges': [[[1, 5], [2, 1]], [[3, 1]], [[1, 1]], [[4, 10]], []],
            'start': 0,
            'goal': 4,
            'h': [0, 0, 0, 0, 0]},
  'expected': 13,
  'maxExpanded': 4},
 {'id': 'informed',
  'graph': {'grid': {'width': 15, 'height': 15},
            'start': 0,
            'goal': 224,
            'h': 'manhattan'},
  'expected': 28,
  'maxExpanded': 40}]

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

if __name__ == "__main__":
    results = run_astar_code_tests(astar_cost)
    raise SystemExit(0 if all(row["passed"] for row in results) else 1)
