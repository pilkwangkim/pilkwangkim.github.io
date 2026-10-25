#!/usr/bin/env python3
"""Independent exact checks for the undergraduate Galois-theory tutorial.

The core verifier uses only the standard library.  It audits the finite examples
used by the diagrams and activities; it is not a proof of the general Galois
correspondence or radical-solvability theorem.  Optional SymPy checks provide a
separate symbolic implementation, not the mathematical argument in the article.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, deque
import contextlib
from fractions import Fraction
import hashlib
import importlib.util
import io
import itertools
import json
from math import comb
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


class Audit:
    def __init__(self):
        self.counts = Counter()

    def check(self, condition, description, group):
        self.counts[group] += 1
        if not condition:
            raise AssertionError(description)


def identity(n):
    return tuple(range(n))


def compose(p, q):
    """Rightmost first: p o q sends i to p[q[i]]."""
    return tuple(p[q[i]] for i in range(len(p)))


def inverse(p):
    result = [None] * len(p)
    for source, target in enumerate(p):
        result[target] = source
    return tuple(result)


def cycle(n, labels):
    result = list(range(n))
    for source, target in zip(labels, labels[1:] + labels[:1]):
        result[source] = target
    return tuple(result)


def commutator(p, q):
    return compose(compose(compose(p, q), inverse(p)), inverse(q))


def generated(generators, n):
    """Finite closure by exact permutation multiplication, without SymPy."""
    generators = set(generators)
    generators.update(inverse(p) for p in tuple(generators))
    found = {identity(n)}
    queue = deque(found)
    while queue:
        p = queue.popleft()
        for q in generators:
            candidate = compose(p, q)
            if candidate not in found:
                found.add(candidate)
                queue.append(candidate)
    return found


def parity(p):
    return sum(p[i] > p[j] for i in range(len(p))
               for j in range(i + 1, len(p))) % 2


def derived_subgroup(group, n):
    return generated({commutator(p, q) for p in group for q in group}, n)


def derived_orders(group, n):
    orders = [len(group)]
    while len(group) > 1:
        following = derived_subgroup(group, n)
        orders.append(len(following))
        if following == group:
            break
        group = following
    return orders


def permutation_checks(audit):
    expected = {2: [2, 1], 3: [6, 3, 1],
                4: [24, 12, 4, 1], 5: [120, 60, 60]}
    group_orders = {}
    for n in range(2, 6):
        group = set(itertools.permutations(range(n)))
        actual = derived_orders(group, n)
        audit.check(actual == expected[n], f"S{n} derived orders: {actual}", "derived")
        group_orders[f"S{n}"] = actual
        audit.check(2 * sum(parity(p) == 0 for p in group) == len(group),
                    "Even and odd permutations must split Sn equally", "learning_examples")
        for p in group:
            audit.check(compose(p, inverse(p)) == identity(n),
                        "Permutation inverse/composition mismatch", "permutations")
            audit.check(compose(identity(n), p) == p == compose(p, identity(n)),
                        "Permutation identity mismatch", "permutations")
        adjacent = [cycle(n, [j, j + 1]) for j in range(n - 1)]
        audit.check(generated(adjacent, n) == group,
                    "Adjacent transpositions did not generate Sn", "permutations")
    a5 = {p for p in itertools.permutations(range(5)) if parity(p) == 0}
    three_cycles = {cycle(5, list(labels))
                    for labels in itertools.permutations(range(5), 3)}
    audit.check(len(three_cycles) == 20, "A5 must have twenty 3-cycles", "a5")
    audit.check(generated(three_cycles, 5) == a5,
                "3-cycles did not generate A5", "a5")
    audit.check(derived_subgroup(a5, 5) == a5, "A5 is not perfect", "a5")
    g = compose(cycle(5, [0, 1]), cycle(5, [2, 3]))
    h = compose(cycle(5, [0, 1]), cycle(5, [2, 4]))
    c = commutator(g, h)
    audit.check(c == cycle(5, [2, 3, 4]),
                "Rightmost-first [(12)(34),(12)(35)] must be (345)", "a5")
    conjugates = {compose(compose(p, c), inverse(p)) for p in a5}
    audit.check(conjugates == three_cycles,
                "A5 conjugacy class of (345) must be all 3-cycles", "a5")
    five_cycle = cycle(5, [0, 1, 2, 3, 4])
    for i, j in itertools.combinations(range(5), 2):
        audit.check(len(generated([five_cycle, cycle(5, [i, j])], 5)) == 120,
                    "A 5-cycle and transposition did not generate S5", "prime_degree")
    # The prime-degree hypothesis is material: n=4 has counterexamples.
    audit.check(len(generated([cycle(4, [0, 1, 2, 3]), cycle(4, [0, 2])], 4)) == 8,
                "The composite-degree contrast must generate D4 of order 8", "prime_degree")
    return group_orders


def qadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def qmul(a, b):
    return a[0] * b[0] + 2 * a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def qinverse(a):
    denominator = a[0] ** 2 - 2 * a[1] ** 2
    if denominator == 0:
        raise ZeroDivisionError("Cannot invert zero in Q(sqrt(2))")
    return a[0] / denominator, -a[1] / denominator


def bimul(a, b):
    """Basis masks 0,1,2,3 denote 1,sqrt2,sqrt3,sqrt6."""
    result = [Fraction(0)] * 4
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            repeated = i & j
            rational = (2 if repeated & 1 else 1) * (3 if repeated & 2 else 1)
            result[i ^ j] += left * right * rational
    return tuple(result)


def sign_action(coefficients, signs):
    s, t = signs
    return tuple(a * factor for a, factor in
                 zip(coefficients, (1, s, t, s * t)))


def valid_root_permutations():
    roots = [(0, 1, 0, 0), (0, -1, 0, 0),
             (0, 0, 1, 0), (0, 0, -1, 0)]
    valid = []
    for p in itertools.permutations(range(4)):
        u, minus_u, v, minus_v = [roots[i] for i in p]
        if (bimul(u, u) == (2, 0, 0, 0)
                and bimul(v, v) == (3, 0, 0, 0)
                and tuple(-c for c in u) == minus_u
                and tuple(-c for c in v) == minus_v):
            valid.append(p)
    return valid


def field_checks(audit):
    values = [(Fraction(a), Fraction(b)) for a, b in itertools.product(range(-2, 3), repeat=2)]
    for a in values:
        conjugate = (a[0], -a[1])
        audit.check(qmul(a, conjugate) == (a[0] ** 2 - 2 * a[1] ** 2, 0),
                    "Quadratic conjugate/norm identity failed", "quadratic_field")
        if a != (0, 0):
            audit.check(qmul(a, qinverse(a)) == (1, 0),
                        "Quadratic field inverse failed", "quadratic_field")
        for b in values:
            product = qmul(a, b)
            audit.check(qmul(b, a) == product,
                        "Quadratic field multiplication did not commute", "quadratic_field")
            audit.check((product[0], -product[1]) == qmul(conjugate, (b[0], -b[1])),
                        "Conjugation failed to preserve multiplication", "quadratic_field")
    valid = valid_root_permutations()
    audit.check(set(valid) == {(0, 1, 2, 3), (1, 0, 2, 3),
                              (0, 1, 3, 2), (1, 0, 3, 2)},
                "Expected exactly four sign-preserving root permutations", "root_permutations")
    basis = [tuple(Fraction(i == j) for i in range(4)) for j in range(4)]
    signs = tuple(itertools.product((1, -1), repeat=2))
    for s in signs:
        for a, b in itertools.product(basis, repeat=2):
            audit.check(sign_action(bimul(a, b), s) == bimul(sign_action(a, s), sign_action(b, s)),
                        "Biquadratic sign map is not multiplicative", "root_permutations")
    subgroups = []
    for mask in range(1 << len(signs)):
        subgroup = frozenset(s for index, s in enumerate(signs) if mask & (1 << index))
        if (1, 1) not in subgroup:
            continue
        if all((a[0] * b[0], a[1] * b[1]) in subgroup
               for a, b in itertools.product(subgroup, repeat=2)):
            subgroups.append(subgroup)
    audit.check(len(subgroups) == 5, "V4 must have exactly five subgroups", "fixed_fields")
    fixed = {}
    for subgroup in subgroups:
        indices = tuple(i for i, a in enumerate(basis)
                        if all(sign_action(a, s) == a for s in subgroup))
        fixed[subgroup] = indices
        audit.check(len(indices) * len(subgroup) == 4,
                    "Fixed-field dimension times subgroup order must equal four", "fixed_fields")
    expected = {
        frozenset({(1, 1)}): (0, 1, 2, 3),
        frozenset({(1, 1), (-1, 1)}): (0, 2),
        frozenset({(1, 1), (1, -1)}): (0, 1),
        frozenset({(1, 1), (-1, -1)}): (0, 3),
        frozenset(signs): (0,),
    }
    audit.check(fixed == expected, "Incorrect fixed basis for a V4 subgroup", "fixed_fields")
    for first, second in itertools.product(subgroups, repeat=2):
        if first <= second:
            audit.check(set(fixed[second]) <= set(fixed[first]),
                        "Galois correspondence did not reverse inclusion", "fixed_fields")
    return {"valid_root_permutations": [list(p) for p in valid],
            "v4_subgroups": [{"signs": sorted(map(list, group)), "fixed_basis": list(indices)}
                             for group, indices in fixed.items()]}


def learning_example_checks(audit):
    """Exact checks for the short worked examples in the learning-layer edit."""
    audit.check(qmul((3, 2), (3, -2)) == (1, 0),
                "The concrete quadratic inverse is incorrect", "learning_examples")
    audit.check(qmul((-1, 1), (1, 1)) == (1, 0),
                "The Bezout inverse (sqrt2+1)^-1 is incorrect", "learning_examples")
    s = cycle(3, [0, 1])
    t = cycle(3, [1, 2])
    r = cycle(3, [0, 1, 2])
    audit.check(compose(s, t) == r and compose(t, s) == inverse(r),
                "The noncommuting cycle example is incorrect", "learning_examples")
    audit.check(commutator(s, t) == inverse(r),
                "The S3 commutator must be (132)", "learning_examples")
    audit.check(compose(compose(s, t), s) == cycle(3, [0, 2]),
                "Adjacent path swaps must give the endpoint swap", "learning_examples")
    h = {identity(3), t}
    audit.check({compose(r, p) for p in h} == {r, s},
                "The S3 left coset is incorrect", "learning_examples")
    audit.check({compose(p, r) for p in h} == {r, cycle(3, [0, 2])},
                "The S3 right coset is incorrect", "learning_examples")
    audit.check({compose(compose(r, p), inverse(r)) for p in h}
                == {identity(3), cycle(3, [0, 2])},
                "The nonnormal S3 subgroup conjugate is incorrect", "learning_examples")
    # V4 is encoded by the independent signs of sqrt2 and sqrt3.
    v4 = set(itertools.product((1, -1), repeat=2))
    kernel = {(1, 1), (-1, 1)}
    tau_h = {(a, -b) for a, b in kernel}
    audit.check(kernel.isdisjoint(tau_h) and kernel | tau_h == v4,
                "The two V4 cosets must partition the group", "learning_examples")
    audit.check({b for a, b in kernel} == {1} and {b for a, b in tau_h} == {-1},
                "The cosets must give the two restrictions on Q(sqrt3)", "learning_examples")
    for a, b in itertools.product(v4, repeat=2):
        audit.check((a[1] * b[1]) == (a[0] * b[0], a[1] * b[1])[1],
                    "Restriction must preserve composition", "learning_examples")
    roots = [(0, 1, 0, 0), (0, -1, 0, 0),
             (0, 0, 1, 0), (0, 0, -1, 0)]
    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    values = []
    for first, second in pairings:
        a = bimul(roots[first[0]], roots[first[1]])
        b = bimul(roots[second[0]], roots[second[1]])
        values.append(tuple(x + y for x, y in zip(a, b)))
    audit.check(values == [(-5, 0, 0, 0), (0, 0, 0, 2), (0, 0, 0, -2)],
                "The concrete quartic resolvent values are incorrect", "learning_examples")
    # Expand product (t-y_i) using the four-coordinate exact field arithmetic.
    one, zero = (1, 0, 0, 0), (0, 0, 0, 0)
    polynomial = [one]
    for value in values:
        following = [zero] * (len(polynomial) + 1)
        for power, coefficient in enumerate(polynomial):
            constant = bimul(coefficient, tuple(-x for x in value))
            following[power] = tuple(a + b for a, b in zip(following[power], constant))
            following[power + 1] = tuple(a + b for a, b in zip(following[power + 1], coefficient))
        polynomial = following
    audit.check(polynomial == [(-120, 0, 0, 0), (-24, 0, 0, 0),
                               (5, 0, 0, 0), one],
                "The quartic resolvent must be t^3+5t^2-24t-120", "learning_examples")
    return {"status": "passed", "quartic_resolvent_ascending": [-120, -24, 5, 1],
            "v4_coset_count": 2, "s3_derived_orders": [6, 3, 1]}


def polynomial_value(coefficients, x):
    result = 0
    for coefficient in reversed(coefficients):
        result = result * x + coefficient
    return result


def polynomial_shift(coefficients, shift):
    result = [0] * len(coefficients)
    for degree, coefficient in enumerate(coefficients):
        for power in range(degree + 1):
            result[power] += coefficient * comb(degree, power) * shift ** (degree - power)
    return result


def quintic_checks(audit):
    coefficients = [1, -5, 0, 0, 0, 1]
    shifted = polynomial_shift(coefficients, -1)
    audit.check(shifted == [5, 0, -10, 10, -5, 1],
                "Incorrect expansion of f(x-1)", "quintic")
    audit.check(all(c % 5 == 0 for c in shifted[:-1])
                and shifted[-1] % 5 != 0 and shifted[0] % 25 != 0,
                "The translated quintic does not meet Eisenstein at five", "quintic")
    for x in [Fraction(-2), Fraction(-1), Fraction(0), Fraction(1), Fraction(2)]:
        audit.check(polynomial_value(shifted, x) == polynomial_value(coefficients, x - 1),
                    "Translation identity failed", "quintic")
    audit.check(polynomial_value(coefficients, -1) == 5
                and polynomial_value(coefficients, 1) == -3,
                "Incorrect critical-point values", "quintic")
    audit.check(polynomial_value(coefficients, 0) == 1
                and polynomial_value(coefficients, Fraction(1, 2)) == Fraction(-47, 32),
                "The two explicit bisection decisions are incorrect", "learning_examples")
    intervals = [(Fraction(-1542, 1000), Fraction(-1541, 1000)),
                 (Fraction(200, 1000), Fraction(201, 1000)),
                 (Fraction(1440, 1000), Fraction(1441, 1000))]
    for left, right in intervals:
        audit.check(polynomial_value(coefficients, left) * polynomial_value(coefficients, right) < 0,
                    "Exact rational bracket does not change sign", "quintic")
    derivative = [degree * coefficients[degree] for degree in range(1, len(coefficients))]
    audit.check(derivative == [-5, 0, 0, 0, 5], "Incorrect derivative coefficients", "quintic")
    # This identity supplies the exact sign analysis, rather than sampling a plot.
    audit.check([5 * c for c in [-1, 0, 0, 0, 1]] == derivative,
                "f'(x) must equal 5(x^4-1)", "quintic")
    return {"coefficients_ascending": coefficients, "translated_ascending": shifted,
            "critical_values": {"-1": 5, "1": -3},
            "real_root_brackets": [[str(a), str(b)] for a, b in intervals]}


def sympy_checks(audit, mode):
    if mode == "off":
        return {"status": "disabled"}
    if importlib.util.find_spec("sympy") is None:
        if mode == "required":
            raise RuntimeError("SymPy requested but not installed in this Python runtime")
        return {"status": "unavailable; standard-library checks still executed"}
    import sympy as sp
    from sympy.polys.numberfields import galois_group
    x = sp.symbols("x")
    f = x ** 5 - 5 * x + 1
    audit.check(sp.Poly(f, x).is_irreducible, "SymPy irreducibility cross-check failed", "sympy")
    audit.check(sp.expand(f.subs(x, x - 1)) == x ** 5 - 5 * x ** 4 + 10 * x ** 3 - 10 * x ** 2 + 5,
                "SymPy shift cross-check failed", "sympy")
    group, _ = galois_group(f, x)
    audit.check(group.order() == 120 and not group.is_solvable,
                "SymPy quintic Galois-group cross-check failed", "sympy")
    audit.check(sp.gcd(f, sp.diff(f, x)) == 1, "The quintic must be separable", "sympy")
    u3, v3 = (-1 + sp.sqrt(5)) / 2, (-1 - sp.sqrt(5)) / 2
    audit.check(sp.simplify(u3 + v3) == -1 and sp.simplify(u3 * v3) == -1,
                "The numerical Cardano example has incompatible cubes", "sympy")
    audit.check(sp.degree(sp.minpoly(sp.sqrt(2) + sp.I, x), x) == 4,
                "The Q(sqrt2,i) product-basis example must have degree four", "sympy")
    return {"status": "passed", "version": sp.__version__,
            "quintic_galois_group_order": int(group.order()), "solvable": bool(group.is_solvable)}


def canonical_hash(value):
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def python_asset_checks(audit, path):
    spec = importlib.util.spec_from_file_location("galois_reference_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("compose", "inverse", "commutator", "parity", "group_closure",
                "derived_subgroup", "derived_series")
    audit.check(all(callable(getattr(module, name, None)) for name in required),
                "Python reference is missing a required finite-group API", "python_asset")
    for n in range(2, 6):
        group = set(itertools.permutations(range(n)))
        for p in group:
            audit.check(module.inverse(p) == inverse(p)
                        and module.parity(p) == parity(p),
                        "Python reference inverse/parity disagrees", "python_asset")
            for q in group:
                audit.check(module.compose(p, q) == compose(p, q),
                            "Python reference composition disagrees", "python_asset")
                audit.check(module.commutator(p, q) == commutator(p, q),
                            "Python reference commutator disagrees", "python_asset")
        generators = [cycle(n, [i, i + 1]) for i in range(n - 1)]
        audit.check(set(module.group_closure(generators, n)) == group,
                    "Python reference generator closure disagrees", "python_asset")
        audit.check(set(module.derived_subgroup(group)) == derived_subgroup(group, n),
                    "Python reference derived subgroup disagrees", "python_asset")
        audit.check([len(h) for h in module.derived_series(group)] == derived_orders(group, n),
                    "Python reference derived series disagrees", "python_asset")
    return {"status": "passed", "permutation_pairs": 15016,
            "scope": "all products and commutators in S2,S3,S4,S5; derived groups"}


def javascript_asset_checks(audit, path):
    # Node reads the actual browser bundle's pure exports.  Hashing complete
    # tables keeps the subprocess output compact without sampling their entries.
    script = r"""
const model = require(process.argv[1]);
const crypto = require('node:crypto');
const hash = x => crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex');
const groups = {};
for(let n=2;n<=5;n++) {
  const p=model.permutations(n), table=fn=>p.map(a=>p.map(b=>fn(a,b)));
  groups[n]={permutations:p,inverses:p.map(model.inverse),parities:p.map(model.parity),
    productHash:hash(table(model.compose)),commutatorHash:hash(table(model.commutator)),
    derived:model.derivedSeries(p).map(g=>g.length)};
}
const pairs=[];
for(let a=-3;a<=3;a++)for(let b=-3;b<=3;b++)pairs.push(model.pair(a,b));
const inverses=[];
for(let a=-9;a<=9;a++)for(let b=-9;b<=9;b++)
  inverses.push({pair:[a,b],value:model.inversePair(model.pair(a,b))});
const all4=model.permutations(4);
const v4=Array.from({length:16},(_,mask)=>{
  const generators=[0,1,2,3].filter(g=>mask&(1<<g)),subgroup=model.v4Closure(generators);
  return {generators,subgroup,basis:model.fixedBasis(subgroup)};
});
const bisections=[];
for(const name of ['obstruction','radical'])for(let root=0;root<(name==='obstruction'?3:1);root++)
  for(const count of [0,1,10,20,40]) {
    const result=model.bisection(name,root,count);
    bisections.push({name,root,count,...result,formatted:model.formatBisectionBracket(result.lo,result.hi)});
  }
console.log(JSON.stringify({groups,quadratic:{pairs,
  addHash:hash(pairs.map(a=>pairs.map(b=>model.addPair(a,b)))),
  multiplyHash:hash(pairs.map(a=>pairs.map(b=>model.mulPair(a,b)))),
  conjugates:pairs.map(model.conjugatePair),inverses},
  rootPermutations:all4.map(p=>({p,...model.rootPermutationChecks(p)})),v4,bisections}));
"""
    completed = subprocess.run(["node", "-e", script, str(path)], check=True,
                               capture_output=True, text=True, timeout=30)
    data = json.loads(completed.stdout)
    for n in range(2, 6):
        group = list(itertools.permutations(range(n)))
        row = data["groups"][str(n)]
        audit.check(row["permutations"] == [list(p) for p in group],
                    "Browser permutation enumeration disagrees", "javascript_asset")
        audit.check(row["inverses"] == [list(inverse(p)) for p in group]
                    and row["parities"] == [parity(p) for p in group],
                    "Browser inverse/parity disagrees", "javascript_asset")
        for field, operation in [("productHash", compose), ("commutatorHash", commutator)]:
            expected = canonical_hash([[operation(p, q) for q in group] for p in group])
            audit.check(row[field] == expected,
                        f"Browser complete {field} table disagrees for S{n}", "javascript_asset")
        audit.check(row["derived"] == derived_orders(set(group), n),
                    "Browser derived series disagrees", "javascript_asset")
    q = data["quadratic"]
    pairs = list(itertools.product(range(-3, 4), repeat=2))
    audit.check(q["pairs"] == [list(p) for p in pairs],
                "Browser quadratic inputs changed", "javascript_asset")
    audit.check(q["addHash"] == canonical_hash([[qadd(a, b) for b in pairs] for a in pairs]),
                "Browser complete quadratic-addition table disagrees", "javascript_asset")
    audit.check(q["multiplyHash"] == canonical_hash([[qmul(a, b) for b in pairs] for a in pairs]),
                "Browser complete quadratic-product table disagrees", "javascript_asset")
    audit.check(q["conjugates"] == [[a, -b] for a, b in pairs],
                "Browser quadratic conjugation disagrees", "javascript_asset")
    audit.check(len(q["inverses"]) == 361,
                "All 361 allowed coefficient pairs must be checked", "javascript_asset")
    for row in q["inverses"]:
        a = tuple(map(Fraction, row["pair"]))
        if a == (0, 0):
            audit.check(row["value"] is None, "Browser must not invert zero", "javascript_asset")
        else:
            actual = tuple(Fraction(num, den) for num, den in row["value"])
            audit.check(actual == qinverse(a), "Browser exact quadratic inverse disagrees", "javascript_asset")
    valid = set(valid_root_permutations())
    audit.check(len(data["rootPermutations"]) == 24,
                "All root permutations must be checked", "javascript_asset")
    for row in data["rootPermutations"]:
        audit.check(row["valid"] == (tuple(row["p"]) in valid),
                    "Browser root-permutation decision disagrees", "javascript_asset")
    for row in data["v4"]:
        masks = row["generators"]
        subgroup = {0}
        for g in masks:
            subgroup |= {h ^ g for h in tuple(subgroup)}
        basis = [b for b in range(4) if all((g & b).bit_count() % 2 == 0 for g in subgroup)]
        audit.check(row["subgroup"] == sorted(subgroup) and row["basis"] == basis,
                    "Browser V4 closure/fixed basis disagrees", "javascript_asset")
    for row in data["bisections"]:
        # Every listed midpoint is a dyadic rational represented exactly by the
        # binary64 coordinate.  Re-evaluate endpoint signs with Fraction.
        coefficients = [1, -5, 0, 0, 0, 1] if row["name"] == "obstruction" else [-2, 0, 0, 0, 0, 1]
        lo, hi = Fraction(row["lo"]), Fraction(row["hi"])
        audit.check(lo <= hi and polynomial_value(coefficients, lo) * polynomial_value(coefficients, hi) <= 0,
                    "Browser bisection lost an exact sign-changing bracket", "javascript_asset")
        initial_width = Fraction(row["initial"][1]) - Fraction(row["initial"][0])
        audit.check(hi - lo == initial_width / (2 ** row["count"]),
                    "Browser bisection interval did not halve at each step", "javascript_asset")
        displayed = row["formatted"][1:-1].split(", ")
        audit.check(len(displayed) == 2 and float(displayed[0]) == row["lo"]
                    and float(displayed[1]) == row["hi"],
                    "Browser bracket labels must round-trip to their binary64 endpoints", "javascript_asset")
        audit.check(lo == hi or displayed[0] != displayed[1],
                    "Distinct bisection endpoints must not display identical labels", "javascript_asset")
    return {"status": "passed", "permutation_pairs": 15016,
            "quadratic_operation_pairs": len(pairs) ** 2,
            "quadratic_inverses": len(q["inverses"]),
            "root_permutations": len(data["rootPermutations"]),
            "v4_generator_subsets": len(data["v4"]),
            "exact_bisection_brackets": len(data["bisections"])}


def asset_crosschecks(audit, require_assets):
    result = {}
    for label, relative, checker in [
            ("python", "assets/code/galois-reference.py", python_asset_checks),
            ("javascript", "assets/js/article-labs-galois.js", javascript_asset_checks)]:
        path = ROOT / relative
        if not path.exists():
            if require_assets:
                raise AssertionError(f"Required tutorial asset missing: {relative}")
            result[label] = {"status": "not present yet", "path": relative}
            continue
        result[label] = {"path": relative, **checker(audit, path)}
    return result


def article_checks(audit, sympy_mode):
    """Execute available source fences; absent translations are not required."""
    result = {}
    stem = "2026-10-25-Galois-Theory-From-Scratch"
    for language, suffix in [("en", ""), ("ko", "-KR")]:
        path = ROOT / "_posts/2026-4Q" / f"{stem}{suffix}.md"
        if not path.exists():
            result[language] = {"status": "not present yet"}
            continue
        source = path.read_text(encoding="utf-8")
        blocks = re.findall(r"^```python\s*\n(.*?)^```\s*$", source, flags=re.M | re.S)
        namespace = {"__name__": f"galois_article_{language}_audit"}
        output, executed, skipped = io.StringIO(), 0, []
        for index, code in enumerate(blocks, 1):
            filename = f"{path.name}:python-fence-{index}"
            parsed = ast.parse(code, filename=filename)
            imports = set()
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    imports.update(name.name.partition(".")[0] for name in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.partition(".")[0])
            third_party = imports - set(sys.stdlib_module_names)
            if third_party - {"sympy"}:
                raise AssertionError(f"Unexpected article dependency: {third_party}")
            if "sympy" in third_party and (sympy_mode == "off"
                    or importlib.util.find_spec("sympy") is None):
                if sympy_mode == "required":
                    raise AssertionError("A required SymPy article fence cannot execute")
                skipped.append({"fence": index, "reason": "optional SymPy unavailable or disabled"})
                continue
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(parsed, filename, "exec"), namespace)
            executed += 1
            audit.check(True, f"Executed {filename}", "article_python")
        if "qmul" in namespace:
            for a in itertools.product(range(-2, 3), repeat=2):
                for b in itertools.product(range(-2, 3), repeat=2):
                    audit.check(tuple(namespace["qmul"](a, b)) == qmul(a, b),
                                "Article quadratic multiplication disagrees", "article_python")
        if "qinv" in namespace:
            for a in itertools.product(range(-2, 3), repeat=2):
                if a != (0, 0):
                    actual = tuple(namespace["qinv"](a))
                    audit.check(actual == qinverse(tuple(map(Fraction, a))),
                                "Article exact quadratic inverse disagrees", "article_python")
        # This example must work when copied alone, without imports from section 2.
        bisection_blocks = [code for code in blocks if "def bisect_exact(" in code]
        audit.check(len(bisection_blocks) == 1,
                    "Expected one standalone exact-bisection example", "article_python")
        standalone, printed = {}, io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(bisection_blocks[0], f"{path.name}:bisection-alone", "exec"), standalone)
        rows = printed.getvalue().splitlines()
        brackets = [(-3, -1), (-1, 1), (1, 3)]
        audit.check(len(rows) == len(brackets),
                    "Bisection example must print one result per real-root bracket", "article_python")
        for bracket, row in zip(brackets, rows):
            match = re.fullmatch(r"root ≈ (-?\d+\.\d{12}); error ≤ (\d+e-15)", row)
            audit.check(match is not None, "Unrecognized bisection error-bound output", "article_python")
            displayed, printed_bound = map(Fraction, match.groups())
            lo, hi = standalone["bisect_exact"](standalone["quintic"], *bracket)
            midpoint = (lo + hi) / 2
            required_bound = abs(displayed - midpoint) + (hi - lo) / 2
            audit.check(required_bound <= printed_bound < required_bound + Fraction(1, 10**15),
                        "Printed error bound must include decimal rounding and round upward", "article_python")
            audit.check(printed_bound >= max(lo - displayed, displayed - hi, Fraction(0)),
                        "Printed decimal lies farther from the root bracket than its claimed error", "article_python")
        result[language] = {"status": "passed", "python_fences": len(blocks),
                            "executed": executed, "skipped": skipped,
                            "stdout_characters": len(output.getvalue()),
                            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sympy", choices=["auto", "required", "off"], default="auto")
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--output", type=Path, help="Also save the generated JSON report")
    args = parser.parse_args()
    audit = Audit()
    result = {"status": "passed", "derived_orders": permutation_checks(audit),
              "field_examples": field_checks(audit), "quintic": quintic_checks(audit),
              "learning_examples": learning_example_checks(audit),
              "sympy": sympy_checks(audit, args.sympy),
              "asset_crosschecks": asset_crosschecks(audit, args.require_assets),
              "article_python": article_checks(audit, args.sympy)}
    result["assertions"] = sum(audit.counts.values())
    result["assertions_by_group"] = dict(audit.counts)
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
