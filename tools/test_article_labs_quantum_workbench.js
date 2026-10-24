'use strict';
const assert = require('node:assert/strict');
const q = require('../assets/js/article-labs-quantum-workbench.js');
let checks = 0;
const near = (a, b, tolerance = 1e-10) => { assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`); checks += 1; };
const norm = state => state.reduce((sum, a) => sum + a[0] ** 2 + a[1] ** 2, 0);
const X = q.gate('X', 0), H = q.gate('H', 0), Z = q.gate('Z', 0), S = q.gate('S', 0), T = q.gate('T', 0);
// Q0 is the most significant bit, not the least significant simulator wire.
assert.deepEqual(q.simulate([X], 2, 0), q.basis(2, 2)); checks += 1;
assert.deepEqual(q.simulate([q.gate('CX', 1, [0])], 2, 2), q.basis(2, 3)); checks += 1;
assert.deepEqual(q.simulate([q.gate('CCX', 2, [0, 1])], 3, 6), q.basis(3, 7)); checks += 1;
near(q.simulate([S], 2, 2)[2][1], 1);
near(q.simulate([T], 2, 2)[2][0], Math.SQRT1_2);
near(q.simulate([T], 2, 2)[2][1], Math.SQRT1_2);
for (const n of [2, 3]) for (const gate of q.palette(n)) for (let input = 0; input < 2 ** n; input += 1) {
  near(norm(q.simulate([gate], n, input)), 1);
  // Every supplied gate has its expected finite inverse, including complex S and T.
  const repetitions = gate.op === 'T' ? 8 : gate.op === 'S' ? 4 : 2;
  assert.equal(q.compare(Array.from({ length: repetitions }, () => gate), [], n).equivalent, true); checks += 1;
}
assert.equal(q.compare([H, Z, H], [X], 2).equivalent, true); checks += 1;
assert.equal(q.compare([S, S], [Z], 2).equivalent, true); checks += 1;
assert.equal(q.compare([T, T], [S], 2).equivalent, true); checks += 1;
// One phase for the whole map accepts -I, rather than incorrectly rejecting global phase.
const minusIdentity = q.compare([X, Z, X, Z], [], 2);
assert.equal(minusIdentity.equivalent, true); near(minusIdentity.sharedPhase[0], -1); checks += 1;
// But independent per-input phase alignment would incorrectly accept Z vs I.
const phaseFailure = q.compare([], [Z], 2);
assert.equal(phaseFailure.equivalent, false); near(phaseFailure.probabilityError, 0);
assert.equal(phaseFailure.witness, 2); assert.ok(phaseFailure.coherent); checks += 3;
near(norm(phaseFailure.coherent.input), 1);
assert.ok(phaseFailure.coherent.actual.some((a, i) => Math.abs(a[0] - phaseFailure.coherent.expected[i][0]) > 1)); checks += 1;
// Relative complex phase is also rejected despite identical basis probabilities.
const sFailure = q.compare([S], [], 2);
assert.equal(sFailure.equivalent, false); assert.ok(sFailure.coherent); near(sFailure.probabilityError, 0); checks += 2;
const p = q.PUZZLES.find(p => p.id === 'scratch');
const dirty = q.compare(p.initial, p.target, p.n, p.data);
assert.equal(dirty.equivalent, false); near(dirty.leakage, 1); assert.equal(dirty.witness, 3); checks += 2;
assert.equal(q.compare(p.target, p.target, 3, 2).equivalent, true); checks += 1;
assert.equal(q.compare([q.gate('CZ', 1, [0])], p.target, 3, 2).equivalent, true); checks += 1;
// A scratch-only phase is irrelevant on the allowed clean input subspace.
assert.equal(q.compare([q.gate('Z', 2)], [], 3, 2).equivalent, true); checks += 1;
for (const puzzle of q.PUZZLES) {
  assert.equal(q.compare(puzzle.target, puzzle.target, puzzle.n, puzzle.data).equivalent, true);
  near(norm(q.simulate(puzzle.initial, puzzle.n, 0)), 1); checks += 1;
}
const original = [H, Z], replaced = q.edit(original, 'replace', 1, X, 2);
assert.deepEqual(original, [H, Z]); assert.deepEqual(replaced, [H, X]); checks += 2;
assert.deepEqual(q.edit(original, 'insert', 2, X, 2), [H, Z, X]); checks += 1;
assert.deepEqual(q.edit(original, 'remove', 0, null, 2), [Z]); checks += 1;
assert.deepEqual(q.edit(original, 'right', 0, null, 2), [Z, H]); checks += 1;
assert.deepEqual(q.edit(original, 'left', 1, null, 2), [Z, H]); checks += 1;
for (const action of [() => q.edit(original, 'remove', 2, null, 2), () => q.edit(original, 'left', 0, null, 2),
  () => q.edit(original, 'right', 1, null, 2), () => q.edit(Array(12).fill(H), 'insert', 12, X, 2),
  () => q.validateCircuit([q.gate('CX', 0, [0])], 2), () => q.validateCircuit([q.gate('CCX', 2, [0, 0])], 3),
  () => q.validateCircuit([q.gate('eval', 0)], 2), () => q.validateCircuit([q.gate('H', 3)], 3),
  () => q.restore({ puzzle: 'unknown', circuit: [] }), () => q.restore({ puzzle: 'phase', circuit: Array(13).fill(H) })]) {
  assert.throws(action, RangeError); checks += 1;
}
const saved = { puzzle: 'scratch', circuit: p.target, selected: 2, step: 1, input: '3' };
assert.deepEqual(q.restore(JSON.parse(JSON.stringify(saved))), saved); checks += 1;
const restored = q.restore({ puzzle: 'phase', circuit: [], selected: 999, step: -8, input: 'plus' });
assert.equal(restored.selected, 0); assert.equal(restored.step, 0); assert.equal(restored.input, 'plus'); checks += 3;
// Gates on independent wires commute, but reversing H and Z on one wire does not.
assert.equal(q.compare([H, q.gate('S', 1)], [q.gate('S', 1), H], 2).equivalent, true); checks += 1;
assert.equal(q.compare([H, Z], [Z, H], 2).equivalent, false); checks += 1;
console.log(`Quantum workbench: ${checks} checks passed (complex unitary gates, common global phase, coherent counterexamples, clean ancilla, bounded immutable edits and shared-state validation).`);
