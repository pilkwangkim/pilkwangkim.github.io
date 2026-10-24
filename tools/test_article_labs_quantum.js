'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const q = require('../assets/js/article-labs-quantum.js');
let checks = 0;
function near(actual, expected, tolerance = 1e-12) { assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`); checks += 1; }
near(q.phase(0).p0, 1); near(q.phase(Math.PI).p1, 1); near(q.phase(Math.PI / 2).p0, 0.5);
for (let i = 0; i <= 360; i += 1) { const r = q.phase(i * Math.PI / 180); near(r.p0 + r.p1, 1); }
for (let stage = 0; stage <= 5; stage += 1) for (const clean of [false, true]) for (const phase of [false, true]) {
  const r = q.cleanup(stage, clean, phase); near(r.probabilities.reduce((a, b) => a + b), 1); near(r.p0 + r.p1, 1);
}
near(q.cleanup(5, true, true).p1, 1); near(q.cleanup(5, true, true).scratch1, 0);
near(q.cleanup(5, false, true).p1, 0.5); near(q.cleanup(5, true, false).p0, 1);
const normalized64 = q.normalize(64); assert.equal(normalized64.s, 7); near(normalized64.m, 1); checks += 1;
for (const x of [1, 2, 3, 4, 16, 63, 64, 65, 4095, 4096, 1048575, 1048576, 3133452]) {
  const r = q.normalize(x); assert.ok(r.m >= 1 && r.m < 4); near(2 ** (r.s - 10) / Math.sqrt(r.m), 1 / Math.sqrt(x));
  for (const e0 of [-0.01, 0, 0.0001, 0.01]) for (const bits of [8, 17, 28]) {
    const f = q.fixedpoint(x, e0, bits), storedError = f.y0 / f.exact - 1;
    near(f.errors[2], -1.5 * storedError ** 2 - 0.5 * storedError ** 3, 8e-16);
    assert.ok(f.next >= f.rounded && f.next - f.rounded < 2 ** -bits + 1e-15); checks += 1;
  }
}
assert.throws(() => q.normalize(0)); assert.throws(() => q.normalize(3133453)); assert.throws(() => q.normalize(1.5)); checks += 3;
near(q.gfMul(2, 2), 3); near(q.gfPow(2, 3), 1); near(q.gfMul(3, 2), 1);
for (let a = 0; a < 4; a += 1) for (let b = 0; b < 4; b += 1) {
  assert.equal(q.gfMul(a, b), q.gfMul(b, a)); if (b) near(q.gfDiv(q.gfMul(a, b), b), a); checks += 1;
}
for (let position = 0; position < 3; position += 1) for (let value = 1; value < 4; value += 1) for (let mask = 0; mask < 4; mask += 1) {
  const r = q.syndrome(position, value, mask); assert.equal(r.position, position); assert.equal(r.value, value);
  assert.equal(r.sign, ((value & mask).toString(2).replace(/0/g, '').length % 2) ? -1 : 1); checks += 3;
}
assert.deepEqual(q.decode(0, 0), { position: -1, value: 0 }); assert.throws(() => q.gfDiv(1, 0)); checks += 2;
const d = q.syndrome(1, 3, 2); assert.equal(d.s0, 1); assert.equal(d.s1, 2); assert.equal(d.sign, -1); checks += 3;
const pi4 = q.givens(Math.PI / 4, Math.PI / 4, false, 12);
[0.5, Math.SQRT1_2, 0.5].forEach((v, i) => near(pi4.actual[i], v));
const rev = q.givens(Math.PI / 4, Math.PI / 4, true, 12);
[0.5, 0.5, Math.SQRT1_2].forEach((v, i) => near(rev.actual[i], v));
for (let a = 0; a < 360; a += 13) for (let b = 0; b < 360; b += 17) for (const reverse of [false, true]) for (const bits of [2, 5, 12]) {
  const r = q.givens(a * Math.PI / 180, b * Math.PI / 180, reverse, bits);
  near(r.norm, 1); assert.ok(r.error <= 2 * Math.PI / 2 ** bits + 1e-12); checks += 1;
}
near(q.resource('dqi', 0.9, 1.2, 1).ratio, 0.972); near(q.resource('invsqrt', 1.2, 0.9, 2).ratio, 1.08);
for (const task of ['dqi', 'invsqrt', 'chem']) near(q.resource(task, 1, 1, 1).overall, 1);
const counts = JSON.parse(fs.readFileSync(path.join(__dirname, '../assets/img/posts/quantum-circuit-compilation/invsqrt-stage-counts.json')));
assert.equal(counts.forward.reduce((a, b) => a + b) + counts.reverse.reduce((a, b) => a + b), 7735); checks += 1;
console.log(`Quantum labs: ${checks} numerical checks passed (phase, unitary cleanup, normalization, Newton, GF(4), rotation order/quantization, costs).`);
