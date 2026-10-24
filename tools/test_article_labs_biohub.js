'use strict';
const assert = require('node:assert/strict');
const lab = require('../assets/js/article-labs-biohub.js');
const near = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
near(lab.distance([0, 0, 0], [1, 0, 0]), 1.625);
near(lab.distance([0, 0, 0], [0, 0, 4]), 1.625);
const matching = lab.matchPoints([[0, 0, 0], [0, 0, 10]], [[0, 0, 1], [0, 0, 11]]);
assert.equal(matching.pairs.length, 2); assert.equal(new Set(matching.pairs.map(p => p[1])).size, 2);
assert.equal(lab.matchPoints([[0, 0, 0]], [[5, 0, 0]]).pairs.length, 0);
// The annotated source makes its outgoing prediction evaluable even without a matched target.
const missingTarget = lab.lineageCounts([0, 1, 2], new Map([[0, 0], [2, 2]]));
assert.deepEqual(missingTarget, { tp: 2, fp: 1, fn: 1 });
near(lab.edgeScore(missingTarget.tp, missingTarget.fp, missingTarget.fn, 6, 6).adjusted, .5);
near(lab.edgeScore(3, 0, 0, 6, 6).adjusted, 1);
assert.ok(lab.edgeScore(3, 0, 0, 7, 6).adjusted < 1);
for (const p of [.05, .17, .4, .8, 1]) {
  const r = lab.divisionChange(32, 20, 48, 8, p);
  assert.equal(r.weightedDelta > 0, p > r.precisionBoundary);
  near((32 + 8) / ((32 + 8) + 20 + r.addedFP + 40), r.next);
}
const at = lab.divisionChange(32, 20, 48, 8, .32 / 1.32); near(at.weightedDelta, 0);
const unchanged = lab.graphStages(false, 3), changed = lab.graphStages(true, 3);
assert.ok(changed.before.tp > unchanged.before.tp);
assert.ok(changed.after.score < unchanged.after.score);
assert.ok(lab.graphStages(true, 2).after.score > unchanged.after.score);
for (let shift = 0; shift <= 9; shift += .25) assert.equal(lab.jitterLinks(shift, true).correct, 4);
assert.ok(lab.jitterLinks(9, false).correct < 4);
for (const [_, a, b] of lab.tailRecords) { near(lab.mixture(a, b, 0), b); near(lab.mixture(a, b, 1), a); near(lab.mixture(a, b, -b / (a - b)), 0); }
const fields = lab.alignFields([-10, -8, -5], [-2, 0, 4], .475, true);
near(fields.aligned.reduce((a, b) => a + b, 0) / 3, -23 / 3);
assert.equal(lab.foldRows('embryo').every(r => !((r[1] && r[3]) || (r[2] && r[4]))), true);
assert.equal(lab.foldRows('balanced').every(r => r[1] && r[2] && r[3] && r[4]), true);
const factories = {};
global.window = { ArticleLabs: { register: (id, factory) => { factories[id] = factory; } } };
delete require.cache[require.resolve('../assets/js/article-labs-biohub.js')]; require('../assets/js/article-labs-biohub.js'); delete global.window;
function node() { return { append() {}, replaceChildren() {}, addEventListener() {}, textContent: '' }; }
let renders = 0;
for (const [id, factory] of Object.entries(factories)) for (const ko of [false, true]) {
  const controls = [], ctx = { root: { dataset: { preset: id === 'biohub-folds' ? 'embryo' : '' } }, ko, text: (en, kr) => ko ? kr : en, el: node, svg: node, view: node(), status: node(), metric: node, format: (v, n = 4) => Number(v).toFixed(n), table: () => node(),
    control(en, kr, opts) { const input = { ...node(), value: String(opts.value), checked: Boolean(opts.value), opts }; controls.push(input); return input; }, onChange(fn) { ctx.render = fn; fn(); },
    plot(spec) { for (const s of spec.series || []) for (const [x, y] of s.points) { assert.ok(Number.isFinite(x) && Number.isFinite(y)); assert.ok(x >= spec.x[0] - 1e-8 && x <= spec.x[1] + 1e-8, id); assert.ok(y >= spec.y[0] - 1e-8 && y <= spec.y[1] + 1e-8, id); } return node(); }
  };
  factory(ctx); renders++;
  for (const input of controls) {
    const original = input.value, checked = input.checked;
    const values = input.opts.type === 'select' ? input.opts.options.map(x => x.value) : input.opts.type === 'checkbox' ? [false, true] : [input.opts.min, input.opts.max];
    for (const v of values) { if (input.opts.type === 'checkbox') input.checked = v; else input.value = String(v); ctx.render(); renders++; }
    input.value = original; input.checked = checked;
  }
}
console.log(`BioHub matching, graph, split and score invariants and ${renders} bilingual factory/control renders passed (9 factories).`);
