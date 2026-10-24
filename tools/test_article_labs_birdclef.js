'use strict';
const assert = require('node:assert/strict');
const lab = require('../assets/js/article-labs-birdclef.js');
const near = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
assert.deepEqual(lab.columnRanks([[.9, .1], [.4, .8], [.2, .2]]), [[1, 1 / 3], [2 / 3, 1], [1 / 3, 2 / 3]]);
assert.deepEqual(lab.ranks([1, 1, 3]), [.5, .5, 1]);
assert.deepEqual(lab.topIndices([.8, .8, .4], 1), [0]);
const smoothed = lab.smooth([[.1, .9, .5]], [[0, 1]], .2);
near(smoothed[0][0], .18); near(smoothed[0][1], .82); near(smoothed[0][2], .5);
const gates = [1, 0, .7, 1, 0, .4, 1, 0, .8, 0, 1, .5];
for (const weight of [0, .03, .3]) for (const budget of [0, .001, .004, .05]) for (const anchorK of [0, 6, 12]) {
  const r = lab.correction(lab.anchor, lab.side, { anchorK, sideK: 4, tau: .55, weight, budget, gates });
  assert.ok(r.D <= budget + 1e-12);
  r.A.forEach((row, i) => row.forEach((v, c) => {
    assert.ok(r.attempted[i][c] >= 0 && r.attempted[i][c] <= 1);
    if (!gates[c] || !r.mask[i][c]) near(r.attempted[i][c], v);
    if (!r.accepted) near(r.final[i][c], v);
  }));
  assert.ok(r.top3 >= 0 && r.top3 <= 1 && r.top10 >= 0 && r.top10 <= 1);
}
const factories = {};
global.window = { ArticleLabs: { register: (id, factory) => { factories[id] = factory; } } };
delete require.cache[require.resolve('../assets/js/article-labs-birdclef.js')]; require('../assets/js/article-labs-birdclef.js'); delete global.window;
function node() { return { append() {}, replaceChildren() {}, addEventListener() {}, textContent: '' }; }
let renders = 0;
for (const ko of [false, true]) {
  const controls = [], ctx = { root: { dataset: {} }, ko, text: (en, kr) => ko ? kr : en, el: node, svg: node, view: node(), status: node(), metric: node, format: (v, n = 4) => Number(v).toFixed(n), table: () => node(),
    control(en, kr, opts) { const input = { ...node(), value: String(opts.value), checked: Boolean(opts.value), opts }; controls.push(input); return input; }, onChange(fn) { ctx.render = fn; fn(); },
    plot(spec) { for (const s of spec.series || []) for (const [x, y] of s.points) { assert.ok(Number.isFinite(x) && Number.isFinite(y)); assert.ok(x >= spec.x[0] && x <= spec.x[1]); assert.ok(y >= spec.y[0] && y <= spec.y[1]); } return node(); }
  };
  factories['birdclef-gated-ranks'](ctx); renders++;
  for (const input of controls) {
    const original = input.value, checked = input.checked;
    const values = input.opts.type === 'select' ? input.opts.options.map(x => x.value) : input.opts.type === 'checkbox' ? [false, true] : [input.opts.min, input.opts.max];
    for (const v of values) { if (input.opts.type === 'checkbox') input.checked = v; else input.value = String(v); ctx.render(); renders++; }
    input.value = original; input.checked = checked;
  }
}
console.log(`BirdCLEF column-rank, mask, class-gate and movement-budget invariants and ${renders} bilingual control renders passed.`);
