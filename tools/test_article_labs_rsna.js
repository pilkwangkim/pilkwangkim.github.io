'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const lab = require('../assets/js/article-labs-rsna.js');
const { parseCSV } = require('../assets/js/article-labs.js');
const near = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
near(lab.auc([1, 0], [.8, .2]), 1);
near(lab.auc([1, 0], [.2, .8]), 0);
near(lab.auc([1, 0], [.5, .5]), .5);
const labels = [1, 1, 0, 0], scores = [.9, .4, .8, .1];
near(lab.auc(labels, scores), lab.auc(labels, scores.map(x => x ** 3)));
assert.deepEqual(lab.ranks([.2, .2, .9]), [.5, .5, 1]);
for (const q of [0, .02, .6, 1]) for (const p1 of [.01, .2, .8, .99]) for (const p2 of [.01, .3, .9, .99]) {
  const r = lab.windowLoss(q, p1, p2); assert.ok(r.combined <= r.separate + 1e-12);
  const h = 1e-6, z = lab.logit(p1), numeric = (lab.windowLoss(q, lab.sigmoid(z + h), p2).combined - lab.windowLoss(q, lab.sigmoid(z - h), p2).combined) / (2 * h);
  near(r.combinedGrad[0], numeric, 1e-7);
}
assert.equal(lab.teacherStatus(0, 1, [0, 1]), 'double-exclusion');
assert.equal(lab.teacherStatus(0, 1, [1]), 'evaluation-path');
assert.equal(lab.teacherStatus(0, 1, [0]), 'evaluation-excluded');
near(lab.geometry(130, 448, 16).pitch, 130 / 448);
near(lab.geometry(130, 448, 16).bytes / lab.geometry(130, 224, 16).bytes, 4);
const matrix = [[.99, .1, .3, .4], [.4, .8, .5, .6]];
assert.notDeepEqual(lab.reduce(matrix, .5, 'prob'), lab.reduce(matrix, .5, 'logit'));
assert.equal(lab.budgetRecords.length, 6);
// A lightweight factory harness checks every control endpoint and both languages.
const factories = {};
global.window = { ArticleLabs: { register: (id, factory) => { factories[id] = factory; } } };
delete require.cache[require.resolve('../assets/js/article-labs-rsna.js')];
require('../assets/js/article-labs-rsna.js');
delete global.window;
function node() { return { append() {}, replaceChildren() {}, addEventListener() {}, textContent: '' }; }
function context(ko, preset) {
  const controls = [], ctx = { root: { dataset: { preset } }, ko, text: (en, kr) => ko ? kr : en, el: node, svg: node, view: node(), status: node(), metric: node, format: (v, n = 4) => Number(v).toFixed(n), table: () => node(),
    control(en, kr, opts) { const input = { ...node(), value: String(opts.value), checked: Boolean(opts.value), opts }; controls.push(input); return input; },
    onChange(fn) { ctx.render = fn; fn(); },
    plot(spec) { for (const s of spec.series || []) for (const [x, y] of s.points) { assert.ok(Number.isFinite(x) && Number.isFinite(y)); assert.ok(x >= spec.x[0] - 1e-8 && x <= spec.x[1] + 1e-8, `x outside ${spec.title}: ${x}`); assert.ok(y >= spec.y[0] - 1e-8 && y <= spec.y[1] + 1e-8, `y outside ${spec.title}: ${y}`); } return node(); }
  }; return { ctx, controls };
}
const rows = parseCSV(fs.readFileSync(path.join(__dirname, '../assets/img/posts/rsna-knee-series/part5-submission-history.csv'), 'utf8'));
assert.equal(rows.length, 154); assert.equal(lab.historyRows(rows, false, false).length, 153);
assert.equal(lab.historyRows(rows, false, true).length, 154);
assert.ok(lab.historyRows(rows, true, false).length < 154);
(async () => {
  let renders = 0;
  for (const [id, factory] of Object.entries(factories)) for (const ko of [false, true]) {
    const { ctx, controls } = context(ko, id === 'rsna-window-loss' ? 'windows' : '');
    ctx.fetchCSV = async () => rows;
    await factory(ctx); renders++;
    for (const input of controls) {
      const original = input.value, checked = input.checked;
      const values = input.opts.type === 'select' ? input.opts.options.map(x => x.value) : input.opts.type === 'checkbox' ? [false, true] : [input.opts.min, input.opts.max];
      for (const v of values) { if (input.opts.type === 'checkbox') input.checked = v; else input.value = String(v); ctx.render(); renders++; }
      input.value = original; input.checked = checked;
    }
  }
  console.log(`RSNA numerical invariants and ${renders} bilingual factory/control renders passed (9 factories; 154 history rows).`);
})().catch(error => { console.error(error); process.exitCode = 1; });
