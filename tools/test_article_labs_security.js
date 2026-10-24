'use strict';
const assert = require('node:assert/strict');
const s = require('../assets/js/article-labs-security.js');
assert.equal(s.rawScore(400, 1, 400) / 200, 36);
assert.equal(s.rawScore(1, 4, 1), 66);
assert.equal(s.rawScore(4, 1, 4), 72);
assert.equal(s.rawScore(4, 1, 99), 72);
assert.deepEqual(s.fillRows(9000, 24, 12, 357), [357, 357]);
assert.deepEqual(s.fillRows(9000, 24, 12, 0), [375, 750]);
assert(s.density(3, .06, .75) < s.density(1, .06, .75));
assert(s.density(2, 1, .75) > s.density(1, 1, .75));
assert.equal(s.stopCount(.8).count, s.stopCount(.99).count);
assert.equal(s.stopCount(.95).bound, 'wall');
assert.equal(s.stopCount(1.05).bound, 'replay');
assert.equal(s.deadline(1500, 7.5, 9000, false).credited, 0);
assert.equal(s.deadline(1500, 7.5, 8750, true).credited, 1166);
assert.equal(s.deadline(100, 1, 8750, true, true).valid, false);
const slow = s.ordered(256, 1744, 34, 7.5, 8750), fast = s.ordered(256, 1744, .25, 7.5, 8750);
assert(fast.second > slow.second);
assert(slow.time <= 8750 && fast.time <= 8750);
assert.equal(s.ordered(2, 3, 5, 2, 10).second, 0);
assert.equal(s.ordered(2, 3, 5, 2, 12).second, 1);
assert.equal(Math.max(s.pairs[9][2], s.pairs[8][2]), 26.010);
assert(Math.abs(s.pooledRate([[100, 10], [100, 100]]) - 200 / 110) < 1e-12);
// Native range inputs snap off-grid defaults; preserve the reported 11.942 s.
const factories = {}, fs = require('node:fs'), vm = require('node:vm');
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-security.js'), 'utf8'), { window: { ArticleLabs: { register: (id, factory) => { factories[id] = factory; } } } });
const output = { textContent: '' }, empty = () => ({ append() {}, replaceChildren() {} });
factories['security-accounting']({
  control(_en, _ko, options) {
    const steps = (options.value - options.min) / options.step;
    assert(Math.abs(steps - Math.round(steps)) < 1e-7, 'Reported default must fit the native range grid');
    return { value: String(options.min + Math.round(steps) * options.step) };
  },
  metric: () => output, onChange: render => render(), view: empty(), status: {},
  table: empty, text: en => en, format: (value, digits) => value.toFixed(digits)
});
assert.equal(output.textContent, '+6.263');
console.log('Security article labs: scoring, bounds, prefix, and identity invariants pass.');
