'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const g = require('../assets/js/article-labs-galois.js');
let checks = 0;
const check = value => { assert(value); checks++; };
for (let n = 1; n <= 5; n++) {
  const all = g.permutations(n), e = g.identity(n);
  for (const p of all) { assert.deepEqual(g.compose(p, g.inverse(p)), e); checks++; assert.deepEqual(g.compose(g.inverse(p), p), e); checks++; }
}
for (const [name, orders] of Object.entries({ C5: [5, 1], S3: [6, 3, 1], S4: [24, 12, 4, 1], A5: [60, 60], S5: [120, 60, 60] })) { assert.deepEqual(g.namedGroup(name).map(h => h.length), orders); checks++; }
for (let a = -9; a <= 9; a++) for (let b = -9; b <= 9; b++) {
  const u = g.pair(a, b), inverse = g.inversePair(u);
  assert.deepEqual(g.conjugatePair(g.conjugatePair(u)), u); checks++;
  if (a || b) { const x = inverse[0][0] / inverse[0][1], y = inverse[1][0] / inverse[1][1]; check(Math.abs(a * x + 2 * b * y - 1) < 1e-10); check(Math.abs(a * y + b * x) < 1e-10); } else check(inverse === null);
  for (const v of [[0, 0], [1, -1], [2, 3]]) { assert.deepEqual(g.conjugatePair(g.mulPair(u, v)), g.mulPair(g.conjugatePair(u), g.conjugatePair(v))); checks++; }
}
const rootPermutations = g.permutations(4), allowed = rootPermutations.filter(p => g.rootPermutationChecks(p).valid);
assert.equal(allowed.length, 4); checks++;
assert(!g.rootPermutationChecks([2, 3, 0, 1]).valid); checks++;
assert(g.rootPermutationChecks([1, 0, 3, 2]).valid); checks++;
const subgroups = new Set();
for (let mask = 0; mask < 8; mask++) {
  const h = g.v4Closure([1, 2, 3].filter((_, i) => mask & 1 << i)); subgroups.add(h.join(','));
  check(h.length * g.fixedBasis(h).length === 4);
  for (const a of h) for (const b of h) check(h.includes(a ^ b));
}
check(subgroups.size === 5);
assert.deepEqual(g.fixedBasis([0, 1]), [0, 2]); checks++;
assert.deepEqual(g.fixedBasis([0, 2]), [0, 1]); checks++;
assert.deepEqual(g.fixedBasis([0, 3]), [0, 3]); checks++;
for (const [name, model] of Object.entries(g.QUINTICS)) for (let i = 0; i < model.brackets.length; i++) {
  let previous = Infinity;
  for (let steps = 0; steps <= 40; steps++) {
    const result = g.bisection(name, i, steps);
    check(result.width <= previous); check(model.f(result.lo) * model.f(result.hi) <= 0); previous = result.width;
    const endpoints = g.formatBisectionBracket(result.lo, result.hi).slice(1, -1).split(', ');
    check(Number(endpoints[0]) === result.lo); check(Number(endpoints[1]) === result.hi);
    check(result.lo === result.hi || endpoints[0] !== endpoints[1]);
  }
  const result = g.bisection(name, i, 40); check(Math.abs(result.residual) < 1e-9); check(result.width < 2e-12);
}
for (const bad of [['unknown', 0, 1], ['obstruction', -1, 0], ['radical', 1, 1], ['radical', 0, 41]]) { assert.throws(() => g.bisection(...bad)); checks++; }
assert.equal(g.QUIZ.length, 5); checks++;
for (const q of g.QUIZ) { check(q.options.length === 3 && q.optionReasons.length === 3); check(q.diagram.length >= 3); }
// Focused factory-state tests: invalid payloads must not partially mutate fields.
// Browser geometry and native validation remain separate real-browser checks.
class Element {
  constructor(tag = 'div', attrs = {}, children = []) { this.tagName = tag.toUpperCase(); Object.assign(this, attrs); this.children = [children].flat(Infinity).filter(x => x !== null); this.classList = { add() {} }; this.checked = false; this.value = ''; this.listeners = {}; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
  click() { for (const callback of this.listeners.click || []) callback(); }
  get options() { return this.children.filter(x => x && x.tagName === 'OPTION'); }
}
const factories = new Map();
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-galois.js'), 'utf8'), { window: { ArticleLabs: { register: (id, factory) => factories.set(id, factory) } } });
let stateChecks = 0;
const visibleText = node => typeof node === 'string' ? node : [node.textContent || '', ...(node.children || []).map(visibleText)].join(' ');
function mount(id, factory, ko) {
  const fields = [], defaults = [], buttons = [], metrics = []; let state, render, reset = () => {};
  const el = (tag, attrs, children) => new Element(tag, attrs, children);
  const ctx = { root: new Element(), ui: new Element(), view: new Element(), status: new Element(), metrics: new Element(), ko,
    el, svg: el, text: (en, korean) => ko ? korean : en, metric: () => { const node = new Element('output'); metrics.push(node); return node; },
    plot: () => new Element(), button: (en, korean, callback) => { buttons.push(callback); return new Element('button'); },
    onChange: callback => { render = callback; render(); }, onReset: callback => { reset = callback; },
    stateLink: (name, provider) => { assert.equal(name, id); state = provider; },
    control: (_en, _ko, options) => {
      const field = new Element(options.type === 'select' ? 'select' : 'input', { type: options.type, min: options.min, max: options.max });
      field.value = String(options.value); field.checked = Boolean(options.value);
      if (options.options) field.children = options.options.map(o => Object.assign(new Element('option'), { value: o.value }));
      fields.push(field); defaults.push({ value: field.value, checked: field.checked }); return field;
    }
  };
  factory(ctx);
  return { fields, defaults, buttons, state, ctx, render: () => render(), reset: () => reset(),
    snapshot: () => JSON.stringify({ model: state.read(), view: visibleText(ctx.view), metrics: metrics.map(node => node.textContent), disabled: fields.map(node => Boolean(node.disabled)) }) };
}
for (const [id, factory] of factories) for (const ko of [false, true]) {
  const { fields, defaults, buttons, state, ctx, render, reset } = mount(id, factory, ko);
  const saved = JSON.parse(JSON.stringify(state.read())), baseline = JSON.stringify(saved);
  const reject = payload => { assert.equal(state.write(payload), false, id + ': invalid state accepted'); assert.equal(JSON.stringify(state.read()), baseline, id + ': invalid state partially changed fields'); stateChecks++; };
  reject({}); reject({ fields: [] }); reject({ ...saved, fields: [...saved.fields, 'extra'] });
  for (let i = 0; i < fields.length; i++) {
    const invalid = JSON.parse(baseline); invalid.fields[i] = fields[i].type === 'checkbox' ? 'true' : '__invalid__'; reject(invalid);
  }
  if (id === 'galois-quadratic') {
    for (const bad of ['', ' ', '0x1', '0b1', '0o1', '+1', '1.', '1e', 'NaN', 'Infinity', '-10', '10', '0.5']) {
      const invalid = JSON.parse(baseline); invalid.fields[1] = bad; reject(invalid);
    }
    for (const valid of ['0', '-0', '01', '1.0', '.0', '1e0', '1E+0', '-9', '9']) {
      const payload = JSON.parse(baseline); payload.fields[1] = valid;
      assert.equal(state.write(payload), true); stateChecks++;
      assert.equal(state.write(saved), true); stateChecks++;
    }
    const zero = JSON.parse(baseline); zero.fields[0] = '0'; zero.fields[1] = '0';
    assert.equal(state.write(zero), true); stateChecks++;
    assert(visibleText(ctx.view).includes(ko ? '정의되지 않음 (u = 0)' : 'not defined (u = 0)')); stateChecks++;
    assert(!visibleText(ctx.view).includes('undefined')); stateChecks++;
    assert.equal(state.write(saved), true); stateChecks++;
  }
  if (id === 'galois-derived') for (const bad of [-1, 99, .5, '1', null]) reject({ ...saved, step: bad });
  if (id === 'galois-quintic') {
    for (const bad of [-1, 41, .5, '1', null]) reject({ ...saved, steps: bad });
    reject({ ...saved, fields: ['radical', '1'] });
    for (let i = 0; i < 100; i++) buttons[1]();
    assert.equal(state.read().steps, 40); stateChecks++;
    const bracket = g.bisection('obstruction', 0, 40);
    assert(visibleText(ctx.view).includes(g.formatBisectionBracket(bracket.lo, bracket.hi))); stateChecks++;
    assert.equal(state.write(saved), true);
  }
  if (id === 'galois-quiz') {
    reject({ ...saved, answers: [3, null, null, null, null] });
    reject({ ...saved, hints: [true, true] });
  }
  assert.equal(state.write(saved), true); assert.equal(JSON.stringify(state.read()), baseline); stateChecks++;
  fields.forEach((field, i) => Object.assign(field, defaults[i])); reset(); render();
  assert.equal(JSON.stringify(state.read()), baseline, id + ': Reset did not restore all defaults'); stateChecks++;
}
// The real checkpoint bridge invokes onChange after provider.write. Exercise
// progressed models on a fresh factory, then repeat that lifecycle render.
let lifecycleChecks = 0, progressedMounts = 0;
function descendants(node) { return (node.children || []).flatMap(child => child instanceof Element ? [child, ...descendants(child)] : []); }
for (const [id, factory] of factories) for (const ko of [false, true]) {
  const active = mount(id, factory, ko), { fields, buttons, render, ctx } = active;
  if (id === 'galois-quadratic') { ['-3', '2', '0', '-4'].forEach((value, i) => { fields[i].value = value; }); render(); }
  if (id === 'galois-permutations') buttons[2]();
  if (id === 'galois-correspondence') { fields.forEach((field, i) => { field.checked = i === 2; }); render(); }
  if (id === 'galois-derived') { fields[0].value = 'S4'; render(); buttons[0](); buttons[0](); }
  if (id === 'galois-quintic') { fields[1].value = '2'; render(); buttons[1](); buttons[0](); }
  if (id === 'galois-quiz') {
    fields[0].value = '1'; render(); descendants(ctx.view).filter(node => node.tagName === 'BUTTON')[0].click(); buttons[2]();
    buttons[1](); descendants(ctx.view).filter(node => node.tagName === 'BUTTON')[2].click();
  }
  const cases = [JSON.parse(JSON.stringify(active.state.read()))];
  if (id === 'galois-derived') { fields[0].value = 'S5'; render(); buttons[1](); cases.push(JSON.parse(JSON.stringify(active.state.read()))); }
  if (id === 'galois-quintic') { fields[0].value = 'radical'; render(); buttons[1](); buttons[1](); cases.push(JSON.parse(JSON.stringify(active.state.read()))); }
  for (const saved of cases) {
    assert.equal(active.state.write(saved), true); const expected = active.snapshot();
    const restored = mount(id, factory, ko); assert.equal(restored.state.write(saved), true); lifecycleChecks++;
    for (let repeat = 0; repeat < 3; repeat++) {
      restored.render(); assert.equal(restored.snapshot(), expected, id + ': checkpoint was lost on lifecycle render'); lifecycleChecks++;
    }
    const reject = payload => {
      assert.equal(restored.state.write(payload), false, id + ': invalid progressed state accepted');
      assert.equal(restored.snapshot(), expected, id + ': rejected state changed model or displayed results'); lifecycleChecks++;
    };
    reject({}); reject({ ...saved, fields: [...saved.fields, 'extra'] });
    for (let i = 0; i < fields.length; i++) for (const invalid of [null, {}, [], '__proto__', fields[i].type === 'checkbox' ? 'false' : false, 999]) {
      const bad = JSON.parse(JSON.stringify(saved)); bad.fields[i] = invalid; reject(bad);
    }
    if (id === 'galois-derived') for (const bad of [-1, 99, .5, '1', null]) reject({ ...saved, step: bad });
    if (id === 'galois-quintic') for (const bad of [-1, 41, .5, '1', null]) reject({ ...saved, steps: bad });
    if (id === 'galois-quiz') { reject({ ...saved, answers: [3, null, null, null, null] }); reject({ ...saved, hints: [true, 'true', false, false, false] }); }
    restored.fields.forEach((field, i) => Object.assign(field, restored.defaults[i])); restored.reset(); restored.render();
    assert.equal(restored.snapshot(), mount(id, factory, ko).snapshot(), id + ': progressed Reset differs from a fresh factory'); lifecycleChecks++; progressedMounts++;
  }
}
console.log(JSON.stringify({ status: 'passed', checks, stateChecks, lifecycleChecks, progressedMounts, rootPermutations: 24, validAutomorphisms: 4, intermediateFields: subgroups.size, derivedOrders: Object.fromEntries(['C5', 'S3', 'S4', 'A5', 'S5'].map(name => [name, g.namedGroup(name).map(h => h.length)])), quizzes: 5 }, null, 2));
