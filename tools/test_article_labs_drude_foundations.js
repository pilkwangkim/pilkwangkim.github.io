'use strict';
/* Independent physical oracles and bilingual UI/state contracts.
 * A DOM adapter is not browser geometry or actual screen-reader evidence. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const m = require('../assets/js/article-labs-drude-foundations.js');
const electronCharge = -1.602176634e-19;
let physics = 0, lifecycle = 0, plotted = 0;
function check(v, message) { assert.ok(v, message); physics++; }
function close(a, b, tolerance = 1e-10) { check(Number.isFinite(a) && Math.abs(a - b) <= tolerance, `${a} != ${b} (tol ${tolerance})`); }
function rejected(fn) { assert.throws(fn); physics++; }

// Charge is the signed difference of boundary charge fluxes, with no source.
for (const left of [-3, -1.25, 0, 1, 2, 3]) for (const right of [-3, -.5, 0, 1, 2, 3]) for (let i = 0; i <= 40; i++) {
  const state = Object.freeze({ left, right, time: i * .05 }), p = m.continuityModel(state);
  const enteringCharge = left * 1e-6 * state.time * 1e-9, leavingCharge = right * 1e-6 * state.time * 1e-9;
  close(p.chargeC, enteringCharge - leavingCharge, 1e-28);
  close(p.chargeFC * 1e-15, p.chargeC, 1e-28);
  close(p.electronCountChange * electronCharge, p.chargeC, 1e-28);
  close(p.electronRateLeft * electronCharge, left * 1e-6, 1e-20);
  close(p.electronRateRight * electronCharge, right * 1e-6, 1e-20);
  if (i < 40) {
    const next = m.continuityModel({ left, right, time: (i + 1) * .05 });
    close((next.chargeC - p.chargeC) / (.05e-9), (left - right) * 1e-6, 1e-18);
  }
  if (left === right) close(p.chargeC, 0);
}
const charged = m.continuityModel();
close(charged.chargeC, 1e-15, 1e-28); close(charged.electronCountChange, -6241.509074460763, 1e-9);
check(charged.electronCountChange < 0 && charged.rateMicroamp > 0);

// An independent explicit complex matrix multiplication checks all four modes.
const matrix = [[2, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 2, -1], [-1, 0, -1, 2]];
const exact = [
  [[.5, 0], [.5, 0], [.5, 0], [.5, 0]],
  [[.5, 0], [0, .5], [-.5, 0], [0, -.5]],
  [[.5, 0], [-.5, 0], [.5, 0], [-.5, 0]],
  [[.5, 0], [0, -.5], [-.5, 0], [0, .5]]
];
for (let mode = 0; mode < 4; mode++) for (let it = 1; it <= 30; it++) {
  const hopping = it / 10, p = m.blochModel(Object.freeze({ mode, hopping }));
  close(p.energyEV, [0, 2, 4, 2][mode] * hopping, 5e-15); close(p.norm, 1);
  for (let j = 0; j < 4; j++) {
    close(p.coefficients[j].re, exact[mode][j][0]); close(p.coefficients[j].im, exact[mode][j][1]); close(p.probabilities[j], .25);
    const re = matrix[j].reduce((sum, h, q) => sum + h * hopping * exact[mode][q][0], 0);
    const im = matrix[j].reduce((sum, h, q) => sum + h * hopping * exact[mode][q][1], 0);
    close(re, p.energyEV * p.coefficients[j].re, 5e-15); close(im, p.energyEV * p.coefficients[j].im, 5e-15);
  }
  close(Math.cos(4 * p.ka), 1); close(Math.sin(4 * p.ka), 0); check(p.residualEV < 5e-15);
}
for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
  const ca = m.blochModel({ mode: a, hopping: 1 }).coefficients, cb = m.blochModel({ mode: b, hopping: 1 }).coefficients;
  close(ca.reduce((sum, c, j) => sum + c.re * cb[j].re + c.im * cb[j].im, 0), a === b ? 1 : 0);
  close(ca.reduce((sum, c, j) => sum + c.re * cb[j].im - c.im * cb[j].re, 0), 0);
}

// Occupation is bounded per state, not a group-choice probability. Equal
// state-count weights make opposite changes conserve number exactly.
for (let id = 0; id <= 100; id++) for (let iv = 1; iv <= 20; iv++) {
  const delta = -.5 + id / 100, speed = iv / 10, p = m.occupationModel(Object.freeze({ delta, speed }));
  check(p.fPlus >= 0 && p.fPlus <= 1 && p.fMinus >= 0 && p.fMinus <= 1);
  close(p.fPlus + p.fMinus, 1); close(p.density, 1e23, 1e8); close(p.densityOverG, 1);
  const g = 1e23, v = speed * 1e5, nPlus = g * (.5 - delta), nMinus = g * (.5 + delta);
  const expectedJ = electronCharge * (v * nPlus - v * nMinus);
  close(p.currentDensity, expectedJ, 2e-6);
  close(p.currentNormalized, 2 * speed * delta, 1e-14);
  close(p.meanVelocity, (v * nPlus - v * nMinus) / (nPlus + nMinus), 1e-10);
  close(p.currentDensity, electronCharge * p.density * p.meanVelocity, 2e-6);
  if (delta !== 0) check(p.currentDensity * delta > 0 && p.meanVelocity * delta < 0);
}
for (const speed of [.1, 1, 2]) {
  close(m.occupationModel({ delta: 0, speed }).currentDensity, 0);
  close(m.occupationModel({ delta: .5, speed }).fPlus, 0); close(m.occupationModel({ delta: -.5, speed }).fMinus, 0);
}
const specs = [
  ['drude-continuity', m.validContinuityState, m.continuityModel, { left: 2, right: 1, time: 1 }, [{ left: 3.25 }, { right: -3.25 }, { left: .1 }, { time: .025 }, { time: 2.05 }]],
  ['drude-bloch', m.validBlochState, m.blochModel, { mode: 1, hopping: 1 }, [{ mode: -1 }, { mode: 4 }, { mode: .5 }, { mode: Math.PI / 2 }, { hopping: .15 }, { hopping: 0 }, { hopping: 3.1 }]],
  ['drude-occupation', m.validOccupationState, m.occupationModel, { delta: .1, speed: 1 }, [{ delta: -.51 }, { delta: .51 }, { delta: .005 }, { speed: 0 }, { speed: .15 }, { speed: 2.1 }]]
];
for (const [, valid, model, state, patches] of specs) {
  check(valid(state));
  const invalid = [null, false, [], {}, { ...state, extra: 0 }, ...patches.map(p => ({ ...state, ...p }))];
  for (const key of Object.keys(state)) {
    const missing = { ...state }; delete missing[key]; invalid.push(missing);
    for (const bad of [NaN, Infinity, -Infinity, null, false, true, '1', {}, []]) invalid.push({ ...state, [key]: bad });
  }
  for (const bad of invalid) { check(valid(bad) === false); rejected(() => model(bad)); }
}

// Bilingual factories exercise the same state and plot outputs as the browser.
class Element {
  constructor(tag = 'div', attrs = {}, children = []) {
    this.tagName = tag.toUpperCase(); this.attrs = attrs || {}; this.children = []; this.listeners = {}; this._value = ''; this._text = ''; this.checked = false;
    this.classes = []; this.classList = { add: (...names) => this.classes.push(...names) }; this.append(children);
  }
  set value(v) { this._value = String(v); }
  get value() { return this._value; }
  append(...items) { this.children.push(...items.flat(Infinity).filter(x => x !== null && x !== undefined)); }
  replaceChildren(...items) { this.children = []; this._text = ''; this.append(items); }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c instanceof Element ? c.textContent : String(c)).join(' '); }
}
const factories = new Map();
const source = fs.readFileSync(require.resolve('../assets/js/article-labs-drude-foundations.js'), 'utf8');
vm.runInNewContext(source, { window: { ArticleLabs: { register: (id, fn) => { assert.ok(!factories.has(id)); factories.set(id, fn); } } } });
assert.deepEqual([...factories.keys()], specs.map(s => s[0])); lifecycle++;
const registry = fs.readFileSync(require.resolve('../_data/article_labs_drude_foundations.yml'), 'utf8');
assert.deepEqual(registry.match(/^drude-[a-z]+(?=:)/gm), [...factories.keys()]); lifecycle++;
assert.equal((registry.match(/bundle: drude-foundations/g) || []).length, 3); lifecycle++;
assert.ok(!registry.includes('stylesheet:') && fs.readFileSync(require.resolve('../_data/article_labs_drude.yml'), 'utf8').includes('stylesheet: /assets/css/article-labs-drude.css')); lifecycle++;
assert.ok(!registry.includes('중성') && !source.includes('중성')); lifecycle++;
function mount(id, ko) {
  const fields = [], defaults = [], metrics = [], buttons = []; let state, render, reset;
  const ctx = { root: new Element(), ui: new Element(), view: new Element(), metrics: new Element(), status: new Element(), ko,
    text: (en, korean) => ko ? korean : en, format: (v, digits = 4) => String(Number(v.toFixed(digits))),
    el: (tag, attrs, children) => new Element(tag, attrs, children), svg: (tag, attrs, children) => new Element(tag, attrs, children),
    metric: (en, korean) => { const output = new Element('output', { label: ko ? korean : en }); metrics.push(output); return output; },
    control: (en, korean, config) => { const field = new Element(config.type === 'select' ? 'select' : 'input', { label: ko ? korean : en }); field.config = config; field.value = config.value; fields.push(field); defaults.push(field.value); return field; },
    button: (en, korean, fn) => { buttons.push({ text: ko ? korean : en, click: fn }); return new Element('button'); },
    plot: spec => {
      assert.ok(spec.x[0] < spec.x[1] && spec.y[0] < spec.y[1]);
      const inspect = (x, y) => { assert.ok(Number.isFinite(x) && Number.isFinite(y)); assert.ok(x >= spec.x[0] - 1e-10 && x <= spec.x[1] + 1e-10); assert.ok(y >= spec.y[0] - 1e-10 && y <= spec.y[1] + 1e-10); plotted++; };
      (spec.series || []).forEach(s => s.points.forEach(([x, y]) => inspect(x, y)));
      (spec.markers || []).forEach(p => inspect(p.x, p.y));
      (spec.bars || []).forEach(b => { inspect(b.x - b.width / 2, b.y); inspect(b.x + b.width / 2, b.y); });
      assert.ok((spec.xTicks || []).every(x => x >= spec.x[0] && x <= spec.x[1])); assert.ok((spec.yTicks || []).every(y => y >= spec.y[0] && y <= spec.y[1]));
      return new Element('div', { spec });
    },
    onChange: fn => { render = fn; fn(); }, onReset: fn => { reset = fn; }, stateLink: (key, provider) => { assert.equal(key, id); assert.equal(state, undefined); state = provider; }
  };
  factories.get(id)(ctx);
  assert.deepEqual(ctx.root.classes, ['drude-lab', 'al-workshop']); lifecycle++;
  return { ctx, fields, metrics, buttons, state, render,
    reset: () => { fields.forEach((f, i) => { f.value = defaults[i]; }); reset?.(); render(); },
    snapshot: () => JSON.stringify({ fields: fields.map(f => f.value), state: state.read(), view: ctx.view, metrics: metrics.map(o => o.textContent), status: ctx.status.textContent }) };
}
function roundTrip(lab, id, ko) {
  const other = mount(id, ko); assert.equal(other.state.write(JSON.parse(JSON.stringify(lab.state.read()))), true); assert.equal(other.snapshot(), lab.snapshot()); lifecycle += 2;
}
for (const [id, , , base, patches] of specs) for (const ko of [false, true]) {
  const lab = mount(id, ko), initial = lab.snapshot();
  const invalid = [null, false, [], {}, { ...base, extra: 0 }, ...patches.map(p => ({ ...base, ...p }))];
  for (const key of Object.keys(base)) {
    const missing = { ...base }; delete missing[key]; invalid.push(missing);
    for (const value of [undefined, null, NaN, Infinity, -Infinity, '1', false, {}, []]) invalid.push({ ...base, [key]: value });
  }
  for (const state of invalid) { assert.equal(lab.state.write(state), false); assert.equal(lab.snapshot(), initial, 'invalid state partially mutated UI'); lifecycle += 2; }
  roundTrip(lab, id, ko);
  for (const field of lab.fields) {
    const config = field.config, values = config.type === 'select' ? config.options.map(o => o.value) : [config.min, config.min + config.step, config.max];
    for (const value of values) { field.value = value; lab.render(); assert.ok(lab.ctx.view.children.length); roundTrip(lab, id, ko); lifecycle++; }
    lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++;
  }
  for (const button of lab.buttons) { assert.ok(button.text.length > 0); button.click(); roundTrip(lab, id, ko); lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle += 2; }
  if (id === 'drude-continuity') for (const left of [-3, 0, 3]) for (const right of [-3, 0, 3]) for (const time of [0, 2]) {
    assert.equal(lab.state.write({ left, right, time }), true);
    close(Number(lab.metrics[1].textContent), (left - right) * time); close(Number(lab.metrics[0].textContent), left - right); roundTrip(lab, id, ko);
  }
  if (id === 'drude-bloch') for (let mode = 0; mode < 4; mode++) for (const hopping of [.1, 3]) {
    assert.equal(lab.state.write({ mode, hopping }), true);
    const coefficientBars = lab.ctx.view.children[0].attrs.spec.bars, probabilityBars = lab.ctx.view.children[1].attrs.spec.bars;
    for (let j = 0; j < 4; j++) { close(coefficientBars[j * 2].y, exact[mode][j][0]); close(coefficientBars[j * 2 + 1].y, exact[mode][j][1]); close(probabilityBars[j].y, .25); }
    roundTrip(lab, id, ko);
  }
  if (id === 'drude-occupation') for (const delta of [-.5, -.1, 0, .1, .5]) for (const speed of [.1, 2]) {
    assert.equal(lab.state.write({ delta, speed }), true); const bars = lab.ctx.view.children[0].attrs.spec.bars, chart = lab.ctx.view.children[1].attrs.spec;
    close(bars[0].y, .5 + delta); close(bars[1].y, .5 - delta); close(chart.markers[0].y, 2 * delta * speed); close(Number(lab.metrics[0].textContent), 1); roundTrip(lab, id, ko);
  }
  // Rejection is also transactional after changing multiple fields away from
  // defaults: an invalid last property cannot partially restore earlier ones.
  const changed = JSON.parse(JSON.stringify(lab.state.read())), before = lab.snapshot();
  for (const bad of invalid) { assert.equal(lab.state.write(bad), false); assert.equal(lab.snapshot(), before); lifecycle += 2; }
  assert.deepEqual(JSON.parse(JSON.stringify(lab.state.read())), changed); lifecycle++;
  lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++;
}
console.log(`Drude foundations: ${physics} independent physics/model assertions; ${lifecycle} bilingual UI/state/reset assertions; ${plotted} finite in-domain plot coordinates. Browser geometry is not tested by this adapter.`);
