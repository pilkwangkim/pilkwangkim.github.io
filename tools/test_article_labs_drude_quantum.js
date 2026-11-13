'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const m = require('../assets/js/article-labs-drude-quantum.js');
let physics = 0, lifecycle = 0, plotted = 0;
function close(a, b, tolerance = 1e-10) { assert.ok(Number.isFinite(a) && Math.abs(a - b) <= tolerance, `${a} != ${b} (tol=${tolerance})`); physics++; }
function check(condition, message) { assert.ok(condition, message); physics++; }
function throws(fn) { assert.throws(fn); physics++; }
function integrate(fn, lo, hi, n = 12000) {
  assert.equal(n % 2, 0);
  const h = (hi - lo) / n; let sum = fn(lo) + fn(hi);
  for (let i = 1; i < n; i++) sum += (i % 2 ? 4 : 2) * fn(lo + i * h);
  return sum * h / 3;
}

// Packet density: normalization, moments and force-independent spreading.
// Physical plotting references are constants, independently converted from SI.
const initialWidthNm = 10, massKg = 9.1093837139e-31;
const packetTimePs = 2 * massKg * (initialWidthNm * 1e-9) ** 2 / (6.62607015e-34 / (2 * Math.PI)) * 1e12;
const packetForceEVNm = 2 * massKg * initialWidthNm * 1e-9 / (packetTimePs * 1e-12) ** 2 / 1.602176634e-19 * 1e-9;
close(m.PACKET_WIDTH_NM, initialWidthNm); close(m.PACKET_MASS_KG, massKg, 1e-40);
close(m.PACKET_TIME_PS, packetTimePs, 1e-12); close(m.PACKET_FORCE_EV_NM, packetForceEVNm, 1e-15);
check(9 + 5 * Math.sqrt(10) < 25, 'Fixed ±250 nm window must cover five standard deviations at every control setting.');
check(1 / (Math.sqrt(2 * Math.PI) * initialWidthNm) < .042, 'Initial peak must fit fixed density axis.');
check(3 * packetTimePs < 6, 'Every selectable time must fit fixed ps axis.');
for (const alpha of [-1, -.3, 0, 1]) for (const s of [0, .7, 3]) {
  const p = m.packetState(alpha, s), density = x => m.packetDensity(x, alpha, s);
  const norm = integrate(density, -60, 60), mean = integrate(x => x * density(x), -60, 60), second = integrate(x => x * x * density(x), -60, 60);
  close(norm, 1); close(mean, alpha * s * s); close(second - mean * mean, 1 + s * s, 2e-10);
  close(p.center, mean); close(p.width * p.width, second - mean * mean, 2e-10); close(p.velocity, 2 * alpha * s); close(p.acceleration, 2 * alpha);
  close(p.peak, density(p.center)); close(p.allSpaceProbability, 1); check(p.width >= 1);
  for (const distance of [0, .2, 1, 5]) close(density(p.center - distance), density(p.center + distance));
  if (s > 0 && s < 3) {
    const h = 1e-4, left = m.packetState(alpha, s - h), right = m.packetState(alpha, s + h);
    close((right.center - left.center) / (2 * h), p.velocity, 2e-10);
    close((right.center - 2 * p.center + left.center) / h ** 2, p.acceleration, 5e-8);
    close((right.width - left.width) / (2 * h), s / Math.sqrt(1 + s * s), 2e-9);
  }
}
// Independent complex solution in units ℏ=m*=Δx0=1 (therefore t0=2).
// A boost of the free Gaussian verifies both the density and Schrödinger law;
// the test does not infer the Hamiltonian merely from an Ehrenfest trajectory.
function exactPsi(x, t, force) {
  const b = t / 2, y = x - force * t * t / 2;
  const amplitude = (2 * Math.PI) ** (-.25) / (1 + b * b) ** .25 * Math.exp(-y * y / (4 * (1 + b * b)));
  const phase = -.5 * Math.atan(b) + y * y * b / (4 * (1 + b * b)) + force * t * x - force * force * t ** 3 / 6;
  return [amplitude * Math.cos(phase), amplitude * Math.sin(phase)];
}
for (const alpha of [-1, 0, .7, 1]) for (const t of [.5, 2, 5]) for (const x of [-1, .3, 3]) {
  const force = alpha / 2, p = exactPsi(x, t, force), ht = 1e-5, hx = 2e-4;
  const plusT = exactPsi(x, t + ht, force), minusT = exactPsi(x, t - ht, force), plusX = exactPsi(x + hx, t, force), minusX = exactPsi(x - hx, t, force);
  close(p[0] ** 2 + p[1] ** 2, m.packetDensity(x, alpha, t / 2), 1e-12);
  const dt = p.map((_, j) => (plusT[j] - minusT[j]) / (2 * ht));
  const dxx = p.map((v, j) => (plusX[j] - 2 * v + minusX[j]) / hx ** 2);
  close(-dt[1] + .5 * dxx[0] + force * x * p[0], 0, 8e-8);
  close(dt[0] + .5 * dxx[1] + force * x * p[1], 0, 8e-8);
}

// Finite-time transition: independently convert ℏ from SI into eV fs.
const hbar = (6.62607015e-34 / (2 * Math.PI)) / 1.602176634e-19 * 1e15;
close(m.HBAR_EV_FS, hbar, 1e-15);
close(m.sinc(0), 1); close(m.sinc(1e-9), 1); close(m.sinc(Math.PI), 0, 1e-15);
for (const coupling of [0, .1, .5, 1]) for (const time of [0, 1, 10, 25, 50]) {
  const p = m.transitionSummary(coupling, time);
  close(p.peakProbability, (coupling * .001 * time / hbar) ** 2);
  close(p.energyAreaEV, 2 * Math.PI * (coupling * .001) ** 2 * time / hbar);
  check(p.peakProbability < .006 && p.peakProbability >= 0);
  if (time === 0) assert.equal(p.energyScaleEV, null); else close(p.energyScaleEV, hbar / time);
  for (const detuning of [0, .005, .05, .15, 1]) {
    const prob = m.transitionProbability(detuning, coupling, time);
    check(prob >= 0 && prob <= p.peakProbability + 1e-15);
    close(prob, m.transitionProbability(-detuning, coupling, time));
    if (time > 0 && detuning > 0) close(prob, 4 * (coupling * .001 / detuning) ** 2 * Math.sin(detuning * time / (2 * hbar)) ** 2, 1e-15);
  }
  if (time > 0) close(m.transitionProbability(2 * Math.PI * hbar / time, coupling, time), 0, 1e-28);
}
close(m.transitionSummary(1, 50).peakProbability / m.transitionSummary(1, 25).peakProbability, 4);
close(m.transitionSummary(1, 50).energyAreaEV / m.transitionSummary(1, 25).energyAreaEV, 2);
close(m.transitionSummary(.5, 50).peakProbability / m.transitionSummary(1, 50).peakProbability, .25);
for (const time of [5, 25, 50]) {
  // A finite ±100 eV quadrature approaches the all-energy analytic area from
  // below; it is not relabelled as total probability or renormalized to one.
  const area = integrate(e => m.transitionProbability(e, 1, time), -100, 100, 200000), full = 2 * Math.PI * 1e-6 * time / hbar;
  check(area <= full * (1 + 1e-6)); close(area / full, 1, .001);
}

// Fourier transform: integrate real space independently, with the two different
// normalization choices. No scattering rate is inferred without its prefactor.
for (const width of [.5, 1, 1.3, 2, 3]) for (const mode of ['integral', 'peak']) {
  const strength = mode === 'integral' ? 1 : width;
  close(integrate(x => m.gaussianPotential(x, width, mode), -40, 40), strength, 1e-11);
  close(m.gaussianPotential(0, width, mode), strength / (Math.sqrt(2 * Math.PI) * width));
  for (const q of [0, .2, .7, 1, 2, 4]) {
    const p = m.gaussianFourier(q, width, mode);
    const cosineIntegral = integrate(x => m.gaussianPotential(x, width, mode) * Math.cos(q * x), -40, 40);
    const sineIntegral = integrate(x => m.gaussianPotential(x, width, mode) * Math.sin(q * x), -40, 40);
    close(cosineIntegral, strength * Math.exp(-q * q * width * width / 2), 2e-12);
    close(sineIntegral, 0, 2e-12); close(p.amplitude, cosineIntegral, 2e-12);
    close(p.rateFactor, p.amplitude * p.amplitude); close(p.strength, strength);
    close(p.amplitude, m.gaussianFourier(-q, width, mode).amplitude);
    check(p.amplitude >= 0 && p.rateFactor >= 0);
  }
}
close(m.gaussianFourier(1, 1).amplitude, .6065306597126334);
close(m.gaussianFourier(1, 2).amplitude, .1353352832366127);
close(m.gaussianFourier(1, 1).rateFactor, .36787944117144233);
close(m.gaussianFourier(1, 2).rateFactor, .01831563888873418);
check(m.gaussianFourier(4, 3).rateFactor > 0, 'Tiny positive Fourier power must not be forced to zero.');
for (let i = 0; i < 25; i++) {
  const width = .5 + i / 10;
  check(m.gaussianFourier(1, width).amplitude > m.gaussianFourier(1, width + .1).amplitude);
}
for (const fn of [() => m.packetState(NaN, 1), () => m.packetState('1', 1), () => m.packetState(1.1, 1), () => m.packetState(0, -1), () => m.packetState(0, 3.1), () => m.packetDensity(Infinity, 0, 1), () => m.sinc(NaN), () => m.transitionProbability(0, 1.1, 5), () => m.transitionProbability(0, -1, 5), () => m.transitionProbability(0, 1, 51), () => m.transitionProbability(101, 1, 1), () => m.transitionSummary(1, '1'), () => m.gaussianStrength(.4, 'integral'), () => m.gaussianStrength(3.1, 'peak'), () => m.gaussianStrength(1, 'wrong'), () => m.gaussianPotential(NaN, 1), () => m.gaussianFourier(Infinity, 1), () => m.gaussianFourier(0, '1')]) throws(fn);

// Focused UI/state adapter: these assertions verify state, labels, plot data,
// presets and Reset, not browser geometry or native range event behaviour.
class Element {
  constructor(tag = 'div', attrs = {}, children = []) { this.tagName = tag.toUpperCase(); this.attrs = attrs; this.children = []; this.listeners = {}; this._value = ''; this.checked = false; this.disabled = false; this._text = ''; this.classList = { add() {} }; this.append(children); }
  set value(v) { this._value = String(v); }
  get value() { return this._value; }
  append(...values) { this.children.push(...values.flat(Infinity).filter(x => x !== null && x !== undefined)); }
  replaceChildren(...values) { this.children = []; this._text = ''; this.append(values); }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  click() { this.listeners.click?.(); }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(x => x instanceof Element ? x.textContent : String(x)).join(' '); }
}
const factories = new Map();
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-drude-quantum.js'), 'utf8'), { window: { ArticleLabs: { register: (id, fn) => { assert.ok(!factories.has(id)); factories.set(id, fn); } } } });
const ids = ['drude-packet', 'drude-transition', 'drude-gaussian'];
assert.deepEqual([...factories.keys()], ids); lifecycle++;
const registry = fs.readFileSync(require.resolve('../_data/article_labs_drude_quantum.yml'), 'utf8');
assert.deepEqual(registry.match(/^drude-[a-z]+(?=:)/gm), ids); lifecycle++;
assert.equal((registry.match(/bundle: drude-quantum/g) || []).length, 3); lifecycle++;
// The mandatory base drude bundle owns the one shared stylesheet declaration.
assert.equal((registry.match(/stylesheet:/g) || []).length, 0); lifecycle++;
function mount(id, ko = false) {
  const fields = [], defaults = [], metrics = [], buttons = [], labels = []; let state, render, reset;
  const ctx = { root: new Element(), ui: new Element(), view: new Element(), metrics: new Element(), status: new Element(), ko,
    text: (en, korean) => ko ? korean : en, el: (tag, attrs, children) => new Element(tag, attrs, children),
    metric: (en, korean) => { const output = new Element('output'); metrics.push(output); labels.push(ko ? korean : en); return output; },
    control: (en, korean, config) => { const el = new Element(config.type === 'select' ? 'select' : 'input'); el.config = config; el.value = config.value; el.checked = config.type === 'checkbox' && Boolean(config.value); fields.push(el); labels.push(ko ? korean : en); defaults.push({ value: el.value, checked: el.checked }); return el; },
    button: (en, korean, handler) => { const el = new Element('button'); el.textContent = ko ? korean : en; el.addEventListener('click', handler); buttons.push(el); return el; },
    note: (en, korean) => { ctx.modelNote = ko ? korean : en; },
    plot: spec => {
      assert.ok(spec.x[0] < spec.x[1] && spec.y[0] < spec.y[1]); assert.ok([...spec.x, ...spec.y].every(Number.isFinite));
      assert.ok(spec.xLabel && spec.yLabel && spec.title, 'Accessible plot labels missing');
      const inspect = (x, y) => { assert.ok(Number.isFinite(x) && Number.isFinite(y)); assert.ok(x >= spec.x[0] - 1e-9 && x <= spec.x[1] + 1e-9, 'x outside plot'); assert.ok(y >= spec.y[0] - 1e-9 && y <= spec.y[1] + 1e-9, `y=${y} outside ${spec.y}`); plotted++; };
      (spec.series || []).forEach(line => { assert.ok(line.label && line.points.length >= 2); line.points.forEach(([x, y]) => inspect(x, y)); }); (spec.markers || []).forEach(p => inspect(p.x, p.y));
      for (const axis of ['x', 'y']) for (const tick of spec[axis + 'Ticks'] || []) { assert.ok(Number.isFinite(tick) && tick >= spec[axis][0] && tick <= spec[axis][1]); }
      return new Element('div', { spec }, spec.title);
    },
    onChange: fn => { render = fn; fn(); }, onReset: fn => { reset = fn; }, stateLink: (key, provider) => { assert.equal(key, id); assert.equal(state, undefined); state = provider; }
  };
  factories.get(id)(ctx);
  const snapshot = () => JSON.stringify({ state: state.read(), view: ctx.view, metrics: metrics.map(v => v.textContent), status: ctx.status.textContent, note: ctx.modelNote, disabled: fields.map(f => f.disabled) }, (key, value) => key === 'listeners' ? undefined : value);
  return { fields, metrics, buttons, labels, ctx, state, render, snapshot, reset: () => { fields.forEach((field, i) => Object.assign(field, defaults[i])); reset?.(); render(); } };
}
function roundTrip(lab, id, ko) {
  const restored = mount(id, ko), state = JSON.parse(JSON.stringify(lab.state.read()));
  assert.equal(restored.state.write(state), true); assert.equal(restored.snapshot(), lab.snapshot()); restored.render(); assert.equal(restored.snapshot(), lab.snapshot()); lifecycle += 3;
}
function verifyPlottedModel(lab, id) {
  const state = lab.state.read(), [first, second, third] = lab.ctx.view.children.map(node => node.attrs.spec);
  assert.ok(first && second); lifecycle++;
  // Exact axis contracts, checked after every slider value, preset, restore
  // and corner case. No parameter is allowed to rescale or recenter a plot.
  const expectedAxes = {
    'drude-packet': [[[-250, 250], [0, .042]], [[0, 6], [-100, 100]]],
    'drude-transition': [[[-150, 150], [0, 6]], [[0, 50], [0, .5]]],
    'drude-gaussian': [[[-10, 10], [0, .85]], [[0, 4], [0, 3.2]], [[0, 4], [0, 10]]]
  };
  const actualAxes = lab.ctx.view.children.filter(node => node.attrs.spec).map(node => [node.attrs.spec.x, node.attrs.spec.y]);
  assert.deepEqual(JSON.parse(JSON.stringify(actualAxes)), expectedAxes[id], `${id}: a control changed the absolute plot axes`); lifecycle++;
  if (id === 'drude-packet') {
    close(first.markers[0].x, initialWidthNm * state.alpha * state.s ** 2);
    for (const [x, y] of first.series[1].points) close(y, Math.exp(-((x / initialWidthNm - state.alpha * state.s ** 2) ** 2) / (2 * (1 + state.s ** 2))) / (initialWidthNm * Math.sqrt(2 * Math.PI * (1 + state.s ** 2))), 1e-14);
    close(second.markers[0].x, state.s * packetTimePs);
    close(second.markers[0].y, initialWidthNm * state.alpha * state.s ** 2); close(second.markers[1].y, initialWidthNm * Math.sqrt(1 + state.s ** 2));
    check(/nm/.test(first.xLabel) && /nm⁻¹/.test(first.yLabel) && /ps/.test(second.xLabel) && /nm/.test(second.yLabel));
    for (const [t, x] of second.series[0].points) close(x, initialWidthNm * state.alpha * (t / packetTimePs) ** 2, 1e-12);
    for (const [t, width] of second.series[1].points) close(width, initialWidthNm * Math.sqrt(1 + (t / packetTimePs) ** 2), 1e-12);
    assert.equal(lab.metrics[0].textContent, (initialWidthNm * state.alpha * state.s ** 2).toFixed(2));
    assert.equal(lab.metrics[1].textContent, (initialWidthNm * Math.sqrt(1 + state.s ** 2)).toFixed(2));
    assert.equal(lab.metrics[3].textContent, '1'); lifecycle++;
  } else if (id === 'drude-transition') {
    assert.equal(first.series.length, state.reference ? 2 : 1); lifecycle++;
    for (const [e, y] of first.series[0].points) {
      const delta = e / 1000, arg = delta * state.timeFs / (2 * hbar), independentSinc = arg === 0 ? 1 : Math.sin(arg) / arg;
      close(y, 1000 * (state.couplingMeV * .001 * state.timeFs / hbar) ** 2 * independentSinc ** 2, 1e-12);
    }
    for (const [t, y] of second.series[0].points) close(y, 1000 * 2 * Math.PI * (state.couplingMeV * .001) ** 2 * t / hbar, 1e-13);
    close(first.markers[0].x, state.probeMeV); check(/meV/.test(second.yLabel));
    const scaleMetric = lab.metrics[3];
    assert.equal(scaleMetric.ariaLabel, scaleMetric.title);
    if (state.timeFs === 0) {
      assert.equal(scaleMetric.textContent, '— (t=0)');
      assert.ok(/undefined|정의되지/.test(scaleMetric.ariaLabel));
      assert.ok(scaleMetric.ariaLabel.includes('t=0'));
      assert.ok(!scaleMetric.textContent.includes('∞'));
    } else {
      assert.ok(scaleMetric.ariaLabel.includes(scaleMetric.textContent));
      assert.ok(scaleMetric.ariaLabel.includes('meV'));
      assert.ok(!scaleMetric.ariaLabel.includes('t=0'));
    }
  } else {
    assert.ok(third, 'Fourier amplitude and its square require separate physical-unit plots'); lifecycle++;
    const strength = state.mode === 'integral' ? 1 : state.width;
    for (const [x, y] of first.series[1].points) close(y, strength / (Math.sqrt(2 * Math.PI) * state.width) * Math.exp(-x * x / (2 * state.width * state.width)), 1e-14);
    for (const [q, y] of second.series[1].points) close(y, strength * Math.exp(-q * q * state.width * state.width / 2), 1e-14);
    for (let i = 0; i < second.series[1].points.length; i++) close(third.series[1].points[i][1], second.series[1].points[i][1] ** 2, 1e-14);
    check(/nm/.test(first.xLabel) && /eV/.test(first.yLabel) && /nm⁻¹/.test(second.xLabel) && /eV nm/.test(second.yLabel) && /eV² nm²/.test(third.yLabel));
    // q selects a visible point AND a full-height vertical probe on both
    // reciprocal-space plots, including when the amplitude is vanishingly small.
    for (const chart of [second, third]) {
      close(chart.markers[0].x, state.q);
      assert.deepEqual(JSON.parse(JSON.stringify(chart.series[2].points)), [[state.q, chart.y[0]], [state.q, chart.y[1]]]); lifecycle++;
    }
    close(second.markers[0].y, strength * Math.exp(-(state.q ** 2) * state.width ** 2 / 2), 1e-14);
    close(third.markers[0].y, strength ** 2 * Math.exp(-(state.q ** 2) * state.width ** 2), 1e-14);
    for (const metric of lab.metrics) { assert.match(metric.textContent, /^\d\.\d{3}e[+-]\d{2}$/); assert.equal(metric.textContent.length, 9); lifecycle += 2; }
    if (state.width === 3 && state.q === 4) assert.notEqual(lab.metrics[3].textContent, '0');
  }
}
const corners = {
  'drude-packet': [-1, -.3, 0, .7, 1].flatMap(alpha => [0, .1, 1, 3].map(s => ({ alpha, s }))),
  'drude-transition': [0, .1, 1].flatMap(couplingMeV => [0, 1, 25, 50].flatMap(timeFs => [-150, 0, 150].flatMap(probeMeV => [false, true].map(reference => ({ couplingMeV, timeFs, probeMeV, reference }))))),
  'drude-gaussian': [.5, 1, 1.7, 2, 3].flatMap(width => [0, .05, 1, 4].flatMap(q => ['integral', 'peak'].map(mode => ({ width, q, mode }))))
};
for (const id of ids) for (const ko of [false, true]) {
  const lab = mount(id, ko), initial = lab.snapshot(), saved = JSON.parse(JSON.stringify(lab.state.read()));
  if (id === 'drude-gaussian') { assert.equal(saved.width, 2, 'Start away from the width=1 nm coincidence so changing the mode changes the curves.'); lifecycle++; }
  assert.equal(lab.buttons.length, 3); assert.ok(lab.ctx.modelNote.length > 100); assert.ok(lab.labels.every(label => typeof label === 'string' && label.length)); lifecycle += 3;
  for (const bad of [null, false, [], {}, { wrong: 1 }, { ...saved, extra: 1 }]) { assert.equal(lab.state.write(bad), false); assert.equal(lab.snapshot(), initial); lifecycle += 2; }
  for (const key of Object.keys(saved)) {
    const missing = { ...saved }; delete missing[key]; assert.equal(lab.state.write(missing), false); assert.equal(lab.snapshot(), initial); lifecycle += 2;
    for (const bad of ['__bad__', null, NaN, Infinity, -Infinity, {}, [], undefined]) { assert.equal(lab.state.write({ ...saved, [key]: bad }), false); assert.equal(lab.snapshot(), initial); lifecycle += 2; }
  }
  roundTrip(lab, id, ko);
  for (const field of lab.fields) {
    const config = field.config, values = config.type === 'select' ? config.options.map(o => o.value) : config.type === 'checkbox' ? [false, true] : Array.from({ length: Math.round((config.max - config.min) / config.step) + 1 }, (_, i) => Number((config.min + i * config.step).toFixed(10)));
    for (const value of values) { if (config.type === 'checkbox') field.checked = value; else field.value = value; lab.render(); verifyPlottedModel(lab, id); lifecycle++; }
    lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++;
  }
  for (const state of corners[id]) { assert.equal(lab.state.write(state), true); verifyPlottedModel(lab, id); roundTrip(lab, id, ko); lifecycle++; }
  for (const button of lab.buttons) { button.click(); verifyPlottedModel(lab, id); roundTrip(lab, id, ko); lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++; }
  const invalids = {
    'drude-packet': [{ alpha: -1.1 }, { alpha: .05 }, { s: 3.1 }, { s: .05 }],
    'drude-transition': [{ couplingMeV: 1.1 }, { couplingMeV: .05 }, { timeFs: .5 }, { timeFs: 51 }, { probeMeV: 2 }, { probeMeV: 155 }, { reference: 1 }],
    'drude-gaussian': [{ width: .4 }, { width: .55 }, { q: -.05 }, { q: .025 }, { q: 4.05 }, { mode: 'both' }]
  };
  const current = JSON.parse(JSON.stringify(lab.state.read())), snapshot = lab.snapshot();
  for (const partial of invalids[id]) { assert.equal(lab.state.write({ ...current, ...partial }), false); assert.equal(lab.snapshot(), snapshot); lifecycle += 2; }
  // Even a payload with several valid changed fields plus a bad final field
  // must leave all controls and rendered state untouched.
  const candidate = { ...corners[id][corners[id].length - 1], ...invalids[id][invalids[id].length - 1] };
  assert.equal(lab.state.write(candidate), false); assert.equal(lab.snapshot(), snapshot); lifecycle += 2;
  lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++;
}
console.log(`Drude quantum: ${physics} physics/plot-data assertions; ${lifecycle} bilingual UI/state/preset/reset assertions; ${plotted} finite, in-domain plotted points checked. Browser geometry not tested by this adapter.`);
