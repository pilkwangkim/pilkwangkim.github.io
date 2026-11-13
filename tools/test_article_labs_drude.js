'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const m = require('../assets/js/article-labs-drude.js');
let physics = 0, lifecycle = 0, plotted = 0;
function close(a, b, tolerance = 1e-10) { assert.ok(Number.isFinite(a) && Math.abs(a - b) <= tolerance, `${a} != ${b} (tol=${tolerance})`); physics++; }
function check(value) { assert.ok(value); physics++; }
function throws(fn) { assert.throws(fn); physics++; }
const trapezoid = points => points.slice(1).reduce((sum, p, i) => sum + (p[0] - points[i][0]) * (p[1] + points[i][1]) / 2, 0);
const sample = (lo, hi, count, fn) => Array.from({ length: count }, (_, i) => { const x = lo + (hi - lo) * i / (count - 1); return [x, fn(x)]; });

const base = m.stepResponse({ s: 100 });
close(base.mobility * 1e4, 879.410004189, 1e-8); close(base.conductivity, 1408.97016042, 1e-8); close(base.drift, -87.941000419, 1e-8); close(base.currentNA, 14.0897016042, 1e-9);
close(base.voltage, .001); close(base.currentNA, base.currentDensity * 1e-5);
for (const field of [-2000, -100, 0, 100, 2000]) for (let j = 0; j <= 60; j++) {
  const s = j / 10, p = m.stepResponse({ field, s });
  close(p.currentDensity, -m.E * 1e23 * p.drift, 1e-8); check(p.currentDensity * field >= 0);
  close(p.drift, -base.mobility * field * (1 - Math.exp(-s)), 1e-12);
  if (s > 0) {
    const h = 1e-5, derivative = (m.stepResponse({ field, s: s + h }).drift - m.stepResponse({ field, s: s - h }).drift) / (2 * h);
    close(derivative, -base.mobility * field - p.drift, 1e-7);
  }
}
for (let j = 0; j <= 100; j++) {
  const w = j / 20, p = m.acResponse(w);
  close(p.real + w * p.imaginary, 1); close(p.imaginary - w * p.real, 0);
  close(p.magnitude, Math.hypot(p.real, p.imaginary)); check(p.real >= 0 && p.imaginary >= 0);
  close(m.acResponse(w, p.lag).current, p.magnitude);
  close((m.acResponse(w, .7 + 1e-5).current - m.acResponse(w, .7 - 1e-5).current) / 2e-5 * w + m.acResponse(w, .7).current, Math.cos(.7), 2e-10);
}
for (const t of [.1, 1, 3]) for (const a of [.1, .5, 1]) for (let j = -100; j <= 100; j++) {
  const k = j / 100 * Math.PI, p = m.tightBinding(k, t, a), h = 1e-4;
  check(p.energyEV >= 0 && p.energyEV <= 4 * t + 1e-12);
  if (Math.abs(k) < Math.PI - h) {
    const plus = m.tightBinding(k + h, t, a).energyEV, minus = m.tightBinding(k - h, t, a).energyEV;
    close((plus - minus) / (2 * h) * m.E * a * 1e-9 / m.HBAR, p.velocity, Math.max(1e-4, Math.abs(p.velocity) * 1e-7));
    close((plus - 2 * p.energyEV + minus) / h ** 2 * m.E * (a * 1e-9) ** 2 * m.M0 / m.HBAR ** 2, p.inverseMassM0, 5e-5);
  }
}
for (const t of [.1, 1, 3]) for (const a of [.1, .5, 1]) for (const kNM of [-32, -15, -.25, 0, .25, 15, 32]) {
  const p = m.bandAtK(kNM, t, a), ka = kNM * a;
  close(p.energyEV, 2 * t * (1 - Math.cos(ka)), 1e-12);
  close(p.velocity, 2 * t * m.E * a * 1e-9 / m.HBAR * Math.sin(ka), 1e-7);
  close(p.inverseMassM0, 2 * t * m.E * (a * 1e-9) ** 2 * m.M0 / m.HBAR ** 2 * Math.cos(ka), 1e-12);
}
for (const x of [0, 1e-12, 1e-8, .0001, .01, .1, 1, 10, 400]) {
  const normalized = sample(0, Math.PI, 20001, t => m.scatteringShape(x, t).pdf), weighted = sample(0, Math.PI, 20001, t => m.scatteringShape(x, t).transportWeighted), p = m.scatteringShape(x);
  close(trapezoid(normalized), 1, 1e-6); close(trapezoid(weighted), p.meanLoss, 1e-7); close(1 / p.meanLoss, p.ratio); check(p.ratio >= 1 - 1e-12);
}
close(m.scatteringShape(0).ratio, 1); close(m.scatteringShape(1e-8).ratio, 1 + 1e-8 / 3, 1e-15);
for (const w of [0, .1, .49, .5 - 1e-12, .5, .5 + 1e-12, .51, 1, 3]) {
  close(m.chargeMode(0, w).a, 1); close(m.chargeMode(0, w).derivative, 0);
  for (let j = 0; j <= 120; j++) {
    const s = j / 10, p = m.chargeMode(s, w);
    check(Math.abs(p.a) <= 1 + 1e-12); close(p.secondDerivative + p.derivative + w ** 2 * p.a, 0);
    if (s > 0) {
      const h = 1e-5, plus = m.chargeMode(s + h, w), minus = m.chargeMode(s - h, w);
      close((plus.a - minus.a) / (2 * h), p.derivative, 1e-8);
      close((plus.derivative - minus.derivative) / (2 * h) + p.derivative + w ** 2 * p.a, 0, 2e-8);
      const x = .23, xp = m.chargeProfile(x + h, s, w), xm = m.chargeProfile(x - h, s, w), tp = m.chargeProfile(x, s + h, w), tm = m.chargeProfile(x, s - h, w);
      close(-(tp.densityFraction - tm.densityFraction) / (.05 * 2 * h) + (xp.current - xm.current) / (4 * Math.PI * h), 0, 2e-8);
      close(m.chargeProfile(x, s, w).continuityResidual, 0);
    }
  }
}
for (const s of [0, 1, 12, 100]) { close(m.chargeMode(s, .5).a, Math.exp(-s / 2) * (1 + s / 2)); close(m.chargeMode(s, .5).derivative, -s / 4 * Math.exp(-s / 2)); }
for (const omega of [.1, .5, 1, 3]) {
  const u=m.chargeUnits(omega);
  close(u.density*m.E**2/(u.permittivity*u.mass),(omega/u.tau)**2,1e12);
  close(u.initialAmplitude/u.density,.05);
  close(u.q*u.length,2*Math.PI);
  close(u.fieldScale*u.permittivity*u.q/(m.E*u.initialAmplitude),1);
  close(u.currentScale*u.q*u.tau/(m.E*u.initialAmplitude),1);
}
for (let i = 0; i < 5; i++) { const q = m.quizProblem(i); check(q.choices.length === 3 && q.answer >= 0 && q.answer < 3 && q.en && q.ko && q.explanation.length === 2); }
for (const fn of [() => m.stepResponse({ density: NaN }), () => m.stepResponse({ massRatio: 0 }), () => m.stepResponse({ s: -1 }), () => m.stepResponse({ tauFs: '100' }), () => m.acResponse(-1), () => m.acResponse(Infinity), () => m.tightBinding(Math.PI + .1), () => m.tightBinding(0, 0), () => m.screeningIntegral(-1), () => m.scatteringShape(1, -1), () => m.scatteringShape(NaN), () => m.chargeMode(-1, 1), () => m.chargeMode(1, NaN), () => m.chargeProfile(2, 1, 1), () => m.quizProblem(5), () => m.quizProblem('0')]) throws(fn);

// Focused state/UI adapter; browser rendering and draft validation are tested
// by the shared runtime's own suite, not represented as geometry evidence here.
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
  allButtons() { return this.children.flatMap(x => x instanceof Element ? [...(x.tagName === 'BUTTON' ? [x] : []), ...x.allButtons()] : []); }
}
const factories = new Map();
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-drude.js'), 'utf8'), { window: { ArticleLabs: { register: (id, fn) => { assert.ok(!factories.has(id)); factories.set(id, fn); } } } });
assert.deepEqual([...factories.keys()], ['drude-step', 'drude-ac', 'drude-band', 'drude-scattering', 'drude-charge', 'drude-quiz']); lifecycle++;
const registry = fs.readFileSync(require.resolve('../_data/article_labs_drude.yml'), 'utf8');
assert.deepEqual(registry.match(/^drude-[a-z]+(?=:)/gm), [...factories.keys()]); lifecycle++;
function mount(id, ko = false) {
  const fields = [], defaults = [], metrics = []; let state, render, reset;
  const ctx = { root: new Element(), ui: new Element(), view: new Element(), metrics: new Element(), status: new Element(), ko,
    text: (en, korean) => ko ? korean : en, format: (v, digits = 4) => String(Number(v.toFixed(digits))),
    el: (tag, attrs, children) => new Element(tag, attrs, children),
    metric: () => { const output = new Element('output'); metrics.push(output); return output; },
    control: (_en, _ko, config) => { const el = new Element(config.type === 'select' ? 'select' : 'input'); el.config = config; el.value = config.value; el.checked = config.type === 'checkbox' && Boolean(config.value); fields.push(el); defaults.push({ value: el.value, checked: el.checked }); return el; },
    plot: spec => {
      assert.ok(spec.x[0] < spec.x[1] && spec.y[0] < spec.y[1]);
      const inspect = (x, y) => { assert.ok(Number.isFinite(x) && Number.isFinite(y)); assert.ok(x >= spec.x[0] - 1e-9 && x <= spec.x[1] + 1e-9, 'x outside plot'); assert.ok(y >= spec.y[0] - 1e-9 && y <= spec.y[1] + 1e-9, `y=${y} outside ${spec.y}`); plotted++; };
      (spec.series || []).forEach(line => line.points.forEach(([x, y]) => inspect(x, y))); (spec.markers || []).forEach(p => inspect(p.x, p.y)); (spec.bars || []).forEach(p => inspect(p.x, p.y));
      return new Element('div', { spec }, spec.title);
    },
    onChange: fn => { render = fn; fn(); }, onReset: fn => { reset = fn; }, stateLink: (key, provider) => { assert.equal(key, id); state = provider; }
  };
  factories.get(id)(ctx);
  const lab = { fields, metrics, ctx, state, render, reset: () => { fields.forEach((field, i) => Object.assign(field, defaults[i])); reset?.(); render(); }, snapshot: () => JSON.stringify({ state: state.read(), view: ctx.view, metrics: metrics.map(v => v.textContent), status: ctx.status.textContent, disabled: fields.map(f => f.disabled) }, (key, value) => key === 'listeners' ? undefined : value) };
  return lab;
}
function roundTrip(lab, id, ko) { const restored = mount(id, ko), state = JSON.parse(JSON.stringify(lab.state.read())); assert.equal(restored.state.write(state), true); assert.equal(restored.snapshot(), lab.snapshot()); restored.render(); assert.equal(restored.snapshot(), lab.snapshot()); lifecycle += 3; }
for (const id of factories.keys()) for (const ko of [false, true]) {
  const lab = mount(id, ko), initial = lab.snapshot(), saved = JSON.parse(JSON.stringify(lab.state.read()));
  for (const bad of [null, false, [], {}, { wrong: 1 }]) { assert.equal(lab.state.write(bad), false); assert.equal(lab.snapshot(), initial); lifecycle += 2; }
  for (const key of Object.keys(saved)) for (const bad of ['__bad__', null, NaN, Infinity, {}, []]) { assert.equal(lab.state.write({ ...saved, [key]: bad }), false); assert.equal(lab.snapshot(), initial); lifecycle += 2; }
  roundTrip(lab, id, ko);
  // Every range/checkbox/select is exercised directly through live render.
  for (const field of lab.fields) {
    const config = field.config;
    const values = config.type === 'select' ? config.options.map(o => o.value) : config.type === 'checkbox' ? [false, true] : [config.min, config.min + config.step, Number(((config.min + config.max) / 2 / config.step).toFixed(0)) * config.step, config.max];
    for (const value of values) { if (config.type === 'checkbox') field.checked = value; else field.value = value; lab.render(); assert.ok(lab.ctx.view.children.length); lifecycle++; }
    lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++;
  }
  if (id === 'drude-step') for (const n of [.1, 10]) for (const mass of [.1, 2]) for (const tau of [10, 500]) for (const field of [-2000, 0, 2000]) for (const time of [0, 6]) for (const view of [3, 30, 300]) {
    assert.equal(lab.state.write({ n, mass, tau, field, time, view }), true); check(Number(lab.metrics[0].textContent) > 0 && Number(lab.metrics[1].textContent) > 0); roundTrip(lab, id, ko);
    const spec = lab.ctx.view.children[0].attrs.spec, scale = 1e6;
    const expected = m.stepResponse({ density: n * 1e23, massRatio: mass, tauFs: tau, field, s: time });
    assert.deepEqual(JSON.parse(JSON.stringify(spec.x)), [0, 3]); assert.deepEqual(JSON.parse(JSON.stringify(spec.y)), [-view, view]);
    const outside = Math.abs(expected.steadyCurrentDensity) > view * scale;
    assert.equal(spec.series.length, outside ? 2 : 3);
    assert.deepEqual(JSON.parse(JSON.stringify(spec.series.at(-1).points)),[[time*tau/1000,-view],[time*tau/1000,view]]);
    if (!outside) close(spec.series[1].points[0][1] * scale, expected.steadyCurrentDensity, 1e-7);
    const markerOutside = Math.abs(expected.currentDensity) > view * scale;
    close(spec.markers[0].y * scale, markerOutside ? Math.sign(expected.currentDensity) * view * scale : expected.currentDensity, 1e-7);
    if (markerOutside) assert.ok(spec.markers[0].label.length > 0);
    close(spec.markers[0].x, time * tau / 1000);
    assert.ok(spec.series[0].points.every(([ps, value]) => Math.abs(value * scale - m.stepResponse({ density: n * 1e23, massRatio: mass, tauFs: tau, field, s: ps * 1000 / tau }).currentDensity) < Math.max(1e-9, Math.abs(expected.steadyCurrentDensity) * 1e-14)));
    if (outside) { assert.ok(lab.ctx.status.textContent.includes(ko ? '범위 밖' : 'outside')); if (spec.series[0].points.at(-1)[0] < 3) close(Math.abs(spec.series[0].points.at(-1)[1]), view, 1e-10); }
    assert.equal(spec.xLabel, 't (ps)'); lifecycle += 8;
  }
  if (id === 'drude-ac') {
    let acAxes;
    for (const omega of [0, .05, 1, 5]) for (const timePs of [0, .02, 1, 2]) {
      assert.equal(lab.state.write({ omega, timePs }), true); assert.equal(lab.fields[1].disabled, false); roundTrip(lab, id, ko);
      const charts=lab.ctx.view.children.slice(0,2).map(e=>e.attrs.spec), axes=JSON.stringify(charts.map(c=>[c.x,c.y,c.xLabel,c.yLabel]));
      if (acAxes) assert.equal(axes,acAxes); else acAxes=axes;
      const sigma=m.stepResponse().conductivity;
      charts[1].series[0].points.forEach(([ps,y])=>close(y,Math.cos(omega*ps/.1),1e-12));
      charts[1].series[1].points.forEach(([ps,y])=>close(y,sigma*1000/1e6*m.acResponse(omega,omega*ps/.1).current,1e-12));
    }
    lab.state.write({ omega: 0, timePs: 1.5 }); lab.state.write({ omega: 1, timePs: 1.5 }); assert.equal(lab.state.read().timePs, 1.5); lifecycle += 2;
  }
  if (id === 'drude-band') for (const k of [-1, -.5, 0, .5, 1]) for (const hopping of [.1, 3]) for (const lattice of [.1, 1]) for (const parabola of [false, true]) {
    assert.equal(lab.state.write({ k, hopping, lattice, parabola }), true); assert.equal(lab.ctx.view.children[0].attrs.spec.series.length, parabola ? 4 : 3); roundTrip(lab, id, ko);
    const charts = lab.ctx.view.children.slice(0,3).map(e=>e.attrs.spec);
    for (const chart of charts) { assert.deepEqual(JSON.parse(JSON.stringify(chart.x)),[-32,32]); assert.equal(chart.xLabel,'k (nm⁻¹)'); close(chart.markers[0].x,k*Math.PI/lattice); }
    assert.deepEqual(JSON.parse(JSON.stringify(charts.map(c=>c.y))), [[0,12.5],[-10,10],[-85,85]]);
    charts[0].series[0].points.forEach(([x,y])=>close(y,m.bandAtK(x,hopping,lattice).energyEV,1e-12));
    charts[1].series[0].points.forEach(([x,y])=>close(y,m.bandAtK(x,hopping,lattice).velocity/1e6,1e-12));
    charts[2].series[0].points.forEach(([x,y])=>close(y,m.bandAtK(x,hopping,lattice).inverseMassM0,1e-12));
    if (Math.abs(k) === 1) { assert.equal(lab.metrics[1].textContent, '0'); lifecycle++; }
    if (Math.abs(k) === .5) { assert.equal(lab.metrics[2].textContent, '0'); lifecycle++; }
  }
  if (id === 'drude-scattering') for (const mode of ['screened', 'isotropic', 'reverse']) for (const shape of [.01, 10, 400]) {
    assert.equal(lab.state.write({ mode, shape }), true); assert.equal(lab.fields[1].disabled, mode !== 'screened');
    if (mode === 'isotropic') assert.equal(lab.metrics[0].textContent, '1'); if (mode === 'reverse') assert.equal(lab.metrics[0].textContent, '0.5');
    roundTrip(lab, id, ko); lifecycle++;
  }
  if (id === 'drude-charge') for (const omega of [.1, .45, .5, .55, 1, 3]) for (const time of [0, .1, 2, 12]) {
    assert.equal(lab.state.write({ omega, time }), true); assert.equal(Number(lab.metrics[4].textContent), 0); roundTrip(lab, id, ko);
    const charts=lab.ctx.view.children.slice(0,4).map(e=>e.attrs.spec), u=m.chargeUnits(omega), p=m.chargeMode(time,omega);
    assert.deepEqual(JSON.parse(JSON.stringify(charts.map(c=>[c.x,c.y]))),[[[0,1.2],[-3.5,3.5]],[[0,100],[-3.5,3.5]],[[0,100],[-1,1]],[[0,100],[-3,3]]]);
    close(charts[0].markers[0].x,time/10); close(charts[0].markers[0].y,u.initialAmplitude/1e22*p.a);
    charts[1].series[0].points.forEach(([x,y])=>close(y,u.initialAmplitude/1e22*p.a*Math.cos(2*Math.PI*x/100)));
    charts[2].series[0].points.forEach(([x,y])=>close(y,u.fieldScale/1e6*m.chargeProfile(x/100,time,omega).field));
    charts[3].series[0].points.forEach(([x,y])=>close(y,u.currentScale/1e9*m.chargeProfile(x/100,time,omega).current));
  }
  if (id === 'drude-quiz') {
    for (let question = 0; question < 5; question++) for (let selected = -1; selected < 3; selected++) {
      assert.equal(lab.state.write({ question, selected }), true); assert.equal(lab.ctx.view.allButtons().length, 3); roundTrip(lab, id, ko);
      if (selected >= 0) { lab.ctx.view.allButtons()[selected].click(); assert.equal(lab.state.read().selected, selected); lifecycle++; }
    }
    lab.state.write({ question: 0, selected: 1 }); lab.fields[0].value = '1'; lab.render(); assert.equal(lab.state.read().selected, -1); lifecycle++;
  }
  // Invalid off-grid/out-of-range shares are transactional, even when the
  // setting currently belongs to an inactive control.
  const current = JSON.parse(JSON.stringify(lab.state.read())), snapshot = lab.snapshot();
  const invalids = { 'drude-step': [{ n: .15 }, { mass: 0 }, { tau: 11 }, { field: 2100 }, { time: 6.1 }, {view:4}], 'drude-ac': [{ omega: .025 }, { timePs: 2.02 }, {timePs:.01}], 'drude-band': [{ k: 1.01 }, { hopping: .15 }, { lattice: .11 }, { parabola: 1 }], 'drude-scattering': [{ mode: 'other' }, { shape: 0 }, { shape: .015 }, { shape: 400.01 }], 'drude-charge': [{ omega: .525 }, { time: 12.1 }], 'drude-quiz': [{ question: 5 }, { selected: 3 }, { selected: .5 }] };
  for (const partial of invalids[id]) { assert.equal(lab.state.write({ ...current, ...partial }), false); assert.equal(lab.snapshot(), snapshot); lifecycle += 2; }
  lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++;
}
console.log(`Drude: ${physics} physics assertions; ${lifecycle} bilingual UI/state/reset assertions; ${plotted} finite, in-domain plotted points checked. Geometry not tested by this adapter.`);
