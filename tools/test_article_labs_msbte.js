'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const m = require('../assets/js/article-labs-msbte.js');
let checks = 0;
function close(a, b, tol = 1e-10) { assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`); checks++; }
function throws(fn) { assert.throws(fn); checks++; }
const modes = m.subbands(10, 8, .19, 8);
assert.deepEqual(modes.slice(0, 3).map(x => [x.ny, x.nz]), [[1, 1], [2, 1], [1, 2]]); checks++;
assert.ok(modes[0].edge > 50 && modes[0].edge < 51); checks++;
const doubled = m.subbands(20, 16, .19, 8); modes.forEach((v, i) => close(v.edge / 4, doubled[i].edge));
close(m.fermi(100, 100, 300), .5);
close(m.fermi(1e9, 0, 300), 0); close(m.fermi(-1e9, 0, 300), 1);
const edges = modes.map(x => x.edge);
close(m.ballisticCurrent(edges, 100, 100, 300), 0);
close(m.ballisticCurrent(edges, 115, 95, 300), -m.ballisticCurrent(edges, 95, 115, 300));
assert.ok(m.ballisticCurrent(edges, 115, 95, 300) < 0); checks++;
// Numerical quadrature independently checks the analytic Landauer integral.
const de = .02; let integral = 0;
edges.forEach(edge => { for (let i = 0; i < 50000; i++) { const e = edge + (i + .5) * de; integral += (m.fermi(e, 115, 300) - m.fermi(e, 95, 300)) * de; } });
close(-m.G0 / 1000 * integral, m.ballisticCurrent(edges, 115, 95, 300), 1e-8);
for (const open of [1, 2]) for (const beta of [0, .005, .04]) for (const gamma of [0, .01, .04]) {
  const result = m.solveShell({ modes: open, beta, gamma, cells: 12 });
  result.occupation.flat().forEach(f => { assert.ok(f >= .2 - 1e-12 && f <= .9 + 1e-12); checks++; });
  result.flux.forEach(f => close(f, result.flux[0]));
  const equilibrium = m.solveShell({ modes: open, beta, gamma, cells: 12, fL: .4, fR: .4 });
  equilibrium.occupation.flat().forEach(f => close(f, .4)); equilibrium.flux.forEach(f => close(f, 0));
}
const ballistic = m.solveShell({ modes: 2, beta: 0, gamma: 0 });
ballistic.occupation.forEach(row => { close(row[0], .9); close(row[1], .2); close(row[2], .9); close(row[3], .2); });
for (const cells of [12, 24, 36]) {
  const beta = .01, result = m.solveShell({ modes: 1, cells, beta, gamma: .04 });
  const expected = .7 * (1 + beta * result.dx) / (1 + beta * 100 + beta * result.dx);
  result.flux.forEach(f => close(f, expected));
  const noGamma = m.solveShell({ modes: 1, cells, beta, gamma: 0 }); close(result.flux[0], noGamma.flux[0]);
}
const error = n => Math.abs(m.solveShell({ modes: 1, cells: n, beta: .01 }).flux[0] - .7 / 2);
assert.ok(error(36) < error(24) && error(24) < error(12)); checks++;
const kappa = m.conductances(2, .01, .02), speeds = [.2, .2, .5, .5], c = m.collision([.9, .4, .7, .2], speeds, kappa);
close(c.reduce((s, v, i) => s + v / speeds[i], 0), 0);
m.collision([.4, .4, .4, .4], speeds, kappa).forEach(v => close(v, 0));
const filtered = m.solveShell({ modes: 2, fL: [.9, .3], fR: [.2, .2], gamma: .02 });
filtered.flux.forEach(f => close(f, filtered.flux[0]));
assert.ok(Math.abs(filtered.modeFlux[0][0] - filtered.modeFlux.at(-1)[0]) > .05); checks++;
// Symmetric elastic transition factors cancel even at large occupations.
for (const f of [.1, .8, .99]) for (const g of [.05, .7, 1]) close(g * (1 - f) - f * (1 - g), g - f);
for (let i = 0; i < 4; i++) { const q = m.quizProblem(i); assert.ok(q.answer >= 0 && q.answer < q.choices.length); checks++; }
throws(() => m.subbands(0)); throws(() => m.subbands(10, 8, .19, 0)); throws(() => m.fermi(0, 0, 0)); throws(() => m.solveShell({ cells: 100 })); throws(() => m.solveShell({ beta: -.1 })); throws(() => m.solveShell({ fL: 2 })); throws(() => m.solveShell({ gamma: NaN })); throws(() => m.quizProblem(4));
// Sparse JS arrays must not bypass finite-state validation via Array.every.
throws(() => m.ballisticCurrent(Array(2), 115, 95, 300));
throws(() => m.ballisticCurrent([50, , 110], 115, 95, 300));
throws(() => m.collision([, .2], [1, 1], [[0, .1], [.1, 0]]));
throws(() => m.collision([.8, .2], [, 1], [[0, .1], [.1, 0]]));
throws(() => m.collision([.8, .2], [1, 1], [[0, .1], Array(2)]));
throws(() => m.collision([.8, .2], [1, 1], Array(2)));
throws(() => m.collision([.8, .2], [1, 1], [[1, .1], [.1, 1]]));
throws(() => m.collision([], [], []));
const previousModelChecks = checks;
// Exact initial-value examples: no contact solver or combined dynamics.
for (let j = 0; j <= 40; j++) {
  const t = j / 10, centre = -30 + 20 * t;
  close(m.packetOccupation(centre, t), .9, 1e-14);
  close(m.packetOccupation(centre - 12, t), .1 + .8 * Math.exp(-.5), 1e-14);
  close(m.packetOccupation(centre + 12, t), .1 + .8 * Math.exp(-.5), 1e-14);
  for (const x of [-100, -30, 0, 80, 140]) {
    close(m.packetOccupation(x, t), m.packetOccupation(x - 20 * t, 0), 1e-14);
    assert.ok(m.packetOccupation(x, t) >= .1 && m.packetOccupation(x, t) <= .9); checks++;
  }
  for (let k = 0; k <= 20; k++) {
    const r = k / 20, pair = m.relaxingPair(t, r);
    close(pair.plus + pair.minus, 1, 1e-14); close(pair.mean, .5, 1e-14);
    close(pair.plus - pair.minus, pair.difference, 1e-14);
    close(pair.difference, .6 * Math.exp(-2 * r * t), 1e-14);
    assert.ok(pair.plus >= .5 && pair.plus <= .8 && pair.minus >= .2 && pair.minus <= .5); checks++;
    if (r === 0 || t === 0) { close(pair.plus, .8, 1e-14); close(pair.minus, .2, 1e-14); }
  }
}
for (const x of [-35, 0, 60]) {
  const dx = 1e-3, dt = 1e-5, t = 1.3;
  const temporal = (m.packetOccupation(x, t + dt) - m.packetOccupation(x, t - dt)) / (2 * dt);
  const spatial = (m.packetOccupation(x + dx, t) - m.packetOccupation(x - dx, t)) / (2 * dx);
  close(temporal + 20 * spatial, 0, 1e-7);
}
for (const r of [0, .4, 1]) {
  const dt = 1e-5, t = 1.3, a = m.relaxingPair(t - dt, r), b = m.relaxingPair(t + dt, r), pair = m.relaxingPair(t, r);
  close((b.plus - a.plus) / (2 * dt), r * (pair.minus - pair.plus), 1e-10);
  close((b.minus - a.minus) / (2 * dt), -r * (pair.minus - pair.plus), 1e-10);
  const blockedGainMinusLoss = r * pair.minus * (1 - pair.plus) - r * pair.plus * (1 - pair.minus);
  close(blockedGainMinusLoss, r * (pair.minus - pair.plus), 1e-14);
  close(m.relaxingPair(1.9, r).difference, m.relaxingPair(.7, r).difference * Math.exp(-2 * r * 1.2), 1e-14);
}
close(m.relaxingPair(1, .5).difference, .6 / Math.E, 1e-14);
for (const run of [() => m.packetOccupation(NaN, 0), () => m.packetOccupation(Infinity, 0), () => m.packetOccupation(0, -.1), () => m.packetOccupation(0, 4.1), () => m.packetOccupation(0, '1'), () => m.relaxingPair(NaN, .4), () => m.relaxingPair(-.1, .4), () => m.relaxingPair(4.1, .4), () => m.relaxingPair(1, -.1), () => m.relaxingPair(1, 1.1), () => m.relaxingPair(1, Infinity), () => m.relaxingPair(1, '.4')]) throws(run);
const basicsModelChecks = checks - previousModelChecks;
const auditModelStart = checks;
for (const modes of [1, 2]) for (const beta of [0, .005, .04]) for (const gamma of [0, .003, .04]) for (const cells of [12, 24, 36]) {
  const estimate = m.symmetricTransmission(modes, 100, beta, gamma, cells);
  const solved = m.solveShell({ modes, length: 100, beta, gamma, fL: .9, fR: .2, cells });
  // Compare an independent closed-form symmetry reduction with the solved
  // many-cell equations, not with a ratio of possibly zero contact currents.
  close(solved.flux[0] / (modes * .7), estimate.grid, 2e-12);
  close(estimate.alpha, beta + (modes - 1) * gamma / 2);
  close(estimate.relativeErrorPercent, 100 * (estimate.grid / estimate.continuum - 1));
  assert.ok(estimate.relativeErrorPercent >= -1e-12); checks++;
}
const coarseSymmetry = m.symmetricTransmission(2, 100, .04, .04, 12);
close(coarseSymmetry.grid, .2); close(coarseSymmetry.continuum, 1 / 7); close(coarseSymmetry.relativeErrorPercent, 40);
close(m.symmetricTransmission(2, 100, .04, .04, 36).relativeErrorPercent, 600 / 43);
close(m.symmetricTransmission(2, 100, .005, .003, 24).relativeErrorPercent, 169 / 161);
for (const cells of [12, 24, 36]) close(m.symmetricTransmission(1, 100, 0, .04, cells).relativeErrorPercent, 0);
const counterflow = m.solveShell({ modes: 2, fL: [.9, .1], fR: [.5, .5], beta: .005, gamma: .04, cells: 36 });
counterflow.flux.forEach(q => close(q, 0));
assert.ok(counterflow.modeFlux[0][0] > .02 && counterflow.modeFlux[0][1] < -.02); checks++;
for (const args of [[0, 100, .01, .01, 24], [2, 0, .01, .01, 24], [2, 100, -.01, .01, 24], [2, 100, .01, -.01, 24], [2, 100, .01, .01, 1], [2, 100, .01, .01, 37]]) throws(() => m.symmetricTransmission(...args));
const auditModelChecks = checks - auditModelStart;
const finalAuditStart = checks;
// Particle-hole symmetry and the Fermi slope are independent analytic
// checks, including the pure model's 1 K and 2000 K endpoints.
for (const temperature of [1, 300, 2000]) {
  const thermal = m.KB * temperature, mu = 110, delta = thermal * 1e-4;
  close((m.fermi(mu, mu + delta, temperature) - m.fermi(mu, mu - delta, temperature)) / (2 * delta) * 4 * thermal, 1, 2e-9);
  for (const z of [-1000, -20, -1, 0, 1, 20, 1000]) close(m.fermi(mu + z * thermal, mu, temperature) + m.fermi(mu - z * thermal, mu, temperature), 1, 2e-14);
}
// Exactly at a subband edge, integral_edge^infinity f(E;mu=edge)dE
// is kBT*ln(2). A degenerate pair contributes twice, with charge sign intact.
close(m.ballisticCurrent([110, 110], 110, -1e6, 1), -m.G0 * m.KB * Math.log(2) * 2 / 1000, 1e-14);
const extremeEdges = [50, 50, 110, 300], zeroTOverlap = extremeEdges.reduce((sum, edge) => sum + Math.max(200 - edge, 0), 0);
close(m.ballisticCurrent(extremeEdges, 200, -1e6, 1), -m.G0 * zeroTOverlap / 1000, 1e-11);
close(m.ballisticCurrent(extremeEdges, -1e6, 200, 1), m.G0 * zeroTOverlap / 1000, 1e-11);
// Brute-force a larger index square independently of subbands' count bound.
// These corners include severe aspect ratios and exact square-wire degeneracy.
for (const [width, height, mass] of [[2, 100, .01], [100, 2, 10], [8, 8, .19]]) {
  const exhaustive = [];
  for (let ny = 1; ny <= 40; ny++) for (let nz = 1; nz <= 40; nz++) exhaustive.push({ ny, nz, edge: Math.PI ** 2 * m.A0 / mass * (ny ** 2 / width ** 2 + nz ** 2 / height ** 2) });
  exhaustive.sort((a, b) => a.edge - b.edge || a.ny - b.ny || a.nz - b.nz);
  const actual = m.subbands(width, height, mass, 16);
  assert.deepEqual(actual, exhaustive.slice(0, 16)); checks++;
  const swapped = m.subbands(height, width, mass, 16);
  actual.forEach((band, i) => close(band.edge, swapped[i].edge, 1e-9));
}
const squareBands = m.subbands(8, 8, .19, 3);
assert.deepEqual(squareBands.map(band => [band.ny, band.nz]), [[1, 1], [1, 2], [2, 1]]); checks++;
close(squareBands[1].edge, squareBands[2].edge, 0.000000000001);
// Rescale collision units without changing the physical inverse-time rate.
const unitF = [.9, .1, .3, .7], unitV = [2, 2, 1, 1], unitK = m.conductances(2, .003, .02);
const perPs = m.collision(unitF, unitV, unitK), perS = m.collision(unitF, unitV.map(v => v * 1e3), unitK.map(row => row.map(k => k * 1e9)));
perPs.forEach((c, a) => close(perS[a], c * 1e12, 1e-4));
// Exercise the pure solver's stiffest length/rate and coarsest allowed grid.
const stiff = m.solveShell({ modes: 2, length: 1000, beta: .1, gamma: .1, fL: 1, fR: 0, cells: 2 });
const stiffTransmission = (1 + .15 * 500) / (1 + .15 * 1000 + .15 * 500);
stiff.flux.forEach(q => close(q, 2 * stiffTransmission, 1e-12));
stiff.occupation.flat().forEach(f => { assert.ok(f >= 0 && f <= 1); checks++; });
const stiffEquilibrium = m.solveShell({ modes: 2, length: 1000, beta: .1, gamma: .1, fL: 1, fR: 1, cells: 2 });
stiffEquilibrium.occupation.flat().forEach(f => close(f, 1, 1e-12));
stiffEquilibrium.flux.forEach(q => close(q, 0, 1e-12));
throws(() => m.solveShell({ fL: Array(2) })); throws(() => m.solveShell({ fR: [.2, ,] }));
throws(() => m.collision([.8, .2], [1, 1], [[0, .1], [.2, 0]]));
throws(() => m.collision([.8, .2], [1, 1], [[0, NaN], [NaN, 0]]));
const finalAuditChecks = checks - finalAuditStart;
// Focused UI/state adapter: this checks logic, not actual browser geometry.
class Element {
  constructor(tag = 'div', attrs = {}, children = []) { this.tagName = tag.toUpperCase(); this.attrs = attrs; this.children = []; this.listeners = {}; this.value = ''; this.checked = false; this._text = ''; this.classList = { add: () => {} }; this.append(children); }
  append(...children) { this.children.push(...children.flat(Infinity).filter(x => x !== null && x !== undefined)); }
  replaceChildren(...children) { this.children = []; this._text = ''; this.append(children); }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  click() { if (this.listeners.click) this.listeners.click(); }
  focus() { Element.active = this; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(x => x instanceof Element ? x.textContent : String(x)).join(' '); }
  querySelectorAll(selector) { const klass = selector.slice(1); return this.children.flatMap(x => x instanceof Element ? [...(String(x.attrs.class || '').split(' ').includes(klass) ? [x] : []), ...x.querySelectorAll(selector)] : []); }
}
const factories = new Map();
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-msbte.js'), 'utf8'), { window: { ArticleLabs: { register: (id, fn) => factories.set(id, fn) } } });
assert.deepEqual([...factories.keys()], ['msbte-basics', 'msbte-confinement', 'msbte-injection', 'msbte-scattering', 'msbte-quiz']);
function mount(id, ko) {
  const fields = [], defaults = [], buttons = [], metrics = []; let state, render, customReset;
  const ctx = { root: new Element(), ui: new Element(), view: new Element(), metrics: new Element(), status: new Element(), ko,
    text: (en, korean) => ko ? korean : en,
    el: (tag, attrs, children) => new Element(tag, attrs, children),
    svg: (tag, attrs, children) => new Element(tag, attrs, children),
    metric: () => { const v = new Element('output'); metrics.push(v); return v; },
    control: (_en, _ko, opts) => { const v = new Element('input'); v.value = String(opts.value); v.checked = opts.type === 'checkbox' && opts.value; fields.push(v); defaults.push({ value: v.value, checked: v.checked }); return v; },
    button: (en, korean, fn) => { buttons.push(fn); return new Element('button'); },
    plot: spec => { spec.series.forEach(s => s.points.forEach(([x, y]) => { assert.ok(Number.isFinite(x) && Number.isFinite(y)); assert.ok(x >= spec.x[0] - 1e-8 && x <= spec.x[1] + 1e-8, 'A plot must not hide out-of-range coordinates.'); assert.ok(y >= spec.y[0] - 1e-8 && y <= spec.y[1] + 1e-8, 'A plot must not hide out-of-range values.'); })); (spec.markers || []).forEach(p => { assert.ok(p.x >= spec.x[0] && p.x <= spec.x[1] && p.y >= spec.y[0] && p.y <= spec.y[1]); }); return new Element('div', { spec }, spec.title); },
    onChange: fn => { render = fn; fn(); }, onReset: fn => { customReset = fn; }, stateLink: (key, provider) => { assert.equal(key, id); state = provider; }
  };
  ctx.root.append(ctx.view, ctx.ui, ctx.status); factories.get(id)(ctx);
  return { ctx, fields, buttons, metrics, state, render, reset: () => { fields.forEach((v, i) => Object.assign(v, defaults[i])); if (customReset) customReset(); render(); }, snapshot: () => JSON.stringify({ state: state.read(), text: ctx.view.textContent, status: ctx.status.textContent, metrics: metrics.map(v => v.textContent), customMetrics: ctx.metrics.textContent }) };
}
let lifecycle = 0;
for (const id of factories.keys()) for (const ko of [false, true]) {
  const lab = mount(id, ko), initial = lab.snapshot(), saved = JSON.parse(JSON.stringify(lab.state.read()));
  for (const bad of [null, [], {}, { wrong: true }]) { assert.equal(lab.state.write(bad), false); assert.equal(lab.snapshot(), initial); lifecycle++; }
  for (const key of Object.keys(saved)) { assert.equal(lab.state.write({ ...saved, [key]: '__bad__' }), false); assert.equal(lab.snapshot(), initial); lifecycle++; }
  assert.equal(lab.state.write(saved), true); assert.equal(lab.snapshot(), initial); lifecycle++;
  lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++;
  if (id === 'msbte-basics') {
    for (const mode of ['flight', 'relaxation']) for (const time of [0, .1, 2, 4]) for (const rate of [0, .4, 1]) {
      assert.equal(lab.state.write({ mode, time, rate }), true);
      assert.equal(lab.fields[2].disabled, mode === 'flight');
      const rows = lab.ctx.metrics.children;
      assert.equal(rows.length, 6);
      rows.forEach(row => { const key = row.attrs['data-basics-metric']; assert.equal(row.hidden, ['centre', 'peak'].includes(key) ? mode !== 'flight' : mode === 'flight'); });
      const output = key => rows.find(row => row.attrs['data-basics-metric'] === key).children[1].children[0].textContent;
      assert.equal(output('centre'), (-30 + 20 * time).toFixed(1)); assert.equal(output('peak'), '0.9000');
      assert.equal(output('plus'), m.relaxingPair(time, rate).plus.toFixed(4)); assert.equal(output('minus'), m.relaxingPair(time, rate).minus.toFixed(4));
      assert.equal(output('mean'), '0.5000'); assert.equal(output('difference'), m.relaxingPair(time, rate).difference.toFixed(4));
      const snapshot = lab.snapshot(), restored = mount(id, ko);
      assert.equal(restored.state.write(JSON.parse(JSON.stringify(lab.state.read()))), true); assert.equal(restored.snapshot(), snapshot);
      restored.render(); assert.equal(restored.snapshot(), snapshot);
      lifecycle++;
    }
    const checkpoint = lab.snapshot(), base = JSON.parse(JSON.stringify(lab.state.read()));
    for (const bad of [{ ...base, mode: 'combined' }, { ...base, time: .05 }, { ...base, time: -.1 }, { ...base, time: 4.1 }, { ...base, rate: .025 }, { ...base, rate: -.05 }, { ...base, rate: 1.05 }, { ...base, time: NaN }, { ...base, rate: Infinity }]) { assert.equal(lab.state.write(bad), false); assert.equal(lab.snapshot(), checkpoint); lifecycle++; }
    lab.reset(); assert.equal(lab.snapshot(), initial); assert.equal(lab.fields[2].disabled, true); lifecycle++;
  }
  if (id === 'msbte-confinement') for (const width of [6, 10, 20]) for (const level of [0, 1, 2]) {
    assert.equal(lab.state.write({ width, level }), true);
    const curve = lab.ctx.view.children[1].attrs.spec.series[0].points;
    const area = curve.slice(1).reduce((sum, p, i) => sum + (p[0] - curve[i][0]) * (p[1] + curve[i][1]) / 2, 0);
    assert.ok(Math.abs(area - 1) < 1e-12, 'The plotted marginal probability density must normalize to one.'); lifecycle++;
  }
  if (id === 'msbte-injection') for (const muL of [50, 180]) for (const muR of [50, 180]) for (const temperature of [50, 500]) for (const count of [2, 4, 8]) { assert.equal(lab.state.write({ muL, muR, temperature, count }), true); lifecycle++; }
  if (id === 'msbte-injection') {
    lab.state.write({ muL: 50, muR: 180, temperature: 50, count: 8 });
    assert.ok(Number(lab.metrics[0].textContent) < 0 && Number(lab.metrics[1].textContent) > 0); lifecycle++;
    lab.state.write({ muL: 100, muR: 100, temperature: 300, count: 8 });
    assert.equal(Number(lab.metrics[0].textContent), 0); assert.equal(Number(lab.metrics[1].textContent), 0);
    const curves = lab.ctx.view.children[0].attrs.spec.series;
    for (const first of [0, 2]) curves[first].points.forEach((p, i) => assert.ok(Math.abs(p[1] - curves[first + 1].points.at(-1 - i)[1]) < 1e-14)); lifecycle++;
  }
  if (id === 'msbte-scattering') for (const energy of [60, 110, 140]) for (const beta of [0, .04]) for (const gamma of [0, .04]) for (const filter of [false, true]) { assert.equal(lab.state.write({ energy, beta, gamma, fL: 0, fR: 1, cells: 12, filter, fL2: .3 }), true); lifecycle++; }
  if (id === 'msbte-scattering') {
    const symmetric = { energy: 130, beta: .04, gamma: .04, fL: .9, fR: .2, cells: 12, filter: false, fL2: .3 };
    lab.state.write(symmetric); assert.equal(lab.metrics[3].textContent, '40.00');
    assert.ok(lab.ctx.status.textContent.includes('T_FV=0.200000') && lab.ctx.status.textContent.includes('T_cont=0.142857')); lifecycle++;
    lab.state.write({ ...symmetric, cells: 36 }); assert.equal(lab.metrics[3].textContent, '13.95'); lifecycle++;
    lab.state.write({ ...symmetric, fL: .2, fR: .2 }); assert.equal(lab.metrics[3].textContent, '40.00'); assert.equal(lab.metrics[0].textContent, '0.000000'); lifecycle++;
    lab.state.write({ ...symmetric, beta: .005, gamma: .003, cells: 24 }); assert.equal(lab.metrics[3].textContent, '1.05'); lifecycle++;
    lab.state.write({ ...symmetric, filter: true }); assert.equal(lab.metrics[3].textContent, '—'); lifecycle++;
    lab.state.write({ ...symmetric, filter: true, fL2: .9 }); assert.equal(lab.metrics[3].textContent, '40.00'); lifecycle++;
    for (const [energy, modes] of [[110, 1], [115, 2], [140, 2]]) {
      lab.state.write({ ...symmetric, energy, filter: true });
      assert.equal(lab.ctx.view.children[0].attrs.spec.series.length, 2 * modes);
      assert.equal(lab.fields[6].disabled, modes === 1);
      assert.equal(lab.metrics[3].textContent, modes === 1 ? '25.00' : '—'); lifecycle++;
    }
    lab.state.write({ ...symmetric, filter: true, fL: .9, fL2: .1, fR: .5 });
    const fluxChart = lab.ctx.view.children[1].attrs.spec;
    assert.ok(fluxChart.series[0].points[0][1] > 0 && fluxChart.series[1].points[0][1] < 0);
    assert.ok(Math.abs(Number(lab.metrics[0].textContent)) < 1e-12); assert.equal(lab.metrics[3].textContent, '—'); lifecycle++;
    // Hiding the second propagating mode must not erase its independently
    // prescribed inflow. Reopening and sharing must restore the same curves.
    const reopened = { ...symmetric, energy: 115, fL: .05, fL2: .95, fR: .2, filter: true };
    assert.equal(lab.state.write(reopened), true);
    const openSnapshot = lab.snapshot();
    assert.equal(lab.fields[6].disabled, false); assert.equal(lab.metrics[3].textContent, '—'); lifecycle++;
    assert.equal(lab.state.write({ ...reopened, energy: 110 }), true);
    assert.equal(lab.fields[6].disabled, true); assert.equal(lab.state.read().fL2, .95);
    assert.equal(lab.ctx.view.children[0].attrs.spec.series.length, 2); lifecycle++;
    assert.equal(lab.state.write(reopened), true); assert.equal(lab.snapshot(), openSnapshot); lifecycle++;
    const shared = mount(id, ko);
    assert.equal(shared.state.write(JSON.parse(JSON.stringify(lab.state.read()))), true);
    assert.equal(shared.snapshot(), openSnapshot); lifecycle++;
    for (const bad of [{ ...reopened, energy: 112.5 }, { ...reopened, cells: 13 }, { ...reopened, fL2: .975 }, { ...reopened, filter: 1 }]) {
      assert.equal(lab.state.write(bad), false); assert.equal(lab.snapshot(), openSnapshot); lifecycle++;
    }
    lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++;
  }
  if (id === 'msbte-quiz') for (let index = 0; index < 4; index++) {
    const q = m.quizProblem(index); assert.equal(lab.state.write({ index, selected: q.answer, checked: false }), true);
    if (index < 2) {
      const diagram = lab.ctx.view.querySelectorAll('.msbte-energy-diagram')[0];
      assert.ok(diagram && diagram.textContent.includes('Subband 1') && diagram.textContent.includes('Subband 2'));
      assert.ok(!diagram.textContent.includes('0.25') && !diagram.textContent.includes('0.75'));
      assert.equal(diagram.children.filter(node => node instanceof Element && node.tagName === 'CIRCLE').length, q.edges.filter(edge => q.energy > edge).length);
      lifecycle++;
    }
    lab.buttons[0](); assert.ok(lab.ctx.view.textContent.includes(ko ? '맞았습니다' : 'Correct.'));
    assert.equal(lab.ctx.view.querySelectorAll('.msbte-choice').length, 3);
    lab.ctx.view.querySelectorAll('.msbte-choice')[1].click(); assert.equal(Element.active, lab.ctx.view.querySelectorAll('.msbte-choice')[1]); lifecycle++;
  }
  assert.ok(!lab.ctx.view.textContent.includes('undefined')); lifecycle++;
}
console.log(`MS-BTE lab models: ${checks} checks (${previousModelChecks} core + ${basicsModelChecks} BTE basics + ${auditModelChecks} transmission audit + ${finalAuditChecks} final numerical audit); UI/state lifecycle: ${lifecycle} checks passed.`);
