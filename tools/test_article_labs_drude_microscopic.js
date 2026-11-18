'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const m = require('../assets/js/article-labs-drude-microscopic.js');
let physics = 0, lifecycle = 0, plotted = 0;
function close(a, b, tolerance = 1e-10) { assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tolerance, `${a} != ${b}; tolerance ${tolerance}`); physics++; }
function relative(a, b, tolerance = 1e-11) { close(a / b, 1, tolerance); }
function check(value) { assert.ok(value); physics++; }
function throws(fn) { assert.throws(fn); physics++; }
function simpson(fn, lo, hi, count = 20000) {
  const h = (hi - lo) / count; let sum = fn(lo) + fn(hi);
  for (let i = 1; i < count; i++) sum += (i % 2 ? 4 : 2) * fn(lo + i * h);
  return sum * h / 3;
}

// Independent scalar and vector checks, including both angular endpoints.
for (let degree = 0; degree <= 180; degree++) for (const r of [.1, 1, 5]) {
  const p = m.angularChannel(degree, r), theta = degree * Math.PI / 180;
  close(p.forward ** 2 + p.transverse ** 2, 1);
  close(p.loss, 1 - Math.cos(theta)); close(p.transportRate, r * (1 - Math.cos(theta)));
  check(p.loss >= 0 && p.loss <= 2);
  if (degree === 0) { check(p.isIdentity); assert.equal(p.transportTimePs, null); assert.equal(p.timeRatio, null); physics += 2; }
  else { relative(p.transportTimePs * p.transportRate, 1); relative(p.timeRatio, r * p.transportTimePs); }
  // Azimuthal averaging of outgoing vectors has zero transverse components.
  let transverseX = 0, transverseY = 0;
  for (let j = 0; j < 16; j++) { const phi = j * 2 * Math.PI / 16; transverseX += Math.sin(theta) * Math.cos(phi) / 16; transverseY += Math.sin(theta) * Math.sin(phi) / 16; }
  close(transverseX, 0); close(transverseY, 0);
}
close(m.angularChannel(90).transportTimePs, 1); close(m.angularChannel(180).transportTimePs, .5);
relative(m.angularChannel(1e-7).loss, .5 * (1e-7 * Math.PI / 180) ** 2);

// Derive the same absolute rate via DOS and incident flux, not normalized ratios.
const impurity = m.shortRange(50);
close(impurity.tau * 1e12, .67086393263996, 2e-12);
close(impurity.meanFreePath * 1e9, 198.9436788649, 1e-9);
for (const energy of [1, 10, 50, 150, 500]) for (const density of [.1, .2, 1, 2, 5]) for (const amplitude of [.1, .5, 1, 1.5, 3]) {
  const p = m.shortRange(energy, density, amplitude);
  const dosRate = 2 * Math.PI / m.HBAR * 1e25 * density * p.u0 ** 2 * p.dosOneSpin;
  relative(p.rate, dosRate); relative(p.rate / impurity.rate, density * amplitude ** 2 * Math.sqrt(energy / 50));
  relative(p.dosOneSpin / impurity.dosOneSpin, Math.sqrt(energy / 50));
  relative(p.meanFreePath, 1 / (1e25 * density * p.crossSection));
  relative(p.tau * p.rate, 1); relative(p.speed, Math.sqrt(2 * p.kinetic / m.MASS));
  relative(p.kMeanFreePath, p.k * p.speed * p.tau);
}

// Bose detailed balance and exact parabolic energy/momentum conservation.
for (const q of [.05, .1, .5, 1, 2]) for (const temperature of [5, 10, 100, 300, 600]) for (const energy of [.1, 1, 10, 50, 100]) {
  const p = m.phononMode(q, temperature, energy), kinetic = energy * 1e-3 * m.E;
  relative(p.emissionFactor / p.population, Math.exp(p.z));
  close(p.emissionFactor - p.population, 1, 1e-12);
  relative(p.phononEnergy, m.HBAR * m.VS * q * 1e9);
  check(p.population > 0 && p.population < p.equipartitionPopulation);
  relative(p.energyRatio, p.phononEnergy / kinetic);
  const absorptionK = m.HBAR ** 2 / (2 * m.MASS) * (p.k ** 2 + p.q ** 2 + 2 * p.k * p.q * p.absorptionCosine);
  const emissionK = m.HBAR ** 2 / (2 * m.MASS) * (p.k ** 2 + p.q ** 2 - 2 * p.k * p.q * p.emissionCosine);
  relative(absorptionK, kinetic + p.phononEnergy, 1e-12);
  close((emissionK - kinetic + p.phononEnergy) / kinetic, 0, 1e-12);
  assert.equal(p.absorptionAllowed, Math.abs(p.absorptionCosine) <= 1 + 1e-12);
  assert.equal(p.emissionAllowed, kinetic >= p.phononEnergy && Math.abs(p.emissionCosine) <= 1 + 1e-12); physics += 2;
  if (p.absorptionAllowed) {
    const finalK = m.HBAR ** 2 / (2 * m.MASS) * ((p.k + p.q * p.absorptionCosine) ** 2 + p.q ** 2 * (1 - p.absorptionCosine ** 2));
    relative(finalK, kinetic + p.phononEnergy, 1e-12);
  }
}
const contrasting = m.phononMode(.05, 300, .1);
check(contrasting.z < .01 && contrasting.energyRatio > 1 && contrasting.absorptionAllowed && !contrasting.emissionAllowed);
const occupied = m.phononMode(1, 300, 50);
check(occupied.z < .13 && occupied.energyRatio < .07 && occupied.absorptionAllowed && occupied.emissionAllowed);
const momentumBlocked = m.phononMode(2, 300, 10);
check(momentumBlocked.emissionEnergyAllowed && !momentumBlocked.emissionAllowed);

// Independent gamma values and numerical integrals after y=t² remove the
// square-root endpoint. r=-2 has a finite conductivity integral but not a
// particle integral; no divergent function is sampled at zero or disguised.
close(m.gamma(.5), Math.sqrt(Math.PI), 1e-14); close(m.gamma(1), 1, 1e-14);
for (let z = .5; z <= 4; z += .5) relative(m.gamma(z + 1), z * m.gamma(z));
relative(m.gamma(6), 120);
for (const transport of [false, true]) {
  const norm = simpson(t => 2 * t * m.mbWeight(t*t, transport), 0, 9);
  close(norm, 1, 1e-11);
  const mean = simpson(t => 2 * t ** 3 * m.mbWeight(t*t, transport), 0, 9);
  close(mean, transport ? 2.5 : 1.5, 1e-10);
}
for (const temperature of [50, 100, 300, 600]) for (const r of [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2]) {
  const p = m.thermalAverage(temperature, r), b = m.KB * temperature / m.KREF;
  // Under y=t² the transport integrand is 2*t^(4+2r)*exp(-t²).
  const transport = b ** r * simpson(t => 2 * t ** (4 + 2 * r) * Math.exp(-t*t), 0, 9) / (3 * Math.sqrt(Math.PI) / 4);
  relative(p.transportRatio, transport, 1e-10);
  relative(p.tauEffective, m.TAUREF * p.transportRatio);
  relative(p.mobility, m.E * p.tauEffective / m.MASS);
  if (r <= -1.5) { assert.equal(p.particleRatio, null); assert.equal(p.tauParticle, null); assert.equal(p.particleToTransport, null); physics += 3; }
  else {
    const particle = b ** r * simpson(t => 2 * t ** (2 + 2 * r) * Math.exp(-t*t), 0, 9) / (Math.sqrt(Math.PI) / 2);
    relative(p.particleRatio, particle, 1e-10);
    relative(p.particleToTransport, 1.5 / (r + 1.5));
  }
}
relative(m.thermalAverage(300, 0).transportRatio, 1);
relative(m.thermalAverage(300, 0).particleRatio, 1);
const avg = m.thermalAverage(300, -.5);
close(avg.tauEffective * 1e12, .701836862, 2e-9); close(avg.tauParticle * 1e12, 1.052755292, 2e-9);
close(avg.mobility, .617202357, 2e-9); close(avg.particleToTransport, 1.5);
check(m.thermalAverage(300, -1.49).particleRatio > 100);
for (const fn of [() => m.angularChannel(-1), () => m.angularChannel(181), () => m.angularChannel(90, 0), () => m.angularChannel('60'), () => m.shortRange(0), () => m.shortRange(50, 0), () => m.shortRange(50, 1, NaN), () => m.phononMode(0, 300, 50), () => m.phononMode(1, 0, 50), () => m.phononMode(1, 300, 0), () => m.gamma(0), () => m.gamma(NaN), () => m.mbWeight(-1), () => m.mbWeight(1, 1), () => m.thermalAverage(0, 0), () => m.thermalAverage(300, -2.5), () => m.thermalAverage(300, Infinity)]) throws(fn);

// Lightweight API adapter checks model/UI lifecycle, never browser geometry.
class Element {
  constructor(tag = 'div', attrs = {}, children = []) { this.tagName = tag.toUpperCase(); this.attrs = attrs; this.children = []; this.listeners = {}; this._value = ''; this.checked = false; this.disabled = false; this._text = ''; this.classList = { add() {} }; this.append(children); }
  set value(value) { this._value = String(value); }
  get value() { return this._value; }
  append(...values) { this.children.push(...values.flat(Infinity).filter(x => x !== null && x !== undefined)); }
  replaceChildren(...values) { this.children = []; this._text = ''; this.append(values); }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  click() { this.listeners.click?.(); }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(x => x instanceof Element ? x.textContent : String(x)).join(' '); }
}
const factories = new Map();
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-drude-microscopic.js'), 'utf8'), { window: { ArticleLabs: { register: (id, fn) => { assert.ok(!factories.has(id)); factories.set(id, fn); } } } });
assert.deepEqual([...factories.keys()], ['drude-angle', 'drude-impurity', 'drude-phonon', 'drude-average']); lifecycle++;
const registry = fs.readFileSync(require.resolve('../_data/article_labs_drude_microscopic.yml'), 'utf8');
assert.deepEqual(registry.match(/^drude-[a-z]+(?=:)/gm), [...factories.keys()]); lifecycle++;
assert.equal((registry.match(/bundle: drude-microscopic/g) || []).length, 4); lifecycle++;
function mount(id, ko = false) {
  const fields = [], defaults = [], metrics = [], buttons = [], fixedAxes = [];
  let state, render, plotIndex = 0;
  const plotCount = id === 'drude-angle' ? 1 : 2;
  const ctx = { root: new Element(), ui: new Element(), view: new Element(), metrics: new Element(), status: new Element(), ko,
    text: (en, korean) => ko ? korean : en, format: (v, digits = 4) => String(Number(v.toFixed(digits))),
    el: (tag, attrs, children) => new Element(tag, attrs, children), svg: (tag, attrs, text) => new Element(tag, attrs, text),
    metric: () => { const output = new Element('output'); metrics.push(output); return output; },
    control: (_en, _ko, config) => { const field = new Element(config.type === 'select' ? 'select' : 'input'); field.config = config; field.value = config.value; fields.push(field); defaults.push(field.value); return field; },
    button: (en, korean, fn) => { const button = new Element('button', {}, ko ? korean : en); button.addEventListener('click', fn); buttons.push(button); return button; },
    plot: spec => {
      assert.ok(spec.x[0] < spec.x[1] && spec.y[0] < spec.y[1]);
      // Axis domains, tick spacing and units cannot track changing controls.
      // This assertion runs for every render, shared-state restore and reset.
      const slot = plotIndex++ % plotCount;
      const axes = JSON.stringify([spec.x, spec.y, spec.xTicks, spec.yTicks, spec.xLabel, spec.yLabel]);
      if (fixedAxes[slot] === undefined) fixedAxes[slot] = axes;
      else assert.equal(axes, fixedAxes[slot], `${id}: plot ${slot} axes changed with an input`);
      lifecycle++;
      function point(x, y) { assert.ok(Number.isFinite(x) && Number.isFinite(y)); assert.ok(x >= spec.x[0] - 1e-9 && x <= spec.x[1] + 1e-9); assert.ok(y >= spec.y[0] - 1e-9 && y <= spec.y[1] + 1e-9, `${id}: y=${y} outside ${spec.y}`); plotted++; }
      (spec.series || []).forEach(s => s.points.forEach(([x, y]) => point(x, y))); (spec.bars || []).forEach(b => { point(b.x, b.y); check(b.x - b.width/2 >= spec.x[0] && b.x + b.width/2 <= spec.x[1]); }); (spec.markers || []).forEach(p => point(p.x, p.y));
      return new Element('div', { spec }, spec.title);
    },
    onChange: fn => { render = fn; fn(); }, stateLink: (key, provider) => { assert.equal(key, id); state = provider; }
  };
  factories.get(id)(ctx);
  return { fields, metrics, buttons, state, ctx, render, reset: () => { fields.forEach((f, i) => { f.value = defaults[i]; }); render(); },
    snapshot: () => JSON.stringify({ state: state.read(), view: ctx.view, metrics: metrics.map(x => x.textContent), status: ctx.status.textContent }, (key, value) => key === 'listeners' ? undefined : value) };
}
function roundtrip(lab, id, ko) { const restored = mount(id, ko); assert.equal(restored.state.write(JSON.parse(JSON.stringify(lab.state.read()))), true); assert.equal(restored.snapshot(), lab.snapshot()); restored.render(); assert.equal(restored.snapshot(), lab.snapshot()); lifecycle += 3; }
const corners = {
  'drude-angle': [{ angle: 0, clock: .1 }, { angle: 0, clock: 5 }, { angle: 1, clock: .1 }, { angle: 90, clock: 1 }, { angle: 180, clock: 5 }],
  'drude-impurity': [], 'drude-phonon': [], 'drude-average': []
};
for (const energy of [10, 50, 150]) for (const density of [.2, 1, 2]) for (const amplitude of [.5, 1, 1.5]) corners['drude-impurity'].push({ energy, density, amplitude });
for (const q of [.05, 1, 2]) for (const temperature of [5, 300, 600]) for (const energy of [.1, 10, 50, 100]) corners['drude-phonon'].push({ q, temperature, energy });
for (const temperature of [50, 300, 600]) for (const exponent of [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2]) corners['drude-average'].push({ temperature, exponent });
function plots(lab) { return lab.ctx.view.children.filter(node => node.attrs?.spec).map(node => node.attrs.spec); }
function graphicSignature(lab) {
  return JSON.stringify(plots(lab).map(p => ({ series: p.series?.map(s => s.points), markers: p.markers, bars: p.bars })));
}
const fixedDomains = {
  'drude-angle': [[[0, 180], [0, 10.5]]],
  'drude-impurity': [[[10, 150], [0, 12.5]], [[10, 150], [1.5, 4]]],
  'drude-phonon': [[[5, 600], [-7, 3]], [[.05, 2], [-7, 108]]],
  'drude-average': [[[0, 400], [0, .12]], [[0, 400], [0, 1]]]
};
for (const id of factories.keys()) for (const ko of [false, true]) {
  const lab = mount(id, ko), initial = lab.snapshot(), saved = JSON.parse(JSON.stringify(lab.state.read()));
  assert.equal(JSON.stringify(plots(lab).map(p => [p.x, p.y])), JSON.stringify(fixedDomains[id])); lifecycle++;
  for (const bad of [null, false, [], {}, { wrong: 1 }, { ...saved, extra: 1 }]) { assert.equal(lab.state.write(bad), false); assert.equal(lab.snapshot(), initial); lifecycle += 2; }
  for (const key of Object.keys(saved)) {
    const missing = { ...saved }; delete missing[key];
    for (const bad of ['0', null, NaN, Infinity, {}, []]) { assert.equal(lab.state.write({ ...saved, [key]: bad }), false); assert.equal(lab.snapshot(), initial); lifecycle += 2; }
    assert.equal(lab.state.write(missing), false); assert.equal(lab.snapshot(), initial); lifecycle += 2;
  }
  roundtrip(lab, id, ko);
  for (const field of lab.fields) {
    const c = field.config;
    const values = c.type === 'select' ? c.options.map(o => o.value) : [c.min, c.min + c.step, c.max];
    const defaultGraphic = graphicSignature(lab);
    for (const value of values) {
      field.value = value; lab.render(); assert.ok(lab.ctx.view.children.length);
      if (String(value) !== String(c.value)) assert.notEqual(graphicSignature(lab), defaultGraphic, `${id}: a control changed only the numbers`);
      lifecycle++;
    }
    lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++;
  }
  for (const button of lab.buttons) { button.click(); roundtrip(lab, id, ko); lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++; }
  for (const state of corners[id]) {
    assert.equal(lab.state.write(state), true); roundtrip(lab, id, ko); lifecycle++;
    if (id === 'drude-phonon') {
      const p = m.phononMode(state.q, state.temperature, state.energy);
      for (const [index, permitted] of [[5, p.absorptionAllowed], [6, p.emissionAllowed]]) {
        const metric = lab.metrics[index];
        assert.equal(metric.textContent, ko ? (permitted ? '가능' : '불가') : (permitted ? 'Allowed' : 'Forbidden'));
        assert.equal(metric.ariaLabel, metric.title);
        assert.ok(metric.ariaLabel.includes(ko ? 'Energy·momentum 보존' : 'Kinematically'));
        assert.ok(metric.ariaLabel.includes(ko ? '선택한 q' : 'selected q'));
      }
      const energyPlot = lab.ctx.view.children[1].attrs.spec;
      assert.ok(energyPlot.markers.every(marker => /Kinematically|Energy·momentum 보존/.test(marker.label)), 'Full kinematic labels must remain on plot markers');
    }
  }
  if (id === 'drude-angle') {
    lab.state.write({ angle: 0, clock: 1 }); assert.equal(lab.metrics[3].textContent, '∞');
    assert.equal(lab.metrics[3].ariaLabel, lab.metrics[3].title);
    assert.ok(lab.metrics[3].ariaLabel.includes('θ=0'));
    assert.ok(/Infinite|무한대/.test(lab.metrics[3].ariaLabel));
    lab.state.write({ angle: 180, clock: 1 }); assert.equal(lab.metrics[3].textContent, '0.5');
    assert.ok(lab.metrics[3].ariaLabel.includes('0.5 ps'));
    assert.ok(!lab.metrics[3].ariaLabel.includes('θ=0')); lifecycle += 2;
  }
  if (id === 'drude-average') {
    lab.state.write({ temperature: 300, exponent: -2 }); assert.equal(lab.ctx.view.children[1].attrs.spec.series.length, 1);
    assert.equal(lab.metrics[1].textContent, ko ? '발산' : 'Diverges'); assert.equal(lab.metrics[3].textContent, '∞');
    assert.equal(lab.metrics[1].ariaLabel, lab.metrics[1].title); assert.equal(lab.metrics[3].ariaLabel, lab.metrics[3].title);
    assert.ok(lab.metrics[1].ariaLabel.includes('K=0')); assert.ok(/positive infinity|양의 무한대/.test(lab.metrics[3].ariaLabel));
    lab.state.write({ temperature: 300, exponent: -.5 }); assert.equal(lab.metrics[3].textContent, '1.5');
    assert.ok(lab.metrics[1].ariaLabel.includes('ps')); assert.ok(lab.metrics[3].ariaLabel.includes('1.5'));
    assert.ok(!lab.metrics[1].ariaLabel.includes('K=0')); lifecycle += 2;
  }
  const invalid = { 'drude-angle': [{ angle: .5 }, { angle: 181 }, { clock: .15 }, { clock: 0 }], 'drude-impurity': [{ energy: 51 }, { density: .25 }, { amplitude: .55 }, { energy: 155 }], 'drude-phonon': [{ q: .075 }, { temperature: 301 }, { energy: .15 }, { q: 2.05 }], 'drude-average': [{ temperature: 55 }, { exponent: -2.5 }, { exponent: -.25 }] };
  const current = JSON.parse(JSON.stringify(lab.state.read())), snapshot = lab.snapshot();
  for (const patch of invalid[id]) { assert.equal(lab.state.write({ ...current, ...patch }), false); assert.equal(lab.snapshot(), snapshot); lifecycle += 2; }
  lab.reset(); assert.equal(lab.snapshot(), initial); lifecycle++;
}

// Probe every selectable value of every field (others at their defaults),
// in addition to full corner products above. Fixed-domain assertions run in
// the adapter and every physical point is required to remain inside them.
for (const id of factories.keys()) {
  const lab = mount(id);
  for (const field of lab.fields) {
    const c = field.config;
    const values = c.type === 'select' ? c.options.map(o => o.value) : Array.from({ length: Math.round((c.max - c.min) / c.step) + 1 }, (_, i) => Number((c.min + i * c.step).toFixed(9)));
    for (const value of values) { field.value = value; lab.render(); }
    lab.reset();
  }
}

// Absolute plotted values, not just moved markers or changed text.
{
  const lab = mount('drude-angle');
  for (const clock of [.1, 1, 5]) {
    lab.state.write({ angle: 90, clock }); const [chart] = plots(lab);
    close(chart.series[0].points[90][1], clock);
    close(chart.series[0].points[180][1], 2 * clock);
    close(chart.markers[0].y, clock);
    close(chart.series[1].points[0][1], clock);
  }
}
{
  const lab = mount('drude-impurity');
  for (const density of [.2, 1, 2]) for (const amplitude of [.5, 1, 1.5]) {
    lab.state.write({ energy: 50, density, amplitude }); const [rate, path] = plots(lab);
    const expected = impurity.rate * density * amplitude ** 2 * 1e-12;
    relative(rate.series[0].points[40][1], expected);
    const expectedPath = Math.log10(1e9 / (1e25 * density * 4 * Math.PI * (.2e-9 * amplitude) ** 2));
    for (const [, y] of path.series[1].points) close(y, expectedPath);
    relative(rate.markers[0].y, expected); close(path.markers[0].y, expectedPath);
    relative(rate.series[1].points[40][1], impurity.rate * 1e-12);
  }
}
{
  const lab = mount('drude-phonon');
  for (const q of [.05, 1, 2]) for (const temperature of [5, 300, 600]) for (const energy of [.1, 10, 50, 100]) {
    lab.state.write({ q, temperature, energy }); const [population, final] = plots(lab), p = m.phononMode(q, temperature, energy);
    close(population.markers[0].y, Math.log10(p.population));
    close(population.markers[1].y, Math.log10(p.emissionFactor));
    close(final.markers[0].x, q); close(final.markers[0].y, energy + p.phononEnergy / m.E * 1e3);
    close(final.markers[1].y, energy - p.phononEnergy / m.E * 1e3);
    assert.equal(final.markers[0].color, p.absorptionAllowed ? 'blue' : 'red');
    assert.equal(final.markers[1].color, p.emissionAllowed ? 'gold' : 'red'); lifecycle += 2;
    const endEnergy = m.HBAR * m.VS * 2e9 / m.E * 1e3;
    close(final.series[0].points.at(-1)[1], energy + endEnergy);
    close(final.series[1].points.at(-1)[1], energy - endEnergy);
    for (const [color, window] of Object.entries(m.phononWindows(energy))) {
      const series = final.series.find(s => s.color === (color === 'absorption' ? 'blue' : 'gold'));
      const start = Math.max(.05, window?.[0] ?? Infinity), end = Math.min(2, window?.[1] ?? -Infinity);
      assert.equal(Boolean(series), end > start); lifecycle++;
      if (series) {
        close(series.points[0][0], start);
        close(series.points[1][0], end);
      }
    }
  }
  const threshold = m.phononMode(.2, 300, 2);
  check(threshold.population > 30 && threshold.absorptionAllowed && threshold.emissionEnergyAllowed && !threshold.emissionAllowed);
  for (const energy of [.1, .4, 2, 10, 20, 50, 100]) for (let i = 1; i <= 40; i++) {
    const q = i / 20, mode = m.phononMode(q, 300, energy), window = m.phononWindows(energy);
    for (const [key, allowed] of [['absorption', mode.absorptionAllowed], ['emission', mode.emissionAllowed]]) {
      assert.equal(Boolean(window[key] && q >= window[key][0] && q <= window[key][1]), allowed);
      physics++;
    }
  }
}
{
  const lab = mount('drude-average');
  for (const temperature of [50, 300, 600]) for (const exponent of [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2]) {
    lab.state.write({ temperature, exponent }); const [weights, fractions] = plots(lab);
    const thermalMeV = m.KB * temperature / m.E * 1e3;
    for (let index = 0; index < 2; index++) {
      const shape = index ? 1.5 : .5, norm = index ? 3 * Math.sqrt(Math.PI) / 4 : Math.sqrt(Math.PI) / 2;
      const energy = weights.series[index].points[200][0], y = energy / thermalMeV;
      relative(weights.series[index].points[200][1], y ** shape * Math.exp(-y) / (norm * thermalMeV));
      close(weights.markers[index].x, (index ? 2.5 : 1.5) * thermalMeV);
    }
    const p = m.thermalAverage(temperature, exponent);
    assert.equal(fractions.series.length, p.tauParticle === null ? 1 : 2);
    fractions.series.forEach((series, index) => {
      const shape = exponent + (index === 0 ? 2.5 : 1.5);
      close(series.points[0][1], 0);
      close(fractions.markers[index].x, thermalMeV);
      close(fractions.markers[index].y, m.averageFraction(1, shape));
      close(series.points.at(-1)[1], m.averageFraction(400 / thermalMeV, shape));
      assert.ok(series.points.every((point, i) => !i || point[1] + 1e-13 >= series.points[i - 1][1]));
    });
  }
  // Independent exact gamma integrals for integer shape, not the lab's
  // positive-series implementation. Endpoint normalization uses y=∞.
  for (const y of [.01, .5, 1, 5, 20, 60, 70]) {
    close(m.averageFraction(y, 1), -Math.expm1(-y));
    close(m.averageFraction(y, 2), 1 - (1 + y) * Math.exp(-y));
    close(m.averageFraction(y, 3), 1 - (1 + y + y ** 2 / 2) * Math.exp(-y));
  }
  for (const y of [.01, .5, 1, 5, 20]) for (const shape of [.5, 1.5, 2.5, 3.5]) {
    close(m.averageFraction(y, shape + 1), m.averageFraction(y, shape) - y ** shape * Math.exp(-y) / m.gamma(shape + 1));
  }
}
console.log(`Drude microscopic: ${physics} physics assertions; ${lifecycle} bilingual UI/state/reset assertions; ${plotted} finite, in-domain plotted points. This adapter does not test browser geometry.`);
