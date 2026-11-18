#!/usr/bin/env node
'use strict';

// A deterministic DOM contract test, not a substitute for responsive browser QA.
// It executes the production renderer and the actual NEGF core in both languages.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const core = require('../assets/js/negf-core.js');
const source = fs.readFileSync(path.join(__dirname, '../assets/js/negf-interactive.js'), 'utf8');
const include = fs.readFileSync(path.join(__dirname, '../_includes/interactive/negf-barrier.html'), 'utf8');

class Element {
  constructor(tag, attrs = {}) {
    this.tagName = tag;
    this.attributes = { ...attrs };
    this.children = [];
    this.listeners = new Map();
    this.dataset = {};
    this.value = attrs.value || '';
    this.min = attrs.min || '';
    this.max = attrs.max || '';
    this.disabled = true;
    this.hidden = false;
    this.textContent = '';
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(listener);
  }
  dispatch(type, event = {}) {
    for (const listener of this.listeners.get(type) || []) listener(event);
  }
  matches(selector) {
    if (selector.startsWith('.')) return (this.attributes.class || this.className || '').split(/\s+/).includes(selector.slice(1));
    if (selector.startsWith('[')) return Object.hasOwn(this.attributes, selector.slice(1, -1));
    return this.tagName === selector;
  }
  querySelectorAll(selector) {
    const alternatives = selector.split(',').map(value => value.trim());
    return this.children.flatMap(child => [
      ...(alternatives.some(value => child.matches(value)) ? [child] : []),
      ...child.querySelectorAll(selector)
    ]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

function harness(lang, withCore = true) {
  const root = new Element('section', { 'data-negf-lab': '' });
  root.dataset.lang = lang;
  const names = new Map();
  function add(name, tag = 'span', attrs = {}) {
    assert.ok(include.includes(`data-negf-${name}`), `include is missing ${name}`);
    const node = new Element(tag, { [`data-negf-${name}`]: '', ...attrs });
    names.set(name, node);
    root.append(node);
    return node;
  }
  add('height', 'input', { min: '0', max: '0.65', value: '0.4' });
  add('width', 'input', { min: '6', max: '30', value: '16' });
  add('energy', 'input', { min: '0.001', max: '0.6', value: '0.1413' });
  for (const name of ['reset', 'scenario-high', 'scenario-wide']) add(name, 'button');
  for (const name of ['status', 'resonances', 'height-value', 'width-value', 'energy-value',
    'probe-energy', 'probe-transmission', 'first-resonance', 'first-width', 'width-note']) add(name);
  for (const name of ['potential', 'transmission', 'linewidth']) {
    const svg = add(name, 'svg');
    svg.append(new Element('g', { 'data-negf-chart': '' }));
    svg.createSVGPoint = () => ({ x: 0, y: 0, matrixTransform() { return this; } });
    svg.getScreenCTM = () => ({ inverse: () => ({}) });
  }
  const frames = new Map();
  const errors = [];
  let nextFrame = 0;
  const document = {
    createElementNS: (_namespace, tag) => new Element(tag),
    createElement: tag => new Element(tag),
    querySelectorAll: selector => selector === '[data-negf-lab]' ? [root] : []
  };
  const window = {
    NEGFCore: withCore ? core : null,
    location: { hash: '' },
    requestAnimationFrame(fn) { frames.set(++nextFrame, fn); return nextFrame; },
    cancelAnimationFrame(id) { frames.delete(id); }
  };
  vm.runInNewContext(source, { window, document, console: { error: (...args) => errors.push(args) } });
  function flush() {
    while (frames.size) {
      const pending = Array.from(frames.values());
      frames.clear();
      pending.forEach(fn => fn());
    }
    assert.deepEqual(errors, [], 'renderer reported an error');
    if (withCore) assert.equal(root.dataset.ready, 'true');
  }
  flush();
  return {
    root, names, flush,
    input(name, value) { names.get(name).value = String(value); names.get(name).dispatch('input'); flush(); },
    click(name) { names.get(name).dispatch('click'); flush(); }
  };
}

function one(root, selector) {
  const result = root.querySelector(selector);
  assert.ok(result, `missing rendered ${selector}`);
  return result;
}
function graph(h) {
  const attr = (selector, name) => one(h.root, selector).getAttribute(name);
  return {
    potential: attr('.negf-potential-line', 'd'),
    referencePotential: attr('.negf-reference-potential', 'd'),
    transmission: attr('.negf-transmission-line', 'd'),
    referenceTransmission: attr('.negf-reference-transmission', 'd'),
    linewidth: attr('.negf-linewidth-current', 'd'),
    referenceLinewidth: attr('.negf-linewidth-reference', 'd'),
    probeX: attr('.negf-probe-dot', 'cx'),
    probeY: attr('.negf-probe-dot', 'cy'),
    columnHeight: attr('.negf-probe-column', 'height'),
    potentialProbeY: one(h.names.get('potential'), '.negf-probe-line').getAttribute('y1')
  };
}
function assertReferenceUnchanged(before, after) {
  for (const key of ['referencePotential', 'referenceTransmission', 'referenceLinewidth']) assert.equal(after[key], before[key], `${key} must retain a fixed reference`);
}
function halfMaximumSeparation(h) {
  const positions = h.root.querySelectorAll('.negf-half-crossing').map(node => Number(node.getAttribute('cx')));
  assert.equal(positions.length, 2);
  return Math.abs(positions[1] - positions[0]);
}

let checks = 0;
function check(name, fn) { fn(); checks += 1; console.log('PASS ' + name); }

for (const lang of ['en', 'ko']) {
  check(`${lang}: all original and scenario controls enable; default plots agree with physics`, () => {
    const h = harness(lang);
    for (const name of ['height', 'width', 'energy', 'reset', 'scenario-high', 'scenario-wide']) assert.equal(h.names.get(name).disabled, false);
    const g = graph(h);
    assert.equal(g.potential, g.referencePotential);
    assert.equal(g.transmission, g.referenceTransmission);
    assert.ok(Math.abs(Number(h.names.get('probe-transmission').textContent) - core.transmission(0.1413, core.profile())) < 0.000006);
    assert.ok(halfMaximumSeparation(h) > 0);
  });
  check(`${lang}: barrier height changes plotted geometry, spectrum and absolute resonance width`, () => {
    const h = harness(lang), before = graph(h), firstWidth = halfMaximumSeparation(h);
    h.input('height', 0.6);
    const after = graph(h);
    for (const key of ['potential', 'transmission', 'linewidth']) assert.notEqual(after[key], before[key], key);
    assert.ok(halfMaximumSeparation(h) < firstWidth, 'higher barriers must visibly narrow this first resonance');
    assertReferenceUnchanged(before, after);
  });
  check(`${lang}: well width moves peak energies and changes the potential and width comparison`, () => {
    const h = harness(lang), before = graph(h);
    const firstEnergy = parseFloat(h.names.get('first-resonance').textContent);
    h.input('width', 24);
    const after = graph(h);
    for (const key of ['potential', 'transmission', 'linewidth']) assert.notEqual(after[key], before[key], key);
    assert.ok(parseFloat(h.names.get('first-resonance').textContent) < firstEnergy);
    assertReferenceUnchanged(before, after);
  });
  check(`${lang}: incident energy changes linked graphical selection but not the Hamiltonian/spectrum`, () => {
    const h = harness(lang), before = graph(h);
    h.input('energy', 0.05);
    const after = graph(h);
    for (const key of ['probeX', 'probeY', 'columnHeight', 'potentialProbeY']) assert.notEqual(after[key], before[key], key);
    for (const key of ['potential', 'transmission', 'linewidth']) assert.equal(after[key], before[key], key);
    for (const energy of [0.001, 0.6]) {
      h.input('energy', energy);
      const g = graph(h);
      assert.ok(Number(g.probeX) >= 68 && Number(g.probeX) <= 620);
      assert.ok(Number(g.columnHeight) >= 0 && Number(g.columnHeight) <= 140 + 1e-8);
    }
  });
  check(`${lang}: exact peak buttons select unit transmission and expose selection to assistive technology`, () => {
    const h = harness(lang);
    const buttons = h.names.get('resonances').children;
    assert.equal(buttons.length, 6);
    buttons[0].dispatch('click');
    assert.equal(buttons[0].getAttribute('aria-pressed'), 'true');
    assert.equal(buttons[1].getAttribute('aria-pressed'), 'false');
    assert.ok(Math.abs(Number(h.names.get('probe-transmission').textContent) - 1) < 1e-9);
    assert.ok(Math.abs(Number(graph(h).probeY) - 40) < 1e-8);
  });
  check(`${lang}: no-barrier case, scenario buttons and Reset restore all graphical state`, () => {
    const h = harness(lang), before = graph(h);
    h.input('height', 0);
    assert.equal(h.names.get('resonances').hidden, true);
    assert.equal(h.root.querySelectorAll('.negf-half-crossing').length, 0);
    assert.equal(one(h.root, '.negf-linewidth-current').getAttribute('d'), 'M68,40H620');
    assert.equal(h.names.get('probe-transmission').textContent, '1.00000');
    assert.ok(h.names.get('width-note').textContent.includes(lang === 'ko' ? '따로 구분되는' : 'no distinct'));
    h.click('scenario-high');
    assert.equal(h.names.get('height').value, '0.6');
    assert.equal(h.names.get('width').value, '16');
    assert.notEqual(graph(h).linewidth, before.linewidth);
    h.click('scenario-wide');
    assert.equal(h.names.get('height').value, '0.4');
    assert.equal(h.names.get('width').value, '24');
    h.click('reset');
    assert.deepEqual(graph(h), before);
  });
  check(`${lang}: plot selection and live-preview restoration preserve exact peak energy`, () => {
    const h = harness(lang);
    const peak = core.spectrum(core.profile()).peaks[0];
    h.names.get('transmission').dispatch('click', { clientX: 68 + 552 * (peak.energyEV - 0.001) / 0.599, clientY: 80 });
    assert.equal(h.names.get('probe-transmission').textContent, '1.00000');
    const capture = { detail: {} };
    h.root.dispatch('article-preview:capture', capture);
    assert.ok(Math.abs(capture.detail.state.energy - peak.energyEV) < 1e-14);
    const selected = graph(h);
    h.click('scenario-wide');
    const restore = { detail: { state: capture.detail.state } };
    h.root.dispatch('article-preview:restore', restore);
    h.flush();
    assert.equal(restore.detail.restored, true);
    assert.deepEqual(graph(h), selected);
  });
}

check('missing physics bundle fails visibly and keeps all controls disabled', () => {
  const h = harness('en', false);
  assert.ok(h.names.get('status').textContent.includes('could not load'));
  for (const input of h.root.querySelectorAll('input, button')) assert.equal(input.disabled, true);
});

console.log(`${checks} NEGF interactive contract checks passed.`);
