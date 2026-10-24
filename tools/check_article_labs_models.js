'use strict';
/* Contract smoke test only: no browser/layout claims, model calls, or network access. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const base = path.resolve(__dirname, '..');
const { parseCSV, format } = require('../assets/js/article-labs.js');
class StubEvent {
  constructor(type, options = {}) { this.type = type; this.bubbles = Boolean(options.bubbles); this.defaultPrevented = false; Object.assign(this, options); }
  preventDefault() { this.defaultPrevented = true; }
}
class StubNode {
  constructor(tag = 'div', attrs = {}, children = []) {
    this.tagName = tag.toUpperCase(); this.attributes = {}; this.dataset = {}; this.listeners = {}; this.children = [];
    this.parentNode = null; this._text = ''; this._value = ''; this.checked = false; this.disabled = false;
    this.classList = {
      add: (...names) => this.setAttribute('class', [...new Set((this.className || '').split(/\s+/).filter(Boolean).concat(names))].join(' ')),
      contains: name => (this.className || '').split(/\s+/).includes(name),
      remove: (...names) => this.setAttribute('class', (this.className || '').split(/\s+/).filter(name => !names.includes(name)).join(' '))
    };
    for (const [name, value] of Object.entries(attrs || {})) if (value !== undefined && value !== null && value !== false) this.setAttribute(name, value);
    this.append(...[children].flat(Infinity));
  }
  append(...nodes) {
    for (const node of nodes.flat(Infinity)) {
      if (node === undefined || node === null || node === false) continue;
      if (node instanceof StubNode) {
        if (node.parentNode) node.parentNode.children = node.parentNode.children.filter(child => child !== node);
        node.parentNode = this;
      }
      this.children.push(node instanceof StubNode ? node : String(node));
    }
  }
  replaceChildren(...nodes) { this.children.forEach(child => { if (child instanceof StubNode) child.parentNode = null; }); this.children = []; this._text = ''; this.append(...nodes); }
  get textContent() { return this._text + this.children.map(child => child instanceof StubNode ? child.textContent : child).join(''); }
  set textContent(text) { this.replaceChildren(); this._text = String(text); }
  get options() { return this.tagName === 'SELECT' ? this.children.filter(child => child instanceof StubNode && child.tagName === 'OPTION') : undefined; }
  get value() { return this.tagName === 'SELECT' ? this.options.some(option => option.value === this._value) ? this._value : '' : this._value; }
  set value(value) { this._value = String(value); }
  get validity() {
    if (this.tagName !== 'INPUT' || this.type !== 'number') return { valid: true };
    // Native optional number fields allow an empty value; the runtime separately
    // rejects blank drafts before rendering. Bounds and integer step still apply.
    if (this.value.trim() === '') return { valid: !this.required };
    const value = Number(this.value);
    const inBounds = Number.isFinite(value) && (this.min === undefined || value >= Number(this.min)) &&
      (this.max === undefined || value <= Number(this.max));
    const step = this.step === undefined ? 1 : this.step;
    const offset = (value - Number(this.min || 0)) / Number(step);
    return { valid: inBounds && (step === 'any' || Math.abs(offset - Math.round(offset)) < 1e-7) };
  }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  dispatchEvent(event) {
    if (!event.target) event.target = this;
    event.currentTarget = this;
    for (const fn of this.listeners[event.type] || []) fn(event);
    if (typeof this['on' + event.type] === 'function') this['on' + event.type](event);
    if (event.bubbles && this.parentNode) this.parentNode.dispatchEvent(event);
    return !event.defaultPrevented;
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === 'class') this.className = String(value);
    else if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_match, char) => char.toUpperCase())] = String(value);
    else this[name] = typeof value === 'boolean' ? value : String(value);
  }
  getAttribute(name) {
    if (name.startsWith('data-')) return this.dataset[name.slice(5).replace(/-([a-z])/g, (_match, char) => char.toUpperCase())] ?? null;
    return this.attributes[name] ?? null;
  }
  matches(selector) {
    const tag = /^[a-z][\w-]*/i.exec(selector)?.[0];
    if (tag && this.tagName !== tag.toUpperCase()) return false;
    for (const [, name] of selector.matchAll(/\.([\w-]+)/g)) if (!this.classList.contains(name)) return false;
    for (const [, name, value] of selector.matchAll(/\[([^=\]]+)(?:="([^"]*)")?\]/g)) if (this.getAttribute(name) === null || value !== undefined && this.getAttribute(name) !== value) return false;
    return true;
  }
  querySelectorAll(selector) {
    const direct = /^:scope\s*>\s*(.+)$/.exec(selector);
    if (direct) return this.children.filter(child => child instanceof StubNode && child.matches(direct[1]));
    return this.children.flatMap(child => child instanceof StubNode ? [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)] : []);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  get parentElement() { return this.parentNode; }
  closest(selector) { for (let node = this; node; node = node.parentElement) if (node.matches(selector)) return node; return null; }
  focus() { this.focused = true; }
}
function walk(dir) { return fs.readdirSync(dir, { withFileTypes: true }).flatMap(p => p.isDirectory() ? walk(path.join(dir, p.name)) : [path.join(dir, p.name)]); }
function localAsset(asset) {
  const file = path.resolve(base, asset.replace(/^\//, ''));
  assert(file.startsWith(base + path.sep), 'Asset leaves repository');
  return fs.readFileSync(file, 'utf8');
}
async function main() {
  const factories = new Map(), presets = new Map();
  const registry = { register(id, fn) { assert(!factories.has(id), 'Duplicate factory ' + id); factories.set(id, fn); } };
  const sandbox = { window: { ArticleLabs: registry }, ArticleLabs: registry, console, Math, Number, Array, String, Map, Set, Infinity, Event: StubEvent };
  const context = vm.createContext(sandbox);
  for (const file of fs.readdirSync(path.join(base, 'assets/js')).filter(f => /^article-labs-.+\.js$/.test(f))) {
    vm.runInContext(fs.readFileSync(path.join(base, 'assets/js', file), 'utf8'), context, { filename: file });
  }
  for (const file of walk(path.join(base, '_posts')).filter(p => p.endsWith('.md'))) {
    for (const match of fs.readFileSync(file, 'utf8').matchAll(/{%\s*include\s+interactive\/article-lab\.html\s+([^%]+)%}/g)) {
      const id = /\bid="([^"]+)"/.exec(match[1])?.[1], preset = /\bpreset="([^"]*)"/.exec(match[1])?.[1] || '';
      if (id) { if (!presets.has(id)) presets.set(id, new Set()); presets.get(id).add(preset); }
    }
  }
  let cases = 0, plots = 0, renders = 0, sharedCases = 0, customInputCases = 0, numericEditCases = 0;
  for (const [id, factory] of factories) for (const preset of presets.get(id) || ['']) for (const ko of [false, true]) {
    const controls = [], buttons = [], customFields = [], defaults = []; let render = () => {}, resetState = () => {}, sharedState;
    const root = new StubNode('section', { id: 'lab-' + id, 'data-article-lab': id, 'data-language': ko ? 'ko' : 'en', 'data-preset': preset });
    const makeElement = (tag, attrs, children) => {
      const node = new StubNode(tag, attrs, children);
      if (['INPUT', 'TEXTAREA'].includes(node.tagName)) customFields.push(node);
      return node;
    };
    const ctx = {
      root, ko, preset, text: (en, korean) => ko ? korean : en,
      ui: new StubNode('fieldset', { class: 'al-controls' }), view: new StubNode('div', { class: 'al-view' }), metrics: new StubNode('dl', { class: 'al-metrics' }), status: new StubNode('p', { class: 'al-status' }),
      el: makeElement, svg: (tag, attrs, text) => new StubNode(tag, attrs, text), table: () => new StubNode(),
      metric: (en, korean) => { const output = new StubNode('output'); ctx.metrics.append(new StubNode('div', {}, [new StubNode('dt', {}, ko ? korean : en), new StubNode('dd', {}, output)])); return output; },
      format, asset: p => p, note: () => {},
      fetchJSON: async p => JSON.parse(localAsset(p)), fetchCSV: async p => parseCSV(localAsset(p)),
      control: (_en, _ko, options = {}) => {
        const type = options.type || 'range';
        const node = new StubNode(type === 'select' ? 'select' : 'input', { type, ...Object.fromEntries(['min', 'max', 'step'].filter(key => options[key] !== undefined).map(key => [key, options[key]])) });
        if (type === 'select') node.append(...options.options.map(option => new StubNode('option', { value: option.value }, option.en || option.value)));
        node.value = options.value === undefined ? '' : String(options.value); node.checked = Boolean(options.value);
        if (type === 'range' || type === 'number') {
          const value = Number(node.value); assert(Number.isFinite(value), id + ': nonfinite default');
          if (node.min !== undefined) assert(value >= node.min, id + ': default below minimum');
          if (node.max !== undefined) assert(value <= node.max, id + ': default above maximum');
          if (node.step !== undefined && node.step !== 'any') {
            const position = (value - Number(node.min || 0)) / Number(node.step);
            assert(Math.abs(position - Math.round(position)) < 1e-7, id + ': default off native step grid');
          }
        }
        controls.push(node); defaults.push({ node, value: node.value, checked: node.checked });
        ctx.ui.append(new StubNode('div', { class: 'al-control' + (type === 'checkbox' ? ' al-checkbox' : '') }, node)); return node;
      },
      button: (_en, _ko, handler) => { const node = new StubNode('button', { type: 'button', class: 'al-button' }); buttons.push(handler); node.addEventListener('click', handler); ctx.ui.append(node); return node; },
      onReset: fn => { resetState = fn; }, onChange: fn => { render = () => { renders++; fn(); }; render(); },
      stateLink: (key, config) => { assert.equal(key, id, id + ': share key does not match factory'); assert.equal(typeof config.read, 'function'); assert.equal(typeof config.write, 'function'); sharedState = config; },
      plot: spec => {
        plots++; assert(spec.x[1] > spec.x[0] && spec.y[1] > spec.y[0], id + ': invalid plot domain');
        assert([...spec.x, ...spec.y].every(Number.isFinite), id + ': nonfinite domain');
        const points = [...(spec.series || []).flatMap(s => s.points), ...(spec.markers || []).map(m => [m.x, m.y]), ...(spec.bars || []).map(b => [b.x, b.y])];
        for (const [x, y] of points) {
          assert(Number.isFinite(x) && Number.isFinite(y), id + ': nonfinite point');
          assert(x >= spec.x[0] - 1e-8 && x <= spec.x[1] + 1e-8, id + ': point outside x domain');
          assert(y >= spec.y[0] - 1e-8 && y <= spec.y[1] + 1e-8, id + ': point outside y domain');
        }
        return new StubNode();
      }
    };
    root.append(ctx.ui, ctx.view, ctx.metrics, ctx.status);
    const renderIfValid = () => {
      if (controls.some(node => node.type === 'number' && (node.value.trim() === '' || !node.validity.valid))) return;
      render();
    };
    ctx.ui.oninput = renderIfValid; ctx.ui.onchange = renderIfValid;
    await factory(ctx);
    assert.equal(root.querySelectorAll('.al-metrics').length, 1, id + ': metric container was lost or duplicated');
    assert.equal(ctx.metrics.closest('[data-article-lab]'), root, id + ': moved metrics left the lab');
    function sharedRoundTrip() {
      const saved = JSON.stringify(sharedState.read());
      assert(saved && saved.length < 16384, id + ': invalid or unbounded shared state');
      assert.notEqual(sharedState.write(JSON.parse(saved)), false, id + ': own shared state rejected');
      assert.equal(JSON.stringify(sharedState.read()), saved, id + ': shared state round-trip changed settings');
      const before = JSON.stringify(sharedState.read()); let rejected = false;
      try { rejected = sharedState.write({}) === false; } catch (_) { rejected = true; }
      assert(rejected, id + ': empty shared state accepted');
      assert.equal(JSON.stringify(sharedState.read()), before, id + ': invalid shared state partially mutated settings');
      sharedCases++;
    }
    if (sharedState) sharedRoundTrip();
    if (id === 'quantum-circuit-puzzle') {
      const selected = controls.find(node => node.type === 'number');
      assert(selected && sharedState, id + ': editable slot or circuit state missing');
      const initial = JSON.parse(JSON.stringify(sharedState.read()));
      const initialCircuit = JSON.stringify(initial.circuit);
      // Native input bubbles through the runtime's invalid-number gate; clicking
      // Insert still reaches the actual factory handler, which must not coerce
      // an unfinished empty draft to slot zero or mutate at an invalid slot.
      for (const invalid of ['', '-1', '0.5', String(initial.circuit.length + 1)]) {
        selected.value = invalid;
        selected.dispatchEvent(new StubEvent('input', { bubbles: true }));
        assert.equal(JSON.stringify(sharedState.read().circuit), initialCircuit, id + ': invalid input changed the circuit before editing');
        buttons[0]();
        assert.equal(JSON.stringify(sharedState.read().circuit), initialCircuit, id + ': Insert changed the circuit for invalid slot ' + JSON.stringify(invalid));
        assert.equal(Number(root.dataset.gateCount), initial.circuit.length, id + ': invalid Insert changed the displayed gate count');
      }
      selected.value = ''; selected.dispatchEvent(new StubEvent('input', { bubbles: true }));
      buttons[5]();
      assert.equal(sharedState.read().circuit.length, 0, id + ': Clear could not recover from an empty slot draft');
      assert.equal(selected.value, '0'); assert.equal(selected.validity.valid, true);
      selected.value = ''; selected.dispatchEvent(new StubEvent('input', { bubbles: true }));
      defaults.forEach(({ node, value, checked }) => { node.value = value; node.checked = checked; });
      resetState(); render();
      assert.equal(JSON.stringify(sharedState.read().circuit), initialCircuit, id + ': Reset did not restore the original circuit');
      assert.equal(selected.value, '0'); assert.equal(selected.validity.valid, true);
      buttons[0]();
      assert.equal(sharedState.read().circuit.length, initial.circuit.length + 1, id + ': valid Insert remained blocked after recovery');
      defaults.forEach(({ node, value, checked }) => { node.value = value; node.checked = checked; });
      resetState(); render();
      numericEditCases++;
    }
    for (const node of controls) {
      const original = node.value, checked = node.checked;
      const change = () => { node.dispatchEvent({ type: 'change' }); render(); };
      if (node.type === 'select') for (const option of node.options) { node.value = String(option.value); change(); }
      if (node.type === 'range' || node.type === 'number') for (const value of [node.min, node.max]) if (value !== undefined) { node.value = String(value); change(); }
      if (node.type === 'checkbox') { node.checked = !checked; change(); }
      node.value = original; node.checked = checked; change();
    }
    for (const handler of buttons) handler();
    // Custom view buttons (gate selection and plot selection) participate too.
    for (const node of ctx.view.querySelectorAll('button')) node.dispatchEvent(new StubEvent('click', { bubbles: true }));
    if (sharedState) sharedRoundTrip();
    if (customFields.length) {
      const values = customFields.map(node => node.value);
      const previousFeedback = ctx.status.textContent;
      customFields.forEach(node => { node.value = node.tagName === 'TEXTAREA' ? '{invalid json' : 'invalid expected value'; node.dispatchEvent(new StubEvent('input', { bubbles: true })); });
      assert(buttons.length > 0, id + ': custom inputs have no action to check');
      buttons[0]();
      assert.notEqual(ctx.status.textContent, previousFeedback, id + ': invalid custom input did not change feedback');
      for (const handler of buttons) handler();
      assert(ctx.status.textContent.length > 0, id + ': invalid custom input gave no feedback');
      customFields.forEach((node, i) => { node.value = values[i]; }); customInputCases++;
    }
    defaults.forEach(({ node, value, checked }) => { node.value = value; node.checked = checked; });
    resetState(); render();
    assert.equal(root.querySelectorAll('.al-metrics').length, 1, id + ': reset lost or duplicated metric container');
    cases++;
  }
  for (const id of presets.keys()) assert(factories.has(id), 'Included factory missing: ' + id);
  console.log(JSON.stringify({ status: 'passed', factories: factories.size, language_preset_cases: cases, renders, plots, shared_state_cases: sharedCases, custom_invalid_input_cases: customInputCases, numeric_edit_recovery_cases: numericEditCases, scope: 'DOM-contract stub, endpoints, dynamic options, custom buttons, sharing and invalid-input feedback; browser geometry still requires QA' }, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
