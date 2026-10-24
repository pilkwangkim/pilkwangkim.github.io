'use strict';
/* Contract smoke test only: no browser/layout claims, model calls, or network access. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const base = path.resolve(__dirname, '..');
const { parseCSV, format } = require('../assets/js/article-labs.js');
class StubNode {
  constructor() { this.dataset = {}; this.listeners = {}; this.children = []; this.textContent = ''; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  dispatchEvent(event) { for (const fn of this.listeners[event.type] || []) fn(event); }
  setAttribute(name, value) { this[name] = value; }
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
  const sandbox = { window: { ArticleLabs: registry }, ArticleLabs: registry, console, Math, Number, Array, String, Map, Set, Infinity };
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
  let cases = 0, plots = 0, renders = 0;
  for (const [id, factory] of factories) for (const preset of presets.get(id) || ['']) for (const ko of [false, true]) {
    const controls = [], buttons = []; let render = () => {}, resetState = () => {};
    const ctx = {
      root: Object.assign(new StubNode(), { dataset: { preset } }), ko, preset, text: (en, korean) => ko ? korean : en,
      ui: new StubNode(), view: new StubNode(), metrics: new StubNode(), status: new StubNode(),
      el: () => new StubNode(), svg: () => new StubNode(), table: () => new StubNode(), metric: () => new StubNode(),
      format, asset: p => p, note: () => {},
      fetchJSON: async p => JSON.parse(localAsset(p)), fetchCSV: async p => parseCSV(localAsset(p)),
      control: (_en, _ko, options) => {
        const type = options.type || 'range';
        const node = Object.assign(new StubNode(), options, { type, value: String(options.value), checked: Boolean(options.value) });
        if (type === 'range' || type === 'number') {
          const value = Number(node.value); assert(Number.isFinite(value), id + ': nonfinite default');
          if (node.min !== undefined) assert(value >= node.min, id + ': default below minimum');
          if (node.max !== undefined) assert(value <= node.max, id + ': default above maximum');
          if (node.step !== undefined && node.step !== 'any') {
            const position = (value - Number(node.min || 0)) / Number(node.step);
            assert(Math.abs(position - Math.round(position)) < 1e-7, id + ': default off native step grid');
          }
        }
        controls.push(node); return node;
      },
      button: (_en, _ko, handler) => { buttons.push(handler); return new StubNode(); },
      onReset: fn => { resetState = fn; }, onChange: fn => { render = () => { renders++; fn(); }; render(); },
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
    await factory(ctx);
    for (const node of controls) {
      const original = node.value, checked = node.checked;
      const change = () => { node.dispatchEvent({ type: 'change' }); render(); };
      if (node.type === 'select') for (const option of node.options) { node.value = String(option.value); change(); }
      if (node.type === 'range' || node.type === 'number') for (const value of [node.min, node.max]) if (value !== undefined) { node.value = String(value); change(); }
      if (node.type === 'checkbox') { node.checked = !checked; change(); }
      node.value = original; node.checked = checked; change();
    }
    for (const handler of buttons) handler();
    resetState(); render(); cases++;
  }
  for (const id of presets.keys()) assert(factories.has(id), 'Included factory missing: ' + id);
  console.log(JSON.stringify({ status: 'passed', factories: factories.size, language_preset_cases: cases, renders, plots, scope: 'DOM-contract stub, endpoints and options; browser geometry still requires QA' }, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
