'use strict';
const assert = require('node:assert/strict');
const a = require('../assets/js/article-labs-agent-challenges.js');
let assertions = 0;
for (const s of a.scenarios) {
  const before = JSON.stringify(s);
  assert.equal(s.actions.filter(x => x.safe && x.done).length, 1);
  for (const authority of ['user', 'external', 'none']) for (const action of s.actions) {
    const r = a.checkDecision(s.id, authority, action.id);
    assert.equal(r.solved, authority === s.authority && action.safe && action.done);
    assert.equal(r.safe, action.safe);
    assert.equal(r.done, action.done);
    if (!action.safe) assert.equal(r.solved, false);
    assertions += 4;
  }
  const denied = a.checkDecision(s.id, s.authority, 'deny');
  assert(denied.safe && !denied.done && !denied.solved);
  assert.equal(JSON.stringify(s), before);
}
assert.throws(() => a.checkDecision('missing', 'none', 'deny'));
assert.throws(() => a.checkDecision('invoice', 'system', 'summary'));
assert.throws(() => a.checkDecision('invoice', 'none', 'run-shell'));
for (const fallback of [0, 1, 10, 1000]) for (const value of [null, 0, 1, 5, 1000]) {
  const input = Object.freeze({ value, default: fallback });
  const oracle = value === null ? fallback : value;
  const r = a.evaluateTest('limit', input, oracle);
  assert.equal(r.oracle, oracle);
  assert.equal(r.b, oracle);
  assert.equal(r.bPass, true);
  assert.equal(r.separates, value === 0 && fallback !== 0);
  assertions += 4;
}
assert.deepEqual(a.parseTest('limit', '{"value":0}', '0').result, { oracle: 0, expectedValid: true, a: 10, b: 0, aPass: false, bPass: true, separates: true });
assert.equal(a.evaluateTest('limit', { value: 0 }, 10).expectedValid, false);
assert.equal(a.evaluateTest('limit', { value: null }, 10).expectedValid, true);
for (let n = 0; n <= 12; n++) for (let target = -1; target <= n; target++) {
  const values = Object.freeze(Array.from({ length: n }, (_, i) => i));
  const input = Object.freeze({ values, target });
  const oracle = target >= 0 && target < n ? target : -1;
  const r = a.evaluateTest('index', input, oracle);
  assert.equal(r.b, oracle);
  assert.equal(r.a, oracle === 0 ? -1 : oracle);
  assert.equal(r.separates, oracle === 0);
  assertions += 3;
}
assert.equal(a.runPatch('index', 'b', { values: [4, 7, 4], target: 4 }), 0);
assert.equal(a.runPatch('index', 'b', { values: [4, 7, 4], target: 8 }), -1);
for (const [issue, input] of [
  ['limit', {}], ['limit', { value: false }], ['limit', { value: '' }], ['limit', { value: -1 }], ['limit', { value: 1001 }], ['limit', { value: null, default: null }], ['limit', { value: 0, code: 'alert(1)' }],
  ['index', { values: [1], target: '1' }], ['index', { values: Array(13).fill(0), target: 0 }], ['index', { values: [Infinity], target: 0 }], ['index', { values: [1], target: 1, extra: 0 }], ['__proto__', {}]
]) assert.throws(() => a.validateInput(issue, input));
for (const source of ['undefined', 'function(){return 1}', '{"value":NaN}', '<script>alert(1)</script>', '[]', '{"value":false}']) assert.throws(() => a.parseTest('limit', source, '0'));
for (const expected of ['"0"', 'null', 'false', '0.5', '{}', '']) assert.throws(() => a.parseTest('limit', '{"value":0}', expected));
assert.throws(() => a.parseTest('limit', ' '.repeat(1201), '0'));
assert.throws(() => a.runPatch('limit', 'eval', { value: 1 }));
console.log(`Agent challenges: 4 trust scenarios, both issue contracts, ${assertions} evaluator checks, invalid input and immutable-input invariants pass.`);

// Focused DOM contract exercises stateful controls, not browser layout.
const fs = require('node:fs'), vm = require('node:vm');
class NodeStub {
  constructor(tag = 'div', attrs = {}, children = []) {
    this.tagName = tag.toUpperCase(); this.attrs = { ...attrs }; this.children = []; this.listeners = {}; this.textContent = ''; this.value = attrs.value || ''; this.dataset = {}; this.id = attrs.id || ''; this.classList = { add() {} }; this.append(...(Array.isArray(children) ? children : [children]));
  }
  append(...nodes) { this.children.push(...nodes.filter(n => n !== undefined && n !== null)); }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(event, fn) { (this.listeners[event] ||= []).push(fn); }
  fire(event) { for (const fn of this.listeners[event] || []) fn(); }
}
const factories = new Map();
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-agent-challenges.js'), 'utf8'), { window: { ArticleLabs: { register(id, fn) { factories.set(id, fn); } } }, console });
function mount(id, ko) {
  const controls = [], buttons = [], metrics = []; let render, resetState, share;
  const ctx = {
    root: new NodeStub('section', { id: 'lab-' + id }), ui: new NodeStub(), view: new NodeStub(), status: new NodeStub(), ko,
    text: (en, korean) => ko ? korean : en, el: (tag, attrs, children) => new NodeStub(tag, attrs, children),
    metric: () => { const node = new NodeStub('output'); metrics.push(node); return node; },
    control: (_en, _ko, opts) => { const node = new NodeStub(opts.type === 'select' ? 'select' : 'input', opts); node.value = String(opts.value); controls.push({ node, initial: node.value }); return node; },
    button: (_en, _ko, fn) => buttons.push(fn), table: (headers, rows) => new NodeStub('table', {}, [headers, rows]),
    onChange: fn => { render = fn; fn(); }, onReset: fn => { resetState = fn; },
    stateLink: (key, hooks) => { assert.equal(key, id); share = hooks; }
  };
  factories.get(id)(ctx);
  return { ctx, controls: controls.map(x => x.node), buttons, metrics, render: () => render(), share: () => share, reset: () => { for (const c of controls) c.node.value = c.initial; resetState(); render(); } };
}
for (const ko of [false, true]) {
  const s = mount('security-trust-puzzle', ko);
  s.buttons[0](); assert.match(s.ctx.status.textContent, ko ? /모두/ : /both/);
  s.controls[1].value = 'none'; s.controls[2].value = 'summary'; s.buttons[0](); assert.equal(s.metrics[0].textContent, '1 / 4');
  const solved = JSON.parse(JSON.stringify(s.share().read()));
  s.controls[0].value = 'meeting'; s.render(); assert.equal(s.controls[1].value, ''); assert.equal(s.controls[2].value, '');
  s.controls[1].value = 'user'; s.controls[2].value = 'schedule'; s.buttons[0](); assert.equal(s.metrics[0].textContent, '2 / 4');
  s.reset(); assert.equal(s.metrics[0].textContent, '0 / 4'); assert.equal(s.controls[0].value, 'invoice'); assert.equal(s.controls[1].value, ''); assert.equal(s.controls[2].value, '');
  assert.notEqual(s.share().write(solved), false); assert.equal(s.metrics[0].textContent, '1 / 4'); assert.equal(s.controls[2].value, 'summary');
  const old = JSON.stringify(s.share().read()); assert.equal(s.share().write({ ...solved, scenario: 'missing' }), false); assert.equal(JSON.stringify(s.share().read()), old);
  s.share().write({ ...solved, scenario: 'export', authority: 'user', action: 'redirect', checked: true }); assert.equal(s.controls[2].value, 'redirect');
  const g = mount('gemma-test-challenge', ko);
  const input = g.ctx.ui.children[0].children[1], expected = g.ctx.ui.children[1].children[1];
  assert.equal(expected.attrs.inputmode, 'text', 'Expected results need a keyboard that can enter a minus sign.');
  assert.equal(input.value, a.issues.limit.example); g.buttons[0](); assert.equal(g.metrics[0].textContent, '1');
  input.value = '{"value":0}'; expected.value = '0'; g.buttons[0](); assert.equal(g.metrics[0].textContent, '2'); assert.equal(g.metrics[1].textContent, '1');
  g.buttons[0](); assert.equal(g.metrics[0].textContent, '2'); // duplicate case does not grow suite
  expected.value = '10'; g.buttons[0](); assert.equal(g.metrics[0].textContent, '2');
  expected.value = '0'; const shared = JSON.parse(JSON.stringify(g.share().read()));
  input.value = '{"value":false}'; g.buttons[0](); assert.match(g.ctx.status.textContent, ko ? /유효하지/ : /Invalid/); assert.equal(g.metrics[0].textContent, '2');
  g.reset(); assert.equal(g.metrics[0].textContent, '0'); assert.equal(input.value, a.issues.limit.example); assert.equal(expected.value, '5');
  assert.notEqual(g.share().write(shared), false); assert.equal(g.metrics[0].textContent, '2'); assert.equal(input.value, '{"value":0}'); assert.equal(expected.value, '0');
  const oldG = JSON.stringify(g.share().read()); assert.equal(g.share().write({ ...shared, suite: [{ input: { value: 0 }, expected: 10 }] }), false); assert.equal(JSON.stringify(g.share().read()), oldG);
  g.controls[0].value = 'index'; g.render(); assert.equal(g.metrics[0].textContent, '0');
  assert.equal(expected.attrs.inputmode, 'text', 'The first-index issue must keep the minus key available.');
  input.value = '{"values":[4,7,4],"target":9}'; expected.value = '-1'; g.buttons[0]();
  assert.equal(g.metrics[0].textContent, '1'); assert.equal(g.metrics[1].textContent, '0');
  assert.equal(g.share().read().suite[0].expected, -1, 'An absent match must accept and retain the specified negative result.');
  input.value = '{"values":[4,7,4],"target":4}'; expected.value = '0'; g.buttons[0](); assert.equal(g.metrics[1].textContent, '1');
  for (let i = 1; i <= 10; i++) { input.value = JSON.stringify({ values: [i], target: i }); expected.value = '0'; g.buttons[0](); }
  assert.equal(g.metrics[0].textContent, '8'); assert.equal(g.share().read().suite.length, 8);
  g.buttons[1](); assert.equal(g.metrics[0].textContent, '0');
}
console.log('Agent challenge DOM contract: EN/KO decision feedback, custom input, duplicate/bounded suites, Reset, validated state restore and failed-restore immutability pass.');
