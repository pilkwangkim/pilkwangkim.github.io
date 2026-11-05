'use strict';
/* Exact finite models and focused DOM/state lifecycle tests.
 * Actual browser geometry is checked separately. No dependencies are used. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const ecc = require('../assets/js/article-labs-ecc.js');
const copy = object => JSON.parse(JSON.stringify(object));
let modelChecks = 0, hammingSingles = 0, hammingDoubles = 0, extendedPatterns = 0;
const check = value => { assert(value); modelChecks++; };
const equal = (a, b) => { assert.deepEqual(a, b); modelChecks++; };
equal(ecc.encode('1011').join(''), '0110011');
equal(ecc.syndrome('0110111'), [1, 0, 1]);
equal(ecc.decode('0110111').data, [1, 0, 1, 1]);
equal(ecc.encodeExtended('1011').join(''), '01100110');
const codewords = Array.from({ length: 16 }, (_, n) => ecc.encode(ecc.fromInteger(n, 4)));
check(new Set(codewords.map(c => c.join(''))).size === 16);
let minimum = 7;
for (let i = 0; i < 16; i++) for (let j = i + 1; j < 16; j++) minimum = Math.min(minimum, ecc.distance(codewords[i], codewords[j]));
equal(minimum, 3);
for (let n = 0; n < 16; n++) {
  const m = ecc.fromInteger(n, 4), code = ecc.encode(m), ext = ecc.encodeExtended(m), before = copy(code);
  equal(ecc.syndrome(code), [0, 0, 0]);
  for (let position = 0; position <= 7; position++) {
    const received = ecc.applyFlips(code, position ? [position] : []), original = copy(received), decoded = ecc.decode(received);
    equal(decoded.data, m); equal(decoded.corrected, code); equal(decoded.index, position); equal(received, original); hammingSingles++;
  }
  for (let i = 1; i <= 7; i++) for (let j = i + 1; j <= 7; j++) {
    const received = ecc.applyFlips(code, [i, j]), decoded = ecc.decode(received);
    equal(decoded.index, i ^ j); equal(ecc.syndrome(decoded.corrected), [0, 0, 0]); check(decoded.data.join('') !== m.join('')); hammingDoubles++;
  }
  const patterns = [[], ...Array.from({ length: 8 }, (_, i) => [i + 1])];
  for (let i = 1; i <= 8; i++) for (let j = i + 1; j <= 8; j++) patterns.push([i, j]);
  for (const positions of patterns) {
    const received = ecc.applyFlips(ext, positions), beforeReceived = copy(received), decoded = ecc.decodeExtended(received);
    equal(received, beforeReceived);
    if (positions.length === 2) { equal(decoded.status, 'double-detected'); equal(decoded.data, null); equal(decoded.corrected, null); }
    else { equal(decoded.data, m); equal(decoded.corrected, ext); equal(decoded.status, positions.length ? positions[0] === 8 ? 'parity-corrected' : 'single-corrected' : 'clean'); }
    extendedPatterns++;
  }
  equal(code, before);
  const triple = ecc.applyFlips(code, [3, 5, 6]); equal(ecc.syndrome(triple), [0, 0, 0]); check(ecc.dataBits(triple).join('') !== m.join(''));
}
equal(hammingSingles, 128); equal(hammingDoubles, 336); equal(extendedPatterns, 592);
for (const n of [1, 3, 5, 7, 9]) for (const p of [0, .01, .1, .25, .5, 1]) {
  let oracle = 0;
  for (let mask = 0; mask < 2 ** n; mask++) {
    const r = mask.toString(2).padStart(n, '0').split('').map(Number), weight = r.reduce((a, b) => a + b, 0);
    if (ecc.majority(r)) oracle += p ** weight * (1 - p) ** (n - weight);
  }
  check(Math.abs(oracle - ecc.repetitionError(n, p)) < 1e-12);
}
check(Math.abs(ecc.repetitionError(3, .1) - .028) < 1e-14);
for (let n = 0; n < 8; n++) for (const t of [0, 1, 2]) {
  const r = ecc.fromInteger(n, 3), candidates = ecc.distanceCandidates(r, t);
  equal(candidates, ['000', '111'].filter(c => [...c].reduce((sum, bit, i) => sum + (Number(bit) !== r[i]), 0) <= t));
  if (t === 1) equal(candidates.length, 1);
}
for (let n = 0; n < 32; n++) {
  const data = ecc.fromInteger(n >> 1, 4), c = [...data, ecc.parity(data)], flip = ecc.fromInteger(n, 5).flatMap((b, i) => b ? [i + 1] : []);
  equal(ecc.parity(ecc.applyFlips(c, flip)), flip.length % 2);
}
const quizPairs = new Set();
for (let n = 0; n < 112; n++) {
  const q = ecc.quizProblem(n); equal(ecc.distance(q.received, q.codeword), 1); equal(ecc.decode(q.received).data, q.message); equal(ecc.decode(q.received).index, q.position); quizPairs.add(q.message.join('') + ':' + q.position);
}
equal(quizPairs.size, 112);
for (const run of [() => ecc.bits(Array(4), 4), () => ecc.bits('101x', 4), () => ecc.encode([1, 0, true, 1]), () => ecc.syndrome('010'), () => ecc.decodeExtended('0110011'), () => ecc.applyFlips('101', [0]), () => ecc.applyFlips('101', [1, 1]), () => ecc.majority('10'), () => ecc.repetitionError(2, .1), () => ecc.repetitionError(3, NaN), () => ecc.repetitionError(3, -1), () => ecc.distanceCandidates('000', .5), () => ecc.quizProblem(112), () => ecc.fromInteger(16, 4)]) {
  assert.throws(run); modelChecks++;
}

// Lightweight DOM adapter verifies control logic, focus targets and state
// restoration only. It does not stand in for browser geometry or native keys.
class Element {
  constructor(tag = 'div', attrs = {}, children = []) {
    this.tagName = tag.toUpperCase(); this.attrs = attrs; this.children = []; this.listeners = {}; this.value = ''; this.checked = false; this._text = '';
    this.classList = { add: () => {} }; this.append(children);
  }
  append(...children) { this.children.push(...children.flat(Infinity).filter(c => c !== undefined && c !== null && c !== false)); }
  replaceChildren(...children) {
    if (elements(this).includes(Element.activeElement)) {
      Element.activeElement.focused = false;
      Element.activeElement = null;
    }
    this.children = []; this._text = ''; this.append(children);
  }
  addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
  fire(type) { for (const callback of this.listeners[type] || []) callback(); }
  click() { this.fire('click'); }
  getAttribute(name) { return this.attrs[name] === undefined ? null : this.attrs[name]; }
  matches(selector) {
    const attribute = /^\[([\w-]+)="([^"]+)"\]$/.exec(selector);
    if (attribute) return String(this.getAttribute(attribute[1])) === attribute[2];
    if (selector.startsWith('.')) return String(this.attrs.class || '').split(/\s+/).includes(selector.slice(1));
    return this.tagName === selector.toUpperCase();
  }
  querySelectorAll(selector) { return elements(this).filter(node => node.matches(selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  focus() {
    if (Element.activeElement) Element.activeElement.focused = false;
    Element.activeElement = this; this.focused = true;
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c instanceof Element ? c.textContent : String(c)).join(' '); }
}
const factories = new Map();
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-ecc.js'), 'utf8'), { window: { ArticleLabs: { register: (id, factory) => factories.set(id, factory) } } });
assert.deepEqual([...factories.keys()].sort(), ['ecc-distance', 'ecc-hamming', 'ecc-parity', 'ecc-quiz', 'ecc-repetition']);
function mount(id, ko) {
  const fields = [], defaults = [], buttons = [], metrics = []; let provider, render, reset;
  const el = (tag, attrs, children) => new Element(tag, attrs, children);
  const ctx = {
    root: new Element(), ui: new Element(), view: new Element(), status: new Element(), metrics: new Element(), ko,
    el, svg: el, text: (en, korean) => ko ? korean : en,
    metric: () => { const output = new Element('output'); metrics.push(output); return output; },
    plot: spec => { assert(spec.series.every(s => s.points.every(p => p.every(Number.isFinite)))); return el('div', { plot: spec }, spec.title); },
    table: (headers, rows) => el('table', {}, [headers, rows].flat(Infinity)),
    button: (en, korean, callback) => { buttons.push({ label: ko ? korean : en, callback }); return new Element('button'); },
    control: (_en, _ko, options) => { const field = new Element(options.type === 'select' ? 'select' : 'input'); field.value = String(options.value); fields.push(field); defaults.push(field.value); return field; },
    onChange: fn => { render = fn; render(); }, onReset: fn => { reset = fn; }, stateLink: (key, value) => { assert.equal(key, id); provider = value; }
  };
  ctx.root.append(ctx.ui, ctx.view, ctx.metrics, ctx.status);
  factories.get(id)(ctx);
  return { ctx, fields, buttons, metrics, state: provider, render: () => render(), reset: () => { fields.forEach((f, i) => { f.value = defaults[i]; }); reset(); render(); }, snapshot: () => JSON.stringify({ state: provider.read(), view: ctx.view.textContent, status: ctx.status.textContent, metrics: metrics.map(m => m.textContent) }) };
}
function elements(node) { return node.children.flatMap(c => c instanceof Element ? [c, ...elements(c)] : []); }
let lifecycleChecks = 0, focusChecks = 0, presentationChecks = 0;
for (const id of factories.keys()) for (const ko of [false, true]) {
  const lab = mount(id, ko), original = copy(lab.state.read()), originalSnapshot = lab.snapshot();
  for (const payload of [null, [], {}, { nonsense: true }]) { assert.equal(lab.state.write(payload), false); assert.equal(lab.snapshot(), originalSnapshot); lifecycleChecks++; }
  // Each view contains actual bit controls. Clicking one changes the model.
  const bitControls = elements(lab.ctx.view).filter(node => node.tagName === 'BUTTON');
  const bit = bitControls[id === 'ecc-quiz' ? 1 : 0]; assert(bit); bit.click();
  assert.notEqual(lab.snapshot(), originalSnapshot); lifecycleChecks++;
  // Every control preset, including beyond-guarantee error patterns, renders.
  for (const button of lab.buttons) { button.callback(); assert(!lab.ctx.view.textContent.includes('undefined')); lifecycleChecks++; }
  const progressed = copy(lab.state.read()), snapshot = lab.snapshot(), restored = mount(id, ko);
  assert.equal(restored.state.write(progressed), true);
  assert.equal(restored.snapshot(), snapshot); lifecycleChecks++;
  for (let repeat = 0; repeat < 3; repeat++) { restored.render(); assert.equal(restored.snapshot(), snapshot); lifecycleChecks++; }
  for (const key of Object.keys(progressed)) {
    const bad = { ...copy(progressed), [key]: key === 'p' ? NaN : '__invalid__' };
    assert.equal(restored.state.write(bad), false, id + ':' + key); assert.equal(restored.snapshot(), snapshot); lifecycleChecks++;
  }
  restored.reset(); assert.equal(restored.snapshot(), originalSnapshot); lifecycleChecks++;
  assert.equal(lab.state.write(original), true); assert.equal(lab.snapshot(), originalSnapshot); lifecycleChecks++;
}

// Every editable row has a stable identity. Replacing the view must focus the
// new button in that row, not its removed predecessor or an equal position in
// the message/received row next to it.
for (const id of factories.keys()) for (const ko of [false, true]) {
  const lab = mount(id, ko);
  const rows = elements(lab.ctx.view).filter(node => node.getAttribute('data-ecc-row')
    && node.children.some(child => child instanceof Element && child.tagName === 'BUTTON'));
  assert.equal(new Set(rows.map(row => row.getAttribute('data-ecc-row'))).size, rows.length); focusChecks++;
  const identities = rows.map(row => ({ key: row.getAttribute('data-ecc-row'),
    positions: row.children.filter(child => child instanceof Element && child.tagName === 'BUTTON')
      .map(child => child.getAttribute('data-ecc-position')) }));
  for (const row of lab.ctx.view.querySelectorAll('.ecc-bits')) {
    assert.equal(Number(row.getAttribute('data-ecc-count')), row.children.length); presentationChecks++;
  }
  for (const { key, positions } of identities) for (const position of positions) {
    const oldRow = lab.ctx.root.querySelector(`[data-ecc-row="${key}"]`);
    const oldButton = oldRow.querySelector(`[data-ecc-position="${position}"]`);
    oldButton.focus(); oldButton.click();
    const newRow = lab.ctx.root.querySelector(`[data-ecc-row="${key}"]`);
    const newButton = newRow.querySelector(`[data-ecc-position="${position}"]`);
    assert.notEqual(newButton, oldButton); focusChecks++;
    assert.equal(Element.activeElement, newButton); focusChecks++;
    assert.equal(newButton.focused, true); focusChecks++;
    assert.equal(oldButton.focused, false); focusChecks++;
  }
}
{
  const first = mount('ecc-hamming', false), second = mount('ecc-hamming', false);
  first.ctx.root.querySelector('[data-ecc-row="received"]').querySelector('[data-ecc-position="5"]').click();
  assert.equal(Element.activeElement, first.ctx.root.querySelector('[data-ecc-row="received"]').querySelector('[data-ecc-position="5"]')); focusChecks++;
  assert.notEqual(Element.activeElement, second.ctx.root.querySelector('[data-ecc-row="received"]').querySelector('[data-ecc-position="5"]')); focusChecks++;
}

for (const ko of [false, true]) {
  const lab = mount('ecc-hamming', ko), base = copy(lab.state.read());
  assert.equal(lab.ctx.view.querySelector('.ecc-check-corner').textContent, 'r_j'); presentationChecks++;
  assert(lab.ctx.view.querySelector('.ecc-check-key').textContent.includes(ko ? 'H의 계수가 아니라' : 'not the coefficients of H')); presentationChecks++;
  assert.equal(lab.ctx.view.querySelector('.ecc-transmitted').querySelector('code').textContent, '0110011'); presentationChecks++;
  lab.state.write({ ...copy(base), flips: Array.from({ length: 8 }, (_, i) => i === 4) });
  assert.equal(lab.ctx.view.querySelector('.ecc-transmitted').querySelector('code').textContent, '0110011'); presentationChecks++;
  const received = lab.ctx.root.querySelector('[data-ecc-row="received"]');
  assert.equal(received.querySelector('[data-ecc-position="5"]').querySelector('strong').textContent, '1'); presentationChecks++;
  const flipMark = received.querySelector('[data-ecc-position="5"]').querySelector('.ecc-bit-mark');
  assert.equal(flipMark.textContent, ko ? '변경' : 'flip'); presentationChecks++;
  assert(received.querySelector('[data-ecc-position="5"]').getAttribute('aria-label').includes(ko ? '뒤집힘' : 'flipped')); presentationChecks++;
  lab.state.write({ ...copy(base), message: [0, 0, 0, 0] });
  assert(lab.ctx.view.querySelectorAll('.ecc-check-member').every(cell => cell.textContent === '0')); presentationChecks++;
  lab.state.write({ ...copy(base), mode: 'extended' });
  assert.equal(lab.ctx.view.querySelector('.ecc-transmitted').querySelector('code').textContent, '01100110'); presentationChecks++;
  lab.state.write({ ...copy(base), flips: Array.from({ length: 8 }, (_, i) => [2, 4].includes(i)) });
  assert(lab.ctx.view.textContent.includes(ko ? 'Decoder가 내놓은 message' : 'Decoder output message')); presentationChecks++;
  assert(lab.ctx.view.textContent.includes(ko ? 'Decoder가 내놓은 message가 원본과 다릅니다' : 'The output message is WRONG.')); presentationChecks++;
  const quiz = mount('ecc-quiz', ko);
  quiz.buttons.find(button => button.label === (ko ? 'Parity check 보기' : 'Show parity checks')).callback();
  assert.equal(quiz.ctx.view.querySelector('.ecc-check-corner').textContent, 'r_j'); presentationChecks++;
  assert(quiz.ctx.view.querySelector('.ecc-check-key').textContent.includes(ko ? '오른쪽 s' : 'rightmost s')); presentationChecks++;
  const repetition = mount('ecc-repetition', ko), repetitionState = copy(repetition.state.read());
  for (const n of [1, 3, 5, 7, 9]) {
    repetition.state.write({ ...copy(repetitionState), n });
    const row = repetition.ctx.view.querySelector('.ecc-bits');
    assert.equal(Number(row.getAttribute('data-ecc-count')), n); presentationChecks++;
    assert.equal(row.children.length, n); presentationChecks++;
  }
}

// Source contract only: actual row heights, wrapping and touch geometry are
// checked in the browser. Breakpoints are the exact widths at which all cells
// fit at 44px minimum plus four-pixel gaps.
const css = fs.readFileSync(require.resolve('../assets/css/article-labs-ecc.css'), 'utf8');
assert(css.includes('container:eccview / inline-size')); presentationChecks++;
assert(/\.ecc-bits \{[^}]*gap:4px/.test(css)); presentationChecks++;
for (const count of [7, 8, 9]) {
  const minimumWidth = count * 44 + (count - 1) * 4;
  const selector = `[data-ecc-count="${count}"]`;
  const containerBlock = css.slice(css.indexOf(`@container eccview (min-width:${minimumWidth}px)`)).split('\n}')[0];
  assert(containerBlock.includes(selector)); presentationChecks++;
  assert(containerBlock.includes(`repeat(${count},minmax(44px,1fr))`)); presentationChecks++;
  for (const width of [266, 336, 470]) {
    const columns = width >= minimumWidth ? count : count === 9 ? 3 : 4;
    assert(columns * 44 + (columns - 1) * 4 <= width); presentationChecks++;
  }
}

for (const ko of [false, true]) {
  const lab = mount('ecc-hamming', ko), base = copy(lab.state.read());
  const patterns = [[], [5], [3, 5], [3, 5, 6], [8]];
  for (const mode of ['hamming', 'extended']) for (const positions of patterns) {
    assert.equal(lab.state.write({ ...copy(base), mode, flips: Array.from({ length: 8 }, (_, i) => positions.includes(i + 1)) }), true);
    assert(!lab.ctx.view.textContent.includes('undefined')); lifecycleChecks++;
    if (mode === 'extended' && positions.length === 2) assert(lab.ctx.view.textContent.includes(ko ? '수정하지 않습니다' : 'refuse correction'));
    if (mode === 'hamming' && positions.length >= 2) assert(lab.ctx.view.textContent.includes(ko ? '원본과 다릅니다' : 'WRONG'));
  }
  const parityLab = mount('ecc-parity', ko), state = copy(parityLab.state.read());
  assert.equal(parityLab.state.write({ ...state, message: '1011' }), false); lifecycleChecks++;
  const quiz = mount('ecc-quiz', ko);
  for (let n = 0; n < 112; n++) {
    const q = ecc.quizProblem(n);
    assert.equal(quiz.state.write({ number: n, chosen: q.position, checked: true, hint: true }), true);
    assert(quiz.ctx.view.textContent.includes(ko ? '맞았습니다' : 'Correct.')); lifecycleChecks++;
  }
}
console.log(`PASS ECC: ${modelChecks} exact model checks; ${hammingSingles} Hamming clean/single cases; ${hammingDoubles} double miscorrections; ${extendedPatterns} SECDED <=2-error cases; ${lifecycleChecks} bilingual factory/state checks; ${focusChecks} focus assertions; ${presentationChecks} presentation checks.`);
