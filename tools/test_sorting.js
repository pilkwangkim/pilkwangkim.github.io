/* Independent finite-input checks and bilingual DOM-contract tests; not browser geometry QA. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const m = require('../assets/js/article-labs-sorting.js');
let checks = 0, arraysChecked = 0, colorInputs = 0, factoryCases = 0;
function ok(value, message) { assert.ok(value, message); checks++; }
function eq(actual, expected, message) { assert.deepEqual(actual, expected, message); checks++; }
function throws(fn) { assert.throws(fn); checks++; }
function enumerate(n, alphabet, visit, a = []) { if (a.length === n) { visit(a.slice()); return; } for (const key of alphabet) { a.push(key); enumerate(n, alphabet, visit, a); a.pop(); } }
function stableReference(a) { return a.map((key, id) => ({ key, id })).sort((a, b) => a.key - b.key || a.id - b.id); }
function independentInversions(a) { return a.reduce((n, x, i) => n + a.slice(i + 1).filter(y => x > y).length, 0); }
const stableKinds = new Set(['bubble', 'insertion', 'binary-insertion', 'merge']);
const unstableWitness = new Map();
for (let n = 0; n <= 6; n++) enumerate(n, [0, 1, 2], values => {
  arraysChecked++; const expected = stableReference(values), sorted = expected.map(r => r.key), inv = independentInversions(values);
  eq(m.inversionCount(values), inv);
  for (let i = 0; i + 1 < values.length; i++) {
    const a = values.slice(); [a[i], a[i + 1]] = [a[i + 1], a[i]];
    eq(m.inversionCount(a) - inv, values[i] > values[i + 1] ? -1 : values[i] < values[i + 1] ? 1 : 0, 'adjacent-swap inversion identity');
  }
  for (const algorithm of m.ALGORITHMS) {
    const input = values.slice(), trace = m.sortingTrace(input, algorithm), final = trace.at(-1);
    eq(input, values, 'trace generator preserves caller input'); eq(final.array.map(r => r.key), sorted, `${algorithm} sorts`);
    eq(final.array.map(r => r.id).sort((a, b) => a - b), values.map((_, i) => i), 'record identities preserved');
    ok(final.done && trace.length <= m.MAX_TRACE, 'bounded completed trace');
    for (let k = 1; k < trace.length; k++) {
      ok(trace[k].comparisons >= trace[k - 1].comparisons && trace[k].writes >= trace[k - 1].writes, 'monotonic counter');
      ok(trace[k].active.every(i => Number.isInteger(i) && i >= 0 && i < values.length), 'active indices valid');
    }
    if (stableKinds.has(algorithm)) eq(final.array.map(({ key, id }) => ({ key, id })), expected, `${algorithm} stable`);
    else if (!m.isStable(final.array)) unstableWitness.set(algorithm, values);
    if (algorithm === 'selection') eq(final.comparisons, n * Math.max(0, n - 1) / 2, 'selection exact comparisons');
    if (algorithm === 'bubble') eq(final.swaps, inv, 'bubble swaps exactly inversions');
    if (algorithm === 'insertion' || algorithm === 'binary-insertion') eq(final.writes, inv + Math.max(0, n - 1), 'I shifts plus n-1 final writes');
  }
});
eq([...unstableWitness.keys()].sort(), ['heap', 'quick3', 'selection'], 'all nonstable variants have finite witnesses');
for (const algorithm of m.ALGORITHMS) for (const values of [m.PRESETS.random, m.PRESETS.sorted, m.PRESETS.reverse, m.PRESETS.nearly, m.PRESETS.duplicates, Array.from({ length: 24 }, (_, i) => 23 - i), Array(24).fill(7)]) {
  const final = m.sortingTrace(values, algorithm).at(-1); eq(final.array.map(r => r.key), values.slice().sort((a, b) => a - b));
}
for (let n = 2; n <= 24; n++) {
  const sorted = Array.from({ length: n }, (_, i) => i), reversed = sorted.slice().reverse(), equal = Array(n).fill(7);
  const bubble = m.sortingTrace(sorted, 'bubble').at(-1); eq(bubble.comparisons, n - 1); eq(bubble.writes, 0, 'early exit has linear sorted-input work');
  const linear = m.sortingTrace(reversed, 'insertion').at(-1), binary = m.sortingTrace(reversed, 'binary-insertion').at(-1);
  eq(linear.comparisons, n * (n - 1) / 2); eq(linear.writes, binary.writes, 'binary search does not eliminate moves');
  ok(binary.comparisons <= (n - 1) * Math.ceil(Math.log2(n)), 'binary insertion logarithmic comparisons per insertion');
  eq(m.sortingTrace(equal, 'quick3').at(-1).comparisons, 2 * (n - 1), 'all-equal 3-way partition scans once with two tests per record');
  eq(m.sortingTrace(reversed, 'quick3').at(-1).comparisons, n * (n - 1) / 2, 'deterministic first-pivot reverse worst case');
}
// Distribution reference checks compare full record identities, not just sorted keys.
for (let n = 2; n <= 5; n++) enumerate(n, [0, 1, 10], values => {
  for (const method of ['counting', 'radix']) for (const base of [2, 10]) {
    const trace = m.distributionTrace(values, method, base), final = trace.at(-1), expected = stableReference(values);
    eq(final.array.map(({ key, id }) => ({ key, id })), expected, `${method}/${base} stable result`);
    const passes = method === 'counting' ? 1 : Math.max(1, Math.floor(Math.log(Math.max(1, ...values)) / Math.log(base)) + 1);
    eq(final.writes, 2 * n * passes, 'one bucket and one main-array write per record per pass');
    for (const state of trace) for (const bucket of state.buckets) {
      // Bucket order is arrival order from that pass, which may differ from initial ID order.
      ok(new Set(bucket.map(r => r.id)).size === bucket.length, 'no identity duplicated within a bucket');
    }
  }
});
for (const method of ['counting', 'radix']) for (const base of [2, 10]) {
  const values = [99, 0, 21, 11, 12, 22, 11, 99], final = m.distributionTrace(values, method, base).at(-1);
  eq(final.array.map(({ key, id }) => ({ key, id })), stableReference(values));
}
// Enumerate all legal colors through length eight, checking every region invariant.
for (let n = 0; n <= 8; n++) enumerate(n, [0, 1, 2], values => {
  colorInputs++; const result = m.colorsTrace(values);
  eq(result.array.map(x => x.key), values.slice().sort((a, b) => a - b)); ok(result.passed && result.done && !result.limit && !result.error);
  eq(result.iterations, n, 'one unknown record classified per loop'); ok(result.writes <= 2 * n);
  for (const s of result.steps) {
    ok(0 <= s.low && s.low <= s.mid && s.mid <= s.high + 1 && s.high < n, 'pointer interval invariant');
    ok(s.array.slice(0, s.low).every(r => r.key === 0), 'zero region'); ok(s.array.slice(s.low, s.mid).every(r => r.key === 1), 'one region'); ok(s.array.slice(s.high + 1).every(r => r.key === 2), 'two region');
    eq(Math.max(0, s.high - s.mid + 1), n - s.iterations, 'unknown region is a decreasing potential');
  }
});
let rejectedMutations = 0;
for (const condition of ['<', '<=']) for (const zero of ['advance', 'stay']) for (const two of ['advance', 'stay']) {
  const cases = m.judgeColors({ condition, zero, two }), correct = condition === '<=' && zero === 'advance' && two === 'stay';
  eq(cases.every(c => c.passed), correct, 'eight fixtures distinguish every guided mutant'); if (!correct) rejectedMutations++;
  for (const c of cases) ok(c.iterations <= c.input.length && c.steps.length <= c.input.length + 2, 'wrong variants remain bounded');
  for (const c of cases) {
    ok(c.diagnosticEn.length > 20 && c.diagnosticKo.length > 10, 'each case has bilingual outcome-specific diagnostics');
    eq(c.purposeEn, m.CODE_CASE_PURPOSES[c.id][0]); eq(c.purposeKo, m.CODE_CASE_PURPOSES[c.id][1]);
    eq(c.expected, c.input.slice().sort((a, b) => a - b));
    ok([c.low, c.mid, c.high].every(Number.isInteger), 'terminal pointers are reported');
  }
}
eq(rejectedMutations, 7);
eq(m.judgeColors({ condition: '<', zero: 'advance', two: 'advance' }).filter(c => c.passed).map(c => c.id), ['empty', 'sorted', 'all-twos'], 'runnable wrong starter has genuine example passes, not artificial failure');
for (const n of [3, 4, 5, 6, 7]) for (const seed of [0, 17, 9999]) {
  const run = m.bogoExperiment(n, seed); eq(run, m.bogoExperiment(n, seed), 'seed reproducible');
  ok(run.attempts.length <= 2000); eq(run.expected, Array.from({ length: n }, (_, i) => i + 1).reduce((a, b) => a * b, 1));
  for (const a of run.attempts) eq(a.slice().sort((a, b) => a - b), Array.from({ length: n }, (_, i) => i + 1), 'shuffle preserves permutation');
  eq(run.success, run.attempts.at(-1).join(',') === Array.from({ length: n }, (_, i) => i + 1).join(','));
}
const permutations = new Set();
for (let j = 0; j < 3; j++) for (let k = 0; k < 2; k++) { const draws = [(j + .25) / 3, (k + .25) / 2]; permutations.add(m.shuffled([0, 1, 2], () => draws.shift()).join('')); }
eq(permutations.size, 6, 'all Fisher–Yates branches cover exactly 3! permutations');
for (const text of ['', '1', '1,,2', '1, -2', '1,1.2', '1,100', '1e1,2', 'NaN,2', '1,2,', '1,'.repeat(200), '0,'.repeat(24) + '0']) throws(() => m.parseInput(text));
eq(m.parseInput(' 0, 2, 99 '), [0, 2, 99]);
throws(() => m.sortingTrace([100])); throws(() => m.sortingTrace([1], 'unknown')); throws(() => m.colorsTrace([3])); throws(() => m.colorsTrace([1], {})); throws(() => m.bogoExperiment(8)); throws(() => m.bogoExperiment(3, -1)); throws(() => m.bogoExperiment(3, 1, 2001)); throws(() => m.distributionTrace([1, 2], 'bad')); throws(() => m.distributionTrace([1, 2], 'radix', 3));
eq(m.QUIZ.length, 5); eq(m.CODE_CASES.length, 8); eq(m.QUIZ.map(q => q.answer), [0, 1, 1, 0, 1]);
for (const q of m.QUIZ) {
  for (const field of ['en', 'ko', 'skillEn', 'skillKo', 'promptEn', 'promptKo', 'hintEn', 'hintKo', 'explainEn', 'explainKo']) ok(typeof q[field] === 'string' && q[field].length > 0, 'bilingual quiz field');
  eq(q.optionReasons.length, q.options.length);
  for (const reason of q.optionReasons) { eq(reason.length, 2); ok(reason.every(x => typeof x === 'string' && x.length > 20), 'every option has substantive bilingual feedback'); }
}
// Minimal actual handlers: native controls, custom inputs, answers, timer lifecycle, Reset, state rollback.
class StubNode {
  constructor(tag = 'div', attrs = {}, children = []) { this.tagName = tag.toUpperCase(); this.attrs = { ...attrs }; this.children = []; this.listeners = {}; this._text = ''; this.value = attrs.value || ''; this.checked = false; this.disabled = false; this.isConnected = true; this.className = attrs.class || ''; this.classList = { add: (...s) => { this.className += ' ' + s.join(' '); } }; this.append(...[children].flat()); }
  append(...nodes) { this.children.push(...nodes.filter(x => x !== undefined && x !== null)); }
  replaceChildren(...nodes) { this.children = []; this._text = ''; this.append(...nodes); }
  set textContent(value) { this.children = []; this._text = String(value); }
  get textContent() { return this._text + this.children.map(n => n instanceof StubNode ? n.textContent : n).join(''); }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'class') this.className = String(v); }
  addEventListener(event, fn) { (this.listeners[event] ||= []).push(fn); }
  fire(event, payload = {}) { for (const fn of this.listeners[event] || []) fn(payload); }
}
function descendants(n) { return n instanceof StubNode ? [n, ...n.children.flatMap(descendants)] : []; }
const factories = new Map(), timers = new Map(); let timerId = 0;
const win = { ArticleLabs: { register(id, fn) { ok(!factories.has(id)); factories.set(id, fn); } }, setTimeout(fn) { timers.set(++timerId, fn); return timerId; }, clearTimeout(id) { timers.delete(id); } };
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-sorting.js'), 'utf8'), { window: win, console });
function tick() { const entry = timers.entries().next().value; if (entry) { timers.delete(entry[0]); entry[1](); } }
function drain() { let count = 0; while (timers.size) { ok(count++ < 15000, 'animation bounded'); tick(); } }
function mount(id, ko) {
  const controls = [], buttons = [], metrics = [], custom = []; let render, reset, state;
  const ctx = { root: new StubNode('section'), ui: new StubNode(), view: new StubNode(), status: new StubNode(), ko,
    text: (en, korean) => ko ? korean : en,
    el: (tag, attrs, children) => { const n = new StubNode(tag, attrs, children); if (['input', 'textarea'].includes(tag)) custom.push(n); return n; },
    svg: (tag, attrs, text) => new StubNode(tag, attrs, text),
    control: (en, _ko, options) => { const n = new StubNode(options.type === 'select' ? 'select' : 'input'); n.value = String(options.value); n.checked = Boolean(options.value); n.initial = { value: n.value, checked: n.checked }; controls.push({ en, node: n, options }); return n; },
    metric: (en) => { const n = new StubNode('output'); metrics.push({ en, node: n }); return n; },
    button: (en, _ko, handler) => { const n = new StubNode('button'); n.addEventListener('click', handler); buttons.push({ en, node: n }); return n; },
    onChange: fn => { render = fn; fn(); }, onReset: fn => { reset = fn; }, table: (headers, rows) => new StubNode('table', {}, JSON.stringify({ headers, rows })),
    stateLink: (key, hooks) => { eq(key, id); state = hooks; }, checkpointState: hooks => { ok(!state); state = hooks; }
  };
  factories.get(id)(ctx); factoryCases++;
  return { ctx, controls, buttons, metrics, custom, state: () => state, render: () => render(), control: label => controls.find(c => c.en === label).node, metric: label => metrics.find(c => c.en === label).node.textContent,
    click: label => { const button = buttons.find(c => c.en === label); ok(button, `button ${label}`); button.node.fire('click'); },
    reset: () => { controls.forEach(c => { c.node.value = c.node.initial.value; c.node.checked = c.node.initial.checked; }); reset(); render(); }
  };
}
eq(factories.size, 8);
for (const ko of [false, true]) {
  for (const id of factories.keys()) {
    const lab = mount(id, ko), hooks = lab.state(); ok(hooks, 'every custom workbench has checkpoint/share state');
    const saved = JSON.stringify(hooks.read()); ok(saved.length < 16384, 'bounded state'); ok(hooks.write(JSON.parse(saved)) !== false); eq(JSON.stringify(hooks.read()), saved);
    eq(hooks.write({}), false, 'invalid settings rejected'); eq(JSON.stringify(hooks.read()), saved, 'invalid write does not mutate');
    for (const { node, options } of lab.controls) if (options.type === 'select') { const original = node.value; for (const option of options.options) { node.value = option.value; lab.render(); ok(lab.ctx.status.textContent.length > 0); } node.value = original; lab.render(); }
    for (const { en } of lab.buttons) { lab.click(en); drain(); ok(lab.ctx.status.textContent.length > 0); }
    // Restore progressed state into a fresh factory, as an editor reload does.
    const progressed = JSON.parse(JSON.stringify(lab.state().read()));
    const rebuilt = mount(id, ko);
    ok(rebuilt.state().write(progressed) !== false, 'fresh factory accepts progressed checkpoint');
    eq(JSON.stringify(rebuilt.state().read()), JSON.stringify(progressed), 'progress survives rebuilding');
    // The shared runtime renders after provider.write and accepted controls.
    // Re-rendering unchanged settings must preserve cursors and test results.
    for (let pass = 0; pass < 3; pass++) {
      rebuilt.render();
      eq(JSON.stringify(rebuilt.state().read()), JSON.stringify(progressed), 'post-restore render preserves progress');
    }
    const invalid = {
      'sorting-playground': [{ cursor: 1e9 }, { values: Array(25).fill(0) }, { speed: 81 }],
      'sorting-random': [{ cursor: 1e9 }, { n: 8 }, { seed: 10000 }],
      'sorting-inversions': [{ cursor: 1e9 }, { values: Array(9).fill(0) }, { swaps: 100001 }, { feedback: 'x'.repeat(501) }],
      'sorting-merge': [{ picks: Array(7).fill('L') }, { picks: ['L', 'L', 'L', 'L'] }, { feedback: 'x'.repeat(501) }],
      'sorting-partition': [{ cursor: 1e9 }, { values: [0, 3] }],
      'sorting-distribution': [{ cursor: 1e9 }, { values: Array(17).fill(0) }, { base: 3 }],
      'sorting-quiz': [{ question: 5 }, { answers: Array(5).fill(3) }, { hints: Array(5).fill('true') }],
      'sorting-code': [{ condition: 'while(true)' }, { zero: 'invalid' }, { tested: 'true' }]
    }[id];
    for (const patch of invalid) {
      eq(rebuilt.state().write({ ...progressed, ...patch }), false, 'bounded provider rejects corrupt progressed checkpoint');
      eq(JSON.stringify(rebuilt.state().read()), JSON.stringify(progressed), 'rejected checkpoint preserves active experiment');
    }
    lab.reset(); ok(lab.ctx.status.textContent.length > 0); eq(timers.size, 0, 'Reset cancels timers');
  }
  const playground = mount('sorting-playground', ko); playground.click('Step'); const beforeInvalid = JSON.stringify(playground.state().read()); playground.custom[0].value = 'x, -1'; playground.click('Apply array'); eq(JSON.stringify(playground.state().read()), beforeInvalid, 'invalid custom array preserves complete trace state'); ok(/unchanged|유지/.test(playground.ctx.status.textContent));
  playground.custom[0].value = '4, 2, 4, 1'; playground.click('Apply array'); eq(JSON.parse(JSON.stringify(playground.state().read().values)), [4, 2, 4, 1]);
  playground.click('Run'); eq(timers.size, 1); tick(); playground.click('Back'); eq(timers.size, 0, 'Back pauses'); playground.click('Run'); eq(timers.size, 1); playground.reset(); eq(timers.size, 0); eq(playground.state().read().cursor, 0);
  playground.click('Run'); playground.ctx.root.isConnected = false; tick(); eq(timers.size, 0, 'detached root stops'); playground.ctx.root.isConnected = true;
  const random = mount('sorting-random', ko); random.control('Distinct keys n').value = '7'; random.render(); random.click('Finish'); eq(random.metric('Expected shuffles'), '5040 = 7!'); ok(/2,000|2,000/.test(random.ctx.status.textContent) || /seed|seed/.test(random.ctx.status.textContent));
  const inversions = mount('sorting-inversions', ko); eq(inversions.metric('Current inversions I'), '6'); inversions.click('Swap one inverted pair'); eq(inversions.metric('Current inversions I'), '5'); inversions.control('Insertion trace').value = 'binary-insertion'; inversions.render(); inversions.click('Finish'); ok(Number(inversions.metric('Trace comparisons / writes').split(' / ')[1]) >= 5);
  const manualInversions = mount('sorting-inversions', ko);
  function swapFirstPair() { descendants(manualInversions.ctx.view).find(n => n.attrs['aria-label'] === (ko ? 'Index 0와 1 바꾸기' : 'Swap indices 0 and 1')).fire('click'); }
  swapFirstPair();
  ok(manualInversions.ctx.view.textContent.includes(ko ? '하나 줄었습니다' : 'removes exactly one'), 'inverted swap feedback reports decrease');
  swapFirstPair();
  ok(manualInversions.ctx.view.textContent.includes(ko ? '하나 늘었습니다' : 'creates exactly one'), 'ordered swap feedback reports increase');
  const equalSettings = JSON.parse(JSON.stringify(manualInversions.state().read()));
  ok(manualInversions.state().write({ ...equalSettings, values: [2, 2, 1, 1], cursor: 0 }));
  swapFirstPair();
  ok(manualInversions.ctx.view.textContent.includes(ko ? '변하지 않습니다' : 'do not change'), 'equal-key swap feedback reports unchanged count');
  const merge = mount('sorting-merge', ko); const wrongRight = descendants(merge.ctx.view).find(n => n.tagName === 'BUTTON' && /right|오른쪽/.test(n.textContent)); wrongRight.fire('click'); merge.click('Complete stable merge'); eq(merge.metric('Sorted / stable so far'), '✓ / ✗'); merge.reset(); merge.click('Complete stable merge'); eq(merge.metric('Sorted / stable so far'), '✓ / ✓');
  const partition = mount('sorting-partition', ko); partition.control('Input').value = 'recheck'; partition.render(); partition.click('Step'); const partitionStep = JSON.stringify(partition.state().read()); partition.render(); eq(JSON.stringify(partition.state().read()), partitionStep, 'render keeps partition step'); partition.click('Finish'); eq(partition.metric('Unknown records'), '0'); eq(partition.metric('Iterations / n'), '3 / 3');
  const distribution = mount('sorting-distribution', ko); distribution.control('Radix base').value = '2'; distribution.render(); distribution.click('Run'); eq(timers.size, 1); distribution.reset(); eq(timers.size, 0, 'distribution reset cancels timer');
  const quiz = mount('sorting-quiz', ko); for (let i = 0; i < 5; i++) {
    quiz.control('Question').value = String(i); quiz.render();
    const answers = descendants(quiz.ctx.view).filter(n => n.attrs.role === 'radio'); eq(answers.length, 3);
    for (let choice = 0; choice < answers.length; choice++) {
      answers[choice].fire('click');
      ok(quiz.ctx.view.textContent.includes(m.QUIZ[i].optionReasons[choice][ko ? 1 : 0]), 'feedback explains the selected option, not just the correct answer');
    }
    answers[m.QUIZ[i].answer].fire('click'); quiz.click('Hint');
    ok(quiz.ctx.view.textContent.includes(m.QUIZ[i][ko ? 'hintKo' : 'hintEn']));
    ok(quiz.ctx.view.textContent.includes(m.QUIZ[i][ko ? 'skillKo' : 'skillEn']));
  }
  eq(quiz.metric('Correct / answered'), '5 / 5'); quiz.click('Previous question'); eq(quiz.metric('Correct / answered'), '5 / 5'); quiz.reset(); eq(quiz.metric('Correct / answered'), '0 / 0');
  const code = mount('sorting-code', ko); code.click('Run eight tests'); eq(code.metric('Passed / cases'), '3 / 8', 'executable wrong starter matches Python');
  const cards = descendants(code.ctx.view).filter(n => n.tagName === 'DETAILS'); eq(cards.length, 8);
  eq(cards.filter(n => Object.hasOwn(n.attrs, 'open')).length, 1, 'open only the first failure; all eight cases remain expandable');
  ok(code.ctx.view.textContent.includes(ko ? '마지막 unknown record' : 'final unknown record'), 'a sorted-looking singleton still reports unfinished classification');
  ok(code.ctx.view.textContent.includes(ko ? '정답' : 'Expected'), 'expected arrays are displayed beside actual arrays');
  ok(code.ctx.view.textContent.includes('return a'), 'displayed API returns the same list');
  code.click('Fill reference solution'); eq(code.metric('Passed / cases'), '8 / 8'); code.render(); eq(code.metric('Passed / cases'), '8 / 8', 'render keeps judged results'); code.control('Loop while').value = '<'; code.render(); eq(code.metric('Passed / cases'), '— / 8', 'changed rules invalidate old judged results'); code.reset(); eq(code.metric('Passed / cases'), '— / 8');
}
console.log(JSON.stringify({ status: 'passed', checks, sort_arrays: arraysChecked, exhaustive_color_inputs: colorInputs, rejected_guided_mutants: rejectedMutations, factories: factories.size, bilingual_factory_mounts: factoryCases, unstable_witnesses: Object.fromEntries(unstableWitness), scope: 'Pure engines, exact fixtures, counters, stability, bounded invalid input, bilingual handlers, timer and checkpoint contracts. Browser layout is checked separately.' }, null, 2));
