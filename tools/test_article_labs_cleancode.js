'use strict';
/* Independent rule oracle, Python fixture parity and focused DOM contracts.
 * No browser geometry claim, services, dependencies or arbitrary code execution. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const lab = require('../assets/js/article-labs-cleancode.js');
const base = path.resolve(__dirname, '..');
const copy = value => JSON.parse(JSON.stringify(value));
const fixtures = JSON.parse(fs.readFileSync(path.join(base, 'assets/data/cleancode/workshop-fixtures.json'), 'utf8'));
const runs = JSON.parse(fs.readFileSync(path.join(base, 'assets/data/cleancode/tdd-runs.json'), 'utf8'));

function reference(state, action, id) {
  const c = state.confirmed, w = state.waiting;
  if (action === 'register') {
    if (c.concat(w).includes(id)) return copy(state);
    return c.length < state.capacity ? { capacity: state.capacity, confirmed: c.concat(id), waiting: w.slice() } : { capacity: state.capacity, confirmed: c.slice(), waiting: w.concat(id) };
  }
  const remaining = c.filter(value => value !== id);
  return remaining.length < c.length && w.length ? { capacity: state.capacity, confirmed: remaining.concat(w[0]), waiting: w.slice(1) } : { capacity: state.capacity, confirmed: remaining, waiting: w.filter(value => value !== id) };
}
function orderedSubsets(remaining, prefix = []) {
  return [prefix, ...remaining.flatMap((id, i) => orderedSubsets(remaining.filter((_, j) => j !== i), prefix.concat(id)))];
}
let exhaustiveStates = 0, exhaustiveTransitions = 0, fixtureTransitions = 0;
for (let capacity = 1; capacity <= 4; capacity++) {
  for (const participants of orderedSubsets(['A', 'B', 'C', 'D'])) {
    for (let split = 0; split <= Math.min(capacity, participants.length); split++) {
      if (split < participants.length && split !== capacity) continue;
      const state = { capacity, confirmed: participants.slice(0, split), waiting: participants.slice(split) };
      lab.validateState(state); exhaustiveStates++;
      for (const action of ['register', 'cancel']) for (const id of ['A', 'B', 'C', 'D', 'E']) {
        const original = copy(state), result = lab[action](state, id);
        assert.deepEqual(result, reference(state, action, id));
        assert.deepEqual(state, original);
        assert.notEqual(result, state); assert.notEqual(result.confirmed, state.confirmed); assert.notEqual(result.waiting, state.waiting);
        lab.validateState(result);
        result.confirmed.push('unshared'); result.waiting.push('unshared'); assert.deepEqual(state, original);
        exhaustiveTransitions++;
      }
    }
  }
}
for (const fixture of fixtures.cases) {
  let state = copy(fixture.initial);
  for (const step of fixture.steps) {
    const before = copy(state), next = lab.operation(state, step);
    assert.deepEqual(next, step.expected, fixture.id);
    assert.deepEqual(state, before, fixture.id + ': source was mutated');
    assert.notEqual(next.confirmed, state.confirmed); assert.notEqual(next.waiting, state.waiting);
    state = next; fixtureTransitions++;
  }
}
for (const fixture of fixtures.invalid_cases) {
  const run = fixture.operation === 'new_workshop' ? () => lab.createState(fixture.capacity) : () => lab.operation(copy(fixture.initial), fixture);
  assert.throws(run, undefined, fixture.id);
}
for (const state of [null, [], { capacity: 1.5, confirmed: [], waiting: [] }, { capacity: 1, confirmed: ['A'], waiting: [2] }, { capacity: 1, confirmed: Array(1), waiting: [] }]) assert.throws(() => lab.validateState(state));
assert.deepEqual(lab.register(lab.createState(1), ' '), { capacity: 1, confirmed: [' '], waiting: [] });
assert.deepEqual(lab.replay(1, [{ operation: 'register', participant_id: 'A' }, { operation: 'register', participant_id: 'B' }, { operation: 'cancel', participant_id: 'A' }]).current, { capacity: 1, confirmed: ['B'], waiting: [] });
for (const args of [[0, []], [5, []], [2, Array(31).fill({ operation: 'cancel', participant_id: 'A' })], [2, [{ operation: 'edit', participant_id: 'A' }]], [2, [{ operation: 'register', participant_id: 'Z' }]], [2, [{ operation: 'cancel', participant_id: 'A', extra: 1 }]]]) assert.throws(() => lab.replay(...args));

const expectedDetection = { boundary: 'at-capacity', duplicate: 'duplicate-waiting', lifo: 'fifo', mutation: 'nonmutation' };
for (const test of lab.testCases) assert.equal(lab.runTest(test.id).passed, true);
for (const bug of lab.bugIds) {
  assert.deepEqual(lab.testCases.filter(test => !lab.runTest(test.id, bug).passed).map(test => test.id), [expectedDetection[bug]]);
  const result = lab.runTest(expectedDetection[bug], bug); assert.notDeepEqual(result.actual, result.expected);
}
for (let mask = 0; mask < 1 << lab.testCases.length; mask++) {
  const selected = lab.testCases.filter((_, i) => mask & 1 << i).map(test => test.id);
  assert.equal(lab.testMatrix(selected).length, selected.length);
}
assert.throws(() => lab.testMatrix(['fifo', 'fifo'])); assert.throws(() => lab.testMatrix(['missing']));
for (const preset of lab.refactorCases) assert.equal(lab.compareRefactor(preset.id, 'after').same, true);
assert.equal(lab.compareRefactor('at-capacity', 'faulty').same, false);
for (let capacity = 1; capacity <= 4; capacity++) for (const values of orderedSubsets(['A', 'B', 'C', 'D'])) {
  const state = { capacity, confirmed: values.slice(0, capacity), waiting: values.slice(capacity) };
  for (const id of ['A', 'B', 'C', 'D', 'E']) assert.deepEqual(lab.beforeRegister(state, id), lab.afterRegister(state, id));
}
assert.equal(lab.quizQuestions.length, 5);
for (const question of lab.quizQuestions) {
  assert.equal(question.options.filter(option => option.correct).length, 1);
  for (const option of question.options) {
    assert.deepEqual(lab.answerQuiz(question.id, option.id), { correct: option.correct === true, feedback: option.feedback });
    assert(option.feedback.every(value => typeof value === 'string' && value.length > 20));
  }
}
assert.equal(new Set(lab.quizQuestions.map(question => question.options.findIndex(option => option.correct))).size, 3);
assert.throws(() => lab.answerQuiz('missing', 'missing'));
assert.equal(runs.snapshots.length, 11);
for (const record of runs.snapshots) {
  assert.equal(record.outcome, record.exit_code === 0 ? 'passed' : 'failed');
  if (record.stage === 'red') { assert.equal(record.failed, 1); assert.match(record.stdout, /AssertionError/); assert.doesNotMatch(record.stdout, /ModuleNotFoundError|ImportError/); }
  else { assert.equal(record.failed, 0); assert.equal(record.exit_code, 0); }
  assert(record.failure_reason.en && record.failure_reason.ko);
}
const finalGreen = runs.snapshots.at(-2), refactor = runs.snapshots.at(-1);
assert.equal(finalGreen.test, refactor.test); assert.equal(finalGreen.passed, refactor.passed);

class Node {
  constructor(tag = 'div', attrs = {}, children = []) {
    this.tagName = tag; this.attrs = attrs; this.children = []; this.listeners = {}; this._text = ''; this.value = attrs.value || ''; this.checked = false; this.disabled = false;
    this.classList = { add: (...classes) => { this.attrs.class = `${this.attrs.class || ''} ${classes.join(' ')}`.trim(); } };
    this.append(children);
  }
  append(...nodes) { for (const node of nodes.flat(Infinity)) if (node !== undefined && node !== null && node !== false) this.children.push(node); }
  replaceChildren(...nodes) { this.children = []; this._text = ''; this.append(...nodes); }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(node => node instanceof Node ? node.textContent : String(node)).join(''); }
  addEventListener(event, callback) { (this.listeners[event] ||= []).push(callback); }
  focus(options) { this.focusOptions = options; }
  find(predicate) { return this.children.flatMap(node => node instanceof Node ? [...(predicate(node) ? [node] : []), ...node.find(predicate)] : []); }
}
const factories = {};
vm.runInNewContext(fs.readFileSync(path.join(base, 'assets/js/article-labs-cleancode.js'), 'utf8'), { window: { ArticleLabs: { register: (id, factory) => { factories[id] = factory; } } } });
async function domContracts() {
  let domCases = 0;
  assert.deepEqual(Object.keys(factories), ['cleancode-state', 'cleancode-tdd', 'cleancode-refactor', 'cleancode-tests', 'cleancode-quiz']);
  for (const [id, factory] of Object.entries(factories)) for (const ko of [false, true]) {
    const controls = [], buttons = []; let paint, reset = () => {}, sharing;
    const ctx = {
      ko, root: new Node('section'), ui: new Node('fieldset'), view: new Node(), metrics: new Node('dl'), status: new Node('p'),
      text: (en, korean) => ko ? korean : en, el: (tag, attrs, children) => new Node(tag, attrs, children),
      control: (en, korean, options) => { const node = new Node(options.type, {}, []); node.label = ko ? korean : en; node.value = String(options.value); node.checked = Boolean(options.value); controls.push(node); return node; },
      metric: () => { const node = new Node('output'); ctx.metrics.append(node); return node; },
      button: (en, korean, callback) => { const node = new Node('button', {}, ko ? korean : en); node.callback = callback; buttons.push(node); return node; },
      onChange: callback => { paint = callback; paint(); }, onReset: callback => { reset = callback; },
      stateLink: (key, state) => { assert.equal(key, id); assert.equal(sharing, undefined); sharing = state; },
      fetchJSON: async asset => JSON.parse(fs.readFileSync(path.join(base, asset), 'utf8'))
    };
    await factory(ctx);
    assert.match(ctx.root.attrs.class, /cc-lab/); assert(ctx.view.textContent.length > 20); assert(ctx.status.textContent.length > 20);
    const checkpoint = JSON.stringify(sharing.read()); assert(checkpoint.length < 16384);
    assert.notEqual(sharing.write(JSON.parse(checkpoint)), false); assert.equal(JSON.stringify(sharing.read()), checkpoint);
    const invalids = [{}, { unknown: 'invalid' }, null];
    if (id === 'cleancode-state') invalids.push({ capacity: 0, participant: 'A', history: [] }, { capacity: 1, participant: 'A', history: [{ operation: 'drop', participant_id: 'B' }] }, { capacity: 1, participant: 'A', history: [{ operation: 'register', participant_id: 'Z' }] });
    if (id === 'cleancode-tests') invalids.push({ tests: ['fifo', 'fifo'] }, { tests: ['missing'] });
    if (id === 'cleancode-quiz') invalids.push({ index: 10, answers: Array(5).fill(null) }, { index: 0, answers: ['invalid', null, null, null, null] }, { index: 0, answers: Array(5) });
    for (const invalid of invalids) { let rejected; try { rejected = sharing.write(invalid) === false; } catch (_) { rejected = true; } assert(rejected); assert.equal(JSON.stringify(sharing.read()), checkpoint, id + ': failed restore changed state'); }
    if (id === 'cleancode-state') {
      controls[0].value = '1'; controls[1].value = 'A'; buttons[0].callback();
      controls[0].value = '4'; controls[1].value = 'B'; paint(); buttons[0].callback();
      assert.equal(sharing.read().capacity, 1); assert.equal(controls[0].disabled, true); assert.match(ctx.view.textContent, /B/);
      for (let i = 0; i < 35; i++) buttons[0].callback(); assert.equal(sharing.read().history.length, 30);
      controls[0].value = '2'; reset(); paint(); assert.equal(controls[0].disabled, false); assert.equal(sharing.read().history.length, 0);
    }
    if (id === 'cleancode-tdd') {
      for (const record of runs.snapshots) { controls[0].value = record.id; paint(); assert(ctx.view.textContent.includes(record.failure_reason[ko ? 'ko' : 'en'])); assert(ctx.view.textContent.includes(record.output_excerpt)); }
      assert.match(ctx.view.textContent, ko ? /저장된 Python 실행 기록/ : /Saved Python execution record/);
    }
    if (id === 'cleancode-refactor') { controls[0].value = 'at-capacity'; controls[1].value = 'faulty'; paint(); assert.match(ctx.metrics.textContent, ko ? /반례/ : /Counterexample/); }
    if (id === 'cleancode-tests') { controls.forEach(node => { node.checked = true; }); paint(); assert.match(ctx.metrics.textContent, /4 \/ 4/); assert.equal(ctx.view.find(node => node.tagName === 'details').length, 4); }
    if (id === 'cleancode-quiz') {
      for (let i = 0; i < 5; i++) {
        const choices = ctx.view.find(node => node.tagName === 'button'); assert.equal(choices.length, 3);
        const correctIndex = lab.quizQuestions[i].options.findIndex(option => option.correct);
        choices[correctIndex].listeners.click[0]();
        const selected = ctx.view.find(node => node.tagName === 'button')[correctIndex];
        assert.equal(selected.attrs['aria-pressed'], 'true');
        assert.equal(selected.focusOptions?.preventScroll, true);
        assert(ctx.view.textContent.includes(lab.quizQuestions[i].options[correctIndex].feedback[ko ? 1 : 0]));
        if (i < 4) buttons[1].callback();
      }
      assert.match(ctx.metrics.textContent, /5 \/ 5/); reset(); paint(); assert.equal(sharing.read().index, 0); assert(sharing.read().answers.every(value => value === null));
    }
    assert.notEqual(sharing.write(JSON.parse(JSON.stringify(sharing.read()))), false); domCases++;
  }
  return domCases;
}
domContracts().then(domCases => console.log(JSON.stringify({ status: 'passed', exhaustive_states: exhaustiveStates, independent_model_transitions: exhaustiveTransitions, python_fixture_sequences: fixtures.cases.length, python_fixture_transitions: fixtureTransitions, invalid_fixture_cases: fixtures.invalid_cases.length, finite_test_selections: 64, test_fault_executions: lab.testCases.length * lab.bugIds.length, saved_pytest_snapshots: runs.snapshots.length, quiz_options: lab.quizQuestions.reduce((n, question) => n + question.options.length, 0), bilingual_dom_contracts: domCases, scope: 'finite model, recorded-data parity and DOM stub; browser layout requires separate QA' }, null, 2))).catch(error => { console.error(error); process.exitCode = 1; });
