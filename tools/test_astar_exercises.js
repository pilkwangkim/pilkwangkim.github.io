/* Finite A* exercise/judge checks, independent Bellman references and DOM contracts.
   No browser, network, Python interpreter or general code evaluation is used here. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const base = path.resolve(__dirname, '..');
const data = require('../assets/data/astar-exercises.json');
const model = require('../assets/js/article-labs-pathfinding.js');
let checks = 0;
const ok = (value, message) => { assert.ok(value, message); checks++; };
const eq = (actual, expected, message) => { assert.deepEqual(actual, expected, message); checks++; };
const throws = (fn, message) => { assert.throws(fn, undefined, message); checks++; };
const solution = { priority: 'g + h', relaxation: 'candidate < old_g', goalCheck: 'pop' };
// Independent materialization: test every pair's coordinate difference instead
// of using the runtime's direction-list neighbor generator.
function referenceGraph(test) {
  const graph = test.graph;
  if (!graph.grid) return graph;
  const { width, height } = graph.grid, n = width * height;
  const edges = Array.from({ length: n }, (_, u) => Array.from({ length: n }, (_, v) => v).filter(v => Math.abs(u % width - v % width) + Math.abs(Math.floor(u / width) - Math.floor(v / width)) === 1).map(v => [v, 1]));
  const h = edges.map((_, u) => Math.abs(u % width - graph.goal % width) + Math.abs(Math.floor(u / width) - Math.floor(graph.goal / width)));
  return { edges, h, start: graph.start, goal: graph.goal };
}
function bellman(graph, reverse = false) {
  const distance = Array(graph.edges.length).fill(Infinity);
  distance[reverse ? graph.goal : graph.start] = 0;
  for (let pass = 1; pass < graph.edges.length; pass++) {
    let changed = false;
    for (let u = 0; u < graph.edges.length; u++) for (const [v, cost] of graph.edges[u]) {
      const a = reverse ? v : u, b = reverse ? u : v;
      if (distance[a] + cost < distance[b]) { distance[b] = distance[a] + cost; changed = true; }
    }
    if (!changed) break;
  }
  return distance;
}
eq(data.version, 1); eq(data.questions.length, 5); eq(data.coding.cases.length, 8);
for (const item of [...data.questions, data.coding]) {
  for (const field of ['desktop', 'mobile', 'png']) ok(fs.existsSync(path.join(base, item.image[field].replace(/^\//, ''))), 'Image exists: ' + item.image[field]);
  for (const language of ['en', 'ko']) for (const field of ['alt', 'caption']) ok(item.image[field][language].length > 10, 'Bilingual image text');
}
for (const question of data.questions) {
  ok(Number.isInteger(question.answer) && question.answer >= 0 && question.answer < question.options.length, 'Answer index');
  for (const language of ['en', 'ko']) {
    for (const field of ['nav', 'title', 'prompt', 'hint', 'explanation', 'followup', 'anchors']) ok(typeof question[field][language] === 'string' && question[field][language].length > 3, 'Bilingual question field');
    for (const option of question.options) ok(option[language].length > 0, 'Bilingual option');
  }
}
for (const test of data.coding.cases) {
  const graph = referenceGraph(test), exact = bellman(graph)[graph.goal], remaining = bellman(graph, true);
  eq(Number.isFinite(exact) ? exact : null, test.expected, test.id + ': independent expected cost');
  const normalized = value => ({ start: value.start, goal: value.goal, h: value.h, edges: value.edges.map(row => row.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1])) });
  eq(normalized(model.materializeCodeCase(test)), normalized(graph), test.id + ': grid/materialization parity');
  for (let u = 0; u < graph.edges.length; u++) ok(graph.h[u] <= remaining[u], test.id + ': every h is admissible');
}
const correct = model.judgeCodeCases(data.coding.cases, solution);
ok(correct.every(row => row.passed), 'Reference solution passes every cost and effort check');
eq(correct.find(row => row.test.id === 'stale').stale, 1, 'Obsolete entry is actually encountered');
eq(correct.find(row => row.test.id === 'reopening').reopened, 1, 'Expanded state is actually reinserted');
eq(correct.find(row => row.test.id === 'informed').expanded, 28, 'Authored grid: A* expands28');
eq(correct.find(row => row.test.id === 'start-goal').expanded, 0, 'Accepted goal is not expanded');
const dijkstra = model.judgeCodeCases(data.coding.cases, { ...solution, priority: 'g' });
ok(dijkstra.every(row => row.costPassed), 'Dijkstra remains cost-optimal on every test');
eq(dijkstra.filter(row => !row.passed).map(row => row.test.id), ['informed'], 'Only the explicit heuristic-use budget rejects Dijkstra');
eq(dijkstra.at(-1).expanded, 224, 'Dijkstra expansion count excludes accepted goal');
const greedy = model.judgeCodeCases(data.coding.cases, { ...solution, priority: 'h' });
eq(greedy[0].cost, 21, 'Illustrated example exposes greedy');
eq(greedy.find(row => row.test.id === 'reopening').cost, 6, 'Greedy reopening counterexample');
const weighted = model.judgeCodeCases(data.coding.cases, { ...solution, priority: 'g + 2*h' });
eq(weighted.find(row => row.test.id === 'reopening').cost, 6, 'Overweighted h loses optimality');
const discovered = model.judgeCodeCases(data.coding.cases, { ...solution, goalCheck: 'discovery' });
eq(discovered.find(row => row.test.id === 'first-goal').cost, 10, 'First discovery is not an optimality certificate');
const zero = data.coding.cases.find(test => test.id === 'zero-cycle');
const cycling = model.runCodeCase(zero, { ...solution, relaxation: 'candidate <= old_g' });
ok(cycling.limited && !cycling.error && !cycling.found, 'Equal-cost cycle hits teaching limit, not no-route');
eq(cycling.pops, 5000, 'Limit counts exactly5000 raw pops');
ok(!model.judgeCodeCases([zero], { ...solution, relaxation: 'candidate <= old_g' })[0].costPassed, 'A limited run never passes as unreachable');
const invalid = ['', 'candidate', 'g < h', 'candidate = old_g', 'candidate < old_g or True', 'candidate < old_g; alert(1)', 'candidate < old_g.__class__', 'candidate < old_g()'];
for (const expression of invalid) throws(() => model.parseRelaxation(expression), 'Reject unsupported comparison: ' + expression);
for (const expression of ['candidate < old_g', ' candidate <= old_g ', 'candidate==old_g', 'candidate != old_g', 'candidate > old_g', 'candidate >= old_g']) eq(typeof model.parseRelaxation(expression), 'function', 'Accepted bounded comparison');
for (const changes of [{ priority: '' }, { priority: 'Math.min(g,h)' }, { priority: 'g.__class__' }, { relaxation: '' }, { goalCheck: '' }, { popLimit: 0 }, { popLimit: 5001 }]) ok(Boolean(model.runCodeCase(zero, { ...solution, ...changes }).error), 'Invalid fragment/limit reports an error');
const limited = model.runCodeCase(data.coding.cases[0], { ...solution, popLimit: 1 });
ok(limited.limited && limited.pops === 1, 'Small explicit limit is also exact');
const bad = { graph: { edges: [[[1, -1]], []], h: [0, 0], start: 0, goal: 1 } };
ok(Boolean(model.runCodeCase(bad, solution).error), 'Negative-cost edge rejected');
// Exhaust the supported mutation dimensions on the finite authored suite.
let mutationRuns = 0;
for (const priority of ['g+h', 'g', 'h', 'g+2*h', 'g+0.5*h', '0']) for (const comparison of ['<', '<=', '>', '>=', '==', '!=']) for (const goalCheck of ['pop', 'discovery']) {
  const rows = model.judgeCodeCases(data.coding.cases, { priority, relaxation: `candidate ${comparison} old_g`, goalCheck });
  for (const row of rows) {
    mutationRuns++;
    ok(row.pops <= 5000 && row.expanded <= row.pops && row.stale <= row.pops, 'Every mutation stays bounded');
    ok(row.cost === null || Number.isFinite(row.cost), 'No NaN/Infinity output');
    if (row.limited || row.error) ok(!row.passed, 'Failure is not an unreachable success');
  }
}
// Additional random nonnegative directed graphs include zero edges and cycles.
let seed = 73129;
const random = () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; };
for (let trial = 0; trial < 120; trial++) {
  const n = 2 + Math.floor(random() * 7), graph = { start: 0, goal: n - 1, edges: Array.from({ length: n }, () => []), h: [] };
  for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) if (u !== v && random() < .3) graph.edges[u].push([v, Math.floor(random() * 9)]);
  const exact = bellman(graph)[graph.goal], back = bellman(graph, true);
  graph.h = back.map(value => Number.isFinite(value) ? value * random() : 0); graph.h[graph.goal] = 0;
  for (const priority of ['g+h', 'g']) {
    const result = model.runCodeCase({ graph }, { ...solution, priority });
    ok(!result.error && !result.limited, 'Finite nonnegative graph terminates');
    ok(exact === Infinity ? result.cost === null : model.sameCost(result.cost, exact), 'Random graph versus independent Bellman relaxation');
  }
}

// Actual factories in both languages, using a small DOM stub. This is not a
// layout test: desktop/mobile pixels must still be checked in a real browser.
class Node {
  constructor(tag = 'div', attrs = {}, children = []) {
    this.tagName = tag.toUpperCase(); this.attributes = {}; this.children = []; this.listeners = {}; this.dataset = {}; this._text = ''; this.value = ''; this.disabled = false;
    this.classList = { add() {} };
    for (const [key, value] of Object.entries(attrs || {})) this.setAttribute(key, value);
    this.append(...(Array.isArray(children) ? children : [children]));
  }
  append(...nodes) { this.children.push(...nodes.filter(node => node !== undefined && node !== null)); }
  replaceChildren(...nodes) { this.children = []; this._text = ''; this.append(...nodes); }
  get textContent() { return this._text + this.children.map(node => node instanceof Node ? node.textContent : String(node)).join(''); }
  set textContent(value) { this.replaceChildren(); this._text = String(value); }
  setAttribute(key, value) { this.attributes[key] = String(value); if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-([a-z])/g, (_m, letter) => letter.toUpperCase())] = String(value); else this[key] = String(value); }
  getAttribute(key) { return this.attributes[key] ?? null; }
  addEventListener(event, handler) { (this.listeners[event] ||= []).push(handler); }
  fire(event, payload = {}) { for (const handler of this.listeners[event] || []) handler({ preventDefault() {}, ...payload }); }
  focus() { this.focused = true; }
  all(predicate) { return this.children.flatMap(node => node instanceof Node ? [...(predicate(node) ? [node] : []), ...node.all(predicate)] : []); }
}
const factories = new Map(), hashListeners = [];
const win = { ArticleLabs: { register(id, factory) { factories.set(id, factory); } }, location: { hash: '' }, addEventListener(type, handler) { if (type === 'hashchange') hashListeners.push(handler); }, removeEventListener(type, handler) { if (type === 'hashchange') { const index = hashListeners.indexOf(handler); if (index >= 0) hashListeners.splice(index, 1); } } };
vm.runInNewContext(fs.readFileSync(path.join(base, 'assets/js/article-labs-pathfinding.js'), 'utf8'), { window: win, console });
async function mount(id, ko) {
  const controls = [], fields = [], buttons = [], metrics = []; let render, reset;
  const ctx = {
    root: new Node('section', { 'data-asset-version': 'test-asset-version' }), ui: new Node(), view: new Node(), status: new Node(), ko,
    text: (en, korean) => ko ? korean : en, asset: p => '/base' + p,
    el(tag, attrs, children) { const node = new Node(tag, attrs, children); if (tag === 'input') fields.push(node); return node; },
    metric() { const node = new Node('output'); metrics.push(node); return node; },
    control(_en, _ko, options) { const node = new Node('select'); node.value = String(options.value); node.initial = node.value; node.options = options.options; controls.push(node); return node; },
    button(en, _ko, handler) { const node = new Node('button'); node.handler = handler; buttons.push({ en, node }); return node; },
    table(headers, rows) { return new Node('table', {}, [new Node('caption', {}, JSON.stringify({ headers, rows }))]); },
    fetchJSON: async file => { eq(file, '/assets/data/astar-exercises.json', 'Only declared local JSON is fetched'); return data; },
    onReset(fn) { reset = fn; }, onChange(fn) { render = fn; fn(); }, checkpointState(config) { ctx.checkpoint = config; }
  };
  await factories.get(id)(ctx);
  return { ctx, controls, fields, metrics, render, reset() { controls.forEach(node => { node.value = node.initial; }); reset(); render(); }, click(en) { const button = buttons.find(button => button.en === en); ok(Boolean(button), 'Button exists: ' + en); button.node.handler(); } };
}
async function domChecks() {
  for (const ko of [false, true]) {
    const quiz = await mount('astar-quiz', ko);
    eq(quiz.controls[0].options.map(option => option[ko ? 'ko' : 'en']), data.questions.map(question => question.nav[ko ? 'ko' : 'en']), 'Compact question navigation uses short bilingual labels');
    eq(quiz.metrics.map(node => node.textContent), ['0 / 5', '0 / 5'], 'Blank quiz');
    eq(quiz.ctx.view.all(node => node.tagName === 'IMG').length, 1, 'One current diagram, not five duplicate blocks');
    const image = quiz.ctx.view.all(node => node.tagName === 'IMG')[0];
    ok(image.src.endsWith('?v=test-asset-version'), 'Image URL includes asset fingerprint');
    for (let index = 0; index < data.questions.length; index++) {
      quiz.controls[0].value = String(index); quiz.render();
      const buttons = quiz.ctx.view.all(node => node.attributes.role === 'radio');
      eq(buttons.length, data.questions[index].options.length, 'Every answer remains available');
      buttons[data.questions[index].answer].fire('click');
      ok(quiz.ctx.view.textContent.includes(data.questions[index].explanation[ko ? 'ko' : 'en']), 'Immediate bilingual explanation');
      eq(quiz.ctx.view.all(node => node.attributes.role === 'radio').filter(node => node.attributes['aria-checked'] === 'true').length, 1, 'Exactly one answer selected');
    }
    eq(quiz.metrics.map(node => node.textContent), ['5 / 5', '5 / 5'], 'Score uses all five answers');
    quiz.click('Previous exercise'); eq(quiz.controls[0].value, '3', 'Previous question navigation');
    quiz.click('Next exercise'); eq(quiz.controls[0].value, '4', 'Next question navigation');
    quiz.click('Show / hide hint'); ok(quiz.ctx.view.textContent.includes(data.questions[4].hint[ko ? 'ko' : 'en']), 'Hint belongs to the current question');
    quiz.reset(); eq(quiz.metrics.map(node => node.textContent), ['0 / 5', '0 / 5'], 'Reset clears custom answers');
    const firstRadio = quiz.ctx.view.all(node => node.attributes.role === 'radio')[0]; firstRadio.fire('keydown', { key: 'ArrowRight' });
    eq(quiz.ctx.view.all(node => node.attributes.role === 'radio')[1].getAttribute('aria-checked'), 'true', 'Arrow keys select and focus an answer');
    quiz.reset();
    win.location.hash = '#' + encodeURIComponent(data.questions[2].anchors[ko ? 'ko' : 'en']); hashListeners.at(-1)();
    eq(quiz.controls[0].value, '2', 'Legacy question anchor opens the matching panel');
    const retiredHashListener = hashListeners.at(-1), oldListenerCount = hashListeners.length;
    quiz.ctx.root.isConnected = false; win.location.hash = '#' + encodeURIComponent(data.questions[0].anchors.en); retiredHashListener();
    eq(quiz.controls[0].value, '2', 'Detached editor widget ignores article hash navigation');
    eq(hashListeners.length, oldListenerCount - 1, 'Detached quiz releases its hash listener'); win.location.hash = '';
    const code = await mount('astar-code-test', ko);
    eq(code.metrics.map(node => node.textContent), ['—', '—'], 'Coding test starts without a solution');
    code.click('Run coding tests'); ok(code.ctx.view.textContent.includes(ko ? '입력 오류' : 'Invalid fragment'), 'Blank TODOs produce feedback');
    code.click('Load reference solution'); eq(code.metrics.map(node => node.textContent), ['8 / 8', '8 / 8'], 'Reference fills all three decisions and passes');
    code.fields[0].value = 'g'; code.render(); eq(code.metrics[0].textContent, '—', 'Editing clears stale results');
    code.click('Run coding tests'); eq(code.metrics.map(node => node.textContent), ['7 / 8', '8 / 8'], 'Correct cost is distinct from heuristic-use effort');
    ok(code.ctx.view.textContent.includes(ko ? '비용은 정답, 탐색 상한 초과' : 'Cost correct; budget exceeded'), 'Dijkstra feedback does not claim suboptimality');
    code.fields[1].value = 'candidate <= old_g'; code.click('Run coding tests');
    ok(code.ctx.view.textContent.includes(ko ? '실행 상한 도달' : 'Limit reached'), 'Cycle limit is visible');
    code.fields[0].value = 'window.location'; code.click('Run coding tests');
    eq(code.fields[0].getAttribute('aria-invalid'), 'true', 'Unsupported code is visibly invalid');
    code.reset(); eq(code.fields.map(node => node.value), ['', ''], 'Reset clears custom text inputs');
    eq(code.controls[0].value, '', 'Reset clears goal-choice TODO');
    eq(code.metrics.map(node => node.textContent), ['—', '—'], 'Reset clears judge results');
    ok(code.ctx.view.textContent.includes('TODO_PRIORITY') && code.ctx.view.textContent.includes('TODO_RELAXATION') && code.ctx.view.textContent.includes('TODO_GOAL_CHECK'), 'Shared Python starter displays all three TODOs');
  }
  console.log(JSON.stringify({ status: 'passed', checks, authored_cases: 8, illustrated_questions: 5, mutation_runs: mutationRuns, random_nonnegative_graphs: 120, independent_reference: 'Bellman relaxation, independently materialized grid', reference: correct.map(row => ({ id: row.test.id, cost: row.cost, expanded: row.expanded, pops: row.pops, stale: row.stale, reopened: row.reopened })), dijkstra_cost_checks: '8/8; only map-specific effort budget fails', browser_scope: 'Bilingual DOM contract only; real desktop/mobile geometry remains separate' }, null, 2));
}
domChecks().catch(error => { console.error(error); process.exitCode = 1; });
