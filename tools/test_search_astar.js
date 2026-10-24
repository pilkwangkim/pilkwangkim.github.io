/* Independent exact-distance and counterexample checks; no DOM or external data. */
'use strict';
const assert = require('node:assert/strict');
const m = require('../assets/js/article-labs-search-astar.js');
let checks = 0;
function ok(condition, message = '') { assert.ok(condition, message); checks++; }
function eq(actual, expected, message = '') { assert.deepEqual(actual, expected, message); checks++; }
// Independent neighbor implementation: swap adjacent row/column positions.
function adjacent(state) {
  const z = state.indexOf('0'), out = [];
  for (let j = 0; j < 9; j++) if (Math.abs(j % 3 - z % 3) + Math.abs(Math.floor(j / 3) - Math.floor(z / 3)) === 1) {
    const a = state.split(''); a[z] = state[j]; a[j] = '0'; out.push(a.join(''));
  }
  return out;
}
const distance = new Map([[m.GOAL, 0]]), queue = [m.GOAL];
for (let head = 0; head < queue.length; head++) {
  const state = queue[head], depth = distance.get(state);
  if (depth === 12) continue;
  for (const next of adjacent(state)) if (!distance.has(next)) { distance.set(next, depth + 1); queue.push(next); }
}
eq(distance.size, 1850, 'independent reverse BFS through exact depth 12');
for (const [state, depth] of distance) {
  eq(m.puzzleNeighbors(state).map(x => x.state).sort(), adjacent(state).sort(), `legal neighbor set ${state}`);
  ok(m.manhattan(state) <= depth, `admissible Manhattan ${state}`);
  for (const next of adjacent(state)) ok(m.manhattan(state) <= 1 + m.manhattan(next), `consistent Manhattan ${state}→${next}`);
}
const counts = {};
for (const [name, state] of Object.entries(m.PRESETS)) {
  eq(distance.get(state), m.DEPTHS[name], `${name} is an exact-distance preset`);
  const a = new m.PuzzleSearch(state).run(), d = new m.PuzzleSearch(state, 'zero').run();
  eq(a.cost, m.DEPTHS[name], 'A* exact optimum'); eq(d.cost, a.cost, 'Dijkstra agrees');
  ok(a.expanded <= d.expanded, 'Manhattan saves expansions on this authored preset');
  for (let i = 1; i < a.path.length; i++) ok(adjacent(a.path[i - 1]).includes(a.path[i]), 'returned move is legal');
  eq(a.path[0], state); eq(a.path.at(-1), m.GOAL); counts[name] = { depth: a.cost, astarExpanded: a.expanded, dijkstraExpanded: d.expanded };
}
eq(m.manhattan(m.GOAL), 0); eq(new m.PuzzleSearch(m.GOAL).run().cost, 0);
eq(new m.PuzzleSearch(m.PRESETS.twelve, 'zero', 1).run().limitReached, true, 'bounded run stops');
assert.throws(() => new m.PuzzleSearch('123456870')); checks++;
assert.throws(() => new m.PuzzleSearch('123456770')); checks++;
assert.throws(() => new m.PuzzleSearch(m.GOAL, 'bad')); checks++;
// Independent Bellman relaxation obtains exact h* on the directed graph.
const reverse = { S: Infinity, A: Infinity, B: Infinity, G: 0 };
for (let pass = 0; pass < 4; pass++) for (const [u, v, c] of m.EDGES) reverse[u] = Math.min(reverse[u], c + reverse[v]);
eq(reverse, m.HSTAR, 'independent exact remaining costs');
const detective = {};
for (const preset of Object.keys(m.HVALUES)) {
  const info = m.inspectHeuristic(m.HVALUES[preset]);
  eq(info.admissible, preset !== 'overestimate'); eq(info.consistent, ['zero', 'exact'].includes(preset));
  for (const reopen of [true, false]) {
    const search = new m.DetectiveSearch(preset, reopen).run();
    const expected = preset === 'overestimate' || preset === 'inconsistent' && !reopen ? 6 : 5;
    eq(search.cost, expected, `${preset}; reopen=${reopen}`);
    const sum = search.path.slice(1).reduce((total, v, i) => total + m.EDGES.find(e => e[0] === search.path[i] && e[1] === v)[2], 0);
    eq(sum, search.cost, 'parent chain matches returned cost');
    if (preset === 'inconsistent' && reopen) { eq(search.reopened, 1); eq(search.trace.map(e => e.state), ['S', 'A', 'B', 'A', 'G']); }
    detective[`${preset}/${reopen}`] = { cost: search.cost, expanded: search.expanded, reopened: search.reopened, order: search.trace.map(e => e.state) };
  }
}
// Stateful factory contract: exercise actual EN/KO handlers without pretending to test layout.
const fs = require('node:fs'), vm = require('node:vm');
class StubNode {
  constructor(tag = 'div', attrs = {}, children = []) {
    this.tagName = tag.toUpperCase(); this.attrs = { ...attrs }; this.children = []; this.listeners = {}; this.textContent = ''; this.value = ''; this.checked = false; this.isConnected = true; this.classList = { add() {} }; this.append(...(Array.isArray(children) ? children : [children]));
  }
  append(...nodes) { this.children.push(...nodes.filter(n => n !== undefined && n !== null)); }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(event, handler) { (this.listeners[event] ||= []).push(handler); }
  fire(event, payload = {}) { for (const handler of this.listeners[event] || []) handler(payload); }
}
const factories = new Map(), timers = new Map(); let timerId = 0;
const win = { ArticleLabs: { register(id, factory) { factories.set(id, factory); } }, setTimeout(fn) { timers.set(++timerId, fn); return timerId; }, clearTimeout(id) { timers.delete(id); } };
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-search-astar.js'), 'utf8'), { window: win, console });
function drainTimers() {
  let steps = 0;
  while (timers.size) { ok(steps++ < 1000, 'UI animation terminates'); const [id, fn] = timers.entries().next().value; timers.delete(id); fn(); }
}
function mount(id, ko, factoryMap = factories) {
  const controls = [], buttons = [], metrics = []; let render, reset, share;
  const ctx = {
    root: new StubNode('section'), ui: new StubNode(), view: new StubNode(), status: new StubNode(), ko,
    text: (en, korean) => ko ? korean : en, el: (tag, attrs, children) => new StubNode(tag, attrs, children), svg: (tag, attrs, text) => new StubNode(tag, attrs, text === undefined ? [] : text),
    metric: () => { const output = new StubNode('output'); metrics.push(output); return output; },
    control: (_en, _ko, options) => { const node = new StubNode(); node.value = String(options.value); node.checked = Boolean(options.value); node.options = options; node.initial = { value: node.value, checked: node.checked }; controls.push(node); return node; },
    button: (en, _ko, handler) => { const button = new StubNode('button'); button.handler = handler; buttons.push({ en, button }); return button; },
    table: (headers, rows) => new StubNode('table', {}, [headers, rows]), onChange: handler => { render = handler; handler(); }, onReset: handler => { reset = handler; }, stateLink: (key, hooks) => { eq(key, id); share = hooks; }
  };
  factoryMap.get(id)(ctx);
  const click = label => { const button = buttons.find(x => x.en === label); ok(Boolean(button), `button ${label} exists`); button.button.handler(); };
  return { ctx, controls, metrics, click, render: () => render(), share: () => share, reset: () => { controls.forEach(c => { c.value = c.initial.value; c.checked = c.initial.checked; }); reset(); render(); } };
}
function descendants(node) { return node instanceof StubNode ? [node, ...node.children.flatMap(descendants)] : []; }
for (const ko of [false, true]) {
  const p = mount('search-astar-tiles', ko);
  eq(p.share().read().board, m.PRESETS.eight); p.click('Finish search'); drainTimers(); eq(p.metrics[2].textContent, '8 / 8');
  p.controls[1].value = 'zero'; p.render(); p.click('Finish search'); drainTimers(); eq(p.metrics[1].textContent, '155 / 117'); eq(p.metrics[2].textContent, '8 / 0', 'Dijkstra readout uses the selected zero heuristic');
  p.controls[0].value = 'four'; p.render(); eq(p.share().read().moves, 0); eq(p.share().read().board, m.PRESETS.four);
  const board = descendants(p.ctx.view).find(n => n.attrs.class === 'sa-board');
  let prevented = false; board.fire('keydown', { key: 'ArrowRight', preventDefault() { prevented = true; } });
  ok(prevented, 'arrow keys do not scroll the page'); eq(p.share().read().board, '123560478'); eq(p.share().read().moves, 1);
  p.click('Hint: one move'); drainTimers(); ok(board.children.some(n => /sa-hint/.test(n.className || '')), 'hint outlines a legal tile');
  p.click('Play solver solution'); drainTimers(); eq(p.share().read().board, '123560478', 'playback never mutates manual board'); eq(p.share().read().moves, 1);
  const saved = JSON.parse(JSON.stringify(p.share().read())), before = JSON.stringify(p.share().read());
  eq(p.share().write({ ...saved, moves: -1 }), false); eq(JSON.stringify(p.share().read()), before, 'invalid shared settings do not mutate state');
  assert.throws(() => p.share().write({ ...saved, board: '123456870' })); checks++; eq(JSON.stringify(p.share().read()), before);
  p.reset(); eq(p.share().read().board, m.PRESETS.eight); eq(p.share().read().moves, 0); eq(p.controls[1].value, 'manhattan');
  ok(p.share().write(saved), 'valid shared settings restore'); eq(p.share().read().board, saved.board);
  p.click('Run search'); p.reset(); eq(timers.size, 0, 'Reset cancels pending search');
  const d = mount('search-astar-detective', ko);
  d.click('Finish'); eq(d.metrics[0].textContent, '5 / 5'); eq(d.metrics[1].textContent, '4 / 1');
  d.controls[1].checked = false; d.render(); d.click('Finish'); eq(d.metrics[0].textContent, '6 / 5');
  d.controls[0].value = 'overestimate'; d.controls[1].checked = true; d.render(); d.click('Finish'); eq(d.metrics[0].textContent, '6 / 5');
  d.controls[0].value = 'exact'; d.render(); d.click('Run'); drainTimers(); eq(d.metrics[0].textContent, '5 / 5');
  d.click('Restart trace');
  const guesses = descendants(d.ctx.view).filter(n => n.tagName === 'BUTTON');
  guesses[0].fire('click'); ok(/Correct|맞았습니다/.test(descendants(d.ctx.view).find(n => n.attrs.class === 'sa-announcement').textContent), 'prediction reveals the selected state');
  const settings = JSON.parse(JSON.stringify(d.share().read())); eq(d.share().write({ ...settings, reopen: 'yes' }), false); eq(d.share().read().reopen, true);
  d.reset(); eq(d.controls[0].value, 'inconsistent'); eq(d.metrics[1].textContent, '0 / 0');
}
const noTimerFactories = new Map();
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-search-astar.js'), 'utf8'), { window: { ArticleLabs: { register(id, factory) { noTimerFactories.set(id, factory); } } }, console });
for (const ko of [false, true]) {
  const p = mount('search-astar-tiles', ko, noTimerFactories); delete p.ctx.root.isConnected;
  p.click('Run search'); eq(p.metrics[2].textContent, '8 / 8', 'timer-less Run completes through the bounded engine');
  p.click('Hint: one move'); p.click('Play solver solution'); eq(p.share().read().board, m.PRESETS.eight, 'timer-less playback preserves the manual board');
  p.reset(); p.controls[1].value = 'zero'; p.render(); p.click('Finish search'); eq(p.metrics[2].textContent, '8 / 0');
  const d = mount('search-astar-detective', ko, noTimerFactories); delete d.ctx.root.isConnected;
  d.click('Run'); eq(d.metrics[0].textContent, '5 / 5', 'timer-less detective Run completes');
}
console.log(JSON.stringify({ checks, exactDistanceStates: distance.size, puzzle: counts, detective, bilingualFactoryContracts: 8 }, null, 2));
