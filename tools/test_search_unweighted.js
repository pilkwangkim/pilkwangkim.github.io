'use strict';
/* Independent finite-model checks plus a DOM-contract stub; no layout or remote claims. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const a = require('../assets/js/article-labs-search-unweighted.js');
let assertions = 0;
const check = (value, message) => { assert(value, message); assertions++; };
const equal = (actual, expected, message = '') => { assert.deepEqual(actual, expected, message); assertions++; };

// A union–find verifies a spanning tree without using the generator's traversal.
for (const [width, height] of [[4, 4], [6, 4], [7, 5]]) for (let seed = 0; seed < 100; seed++) {
  const maze = new a.MazeCarver(width, height, seed), n = width * height;
  while (!maze.done) {
    const oldVisited = maze.visited.size, oldEdges = maze.passages.size, oldDepth = maze.stack.length;
    maze.step();
    check(maze.stack.every(u => maze.visited.has(u)), 'Every stack cell is visited');
    equal(new Set(maze.stack).size, maze.stack.length, 'DFS path has no duplicate stack entries');
    if (maze.event.type === 'carve') { equal(maze.visited.size, oldVisited + 1); equal(maze.passages.size, oldEdges + 1); equal(maze.stack.length, oldDepth + 1); }
    else { equal(maze.visited.size, oldVisited); equal(maze.passages.size, oldEdges); equal(maze.stack.length, oldDepth - 1); }
    for (let i = 1; i < maze.stack.length; i++) check(maze.linked(maze.stack[i - 1], maze.stack[i]), 'Stack traces carved passages');
  }
  equal(maze.visited.size, n); equal(maze.passages.size, n - 1); equal(maze.steps, 2 * n - 1); equal(maze.backtracks, n);
  const parent = Array.from({ length: n }, (_, i) => i);
  const root = u => { while (parent[u] !== u) u = parent[u]; return u; };
  for (const key of maze.passages) {
    const [u, v] = key.split(':').map(Number), distance = Math.abs(u % width - v % width) + Math.abs(Math.floor(u / width) - Math.floor(v / width));
    equal(distance, 1, 'Every passage joins orthogonal neighbors'); check(root(u) !== root(v), 'No cycles'); parent[root(u)] = root(v);
  }
  equal(new Set(parent.map((_, u) => root(u))).size, 1, 'Spanning tree is connected');
  const route = a.mazeRoute(maze); equal(route[0], 0); equal(route.at(-1), n - 1); equal(new Set(route).size, route.length);
  for (let i = 1; i < route.length; i++) check(maze.linked(route[i - 1], route[i]), 'Route never crosses a wall');
  equal([...new a.MazeCarver(width, height, seed).run().passages], [...maze.passages], 'Seed repeatability');
  const old = maze.steps; maze.step(); equal(maze.steps, old, 'Completed generator is idempotent');
}
equal(a.mazeRoute(new a.MazeCarver(6, 4, 17).run()), [0, 1, 7, 13, 14, 8, 9, 3, 4, 5, 11, 10, 16, 17, 23]);
for (const args of [[1, 4, 0], [4, 20, 0], [4, 4, -1], [4, 4, .2], [4, 4, 1000000]]) { assert.throws(() => new a.MazeCarver(...args)); assertions++; }

// Exhaustive permutations are independent of the solver's candidate/pruning tree.
function bruteQueens(n) {
  const out = [];
  function permute(rows, remaining) {
    if (!remaining.length) {
      let safe = true;
      for (let c = 0; c < n; c++) for (let d = c + 1; d < n; d++) if (Math.abs(rows[c] - rows[d]) === d - c) safe = false;
      if (safe) out.push(rows); return;
    }
    for (const r of remaining) permute([...rows, r], remaining.filter(v => v !== r));
  }
  permute([], Array.from({ length: n }, (_, i) => i)); return out.map(x => x.join(',')).sort();
}
for (const [n, count] of [[4, 2], [5, 10], [6, 4]]) {
  const solver = new a.QueensDFS(n);
  while (!solver.done) {
    solver.step();
    check(a.queensSafe(solver.rows), 'Every partial state remains nonattacking'); check(solver.rows.length <= n);
    if (solver.event.type === 'solution') { equal(solver.event.rows.length, n); check(a.queensSafe(solver.event.rows)); }
  }
  equal(solver.solutions.length, count); equal(solver.solutions.map(x => x.join(',')).sort(), bruteQueens(n));
  equal(new Set(solver.solutions.map(x => x.join(','))).size, count, 'No duplicate solutions');
  check(solver.pruned > 0 && solver.pruned < solver.checked); const steps = solver.steps; solver.step(); equal(solver.steps, steps);
}
equal(a.queenConflict([1, 3], 0), null); check(Boolean(a.queenConflict([1, 3], 2))); equal(a.queenConflict([1], 1), { type: 'row', column: 0 });
for (const n of [0, 3, 7, 4.5]) { assert.throws(() => new a.QueensDFS(n)); assertions++; }

// Floyd–Warshall builds its own adjacency by character comparisons.
function wordDistances(preset) {
  const words = preset.words, n = words.length, d = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 0 : Infinity));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let mismatch = 0; for (let k = 0; k < words[i].length; k++) mismatch += words[i][k] !== words[j][k];
    if (mismatch === 1) d[i][j] = 1;
  }
  for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) d[i][j] = Math.min(d[i][j], d[i][k] + d[k][j]);
  return d;
}
for (const [index, cost] of [[0, 4], [1, 5], [2, 3]]) {
  const p = a.wordPresets[index], frozen = JSON.stringify(p), result = a.wordBFS(p).run(), distance = wordDistances(p), startIndex = p.words.indexOf(p.start);
  equal(new Set(p.words).size, p.words.length); check(p.words.includes(p.start) && p.words.includes(p.goal)); equal(result.cost, cost); equal(result.cost, distance[startIndex][p.words.indexOf(p.goal)]);
  check(result.path.every(w => p.words.includes(w))); equal(new Set(result.queue).size, result.queue.length, 'Discovered words enqueued once');
  for (let i = 1; i < result.path.length; i++) check(a.wordNeighbors(p.words, result.path[i - 1]).includes(result.path[i]));
  result.depth.forEach((depth, word) => equal(depth, distance[startIndex][p.words.indexOf(word)], 'Discovered depth is shortest distance'));
  const depths = result.expanded.map(word => result.depth.get(word)); check(depths.every((v, i) => i === 0 || v >= depths[i - 1]), 'BFS expansions are layer ordered');
  equal(JSON.stringify(p), frozen, 'Dictionary is not mutated');
}
equal(a.wordBFS(a.wordPresets[0]).run().path, ['COLD', 'CORD', 'CARD', 'WARD', 'WARM']);
equal(a.wordNeighbors(['A', 'A', 'B'], 'A'), ['B']);

// Enumerate every rectangular integer state and independently derive physical actions.
function physicalMoves(A, B, state) {
  const [x, y] = state, moves = [[A, y], [x, B], [0, y], [x, 0]];
  let ax = x, by = y; while (ax > 0 && by < B) { ax--; by++; } moves.push([ax, by]);
  ax = x; by = y; while (by > 0 && ax < A) { by--; ax++; } moves.push([ax, by]);
  return moves.filter(s => s[0] !== x || s[1] !== y);
}
function independentJugDistance(A, B, target) {
  const states = [], key = s => s.join(','); for (let x = 0; x <= A; x++) for (let y = 0; y <= B; y++) states.push([x, y]);
  const distances = new Map(states.map(s => [key(s), Infinity])); distances.set('0,0', 0);
  for (let pass = 0; pass < states.length; pass++) {
    let changed = false;
    for (const s of states) for (const next of physicalMoves(A, B, s)) if (distances.get(key(s)) + 1 < distances.get(key(next))) { distances.set(key(next), distances.get(key(s)) + 1); changed = true; }
    if (!changed) break;
  }
  return Math.min(...states.filter(s => s.includes(target)).map(s => distances.get(key(s))));
}
for (let A = 1; A <= 8; A++) for (let B = 1; B <= 8; B++) for (let target = 1; target <= Math.max(A, B) + 1; target++) {
  const p = { a: A, b: B, target }, result = a.jugBFS(p).run(); equal(result.cost, independentJugDistance(A, B, target), 'Jug BFS versus independent unit-transfer reference');
  equal(result.path.length ? result.actions.length : 0, result.found ? result.cost : 0);
  for (let x = 0; x <= A; x++) for (let y = 0; y <= B; y++) {
    const s = [x, y], transitions = a.jugMoves(p, s);
    equal(transitions.map(t => t.state.join(',')).sort(), physicalMoves(A, B, s).map(t => t.join(',')).sort(), 'Every action obeys fill/empty/pour rules');
    check(transitions.every(t => t.state[0] !== x || t.state[1] !== y), 'No self-loop actions');
    for (const t of transitions) {
      check(t.state[0] >= 0 && t.state[0] <= A && t.state[1] >= 0 && t.state[1] <= B);
      if (t.action.startsWith('pour')) equal(t.state[0] + t.state[1], x + y, 'Pour conserves water');
    }
  }
  for (let i = 1; i < result.path.length; i++) check(a.jugMoves(p, result.path[i - 1]).some(t => t.action === result.actions[i - 1] && t.state.join(',') === result.path[i].join(',')), 'Reconstructed action is legal');
}
equal(a.jugBFS(a.jugPresets[0]).run().path, [[0, 0], [0, 5], [3, 2], [0, 2], [2, 0], [2, 5], [3, 4]]);
equal(a.jugBFS(a.jugPresets[1]).run().cost, 8); equal(a.jugBFS(a.jugPresets[2]).run().cost, Infinity); equal(a.jugBFS(a.jugPresets[2]).run().states.size, 6);
for (const state of [[-1, 0], [0, 6], [0.5, 2]]) { assert.throws(() => a.jugMoves(a.jugPresets[0], state)); assertions++; }
const immediate = new a.GraphBFS('A', w => w === 'A', () => []); immediate.run(); equal(immediate.cost, 0); equal(immediate.expanded.length, 0);
const impossible = new a.GraphBFS('A', w => w === 'B', () => []); impossible.run(); equal(impossible.cost, Infinity);

// DOM-contract tests exercise native controls, real activity handlers and Reset.
class NodeStub {
  constructor(tag = 'div', attrs = {}, children = []) {
    this.tagName = tag.toUpperCase(); this.attrs = { ...attrs }; this.children = []; this.listeners = {}; this.textContent = ''; this.value = attrs.value || ''; this.disabled = Boolean(attrs.disabled); this.isConnected = true; this.style = { values: {}, setProperty(k, v) { this.values[k] = String(v); } }; this.classList = { add() {} }; this.append(...(Array.isArray(children) ? children : [children]));
  }
  append(...nodes) { this.children.push(...nodes.filter(n => n !== null && n !== undefined)); }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  addEventListener(event, fn) { (this.listeners[event] ||= []).push(fn); }
  fire(event, data = {}) { if (this.disabled) return; for (const fn of this.listeners[event] || []) fn(data); }
  focus() { this.focused = true; }
}
const factories = new Map(), timers = new Map(); let timerId = 0;
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-search-unweighted.js'), 'utf8'), { window: { ArticleLabs: { register(id, fn) { factories.set(id, fn); } }, setInterval(fn) { timers.set(++timerId, fn); return timerId; }, clearInterval(id) { timers.delete(id); } }, console });
function mount(id, ko) {
  const controls = [], buttons = new Map(), metrics = []; let render, onReset;
  const ctx = {
    root: new NodeStub('section'), ui: new NodeStub(), view: new NodeStub(), status: new NodeStub(), ko,
    text: (en, korean) => ko ? korean : en, el: (tag, attrs, children) => new NodeStub(tag, attrs, children), svg: (tag, attrs, text) => { const node = new NodeStub(tag, attrs); node.textContent = text || ''; return node; },
    metric: () => { const node = new NodeStub('output'); metrics.push(node); return node; },
    control: (_en, _ko, opts) => { const node = new NodeStub(opts.type === 'select' ? 'select' : 'input', opts); node.value = String(opts.value); controls.push({ node, initial: node.value }); return node; },
    button: (en, _ko, fn) => { const node = new NodeStub('button'); node.addEventListener('click', fn); node.textContent = en; buttons.set(en, node); return node; },
    table: (headers, rows) => new NodeStub('table', {}, [headers, rows]), onChange: fn => { render = fn; fn(); }, onReset: fn => { onReset = fn; }, checkpointState: config => { ctx.checkpoint = config; }
  };
  factories.get(id)(ctx);
  return { ctx, metrics, controls: controls.map(c => c.node), click: name => buttons.get(name).fire('click'), buttons, render: () => render(), reset: () => { controls.forEach(c => c.node.value = c.initial); onReset(); render(); } };
}
function findClass(node, cls) {
  if (!(node instanceof NodeStub)) return null;
  if ((node.attrs.class || '').split(' ').includes(cls)) return node;
  for (const child of node.children) { const found = findClass(child, cls); if (found) return found; } return null;
}
for (const ko of [false, true]) {
  const maze = mount('search-dfs-maze', ko), board = findClass(maze.ctx.view, 'su-maze-board');
  equal(maze.metrics[0].textContent, '1 / 24'); maze.click('Step DFS'); equal(maze.metrics[0].textContent, '2 / 24');
  maze.click('Run'); equal(timers.size, 1); maze.reset(); equal(timers.size, 0); equal(maze.metrics[0].textContent, '1 / 24'); equal(maze.metrics[2].textContent, '— / —');
  maze.click('Finish maze'); equal(maze.metrics[2].textContent, '0 / 14');
  for (const u of a.mazeRoute(new a.MazeCarver(6, 4, 17).run()).slice(1)) board.children[u].fire('click');
  equal(maze.metrics[2].textContent, '14 / 14'); check(/Reached|도착/.test(maze.ctx.status.textContent));
  board.children[0].fire('keydown', { key: 'ArrowRight', preventDefault() {} }); check(board.children[1].focused, 'Arrow navigation focuses neighboring cell');
  maze.reset(); maze.controls[0].value = '7x5'; maze.render(); equal(maze.metrics[0].textContent, '1 / 35'); maze.click('Run'); maze.click('Run'); equal(timers.size, 0, 'Pause cancels maze interval');
  maze.reset(); equal(maze.controls[0].value, '6x4');

  const queens = mount('search-dfs-queens', ko), qboard = findClass(queens.ctx.view, 'su-queen-board');
  for (const [c, r] of [1, 3, 0, 2].entries()) qboard.children[r * 4 + c].fire('click');
  equal(queens.metrics[0].textContent, '4 / 4'); check(/Solved|성공/.test(queens.ctx.view.children[0].textContent));
  queens.click('Your turn'); qboard.children[0].fire('click'); qboard.children[1].fire('click'); equal(queens.metrics[0].textContent, '1 / 4'); check(/Prune|겹칩니다/.test(queens.ctx.status.textContent));
  queens.click('Step DFS'); equal(queens.metrics[1].textContent, '1 / 0'); queens.click('Run'); equal(timers.size, 1); queens.reset(); equal(timers.size, 0); equal(queens.metrics[0].textContent, '0 / 4'); equal(queens.metrics[1].textContent, '0 / 0');
  queens.click('Find all solutions'); equal(queens.metrics[2].textContent, '2 / 16'); equal(queens.metrics[0].textContent, '4 / 4', 'A stored complete placement stays visible after enumeration');
  const gallery = findClass(queens.ctx.view, 'su-saved-solutions'), galleryOptions = findClass(queens.ctx.view, 'su-solution-grid');
  check(!gallery.hidden); equal(galleryOptions.children.length, 2); check(/Saved solution 1|저장한 해 1/.test(queens.ctx.view.children[0].textContent));
  const countsBeforeGallery = queens.metrics.slice(1).map(output => output.textContent);
  galleryOptions.children[1].fire('click'); equal(queens.metrics.slice(1).map(output => output.textContent), countsBeforeGallery, 'Viewing a saved solution never advances DFS');
  check(/Saved solution 2|저장한 해 2/.test(queens.ctx.view.children[0].textContent));
  equal(qboard.children.map((cell, u) => cell.textContent === '♛' ? [Math.floor(u / 4), u % 4] : null).filter(Boolean), [[0, 1], [1, 3], [2, 0], [3, 2]], 'The second N=4 solution appears on the read-only board');
  queens.click('Step DFS'); equal(queens.metrics.slice(1).map(output => output.textContent), countsBeforeGallery); equal(queens.metrics[0].textContent, '4 / 4', 'Step on a completed search retains its saved solution');
  queens.click('Your turn'); check(gallery.hidden); equal(galleryOptions.children.length, 0); equal(queens.metrics[0].textContent, '0 / 4', 'Manual mode clears saved-view selection');
  queens.click('Step DFS'); let findFirst = 0; while (galleryOptions.children.length === 0 && findFirst++ < 200) queens.click('Step DFS');
  equal(galleryOptions.children.length, 1); galleryOptions.children[0].fire('click'); check(/Saved solution 1|저장한 해 1/.test(queens.ctx.view.children[0].textContent));
  queens.click('Step DFS'); check(/DFS state/.test(queens.ctx.view.children[0].textContent), 'Step resumes the real partial DFS state from a stored-solution view');
  queens.controls[0].value = '5'; queens.render(); check(gallery.hidden); equal(galleryOptions.children.length, 0); queens.click('Find all solutions'); equal(galleryOptions.children.length, 10, 'Saved gallery stays bounded to the N=5 ten solutions');
  queens.controls[0].value = '6'; queens.render(); check(gallery.hidden); queens.click('Find all solutions'); equal(queens.metrics[2].textContent, '4 / 152'); equal(galleryOptions.children.length, 4);
  queens.reset(); equal(queens.controls[0].value, '4'); check(gallery.hidden); equal(galleryOptions.children.length, 0);

  const words = mount('search-bfs-words', ko), options = findClass(words.ctx.view, 'su-word-options');
  for (const word of ['CORD', 'CARD', 'WARD', 'WARM']) options.children.find(node => node.children.includes(word)).fire('click');
  equal(words.metrics[0].textContent, '4 / 4'); check(/Shortest|최단/.test(words.ctx.status.textContent));
  words.click('Step BFS'); equal(words.metrics[1].textContent, '1 / 3'); words.click('Run'); equal(timers.size, 1); words.reset(); equal(timers.size, 0); equal(words.metrics[0].textContent, '0 / 4'); equal(words.metrics[1].textContent, '0 / 1');
  words.click('Reveal shortest chain'); equal(words.metrics[1].textContent, '16 / 4');
  words.controls[0].value = 'cat-dog'; words.render(); equal(words.metrics[0].textContent, '0 / 3'); words.reset(); equal(words.controls[0].value, 'cold-warm');

  const jugs = mount('search-bfs-jugs', ko), actions = findClass(jugs.ctx.view, 'su-jug-actions');
  equal(jugs.metrics[0].textContent, '0 / 6'); equal(jugs.metrics[1].textContent, '(0, 0) L'); check(actions.children[2].disabled && actions.children[4].disabled);
  // Physical action order in the UI is Fill A, Fill B, Empty A, Empty B, Pour A→B, Pour B→A.
  for (const index of [1, 5, 2, 5, 1, 5]) actions.children[index].fire('click');
  equal(jugs.metrics[0].textContent, '6 / 6'); equal(jugs.metrics[1].textContent, '(3, 4) L'); check(/minimum|최소/.test(jugs.ctx.status.textContent));
  jugs.click('Run'); equal(timers.size, 1); jugs.reset(); equal(timers.size, 0); equal(jugs.metrics[0].textContent, '0 / 6');
  jugs.controls[0].value = '2-4-3'; jugs.render(); jugs.click('Reveal shortest plan'); equal(jugs.metrics[0].textContent, '0 / ∞'); equal(jugs.metrics[2].textContent, '6 / 0'); jugs.reset(); equal(jugs.controls[0].value, '3-5-4');
}
equal(timers.size, 0);
console.log(JSON.stringify({ status: 'passed', assertions, maze_seeds_per_size: 100, maze_sizes: [[4, 4], [6, 4], [7, 5]], queens_n: [4, 5, 6], word_presets: 3, jug_capacities: 'all 1..8', independent_references: ['union–find tree verification', 'all queen row permutations', 'Floyd–Warshall word graph', 'integer-transfer jug relaxation'], dom_contract: 'EN/KO manual wins, invalid queen move, preset changes, timers, Pause, Reset clears custom state, arrow navigation, bounded saved-queen gallery, solver-state/count preservation and resume', scope: 'Browser geometry requires separate real-screen QA.' }, null, 2));
