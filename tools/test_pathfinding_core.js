'use strict';
/* Independent shortest-path checks. No DOM, network, dependencies or article mutation. */
const assert = require('node:assert/strict');
const { Search, makePreset, neighbors, heuristic, runSearch, parsePriority, priorityTests, sameCost, validateMap } = require('../assets/js/article-labs-pathfinding.js');
let assertions = 0;
const check = (condition, message) => { assertions++; assert(condition, message); };
check(!sameCost(Infinity, 1) && !sameCost(1, Infinity) && sameCost(Infinity, Infinity), 'Finite and unreachable costs cannot match');
// Bellman–Ford relaxes every directed edge; it shares neither heap nor search state.
function bellmanFord(map, movement, reverse = false) {
  const n = map.cells.length, distance = Array(n).fill(Infinity); distance[reverse ? map.goal : map.start] = 0;
  for (let pass = 0; pass < n - 1; pass++) {
    let changed = false;
    for (let u = 0; u < n; u++) if (map.cells[u]) for (const [v, cost] of neighbors(map, u, movement)) {
      const a = reverse ? v : u, b = reverse ? u : v;
      if (distance[a] + cost < distance[b]) { distance[b] = distance[a] + cost; changed = true; }
    }
    if (!changed) break;
  }
  return distance;
}
function pathCost(map, result, movement) {
  let total = 0;
  for (let i = 1; i < result.path.length; i++) {
    const edge = neighbors(map, result.path[i - 1], movement).find(([v]) => v === result.path[i]);
    check(Boolean(edge), 'Every path edge must exist'); total += edge[1];
  }
  return total;
}
let seed = 32048;
function random() { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; }
for (let trial = 0; trial < 120; trial++) {
  const width = 3 + Math.floor(random() * 6), height = 3 + Math.floor(random() * 6), n = width * height;
  const map = { width, height, cells: Array.from({ length: n }, () => { const r = random(); return r < .22 ? 0 : r < .43 ? 5 : 1; }), start: 0, goal: n - 1 };
  map.cells[map.start] = 1; map.cells[map.goal] = 1; validateMap(map);
  for (const movement of [4, 8]) {
    const exact = bellmanFord(map, movement)[map.goal], back = bellmanFord(map, movement, true);
    const baseH = movement === 4 ? 'manhattan' : 'octile';
    for (const kind of [baseH, 'euclidean', 'zero']) {
      const result = runSearch(map, { algorithm: 'astar', movement, heuristic: kind });
      check(sameCost(result.cost, exact), `A* versus Bellman–Ford trial ${trial}/${movement}/${kind}`);
      if (result.found) check(sameCost(pathCost(map, result, movement), exact), 'Reconstructed route cost');
      for (let u = 0; u < n; u++) if (map.cells[u]) {
        const h = heuristic(map, u, kind); check(h <= back[u] + 1e-9, 'Base heuristic admissibility');
        for (const [v, c] of neighbors(map, u, movement)) check(h <= c + heuristic(map, v, kind) + 1e-9, 'Base heuristic consistency');
      }
    }
    const dijkstra = runSearch(map, { algorithm: 'dijkstra', movement, heuristic: baseH });
    check(sameCost(dijkstra.cost, exact), 'Dijkstra versus Bellman–Ford');
    // Random downward perturbations preserve admissibility but need not be consistent.
    const hValues = back.map(h => Number.isFinite(h) ? h * random() : 0); hValues[map.goal] = 0;
    check(sameCost(runSearch(map, { movement, heuristic: baseH, hValues }).cost, exact), 'Admissible inconsistent heuristic with reopening');
    for (const weight of [1.25, 2, 3]) {
      const weighted = runSearch(map, { movement, heuristic: baseH, weight });
      check(!Number.isFinite(exact) ? !weighted.found : weighted.cost <= weight * exact + 1e-8, 'Weighted A* bounded suboptimality');
    }
  }
}
const graph = { start: 0, goal: 3, edges: [[[1, 3], [2, 1]], [[3, 3]], [[1, 1], [3, 100]], []] };
const reopen = runSearch(graph, { hValues: [5, 0, 4, 0] });
check(reopen.cost === 5 && reopen.reopened === 1, 'Known inconsistent h requires reopening');
assert.deepEqual(reopen.selectedOrder, [0, 1, 2, 1, 3]); assertions++;
const staleGraph = { start: 0, goal: 3, edges: [[[1, 5], [2, 1]], [], [[1, 1]], []] };
const stale = runSearch(staleGraph, { algorithm: 'dijkstra' });
check(!stale.found && stale.stale === 1 && stale.expanded === 3, 'Stale entries do not expand');
const branch = makePreset('branch'), dfs = runSearch(branch, { algorithm: 'dfs' }), bfs = runSearch(branch, { algorithm: 'bfs' });
check(dfs.cost > bfs.cost && bfs.cost === 6, 'DFS first path differs from BFS shallowest path');
const cheap = runSearch(makePreset('weighted'), { algorithm: 'dijkstra' }), fewest = runSearch(makePreset('weighted'), { algorithm: 'bfs' });
check(fewest.cost === 38 && cheap.cost === 12, 'Weighted road separates fewest steps and least cost');
for (const name of ['branch', 'weighted', 'detour', 'blocked', 'open']) {
  for (const algorithm of ['dfs', 'bfs', 'dijkstra', 'greedy', 'astar']) {
    const a = runSearch(makePreset(name), { algorithm }), b = runSearch(makePreset(name), { algorithm });
    assert.deepEqual(a.selectedOrder, b.selectedOrder); assertions++;
  }
}
const same = makePreset('open'); same.goal = same.start;
for (const algorithm of ['dfs', 'bfs', 'dijkstra', 'greedy', 'astar']) {
  const r = runSearch(same, { algorithm }); check(r.cost === 0 && r.path.length === 1 && r.expanded === 0, 'Start equals goal');
}
const corner = { width: 2, height: 2, start: 0, goal: 3, cells: [1, 0, 0, 1] };
check(neighbors(corner, 0, 8).length === 0, 'Corner cutting forbidden');
check(!runSearch(corner, { movement: 8, heuristic: 'octile' }).found, 'Blocked diagonal is not a path');
assert.throws(() => runSearch(makePreset('open'), { movement: 8, heuristic: 'manhattan' })); assertions++;
check(parsePriority('g + 2*h')(3, 4) === 11, 'Arithmetic parse');
check(parsePriority('-(g-h)/2')(3, 5) === 1, 'Unary and parentheses');
check(parsePriority('.5*g + 1.5*h')(4, 2) === 5, 'Decimal parse');
for (const expression of ['window', 'g=1', 'Math.min(g,h)', 'g;h', 'g**h', 'g()', 'g[0]', 'g.constructor', 'alert(1)', '', 'g h', '1e6', '('.repeat(17) + 'g' + ')'.repeat(17), '1'.repeat(81)]) {
  assert.throws(() => parsePriority(expression), undefined, expression); assertions++;
}
for (const expression of ['1/(g-g)', '1000000*1000000*1000000']) { assert.throws(() => parsePriority(expression)(2, 1)); assertions++; }
check(priorityTests('g+h').every(x => x.optimal), 'g+h solves bounded examples optimally');
check(priorityTests('h').some(x => !x.optimal), 'Tests expose greedy counterexample');
const figure = require('../assets/img/posts/2026-10-24-astar-from-scratch/figure-data.json');
const comparison = makePreset('comparison');
assert.deepEqual(comparison.cells, figure.grid.flatMap(row => [...row].map(cell => cell === '#' ? 0 : cell === '5' ? 5 : 1))); assertions++;
assert.deepEqual([comparison.start % comparison.width, Math.floor(comparison.start / comparison.width)], figure.start); assertions++;
assert.deepEqual([comparison.goal % comparison.width, Math.floor(comparison.goal / comparison.width)], figure.goal); assertions++;
const coordinates = ids => ids.map(u => [u % comparison.width, Math.floor(u / comparison.width)]);
for (const algorithm of ['bfs', 'dijkstra', 'greedy', 'astar']) {
  const result = runSearch(comparison, { algorithm }), expected = figure.results[algorithm];
  check(result.cost === expected.cost && result.expanded === expected.expanded && result.path.length - 1 === expected.steps, 'Figure 7 costs and counts: ' + algorithm);
  assert.deepEqual(coordinates(result.path), expected.path); assertions++;
  assert.deepEqual(coordinates(result.expandedOrder), expected.expanded_nodes); assertions++;
  assert.deepEqual(coordinates(result.selectedOrder), expected.pop_order); assertions++;
}
for (const expected of figure.weighted_astar.filter(row => row.weight <= 3)) {
  const result = runSearch(comparison, { algorithm: 'astar', weight: expected.weight });
  check(result.cost === expected.cost && result.expanded === expected.expanded && result.path.length - 1 === expected.steps, 'Figure 8 weighted costs and counts: ' + expected.weight);
  assert.deepEqual(coordinates(result.path), expected.path); assertions++;
  assert.deepEqual(coordinates(result.expandedOrder), expected.expanded_nodes); assertions++;
  assert.deepEqual(coordinates(result.selectedOrder), expected.pop_order); assertions++;
}
console.log(JSON.stringify({ status: 'passed', assertions, random_maps: 120, movements: [4, 8], independent_reference: 'Bellman–Ford', cases: ['optimality', 'path-cost reconstruction', 'admissibility', 'consistency', 'reopening', 'stale heap', 'weighted bound', 'DFS/BFS contrast', 'corner cutting', 'start=goal', 'deterministic ties', 'parser rejection'] }, null, 2));
