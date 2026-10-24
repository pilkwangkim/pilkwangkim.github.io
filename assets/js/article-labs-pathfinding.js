/* Reproducible graph-search teaching models. No external services or code evaluation. */
(function () {
  'use strict';
  const EPS = 1e-10;
  const sameCost = (a, b) => a === b || Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= EPS * Math.max(1, Math.abs(a), Math.abs(b));
  class Heap {
    constructor(compare) { this.items = []; this.compare = compare; }
    push(value) {
      const a = this.items; a.push(value); let i = a.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (this.compare(a[p], value) <= 0) break; a[i] = a[p]; i = p; } a[i] = value;
    }
    pop() {
      const a = this.items, first = a[0], last = a.pop();
      if (a.length) { let i = 0; while (true) { let child = i * 2 + 1; if (child >= a.length) break; if (child + 1 < a.length && this.compare(a[child + 1], a[child]) < 0) child++; if (this.compare(last, a[child]) <= 0) break; a[i] = a[child]; i = child; } a[i] = last; }
      return first;
    }
    get length() { return this.items.length; }
  }
  function makePreset(name = 'weighted') {
    const width = 13, height = 9, cells = Array(width * height).fill(1);
    const id = (x, y) => y * width + x;
    let start = id(1, 4), goal = id(11, 4);
    if (name === 'comparison') {
      // Same authored map as the article's Figure 7 and its recorded figure-data.json.
      const rows = ['#############', '#...........#', '#..###.###..#', '#..#.....#..#', '#S.5555555.G#', '#..#..5..#..#', '#..###.###..#', '#...........#', '#############'];
      rows.forEach((row, y) => [...row].forEach((cell, x) => { cells[id(x, y)] = cell === '#' ? 0 : cell === '5' ? 5 : 1; }));
    }
    else if (name === 'weighted') { for (let x = 3; x <= 9; x++) cells[id(x, 4)] = 5; }
    else if (name === 'detour') { for (let y = 0; y < 8; y++) cells[id(6, y)] = 0; for (let y = 1; y < 7; y++) cells[id(9, y)] = 5; }
    else if (name === 'blocked') { for (let y = 0; y < height; y++) cells[id(6, y)] = 0; }
    else if (name === 'branch') {
      cells.fill(0); start = id(1, 1); goal = id(11, 1);
      // A short upper path competes with a long first-discovered branch on the right.
      for (let x = 1; x <= 11; x++) { cells[id(x, 1)] = 1; cells[id(x, 7)] = 1; }
      for (let y = 1; y <= 7; y++) { cells[id(1, y)] = 1; cells[id(11, y)] = 1; }
      for (let x = 3; x <= 9; x += 2) for (let y = 2; y <= 5; y++) cells[id(x, y)] = 1;
      // From upper-left to lower-left, the right-first DFS prefers the long perimeter.
      start = id(1, 1); goal = id(1, 7);
    } else if (name !== 'open') throw new Error('Unknown preset');
    return { width, height, cells, start, goal };
  }
  function validateMap(map) {
    if (!map || !Number.isInteger(map.width) || !Number.isInteger(map.height) || map.width < 1 || map.height < 1 || map.width * map.height > 400 || !Array.isArray(map.cells) || map.cells.length !== map.width * map.height || !map.cells.every(v => v === 0 || v === 1 || v === 5) || ![map.start, map.goal].every(v => Number.isInteger(v) && v >= 0 && v < map.cells.length && map.cells[v] > 0)) throw new Error('Invalid map');
    return true;
  }
  function neighbors(map, u, movement = 4) {
    const x = u % map.width, y = Math.floor(u / map.width), out = [];
    const directions = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    if (movement === 8) directions.push([1, 1], [-1, 1], [-1, -1], [1, -1]);
    const open = (a, b) => a >= 0 && b >= 0 && a < map.width && b < map.height && map.cells[b * map.width + a] > 0;
    for (const [dx, dy] of directions) {
      if (!open(x + dx, y + dy)) continue;
      // Both orthogonal neighbors must be open: no diagonal squeezing through walls.
      if (dx && dy && (!open(x + dx, y) || !open(x, y + dy))) continue;
      const v = (y + dy) * map.width + x + dx;
      out.push([v, (dx && dy ? Math.SQRT2 : 1) * map.cells[v]]);
    }
    return out;
  }
  function heuristic(map, u, kind = 'manhattan') {
    const dx = Math.abs(u % map.width - map.goal % map.width), dy = Math.abs(Math.floor(u / map.width) - Math.floor(map.goal / map.width));
    if (kind === 'zero') return 0;
    if (kind === 'euclidean') return Math.hypot(dx, dy);
    if (kind === 'octile') return Math.max(dx, dy) + (Math.SQRT2 - 1) * Math.min(dx, dy);
    if (kind === 'manhattan') return dx + dy;
    throw new Error('Unknown heuristic');
  }
  function parsePriority(source) {
    if (typeof source !== 'string' || source.length > 80 || !source.trim()) throw new Error('Use 1–80 characters.');
    const tokens = []; let i = 0;
    while (i < source.length) {
      if (/\s/.test(source[i])) { i++; continue; }
      const rest = source.slice(i), number = /^(?:\d+(?:\.\d*)?|\.\d+)/.exec(rest);
      if (number) { const v = Number(number[0]); if (!Number.isFinite(v) || v > 1e6) throw new Error('Number is too large.'); tokens.push(v); i += number[0].length; }
      else if ('gh+-*/()'.includes(source[i])) tokens.push(source[i++]);
      else throw new Error('Only g, h, numbers, + - * / and parentheses are allowed.');
      if (tokens.length > 80) throw new Error('Too many tokens.');
    }
    let p = 0;
    function atom(depth) {
      if (depth > 16) throw new Error('Expression is nested too deeply.');
      const t = tokens[p++];
      if (typeof t === 'number') return { type: 'number', value: t };
      if (t === 'g' || t === 'h') return { type: 'variable', value: t };
      if (t === '+' || t === '-') return { type: 'unary', op: t, child: atom(depth + 1) };
      if (t === '(') { const node = sum(depth + 1); if (tokens[p++] !== ')') throw new Error('Missing closing parenthesis.'); return node; }
      throw new Error('Expected a number, g, h or parenthesis.');
    }
    function product(depth) { let node = atom(depth); while (tokens[p] === '*' || tokens[p] === '/') { const op = tokens[p++]; node = { type: 'binary', op, left: node, right: atom(depth) }; } return node; }
    function sum(depth) { let node = product(depth); while (tokens[p] === '+' || tokens[p] === '-') { const op = tokens[p++]; node = { type: 'binary', op, left: node, right: product(depth) }; } return node; }
    const tree = sum(0); if (p !== tokens.length) throw new Error('Unexpected token.');
    function value(node, g, h) {
      if (node.type === 'number') return node.value;
      if (node.type === 'variable') return node.value === 'g' ? g : h;
      if (node.type === 'unary') return (node.op === '-' ? -1 : 1) * value(node.child, g, h);
      const a = value(node.left, g, h), b = value(node.right, g, h);
      return node.op === '+' ? a + b : node.op === '-' ? a - b : node.op === '*' ? a * b : a / b;
    }
    return (g, h) => { const v = value(tree, g, h); if (!Number.isFinite(v) || Math.abs(v) > 1e12) throw new Error('Priority must stay finite and within ±10¹².'); return v; };
  }
  class Search {
    constructor(map, options = {}) {
      this.map = map; this.options = { algorithm: 'astar', movement: 4, heuristic: 'manhattan', weight: 1, ...options };
      const o = this.options;
      if (!['dfs', 'bfs', 'dijkstra', 'greedy', 'astar', 'custom'].includes(o.algorithm)) throw new Error('Unknown algorithm');
      if (![4, 8].includes(o.movement) || !Number.isFinite(o.weight) || o.weight < 0 || o.weight > 3) throw new Error('Invalid movement or weight');
      if (!map.edges) validateMap(map);
      if (o.movement === 8 && o.heuristic === 'manhattan' && !o.hValues) throw new Error('Manhattan is not admissible for this eight-direction cost model');
      this.n = map.edges ? map.edges.length : map.cells.length;
      this.g = Array(this.n).fill(Infinity); this.parent = Array(this.n).fill(-1); this.discovered = new Set([map.start]);
      this.closed = new Set(); this.frontier = new Set([map.start]); this.expandedOrder = []; this.selectedOrder = [];
      this.current = null; this.path = []; this.done = false; this.found = false; this.expanded = 0; this.reopened = 0; this.stale = 0; this.sequence = 0; this.maxFrontier = 1;
      this.queue = []; this.head = 0;
      this.heap = new Heap((a, b) => a.priority - b.priority || a.h - b.h || a.sequence - b.sequence);
      this.g[map.start] = 0; this.insert(map.start);
    }
    h(u) { return this.options.hValues ? this.options.hValues[u] : this.map.edges ? 0 : heuristic(this.map, u, this.options.heuristic); }
    priority(g, h) {
      const o = this.options;
      // Queue/stack order is not a numerical g/h priority; zero is an unused placeholder.
      return o.algorithm === 'dfs' || o.algorithm === 'bfs' ? 0 : o.algorithm === 'dijkstra' ? g : o.algorithm === 'greedy' ? h : o.algorithm === 'custom' ? o.priority(g, h) : g + o.weight * h;
    }
    insert(u) {
      const h = this.h(u), entry = { id: u, g: this.g[u], h, priority: this.priority(this.g[u], h), sequence: this.sequence++ };
      if (this.options.algorithm === 'dfs' || this.options.algorithm === 'bfs') this.queue.push(entry); else this.heap.push(entry);
      this.frontier.add(u); this.maxFrontier = Math.max(this.maxFrontier, this.frontier.size);
    }
    step() {
      if (this.done) return null;
      const simple = this.options.algorithm === 'dfs' || this.options.algorithm === 'bfs'; let entry;
      while (true) {
        if (simple) entry = this.options.algorithm === 'dfs' ? this.queue.pop() : this.queue[this.head++]; else entry = this.heap.pop();
        if (!entry) { this.done = true; this.current = null; return null; }
        // Lazy decrease-key: an improved g inserts a new entry; obsolete entries never expand.
        if (entry.g !== this.g[entry.id] || this.closed.has(entry.id)) { this.stale++; continue; }
        break;
      }
      const u = entry.id; this.current = u; this.frontier.delete(u); this.selectedOrder.push(u);
      if (u === this.map.goal) {
        this.done = true; this.found = true;
        for (let v = u; v !== -1; v = this.parent[v]) this.path.push(v);
        this.path.reverse(); return entry;
      }
      this.closed.add(u); this.expanded++; this.expandedOrder.push(u);
      let edges = this.map.edges ? this.map.edges[u] : neighbors(this.map, u, this.options.movement);
      // Reverse insertion gives stack DFS the same first neighbor as queue BFS.
      if (this.options.algorithm === 'dfs') edges = edges.slice().reverse();
      for (const [v, cost] of edges) {
        if (!Number.isFinite(cost) || cost <= 0) throw new Error('These teaching models require positive edge costs');
        if (simple && this.discovered.has(v)) continue;
        const tentative = this.g[u] + cost;
        if (tentative < this.g[v]) {
          this.g[v] = tentative; this.parent[v] = u; this.discovered.add(v);
          // An admissible but inconsistent h may improve a state already expanded.
          if (this.closed.delete(v)) this.reopened++;
          this.insert(v);
        }
      }
      return entry;
    }
    run(limit = 10000) { let count = 0; while (!this.done && count++ < limit) this.step(); if (!this.done) throw new Error('Search exceeded the 10,000-pop teaching limit.'); return this; }
    entries() {
      const simple = this.options.algorithm === 'dfs' || this.options.algorithm === 'bfs';
      let entries = simple ? this.options.algorithm === 'dfs' ? this.queue.slice().reverse() : this.queue.slice(this.head) : this.heap.items.slice().sort(this.heap.compare);
      const seen = new Set();
      return entries.filter(e => e.g === this.g[e.id] && !this.closed.has(e.id) && !seen.has(e.id) && seen.add(e.id));
    }
    get cost() { return this.found ? this.g[this.map.goal] : Infinity; }
  }
  const runSearch = (map, options) => new Search(map, options).run();
  function priorityTests(expression) {
    const priority = parsePriority(expression);
    return ['weighted', 'detour', 'open'].map(name => {
      const map = makePreset(name), reference = runSearch(map, { algorithm: 'dijkstra' });
      const result = runSearch(map, { algorithm: 'custom', priority });
      return { name, result, reference, optimal: sameCost(result.cost, reference.cost) };
    });
  }
  // These fragments deliberately are not Python execution. Their small grammars
  // map to explicit operations; no eval, Function, assignment or property access.
  function parseRelaxation(source) {
    const match = typeof source === 'string' && /^\s*candidate\s*(<=|>=|==|!=|<|>)\s*old_g\s*$/.exec(source);
    if (!match || source.length > 80) throw new Error('Use candidate < old_g (or another comparison of these two names).');
    const operation = match[1];
    return (candidate, oldG) => operation === '<' ? candidate < oldG : operation === '<=' ? candidate <= oldG : operation === '>' ? candidate > oldG : operation === '>=' ? candidate >= oldG : operation === '==' ? candidate === oldG : candidate !== oldG;
  }
  function materializeCodeCase(test) {
    const source = test && test.graph;
    if (!source) throw new Error('Missing graph.');
    let edges, h;
    if (source.grid) {
      const { width, height } = source.grid;
      if (![width, height].every(x => Number.isInteger(x) && x > 0) || width * height > 400) throw new Error('Invalid bounded grid.');
      edges = Array.from({ length: width * height }, (_, u) => {
        const x = u % width, y = Math.floor(u / width);
        return [[1, 0], [0, 1], [-1, 0], [0, -1]].filter(([dx, dy]) => x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height).map(([dx, dy]) => [(y + dy) * width + x + dx, 1]);
      });
      if (source.h !== 'manhattan') throw new Error('Unknown grid heuristic.');
      h = edges.map((_, u) => Math.abs(u % width - source.goal % width) + Math.abs(Math.floor(u / width) - Math.floor(source.goal / width)));
    } else { edges = source.edges; h = source.h; }
    const n = edges && edges.length;
    if (!Array.isArray(edges) || n < 1 || n > 400 || ![source.start, source.goal].every(u => Number.isInteger(u) && u >= 0 && u < n) || !Array.isArray(h) || h.length !== n || !h.every(x => Number.isFinite(x) && x >= 0 && x <= 1e6) || h[source.goal] !== 0) throw new Error('Invalid graph or heuristic.');
    let edgeCount = 0;
    for (const row of edges) {
      if (!Array.isArray(row)) throw new Error('Invalid adjacency list.');
      for (const edge of row) {
        edgeCount++;
        if (!Array.isArray(edge) || edge.length !== 2 || !Number.isInteger(edge[0]) || edge[0] < 0 || edge[0] >= n || !Number.isFinite(edge[1]) || edge[1] < 0 || edge[1] > 1e6) throw new Error('Edges must have finite nonnegative costs.');
      }
    }
    if (edgeCount > 4000) throw new Error('Too many edges.');
    return { edges, h, start: source.start, goal: source.goal };
  }
  function runCodeCase(test, options = {}) {
    const result = { cost: null, found: false, expanded: 0, pops: 0, stale: 0, reopened: 0, limited: false, error: null };
    try {
      const priority = parsePriority(options.priority), improve = parseRelaxation(options.relaxation);
      if (!['pop', 'discovery'].includes(options.goalCheck)) throw new Error('Choose where to test the goal.');
      const { edges, h, start, goal } = materializeCodeCase(test), best = Array(edges.length).fill(Infinity), expandedBefore = new Set();
      const limit = options.popLimit === undefined ? 5000 : options.popLimit;
      if (!Number.isInteger(limit) || limit < 1 || limit > 5000) throw new Error('Pop limit must be within 1–5000.');
      let sequence = 0;
      const heap = new Heap((a, b) => a.f - b.f || a.h - b.h || a.sequence - b.sequence);
      const insert = (u, g) => heap.push({ u, g, h: h[u], f: priority(g, h[u]), sequence: sequence++ });
      best[start] = 0; insert(start, 0);
      while (heap.length) {
        if (result.pops >= limit) { result.limited = true; return result; }
        const entry = heap.pop(); result.pops++;
        const { u, g } = entry;
        // Lazy snapshots must be rejected before either the goal or successors.
        if (g !== best[u]) { result.stale++; continue; }
        if (options.goalCheck === 'pop' && u === goal) { result.cost = g; result.found = true; return result; }
        result.expanded++; expandedBefore.add(u);
        for (const [v, cost] of edges[u]) {
          const candidate = g + cost;
          if (improve(candidate, best[v])) {
            if (expandedBefore.has(v)) result.reopened++;
            best[v] = candidate;
            if (options.goalCheck === 'discovery' && v === goal) { result.cost = candidate; result.found = true; return result; }
            insert(v, candidate);
          }
        }
      }
    } catch (error) { result.error = error.message; }
    return result;
  }
  function judgeCodeCases(cases, options) {
    if (!Array.isArray(cases) || cases.length > 20) throw new Error('Invalid test suite.');
    return cases.map(test => {
      const result = runCodeCase(test, options), costPassed = !result.error && !result.limited && (test.expected === null ? result.cost === null : sameCost(result.cost, test.expected));
      const effortPassed = test.maxExpanded === undefined || result.expanded <= test.maxExpanded;
      return { test, ...result, costPassed, effortPassed, passed: costPassed && effortPassed };
    });
  }
  const api = { Heap, Search, makePreset, validateMap, neighbors, heuristic, parsePriority, priorityTests, runSearch, sameCost, parseRelaxation, materializeCodeCase, runCodeCase, judgeCodeCases };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const register = window.ArticleLabs.register;
  const presetOptions = [
    { value: 'branch', en: 'Branch maze (unit cost)', ko: 'Branch maze (단위 비용)' },
    { value: 'weighted', en: 'Expensive direct road', ko: '비싼 terrain을 지나는 직선 경로' },
    { value: 'comparison', en: 'Figure 7 comparison maze', ko: 'Figure 7 비교 maze' },
    { value: 'detour', en: 'Wall with a distant gap', ko: '멀리 돌아가야 하는 벽' },
    { value: 'open', en: 'Open grid', ko: '빈 grid' },
    { value: 'blocked', en: 'No route', ko: '경로 없음' }
  ];
  function mapLab(ctx, traversal) {
    ctx.root.classList.add('pf-lab', 'al-workshop');
    const preset = ctx.control('Map preset', 'Map 예제', { type: 'select', value: traversal ? 'branch' : 'weighted', options: traversal ? presetOptions.filter(x => ['branch', 'open', 'blocked'].includes(x.value)) : presetOptions });
    const algorithm = ctx.control('Algorithm', 'Algorithm', { type: 'select', value: traversal ? 'dfs' : 'astar', options: (traversal ? ['dfs', 'bfs'] : ['bfs', 'dijkstra', 'greedy', 'astar']).map(value => ({ value, en: { dfs: 'DFS — stack', bfs: 'BFS — queue', dijkstra: 'Dijkstra — g', greedy: 'Greedy — h', astar: 'A* — g + w h' }[value] })) });
    const movement = traversal ? null : ctx.control('Allowed moves', '이동 방향', { type: 'select', value: '4', options: [{ value: '4', en: '4 directions', ko: '4방향' }, { value: '8', en: '8 directions; no corner cutting', ko: '8방향 · corner cutting 금지' }] });
    const hChoice = traversal ? null : ctx.control('Heuristic h', 'Heuristic h', { type: 'select', value: 'manhattan', options: ['manhattan', 'euclidean', 'octile', 'zero'].map(value => ({ value, en: value === 'zero' ? 'Zero (h = 0)' : value[0].toUpperCase() + value.slice(1) })) });
    const weight = traversal ? null : ctx.control('Heuristic multiplier w', 'Heuristic 계수 w', { type: 'range', min: 0, max: 3, step: .25, value: 1 });
    const edit = ctx.control('Click / Enter action', '클릭 / Enter 동작', { type: 'select', value: 'inspect', options: [
      { value: 'inspect', en: 'Inspect cell', ko: 'Cell 살펴보기' }, { value: 'wall', en: 'Toggle wall', ko: '벽 켜기 / 끄기' }, ...(traversal ? [] : [{ value: 'terrain', en: 'Toggle terrain 1 ↔ 5', ko: 'Terrain 1 ↔ 5' }]), { value: 'start', en: 'Move start S', ko: 'Start S 옮기기' }, { value: 'goal', en: 'Move goal G', ko: 'Goal G 옮기기' }
    ] });
    let map = makePreset(preset.value), search, selected = map.start, timer = null, loadedPreset = preset.value, signature = '';
    const expanded = ctx.metric('Expanded / reopened', 'Expanded / reopened'), frontierMetric = ctx.metric('Frontier / peak', 'Frontier / 최댓값'), cost = ctx.metric('Route cost / optimal', '경로 비용 / 최적 비용');
    const detail = ctx.el('p', { class: 'pf-detail' });
    const board = ctx.el('div', { class: 'pf-board', role: 'grid', 'aria-label': ctx.text('Editable pathfinding grid', '수정 가능한 pathfinding grid') });
    const boardScroll = ctx.el('div', { class: 'pf-grid-scroll', role: 'region', tabindex: '0', 'aria-label': ctx.text('Pathfinding map; scroll sideways on narrow screens', 'Pathfinding map · 좁은 화면에서는 좌우로 스크롤') }, board);
    const legend = ctx.el('div', { class: 'pf-legend' }, [['frontier', 'F · frontier'], ['expanded', '· · expanded'], ['current', '@ · current'], ['path', 'P · path'], ['wall', '# · wall'], ['terrain', '5 · terrain']].map(([type, label]) => ctx.el('span', { class: `pf-key pf-${type}` }, label)));
    const frontierView = ctx.el('div', { class: 'pf-frontier-view' });
    const instruction = ctx.el('p', { class: 'pf-instruction' }, ctx.text('Arrow keys move keyboard focus; Enter or Space applies the selected action. On narrow screens, scroll the map sideways. Any map or search-setting change restarts the search. S/G may share a cell.', '방향키로 cell을 선택하고 Enter나 Space로 위 동작을 적용하세요. 좁은 화면에서는 map을 좌우로 스크롤할 수 있습니다. Map이나 탐색 조건을 바꾸면 처음부터 다시 탐색합니다. S와 G를 같은 cell에 둘 수도 있습니다.'));
    ctx.view.append(boardScroll, legend, detail, frontierView, instruction);
    const fmt = x => Number.isFinite(x) ? ctx.format(x, 3) : '∞';
    function stop() { if (timer !== null) window.clearInterval(timer); timer = null; if (runButton) runButton.textContent = ctx.text('Run', '실행'); }
    function options() {
      if (!traversal && movement.value === '8' && hChoice.value === 'manhattan') hChoice.value = 'octile';
      return { algorithm: algorithm.value, movement: traversal ? 4 : Number(movement.value), heuristic: traversal ? 'manhattan' : hChoice.value, weight: traversal ? 1 : Number(weight.value) };
    }
    function restart() { stop(); search = new Search(map, options()); signature = JSON.stringify(options()); }
    function coordinate(u) { return `(${u % map.width},${Math.floor(u / map.width)})`; }
    function action(u) {
      selected = u;
      if (edit.value !== 'inspect') {
        if (edit.value === 'wall' && u !== map.start && u !== map.goal) map.cells[u] = map.cells[u] ? 0 : 1;
        if (edit.value === 'terrain' && map.cells[u]) map.cells[u] = map.cells[u] === 5 ? 1 : 5;
        if (edit.value === 'start') { map.cells[u] = map.cells[u] || 1; map.start = u; }
        if (edit.value === 'goal') { map.cells[u] = map.cells[u] || 1; map.goal = u; }
        restart();
      }
      paint();
    }
    const cells = [];
    for (let y = 0; y < map.height; y++) {
      const row = ctx.el('div', { role: 'row', class: 'pf-row' });
      for (let x = 0; x < map.width; x++) {
        const u = y * map.width + x, button = ctx.el('button', { type: 'button', role: 'gridcell', class: 'pf-cell', tabindex: u === 0 ? '0' : '-1' });
        button.addEventListener('click', () => { cells.forEach(c => c.tabIndex = -1); button.tabIndex = 0; action(u); });
        button.addEventListener('keydown', event => {
          const offset = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowDown: [0, 1], ArrowUp: [0, -1] }[event.key];
          if (!offset) return;
          event.preventDefault(); const nx = x + offset[0], ny = y + offset[1];
          if (nx >= 0 && nx < map.width && ny >= 0 && ny < map.height) { button.tabIndex = -1; const next = cells[ny * map.width + nx]; next.tabIndex = 0; next.focus(); }
        });
        cells.push(button); row.append(button);
      }
      board.append(row);
    }
    let runButton;
    ctx.button('Step', '한 단계', () => { stop(); search.step(); paint(); });
    runButton = ctx.button('Run', '실행', () => {
      if (timer !== null) { stop(); return; }
      if (search.done) restart();
      // Nonvisual hosts may omit timers; the same bounded engine still runs to completion.
      if (typeof window.setInterval !== 'function') { search.run(); paint(); return; }
      runButton.textContent = ctx.text('Pause', '일시정지');
      timer = window.setInterval(() => { if (!ctx.root.isConnected) { stop(); return; } search.step(); paint(); if (search.done) stop(); }, 100);
    });
    ctx.button('Finish now', '끝까지 계산', () => { stop(); search.run(); paint(); });
    ctx.button('Restart search', '탐색만 다시 시작', () => { restart(); paint(); });
    function paint() {
      const route = new Set(search.path), moves = options().movement;
      cells.forEach((cell, u) => {
        let kind = map.cells[u] === 0 ? 'wall' : route.has(u) ? 'path' : search.current === u ? 'current' : search.frontier.has(u) ? 'frontier' : search.closed.has(u) ? 'expanded' : map.cells[u] === 5 ? 'terrain' : 'open';
        const marker = u === map.start && u === map.goal ? 'S/G' : u === map.start ? 'S' : u === map.goal ? 'G' : kind === 'wall' ? '#' : kind === 'current' ? '@' : kind === 'path' ? 'P' : kind === 'frontier' ? 'F' : kind === 'expanded' ? '·' : map.cells[u] === 5 ? '5' : '';
        cell.className = `pf-cell pf-${kind}${u === selected ? ' pf-selected' : ''}`; cell.textContent = marker;
        cell.setAttribute('aria-label', `${coordinate(u)}, ${kind}, terrain ${map.cells[u]}, g ${fmt(search.g[u])}${u === map.start ? ', start' : ''}${u === map.goal ? ', goal' : ''}`);
        cell.setAttribute('aria-selected', String(u === selected));
      });
      const reference = runSearch(map, { algorithm: 'dijkstra', movement: moves, heuristic: moves === 8 ? 'octile' : 'manhattan' });
      expanded.textContent = `${search.expanded} / ${search.reopened}`; frontierMetric.textContent = `${search.frontier.size} / ${search.maxFrontier}`;
      cost.textContent = `${search.found ? fmt(search.cost) : '—'} / ${fmt(reference.cost)}`;
      const g = search.g[selected], h = search.h(selected), priority = Number.isFinite(g) ? search.priority(g, h) : Infinity;
      const ordered = ['dfs', 'bfs'].includes(algorithm.value), orderLabel = algorithm.value === 'dfs' ? 'LIFO' : 'FIFO';
      detail.textContent = ctx.text(`Selected ${coordinate(selected)}: g=${fmt(g)}, h=${fmt(h)}, ${ordered ? `order=${orderLabel} (g/h unused)` : `priority=${fmt(priority)}`}. Terrain=${map.cells[selected] || 'wall'}.`, `선택 ${coordinate(selected)}: g=${fmt(g)}, h=${fmt(h)}, ${ordered ? `순서=${orderLabel} (g/h 미사용)` : `priority=${fmt(priority)}`}. Terrain=${map.cells[selected] || '벽'}.`);
      const entries = search.entries();
      frontierView.replaceChildren(ctx.el('p', {}, ctx.text('Next frontier entries (at most 8)', '다음 frontier entry (최대 8개)')), ctx.table([ctx.text('Cell', 'Cell'), 'g', 'h', ordered ? ctx.text('Order', '순서') : 'priority'], entries.slice(0, 8).map((e, i) => [coordinate(e.id), fmt(e.g), fmt(e.h), ordered ? i + 1 : fmt(e.priority)])));
      let message = search.done ? search.found ? ctx.text(`Goal removed from frontier. ${search.path.length - 1} moves, cost ${fmt(search.cost)}.`, `Frontier에서 goal을 꺼냈습니다. ${search.path.length - 1}번 이동, 비용 ${fmt(search.cost)}.`) : ctx.text('Frontier exhausted: no route exists.', 'Frontier가 비었습니다. 경로가 없습니다.') : ctx.text('Step removes one valid frontier entry and examines its neighbors. Goal testing happens on removal.', '한 단계마다 유효한 frontier entry 하나를 꺼내 이웃을 검사합니다. 꺼낸 state가 goal인지 먼저 확인합니다.');
      if (!traversal && algorithm.value === 'bfs') message += ctx.text(' BFS minimizes moves, not weighted cost.', ' BFS는 이동 횟수를 최소화하며 weighted 비용은 최소화하지 않습니다.');
      if (!traversal && algorithm.value === 'greedy') message += ctx.text(' Greedy has no optimal-cost guarantee.', ' Greedy에는 최적 비용 보장이 없습니다.');
      if (!traversal && algorithm.value === 'astar' && Number(weight.value) > 1) message += ctx.text(' w>1 is Weighted A*: optimality is not guaranteed; with this admissible base h and reopening, cost ≤ w×optimal.', ' w>1이면 Weighted A*입니다. 최적해는 보장하지 않지만, 이 admissible h와 reopening 조건에서는 비용 ≤ w×최적 비용입니다.');
      ctx.status.textContent = message;
    }
    ctx.onReset(() => { map = makePreset(preset.value); loadedPreset = preset.value; selected = map.start; restart(); });
    ctx.onChange(() => {
      if (loadedPreset !== preset.value) { map = makePreset(preset.value); loadedPreset = preset.value; selected = map.start; signature = ''; }
      const next = JSON.stringify(options()); if (!search || signature !== next) restart(); paint();
    });
    ctx.stateLink(traversal ? 'astar-traversal' : 'astar-weighted', {
      read: () => ({ preset: preset.value, algorithm: algorithm.value, movement: traversal ? 4 : Number(movement.value), heuristic: traversal ? 'manhattan' : hChoice.value, weight: traversal ? 1 : Number(weight.value), map, selected, steps: search.selectedOrder.length + (search.done && !search.found ? 1 : 0) }),
      write: state => {
        if (!state || !presetOptions.some(x => x.value === state.preset) || !(traversal ? ['dfs', 'bfs'] : ['bfs', 'dijkstra', 'greedy', 'astar']).includes(state.algorithm) || ![4, 8].includes(state.movement) || !['zero', 'manhattan', 'euclidean', 'octile'].includes(state.heuristic) || !Number.isFinite(state.weight) || state.weight < 0 || state.weight > 3 || state.weight * 4 !== Math.round(state.weight * 4)) return false;
        validateMap(state.map); if (state.map.width !== 13 || state.map.height !== 9 || traversal && (state.movement !== 4 || state.map.cells.some(x => x === 5) || !['branch', 'open', 'blocked'].includes(state.preset))) return false;
        const steps = state.steps === undefined ? 0 : state.steps, savedSelected = state.selected === undefined ? state.map.start : state.selected;
        if (!Number.isInteger(steps) || steps < 0 || steps > 10000 || !Number.isInteger(savedSelected) || savedSelected < 0 || savedSelected >= state.map.cells.length) return false;
        const nextMap = { ...state.map, cells: state.map.cells.slice() }, nextOptions = { algorithm: state.algorithm, movement: state.movement, heuristic: state.movement === 8 && state.heuristic === 'manhattan' ? 'octile' : state.heuristic, weight: state.weight }, next = new Search(nextMap, nextOptions);
        for (let i = 0; i < steps; i++) { if (next.done) return false; next.step(); }
        stop(); map = nextMap; loadedPreset = state.preset; preset.value = state.preset; algorithm.value = state.algorithm;
        if (!traversal) { movement.value = String(state.movement); hChoice.value = nextOptions.heuristic; weight.value = String(state.weight); }
        selected = savedSelected; search = next; signature = JSON.stringify(options()); paint(); return true;
      }
    });
  }
  register('astar-traversal', ctx => mapLab(ctx, true));
  register('astar-weighted', ctx => mapLab(ctx, false));
  register('astar-priority-lab', ctx => {
    ctx.root.classList.add('pf-lab', 'al-workshop');
    const formula = ctx.el('input', { type: 'text', value: 'g + h', maxlength: '80', spellcheck: 'false', class: 'pf-expression', 'aria-label': ctx.text('Priority expression', 'Priority 수식') });
    const label = ctx.el('label', { class: 'pf-expression-label' }, [ctx.text('Priority expression (g, h, + − * /)', 'Priority 수식 (g, h, + − * /)'), formula]); ctx.ui.append(label);
    const preset = ctx.control('Expression preset', '수식 예제', { type: 'select', value: 'g + h', options: ['g', 'h', 'g + h', 'g + 2*h', 'g + 0.5*h'].map(value => ({ value, en: value })) });
    const result = ctx.metric('Optimal on these test maps', '예제에서 최적 비용 일치'), output = ctx.el('div', { class: 'pf-test-output' }); ctx.view.append(output);
    let chosen = preset.value;
    function render() {
      if (preset.value !== chosen) { chosen = preset.value; formula.value = chosen; }
      try {
        const rows = priorityTests(formula.value); result.textContent = `${rows.filter(x => x.optimal).length} / ${rows.length}`;
        output.replaceChildren(ctx.table([ctx.text('Test map', '예제 map'), ctx.text('Your cost', '수식 비용'), 'Dijkstra', ctx.text('Expanded', 'Expanded'), ctx.text('Match', '일치')], rows.map(x => [x.name, ctx.format(x.result.cost, 3), ctx.format(x.reference.cost, 3), x.result.expanded, x.optimal ? '✓' : '✗'])));
        ctx.status.textContent = ctx.text('Only the arithmetic expression runs. A green result on three maps is a test outcome, not a proof that this formula is optimal on every graph.', '입력한 산술식만 실행합니다. 예제 세 개에서 비용이 일치해도 모든 graph에서 최적이라는 증명은 아닙니다.');
        formula.setAttribute('aria-invalid', 'false');
      } catch (error) {
        result.textContent = '—'; output.replaceChildren(); formula.setAttribute('aria-invalid', 'true');
        ctx.status.textContent = ctx.text(`Expression rejected: ${error.message}`, `수식을 실행할 수 없습니다: ${error.message}`);
      }
    }
    ctx.button('Run test maps', '예제 실행', render);
    ctx.onReset(() => { chosen = 'g + h'; formula.value = 'g + h'; }); ctx.onChange(render);
    ctx.checkpointState({
      read: () => ({ formula: formula.value, preset: preset.value }),
      write: s => { if (!s || typeof s.formula !== 'string' || s.formula.length > 80 || !['g', 'h', 'g + h', 'g + 2*h', 'g + 0.5*h'].includes(s.preset)) return false; preset.value = chosen = s.preset; formula.value = s.formula; render(); return true; }
    });
  });
  const exercisePath = '/assets/data/astar-exercises.json';
  const translated = (ctx, value) => ctx.text(value.en, value.ko);
  function exerciseFigure(ctx) {
    const mobile = ctx.el('source', { media: '(max-width:640px)' });
    const image = ctx.el('img', { loading: 'lazy', decoding: 'async' });
    const caption = ctx.el('figcaption');
    const figure = ctx.el('figure', { class: 'pf-exercise-figure' }, [ctx.el('picture', {}, [mobile, image]), caption]);
    return {
      figure,
      show(data) {
        const version = encodeURIComponent(ctx.root.dataset.assetVersion || 'local');
        mobile.setAttribute('srcset', `${ctx.asset(data.mobile)}?v=${version}`);
        image.setAttribute('src', `${ctx.asset(data.desktop)}?v=${version}`);
        image.setAttribute('alt', translated(ctx, data.alt));
        caption.textContent = translated(ctx, data.caption);
      }
    };
  }
  register('astar-quiz', async ctx => {
    ctx.root.classList.add('pf-lab', 'al-workshop');
    const { questions } = await ctx.fetchJSON(exercisePath);
    const choose = ctx.control('Exercise', '문제 선택', { type: 'select', value: '0', options: questions.map((q, i) => ({ value: String(i), en: (q.nav || q.title).en, ko: (q.nav || q.title).ko })) });
    const progress = ctx.metric('Answered', '푼 문제'), score = ctx.metric('Correct', '정답');
    let answers = questions.map(() => -1), hints = questions.map(() => false);
    // This is the selected exercise's heading, not a permanent article section.
    // Keep its accessible level without adding a stale entry to the article TOC.
    const title = ctx.el('div', { class: 'pf-exercise-title', role: 'heading', 'aria-level': '3' }), diagram = exerciseFigure(ctx), prompt = ctx.el('p', { class: 'pf-exercise-prompt' });
    const answer = ctx.el('div', { class: 'pf-answer-options', role: 'radiogroup', 'aria-label': ctx.text('Answer for the current exercise', '현재 문제의 답') });
    const feedback = ctx.el('div', { class: 'pf-quiz-feedback', 'aria-live': 'polite' }), hint = ctx.el('p', { class: 'pf-exercise-hint' });
    ctx.view.append(title, diagram.figure, prompt, ctx.el('p', { class: 'pf-answer-label' }, ctx.text('Choose an answer', '답 고르기')), answer, hint, feedback);
    function render() {
      const index = Number(choose.value), q = questions[index];
      if (!q) return;
      title.textContent = translated(ctx, q.title); diagram.show(q.image); prompt.textContent = translated(ctx, q.prompt);
      const answerButtons = q.options.map((option, i) => {
        const button = ctx.el('button', { type: 'button', role: 'radio', class: 'pf-answer-option', 'aria-checked': answers[index] === i ? 'true' : 'false', tabindex: i === (answers[index] < 0 ? 0 : answers[index]) ? '0' : '-1' }, `${String.fromCharCode(65 + i)}. ${translated(ctx, option)}`);
        button.addEventListener('click', () => { answers[index] = i; render(); const active = answer.children[i]; if (active && active.focus) active.focus(); });
        button.addEventListener('keydown', event => {
          let target = i;
          if (event.key === 'ArrowDown' || event.key === 'ArrowRight') target = (i + 1) % q.options.length;
          else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') target = (i + q.options.length - 1) % q.options.length;
          else if (event.key === 'Home') target = 0;
          else if (event.key === 'End') target = q.options.length - 1;
          else return;
          event.preventDefault(); answers[index] = target; render(); const active = answer.children[target]; if (active && active.focus) active.focus();
        });
        return button;
      });
      answer.replaceChildren(...answerButtons);
      hint.textContent = hints[index] ? translated(ctx, q.hint) : '';
      const answered = answers.filter(value => value !== -1).length;
      progress.textContent = `${answered} / ${questions.length}`;
      score.textContent = `${answers.filter((value, i) => value === questions[i].answer).length} / ${questions.length}`;
      previous.disabled = index === 0; next.disabled = index === questions.length - 1;
      feedback.replaceChildren();
      if (answers[index] !== -1) {
        const correct = answers[index] === q.answer;
        feedback.append(ctx.el('div', { class: 'pf-answer' }, [ctx.el('strong', {}, correct ? ctx.text('Correct', '정답입니다') : ctx.text('Recheck the given conditions', '조건을 다시 확인해 보세요')), ctx.el('p', {}, translated(ctx, q.explanation)), ctx.el('p', { class: 'pf-exercise-followup' }, translated(ctx, q.followup))]));
      }
      ctx.status.textContent = ctx.text('Each diagram, question and explanation stay together. Choose an answer for immediate feedback; Previous/Next keeps your answers. Reset clears all five.', '그림과 질문, 해설을 한곳에서 확인할 수 있습니다. 답을 고르면 바로 해설이 나옵니다. 이전·다음 문제로 이동해도 고른 답은 유지되고, 초기값으로 돌아가면 모두 지워집니다.');
    }
    const previous = ctx.button('Previous exercise', '이전 문제', () => { choose.value = String(Math.max(0, Number(choose.value) - 1)); render(); });
    const next = ctx.button('Next exercise', '다음 문제', () => { choose.value = String(Math.min(questions.length - 1, Number(choose.value) + 1)); render(); });
    ctx.button('Show / hide hint', '힌트 보기 / 접기', () => { const index = Number(choose.value); hints[index] = !hints[index]; render(); });
    function selectHashQuestion() {
      let hash = '';
      try { hash = decodeURIComponent((window.location && window.location.hash || '').replace(/^#/, '')); } catch (_) { return; }
      const index = questions.findIndex(q => q.anchors && (q.anchors.en === hash || q.anchors.ko === hash));
      if (index >= 0) choose.value = String(index);
    }
    selectHashQuestion();
    if (window.addEventListener) {
      const hashChange = () => {
        // A local-editor refresh replaces the widget's entire DOM. Retired
        // quiz panels must not react to later article navigation.
        if (ctx.root.isConnected === false) { if (window.removeEventListener) window.removeEventListener('hashchange', hashChange); return; }
        selectHashQuestion(); render();
      };
      window.addEventListener('hashchange', hashChange);
    }
    ctx.onReset(() => { answers = questions.map(() => -1); hints = questions.map(() => false); choose.value = '0'; });
    ctx.onChange(render);
    ctx.checkpointState({
      read: () => ({ question: Number(choose.value), answers: answers.slice(), hints: hints.slice() }),
      write: s => {
        if (!s || !Number.isInteger(s.question) || s.question < 0 || s.question >= questions.length || !Array.isArray(s.answers) || s.answers.length !== questions.length || !s.answers.every((v, i) => Number.isInteger(v) && v >= -1 && v < questions[i].options.length) || !Array.isArray(s.hints) || s.hints.length !== questions.length || !s.hints.every(v => typeof v === 'boolean')) return false;
        choose.value = String(s.question); answers = s.answers.slice(); hints = s.hints.slice(); render(); return true;
      }
    });
  });
  register('astar-code-test', async ctx => {
    ctx.root.classList.add('pf-lab', 'al-workshop');
    const { coding } = await ctx.fetchJSON(exercisePath), diagram = exerciseFigure(ctx);
    diagram.show(coding.image);
    const makeFragment = (en, ko, placeholder) => {
      const field = ctx.el('input', { type: 'text', value: '', maxlength: '80', spellcheck: 'false', autocomplete: 'off', placeholder, class: 'pf-expression', 'aria-label': ctx.text(en, ko) });
      ctx.ui.append(ctx.el('label', { class: 'pf-expression-label' }, [ctx.text(en, ko), field])); return field;
    };
    const priority = makeFragment('1. Priority expression (g and h)', '1. Priority 수식 (g와 h)', 'TODO_PRIORITY');
    const relaxation = makeFragment('2. Relaxation condition', '2. 비용 갱신 조건', 'TODO_RELAXATION');
    const goal = ctx.control('3. Goal-check location', '3. Goal을 확인할 위치', { type: 'select', value: '', options: [{ value: '', en: 'Choose a location', ko: '위치 선택' }, { value: 'pop', en: 'After a valid minimum-priority pop', ko: '유효한 최소 priority pop 직후' }, { value: 'discovery', en: 'As soon as a relaxation discovers the goal', ko: '갱신 중 goal을 발견한 즉시' }] });
    const passed = ctx.metric('Tests passed', '통과한 테스트'), costCount = ctx.metric('Correct costs', '비용이 맞는 테스트');
    const code = ctx.el('code'), hint = ctx.el('p', { class: 'pf-exercise-hint' }), output = ctx.el('div', { class: 'pf-code-results', 'aria-live': 'polite' });
    const skeleton = ctx.el('details', { class: 'pf-code-scaffold' }, [ctx.el('summary', {}, ctx.text('Python scaffold — view the three edits in context', 'Python 코드에서 세 빈칸의 위치 확인하기')), ctx.el('pre', {}, code)]);
    ctx.view.append(diagram.figure, ctx.el('p', { class: 'pf-exercise-prompt' }, ctx.text('Complete three blanks in astar_cost(graph, start, goal, h). Return the least cost, or None if no route exists. The fixed scaffold already rejects stale entries and reinserts improved states.', 'astar_cost(graph, start, goal, h)의 빈칸 세 개를 채워 보세요. 최소 비용을 반환하고, 경로가 없으면 None을 반환해야 합니다. 오래된 entry를 건너뛰고 개선된 state를 다시 넣는 부분은 이미 작성되어 있습니다.')), skeleton, hint, output);
    let rows = null, showHint = false, lastDraft = '';
    const draft = () => JSON.stringify([priority.value, relaxation.value, goal.value]);
    function render() {
      const current = draft(); if (current !== lastDraft) { lastDraft = current; rows = null; }
      code.textContent = coding.starter.join('\n').split('TODO_PRIORITY').join(priority.value || 'TODO_PRIORITY').split('TODO_RELAXATION').join(relaxation.value || 'TODO_RELAXATION').split('TODO_GOAL_CHECK').join(goal.value || 'TODO_GOAL_CHECK');
      hint.textContent = showHint ? ctx.text('The priority needs both the cost already paid and an admissible estimate of what remains. Propagate only a genuinely smaller g. Accept the goal only after rejecting stale entries and removing a valid minimum-priority candidate.', 'Priority에는 이미 든 비용과 남은 비용의 admissible 추정값을 함께 써야 합니다. g가 실제로 작아졌을 때만 갱신하고, 오래된 entry를 걸러 낸 뒤 유효한 최소 priority 후보로 goal을 꺼냈을 때 종료해 보세요.') : '';
      output.replaceChildren();
      passed.textContent = rows ? `${rows.filter(row => row.passed).length} / ${rows.length}` : '—';
      costCount.textContent = rows ? `${rows.filter(row => row.costPassed).length} / ${rows.length}` : '—';
      if (!rows) { ctx.status.textContent = ctx.text('Fill the three blanks, then run. The browser executes only these restricted Python-syntax fragments, not arbitrary Python. Edit and run the full function in the downloadable notebook.', '세 빈칸을 채운 뒤 테스트를 실행해 보세요. 브라우저에서는 제한된 Python 문법의 수식·비교식과 선택한 종료 조건만 실행합니다. 전체 함수를 직접 수정하고 실행하려면 내려받은 notebook을 사용하세요.'); return; }
      const cost = value => value === null ? 'None' : String(value);
      output.append(ctx.table([ctx.text('Test', '테스트'), ctx.text('Expected', '정답 비용'), ctx.text('Returned', '반환값'), ctx.text('Expanded / budget', 'Expanded / 상한'), ctx.text('Result', '결과')], rows.map(row => [translated(ctx, row.test.name), cost(row.test.expected), row.limited ? ctx.text('Limit reached', '실행 상한 도달') : row.error ? ctx.text('Invalid fragment', '입력 오류') : cost(row.cost), `${row.expanded} / ${row.test.maxExpanded === undefined ? '—' : row.test.maxExpanded}`, row.passed ? '✓ PASS' : !row.costPassed ? ctx.text('✗ Cost / termination', '✗ 비용·종료 조건') : ctx.text('✗ Cost correct; budget exceeded', '✗ 비용은 정답, 탐색 상한 초과')])));
      for (const row of rows.filter(row => !row.passed)) output.append(ctx.el('p', { class: 'pf-code-failure' }, [ctx.el('strong', {}, `${translated(ctx, row.test.name)}: `), translated(ctx, row.test.purpose), row.error ? ` (${row.error})` : row.limited ? ctx.text(' The 5000-pop limit stopped this run; this is not a no-route result.', ' Pop 5000회 상한으로 중단했습니다. 경로가 없다는 결과와는 다릅니다.') : '']));
      output.append(ctx.el('p', { class: 'pf-code-counter-note' }, ctx.text('Expanded counts states whose successors were generated or examined; expanding the same state again counts again, but an accepted goal is excluded. Ties prefer lower h, then insertion order. The 40-expansion budget belongs only to this open 15×15 grid; it is not a universal A* speed guarantee. All runs stop after 5000 raw heap pops.', 'Expanded는 successor를 생성·검사한 state 수입니다. 같은 state를 재확장하면 다시 세고, 정답으로 받아들인 goal은 제외합니다. 동점이면 h가 작은 쪽, 그다음 먼저 넣은 쪽을 고릅니다. 40회 상한은 이 15×15 빈 grid에만 적용한 테스트 조건이며 A*의 일반적인 성능 보장은 아닙니다. 모든 실행에는 raw heap pop 5000회 상한이 있습니다.')));
      ctx.status.textContent = rows.every(row => row.passed) ? ctx.text('All eight tests passed. The cost checks and the one map-specific heuristic budget are satisfied; finite tests do not prove correctness on every graph.', '테스트 여덟 개를 모두 통과했습니다. 비용 검사와 해당 grid의 탐색 상한을 만족했습니다. 이 결과만으로 모든 graph에서의 정확성이 증명되는 것은 아닙니다.') : ctx.text('Read the cost and effort columns separately. Dijkstra can return every optimal cost and still miss the heuristic-use budget on the open grid.', '비용과 탐색 횟수를 나누어 확인해 보세요. Dijkstra는 모든 비용을 정확하게 구해도, 빈 grid에서 heuristic 활용을 확인하는 탐색 상한은 넘길 수 있습니다.');
    }
    function run() {
      lastDraft = draft();
      rows = judgeCodeCases(coding.cases, { priority: priority.value, relaxation: relaxation.value, goalCheck: goal.value });
      priority.setAttribute('aria-invalid', rows.some(row => row.error) ? 'true' : 'false');
      relaxation.setAttribute('aria-invalid', rows.some(row => row.error) ? 'true' : 'false');
      render();
    }
    ctx.button('Run coding tests', '코드 테스트 실행', run);
    ctx.button('Show / hide hint', '힌트 보기 / 접기', () => { showHint = !showHint; render(); });
    ctx.button('Load reference solution', '정답 코드 넣기', () => { priority.value = 'g + h'; relaxation.value = 'candidate < old_g'; goal.value = 'pop'; run(); });
    ctx.onReset(() => { priority.value = ''; relaxation.value = ''; goal.value = ''; showHint = false; rows = null; lastDraft = ''; priority.setAttribute('aria-invalid', 'false'); relaxation.setAttribute('aria-invalid', 'false'); });
    ctx.onChange(render);
    ctx.checkpointState({
      read: () => ({ priority: priority.value, relaxation: relaxation.value, goal: goal.value, showHint, tested: rows !== null }),
      write: s => {
        if (!s || ![s.priority, s.relaxation].every(v => typeof v === 'string' && v.length <= 80) || !['', 'pop', 'discovery'].includes(s.goal) || typeof s.showHint !== 'boolean' || typeof s.tested !== 'boolean') return false;
        priority.value = s.priority; relaxation.value = s.relaxation; goal.value = s.goal; showHint = s.showHint; rows = null; lastDraft = ''; if (s.tested) run(); else render(); return true;
      }
    });
  });
}());
