/* DFS construction/backtracking and BFS state-space puzzles. No services or code evaluation. */
(function () {
  'use strict';
  const edgeKey = (u, v) => `${Math.min(u, v)}:${Math.max(u, v)}`;
  function gridNeighbors(width, height, u) {
    const x = u % width, y = Math.floor(u / width), out = [];
    for (const [dx, dy] of [[1, 0], [0, 1], [-1, 0], [0, -1]]) if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) out.push((y + dy) * width + x + dx);
    return out;
  }
  class MazeCarver {
    constructor(width = 6, height = 4, seed = 17) {
      if (![width, height].every(n => Number.isInteger(n) && n >= 2 && n <= 9) || !Number.isInteger(seed) || seed < 0 || seed > 999999) throw new Error('Invalid maze dimensions or seed');
      this.width = width; this.height = height; this.seed = seed; this.randomState = seed >>> 0;
      this.visited = new Set([0]); this.stack = [0]; this.passages = new Set(); this.current = 0; this.steps = 0; this.backtracks = 0; this.done = false; this.event = { type: 'start', to: 0 };
    }
    random() { this.randomState = (1664525 * this.randomState + 1013904223) >>> 0; return this.randomState / 4294967296; }
    step() {
      if (this.done) return this.event;
      const u = this.stack[this.stack.length - 1], choices = gridNeighbors(this.width, this.height, u).filter(v => !this.visited.has(v));
      this.steps++;
      if (choices.length) {
        const v = choices[Math.floor(this.random() * choices.length)];
        this.passages.add(edgeKey(u, v)); this.visited.add(v); this.stack.push(v); this.current = v;
        this.event = { type: 'carve', from: u, to: v };
      } else {
        this.stack.pop(); this.backtracks++; this.current = this.stack.length ? this.stack[this.stack.length - 1] : null;
        this.event = { type: 'backtrack', from: u, to: this.current };
        this.done = this.stack.length === 0;
      }
      return this.event;
    }
    run(limit = 400) { for (let i = 0; !this.done && i < limit; i++) this.step(); if (!this.done) throw new Error('Maze step limit'); return this; }
    linked(u, v) { return this.passages.has(edgeKey(u, v)); }
  }
  function mazeRoute(maze, start = 0, goal = maze.width * maze.height - 1) {
    const queue = [start], parent = new Map([[start, null]]);
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head]; if (u === goal) { const path = []; for (let p = u; p !== null; p = parent.get(p)) path.push(p); return path.reverse(); }
      for (const v of gridNeighbors(maze.width, maze.height, u)) if (maze.linked(u, v) && !parent.has(v)) { parent.set(v, u); queue.push(v); }
    }
    return [];
  }
  function queenConflict(rows, row, column = rows.length) {
    if (!Array.isArray(rows) || !Number.isInteger(row) || !Number.isInteger(column) || column < 0 || column > rows.length) throw new Error('Invalid queen placement');
    for (let c = 0; c < column; c++) {
      if (rows[c] === row) return { type: 'row', column: c };
      if (Math.abs(rows[c] - row) === column - c) return { type: 'diagonal', column: c };
    }
    return null;
  }
  const queensSafe = rows => rows.every((r, c) => !queenConflict(rows, r, c));
  class QueensDFS {
    constructor(n = 4) {
      if (!Number.isInteger(n) || n < 4 || n > 6) throw new Error('Use N=4, 5 or 6');
      this.n = n; this.rows = []; this.nextRows = Array(n + 1).fill(0); this.solutions = []; this.checked = 0; this.pruned = 0; this.backtracks = 0; this.steps = 0; this.done = false; this.event = { type: 'start' };
    }
    step() {
      if (this.done) return this.event;
      const column = this.rows.length; this.steps++;
      if (column === this.n) {
        this.solutions.push(this.rows.slice()); this.event = { type: 'solution', rows: this.rows.slice() }; this.rows.pop(); this.backtracks++; return this.event;
      }
      if (this.nextRows[column] === this.n) {
        this.nextRows[column] = 0;
        if (column === 0) { this.done = true; this.event = { type: 'done' }; }
        else { const row = this.rows.pop(); this.backtracks++; this.event = { type: 'backtrack', row, column: column - 1 }; }
        return this.event;
      }
      const row = this.nextRows[column]++; this.checked++;
      const conflict = queenConflict(this.rows, row);
      if (conflict) { this.pruned++; this.event = { type: 'reject', row, column, conflict }; }
      else { this.rows.push(row); this.nextRows[column + 1] = 0; this.event = { type: 'place', row, column }; }
      return this.event;
    }
    run(limit = 10000) { for (let i = 0; !this.done && i < limit; i++) this.step(); if (!this.done) throw new Error('Queens step limit'); return this; }
  }
  const wordPresets = Object.freeze([
    { id: 'cold-warm', start: 'COLD', goal: 'WARM', words: ['COLD', 'CORD', 'CARD', 'WARD', 'WARM', 'WORD', 'WORM', 'WOOD', 'GOOD', 'GOAD', 'GOLD', 'BOLD', 'BALD', 'BALL', 'WALL', 'WALK', 'TALK', 'TALL', 'TAIL', 'MAIL', 'MAIN', 'PAIN', 'PAIR', 'FAIR', 'FAIL', 'FALL'] },
    { id: 'head-tail', start: 'HEAD', goal: 'TAIL', words: ['HEAD', 'HEAL', 'TEAL', 'TELL', 'TALL', 'TAIL', 'HEEL', 'FEEL', 'FELL', 'FALL', 'FAIL', 'MAIL', 'MEAL', 'SEAL', 'SEAT', 'HEAT', 'NEAT'] },
    { id: 'cat-dog', start: 'CAT', goal: 'DOG', words: ['CAT', 'COT', 'COG', 'DOG', 'DOT', 'CUT', 'CUP', 'PUP', 'POP', 'POT', 'TOP', 'HOP', 'HOT', 'HAT', 'BAT', 'BAG', 'BOG', 'BIG', 'DIG'] }
  ].map(p => Object.freeze({ ...p, words: Object.freeze(p.words.slice().sort()) })));
  function wordNeighbors(words, word) {
    if (!Array.isArray(words) || typeof word !== 'string') throw new Error('Invalid word graph');
    return words.filter(other => other.length === word.length && [...word].reduce((n, char, i) => n + (char !== other[i]), 0) === 1).sort();
  }
  const jugPresets = Object.freeze([
    Object.freeze({ id: '3-5-4', a: 3, b: 5, target: 4 }),
    Object.freeze({ id: '4-7-2', a: 4, b: 7, target: 2 }),
    Object.freeze({ id: '2-4-3', a: 2, b: 4, target: 3 })
  ]);
  function jugMoves(preset, state) {
    const { a: capA, b: capB } = preset, [a, b] = state;
    if (![capA, capB, a, b].every(Number.isInteger) || capA < 1 || capB < 1 || capA > 12 || capB > 12 || a < 0 || a > capA || b < 0 || b > capB) throw new Error('Invalid jug state');
    const ab = Math.min(a, capB - b), ba = Math.min(b, capA - a);
    const moves = [
      ['fill-a', [capA, b]], ['fill-b', [a, capB]], ['empty-a', [0, b]], ['empty-b', [a, 0]], ['pour-a-b', [a - ab, b + ab]], ['pour-b-a', [a + ba, b - ba]]
    ];
    return moves.filter(([, next]) => next[0] !== a || next[1] !== b).map(([action, next]) => ({ action, state: next }));
  }
  class GraphBFS {
    constructor(start, goalTest, successors, key = state => String(state)) {
      this.key = key; this.goalTest = goalTest; this.successors = successors; this.start = start;
      this.queue = [start]; this.head = 0; this.states = new Map([[key(start), start]]); this.parent = new Map([[key(start), null]]); this.depth = new Map([[key(start), 0]]); this.parentAction = new Map(); this.expanded = []; this.current = null; this.done = false; this.found = false; this.path = []; this.actions = [];
    }
    step() {
      if (this.done) return;
      if (this.head === this.queue.length) { this.done = true; this.current = null; return; }
      const u = this.queue[this.head++], uk = this.key(u); this.current = u;
      if (this.goalTest(u)) {
        this.done = true; this.found = true;
        for (let k = uk; k !== null; k = this.parent.get(k)) { this.path.push(this.states.get(k)); if (this.parentAction.has(k)) this.actions.push(this.parentAction.get(k)); }
        this.path.reverse(); this.actions.reverse(); return;
      }
      this.expanded.push(u);
      for (const { state: v, action } of this.successors(u)) {
        const vk = this.key(v); if (this.parent.has(vk)) continue;
        this.parent.set(vk, uk); this.depth.set(vk, this.depth.get(uk) + 1); this.states.set(vk, v); this.parentAction.set(vk, action); this.queue.push(v);
      }
    }
    run(limit = 1000) { for (let i = 0; !this.done && i < limit; i++) this.step(); if (!this.done) throw new Error('BFS step limit'); return this; }
    get frontier() { return this.queue.slice(this.head); }
    get cost() { return this.found ? this.path.length - 1 : Infinity; }
  }
  const wordBFS = p => new GraphBFS(p.start, w => w === p.goal, w => wordNeighbors(p.words, w).map(v => ({ state: v, action: v })));
  const jugBFS = p => new GraphBFS([0, 0], s => s.includes(p.target), s => jugMoves(p, s), s => s.join(','));
  const api = { MazeCarver, gridNeighbors, mazeRoute, queenConflict, queensSafe, QueensDFS, wordPresets, wordNeighbors, wordBFS, jugPresets, jugMoves, jugBFS, GraphBFS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const register = window.ArticleLabs.register;
  // Rebuild finite teaching models from bounded step counts rather than storing
  // class instances, Sets or functions in a local-editor checkpoint.
  function replay(model, steps, limit) {
    if (!Number.isInteger(steps) || steps < 0 || steps > limit) return null;
    for (let i = 0; i < steps; i++) { if (model.done) return null; model.step(); }
    return model;
  }
  const bfsSteps = search => search.head + (search.done && !search.found ? 1 : 0);
  const validMessage = value => typeof value === 'string' && value.length <= 600;
  function setup(ctx) { ctx.root.classList.add('su-lab', 'al-workshop'); }
  function label(ctx, en, ko, className = 'su-caption') { return ctx.el('p', { class: className }, ctx.text(en, ko)); }
  function choiceButtons(ctx, parent, items, selected, handler) {
    parent.replaceChildren(...items.map(item => {
      const b = ctx.el('button', { type: 'button', class: 'su-chip' + (item.value === selected ? ' su-active' : ''), 'aria-pressed': item.value === selected ? 'true' : 'false' }, ctx.text(item.en, item.ko || item.en));
      b.addEventListener('click', () => handler(item.value)); return b;
    }));
  }
  function clock(ctx, advance, paint, done) {
    let timer = null, button = null;
    const stop = () => { if (timer !== null) window.clearInterval(timer); timer = null; if (button) button.textContent = ctx.text('Run', '실행'); };
    const toggle = () => {
      if (timer !== null) { stop(); return; }
      if (done()) return;
      if (typeof window.setInterval !== 'function') { let count = 0; while (!done() && count++ < 10000) advance(); paint(); return; }
      button.textContent = ctx.text('Pause', '일시정지');
      timer = window.setInterval(() => { if (ctx.root.isConnected === false) { stop(); return; } advance(); paint(); if (done()) stop(); }, 110);
    };
    return { stop, toggle, attach: b => { button = b; } };
  }
  function keyboardGrid(button, cells, width, height, u) {
    button.addEventListener('click', () => { cells.forEach(c => c.tabIndex = -1); button.tabIndex = 0; });
    button.addEventListener('keydown', event => {
      const d = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowDown: [0, 1], ArrowUp: [0, -1] }[event.key]; if (!d) return;
      event.preventDefault(); const x = u % width + d[0], y = Math.floor(u / width) + d[1];
      if (x >= 0 && x < width && y >= 0 && y < height) { button.tabIndex = -1; const next = cells[y * width + x]; next.tabIndex = 0; next.focus(); }
    });
  }
  function pills(ctx, values, className = '') { return ctx.el('div', { class: 'su-pills ' + className }, values.map(v => ctx.el('span', { class: 'su-pill' }, v))); }
  function routeView(ctx, values) { return ctx.el('div', { class: 'su-route', 'aria-label': ctx.text('Current route', '현재 경로') }, values.flatMap((value, i) => [i ? ctx.el('span', { 'aria-hidden': 'true' }, '→') : null, ctx.el('span', { class: 'su-pill' }, value)])); }
  function paintLayers(ctx, parent, search, format) {
    const groups = new Map();
    search.depth.forEach((depth, key) => { if (!groups.has(depth)) groups.set(depth, []); groups.get(depth).push(search.states.get(key)); });
    const active = new Set(search.frontier.map(search.key)), expanded = new Set(search.expanded.map(search.key));
    parent.replaceChildren(...[...groups].map(([depth, states]) => ctx.el('div', { class: 'su-layer' }, [
      ctx.el('strong', {}, `d=${depth}`), ctx.el('div', { class: 'su-pills' }, states.map(s => ctx.el('span', { class: 'su-pill ' + (active.has(search.key(s)) ? 'su-frontier' : expanded.has(search.key(s)) ? 'su-expanded' : 'su-current') }, format(s))))
    ])));
  }
  register('search-dfs-maze', ctx => {
    setup(ctx);
    const size = ctx.control('Maze size', 'Maze 크기', { type: 'select', value: '6x4', options: [{ value: '4x4', en: '4 × 4 — tiny', ko: '4 × 4 · 작게' }, { value: '6x4', en: '6 × 4 — explore', ko: '6 × 4 · 기본' }, { value: '7x5', en: '7 × 5 — longer stack', ko: '7 × 5 · 더 긴 stack' }] });
    const seed = ctx.control('Random seed', 'Random seed', { type: 'number', value: 17, min: 0, max: 999999, step: 1 });
    const visitedMetric = ctx.metric('Visited / all cells', '방문한 cell / 전체'), stackMetric = ctx.metric('Stack depth / backtracks', 'Stack 깊이 / backtrack'), routeMetric = ctx.metric('Your route / unique route', '내 경로 / 유일한 경로');
    let maze, cells = [], signature = '', player = [0], revealed = false, message = '';
    const phase = label(ctx, 'Generate a maze, then walk through it.', 'Maze를 만든 다음 직접 통과해 보세요.', 'su-phase');
    const board = ctx.el('div', { class: 'su-maze-board', role: 'grid', 'aria-label': ctx.text('DFS maze generator and route game', 'DFS maze 생성과 경로 게임') });
    const trace = ctx.el('div', { class: 'su-stack' }), note = ctx.el('p', { class: 'su-caption' });
    ctx.view.append(phase, board, pills(ctx, [ctx.text('Gold @: current', '금색 @: current'), ctx.text('Blue: stack', '파랑: stack'), ctx.text('Gray: finished branch', '회색: 끝난 branch'), ctx.text('Green: your route', '초록: 내 경로')]), trace, note);
    const animation = clock(ctx, () => maze.step(), paint, () => maze.done);
    function coordinate(u) { return `(${u % maze.width},${Math.floor(u / maze.width)})`; }
    function resetMaze() {
      animation.stop(); const [w, h] = size.value.split('x').map(Number); maze = new MazeCarver(w, h, Number(seed.value)); signature = `${size.value}:${seed.value}`; player = [0]; revealed = false; message = '';
      cells = []; board.replaceChildren(); board.style?.setProperty('--su-cols', w);
      for (let u = 0; u < w * h; u++) {
        const b = ctx.el('button', { type: 'button', role: 'gridcell', class: 'su-maze-cell', tabindex: u === 0 ? '0' : '-1' });
        b.addEventListener('click', () => {
          if (!maze.done) { message = ctx.text('Finish generation first: the walls are still changing.', '아직 벽을 만드는 중입니다. 생성이 끝난 뒤 경로를 선택하세요.'); paint(); return; }
          const last = player[player.length - 1];
          if (u === last) return;
          if (player.length > 1 && u === player[player.length - 2]) { player.pop(); revealed = false; }
          else if (!revealed && !player.includes(u) && maze.linked(last, u)) player.push(u);
          else message = ctx.text('Choose a neighboring cell through an open passage. Click the previous cell to undo.', '열린 통로로 이어진 바로 옆 cell을 누르세요. 직전 cell을 누르면 한 칸 돌아갑니다.');
          if (maze.linked(last, u)) message = ''; paint();
        });
        keyboardGrid(b, cells, w, h, u); cells.push(b); board.append(b);
      }
    }
    ctx.button('Step DFS', 'DFS 한 단계', () => { animation.stop(); maze.step(); paint(); });
    const run = ctx.button('Run', '실행', () => animation.toggle()); animation.attach(run);
    ctx.button('Finish maze', 'Maze 완성', () => { animation.stop(); maze.run(); paint(); });
    ctx.button('New seed', '다른 seed', () => { seed.value = String((Number(seed.value) + 1) % 1000000); resetMaze(); paint(); });
    ctx.button('Undo route', '한 칸 돌아가기', () => { if (player.length > 1) player.pop(); revealed = false; message = ''; paint(); });
    ctx.button('Show route', '경로 보기', () => { animation.stop(); maze.run(); player = mazeRoute(maze); revealed = true; message = ''; paint(); });
    function paint() {
      const active = new Set(maze.stack), route = new Set(maze.done ? player : []), n = maze.width * maze.height;
      cells.forEach((cell, u) => {
        const x = u % maze.width, y = Math.floor(u / maze.width);
        cell.className = 'su-maze-cell ' + (route.has(u) ? 'su-path' : u === maze.current ? 'su-current' : active.has(u) ? 'su-frontier' : maze.visited.has(u) ? 'su-expanded' : '');
        for (const [side, v, outer] of [['right', u + 1, x === maze.width - 1], ['bottom', u + maze.width, y === maze.height - 1], ['left', u - 1, x === 0], ['top', u - maze.width, y === 0]]) cell.style?.setProperty(`border-${side}-color`, outer || !maze.linked(u, v) ? 'var(--su-wall)' : 'transparent');
        cell.textContent = u === 0 ? 'S' : u === n - 1 ? 'G' : u === maze.current ? '@' : '';
        cell.setAttribute('aria-label', `${coordinate(u)}${u === 0 ? ', start' : u === n - 1 ? ', goal' : ''}; ${maze.visited.has(u) ? 'visited' : 'unvisited'}`);
      });
      visitedMetric.textContent = `${maze.visited.size} / ${n}`; stackMetric.textContent = `${maze.stack.length} / ${maze.backtracks}`;
      const unique = maze.done ? mazeRoute(maze) : []; routeMetric.textContent = `${maze.done ? player.length - 1 : '—'} / ${unique.length ? unique.length - 1 : '—'}`;
      phase.textContent = maze.done ? ctx.text('Maze ready — click a passage from S toward G.', 'Maze 완성 · S에서 G까지 직접 이동해 보세요.') : ctx.text('Construction: DFS removes walls, not a route search.', '생성 중 · DFS로 벽을 없애고 있습니다. 경로 탐색과는 다릅니다.');
      trace.replaceChildren(label(ctx, 'Stack bottom → top (last 10 cells)', 'Stack 바닥 → 꼭대기 (마지막 10개)'), pills(ctx, maze.stack.slice(-10).map(coordinate)));
      const e = maze.event;
      note.textContent = maze.done ? ctx.text(`${n - 1} passages connect ${n} cells: one simple route joins every pair. Step/backtrack actions: ${maze.steps}.`, `${n}개 cell을 ${n - 1}개 통로로 연결했습니다. 어떤 두 cell 사이에도 simple route가 하나뿐입니다. DFS 동작은 ${maze.steps}번입니다.`) : e.type === 'carve' ? ctx.text(`Unvisited neighbor ${coordinate(e.to)}: carve one wall and push it.`, `아직 방문하지 않은 ${coordinate(e.to)} 쪽 벽을 없애고 stack에 넣습니다.`) : e.type === 'backtrack' ? ctx.text(`No unvisited neighbor at ${coordinate(e.from)}: pop and return.`, `${coordinate(e.from)}에는 방문하지 않은 이웃이 없어 pop하고 돌아갑니다.`) : ctx.text('Every wall starts closed. Choose Step DFS to carve the first passage.', '처음에는 모든 벽이 닫혀 있습니다. DFS 한 단계를 눌러 첫 통로를 만드세요.');
      ctx.status.textContent = message || (maze.done && player[player.length - 1] === n - 1 ? revealed ? ctx.text('The unique route is revealed. Reset for a fresh attempt.', '유일한 경로를 표시했습니다. 초기화하면 다시 도전할 수 있습니다.') : ctx.text(`Reached G in ${player.length - 1} passages. The completed maze is a tree.`, `${player.length - 1}개 통로를 지나 G에 도착했습니다. 완성된 maze는 tree입니다.`) : ctx.text('Arrow keys move focus; Enter/Space selects a cell. A repeated seed makes exactly the same maze.', '방향키로 cell을 고르고 Enter/Space로 누를 수 있습니다. Seed가 같으면 같은 maze가 만들어집니다.'));
    }
    ctx.onReset(resetMaze);
    ctx.onChange(() => { if (signature !== `${size.value}:${seed.value}`) resetMaze(); paint(); });
    ctx.checkpointState({
      read: () => ({ size: size.value, seed: Number(seed.value), steps: maze.steps, player: player.slice(), revealed, message }),
      write: s => {
        if (!s || !['4x4', '6x4', '7x5'].includes(s.size) || !Number.isInteger(s.seed) || s.seed < 0 || s.seed > 999999 || typeof s.revealed !== 'boolean' || !validMessage(s.message)) return false;
        const [w, h] = s.size.split('x').map(Number), next = replay(new MazeCarver(w, h, s.seed), s.steps, 2 * w * h - 1);
        if (!next || !Array.isArray(s.player) || !s.player.length || s.player.length > w * h || s.player[0] !== 0 || new Set(s.player).size !== s.player.length || !s.player.every(u => Number.isInteger(u) && u >= 0 && u < w * h) || s.player.some((u, i) => i && !next.linked(s.player[i - 1], u)) || !next.done && s.player.length !== 1 || s.revealed && (!next.done || JSON.stringify(s.player) !== JSON.stringify(mazeRoute(next)))) return false;
        size.value = s.size; seed.value = String(s.seed); resetMaze(); maze = next; player = s.player.slice(); revealed = s.revealed; message = s.message; paint(); return true;
      }
    });
  });
  register('search-dfs-queens', ctx => {
    setup(ctx);
    const size = ctx.control('Board size N', 'Board 크기 N', { type: 'select', value: '4', options: [4, 5, 6].map(n => ({ value: String(n), en: `${n} × ${n}` })) });
    const queensMetric = ctx.metric('Placed / N', '놓은 Queen / N'), trialsMetric = ctx.metric('DFS checks / pruned', 'DFS 검사 / pruning'), solutionsMetric = ctx.metric('DFS solutions / backtracks', 'DFS 해 / backtrack');
    let solver, manual = [], mode = 'manual', cells = [], signature = '', warning = '', rejected = null, selectedSolution = null;
    const phase = ctx.el('p', { class: 'su-phase' }), board = ctx.el('div', { class: 'su-queen-board', role: 'grid', 'aria-label': ctx.text('N-Queens constraint board', 'N-Queens constraint board') }), detail = ctx.el('p', { class: 'su-caption' });
    const gallery = ctx.el('div', { class: 'su-saved-solutions', hidden: true }), galleryOptions = ctx.el('div', { class: 'su-solution-grid', role: 'group', 'aria-label': ctx.text('Saved queen solutions', '저장한 Queen 배치') });
    gallery.append(label(ctx, 'Saved solutions — choose a placement to inspect', '저장한 해 · 배치를 골라 확인하세요.'), galleryOptions);
    ctx.view.append(phase, board, pills(ctx, [ctx.text('♛: queen', '♛: Queen'), ctx.text('·: safe in next column', '·: 다음 column에서 안전'), ctx.text('×: pruned square', '×: 놓을 수 없음')]), detail, gallery);
    const animation = clock(ctx, () => { mode = 'solver'; solver.step(); }, paint, () => solver.done);
    function rows() { return mode === 'manual' ? manual : selectedSolution !== null ? solver.solutions[selectedSolution] : solver.event.type === 'solution' ? solver.event.rows : solver.rows; }
    function resetBoard() {
      animation.stop(); signature = size.value; solver = new QueensDFS(Number(size.value)); manual = []; mode = 'manual'; warning = ''; rejected = null; selectedSolution = null; cells = []; board.replaceChildren(); board.style?.setProperty('--su-cols', solver.n);
      for (let r = 0; r < solver.n; r++) for (let c = 0; c < solver.n; c++) {
        const u = r * solver.n + c, b = ctx.el('button', { type: 'button', role: 'gridcell', class: 'su-queen-cell', tabindex: u === 0 ? '0' : '-1' });
        b.addEventListener('click', () => {
          if (mode !== 'manual') { warning = ctx.text('Choose Your turn to place queens yourself; the solver and saved-solution boards are read-only.', '직접 놓으려면 직접 도전을 누르세요. Solver의 state와 저장한 해는 읽기 전용입니다.'); paint(); return; }
          rejected = null;
          if (manual.length === solver.n) warning = ctx.text('The board is solved. Undo one queen or choose Your turn for another attempt.', '모든 Queen을 놓았습니다. 하나를 되돌리거나 직접 도전을 눌러 다른 배치를 찾아보세요.');
          else if (c !== manual.length) warning = ctx.text(`Place the next queen in column ${manual.length + 1}. One queen per column is already enforced.`, `다음 Queen은 ${manual.length + 1}번째 column에 놓으세요. Column당 하나씩 놓는 규칙은 이미 적용돼 있습니다.`);
          else {
            const conflict = queenConflict(manual, r);
            if (conflict) { rejected = [r, c]; warning = ctx.text(`Prune: same ${conflict.type} as the queen in column ${conflict.column + 1}.`, `${conflict.column + 1}번째 column의 Queen과 ${conflict.type === 'row' ? 'row' : 'diagonal'}가 겹칩니다. 이 후보는 pruning합니다.`); }
            else { manual.push(r); warning = ''; }
          }
          paint();
        });
        keyboardGrid(b, cells, solver.n, solver.n, u); cells.push(b); board.append(b);
      }
    }
    function prepareSolver() { warning = ''; rejected = null; if (mode !== 'solver') { animation.stop(); solver = new QueensDFS(Number(size.value)); mode = 'solver'; selectedSolution = null; } if (!solver.done) selectedSolution = null; }
    ctx.button('Step DFS', 'DFS 한 단계', () => { animation.stop(); prepareSolver(); solver.step(); paint(); });
    const run = ctx.button('Run', '실행', () => { prepareSolver(); animation.toggle(); paint(); }); animation.attach(run);
    ctx.button('Find all solutions', '모든 해 찾기', () => { animation.stop(); prepareSolver(); solver.run(); paint(); });
    ctx.button('Your turn', '직접 도전', () => { animation.stop(); mode = 'manual'; manual = []; warning = ''; rejected = null; selectedSolution = null; paint(); });
    ctx.button('Undo queen', 'Queen 하나 되돌리기', () => { if (mode === 'manual') { manual.pop(); warning = ''; rejected = null; paint(); } });
    function paint() {
      if (mode === 'solver' && solver.done && solver.solutions.length && selectedSolution === null) selectedSolution = 0;
      const placed = rows(), next = placed.length, event = solver.event;
      cells.forEach((cell, u) => {
        const r = Math.floor(u / solver.n), c = u % solver.n, queen = placed[c] === r, conflict = c === next ? queenConflict(placed, r) : null;
        const isRejected = mode === 'manual' ? rejected && rejected[0] === r && rejected[1] === c : selectedSolution === null && event.type === 'reject' && event.row === r && event.column === c;
        cell.className = `su-queen-cell ${(r + c) % 2 ? 'su-dark-square' : ''}${c === next ? ' su-next-column' : ''}${queen ? ' su-queen' : ''}${isRejected ? ' su-reject' : ''}`;
        cell.textContent = queen ? '♛' : c === next ? conflict ? '×' : '·' : '';
        cell.setAttribute('aria-label', ctx.text(`Row ${r + 1}, column ${c + 1}${queen ? ', queen' : c === next ? conflict ? ', conflict' : ', safe candidate' : ''}`, `${r + 1}행 ${c + 1}열${queen ? ', Queen' : c === next ? conflict ? ', 충돌' : ', 안전한 후보' : ''}`));
      });
      gallery.hidden = mode !== 'solver' || solver.solutions.length === 0;
      choiceButtons(ctx, galleryOptions, mode === 'solver' ? solver.solutions.map((_solution, i) => ({ value: String(i), en: `Solution ${i + 1}`, ko: `해 ${i + 1}` })) : [], selectedSolution === null ? '' : String(selectedSolution), value => {
        animation.stop(); selectedSolution = Number(value); warning = ''; rejected = null; paint(); galleryOptions.children[selectedSolution]?.focus();
      });
      queensMetric.textContent = `${placed.length} / ${solver.n}`; trialsMetric.textContent = `${solver.checked} / ${solver.pruned}`; solutionsMetric.textContent = `${solver.solutions.length} / ${solver.backtracks}`;
      phase.textContent = mode === 'manual' ? next === solver.n ? ctx.text('Solved! No queens attack one another.', '성공! 서로 공격하는 Queen이 없습니다.') : ctx.text(`Your turn — choose a safe square in column ${next + 1}.`, `직접 도전 · ${next + 1}번째 column의 안전한 칸을 고르세요.`) : selectedSolution !== null ? ctx.text(`Saved solution ${selectedSolution + 1} of ${solver.solutions.length}: [${placed.map(r => r + 1).join(', ')}].`, `저장한 해 ${selectedSolution + 1} / ${solver.solutions.length}: [${placed.map(r => r + 1).join(', ')}].`) : ctx.text(`DFS state: [${placed.map(r => r + 1).join(', ')}] — row per column, numbered from 1.`, `DFS state: [${placed.map(r => r + 1).join(', ')}] · 각 column의 row를 1부터 적었습니다.`);
      detail.textContent = mode === 'manual' ? ctx.text('A dead end is useful: undo one queen and try another row. The solver starts independently from the empty board.', '더 놓을 칸이 없으면 Queen 하나를 되돌리고 다른 row를 골라 보세요. Solver는 빈 board에서 별도로 시작합니다.') : selectedSolution !== null ? ctx.text(`Read-only saved placement, not the current DFS stack. The list gives each column's row, numbered from 1. ${solver.done ? 'Choose another saved solution to compare; the search is complete.' : 'Step DFS or Run returns to the paused search without changing its counts.'}`, `현재 DFS stack이 아니라 저장한 배치입니다. 각 column의 row를 1부터 적었습니다. ${solver.done ? '다른 해를 골라 비교해 보세요. 탐색은 이미 끝났습니다.' : 'DFS 한 단계나 실행을 누르면 멈춘 탐색을 이어 갑니다. 배치를 보는 동안 탐색 횟수는 바뀌지 않습니다.'}`) : event.type === 'reject' ? ctx.text(`Check (${event.row + 1},${event.column + 1}): reject a ${event.conflict.type} conflict before creating any descendants.`, `(${event.row + 1},${event.column + 1}) 후보는 ${event.conflict.type === 'row' ? 'row' : 'diagonal'}가 겹칩니다. 그 아래 state는 만들지 않습니다.`) : event.type === 'place' ? ctx.text(`Safe row ${event.row + 1}: push a queen and descend to the next column.`, `${event.row + 1}번째 row는 안전합니다. Queen을 놓고 다음 column으로 내려갑니다.`) : event.type === 'solution' ? ctx.text(`Solution ${solver.solutions.length}: save this complete placement; the next step continues backtracking.`, `${solver.solutions.length}번째 해를 저장했습니다. 다음 단계에서는 backtracking을 계속합니다.`) : event.type === 'backtrack' ? ctx.text('All rows tried in this column: remove the previous queen and resume its next row.', '이 column의 row를 모두 검사했습니다. 이전 Queen을 되돌리고 다음 row를 검사합니다.') : ctx.text('Each step checks one candidate or makes one return; no unsafe partial placement is expanded.', '한 단계마다 후보 하나를 검사하거나 한 번 되돌아갑니다. 충돌하는 partial placement에서는 더 내려가지 않습니다.');
      ctx.status.textContent = warning || (mode === 'solver' && solver.done ? ctx.text(`Search exhausted: ${solver.solutions.length} solutions. Counts include rotated and reflected placements separately.`, `탐색을 마쳤습니다. 해는 ${solver.solutions.length}개입니다. 회전·대칭으로 겹치는 배치도 각각 셉니다.`) : ctx.text('Search state is a partial placement, not a chessboard square. DFS explores the decision tree; constraints prune branches.', 'Search state는 체스판의 칸이 아니라 Queen의 partial placement입니다. DFS로 decision tree를 탐색하고 constraint로 branch를 줄입니다.'));
    }
    ctx.onReset(resetBoard);
    ctx.onChange(() => { if (signature !== size.value) resetBoard(); paint(); });
    ctx.checkpointState({
      read: () => ({ size: Number(size.value), steps: solver.steps, manual: manual.slice(), mode, selectedSolution, rejected, warning }),
      write: s => {
        if (!s || ![4, 5, 6].includes(s.size) || !['manual', 'solver'].includes(s.mode) || !validMessage(s.warning) || !Array.isArray(s.manual) || s.manual.length > s.size || !s.manual.every(r => Number.isInteger(r) && r >= 0 && r < s.size) || !queensSafe(s.manual) || s.rejected !== null && (!Array.isArray(s.rejected) || s.rejected.length !== 2 || !s.rejected.every(r => Number.isInteger(r) && r >= 0 && r < s.size))) return false;
        const next = replay(new QueensDFS(s.size), s.steps, 10000);
        if (!next || s.selectedSolution !== null && (!Number.isInteger(s.selectedSolution) || s.selectedSolution < 0 || s.selectedSolution >= next.solutions.length || s.mode !== 'solver')) return false;
        size.value = String(s.size); resetBoard(); solver = next; manual = s.manual.slice(); mode = s.mode; selectedSolution = s.selectedSolution; rejected = s.rejected && s.rejected.slice(); warning = s.warning; paint(); return true;
      }
    });
  });
  register('search-bfs-words', ctx => {
    setup(ctx);
    const presetField = ctx.control('Word challenge', 'Word 문제', { type: 'select', value: wordPresets[0].id, options: wordPresets.map(p => ({ value: p.id, en: `${p.start} → ${p.goal}` })) });
    const movesMetric = ctx.metric('Your moves / shortest', '내 이동 / 최단 이동'), bfsMetric = ctx.metric('BFS expanded / frontier', 'BFS expanded / frontier'), wordsMetric = ctx.metric('Dictionary / discovered', 'Dictionary / 발견한 word');
    let preset, chain, search, reference, signature = '', message = '';
    const challenge = ctx.el('p', { class: 'su-phase' }), route = ctx.el('div'), moves = ctx.el('div', { class: 'su-word-options' }), solution = ctx.el('div'), layers = ctx.el('div', { class: 'su-layers' }), queue = ctx.el('p', { class: 'su-caption' });
    const dictionary = ctx.el('details', { class: 'su-dictionary' }, ctx.el('summary', {}, ctx.text('Show the complete dictionary', '전체 dictionary 보기'))), words = ctx.el('div'); dictionary.append(words);
    ctx.view.append(challenge, route, label(ctx, 'Choose one neighboring word:', '한 글자만 다른 word를 고르세요.'), moves, solution, label(ctx, 'BFS layers — blue frontier; gray expanded', 'BFS layer · 파랑은 frontier, 회색은 expanded'), layers, queue, dictionary);
    const animation = clock(ctx, () => search.step(), paint, () => search.done);
    function resetWords() { animation.stop(); preset = wordPresets.find(p => p.id === presetField.value); if (!preset) throw new Error('Unknown word preset'); signature = preset.id; chain = [preset.start]; search = wordBFS(preset); reference = wordBFS(preset).run(); message = ''; }
    ctx.button('Undo move', '한 번 되돌리기', () => { if (chain.length > 1) chain.pop(); message = ''; paint(); });
    ctx.button('Restart chain', '처음 word부터', () => { chain = [preset.start]; message = ''; paint(); });
    ctx.button('Step BFS', 'BFS 한 단계', () => { animation.stop(); search.step(); paint(); });
    const run = ctx.button('Run', '실행', () => animation.toggle()); animation.attach(run);
    ctx.button('Reveal shortest chain', '최단 chain 보기', () => { animation.stop(); search.run(); paint(); });
    function paint() {
      const current = chain[chain.length - 1], won = current === preset.goal;
      challenge.textContent = ctx.text(`${preset.start} → ${preset.goal}: replace one letter per move.`, `${preset.start} → ${preset.goal} · 한 번에 한 글자만 바꾸세요.`);
      route.replaceChildren(routeView(ctx, chain));
      choiceButtons(ctx, moves, wordNeighbors(preset.words, current).map(w => ({ value: w, en: w })), '', w => { if (chain.length > 40) { message = ctx.text('40 moves reached. Restart or undo to continue.', '40번까지 이동했습니다. 되돌리거나 처음부터 다시 시작하세요.'); } else { chain.push(w); message = ''; } paint(); });
      words.replaceChildren(pills(ctx, preset.words));
      solution.replaceChildren(...(search.found ? [label(ctx, `BFS shortest chain (${search.cost} moves)`, `BFS 최단 chain (${search.cost}번 이동)`), routeView(ctx, search.path)] : []));
      paintLayers(ctx, layers, search, x => x);
      queue.textContent = ctx.text(`FIFO, next → last: ${search.frontier.slice(0, 12).join(' → ') || '∅'}${search.frontier.length > 12 ? ' …' : ''}`, `FIFO, 다음 → 마지막: ${search.frontier.slice(0, 12).join(' → ') || '∅'}${search.frontier.length > 12 ? ' …' : ''}`);
      movesMetric.textContent = `${chain.length - 1} / ${reference.cost}`; bfsMetric.textContent = `${search.expanded.length} / ${search.frontier.length}`; wordsMetric.textContent = `${preset.words.length} / ${search.states.size}`;
      ctx.status.textContent = message || (won ? chain.length - 1 === reference.cost ? ctx.text('Shortest chain found! Every move has cost 1, so BFS provides the minimum number of changes.', '최단 chain을 찾았습니다! 모든 이동 비용이 1이므로 BFS의 이동 횟수가 최솟값입니다.') : ctx.text(`You reached the goal. Can you reduce ${chain.length - 1} moves to ${reference.cost}?`, `Goal에 도착했습니다. ${chain.length - 1}번을 ${reference.cost}번으로 줄일 수 있을까요?`) : ctx.text('Only the listed dictionary words exist as vertices. BFS marks words when enqueued, and tests the goal when dequeued.', '이 dictionary에 있는 word만 vertex로 취급합니다. BFS는 queue에 넣을 때 discovered로 표시하고 꺼낼 때 goal인지 확인합니다.'));
    }
    ctx.onReset(resetWords); ctx.onChange(() => { if (signature !== presetField.value) resetWords(); paint(); });
    ctx.checkpointState({
      read: () => ({ preset: preset.id, steps: bfsSteps(search), chain: chain.slice(), message }),
      write: s => {
        const p = s && wordPresets.find(p => p.id === s.preset);
        if (!p || !validMessage(s.message) || !Array.isArray(s.chain) || !s.chain.length || s.chain.length > 41 || s.chain[0] !== p.start || s.chain.some((w, i) => !p.words.includes(w) || i && !wordNeighbors(p.words, s.chain[i - 1]).includes(w))) return false;
        const next = replay(wordBFS(p), s.steps, p.words.length + 1); if (!next) return false;
        presetField.value = p.id; resetWords(); search = next; chain = s.chain.slice(); message = s.message; paint(); return true;
      }
    });
  });
  const jugActionNames = {
    'fill-a': ['Fill A', 'A 채우기'], 'fill-b': ['Fill B', 'B 채우기'], 'empty-a': ['Empty A', 'A 비우기'], 'empty-b': ['Empty B', 'B 비우기'], 'pour-a-b': ['Pour A → B', 'A → B 붓기'], 'pour-b-a': ['Pour B → A', 'B → A 붓기']
  };
  register('search-bfs-jugs', ctx => {
    setup(ctx);
    const presetField = ctx.control('Jug challenge', 'Jug 문제', { type: 'select', value: jugPresets[0].id, options: jugPresets.map(p => ({ value: p.id, en: `${p.a}L + ${p.b}L → measure ${p.target}L`, ko: `${p.a}L + ${p.b}L → ${p.target}L 만들기` })) });
    const movesMetric = ctx.metric('Your actions / minimum', '내 동작 / 최소 동작'), stateMetric = ctx.metric('Current state (A, B)', '현재 state (A, B)'), bfsMetric = ctx.metric('BFS expanded / frontier', 'BFS expanded / frontier');
    let preset, history, search, reference, signature = '', message = '';
    const challenge = ctx.el('p', { class: 'su-phase' }), vessels = ctx.el('div', { class: 'su-vessels' }), actions = ctx.el('div', { class: 'su-jug-actions' }), route = ctx.el('div'), solution = ctx.el('div'), layers = ctx.el('div', { class: 'su-layers' });
    ctx.view.append(challenge, vessels, actions, route, solution, label(ctx, 'BFS layers — states are pairs of volumes', 'BFS layer · state는 두 volume의 쌍'), layers);
    const animation = clock(ctx, () => search.step(), paint, () => search.done);
    function resetJugs() { animation.stop(); preset = jugPresets.find(p => p.id === presetField.value); if (!preset) throw new Error('Unknown jug preset'); signature = preset.id; history = [{ state: [0, 0], action: null }]; search = jugBFS(preset); reference = jugBFS(preset).run(); message = ''; }
    ctx.button('Undo action', '한 번 되돌리기', () => { if (history.length > 1) history.pop(); message = ''; paint(); });
    ctx.button('Empty both / restart', '둘 다 비우고 다시', () => { history = [{ state: [0, 0], action: null }]; message = ''; paint(); });
    ctx.button('Step BFS', 'BFS 한 단계', () => { animation.stop(); search.step(); paint(); });
    const run = ctx.button('Run', '실행', () => animation.toggle()); animation.attach(run);
    ctx.button('Reveal shortest plan', '최단 plan 보기', () => { animation.stop(); search.run(); paint(); });
    function jugDrawing(name, capacity, volume) {
      const height = 130 * capacity / Math.max(preset.a, preset.b), top = 154 - height, waterTop = 154 - height * volume / capacity;
      const svg = ctx.svg('svg', { viewBox: '0 0 120 172', role: 'img', 'aria-label': `${name}: ${volume} L / ${capacity} L`, class: 'su-jug-svg' });
      svg.append(ctx.svg('rect', { x: 15, y: waterTop, width: 90, height: height * volume / capacity, rx: 0, class: 'su-water' }), ctx.svg('path', { d: `M15,${top}V154H105V${top}`, class: 'su-jug-outline' }));
      for (let v = 1; v < capacity; v++) svg.append(ctx.svg('line', { x1: 15, x2: 24, y1: 154 - height * v / capacity, y2: 154 - height * v / capacity, class: 'su-jug-tick' }));
      return ctx.el('div', { class: 'su-vessel' }, [ctx.el('strong', {}, `${name} · ${capacity} L`), svg, ctx.el('output', {}, `${volume} L`)]);
    }
    function actionName(id) { return ctx.text(...jugActionNames[id]); }
    function paint() {
      const current = history[history.length - 1].state, legal = jugMoves(preset, current), won = current.includes(preset.target), minimum = reference.found ? reference.cost : '∞';
      challenge.textContent = ctx.text(`Measure exactly ${preset.target} L in either jug.`, `어느 한 jug에 정확히 ${preset.target}L를 담으세요.`);
      vessels.replaceChildren(jugDrawing('A', preset.a, current[0]), jugDrawing('B', preset.b, current[1]));
      actions.replaceChildren(...Object.entries(jugActionNames).map(([id, names]) => {
        const transition = legal.find(t => t.action === id), button = ctx.el('button', { type: 'button', class: 'su-chip', disabled: !transition }, ctx.text(...names));
        button.addEventListener('click', () => { if (!transition) return; if (history.length > 40) message = ctx.text('40 actions reached. Undo or restart.', '40번까지 동작했습니다. 되돌리거나 다시 시작하세요.'); else { history.push(transition); message = ''; } paint(); }); return button;
      }));
      route.replaceChildren(label(ctx, 'Your last 8 states (each arrow costs 1)', '최근 state 8개 (화살표마다 비용 1)'), routeView(ctx, history.slice(-8).map(t => `(${t.state.join(',')})`)));
      const rows = search.found ? search.path.map((state, i) => [i, i ? actionName(search.actions[i - 1]) : ctx.text('Start', '시작'), `(${state.join(',')})`]) : [];
      solution.replaceChildren(...(search.done ? search.found ? [label(ctx, `Shortest plan: ${search.cost} actions`, `최단 plan: ${search.cost}번 동작`), ctx.table(['#', ctx.text('Action', '동작'), '(A, B) L'], rows)] : [label(ctx, 'No solution: BFS exhausted every reachable state.', '해가 없습니다. BFS가 도달할 수 있는 state를 모두 검사했습니다.')] : []));
      paintLayers(ctx, layers, search, s => `(${s.join(',')})`);
      movesMetric.textContent = `${history.length - 1} / ${minimum}`; stateMetric.textContent = `(${current.join(', ')}) L`; bfsMetric.textContent = `${search.expanded.length} / ${search.frontier.length}`;
      ctx.status.textContent = message || (won ? history.length - 1 === reference.cost ? ctx.text('Exactly measured, using a minimum-length plan!', '정확히 만들었습니다. 동작 횟수도 최소입니다!') : ctx.text(`Target reached in ${history.length - 1} actions. The minimum is ${reference.cost}.`, `${history.length - 1}번 동작해 목표를 만들었습니다. 최소 횟수는 ${reference.cost}번입니다.`) : ctx.text('Fill completely, empty completely, or pour until the source empties or destination fills. Unchanged actions are disabled. Every allowed action costs 1, regardless of liters moved.', '가득 채우기, 완전히 비우기, 한쪽이 비거나 다른 쪽이 찰 때까지 붓기만 가능합니다. State가 바뀌지 않는 동작은 꺼져 있습니다. 옮긴 양과 관계없이 동작 한 번의 비용은 1입니다.'));
    }
    ctx.onReset(resetJugs); ctx.onChange(() => { if (signature !== presetField.value) resetJugs(); paint(); });
    ctx.checkpointState({
      read: () => ({ preset: preset.id, steps: bfsSteps(search), actions: history.slice(1).map(t => t.action), message }),
      write: s => {
        const p = s && jugPresets.find(p => p.id === s.preset);
        if (!p || !validMessage(s.message) || !Array.isArray(s.actions) || s.actions.length > 40) return false;
        const nextHistory = [{ state: [0, 0], action: null }];
        for (const action of s.actions) { const transition = jugMoves(p, nextHistory[nextHistory.length - 1].state).find(t => t.action === action); if (!transition) return false; nextHistory.push(transition); }
        const next = replay(jugBFS(p), s.steps, (p.a + 1) * (p.b + 1) + 1); if (!next) return false;
        presetField.value = p.id; resetJugs(); search = next; history = nextHistory; message = s.message; paint(); return true;
      }
    });
  });
})();
