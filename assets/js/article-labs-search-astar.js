/* Bounded, browser-local state-space games. The same pure engines run in Node tests. */
(function () {
  'use strict';
  const GOAL = '123456780', POP_LIMIT = 50000;
  const PRESETS = Object.freeze({ four: '123506478', eight: '123568047', twelve: '123687540' });
  const DEPTHS = Object.freeze({ four: 4, eight: 8, twelve: 12 });
  class Heap {
    constructor(compare) { this.items = []; this.compare = compare; }
    push(value) {
      const a = this.items; let i = a.length; a.push(value);
      while (i) { const p = (i - 1) >> 1; if (this.compare(a[p], value) <= 0) break; a[i] = a[p]; i = p; } a[i] = value;
    }
    pop() {
      if (!this.items.length) return undefined;
      const a = this.items, first = a[0], last = a.pop();
      if (a.length) { let i = 0; while (true) { let c = 2 * i + 1; if (c >= a.length) break; if (c + 1 < a.length && this.compare(a[c + 1], a[c]) < 0) c++; if (this.compare(last, a[c]) <= 0) break; a[i] = a[c]; i = c; } a[i] = last; }
      return first;
    }
  }
  function validatePuzzle(state) {
    if (typeof state !== 'string' || !/^[0-8]{9}$/.test(state) || new Set(state).size !== 9) throw new Error('Expected tiles 0–8 exactly once.');
    const a = [...state].filter(x => x !== '0'); let inversions = 0;
    a.forEach((x, i) => a.slice(i + 1).forEach(y => { if (x > y) inversions++; }));
    if (inversions % 2) throw new Error('This arrangement is not reachable from the goal.');
    return true;
  }
  function puzzleNeighbors(state) {
    const z = state.indexOf('0'), x = z % 3, y = Math.floor(z / 3), out = [];
    for (const [dx, dy, direction] of [[1, 0, 'right'], [0, 1, 'down'], [-1, 0, 'left'], [0, -1, 'up']]) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx > 2 || ny < 0 || ny > 2) continue;
      const j = 3 * ny + nx, a = [...state]; [a[z], a[j]] = [a[j], a[z]];
      out.push({ state: a.join(''), cost: 1, tile: Number(state[j]), index: j, direction });
    }
    return out;
  }
  function manhattan(state) {
    let h = 0;
    [...state].forEach((v, i) => { if (v !== '0') { const j = Number(v) - 1; h += Math.abs(i % 3 - j % 3) + Math.abs(Math.floor(i / 3) - Math.floor(j / 3)); } });
    return h;
  }
  class PuzzleSearch {
    constructor(start, kind = 'manhattan', limit = POP_LIMIT) {
      validatePuzzle(start);
      if (!['manhattan', 'zero'].includes(kind) || !Number.isInteger(limit) || limit < 1 || limit > POP_LIMIT) throw new Error('Invalid solver setting.');
      this.start = start; this.kind = kind; this.limit = limit; this.g = new Map([[start, 0]]); this.parent = new Map(); this.closed = new Set();
      this.sequence = 0; this.expanded = 0; this.pops = 0; this.reopened = 0; this.stale = 0; this.done = false; this.found = false; this.limitReached = false; this.current = null; this.path = []; this.trace = [];
      this.heap = new Heap((a, b) => a.f - b.f || a.h - b.h || a.sequence - b.sequence); this.insert(start);
    }
    h(state) { return this.kind === 'zero' ? 0 : manhattan(state); }
    insert(state) { const g = this.g.get(state), h = this.h(state); this.heap.push({ state, g, h, f: g + h, sequence: this.sequence++ }); }
    entries() {
      const seen = new Set();
      return this.heap.items.slice().sort(this.heap.compare).filter(e => !this.closed.has(e.state) && e.g === this.g.get(e.state) && !seen.has(e.state) && seen.add(e.state));
    }
    step() {
      if (this.done) return null;
      if (this.pops >= this.limit) { this.done = true; this.limitReached = true; return null; }
      let entry;
      do { entry = this.heap.pop(); if (!entry) { this.done = true; return null; } if (this.closed.has(entry.state) || entry.g !== this.g.get(entry.state)) { this.stale++; entry = null; } } while (!entry);
      this.current = entry; this.pops++; this.trace.push(entry);
      if (entry.state === GOAL) {
        this.done = this.found = true;
        for (let s = GOAL; s !== undefined; s = this.parent.get(s)) this.path.push(s);
        this.path.reverse(); return entry;
      }
      this.closed.add(entry.state); this.expanded++;
      for (const { state, cost } of puzzleNeighbors(entry.state)) {
        const tentative = entry.g + cost;
        if (tentative < (this.g.get(state) ?? Infinity)) { this.g.set(state, tentative); this.parent.set(state, entry.state); if (this.closed.delete(state)) this.reopened++; this.insert(state); }
      }
      return entry;
    }
    run() { while (!this.done) this.step(); return this; }
    get cost() { return this.found ? this.g.get(GOAL) : Infinity; }
  }
  const NODES = Object.freeze(['S', 'A', 'B', 'G']);
  const EDGES = Object.freeze([['S', 'A', 3], ['S', 'B', 1], ['B', 'A', 1], ['A', 'G', 3], ['B', 'G', 100]].map(Object.freeze));
  const HSTAR = Object.freeze({ S: 5, A: 3, B: 4, G: 0 });
  const HVALUES = Object.freeze({ zero: Object.freeze({ S: 0, A: 0, B: 0, G: 0 }), exact: HSTAR, inconsistent: Object.freeze({ S: 5, A: 0, B: 4, G: 0 }), overestimate: Object.freeze({ S: 0, A: 0, B: 8, G: 0 }) });
  function inspectHeuristic(h) {
    if (!h || !NODES.every(n => Number.isFinite(h[n]) && h[n] >= 0) || h.G !== 0) throw new Error('Invalid heuristic.');
    return { admissible: NODES.every(n => h[n] <= HSTAR[n]), consistent: EDGES.every(([u, v, c]) => h[u] <= c + h[v]), violations: EDGES.filter(([u, v, c]) => h[u] > c + h[v]), overestimates: NODES.filter(n => h[n] > HSTAR[n]) };
  }
  class DetectiveSearch {
    constructor(preset = 'inconsistent', reopen = true) {
      if (!Object.hasOwn(HVALUES, preset) || typeof reopen !== 'boolean') throw new Error('Invalid detective setting.');
      this.h = HVALUES[preset]; this.reopen = reopen; this.g = { S: 0, A: Infinity, B: Infinity, G: Infinity }; this.parent = {}; this.closed = new Set(); this.sequence = 0; this.expanded = 0; this.reopened = 0; this.current = null; this.trace = []; this.events = []; this.done = false; this.found = false; this.path = [];
      this.heap = new Heap((a, b) => a.f - b.f || a.h - b.h || a.sequence - b.sequence); this.insert('S');
    }
    insert(state) { const g = this.g[state], h = this.h[state]; this.heap.push({ state, g, h, f: g + h, sequence: this.sequence++ }); }
    entries() { const seen = new Set(); return this.heap.items.slice().sort(this.heap.compare).filter(e => e.g === this.g[e.state] && !this.closed.has(e.state) && !seen.has(e.state) && seen.add(e.state)); }
    step() {
      if (this.done) return null;
      let entry;
      do { entry = this.heap.pop(); if (!entry) { this.done = true; return null; } if (entry.g !== this.g[entry.state] || this.closed.has(entry.state)) entry = null; } while (!entry);
      this.current = entry; this.trace.push(entry); this.events = [];
      const u = entry.state;
      if (u === 'G') { this.done = this.found = true; for (let v = 'G'; v !== undefined; v = this.parent[v]) this.path.push(v); this.path.reverse(); return entry; }
      this.closed.add(u); this.expanded++;
      for (const [from, v, c] of EDGES.filter(e => e[0] === u)) {
        const tentative = this.g[u] + c;
        if (tentative >= this.g[v]) continue;
        const wasClosed = this.closed.has(v);
        if (wasClosed && !this.reopen) { this.events.push({ from, to: v, old: this.g[v], improved: tentative, action: 'discarded' }); continue; }
        if (this.closed.delete(v)) { this.reopened++; this.events.push({ from, to: v, old: this.g[v], improved: tentative, action: 'reopened' }); }
        this.g[v] = tentative; this.parent[v] = u; this.insert(v);
      }
      return entry;
    }
    run() { while (!this.done) this.step(); return this; }
    get cost() { return this.found ? this.g.G : Infinity; }
  }
  const api = { GOAL, POP_LIMIT, PRESETS, DEPTHS, Heap, validatePuzzle, puzzleNeighbors, manhattan, PuzzleSearch, NODES, EDGES, HSTAR, HVALUES, inspectHeuristic, DetectiveSearch };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const register = window.ArticleLabs.register;
  // Browser runs yield between batches; timer-less contract hosts use the same bounded engines.
  const hasTimers = typeof window.setTimeout === 'function' && typeof window.clearTimeout === 'function';
  register('search-astar-tiles', ctx => {
    ctx.root.classList.add('sa-lab', 'al-workshop');
    const preset = ctx.control('Challenge', '예제', { type: 'select', value: 'eight', options: Object.keys(PRESETS).map(value => ({ value, en: `${DEPTHS[value]} optimal moves`, ko: `최적 이동 ${DEPTHS[value]}번` })) });
    const heuristic = ctx.control('Solver priority', 'Solver priority', { type: 'select', value: 'manhattan', options: [{ value: 'manhattan', en: 'A*: g + Manhattan h', ko: 'A*: g + Manhattan h' }, { value: 'zero', en: 'Dijkstra: g + 0', ko: 'Dijkstra: g + 0' }] });
    let board = PRESETS.eight, manualMoves = 0, search, signature = '', loadedPreset = preset.value, timer = null, mode = '', playback = -1, hintIndex = -1;
    const movesMetric = ctx.metric('Manual moves / starting optimum', '직접 이동 / 시작 예제 최적 이동'), workMetric = ctx.metric('Expanded / active frontier', 'Expanded / active frontier'), resultMetric = ctx.metric('Solver optimum / current h', 'Solver 최적 이동 / 현재 h');
    const boardNode = ctx.el('div', { class: 'sa-board', tabindex: '0', role: 'group', 'aria-label': ctx.text('Playable 8-puzzle; arrow keys move the blank', '직접 푸는 8-puzzle · 방향키로 빈칸 이동') });
    const traceNode = ctx.el('div', { class: 'sa-board sa-trace-board', role: 'img' });
    const traceLabel = ctx.el('p', { class: 'sa-readout' });
    const instruction = ctx.el('p', { class: 'sa-instruction' }, ctx.text('Click a tile beside the blank. With the board focused, arrow keys move the blank, not the numbered tile. The right-hand board shows the solver’s selected state; it never changes your board.', '빈칸 옆 tile을 누르세요. Board에 focus를 둔 상태에서 방향키를 누르면 숫자 tile이 아니라 빈칸이 이동합니다. Solver board는 탐색 중 선택한 state를 보여 주며, 직접 푸는 board와 별개입니다.'));
    const frontier = ctx.el('div', { class: 'sa-frontier' }), announcement = ctx.el('p', { class: 'sa-announcement', 'aria-live': 'polite' });
    ctx.view.append(ctx.el('div', { class: 'sa-boards' }, [ctx.el('section', {}, [ctx.el('p', { class: 'sa-board-title' }, ctx.text('Your board', '직접 풀기')), boardNode]), ctx.el('section', {}, [ctx.el('p', { class: 'sa-board-title' }, ctx.text('Solver board', 'Solver board')), traceNode, traceLabel])]), instruction, announcement, frontier);
    const cells = Array.from({ length: 9 }, (_, i) => { const cell = ctx.el('button', { type: 'button', class: 'sa-tile' }); cell.addEventListener('click', () => move(i)); boardNode.append(cell); return cell; });
    const traceCells = Array.from({ length: 9 }, () => { const cell = ctx.el('span', { class: 'sa-tile' }); traceNode.append(cell); return cell; });
    let runButton;
    function stop() { if (timer !== null && hasTimers) window.clearTimeout(timer); timer = null; mode = ''; if (runButton) runButton.textContent = ctx.text('Run search', '탐색 실행'); }
    function restart() { stop(); search = new PuzzleSearch(board, heuristic.value); signature = `${board}:${heuristic.value}`; playback = -1; hintIndex = -1; announcement.textContent = ''; }
    function move(index) {
      const next = puzzleNeighbors(board).find(x => x.index === index); if (!next) return;
      board = next.state; manualMoves++; restart(); paint();
      // Keep arrow-key play active after the clicked tile becomes the disabled blank.
      if (typeof boardNode.focus === 'function') boardNode.focus({ preventScroll: true });
      if (board === GOAL) announcement.textContent = ctx.text(`Solved by hand in ${manualMoves} moves! The preset optimum was ${DEPTHS[preset.value]}.`, `${manualMoves}번 움직여 직접 풀었습니다! 시작 예제의 최적 이동 수는 ${DEPTHS[preset.value]}번입니다.`);
    }
    boardNode.addEventListener('keydown', event => { const direction = { ArrowRight: 'right', ArrowDown: 'down', ArrowLeft: 'left', ArrowUp: 'up' }[event.key]; if (!direction) return; event.preventDefault(); const next = puzzleNeighbors(board).find(x => x.direction === direction); if (next) move(next.index); });
    function batch(finish, after) {
      if (ctx.root.isConnected === false) { stop(); return; }
      if (!hasTimers) { search.run(); paint(); stop(); if (after && search.found) after(); return; }
      const count = finish ? 200 : 1;
      for (let i = 0; i < count && !search.done; i++) search.step();
      paint();
      if (search.done) { stop(); if (after && search.found) after(); return; }
      timer = window.setTimeout(() => batch(finish, after), finish ? 0 : 160);
    }
    ctx.button('Step search', '한 단계 탐색', () => { stop(); search.step(); playback = -1; paint(); });
    runButton = ctx.button('Run search', '탐색 실행', () => { if (mode) { stop(); return; } if (search.done) restart(); mode = 'run'; runButton.textContent = ctx.text('Pause', '일시정지'); batch(false); });
    ctx.button('Finish search', '끝까지 탐색', () => { stop(); mode = 'finish'; batch(true); });
    ctx.button('Hint: one move', '힌트: 한 번 이동', () => {
      stop(); mode = 'finish';
      batch(true, () => { if (search.path.length < 2) { announcement.textContent = ctx.text('Already at the goal.', '이미 goal입니다.'); return; } hintIndex = search.path[1].indexOf('0'); paint(); announcement.textContent = ctx.text('The outlined tile starts one optimal route from your current board.', '테두리로 표시한 tile을 누르면 현재 board의 최적 경로를 따라 한 번 이동합니다.'); });
    });
    ctx.button('Play solver solution', 'Solver 해답 재생', () => {
      stop(); mode = 'finish'; batch(true, () => {
        playback = 0; mode = 'play';
        if (!hasTimers) { playback = search.path.length - 1; paint(); stop(); return; }
        function play() { if (ctx.root.isConnected === false) { stop(); return; } paint(); if (playback >= search.path.length - 1) { stop(); return; } timer = window.setTimeout(() => { playback++; play(); }, 500); }
        play();
      });
    });
    ctx.button('Restart solver only', 'Solver만 다시 시작', () => { restart(); paint(); });
    function paint() {
      const shown = playback >= 0 ? search.path[playback] : search.current ? search.current.state : board;
      cells.forEach((cell, i) => { const tile = board[i]; cell.textContent = tile === '0' ? '·' : tile; cell.className = `sa-tile${tile === '0' ? ' sa-blank' : ''}${i === hintIndex ? ' sa-hint' : ''}`; cell.disabled = !puzzleNeighbors(board).some(x => x.index === i); cell.setAttribute('aria-label', tile === '0' ? ctx.text('Blank', '빈칸') : ctx.text(`Move tile ${tile}`, `Tile ${tile} 이동`)); });
      traceCells.forEach((cell, i) => { cell.textContent = shown[i] === '0' ? '·' : shown[i]; cell.className = `sa-tile${shown[i] === '0' ? ' sa-blank' : ''}`; });
      traceNode.setAttribute('aria-label', ctx.text(`Solver state ${shown}`, `Solver state ${shown}`));
      traceLabel.textContent = playback >= 0 ? ctx.text(`Solution move ${playback} / ${search.path.length - 1}`, `해답 이동 ${playback} / ${search.path.length - 1}`) : search.current ? `g=${search.current.g} · h=${search.current.h} · f=${search.current.f}` : ctx.text('No state popped yet.', '아직 state를 꺼내지 않았습니다.');
      const entries = search.entries(); movesMetric.textContent = `${manualMoves} / ${DEPTHS[preset.value]}`; workMetric.textContent = `${search.expanded} / ${entries.length}`; resultMetric.textContent = `${search.found ? search.cost : '—'} / ${search.h(board)}`;
      frontier.replaceChildren(ctx.el('p', { class: 'sa-readout' }, ctx.text('Next frontier states (row-major; 0 is blank)', '다음 frontier state (행 순서 · 0은 빈칸)')), ctx.table(['State', 'g', 'h', 'f'], entries.slice(0, 5).map(e => [e.state, e.g, e.h, e.f])));
      ctx.status.textContent = search.limitReached ? ctx.text('Stopped at 50,000 valid pops. Try Manhattan or choose a preset; no optimality claim is made for an unfinished run.', '유효 pop 50,000번에서 멈췄습니다. Manhattan이나 예제를 선택해 보세요. 끝나지 않은 탐색의 최적 이동 수는 표시하지 않습니다.') : search.found ? ctx.text(`Goal popped: the current board needs ${search.cost} optimal moves. Expanded counts exclude the goal.`, `Goal을 꺼냈습니다. 현재 board의 최적 이동 수는 ${search.cost}번입니다. Expanded에는 goal을 포함하지 않습니다.`) : ctx.text('Solve manually or watch frontier choices. Change h without changing your board to compare search effort.', '직접 풀거나 frontier 선택을 살펴보세요. Board는 그대로 두고 h를 바꾸면 탐색량을 비교할 수 있습니다.');
    }
    ctx.onReset(() => { stop(); board = PRESETS.eight; loadedPreset = 'eight'; manualMoves = 0; signature = ''; });
    ctx.onChange(() => { if (preset.value !== loadedPreset) { loadedPreset = preset.value; board = PRESETS[preset.value]; manualMoves = 0; } if (`${board}:${heuristic.value}` !== signature) restart(); paint(); });
    ctx.stateLink('search-astar-tiles', {
      read: () => ({ preset: preset.value, heuristic: heuristic.value, board, moves: manualMoves, steps: search.pops + (search.done && !search.found ? 1 : 0), playback, hint: hintIndex, announcement: announcement.textContent }),
      write: s => {
        if (!s || !Object.hasOwn(PRESETS, s.preset) || !['zero', 'manhattan'].includes(s.heuristic) || !Number.isInteger(s.moves) || s.moves < 0 || s.moves > 100000) return false;
        validatePuzzle(s.board);
        const steps = s.steps === undefined ? 0 : s.steps, savedPlayback = s.playback === undefined ? -1 : s.playback, savedHint = s.hint === undefined ? -1 : s.hint, savedAnnouncement = s.announcement === undefined ? '' : s.announcement;
        if (!Number.isInteger(steps) || steps < 0 || steps > POP_LIMIT + 1 || !Number.isInteger(savedPlayback) || savedPlayback < -1 || !Number.isInteger(savedHint) || savedHint < -1 || savedHint > 8 || typeof savedAnnouncement !== 'string' || savedAnnouncement.length > 600) return false;
        const next = new PuzzleSearch(s.board, s.heuristic);
        for (let i = 0; i < steps; i++) { if (next.done) return false; next.step(); }
        if (savedPlayback >= 0 && (!next.found || savedPlayback >= next.path.length) || savedHint >= 0 && !puzzleNeighbors(s.board).some(n => n.index === savedHint)) return false;
        stop(); preset.value = loadedPreset = s.preset; heuristic.value = s.heuristic; board = s.board; manualMoves = s.moves; search = next; signature = `${board}:${heuristic.value}`; playback = savedPlayback; hintIndex = savedHint; announcement.textContent = savedAnnouncement; paint(); return true;
      }
    });
  });
  let graphNumber = 0;
  register('search-astar-detective', ctx => {
    ctx.root.classList.add('sa-lab', 'al-workshop');
    const preset = ctx.control('Heuristic evidence', 'Heuristic 예제', { type: 'select', value: 'inconsistent', options: [{ value: 'zero', en: 'Zero: Dijkstra', ko: 'Zero: Dijkstra' }, { value: 'exact', en: 'Exact h*', ko: '정확한 h*' }, { value: 'inconsistent', en: 'Admissible, inconsistent', ko: 'Admissible · inconsistent' }, { value: 'overestimate', en: 'Overestimated h(B)', ko: 'h(B) 과대평가' }] });
    const reopen = ctx.control('Reopen improved states', 'g가 줄면 다시 열기', { type: 'checkbox', value: true });
    let search, signature = '', timer = null, prediction = '';
    const result = ctx.metric('Returned cost / optimum', '반환 비용 / 최적 비용'), count = ctx.metric('Expanded / reopened', 'Expanded / reopened'), guarantee = ctx.metric('Admissible / consistent', 'Admissible / consistent');
    const graph = ctx.svg('svg', { viewBox: '0 0 450 300', class: 'sa-graph', role: 'img', 'aria-label': ctx.text('Directed four-state counterexample graph', 'State 네 개의 directed counterexample graph') });
    const inspection = ctx.el('div', { class: 'sa-inspection' }), frontier = ctx.el('div', { class: 'sa-frontier' }), feedback = ctx.el('p', { class: 'sa-announcement', 'aria-live': 'polite' }), events = ctx.el('p', { class: 'sa-event' }), trail = ctx.el('p', { class: 'sa-readout' });
    const guesses = ctx.el('div', { class: 'sa-guesses' }, [ctx.el('p', {}, ctx.text('Predict the next pop:', '다음에 꺼낼 state를 맞혀 보세요:'))]);
    NODES.forEach(n => { const button = ctx.el('button', { type: 'button', class: 'al-button' }, n); button.addEventListener('click', () => { stop(); if (search.done) return; const next = search.entries()[0]; prediction = next ? ctx.text(`${n === next.state ? 'Correct' : 'Not quite'}: ${next.state} has the next priority f=${next.f}.`, `${n === next.state ? '맞았습니다' : '다시 살펴보세요'}: 다음 state는 ${next.state}, priority는 f=${next.f}입니다.`) : ''; search.step(); paint(); }); guesses.append(button); });
    ctx.view.append(graph, guesses, feedback, events, trail, frontier, inspection);
    let runButton;
    function stop() { if (timer !== null && hasTimers) window.clearTimeout(timer); timer = null; if (runButton) runButton.textContent = ctx.text('Run', '실행'); }
    function restart() { stop(); search = new DetectiveSearch(preset.value, reopen.checked); signature = `${preset.value}:${reopen.checked}`; prediction = ''; }
    ctx.button('Step', '한 단계', () => { stop(); prediction = ''; search.step(); paint(); });
    runButton = ctx.button('Run', '실행', () => { if (timer !== null) { stop(); return; } if (search.done) restart(); if (!hasTimers) { search.run(); paint(); stop(); return; } runButton.textContent = ctx.text('Pause', '일시정지'); function tick() { if (ctx.root.isConnected === false) { stop(); return; } search.step(); paint(); if (search.done) stop(); else timer = window.setTimeout(tick, 700); } timer = window.setTimeout(tick, 50); });
    ctx.button('Finish', '끝까지 실행', () => { stop(); prediction = ''; search.run(); paint(); });
    ctx.button('Restart trace', '탐색 다시 시작', () => { restart(); paint(); });
    const arrow = `sa-arrow-${++graphNumber}`, points = { S: [55, 150], A: [215, 65], B: [215, 235], G: [395, 150] };
    function paint() {
      const check = inspectHeuristic(search.h), pathEdges = new Set(search.path.slice(1).map((v, i) => `${search.path[i]}:${v}`));
      graph.replaceChildren(ctx.svg('title', {}, ctx.text('Route costs: S→A→G is 6; S→B→A→G is 5', 'S→A→G 비용은 6, S→B→A→G 비용은 5')));
      const defs = ctx.svg('defs'), marker = ctx.svg('marker', { id: arrow, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }); marker.append(ctx.svg('path', { d: 'M0 0L10 5L0 10Z', fill: 'context-stroke' })); defs.append(marker); graph.append(defs);
      EDGES.forEach(([u, v, c]) => {
        const [x1, y1] = points[u], [x2, y2] = points[v], d = Math.hypot(x2 - x1, y2 - y1), dx = (x2 - x1) / d, dy = (y2 - y1) / d;
        const violation = search.h[u] > c + search.h[v], parent = search.parent[v] === u, path = pathEdges.has(`${u}:${v}`);
        graph.append(ctx.svg('line', { x1: x1 + 29 * dx, y1: y1 + 29 * dy, x2: x2 - 32 * dx, y2: y2 - 32 * dy, stroke: path ? '#278052' : violation ? 'var(--al-red)' : parent ? 'var(--al-blue)' : 'var(--al-muted)', 'stroke-width': path ? 5 : parent || violation ? 3 : 2, ...(violation ? { 'stroke-dasharray': '6 4' } : {}), 'marker-end': `url(#${arrow})` }));
        const offset = u === 'B' && v === 'A' ? -16 : y2 > y1 ? 16 : -12;
        graph.append(ctx.svg('text', { x: (x1 + x2) / 2 + (u === 'B' && v === 'A' ? offset : 0), y: (y1 + y2) / 2 + (u === 'B' && v === 'A' ? 6 : offset), 'text-anchor': 'middle', class: 'sa-edge-label' }, c));
      });
      NODES.forEach(n => { const [cx, cy] = points[n], current = search.current && search.current.state === n; graph.append(ctx.svg('circle', { cx, cy, r: 27, class: `sa-node${current ? ' sa-node-current' : search.closed.has(n) ? ' sa-node-closed' : ''}` })); graph.append(ctx.svg('text', { x: cx, y: cy + 7, 'text-anchor': 'middle', class: 'sa-node-label' }, n)); });
      const yesno = value => value ? ctx.text('yes', '예') : ctx.text('no', '아니요');
      result.textContent = `${search.found ? search.cost : '—'} / 5`; count.textContent = `${search.expanded} / ${search.reopened}`; guarantee.textContent = `${yesno(check.admissible)} / ${yesno(check.consistent)}`;
      trail.textContent = ctx.text('Valid pop order: ', '유효 pop 순서: ') + (search.trace.map(e => `${e.state}[g=${e.g},f=${e.f}]`).join(' → ') || '—');
      frontier.replaceChildren(ctx.el('p', { class: 'sa-readout' }, ctx.text('Frontier, in pop order', 'Frontier · 꺼낼 순서')), ctx.table(['State', 'g', 'h', 'f'], search.entries().map(e => [e.state, e.g, e.h, e.f])));
      const violations = check.violations.map(([u, v, c]) => `${u}→${v}: ${search.h[u]} > ${c}+${search.h[v]}`).join('; ');
      inspection.replaceChildren(ctx.el('p', { class: 'sa-readout' }, ctx.text('Heuristic evidence: h(n) ≤ h*(n)', 'Heuristic 확인: h(n) ≤ h*(n)')), ctx.table(['State', 'h', 'h*', 'h ≤ h*'], NODES.map(n => [n, search.h[n], HSTAR[n], yesno(search.h[n] <= HSTAR[n])])), ctx.el('p', { class: 'sa-violation' }, violations ? ctx.text(`Dashed edges violate consistency h(u) ≤ c(u,v)+h(v): ${violations}.`, `점선 edge에서는 consistency h(u) ≤ c(u,v)+h(v)가 깨집니다: ${violations}.`) : ctx.text('Every edge passes h(u) ≤ c(u,v)+h(v).', '모든 edge에서 h(u) ≤ c(u,v)+h(v)가 성립합니다.')));
      feedback.textContent = prediction;
      events.textContent = search.events.map(e => ctx.text(`${e.from}→${e.to} improves g: ${e.old}→${e.improved}; ${e.action === 'discarded' ? 'discarded because the state is closed' : 'state reopened'}.`, `${e.from}→${e.to}에서 g가 ${e.old}→${e.improved}로 줄어듭니다. ${e.action === 'discarded' ? '이미 closed라서 개선을 버렸습니다' : 'State를 다시 열었습니다'}.`)).join(' ');
      ctx.status.textContent = search.found ? search.cost === 5 ? ctx.text(`Optimal route: ${search.path.join(' → ')}; cost 5. Blue edges show current parents; green edges show the returned route.`, `최적 경로: ${search.path.join(' → ')} · 비용 5. 파란 선은 현재 parent, 초록 선은 반환한 경로입니다.`) : ctx.text(`Wrong route: ${search.path.join(' → ')}; cost ${search.cost}, but the optimum is 5. ${!check.admissible ? 'Reopening cannot repair an overestimated heuristic’s early goal pop.' : 'The cheaper route through B never propagated through the closed A.'}`, `잘못된 경로: ${search.path.join(' → ')} · 비용 ${search.cost}, 최적 비용은 5입니다. ${!check.admissible ? 'h를 과대평가하면 reopening을 켜도 goal을 너무 일찍 꺼낼 수 있습니다.' : 'B를 지나는 더 싼 경로가 closed인 A를 거쳐 전달되지 못했습니다.'}`) : !reopen.checked ? ctx.text('Closed-once mode deliberately discards improvements to expanded states. Watch what happens when B finds a cheaper A.', 'Closed-once 옵션은 이미 확장한 state의 g 개선을 일부러 버립니다. B에서 A로 더 싸게 도달했을 때를 살펴보세요.') : ctx.text('Predict or step through the next minimum-f pop. Reopening propagates a cheaper g even after expansion.', '다음 최소 f state를 맞히거나 한 단계씩 실행해 보세요. 이미 확장한 state라도 g가 줄면 다시 열어 개선을 전달합니다.');
    }
    ctx.onReset(() => { stop(); signature = ''; });
    ctx.onChange(() => { if (`${preset.value}:${reopen.checked}` !== signature) restart(); paint(); });
    ctx.stateLink('search-astar-detective', {
      read: () => ({ preset: preset.value, reopen: reopen.checked, steps: search.trace.length + (search.done && !search.found ? 1 : 0), prediction }),
      write: s => {
        if (!s || !Object.hasOwn(HVALUES, s.preset) || typeof s.reopen !== 'boolean') return false;
        const steps = s.steps === undefined ? 0 : s.steps, savedPrediction = s.prediction === undefined ? '' : s.prediction;
        if (!Number.isInteger(steps) || steps < 0 || steps > 10 || typeof savedPrediction !== 'string' || savedPrediction.length > 600) return false;
        const next = new DetectiveSearch(s.preset, s.reopen);
        for (let i = 0; i < steps; i++) { if (next.done) return false; next.step(); }
        stop(); preset.value = s.preset; reopen.checked = s.reopen; search = next; signature = `${preset.value}:${reopen.checked}`; prediction = savedPrediction; paint(); return true;
      }
    });
  });
}());
