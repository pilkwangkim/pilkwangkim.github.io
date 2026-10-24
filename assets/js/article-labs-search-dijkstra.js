/* Exact, positive-weight teaching networks. No remote maps, services or eval. */
(function () {
  'use strict';
  const NODES = [
    { id: 'A', x: 50, y: 110, en: 'Depot', ko: 'Depot' },
    { id: 'B', x: 183, y: 110, en: 'Museum', ko: 'Museum' },
    { id: 'C', x: 317, y: 110, en: 'Library', ko: 'Library' },
    { id: 'D', x: 50, y: 300, en: 'Park', ko: 'Park' },
    { id: 'E', x: 183, y: 300, en: 'Market', ko: 'Market' },
    { id: 'F', x: 317, y: 300, en: 'School', ko: 'School' },
    { id: 'G', x: 450, y: 300, en: 'Clinic', ko: 'Clinic' },
    { id: 'H', x: 450, y: 110, en: 'Harbor', ko: 'Harbor' }
  ];
  const BASE_EDGES = [['A', 'H', 18], ['A', 'B', 4], ['B', 'C', 4], ['C', 'H', 4], ['A', 'D', 2], ['D', 'E', 2], ['E', 'F', 2], ['F', 'G', 2], ['G', 'H', 2], ['B', 'E', 3], ['C', 'F', 3]];
  const PRESETS = ['normal', 'traffic', 'bridge', 'isolated'];
  const edgeKey = (a, b) => [a, b].sort().join('-');
  function makeNetwork(preset = 'normal') {
    if (!PRESETS.includes(preset)) throw new Error('Unknown road condition');
    const edges = BASE_EDGES.map(([a, b, minutes]) => ({ a, b, minutes: preset === 'traffic' && ['A-D', 'D-E', 'E-F', 'F-G', 'G-H'].includes(edgeKey(a, b)) ? minutes * 3 : minutes, closed: preset === 'bridge' && edgeKey(a, b) === 'E-F' || preset === 'isolated' && (a === 'H' || b === 'H') }));
    return { nodes: NODES.map(node => ({ ...node })), edges, preset };
  }
  function validateNetwork(graph) {
    if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges) || graph.nodes.length < 1 || graph.nodes.length > 40) throw new Error('Invalid network');
    const ids = graph.nodes.map(n => n.id), keys = new Set();
    if (ids.some(id => typeof id !== 'string') || new Set(ids).size !== ids.length) throw new Error('Invalid node identifiers');
    for (const edge of graph.edges) {
      const key = edgeKey(edge.a, edge.b);
      if (!ids.includes(edge.a) || !ids.includes(edge.b) || edge.a === edge.b || !Number.isFinite(edge.minutes) || edge.minutes <= 0 || keys.has(key)) throw new Error('Invalid positive undirected road');
      keys.add(key);
    }
    return true;
  }
  function adjacent(graph, id) {
    return graph.edges.filter(edge => !edge.closed && (edge.a === id || edge.b === id)).map(edge => ({ id: edge.a === id ? edge.b : edge.a, minutes: edge.minutes })).sort((a, b) => a.id.localeCompare(b.id));
  }
  function manualRoute(graph, route) {
    if (!Array.isArray(route) || route.length === 0 || route.some(id => !graph.nodes.some(node => node.id === id))) return { legal: false, cost: Infinity, reason: 'node' };
    let cost = 0;
    for (let i = 1; i < route.length; i++) {
      const edge = graph.edges.find(e => !e.closed && edgeKey(e.a, e.b) === edgeKey(route[i - 1], route[i]));
      if (!edge) return { legal: false, cost: Infinity, reason: 'road' };
      cost += edge.minutes;
    }
    return { legal: true, cost, reason: null };
  }
  const compare = (a, b) => a.distance - b.distance || a.owner.localeCompare(b.owner) || a.id.localeCompare(b.id) || a.sequence - b.sequence;
  class Dijkstra {
    constructor(graph, sources = ['A'], goal = null) {
      validateNetwork(graph);
      const ids = graph.nodes.map(n => n.id);
      if (!Array.isArray(sources) || !sources.length || sources.some(id => !ids.includes(id)) || goal !== null && !ids.includes(goal)) throw new Error('Invalid source or goal');
      this.graph = graph; this.sources = [...new Set(sources)].sort(); this.goal = goal;
      this.distance = Object.fromEntries(ids.map(id => [id, Infinity]));
      this.owner = Object.fromEntries(ids.map(id => [id, null]));
      this.parent = Object.fromEntries(ids.map(id => [id, null]));
      this.queue = []; this.settled = new Set(); this.sequence = 0; this.pops = 0; this.stale = 0; this.done = false; this.found = false; this.current = null; this.event = null;
      for (const id of this.sources) { this.distance[id] = 0; this.owner[id] = id; this.insert(id); }
    }
    insert(id) { this.queue.push({ id, distance: this.distance[id], owner: this.owner[id], sequence: this.sequence++ }); this.queue.sort(compare); }
    valid(entry) { return entry.distance === this.distance[entry.id] && entry.owner === this.owner[entry.id] && !this.settled.has(entry.id); }
    entries() { return this.queue.map(entry => ({ ...entry, valid: this.valid(entry) })); }
    step() {
      if (this.done) return null;
      const entry = this.queue.shift();
      if (!entry) { this.done = true; this.current = null; this.event = { kind: 'empty', updates: [] }; return this.event; }
      this.pops++; this.current = entry.id;
      if (!this.valid(entry)) { this.stale++; this.event = { kind: 'stale', entry, updates: [] }; if (!this.queue.length) this.done = true; return this.event; }
      this.settled.add(entry.id);
      if (entry.id === this.goal) { this.done = true; this.found = true; this.event = { kind: 'goal', entry, updates: [] }; return this.event; }
      const updates = [];
      for (const next of adjacent(this.graph, entry.id)) {
        const candidate = entry.distance + next.minutes, previous = this.distance[next.id];
        const improved = !this.settled.has(next.id) && (candidate < previous || candidate === previous && entry.owner < this.owner[next.id]);
        updates.push({ from: entry.id, to: next.id, minutes: next.minutes, candidate, previous, improved });
        if (improved) { this.distance[next.id] = candidate; this.parent[next.id] = entry.id; this.owner[next.id] = entry.owner; this.insert(next.id); }
      }
      this.event = { kind: 'settle', entry, updates };
      if (!this.queue.length) this.done = true;
      return this.event;
    }
    run() { let steps = 0; while (!this.done && steps++ < 10000) this.step(); if (!this.done) throw new Error('Teaching limit exceeded'); return this; }
    path(id) {
      if (!Number.isFinite(this.distance[id])) return [];
      const route = [], seen = new Set();
      for (let u = id; u !== null; u = this.parent[u]) { if (seen.has(u) || !Object.hasOwn(this.parent, u)) throw new Error('Invalid predecessor chain'); seen.add(u); route.push(u); }
      return route.reverse();
    }
  }
  const api = { NODES, BASE_EDGES, PRESETS, makeNetwork, validateNetwork, adjacent, manualRoute, Dijkstra, edgeKey };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const fmt = value => Number.isFinite(value) ? String(value) : '∞';
  const join = route => route.join(' → ');
  function networkView(ctx, graph, state, chosen, click, dispatch, stations, manual, solution) {
    const svg = ctx.svg('svg', { viewBox: '0 0 500 365', class: 'sd-network', role: 'group', 'aria-label': ctx.text('Road network. Every edge label is a travel time in minutes. Click a stop or use Tab and Enter.', 'Road network입니다. Edge 숫자는 이동 시간(분)입니다. Stop을 클릭하거나 Tab과 Enter로 선택하세요.') });
    const colors = ['sd-owner-one', 'sd-owner-two'];
    const routeKeys = route => new Set(route.slice(1).map((id, i) => edgeKey(route[i], id)));
    const manualKeys = routeKeys(manual || []), solutionKeys = routeKeys(solution || []);
    for (const edge of graph.edges) {
      const first = graph.nodes.find(n => n.id === edge.a), second = graph.nodes.find(n => n.id === edge.b), key = edgeKey(edge.a, edge.b), direct = key === 'A-H';
      const shape = direct ? 'path' : 'line', coords = direct ? { d: 'M50 110 Q250 -72 450 110' } : { x1: first.x, y1: first.y, x2: second.x, y2: second.y };
      // Keep both encodings visible when the player chose an optimal road.
      if (solutionKeys.has(key) && manualKeys.has(key)) svg.append(ctx.svg(shape, { ...coords, class: 'sd-road sd-solution-road' }));
      svg.append(ctx.svg(shape, { ...coords, class: `sd-road${edge.closed ? ' sd-closed-road' : ''}${solutionKeys.has(key) ? ' sd-solution-road' : ''}${manualKeys.has(key) ? ' sd-manual-road' : ''}` }));
      const x = direct ? 250 : (first.x + second.x) / 2 + (first.x === second.x ? 21 : 0), y = direct ? 27 : (first.y + second.y) / 2 + (first.y === second.y ? -21 : 0);
      const group = ctx.svg('g', { class: 'sd-weight' });
      group.append(ctx.svg('rect', { x: x - 25, y: y - 18, width: 50, height: 33, rx: 7 }), ctx.svg('text', { x, y, 'text-anchor': 'middle', 'dominant-baseline': 'central' }, edge.closed ? '×' : edge.minutes));
      svg.append(group);
    }
    for (const node of graph.nodes) {
      const isStation = dispatch && stations.includes(node.id), ownerIndex = stations.indexOf(state.owner[node.id]);
      const group = ctx.svg('g', { class: `sd-node${chosen === node.id ? ' sd-chosen' : ''}${state.settled.has(node.id) ? ' sd-settled' : ''}${state.current === node.id ? ' sd-current' : ''}${dispatch && ownerIndex >= 0 ? ' ' + colors[ownerIndex] : ''}`, tabindex: 0, role: 'button', 'aria-label': ctx.text(`${node.id}, ${node.en}. ${isStation ? 'Station. ' : ''}g equals ${fmt(state.distance[node.id])} minutes. Select stop.`, `${node.id}, ${node.ko}. ${isStation ? 'Station. ' : ''}g=${fmt(state.distance[node.id])}분. Stop 선택.`) });
      if (isStation) group.append(ctx.svg('rect', { x: node.x - 31, y: node.y - 31, width: 62, height: 62, rx: 12, class: 'sd-station-ring' }));
      group.append(ctx.svg('circle', { cx: node.x, cy: node.y, r: 25 }), ctx.svg('text', { x: node.x, y: node.y, 'text-anchor': 'middle', 'dominant-baseline': 'central', class: 'sd-node-id' }, node.id), ctx.svg('text', { x: node.x, y: node.y + 46, 'text-anchor': 'middle', class: 'sd-distance' }, `g=${fmt(state.distance[node.id])}`));
      group.addEventListener('click', () => click(node.id));
      group.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); click(node.id); } });
      svg.append(group);
    }
    return svg;
  }
  function teachingLab(ctx, dispatch) {
    ctx.root.classList.add('sd-lab', 'al-workshop');
    const road = ctx.control('Road conditions', '도로 상황', { type: 'select', value: 'normal', options: [
      { value: 'normal', en: 'Normal traffic', ko: '평소 교통량' }, { value: 'traffic', en: 'Lower roads ×3 slower', ko: '아래쪽 도로 시간 ×3' }, { value: 'bridge', en: 'Close road E–F', ko: 'E–F 도로 폐쇄' }, { value: 'isolated', en: 'Harbor H isolated', ko: 'Harbor H 진입 불가' }
    ] });
    const stationPreset = dispatch ? ctx.control('Two-station preset', 'Station 두 곳', { type: 'select', value: 'A,H', options: [{ value: 'A,H', en: 'A + H', ko: 'A + H' }, { value: 'A,G', en: 'A + G (tie at E)', ko: 'A + G (E에서 동점)' }, { value: 'B,F', en: 'B + F', ko: 'B + F' }] }) : null;
    const interaction = dispatch ? ctx.control('Click a stop to…', 'Stop 클릭 동작', { type: 'select', value: 'inspect', options: [{ value: 'inspect', en: 'Inspect emergency arrival', ko: '도착 시간과 경로 확인' }, { value: 'station1', en: 'Move station 1', ko: 'Station 1 옮기기' }, { value: 'station2', en: 'Move station 2', ko: 'Station 2 옮기기' }] }) : null;
    const metricCost = ctx.metric(dispatch ? 'Selected arrival' : 'Your route / optimal', dispatch ? '선택한 곳의 도착 시간' : '내 경로 / 최적 시간');
    const metricSettled = ctx.metric('Settled / queue removals', 'Settled / queue에서 꺼낸 수');
    const metricStale = ctx.metric('Discarded old entries', '버린 오래된 entry');
    const board = ctx.el('div', { class: 'sd-board' }), legend = ctx.el('div', { class: 'sd-legend' });
    const routeLine = ctx.el('p', { class: 'sd-route' }), outcome = ctx.el('p', { class: 'sd-outcome', 'aria-live': 'polite' });
    const eventBox = ctx.el('div', { class: 'sd-event' }), queueBox = ctx.el('div', { class: 'sd-queue' });
    const names = ctx.el('details', { class: 'sd-stop-names' }, [ctx.el('summary', {}, ctx.text('Stop names and road model', 'Stop 이름과 도로 모형')), ctx.el('p', {}, ctx.text('A Depot · B Museum · C Library · D Park · E Market · F School · G Clinic · H Harbor. Every road is undirected, and its fixed positive label is the travel time in minutes. The drawing is not to scale.', 'A Depot · B Museum · C Library · D Park · E Market · F School · G Clinic · H Harbor. 모든 도로는 양방향이며, 양수로 표시한 이동 시간(분)은 탐색 중 바뀌지 않습니다. 그림의 길이는 실제 거리와 무관합니다.'))]);
    ctx.view.append(board, legend, routeLine, outcome, eventBox, queueBox, names);
    let graph, search, stations = ['A', 'H'], chosen = dispatch ? 'E' : 'A', manual = ['A'], checked = false, message = '', signature = '', loadedStations = stationPreset ? stationPreset.value : '', timer = null, runButton;
    function stop() { if (timer !== null && typeof window.clearInterval === 'function') window.clearInterval(timer); timer = null; if (runButton) runButton.textContent = ctx.text('Animate Dijkstra', 'Dijkstra 자동 실행'); }
    function restart() { stop(); graph = makeNetwork(road.value); search = new Dijkstra(graph, dispatch ? stations : ['A'], dispatch ? null : 'H'); checked = false; message = ''; signature = road.value + '|' + stations.join(','); }
    function click(id) {
      chosen = id;
      if (dispatch && interaction.value !== 'inspect') {
        const index = interaction.value === 'station1' ? 0 : 1;
        if (stations[1 - index] === id) message = ctx.text('Choose two different station stops.', 'Station은 서로 다른 두 곳에 두세요.');
        else { stations[index] = id; restart(); }
      } else if (!dispatch) {
        const candidate = [...manual, id];
        if (manual[manual.length - 1] === id) message = ctx.text('Already at this stop. Choose a connected stop.', '이미 이곳에 있습니다. 도로로 연결된 다른 stop을 고르세요.');
        else if (!manualRoute(graph, candidate).legal) message = ctx.text('No open road connects the last stop to that stop.', '마지막 stop과 이곳을 잇는 열린 도로가 없습니다.');
        else { manual = candidate; checked = false; message = ''; }
      }
      paint();
    }
    ctx.button('Step: one queue removal', '한 단계: queue에서 하나 꺼내기', () => { stop(); search.step(); paint(); });
    runButton = ctx.button('Animate Dijkstra', 'Dijkstra 자동 실행', () => {
      if (timer !== null) { stop(); return; }
      if (search.done) restart();
      if (typeof window.setInterval !== 'function') { search.run(); paint(); return; }
      runButton.textContent = ctx.text('Pause animation', '일시정지');
      timer = window.setInterval(() => { if (ctx.root.isConnected === false) { stop(); return; } search.step(); paint(); if (search.done) stop(); }, 850);
    });
    ctx.button('Finish Dijkstra', 'Dijkstra 끝까지 계산', () => { stop(); search.run(); paint(); });
    ctx.button('Restart Dijkstra only', 'Dijkstra만 처음부터', () => { restart(); paint(); });
    if (!dispatch) {
      ctx.button('Undo one stop', '마지막 이동 취소', () => { if (manual.length > 1) manual.pop(); checked = false; message = ''; paint(); });
      ctx.button('Clear my route', '내 경로 지우기', () => { manual = ['A']; checked = false; message = ''; paint(); });
      ctx.button('Check my delivery', '내 배달 경로 채점', () => { checked = true; stop(); paint(); });
    }
    function renderEvent() {
      const event = search.event;
      if (!event) { eventBox.replaceChildren(ctx.el('p', {}, ctx.text(`Initial queue: ${dispatch ? stations.slice().sort().join(', ') : 'A'} at g=0. No stop has been settled.`, `초기 queue에는 ${dispatch ? stations.slice().sort().join(', ') : 'A'}가 g=0으로 들어 있습니다. 아직 settled인 stop은 없습니다.`))); return; }
      if (event.kind === 'empty') { eventBox.replaceChildren(ctx.el('p', {}, ctx.text('Queue empty: every reachable stop has its final time.', 'Queue가 비었습니다. 도달 가능한 모든 stop의 시간이 확정되었습니다.'))); return; }
      const entry = event.entry;
      const headline = event.kind === 'stale' ? ctx.text(`Popped ${entry.id} at old g=${entry.distance}. Discarded: a better entry or a settled stop already exists. No neighbors examined.`, `${entry.id}의 오래된 g=${entry.distance} entry를 꺼내 버렸습니다. 더 좋은 entry가 있거나 이미 settled인 stop이므로 이웃은 검사하지 않습니다.`) : event.kind === 'goal' ? ctx.text(`Popped goal H at g=${entry.distance}. This valid minimum entry finalizes the delivery; merely discovering H earlier was not enough.`, `Goal H를 g=${entry.distance}인 유효한 최소 entry로 꺼냈습니다. 여기서 배달 경로를 확정합니다. 앞서 H를 발견했을 때 끝내면 안 됩니다.`) : ctx.text(`Popped ${entry.id} at g=${entry.distance}; settled. Candidate checks below are relaxations, not more queue removals.`, `${entry.id}를 g=${entry.distance}로 꺼내 settled로 표시했습니다. 아래는 이웃의 candidate를 검사한 결과이며, queue에서 추가로 꺼낸 것이 아닙니다.`);
      eventBox.replaceChildren(ctx.el('p', {}, headline));
      if (event.updates.length) eventBox.append(ctx.table([ctx.text('Road', '도로'), ctx.text('Candidate', 'Candidate'), ctx.text('Old g', '이전 g'), ctx.text('Action', '처리')], event.updates.map(update => [`${update.from}–${update.to}`, `${entry.distance}+${update.minutes}=${update.candidate}`, fmt(update.previous), update.improved ? ctx.text('Update + enqueue', '갱신 + queue에 추가') : ctx.text('Keep', '유지')])));
    }
    function paint() {
      const reference = new Dijkstra(graph, dispatch ? stations : ['A'], null).run(), result = manualRoute(graph, manual);
      const finalPath = dispatch ? search.settled.has(chosen) ? search.path(chosen) : [] : search.found ? search.path('H') : [];
      board.replaceChildren(networkView(ctx, graph, search, chosen, click, dispatch, stations, dispatch ? [] : manual, finalPath));
      legend.replaceChildren(...(dispatch ? [ctx.el('span', { class: 'sd-key sd-owner-one' }, `${stations[0]} · station 1`), ctx.el('span', { class: 'sd-key sd-owner-two' }, `${stations[1]} · station 2`), ctx.el('span', { class: 'sd-key' }, ctx.text('Square ring = station; solid outline = settled', '사각 테두리 = station · 실선 테두리 = settled'))] : [ctx.el('span', { class: 'sd-key' }, ctx.text('Blue dashed = your route', '파란 점선 = 내 경로')), ctx.el('span', { class: 'sd-key' }, ctx.text('Gold = final Dijkstra route', '금색 = Dijkstra 최종 경로'))]));
      metricCost.textContent = dispatch ? `${chosen}: ${fmt(search.distance[chosen])} min${search.settled.has(chosen) ? ' ✓' : ' ?'}` : `${fmt(result.cost)} / ${fmt(reference.distance.H)} min`;
      metricSettled.textContent = `${search.settled.size} / ${search.pops}`; metricStale.textContent = String(search.stale);
      if (dispatch) {
        const final = search.settled.has(chosen), route = search.path(chosen), owner = search.owner[chosen];
        routeLine.textContent = Number.isFinite(search.distance[chosen]) ? ctx.text(`From station ${owner}: ${join(route)}; ${search.distance[chosen]} min (${final ? 'final' : 'tentative'}).`, `${chosen}: station ${owner}에서 ${join(route)} 경로로 ${search.distance[chosen]}분 (${final ? '확정' : '잠정'}).`) : ctx.text(`${chosen}: no arrival found${search.done ? '; unreachable from both stations' : ' yet'}.`, `${chosen}: ${search.done ? '어느 station에서도 도달할 수 없습니다.' : '아직 도착 경로를 찾지 못했습니다.'}`);
        const fromEach = stations.map(source => new Dijkstra(graph, [source]).run().distance[chosen]);
        outcome.textContent = message || (search.done ? ctx.text(`Full comparison at ${chosen}: ${stations[0]}=${fmt(fromEach[0])} min, ${stations[1]}=${fmt(fromEach[1])} min. Equal times go to the alphabetically earlier station. Fewer roads do not necessarily mean faster arrival.`, `${chosen}까지의 시간을 비교하면 ${stations[0]}=${fmt(fromEach[0])}분, ${stations[1]}=${fmt(fromEach[1])}분입니다. 동점이면 알파벳이 앞선 station을 선택합니다. 도로를 적게 지난다고 더 빨리 도착하는 것은 아닙니다.`) : ctx.text('Choose an emergency stop; its owner and g may change until it is settled. Move either station, then watch both zero-time sources compete.', '긴급 출동할 stop을 선택하세요. Settled가 되기 전에는 담당 station과 g가 바뀔 수 있습니다. Station을 옮기면 두 출발점의 g=0에서 다시 탐색합니다.'));
      } else {
        routeLine.textContent = ctx.text(`Your delivery: ${join(manual)} · ${result.cost} min. Click an adjacent stop; destination H.`, `내 배달 경로: ${join(manual)} · ${result.cost}분. 연결된 stop을 클릭해 H까지 가세요.`);
        if (message) outcome.textContent = message;
        else if (!checked) outcome.textContent = ctx.text('Can you beat the one-road A–H trip? Plan first, then check. Gray g labels belong to Dijkstra, not your delivery.', 'A–H 도로 하나로 가는 것보다 빨리 갈 수 있을까요? 경로를 고른 뒤 채점해 보세요. 회색 g는 내 이동 시간이 아니라 Dijkstra의 기록입니다.');
        else if (!Number.isFinite(reference.distance.H)) outcome.textContent = ctx.text('H is unreachable under these closures. No legal delivery route exists.', '현재 폐쇄된 도로 때문에 H에 도달할 수 없습니다. 가능한 배달 경로가 없습니다.');
        else if (manual[manual.length - 1] !== 'H') outcome.textContent = ctx.text('The route is legal but has not reached H. Continue the delivery.', '지금까지의 이동은 가능하지만 아직 H에 도착하지 않았습니다. 경로를 이어 보세요.');
        else outcome.textContent = result.cost === reference.distance.H ? ctx.text(`Fastest delivery! ${result.cost} min. One optimal route: ${join(reference.path('H'))}.`, `최단 시간입니다! ${result.cost}분. 최적 경로 예: ${join(reference.path('H'))}.`) : ctx.text(`Delivery arrives in ${result.cost} min, ${result.cost - reference.distance.H} min above optimum. Try ${join(reference.path('H'))} (${reference.distance.H} min).`, `${result.cost}분에 도착하지만 최적 시간보다 ${result.cost - reference.distance.H}분 더 걸립니다. ${join(reference.path('H'))} 경로는 ${reference.distance.H}분입니다.`);
      }
      renderEvent();
      const queue = search.entries();
      queueBox.replaceChildren(ctx.el('p', {}, ctx.text('Priority queue — next removal first (old entries remain visible)', 'Priority queue — 다음에 꺼낼 entry부터 표시 (오래된 entry도 표시)')), ctx.table([ctx.text('Stop', 'Stop'), 'g', ...(dispatch ? [ctx.text('Station', 'Station')] : []), ctx.text('Entry', 'Entry')], queue.map(entry => [entry.id, entry.distance, ...(dispatch ? [entry.owner] : []), entry.valid ? ctx.text('Active', '유효') : ctx.text('Old: skip', '오래됨: 건너뜀')])));
      ctx.status.textContent = search.done ? dispatch ? ctx.text('Queue exhausted. Every reachable stop now has its nearest-station time.', 'Queue를 모두 확인했습니다. 도달 가능한 모든 stop의 최근접 station과 시간이 확정되었습니다.') : search.found ? ctx.text(`H accepted on valid pop: ${search.distance.H} min.`, `유효한 pop으로 H 확정: ${search.distance.H}분.`) : ctx.text('Queue exhausted. H is unreachable.', 'Queue가 비었습니다. H에는 도달할 수 없습니다.') : ctx.text('Step removes one raw queue entry. A valid pop settles one stop; obsolete entries are discarded without expansion.', '한 단계마다 queue entry 하나를 꺼냅니다. 유효하면 stop 하나를 settled로 표시하고, 오래된 entry라면 이웃을 검사하지 않고 버립니다.');
    }
    ctx.onReset(() => { stop(); stations = stationPreset ? stationPreset.value.split(',') : ['A', 'H']; loadedStations = stationPreset ? stationPreset.value : ''; manual = ['A']; chosen = dispatch ? 'E' : 'A'; restart(); });
    ctx.onChange(() => {
      if (stationPreset && loadedStations !== stationPreset.value) { stations = stationPreset.value.split(','); loadedStations = stationPreset.value; }
      if (!search || signature !== road.value + '|' + stations.join(',')) { manual = ['A']; restart(); }
      paint();
    });
    ctx.checkpointState({
      read: () => ({ road: road.value, stations: stations.slice(), loadedStations, chosen, manual: manual.slice(), checked, message, steps: search.pops + (search.event && search.event.kind === 'empty' ? 1 : 0) }),
      write: s => {
        const ids = NODES.map(n => n.id);
        if (!s || !PRESETS.includes(s.road) || !Array.isArray(s.stations) || s.stations.length !== 2 || s.stations[0] === s.stations[1] || !s.stations.every(id => ids.includes(id)) || !ids.includes(s.chosen) || typeof s.checked !== 'boolean' || typeof s.message !== 'string' || s.message.length > 600 || !Number.isInteger(s.steps) || s.steps < 0 || s.steps > 100 || (dispatch ? !['A,H', 'A,G', 'B,F'].includes(s.loadedStations) : s.loadedStations !== '') || !Array.isArray(s.manual) || !s.manual.length || s.manual.length > 1000 || s.manual[0] !== 'A') return false;
        const nextGraph = makeNetwork(s.road), next = new Dijkstra(nextGraph, dispatch ? s.stations : ['A'], dispatch ? null : 'H');
        if (!manualRoute(nextGraph, s.manual).legal) return false;
        for (let i = 0; i < s.steps; i++) { if (next.done) return false; next.step(); }
        stop(); road.value = s.road; stations = s.stations.slice(); loadedStations = s.loadedStations; if (stationPreset) stationPreset.value = loadedStations;
        graph = nextGraph; search = next; signature = road.value + '|' + stations.join(','); chosen = s.chosen; manual = s.manual.slice(); checked = s.checked; message = s.message; paint(); return true;
      }
    });
  }
  window.ArticleLabs.register('search-dijkstra-delivery', ctx => teachingLab(ctx, false));
  window.ArticleLabs.register('search-dijkstra-dispatch', ctx => teachingLab(ctx, true));
}());
