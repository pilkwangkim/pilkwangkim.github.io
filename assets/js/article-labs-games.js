(function () {
  'use strict';
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  function fleetSpeed(ships) { return ships <= 1 ? 1 : 1 + 5 * clamp(Math.log(ships) / Math.log(1000), 0, 1) ** 1.5; }
  function segmentDistance(point, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1], length = dx * dx + dy * dy;
    const t = length < 1e-9 ? 0 : clamp(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length, 0, 1);
    return Math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy);
  }
  function orbitPosition(phase, omega, turns, rho = 30) { return [50 + rho * Math.cos(phase + omega * turns), 50 + rho * Math.sin(phase + omega * turns)]; }
  function routeGeometry(source, target, ships) {
    const angle = Math.atan2(target[1] - source[1], target[0] - source[0]), clearance = 3 + .1;
    const start = [source[0] + Math.cos(angle) * clearance, source[1] + Math.sin(angle) * clearance];
    const distance = Math.max(0, Math.hypot(target[0] - source[0], target[1] - source[1]) - clearance - 2);
    const end = [start[0] + Math.cos(angle) * distance, start[1] + Math.sin(angle) * distance];
    return { start, end, angle, distance, turns: Math.max(1, Math.ceil(distance / fleetSpeed(ships))), safe: segmentDistance([50, 50], start, end) >= 11.5 };
  }
  function intercept(source, ships, phase, omega, horizon = 80) {
    for (let t = 1; t <= horizon; t++) {
      const target = orbitPosition(phase, omega, t), route = routeGeometry(source, target, ships);
      if (route.safe && route.turns === t) return { ...route, target };
    }
    return null;
  }
  const mazeWalls = [[[2, 1], [2, 2]], [[1, 2], [1, 3]]];
  const same = (a, b) => a[0] === b[0] && a[1] === b[1];
  function initialMaze() { return { turn: 0, south: -1, factory: { pos: [2, 0], energy: 24 }, scout: { pos: [2, 1], energy: 8 }, jumpCooldown: 0, lost: false, message: 'start' }; }
  function mazeStep(previous, action) {
    if (previous.lost) return { ...previous, message: 'ended' };
    const state = JSON.parse(JSON.stringify(previous)); state.turn++; state.jumpCooldown = Math.max(0, state.jumpCooldown - 1);
    ['factory', 'scout'].forEach(id => { if (state[id]) state[id].energy = Math.max(0, state[id].energy - 1); });
    const actor = state[action.actor], otherId = action.actor === 'factory' ? 'scout' : 'factory', other = state[otherId];
    state.message = 'idle';
    if (!actor) state.message = 'missing';
    else if (action.type === 'TRANSFER') {
      if (!other || Math.abs(actor.pos[0] - other.pos[0]) + Math.abs(actor.pos[1] - other.pos[1]) !== 1) state.message = 'not-adjacent';
      else { const amount = Math.min(actor.energy, Math.max(0, Number(action.amount) || 0)); actor.energy -= amount; other.energy += amount; state.message = 'transfer'; }
    } else if (action.type !== 'IDLE') {
      const jump = action.type === 'JUMP_N', offset = { MOVE_N: [0, 1], MOVE_E: [1, 0], MOVE_W: [-1, 0], MOVE_S: [0, -1], JUMP_N: [0, 2] }[action.type];
      if (!offset) state.message = 'unknown';
      else {
        const target = [actor.pos[0] + offset[0], actor.pos[1] + offset[1]], cost = jump ? 3 : 1;
        const wall = mazeWalls.some(([a, b]) => (same(actor.pos, a) && same(target, b)) || (same(actor.pos, b) && same(target, a)));
        if (actor.energy < cost) state.message = 'energy';
        else if (jump && (action.actor !== 'factory' || state.jumpCooldown > 0)) state.message = 'cooldown';
        else if (target.some(v => v < 0 || v >= 5) || target[1] < state.south) state.message = 'outside';
        else if (other && same(target, other.pos)) state.message = 'occupied';
        else if (!jump && wall) state.message = 'wall';
        else { actor.pos = target; actor.energy -= cost; if (jump) state.jumpCooldown = 3; state.message = jump ? 'jump' : 'move'; }
      }
    }
    if (state.turn % 2 === 0) state.south++;
    if (state.scout && state.scout.pos[1] < state.south) state.scout = null;
    state.lost = state.factory.pos[1] < state.south;
    return state;
  }
  function legalTopK(scores, indices, minCount, maxCount, requested) {
    if (scores.length !== indices.length || new Set(indices).size !== indices.length || minCount < 0 || maxCount < minCount || maxCount > indices.length) throw new Error('Invalid legal menu');
    const count = clamp(Math.round(requested), minCount, maxCount);
    return scores.map((score, i) => [score, indices[i]]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, count).map(x => x[1]);
  }
  function hybridPolicy(expert, globalScores, rule, weights, coverage, coverageFloor, marginFloor, indices, minCount, maxCount, requested) {
    const sum = weights.reduce((s, x) => s + x, 0), normalized = sum > 0 ? weights.map(x => x / sum) : [0, 0, 1];
    const blended = expert.map((x, i) => normalized[0] * x + normalized[1] * globalScores[i] + normalized[2] * rule[i]);
    const ordered = blended.slice().sort((a, b) => b - a), margin = ordered[0] - ordered[1];
    const reason = sum <= 0 ? 'zero-weights' : coverage < coverageFloor ? 'coverage' : margin < marginFloor ? 'margin' : 'blend';
    const used = reason === 'blend' ? blended : rule;
    return { normalized, blended, used, margin, reason, selected: legalTopK(used, indices, minCount, maxCount, requested) };
  }
  const api = { fleetSpeed, segmentDistance, orbitPosition, routeGeometry, intercept, mazeWalls, initialMaze, mazeStep, legalTopK, hybridPolicy };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const register = window.ArticleLabs.register, number = input => Number(input.value);

  register('games-orbit-intercept', ctx => {
    const ships = ctx.control('Ships in the final launch', '최종 발사 함선 수', { type: 'range', min: 1, max: 1000, step: 1, value: 30 });
    const phase = ctx.control('Current orbit phase (degrees)', '현재 공전 위상 (도)', { type: 'range', min: 0, max: 359, step: 1, value: 120 });
    const omega = ctx.control('Angular speed (radians / turn)', '각속도 (radian / 턴)', { type: 'range', min: 0, max: .08, step: .005, value: .04 });
    const playhead = ctx.control('Inspect turn after launch', '발사 후 살펴볼 턴', { type: 'range', min: 0, max: 80, step: 1, value: 15 });
    const speed = ctx.metric('Fleet speed', '함대 속도'), eta = ctx.metric('Safe exact-turn intercept', '안전한 정수 턴 intercept'), safety = ctx.metric('Aim-at-now route', '현재 위치로 조준한 경로');
    ctx.onChange(() => {
      const source = [12, 25], theta = number(phase) * Math.PI / 180, w = number(omega), n = number(ships), t = number(playhead), now = orbitPosition(theta, w, 0), current = routeGeometry(source, now, n), future = intercept(source, n, theta, w), target = orbitPosition(theta, w, t);
      speed.textContent = ctx.format(fleetSpeed(n), 3); eta.textContent = future ? `${future.turns} ${ctx.text('turns', '턴')}` : ctx.text('None within 80 turns', '80턴 안에 없음');
      safety.textContent = current.safe ? ctx.text(`Sun-safe; static ETA ${current.turns}`, `태양 통과 없음·정적 ETA ${current.turns}`) : ctx.text('Blocked by sun margin', '태양 안전 여유에 막힘');
      ctx.view.replaceChildren();
      const svg = ctx.svg('svg', { class: 'al-chart', viewBox: '0 0 300 260', role: 'img', 'aria-label': ctx.text('Straight fleet routes and a rotating target', '직선 함대 경로와 공전하는 목표') }), xy = p => [30 + 2.4 * p[0], 250 - 2.4 * p[1]];
      const circle = (p, radius, attrs) => { const [cx, cy] = xy(p); svg.append(ctx.svg('circle', { cx, cy, r: radius * 2.4, ...attrs })); };
      svg.append(ctx.svg('rect', { x: 30, y: 10, width: 240, height: 240, fill: 'none', stroke: '#87939c' }));
      circle([50, 50], 30, { fill: 'none', stroke: '#87939c', 'stroke-dasharray': '3 3' });
      circle([50, 50], 11.5, { fill: '#bd8d2d', opacity: .15 }); circle([50, 50], 10, { fill: '#bd8d2d', opacity: .6 });
      function route(r, color, dash) {
        const a = xy(r.start), b = xy(r.end); svg.append(ctx.svg('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: color, 'stroke-width': 2, 'stroke-dasharray': dash ? '4 3' : '' }));
        const traveled = Math.min(r.distance, t * fleetSpeed(n)), position = [r.start[0] + Math.cos(r.angle) * traveled, r.start[1] + Math.sin(r.angle) * traveled];
        circle(position, 1.2, { fill: color });
      }
      route(current, current.safe ? '#87939c' : '#b34c43', true); if (future) { route(future, '#357ca4', false); circle(future.target, 2, { fill: 'none', stroke: '#357ca4', 'stroke-width': 2 }); }
      circle(source, 3, { fill: '#357ca4' }); circle(now, 2, { fill: 'none', stroke: '#87939c', 'stroke-dasharray': '3 2' }); circle(target, 2, { fill: '#bd8d2d' });
      ctx.view.append(svg);
      const actualAtCurrentETA = orbitPosition(theta, w, current.turns), miss = Math.hypot(current.end[0] - actualAtCurrentETA[0], current.end[1] - actualAtCurrentETA[1]);
      ctx.view.append(ctx.table([ctx.text('Route', '경로'), ctx.text('ETA', 'ETA'), ctx.text('Aim point', '조준 위치')], [[ctx.text('Current-position aim', '현재 위치 조준'), current.turns, now.map(x => ctx.format(x, 1)).join(', ')], [ctx.text('Future intercept', '미래 intercept'), future ? future.turns : '—', future ? future.target.map(x => ctx.format(x, 1)).join(', ') : '—']]));
      ctx.status.textContent = ctx.text(`Gold: target at turn ${t}. Blue: safe future aim; dashed gray/red: current-position aim. At the static ETA its endpoint is ${ctx.format(miss, 1)} units from the moving target center. Search requires ceil(D/v)=T exactly.`, `금색은 ${t}턴의 목표, 파랑은 안전한 미래 조준, 회색·빨강 점선은 현재 위치 조준입니다. 정적 ETA에서 그 경로 끝점과 이동한 목표 중심의 거리는 ${ctx.format(miss, 1)}입니다. 탐색은 ceil(D/v)=T를 정확히 만족해야 합니다.`);
    });
  });

  register('games-maze-tempo', ctx => {
    let state = initialMaze(), history = [];
    const actor = ctx.control('Acting unit', '행동할 유닛', { type: 'select', value: 'scout', options: [{ value: 'factory', en: 'Factory F', ko: 'Factory F' }, { value: 'scout', en: 'Scout S', ko: 'Scout S' }] });
    const action = ctx.control('One action this turn', '이번 턴의 행동 하나', { type: 'select', value: 'TRANSFER', options: [{ value: 'TRANSFER', en: 'TRANSFER to adjacent ally', ko: 'TRANSFER: 인접 아군에 전달' }, { value: 'MOVE_N', en: 'MOVE north', ko: 'MOVE 북쪽' }, { value: 'MOVE_E', en: 'MOVE east', ko: 'MOVE 동쪽' }, { value: 'MOVE_W', en: 'MOVE west', ko: 'MOVE 서쪽' }, { value: 'MOVE_S', en: 'MOVE south', ko: 'MOVE 남쪽' }, { value: 'JUMP_N', en: 'Factory JUMP two cells north', ko: 'Factory JUMP 북쪽 두 칸' }, { value: 'IDLE', en: 'IDLE', ko: 'IDLE' }] });
    const amount = ctx.control('Transfer request', '에너지 전달 요청량', { type: 'range', min: 0, max: 10, step: 1, value: 7 });
    const margin = ctx.metric('Turn / factory margin', '턴 / Factory 여유'), energy = ctx.metric('Energy F / S', '에너지 F / S'), cooldown = ctx.metric('Jump cooldown', 'Jump cooldown');
    const messages = { start: ['Choose an action, then advance one turn.', '행동을 고른 뒤 한 턴 진행하세요.'], ended: ['The factory is below the boundary. Reset the board.', 'Factory가 경계 아래입니다. 보드를 초기화하세요.'], idle: ['Unit idles.', '유닛이 대기합니다.'], missing: ['The selected unit has left the board.', '선택한 유닛이 보드를 벗어났습니다.'], 'not-adjacent': ['Transfer requires an adjacent ally.', '전달하려면 아군이 인접해야 합니다.'], transfer: ['Energy moves; the source body stays in its cell.', '에너지는 옮겨지고 source의 몸체는 그대로 남습니다.'], unknown: ['Unknown action becomes idle.', '알 수 없는 행동은 대기가 됩니다.'], energy: ['After upkeep, energy cannot pay the action cost.', '유지비를 낸 뒤 행동 비용을 감당하지 못합니다.'], cooldown: ['Jump is unavailable for this unit or cooldown.', '이 유닛이나 cooldown 상태에서는 jump할 수 없습니다.'], outside: ['The destination is outside the safe board.', '도착점이 안전한 보드 바깥입니다.'], occupied: ['The destination has a friendly blocker.', '도착점에 아군이 막고 있습니다.'], wall: ['A known wall blocks this edge.', '알고 있는 벽이 이 edge를 막습니다.'], jump: ['Jump gains two rows and spends cooldown.', 'Jump로 두 줄을 전진하고 cooldown을 사용했습니다.'], move: ['The unit moves one cell.', '유닛이 한 칸 이동했습니다.'] };
    function render() {
      margin.textContent = `${state.turn} / ${state.factory.pos[1] - state.south}`; energy.textContent = `${state.factory.energy} / ${state.scout ? state.scout.energy : '—'}`; cooldown.textContent = String(state.jumpCooldown);
      ctx.view.replaceChildren();
      const svg = ctx.svg('svg', { class: 'al-chart', viewBox: '0 0 320 260', role: 'img', 'aria-label': ctx.text('Authored five by five survival board', '작성한 5×5 생존 보드') }), pos = p => [50 + p[0] * 44, 230 - p[1] * 44];
      for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) { const [x, y] = pos([c, r]); svg.append(ctx.svg('rect', { x, y: y - 44, width: 44, height: 44, fill: r < state.south ? '#b34c43' : '#87939c', opacity: r < state.south ? .3 : .07, stroke: '#87939c', 'stroke-width': .6 })); }
      mazeWalls.forEach(([a]) => { const [x, y] = pos(a); svg.append(ctx.svg('line', { x1: x, x2: x + 44, y1: y - 44, y2: y - 44, stroke: '#b34c43', 'stroke-width': 5 })); });
      ['factory', 'scout'].forEach(id => { const unit = state[id]; if (!unit) return; const [x, y] = pos(unit.pos); svg.append(ctx.svg('circle', { cx: x + 22, cy: y - 22, r: 16, fill: 'none', stroke: id === 'factory' ? '#357ca4' : '#bd8d2d', 'stroke-width': 3, opacity: state.lost && id === 'factory' ? .25 : 1 })); svg.append(ctx.svg('text', { x: x + 22, y: y - 15, 'text-anchor': 'middle', fill: 'currentColor' }, id === 'factory' ? 'F' : 'S')); });
      ctx.view.append(svg);
      ctx.view.append(ctx.el('p', {}, ctx.text('North ↑ · red edges are walls.', '북쪽 ↑ · 빨간 edge는 벽입니다.')));
      ctx.status.textContent = ctx.text(...messages[state.lost ? 'ended' : state.message]);
    }
    ctx.button('Advance one turn', '한 턴 진행', () => {
      if (!state.lost) {
        const next = { actor: actor.value, type: action.value, amount: number(amount) };
        state = mazeStep(state, next); history.push(next);
      }
      render();
    });
    ctx.button('Restart this board', '보드 다시 시작', () => { state = initialMaze(); history = []; render(); });
    ctx.onReset(() => { state = initialMaze(); history = []; }); ctx.onChange(render);
    if (ctx.checkpointState) ctx.checkpointState({
      read: () => ({ v: 1, history: history.map(item => ({ ...item })) }),
      write: saved => {
        if (!saved || saved.v !== 1 || !Array.isArray(saved.history) || saved.history.length > 20) throw new Error('Invalid maze history');
        let restored = initialMaze();
        const actions = saved.history.map(item => {
          if (!item || !['factory', 'scout'].includes(item.actor) || !['TRANSFER', 'MOVE_N', 'MOVE_E', 'MOVE_W', 'MOVE_S', 'JUMP_N', 'IDLE'].includes(item.type) ||
              !Number.isInteger(item.amount) || item.amount < 0 || item.amount > 10 || restored.lost) throw new Error('Invalid maze action');
          const next = { actor: item.actor, type: item.type, amount: item.amount };
          restored = mazeStep(restored, next);
          return next;
        });
        state = restored; history = actions; render();
      }
    });
  });

  register('games-pokemon-policy', ctx => {
    const expert = ctx.control('Expert weight', 'Expert 가중치', { type: 'range', min: 0, max: 1, step: .05, value: .5 });
    const globalWeight = ctx.control('Global weight', 'Global 가중치', { type: 'range', min: 0, max: 1, step: .05, value: .2 });
    const ruleWeight = ctx.control('Rule weight', 'Rule 가중치', { type: 'range', min: 0, max: 1, step: .05, value: .3 });
    const coverage = ctx.control('Context coverage', '현재 문맥의 지원 범위', { type: 'range', min: 0, max: 1, step: .05, value: .8 });
    const confidence = ctx.control('Required top-two score margin', '필요한 상위 두 점수의 차이', { type: 'range', min: 0, max: .5, step: .01, value: .1 });
    const menu = ctx.control('Current legal menu', '현재 합법 선택 메뉴', { type: 'select', value: 'single', options: [{ value: 'single', en: 'Choose exactly one: [2, 5, 8]', ko: '정확히 하나 선택: [2, 5, 8]' }, { value: 'multiple', en: 'Choose one or two: [1, 4, 7]', ko: '한두 개 선택: [1, 4, 7]' }, { value: 'optional', en: 'Choose zero to two: [3, 9, 12]', ko: '0~2개 선택: [3, 9, 12]' }] });
    const count = ctx.control('Requested number of choices', '요청한 선택 개수', { type: 'range', min: 0, max: 3, step: 1, value: 1 });
    const returned = ctx.metric('Returned option indices', '반환한 option index'), route = ctx.metric('Policy route', '정책 경로'), margin = ctx.metric('Blended top-two margin', '혼합 점수 상위 두 개의 차이');
    ctx.onChange(() => {
      const ids = { single: [2, 5, 8], multiple: [1, 4, 7], optional: [3, 9, 12] }[menu.value], min = menu.value === 'optional' ? 0 : 1, max = menu.value === 'single' ? 1 : 2;
      const E = [.9, .3, .55], G = [.45, .8, .6], R = [.25, .65, .95], r = hybridPolicy(E, G, R, [number(expert), number(globalWeight), number(ruleWeight)], number(coverage), .6, number(confidence), ids, min, max, number(count));
      const reasons = { blend: ['Blend passes both gates', '혼합값이 두 gate 통과'], coverage: ['Rules: coverage below 0.6', '규칙: 지원 범위 0.6 미만'], margin: ['Rules: blended margin too small', '규칙: 혼합 점수 차이 부족'], 'zero-weights': ['Rules: all weights zero', '규칙: 모든 가중치가 0'] };
      returned.textContent = `[${r.selected.join(', ')}]`; route.textContent = ctx.text(...reasons[r.reason]); margin.textContent = ctx.format(r.margin, 3);
      ctx.view.replaceChildren();
      ctx.view.append(ctx.table([ctx.text('Current index', '현재 index'), 'Expert', 'Global', 'Rule', ctx.text('Blend', '혼합'), ctx.text('Chosen', '선택')], ids.map((id, i) => [id, E[i], G[i], R[i], ctx.format(r.blended[i], 3), r.selected.includes(id) ? '✓' : '—'])));
      ctx.view.append(ctx.plot({ x: [0, 2], y: [0, 1], xLabel: ctx.text('Menu position (returned values use original indices)', '메뉴 위치 (반환값은 원래 index)'), yLabel: ctx.text('Authored policy score', '작성한 정책 점수'), title: ctx.text('Different sources favor different options', '서로 다른 선택지를 선호하는 점수'), markers: ids.flatMap((id, i) => [{ x: i, y: r.blended[i], label: String(id), color: 'blue' }, { x: i, y: r.used[i], label: r.selected.includes(id) ? '✓' : '', color: 'gold' }]) }));
      ctx.status.textContent = ctx.text(`Weights are normalized to ${r.normalized.map(x => ctx.format(x, 2)).join(' / ')}. minCount=${min}, maxCount=${max}; the requested count is clamped. Scores are authored examples, not win probabilities. The wrapper returns unique current indices.`, `가중치를 ${r.normalized.map(x => ctx.format(x, 2)).join(' / ')}로 정규화했습니다. minCount=${min}, maxCount=${max}에 맞춰 요청 개수를 제한합니다. 작성한 예시 점수이며 승률이 아닙니다. 반환값은 중복 없는 현재 index입니다.`);
    });
  });
}());
