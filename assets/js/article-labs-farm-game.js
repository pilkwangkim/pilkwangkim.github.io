(function (global) {
  'use strict';
  const DAYS = 10, INITIAL_CASH = 60, BASE_LABOR = 2, MAX_HIRES = 3;
  const CROPS = Object.freeze({
    wheat: Object.freeze({ cost: 10, price: 25, growth: 2, repeat: false }),
    carrot: Object.freeze({ cost: 20, price: 35, growth: 2, repeat: false }),
    tomato: Object.freeze({ cost: 50, price: 60, growth: 8, repeat: true })
  });
  const copy = value => JSON.parse(JSON.stringify(value));
  function validSeed(seed) { return Number.isInteger(seed) && seed >= 1 && seed <= 99999; }
  function random(seed, day, channel) {
    let n = (seed ^ Math.imul(day + 1, 0x9e3779b1) ^ Math.imul(channel + 1, 0x85ebca6b)) >>> 0;
    n ^= n >>> 16; n = Math.imul(n, 0x7feb352d); n ^= n >>> 15; n = Math.imul(n, 0x846ca68b); n ^= n >>> 16;
    return (n >>> 0) / 4294967296;
  }
  function world(seed, day) {
    if (!validSeed(seed) || !Number.isInteger(day) || day < 0 || day >= DAYS) throw new Error('Invalid world day');
    return { rain: random(seed, day, 0) < 0.45,
      prices: Object.fromEntries(Object.entries(CROPS).map(([name, crop], i) => [name, Math.round(crop.price * (0.8 + 0.4 * random(seed, day, i + 1)))])) };
  }
  function initial(seed = 731) {
    if (!validSeed(seed)) throw new Error('Invalid seed');
    return { seed, day: 0, cash: INITIAL_CASH, labor: BASE_LABOR, hires: 0, plots: [null, null, null, null], ledger: [] };
  }
  function cleanAction(action) {
    if (!action || typeof action !== 'object' || Array.isArray(action)) return null;
    if (['end', 'hire'].includes(action.type)) return { type: action.type };
    if (!['plant', 'water', 'harvest'].includes(action.type) || !Number.isInteger(action.plot) || action.plot < 0 || action.plot > 3) return null;
    if (action.type === 'plant' && !Object.hasOwn(CROPS, action.crop)) return null;
    return { type: action.type, plot: action.plot, ...(action.type === 'plant' ? { crop: action.crop } : {}) };
  }
  function transition(previous, requested) {
    const action = cleanAction(requested);
    const fail = error => ({ ok: false, error, state: previous });
    if (!action) return fail('action');
    if (previous.day >= DAYS) return fail('finished');
    const plot = previous.plots[action.plot], weather = world(previous.seed, previous.day);
    if (['plant', 'water', 'harvest'].includes(action.type) && previous.labor < 1) return fail('labor');
    if (action.type === 'plant' && plot) return fail('occupied');
    if (action.type === 'plant' && previous.cash < CROPS[action.crop].cost) return fail('cash');
    if (action.type === 'water' && !plot) return fail('empty');
    if (action.type === 'water' && (plot.watered || weather.rain)) return fail('watered');
    if (action.type === 'harvest' && (!plot || plot.progress < plot.nextYield)) return fail('unripe');
    if (action.type === 'hire' && previous.hires >= MAX_HIRES) return fail('hires');
    const wage = [1, 1, 2][previous.hires];
    if (action.type === 'hire' && previous.cash < wage) return fail('cash');
    const state = copy(previous), event = { day: state.day, ...action, change: 0 };
    if (action.type === 'plant') {
      state.cash -= CROPS[action.crop].cost; state.labor -= 1;
      state.plots[action.plot] = { crop: action.crop, progress: 0, nextYield: CROPS[action.crop].growth, watered: true };
      event.change = -CROPS[action.crop].cost;
    } else if (action.type === 'water') {
      state.plots[action.plot].watered = true; state.labor -= 1;
    } else if (action.type === 'harvest') {
      event.crop = plot.crop; event.change = weather.prices[plot.crop];
      state.cash += event.change; state.labor -= 1;
      if (CROPS[plot.crop].repeat) state.plots[action.plot].nextYield = plot.progress + 1;
      else state.plots[action.plot] = null;
    } else if (action.type === 'hire') {
      state.cash -= wage; state.labor += 1; state.hires += 1; event.change = -wage;
    } else {
      state.plots.forEach(tile => { if (tile) { if (tile.watered || weather.rain) tile.progress += 1; tile.watered = false; } });
      state.day += 1; state.labor = state.day < DAYS ? BASE_LABOR : 0; state.hires = 0;
    }
    event.cash = state.cash; event.labor = state.labor;
    state.ledger.push(event);
    return { ok: true, state, action };
  }
  function replay(seed, actions, count = actions && actions.length) {
    if (!validSeed(seed) || !Array.isArray(actions) || actions.length > 100 || !Number.isInteger(count) || count < 0 || count > actions.length) throw new Error('Invalid farm history');
    let state = initial(seed);
    for (const action of actions.slice(0, count)) {
      const step = transition(state, action);
      if (!step.ok) throw new Error('Invalid farm action: ' + step.error);
      state = step.state;
    }
    return state;
  }
  function cashLine(state) {
    const values = [[0, INITIAL_CASH]];
    for (const row of state.ledger) {
      const day = row.type === 'end' ? row.day + 1 : row.day;
      if (values.at(-1)[0] === day) values[values.length - 1][1] = row.cash;
      else values.push([day, row.cash]);
    }
    return values;
  }
  const model = { DAYS, INITIAL_CASH, BASE_LABOR, MAX_HIRES, CROPS, validSeed, world, initial, cleanAction, transition, replay, cashLine };
  if (typeof module === 'object' && module.exports) module.exports = model;
  if (!global.ArticleLabs) return;
  global.ArticleLabs.register('kaggriculture-farm-game', ctx => {
    ctx.root.classList.add('al-farm-game');
    let seed = 731, history = [], cursor = 0, reference = null, selected = 0, message = '';
    const names = { wheat: ctx.text('Wheat', '밀'), carrot: ctx.text('Carrot', '당근'), tomato: ctx.text('Tomato', '토마토') };
    const seedControl = ctx.control('World seed', '월드 시드', { type: 'number', value: seed, min: 1, max: 99999, step: 1 });
    const cropControl = ctx.control('Seed to plant · cost / growth days', '심을 작물 · 비용 / 성장일', { type: 'select', value: 'wheat', options: Object.entries(CROPS).map(([name, crop]) => ({ value: name, en: names[name] + ' · ' + crop.cost + ' / ' + crop.growth, ko: names[name] + ' · ' + crop.cost + ' / ' + crop.growth })) });
    const rewind = ctx.control('Return to a morning', '이전 날 아침으로 돌아가기', { value: 0, min: 0, max: 0, step: 1 });
    const controls = ctx.el('div', { class: 'fg-actions' }); ctx.ui.append(controls);
    function button(en, ko, handler) { const node = ctx.button(en, ko, handler); controls.append(node); return node; }
    const errors = {
      labor: ctx.text('No actions left. Hire a hand or end the day.', '남은 작업이 없습니다. 일꾼을 고용하거나 하루를 마치세요.'),
      cash: ctx.text('Not enough cash for that purchase.', '구매할 현금이 부족합니다.'),
      occupied: ctx.text('Choose an empty plot for planting.', '씨앗은 빈 밭에 심을 수 있습니다.'),
      empty: ctx.text('There is no crop to water here.', '물을 줄 작물이 없습니다.'),
      watered: ctx.text('This plot already has water today.', '오늘 필요한 물은 이미 공급됐습니다.'),
      unripe: ctx.text('This crop has no ripe yield yet.', '아직 수확할 작물이 익지 않았습니다.'),
      hires: ctx.text('The daily limit is three hired hands.', '하루에는 일꾼을 세 명까지 고용할 수 있습니다.'),
      finished: ctx.text('The season is over. Rewind or keep this path and restart.', '시즌이 끝났습니다. 이전 날로 돌아가거나 경로를 보관하고 다시 시작하세요.')
    };
    function stateNow() { return replay(seed, history, cursor); }
    function act(action) {
      const step = transition(stateNow(), action);
      if (!step.ok) { message = errors[step.error] || ctx.text('Choose a valid action.', '가능한 행동을 선택하세요.'); render(); return; }
      if (cursor < history.length) reference = { seed, actions: history.map(cleanAction) };
      history = history.slice(0, cursor).concat(step.action); cursor = history.length; message = ''; render();
    }
    const plant = button('Plant', '파종', () => act({ type: 'plant', plot: selected, crop: cropControl.value }));
    const water = button('Water', '물 주기', () => act({ type: 'water', plot: selected }));
    const harvest = button('Harvest & sell', '수확·판매', () => act({ type: 'harvest', plot: selected }));
    const hire = button('Hire today', '오늘 일꾼 고용', () => act({ type: 'hire' }));
    const end = button('End day →', '하루 마치기 →', () => act({ type: 'end' })); end.classList.add('fg-end');
    const undo = button('Undo action', '한 행동 되돌리기', () => { if (cursor > 0) cursor -= 1; message = ''; render(); });
    const keep = button('Keep path & restart', '경로 보관·다시 시작', () => { reference = { seed, actions: history.slice(0, cursor).map(cleanAction) }; history = []; cursor = 0; message = ''; render(); }); keep.classList.add('fg-keep');
    const dayMetric = ctx.metric('Day / season', '날짜 / 시즌'), cashMetric = ctx.metric('Cash · change from 60', '현금 · 초기 60 대비'), laborMetric = ctx.metric('Actions remaining', '남은 작업'), compareMetric = ctx.metric('Same morning vs kept path', '같은 아침의 보관 경로 대비');
    const overview = ctx.el('p', { class: 'fg-overview' }), prices = ctx.el('p', { class: 'fg-prices' });
    const grid = ctx.el('div', { class: 'fg-grid', role: 'group', 'aria-label': ctx.text('Choose one of four plots', '네 밭 중 하나 선택') });
    const tiles = Array.from({ length: 4 }, (_, i) => {
      const node = ctx.el('button', { type: 'button', class: 'fg-tile', 'aria-pressed': i === 0 ? 'true' : 'false' });
      node.addEventListener('click', () => { selected = i; render(); }); grid.append(node); return node;
    });
    const chart = ctx.el('div', { class: 'fg-cash-chart' });
    const ledger = ctx.el('details', { class: 'fg-ledger' }), ledgerSummary = ctx.el('summary'), ledgerBody = ctx.el('div'); ledger.append(ledgerSummary, ledgerBody);
    ctx.view.append(overview, prices, grid, chart, ledger);
    function render() {
      if (validSeed(Number(seedControl.value)) && Number(seedControl.value) !== seed) { seed = Number(seedControl.value); history = []; cursor = 0; reference = null; message = ''; }
      const state = stateNow(), finished = state.day === DAYS, weather = world(seed, Math.min(state.day, DAYS - 1));
      const latest = replay(seed, history), past = reference ? replay(seed, reference.actions) : null;
      rewind.max = String(latest.day); rewind.value = String(state.day);
      dayMetric.textContent = finished ? ctx.text('Season complete', '시즌 종료') : (state.day + 1) + ' / ' + DAYS;
      cashMetric.textContent = state.cash + ' (' + (state.cash >= INITIAL_CASH ? '+' : '') + (state.cash - INITIAL_CASH) + ')';
      laborMetric.textContent = String(state.labor);
      let previousMorning = null;
      if (past) {
        const stops = reference.actions.map((a, i) => ({ a, i })).filter(item => item.a.type === 'end');
        const match = state.day === 0 ? 0 : stops[state.day - 1] ? stops[state.day - 1].i + 1 : null;
        if (match !== null) previousMorning = replay(seed, reference.actions, match).cash;
      }
      const thisMorning = state.day ? state.ledger.filter(e => e.type === 'end').at(-1).cash : INITIAL_CASH;
      compareMetric.textContent = previousMorning === null ? '—' : (thisMorning - previousMorning >= 0 ? '+' : '') + (thisMorning - previousMorning);
      overview.textContent = finished ? ctx.text('Season closed: only cash counts; unharvested crops have no salvage value.', '시즌 종료: 현금만 남고, 수확하지 못한 작물은 잔존가치가 없습니다.') : ctx.text('Today: ', '오늘: ') + (weather.rain ? ctx.text('rain · all plots watered', '비 · 모든 밭에 물 공급') : ctx.text('dry · water to advance growth', '건조 · 물을 줘야 성장'));
      prices.textContent = ctx.text('Today’s unit sale prices: ', '오늘의 단위 판매가: ') + Object.keys(CROPS).map(c => names[c] + ' ' + weather.prices[c]).join(' · ');
      tiles.forEach((node, i) => {
        const tile = state.plots[i], ready = tile && tile.progress >= tile.nextYield;
        node.setAttribute('aria-pressed', String(i === selected)); node.dataset.ripe = String(Boolean(ready));
        node.replaceChildren(ctx.el('strong', {}, ctx.text('Plot ', '밭 ') + (i + 1)), ctx.el('span', {}, tile ? names[tile.crop] : ctx.text('Empty', '빈 밭')),
          ctx.el('small', {}, tile ? ready ? ctx.text('Ready to harvest', '수확 가능') : ctx.text('Growth ', '성장 ') + tile.progress + ' / ' + tile.nextYield : ctx.text('Select → plant a seed', '선택 → 씨앗 파종')),
          ctx.el('small', {}, tile ? (tile.watered || weather.rain ? ctx.text('Water supplied', '물 공급됨') : ctx.text('Needs water today', '오늘 물 필요')) : ''));
      });
      const currentLine = cashLine(state), pastLine = past ? cashLine(past) : [];
      const allValues = currentLine.concat(pastLine).map(p => p[1]);
      chart.replaceChildren(ctx.plot({ x: [0, DAYS], y: [0, Math.max(80, ...allValues) * 1.08], xTicks: [0, 2, 4, 6, 8, 10], xLabel: ctx.text('Morning, starting at 0', '0부터 센 아침'), yLabel: ctx.text('Cash (coins)', '현금 (코인)'), series: [{ points: currentLine, color: 'blue', label: ctx.text('Current path', '현재 경로') }, ...(past ? [{ points: pastLine, color: 'gold', dash: true, label: ctx.text('Kept path · same seed', '보관 경로 · 같은 시드') }] : [])] }));
      const daily = Array.from({ length: state.day + Number(!finished) }, (_, day) => {
        const rows = state.ledger.filter(e => e.day === day), spend = rows.filter(e => e.change < 0).reduce((s, e) => s - e.change, 0), sales = rows.filter(e => e.change > 0).reduce((s, e) => s + e.change, 0);
        const balance = rows.length ? rows.at(-1).cash : day === 0 ? INITIAL_CASH : state.ledger.filter(e => e.type === 'end' && e.day === day - 1).at(-1).cash;
        return [day + 1, world(seed, day).rain ? ctx.text('Rain', '비') : ctx.text('Dry', '건조'), spend, sales, balance];
      });
      ledgerSummary.textContent = ctx.text('Daily cash ledger', '일별 현금 장부') + ' (' + daily.length + ')';
      ledgerBody.replaceChildren(ctx.table([ctx.text('Day', '날짜'), ctx.text('Weather', '날씨'), ctx.text('Spent', '지출'), ctx.text('Sales', '매출'), ctx.text('Cash', '현금')], daily));
      [plant, water, harvest, hire, end].forEach(node => { node.disabled = finished; });
      undo.disabled = cursor === 0; keep.disabled = cursor === 0;
      hire.textContent = ctx.text('Hire today', '오늘 일꾼 고용') + ' · ' + (state.hires < MAX_HIRES ? [1, 1, 2][state.hires] : ctx.text('limit', '한도'));
      ctx.status.textContent = message || ctx.text('Select a plot, then act. Planting includes today’s water; each plot action uses one work unit. Rewind before changing a decision to keep the old path for comparison.', '밭을 선택한 뒤 행동하세요. 파종에는 당일 물 주기가 포함되며, 밭 작업마다 작업 1회를 씁니다. 이전으로 돌아가 결정을 바꾸면 기존 경로가 비교용으로 보관됩니다.');
    }
    rewind.addEventListener('input', () => {
      const requested = Number(rewind.value); let target = 0, day = 0;
      while (target < history.length && day < requested) { if (history[target].type === 'end') day += 1; target += 1; }
      cursor = target; message = ''; render();
    });
    ctx.onReset(() => { seed = 731; history = []; cursor = 0; reference = null; selected = 0; message = ''; rewind.max = '0'; });
    ctx.onChange(render);
    if (ctx.stateLink) ctx.stateLink('kaggriculture-farm-game', {
      read: () => ({ seed, actions: history, cursor, selected, ...(reference ? { reference: reference.actions } : {}) }),
      write: data => {
        if (!data || !validSeed(data.seed) || !Number.isInteger(data.cursor) || !Number.isInteger(data.selected) || data.selected < 0 || data.selected > 3) throw new Error('Invalid farm state');
        replay(data.seed, data.actions); replay(data.seed, data.actions, data.cursor);
        if (data.reference !== undefined) replay(data.seed, data.reference);
        seed = data.seed; seedControl.value = String(seed); history = data.actions.map(cleanAction); cursor = data.cursor; selected = data.selected;
        reference = data.reference ? { seed, actions: data.reference.map(cleanAction) } : null; message = ''; render();
      }
    });
  });
})(typeof window !== 'undefined' ? window : globalThis);
