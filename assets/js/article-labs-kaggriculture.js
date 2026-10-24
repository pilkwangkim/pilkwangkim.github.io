(function (global) {
  'use strict';
  const softmax = logits => { const m = Math.max(...logits), e = logits.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
  function bcStep(rate, target, steps) {
    let logits = [0.1, 0.2, 0.7].map(Math.log);
    for (let n = 0; n < steps; n++) { const p = softmax(logits); logits = logits.map((x, i) => x - rate * (p[i] - Number(i === target))); }
    const probabilities = softmax(logits);
    return { probabilities, loss: -Math.log(probabilities[target]) };
  }
  function ppo(ratio, advantage, epsilon) {
    const clipped = Math.max(1 - epsilon, Math.min(1 + epsilon, ratio));
    return { raw: ratio * advantage, clipped: clipped * advantage, objective: Math.min(ratio * advantage, clipped * advantage) };
  }
  function wages(count) { let a = 1, b = 1, total = 0; for (let i = 0; i < count; i++) { total += a; [a, b] = [b, a + b]; } return { total, next: a }; }
  const crops = { wheat: [10, 25, 2, 0], carrot: [20, 35, 2, 0], tomato: [50, 60, 8, 1], strawberry: [100, 120, 10, 2], melon: [80, 250, 10, 0] };
  function harvest(crop, planted, day) {
    const [cost, price, growth, cadence] = crops[crop];
    const elapsed = day - planted - growth;
    const units = elapsed < 0 ? 0 : cadence ? Math.floor(elapsed / cadence) + 1 : 1;
    return { units, revenue: units * price, net: day < planted ? 0 : units * price - cost };
  }
  const factorial = n => { let p = 1; for (let i = 2; i <= n; i++) p *= i; return p; };
  function paired(weight) {
    const worlds = Array.from({ length: 10 }, (_, i) => i < 2 ? 0.25 : i < 4 ? 0.25 * (2 * weight - 1) : 0);
    const distribution = [];
    for (let a = 0; a <= 10; a++) for (let b = 0; b <= 10 - a; b++) {
      const c = 10 - a - b;
      distribution.push({ value: (a * 0.25 + b * 0.25 * (2 * weight - 1)) / 10,
        probability: factorial(10) / (factorial(a) * factorial(b) * factorial(c)) * Math.pow(0.2, a + b) * Math.pow(0.6, c) });
    }
    distribution.sort((a, b) => a.value - b.value);
    const quantile = q => { let cumulative = 0; for (const d of distribution) { cumulative += d.probability; if (cumulative + 1e-12 >= q) return d.value; } return distribution.at(-1).value; };
    return { worlds, mean: worlds.reduce((a, b) => a + b, 0) / 10, distribution, low: quantile(0.025), high: quantile(0.975) };
  }
  function discounted(gamma, step) { const rewards = [-10, 0, 20]; return rewards.slice(step).reduce((sum, value, i) => sum + value * Math.pow(gamma, i), 0); }
  function plans(cash, risk, strawberryScale) {
    const specifications = [[4, 0, 0], [6, 3, 3], [6, 6, 0], [6, 0, 6]];
    const scenarios = [[[100, 80], [70, 60]], [[60, 50], [140, 120]], [[20, 20], [220, 180]]];
    return specifications.map(([workers, tomatoes, strawberries], i) => {
      const plots = tomatoes + strawberries, cost = (workers === 6 ? 8 : 0) + (plots ? 1000 + 50 * tomatoes + 100 * strawberries : 0);
      const values = scenarios.map(([tp, sp]) => [0.9, 0.81].reduce((sum, discount, d) => sum + discount * (3 * tomatoes * tp[d] + (d ? 4 * strawberries * sp[d] * strawberryScale : 0) - [2, 3][d] * plots), 0));
      const mean = values.reduce((a, b) => a + b, 0) / 3, sd = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / 3);
      return { name: 'ABCD'[i], cost, feasible: cost + 75 <= cash, values, mean, sd, score: mean - risk * sd - cost };
    });
  }
  const model = { bcStep, ppo, wages, harvest, paired, discounted, plans };
  if (typeof module === 'object' && module.exports) module.exports = model;
  if (!global.ArticleLabs) return;
  const register = global.ArticleLabs.register;
  register('kaggriculture-return', ctx => {
    const step = ctx.control('Decision time', '의사결정 시점', { value: 0, min: 0, max: 2, step: 1 });
    const gamma = ctx.control('Discount γ', '할인율 γ', { value: 0.9, min: 0, max: 1, step: 0.01 });
    const cash = ctx.metric('Cash now → final', '현재 → 최종 현금'), ret = ctx.metric('Return from here', '이 시점부터의 수익');
    ctx.onChange(() => {
      const t = Number(step.value), g = Number(gamma.value), balances = [30, 20, 20, 40];
      cash.textContent = balances[t] + ' → 40'; ret.textContent = ctx.format(discounted(g, t), 3);
      ctx.view.replaceChildren(ctx.plot({ x: [0, 3], y: [0, 45], xTicks: [0, 1, 2, 3], xLabel: ctx.text('State index', '상태 순서'), yLabel: ctx.text('Cash', '현금'),
        series: [{ points: balances.map((b, i) => [i, b]), color: 'blue' }], markers: [{ x: t, y: balances[t], color: 'gold' }] }),
      ctx.table([ctx.text('Action', '행동'), ctx.text('Cash change', '현금 변화'), ctx.text('Discounted contribution', '할인된 기여')], ['Prepare', 'Tend', 'Sell'].slice(t).map((action, i) => [ctx.ko ? ['준비', '관리', '판매'][t + i] : action, [-10, 0, 20][t + i], ctx.format([-10, 0, 20][t + i] * Math.pow(g, i), 3)])));
      ctx.status.textContent = ctx.text('Cash changes by +10 over the episode; the discounted return is a separate learning quantity.', '전체 현금 증가는 +10입니다. 할인 수익은 학습에 쓰는 별도의 값입니다.');
    });
  });
  register('kaggriculture-bc', ctx => {
    const rate = ctx.control('Learning rate', '학습률', { value: 0.5, min: 0, max: 2, step: 0.05 });
    const target = ctx.control('Teacher action', '교사 행동', { type: 'select', value: 0, options: [{ value: 0, en: 'SELL', ko: 'SELL · 판매' }, { value: 1, en: 'BUY', ko: 'BUY · 구매' }, { value: 2, en: 'HOLD', ko: 'HOLD · 유지' }] });
    const steps = ctx.control('Gradient steps', '경사 갱신 횟수', { value: 1, min: 0, max: 20, step: 1 });
    const loss = ctx.metric('Cross-entropy', '교차 엔트로피'), probability = ctx.metric('Teacher probability', '교사 행동 확률');
    ctx.onChange(() => {
      const result = bcStep(Number(rate.value), Number(target.value), Number(steps.value));
      loss.textContent = ctx.format(result.loss); probability.textContent = ctx.format(result.probabilities[Number(target.value)]);
      ctx.view.replaceChildren(ctx.plot({ x: [-0.5, 2.5], y: [0, 1], xTicks: [0, 1, 2], xLabel: '0 SELL · 1 BUY · 2 HOLD', yLabel: ctx.text('Action probability', '행동 확률'), bars: result.probabilities.map((p, i) => ({ x: i, y: p, width: 0.6, color: i === Number(target.value) ? 'gold' : 'blue' })) }),
        ctx.table([ctx.text('Action', '행동'), ctx.text('Before', '갱신 전'), ctx.text('After', '갱신 후')], ['SELL', 'BUY', 'HOLD'].map((a, i) => [a, [0.1, 0.2, 0.7][i], ctx.format(result.probabilities[i])])));
    });
  });
  register('kaggriculture-ppo', ctx => {
    const ratio = ctx.control('New / old probability', '새 확률 / 이전 확률', { value: 1.1, min: 0.1, max: 2, step: 0.01 });
    const advantage = ctx.control('Advantage A', '어드밴티지 A', { value: 1, min: -2, max: 2, step: 0.1 });
    const epsilon = ctx.control('Clip ε', '클립 ε', { value: 0.2, min: 0, max: 0.5, step: 0.01 });
    const raw = ctx.metric('rA', 'rA'), clipped = ctx.metric('Clipped objective', '클립 목적함수');
    ctx.onChange(() => {
      const r = Number(ratio.value), a = Number(advantage.value), e = Number(epsilon.value), current = ppo(r, a, e);
      raw.textContent = ctx.format(current.raw); clipped.textContent = ctx.format(current.objective);
      const points = Array.from({ length: 96 }, (_, i) => 0.1 + i * 1.9 / 95);
      ctx.view.replaceChildren(ctx.plot({ x: [0.1, 2], y: [-4.2, 4.2], xLabel: ctx.text('Probability ratio r', '확률 비율 r'), yLabel: ctx.text('Objective', '목적함수'), bands: [{ x0: 1 - e, x1: 1 + e, color: 'gold' }],
        series: [{ points: points.map(x => [x, x * a]), color: 'muted', dash: true, label: 'rA' }, { points: points.map(x => [x, ppo(x, a, e).objective]), color: 'blue', label: ctx.text('PPO', 'PPO') }], markers: [{ x: r, y: current.objective, color: 'gold' }] }));
      ctx.status.textContent = ctx.text('Clipping changes this sample’s objective. It does not constrain every updated probability to the shaded interval.', '클리핑은 이 표본의 목적함수를 바꿉니다. 갱신된 모든 확률을 음영 구간에 강제로 가두지는 않습니다.');
    });
  });
  register('kaggriculture-replay', ctx => {
    const time = ctx.control('Observation time t', '관측 시점 t', { value: 0, min: 0, max: 2, step: 1 });
    const correct = ctx.control('Use action row t + 1', '행동 행 t + 1 사용', { type: 'checkbox', value: true });
    const row = ctx.metric('Action row', '행동 행'), action = ctx.metric('Selected action', '선택한 행동');
    ctx.onChange(() => {
      const t = Number(time.value), r = t + Number(correct.checked), actions = ['∅', 'PREPARE', 'TEND', 'SELL'];
      row.textContent = String(r); action.textContent = actions[r];
      ctx.view.replaceChildren(ctx.table([ctx.text('Replay row', '리플레이 행'), ctx.text('Cash / state', '현금 / 상태'), ctx.text('Action that led here', '이 상태를 만든 행동')], ['30 / empty', '20 / growing', '20 / ripe', '40 / done'].map((s, i) => [i, (i === t ? '● ' : '') + s, (i === r ? '→ ' : '') + actions[i]])),
        ctx.el('p', {}, ctx.text('Training pair: observation[' + t + '] → action[' + r + '] = ' + actions[r], '학습 쌍: observation[' + t + '] → action[' + r + '] = ' + actions[r])));
      ctx.status.textContent = correct.checked ? ctx.text('Aligned: the next row records the action chosen from the current observation.', '정렬됨: 다음 행에 현재 관측에서 선택한 행동이 기록됩니다.') : ctx.text('Misaligned: this row records the previous action; row 0 has no action.', '어긋남: 같은 행에는 이전 행동이 기록됩니다. 0번 행에는 행동이 없습니다.');
    });
  });
  register('kaggriculture-paired', ctx => {
    const weight = ctx.control('Opponent E weight', '상대 E 가중치', { value: 0.7, min: 0, max: 1, step: 0.05 });
    const mean = ctx.metric('Weighted mean Δ', '가중 평균 Δ'), interval = ctx.metric('World-bootstrap 95%', '월드 부트스트랩 95%');
    ctx.onChange(() => {
      const result = paired(Number(weight.value));
      mean.textContent = ctx.format(result.mean); interval.textContent = '[' + ctx.format(result.low) + ', ' + ctx.format(result.high) + ']';
      ctx.view.replaceChildren(ctx.plot({ x: [0.4, 10.6], y: [-0.3, 0.3], xTicks: [1, 3, 5, 7, 10], xLabel: ctx.text('Independent world', '독립 월드'), yLabel: ctx.text('Paired weighted Δ', '짝지은 가중 Δ'), bars: result.worlds.map((v, i) => ({ x: i + 1, y: v, width: 0.65, color: v < 0 ? 'red' : 'blue' })) }),
        ctx.el('p', {}, ctx.text('80 game results form 10 world clusters. Resample the entire world, including both seats and opponents.', '80개 게임 결과가 10개 월드 군집을 이룹니다. 양쪽 좌석과 상대를 포함한 월드 전체를 함께 재표집합니다.')));
      ctx.status.textContent = ctx.text('Exact empirical bootstrap percentiles; these can differ from the article’s 1,000 seeded draws.', '경험적 부트스트랩의 정확한 백분위수입니다. 본문의 시드 고정 1,000회 추출값과는 차이가 날 수 있습니다.');
    });
  });
  register('kaggriculture-harvest', ctx => {
    const crop = ctx.control('Crop', '작물', { type: 'select', value: 'tomato', options: Object.keys(crops).map((c, i) => ({ value: c, en: c[0].toUpperCase() + c.slice(1), ko: ['밀', '당근', '토마토', '딸기', '멜론'][i] })) });
    const planted = ctx.control('Planting day', '파종일', { value: 0, min: 0, max: 29, step: 1 });
    const day = ctx.control('Sell through day', '판매 마감일', { value: 29, min: 0, max: 29, step: 1 });
    const units = ctx.metric('Base harvest units', '기본 수확량'), net = ctx.metric('Revenue − seed cost', '매출 − 씨앗 비용');
    ctx.onChange(() => {
      const p = Number(planted.value), d = Number(day.value), result = harvest(crop.value, p, d);
      units.textContent = String(result.units); net.textContent = ctx.format(result.net);
      ctx.view.replaceChildren(ctx.plot({ x: [0, 29], y: [-100, 1800], xTicks: [0, 7, 14, 21, 29], xLabel: ctx.text('Day, starting at 0', '0부터 센 날짜'), yLabel: ctx.text('Net cash at base price', '기준 가격의 순현금'), series: [{ points: Array.from({ length: 30 }, (_, i) => [i, i < p ? 0 : harvest(crop.value, p, i).net]), color: 'blue' }], markers: [{ x: d, y: result.net, color: 'gold' }] }));
    });
  });
  register('kaggriculture-wages', ctx => {
    const workers = ctx.control('Hired hands today', '오늘 고용한 일꾼', { value: 12, min: 0, max: 14, step: 1 });
    const total = ctx.metric('Total daily wage', '하루 총임금'), next = ctx.metric('Next hand alone', '다음 일꾼 한 명');
    ctx.onChange(() => {
      const count = Number(workers.value), result = wages(count); total.textContent = String(result.total); next.textContent = String(result.next);
      ctx.view.replaceChildren(ctx.plot({ x: [-0.5, 14.5], y: [0, 1050], xTicks: [0, 4, 8, 12, 14], xLabel: ctx.text('Hired hands', '고용한 일꾼 수'), yLabel: ctx.text('Cumulative daily wages', '하루 누적 임금'), bars: Array.from({ length: 15 }, (_, i) => ({ x: i, y: wages(i).total, width: 0.65, color: i === count ? 'gold' : 'blue' })) }));
    });
  });
  register('kaggriculture-plans', ctx => {
    const cash = ctx.control('Cash before investment', '투자 전 현금', { value: 4000, min: 0, max: 5000, step: 25 });
    const risk = ctx.control('Scenario deviation penalty', '시나리오 표준편차 페널티', { value: 0.25, min: 0, max: 1, step: 0.05 });
    const price = ctx.control('Strawberry price multiplier', '딸기 가격 배수', { value: 1, min: 0, max: 2, step: 0.05 });
    const winner = ctx.metric('Best feasible plan', '실행 가능한 최선의 계획');
    ctx.onChange(() => {
      const results = plans(Number(cash.value), Number(risk.value), Number(price.value));
      const best = results.filter(r => r.feasible).sort((a, b) => b.score - a.score)[0];
      winner.textContent = best ? best.name + ' · ' + ctx.format(best.score, 2) : ctx.text('No feasible plan', '실행 가능한 계획 없음');
      ctx.view.replaceChildren(ctx.table([ctx.text('Plan', '계획'), ctx.text('Cost today', '당일 비용'), ctx.text('Future mean', '미래 가치 평균'), ctx.text('Score J', '평가값 J')], results.map(r => [r.name + (best?.name === r.name ? ' ●' : ''), r.cost, ctx.format(r.mean, 2), r.feasible ? ctx.format(r.score, 2) : ctx.text('Cash constraint', '현금 부족')])));
      ctx.status.textContent = ctx.text('A: retain · B: 3 tomato + 3 strawberry · C: 6 tomato · D: 6 strawberry. Each new-land plan hires two more workers.', 'A: 유지 · B: 토마토 3 + 딸기 3 · C: 토마토 6 · D: 딸기 6. 확장 계획은 모두 일꾼 두 명을 추가합니다.');
    });
  });
})(typeof globalThis !== 'undefined' ? globalThis : this);
