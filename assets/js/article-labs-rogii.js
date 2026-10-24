/* Authored GR examples and the five historical same-submission score pairs in the article. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else api.install(root.ArticleLabs);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const PORTFOLIO = Object.freeze([
    Object.freeze({ id: 'M4', name: 'Residual direct structural', public: 6.103, private: 8.202 }),
    Object.freeze({ id: 'M5', name: 'TCN dual structural', public: 6.066, private: 8.034 }),
    Object.freeze({ id: 'M3', name: 'HGBR direct structural', public: 6.060, private: 8.132 }),
    Object.freeze({ id: 'PS1', name: 'Bounded residual structural', public: 5.952, private: 8.523 }),
    Object.freeze({ id: 'PS3', name: 'Bounded TCN dual structural', public: 5.998, private: 8.197 })
  ]);
  function gamma(t, distinctive) {
    const periodic = 55 + 12 * Math.sin(2 * Math.PI * t / 20) + 5 * Math.cos(4 * Math.PI * t / 20);
    return periodic + (distinctive ? 10 * Math.exp(-(((t - 11) / 2.5) ** 2)) : 0);
  }
  function noise(t) { return 1.2 * Math.sin(1.7 * t) + 0.8 * Math.cos(3.1 * t); }
  function observation(preset) {
    return Array.from({ length: 49 }, (_, i) => {
      const t = i / 4;
      return { t, value: gamma(t + 5, preset !== 'repeat') + noise(t), present: preset !== 'gap' || t < 4 || t > 9 };
    });
  }
  function shiftCost(delta, preset, rows) {
    const available = (rows || observation(preset)).filter(row => row.present);
    return Math.sqrt(available.reduce((sum, row) => sum + (row.value - gamma(row.t + delta, preset !== 'repeat')) ** 2, 0) / available.length);
  }
  function alignment(preset, tau) {
    if (!['repeat', 'distinct', 'gap'].includes(preset) || !(tau > 0)) throw new RangeError('Invalid alignment model');
    const rows = observation(preset);
    const costs = Array.from({ length: 161 }, (_, i) => { const delta = -5 + i / 4; return { delta, cost: shiftCost(delta, preset, rows) }; });
    const best = costs.reduce((a, b) => b.cost < a.cost - 1e-10 ? b : a);
    const weights = costs.map(row => Math.exp(-(row.cost ** 2 - best.cost ** 2) / tau));
    const total = weights.reduce((a, b) => a + b, 0);
    const q = weights.map(w => w / total);
    const mean = costs.reduce((sum, row, i) => sum + row.delta * q[i], 0);
    const lower = costs.reduce((sum, row, i) => sum + (row.delta < 15 ? q[i] : 0), 0);
    return { rows, costs, q, best, mean, lower };
  }
  function risk(p, a, b, estimate) { return p * (estimate - a) ** 2 + (1 - p) * (estimate - b) ** 2; }
  function hedge(p, separation) {
    const a = 0, b = separation, mean = p * a + (1 - p) * b, hard = p >= 0.5 ? a : b;
    return { a, b, mean, hard, meanRisk: risk(p, a, b, mean), hardRisk: risk(p, a, b, hard) };
  }
  function standings(key) {
    const value = item => key === 'gap' ? item.private - item.public : item[key];
    return PORTFOLIO.slice().sort((a, b) => value(a) - value(b) || a.id.localeCompare(b.id));
  }
  function rank(id, key) { return standings(key).findIndex(item => item.id === id) + 1; }
  function install(labs) {
    if (!labs) return;
    const n = field => Number(field.value), line = (ctx, en, ko) => ctx.el('p', {}, ctx.text(en, ko));
    labs.register('rogii-alignment', ctx => {
      const preset = ctx.control('Authored GR evidence', '구성한 GR 근거', { type: 'select', value: 'repeat', options: [
        { value: 'repeat', en: 'Repeated motif', ko: '반복되는 무늬' }, { value: 'distinct', en: 'Add one distinctive peak', ko: '구별되는 피크 하나 추가' }, { value: 'gap', en: 'Hide the distinctive interval', ko: '구별되는 구간을 결측 처리' }
      ] });
      const shift = ctx.control('Candidate vertical shift δ (ft)', '후보 수직 이동 δ (ft)', { type: 'range', min: -5, max: 35, step: 0.25, value: 5 });
      const tau = ctx.control('Weight width τ (GR-unit²)', '가중치 폭 τ (GR 단위²)', { type: 'range', min: 1, max: 100, step: 1, value: 3 });
      const best = ctx.metric('Lowest-cost shift (ft)', '최소 비용의 이동 (ft)'), selected = ctx.metric('Selected mismatch (GR units)', '선택한 이동의 오차 (GR 단위)'), mean = ctx.metric('q-weighted mean shift (ft)', 'q 가중 평균 이동 (ft)');
      function render() {
        const r = alignment(preset.value, n(tau)), cost = shiftCost(n(shift), preset.value, r.rows);
        best.textContent = ctx.format(r.best.delta, 2); selected.textContent = ctx.format(cost, 3); mean.textContent = ctx.format(r.mean, 3);
        const segments = []; let current = [];
        for (const row of r.rows) {
          if (row.present) current.push([row.t, row.value]);
          else if (current.length) { segments.push(current); current = []; }
        }
        if (current.length) segments.push(current);
        const curve = r.rows.map(row => [row.t, gamma(row.t + n(shift), preset.value !== 'repeat')]);
        const ymax = Math.max(...r.costs.map(row => row.cost)) * 1.08;
        ctx.view.replaceChildren(ctx.plot({ x: [0, 12], y: [30, 85], xLabel: ctx.text('Prefix coordinate t (ft)', 'prefix 좌표 t (ft)'), yLabel: ctx.text('Authored GR units', '구성한 GR 단위'), title: ctx.text('Blue: observation · gold: shifted typewell', '파랑: 관측 · 금색: 이동한 typewell'), series: segments.map(points => ({ points, color: 'blue' })).concat([{ points: curve, color: 'gold' }]), bands: preset.value === 'gap' ? [{ x0: 4, x1: 9, color: 'muted' }] : [] }),
          ctx.plot({ x: [-5, 35], y: [0, ymax], xLabel: ctx.text('Candidate shift δ (ft)', '후보 이동 δ (ft)'), yLabel: ctx.text('Prefix mismatch C(δ)', 'prefix 오차 C(δ)'), title: ctx.text('All tested shifts, with the selected shift marked', '검사한 이동 전체와 선택한 이동'), series: [{ points: r.costs.map(row => [row.delta, row.cost]), color: 'blue' }], markers: [{ x: n(shift), y: cost, color: 'gold', label: ctx.text('Selected shift', '선택한 이동') }] }),
          line(ctx, `Blue: authored observation; gold: reference at δ=${ctx.format(n(shift), 2)} ft. Weight below 15 ft: ${ctx.format(100 * r.lower, 1)}%. Data-generating shift: 5 ft.`, `파랑은 구성한 관측, 금색은 δ=${ctx.format(n(shift), 2)} ft에서의 reference입니다. 15 ft 미만의 가중치 합은 ${ctx.format(100 * r.lower, 1)}%입니다. 자료를 만든 실제 이동은 5 ft입니다.`));
        ctx.status.textContent = ctx.text('q ∝ exp(−C²/τ) on a uniform shift grid. Repeated minima are alternative depth modes; their weighted mean need not itself fit the GR curve.', '균일한 이동 격자에서 q ∝ exp(−C²/τ)로 계산합니다. 반복되는 최소점은 다른 깊이 후보이며, 가중 평균 자체가 GR 곡선에 잘 맞는다는 뜻은 아닙니다.');
      }
      ctx.onChange(render);
    });
    labs.register('rogii-hedge', ctx => {
      const p = ctx.control('Probability of lower mode a (%)', '낮은 mode a일 확률 (%)', { type: 'range', min: 0, max: 100, step: 1, value: 50 });
      const distance = ctx.control('Mode separation b − a (ft)', 'mode 사이 거리 b − a (ft)', { type: 'range', min: 0, max: 40, step: 1, value: 20 });
      const rule = ctx.control('Decision rule', '결정 규칙', { type: 'select', value: 'mean', options: [{ value: 'mean', en: 'Probability-weighted mean', ko: '확률 가중 평균' }, { value: 'hard', en: 'Commit to the more likely mode', ko: '더 가능성 높은 mode로 확정' }] });
      const prediction = ctx.metric('Chosen coordinate (ft)', '선택한 좌표 (ft)'), chosenRisk = ctx.metric('Expected squared loss (ft²)', '기대 제곱 오차 (ft²)'), saved = ctx.metric('Mean saves over hard choice (ft²)', '평균이 확정 선택보다 줄이는 오차 (ft²)');
      function render() {
        const probability = n(p) / 100, d = n(distance), r = hedge(probability, d);
        const estimate = rule.value === 'mean' ? r.mean : r.hard;
        const loss = risk(probability, r.a, r.b, estimate);
        prediction.textContent = ctx.format(estimate, 2); chosenRisk.textContent = ctx.format(loss, 2); saved.textContent = ctx.format(r.hardRisk - r.meanRisk, 2);
        const points = Array.from({ length: 101 }, (_, i) => hedge(i / 100, d));
        ctx.view.replaceChildren(ctx.plot({ x: [0, 100], y: [0, Math.max(1, 0.53 * d * d)], xLabel: ctx.text('P(lower mode a), %', '낮은 mode a의 확률 (%)'), yLabel: ctx.text('Expected squared loss (ft²)', '기대 제곱 오차 (ft²)'), title: ctx.text('Mean risk versus hard selection', '평균과 확정 선택의 위험 비교'), series: [{ points: points.map((v, i) => [i, v.meanRisk]), color: 'blue' }, { points: points.map((v, i) => [i, v.hardRisk]), color: 'gold' }], markers: [{ x: n(p), y: loss, label: ctx.text('Selected rule', '선택한 규칙'), color: 'red' }] }),
          line(ctx, 'Blue: probability-weighted mean; gold: hard mode choice; red: selected rule at the chosen probability. Coordinates are relative to a=0.', '파랑은 확률 가중 평균, 금색은 mode 확정 선택, 빨강은 현재 확률에서 선택한 규칙입니다. 좌표는 a=0을 기준으로 합니다.'),
          ctx.table([ctx.text('Possible true mode', '가능한 실제 mode'), ctx.text('Probability', '확률'), ctx.text('Squared error of this prediction', '이 예측의 제곱 오차')], [['a = 0 ft', ctx.format(probability, 2), ctx.format(estimate ** 2, 2) + ' ft²'], [`b = ${d} ft`, ctx.format(1 - probability, 2), ctx.format((estimate - d) ** 2, 2) + ' ft²']]));
        ctx.status.textContent = ctx.text('The probability is assumed here. A GR cost margin is not automatically this probability. Squared loss can prefer a coordinate between two physical modes.', '여기서 확률은 가정한 값입니다. GR 비용 차이가 자동으로 이 확률이 되는 것은 아닙니다. 제곱 오차 기준에서는 두 물리적 mode 사이의 좌표가 최적일 수 있습니다.');
      }
      ctx.onChange(render);
    });
    labs.register('rogii-portfolio', ctx => {
      const order = ctx.control('Order these five candidates by', '후보 다섯 개의 정렬 기준', { type: 'select', value: 'public', options: [{ value: 'public', en: 'Public RMSE', ko: 'Public RMSE' }, { value: 'private', en: 'Private RMSE (hindsight)', ko: 'Private RMSE (사후 정보)' }, { value: 'gap', en: 'Private − Public gap', ko: 'Private − Public 차이' }] });
      const chosen = ctx.control('Inspect candidate', '살펴볼 후보', { type: 'select', value: 'PS3', options: PORTFOLIO.map(item => ({ value: item.id, en: item.id, ko: item.id })) });
      const publicRank = ctx.metric('Public rank among these five', '이 다섯 후보 중 Public 순위'), privateRank = ctx.metric('Private rank among these five', '이 다섯 후보 중 Private 순위'), gap = ctx.metric('Same-submission RMSE gap (ft)', '같은 제출의 RMSE 차이 (ft)');
      function render() {
        const rows = standings(order.value), selected = PORTFOLIO.find(item => item.id === chosen.value);
        publicRank.textContent = `${rank(selected.id, 'public')} / 5`; privateRank.textContent = `${rank(selected.id, 'private')} / 5`; gap.textContent = '+' + (selected.private - selected.public).toFixed(3);
        ctx.view.replaceChildren(ctx.plot({ x: [0.4, 5.6], y: [0, 10], xTicks: [1, 2, 3, 4, 5], xLabel: ctx.text('Position in the selected ordering', '선택한 정렬에서의 위치'), yLabel: 'RMSE (ft)', title: ctx.text('Blue: Public · gold: Private', '파랑: Public · 금색: Private'), bars: rows.flatMap((item, i) => [{ x: i + 1 - 0.16, y: item.public, width: 0.27, color: 'blue', label: `${item.id} Public ${item.public.toFixed(3)}` }, { x: i + 1 + 0.16, y: item.private, width: 0.27, color: 'gold', label: `${item.id} Private ${item.private.toFixed(3)}` }]), markers: [{ x: rows.findIndex(item => item.id === chosen.value) + 1 + 0.16, y: selected.private, color: 'red', label: selected.id }] }),
          line(ctx, `Order: ${rows.map(item => item.id).join(' → ')}. Blue: Public; gold: Private; red marks ${selected.id}.`, `순서: ${rows.map(item => item.id).join(' → ')}. 파랑은 Public, 금색은 Private, 빨강은 ${selected.id}입니다.`),
          ctx.table([ctx.text('Candidate', '후보'), 'Public', 'Private', ctx.text('Gap', '차이')], rows.map(item => [(item.id === chosen.value ? '● ' : '') + item.id, item.public.toFixed(3), item.private.toFixed(3), '+' + (item.private - item.public).toFixed(3)])));
        ctx.status.textContent = ctx.text('Historical scores from the article. PS3 determined the official team result. M5 is best on Private among these five; Private ordering was unavailable during final selection.', '본문의 과거 점수입니다. 공식 팀 결과는 PS3로 결정됐습니다. 이 다섯 후보 중 Private가 가장 좋은 것은 M5이며, 최종 선택 당시에는 Private 순위를 알 수 없었습니다.');
      }
      ctx.onChange(render);
    });
  }
  return { gamma, noise, observation, shiftCost, alignment, risk, hedge, standings, rank, PORTFOLIO, install };
});
