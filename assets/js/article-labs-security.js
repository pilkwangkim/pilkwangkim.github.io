/* Bounded arithmetic and frozen article records. No model, tool, or network calls. */
(function () {
  'use strict';
  const rawScore = (n, k, cells, weight = 16) => n * k * weight + 2 * Math.min(n, cells);
  const density = (k, fixed, marginal) => (16 * k + 2) / (fixed + k * marginal);
  function fillRows(budget, gpt, gemma, fixed, cap = 2000) {
    return [gpt, gemma].map(t => Math.max(0, Math.min(cap, fixed || Infinity, Math.floor(budget / t))));
  }
  function deadline(n, cost, budget, partial, exception = false) {
    const completed = Math.min(n, Math.floor(budget / cost));
    return { completed, valid: !exception && (partial || completed === n), credited: exception || (!partial && completed < n) ? 0 : completed };
  }
  function ordered(prefix, tail, prefixTime, tailTime, budget, prefixValue = 18, tailValue = 6) {
    const first = Math.min(prefix, Math.floor(budget / prefixTime));
    const second = first === prefix ? Math.min(tail, Math.floor(Math.max(0, budget - first * prefixTime) / tailTime)) : 0;
    return { first, second, time: first * prefixTime + second * tailTime, raw: first * prefixValue + second * tailValue };
  }
  function stopCount(coef, budget = 9000, fraction = .992, warmup = 60, cost = 1, cushion = 1.2) {
    const wall = (fraction * budget - warmup - cushion) / cost;
    const replay = (fraction * budget - warmup - coef * cushion) / (coef * cost);
    return { count: Math.max(0, Math.floor(Math.min(wall, replay))), bound: wall <= replay ? 'wall' : 'replay' };
  }
  const pairs = [
    ['URL18 / Static42', 137.370, 0], ['Letter-K8 / Static42', 134.170, 0],
    ['Prime4', 124, 0], ['PICK2B', 122.625, 0], ['ExactUnion', 119.710, 0],
    ['Both-no-empty', 117.030, 0], ['Proper HTTPS', 63.765, 0],
    ['PICK2BE', 107.690, .960], ['Pure Deputy Mail', 25.695, 26.010], ['Tail-bearing anchor', 137.850, 21.340]
  ];
  function pooledRate(blocks) { return blocks.reduce((s, b) => s + b[0], 0) / blocks.reduce((s, b) => s + b[1], 0); }
  const api = { rawScore, density, fillRows, deadline, ordered, stopCount, pairs, pooledRate };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const reg = (id, fn) => window.ArticleLabs.register(id, fn);
  const val = input => Number(input.value);
  const clear = ctx => ctx.view.replaceChildren();
  function score(ctx) {
    const n = ctx.control('Successful findings', '성공한 finding 수', { type: 'range', min: 0, max: 2000, step: 10, value: 400 });
    const k = ctx.control('Events per finding, K', 'finding당 event 수 K', { type: 'range', min: 1, max: 8, step: 1, value: 1 });
    const unique = ctx.control('Findings with a new cell (%)', '새 cell을 만드는 finding (%)', { type: 'range', min: 0, max: 100, step: 5, value: 100 });
    const out = ctx.metric('One model row', '한 model row');
    ctx.onChange(() => {
      const cells = Math.floor(val(n) * val(unique) / 100), raw = rawScore(val(n), val(k), cells);
      out.textContent = ctx.format(Math.min(1000, raw / 200), 3);
      clear(ctx);
      ctx.view.append(ctx.plot({ x: [0, 2000], y: [0, Math.max(200, rawScore(2000, val(k), 2000) / 200)], xLabel: ctx.text('Successful findings', '성공한 finding 수'), yLabel: ctx.text('One-row score', '한 row 점수'), series: [{ color: 'blue', label: `K=${val(k)}`, points: Array.from({ length: 21 }, (_, i) => [i * 100, Math.min(1000, rawScore(i * 100, val(k), Math.floor(i * 100 * val(unique) / 100)) / 200)]) }], markers: [{ x: val(n), y: Math.min(1000, raw / 200), label: ctx.format(Math.min(1000, raw / 200), 2), color: 'gold' }] }));
      ctx.view.append(ctx.table([ctx.text('Severity mass', 'severity 합'), ctx.text('Cell bonus', 'cell 보너스'), 'Raw'], [[16 * val(n) * val(k), 2 * cells, raw]]));
      ctx.status.textContent = ctx.text('A finding contributes one cell bonus, even when it contains several scored events. The Public score averages two model rows.', '한 finding에 여러 event가 있어도 cell 보너스는 한 번입니다. Public 점수는 두 model row의 평균입니다.');
    });
  }
  reg('security-score', score);
  reg('security-linear', ctx => {
    const n = ctx.control('Successful, distinct K1 findings', '성공하고 cell이 다른 K1 finding 수', { type: 'range', min: 0, max: 800, step: 10, value: 400 });
    const out = ctx.metric('One-row score', '한 row 점수');
    ctx.onChange(() => { out.textContent = ctx.format(.09 * val(n), 3); clear(ctx); ctx.view.append(ctx.plot({ x: [0, 800], y: [0, 72], xLabel: ctx.text('Effective successful findings', '실제로 성공한 finding 수'), yLabel: ctx.text('One-row score', '한 row 점수'), series: [{ color: 'blue', points: [[0, 0], [800, 72]], label: '0.09 × N' }], markers: [[360, 32.4], [400, 36], [450, 40.5]].map(p => ({ x: p[0], y: p[1], label: String(p[1]), color: 'gold' })).concat([{ x: val(n), y: .09 * val(n), label: ctx.format(.09 * val(n), 2), color: 'red' }]) })); ctx.status.textContent = ctx.text('Part 2 historical single-post observations: 360→32.4, 400→36.0, 450→40.5. Each effective finding earns 16+2 raw. Returned count is not effective count, and this identity does not establish a fixed replay ceiling.', 'Part 2의 당시 single-post 관측은 360→32.4, 400→36.0, 450→40.5입니다. 실제로 성공한 finding당 16+2 raw를 얻습니다. 반환 수와 실제 성공 수는 다르며 이 항등식으로 replay 상한이 고정되지는 않습니다.'); });
  });
  reg('security-fill', ctx => {
    const gpt = ctx.control('GPT seconds per finding', 'GPT finding당 초', { type: 'range', min: 8, max: 40, step: .5, value: 24 });
    const gem = ctx.control('Gemma seconds per finding', 'Gemma finding당 초', { type: 'range', min: 8, max: 40, step: .5, value: 12 });
    const policy = ctx.control('Bank policy', 'bank 구성', { type: 'select', value: 'fill', options: [{ value: 'fixed', en: 'Fixed bank: 357', ko: '고정 bank: 357' }, { value: 'fill', en: 'Fill each model independently', ko: '각 model을 독립적으로 채우기' }] });
    const out = ctx.metric('Illustrative Public mean', '계산한 Public 평균');
    ctx.onChange(() => { const rows = fillRows(9000, val(gpt), val(gem), policy.value === 'fixed' ? 357 : 0); out.textContent = ctx.format(.045 * (rows[0] + rows[1]), 3); clear(ctx); ctx.view.append(ctx.plot({ x: [0, 2], y: [0, 2000], xLabel: '0: GPT · 1: Gemma', yLabel: ctx.text('Successful findings', '성공한 finding 수'), bars: rows.map((n, i) => ({ x: i + .3, y: n, width: .4, color: i ? 'gold' : 'blue', label: String(n) })) })); ctx.status.textContent = ctx.text('v3.1.2 planning model: 9,000 seconds per model; every admitted finding succeeds with a distinct cell. These adjustable times are assumptions, not recovered hosted row timings.', 'v3.1.2 계산 모형: model당 9,000초이며 모든 finding이 성공하고 별도 cell을 만듭니다. 조절하는 시간은 가정이며 hosted row 시간을 복원한 값이 아닙니다.'); });
  });
  reg('security-density', ctx => {
    const fixed = ctx.control('Fixed cost F (seconds)', '고정 비용 F (초)', { type: 'range', min: 0, max: 1, step: .01, value: .06 });
    const marginal = ctx.control('Cost per event g (seconds)', 'event당 비용 g (초)', { type: 'range', min: .1, max: 2, step: .01, value: .75 });
    const k = ctx.control('Events per finding K', 'finding당 event 수 K', { type: 'range', min: 1, max: 8, step: 1, value: 3 });
    const out = ctx.metric('Density relative to K1', 'K1 대비 raw/초');
    ctx.onChange(() => { const ratio = density(val(k), val(fixed), val(marginal)) / density(1, val(fixed), val(marginal)); out.textContent = ctx.format(ratio, 3) + '×'; clear(ctx); ctx.view.append(ctx.plot({ x: [1, 8], y: [0, Math.max(...Array.from({ length: 8 }, (_, i) => density(i + 1, val(fixed), val(marginal)))) * 1.1], xLabel: 'K', yLabel: ctx.text('Raw / second', 'raw / 초'), series: [{ color: 'blue', points: Array.from({ length: 8 }, (_, i) => [i + 1, density(i + 1, val(fixed), val(marginal))]) }] })); ctx.status.textContent = ctx.text('R(K)=16K+2 and T(K)=F+Kg. The initial F=0.06 s, g=0.75 s are the Part 4 local measurement; exact execution of K events is assumed.', 'R(K)=16K+2, T(K)=F+Kg입니다. 초기 F=0.06초, g=0.75초는 Part 4의 로컬 측정값이며 K개 event의 정확한 실행을 가정합니다.'); });
  });
  reg('security-cost', ctx => {
    const speed = ctx.control('Speedup of selected component', '선택한 구간의 속도 향상', { type: 'range', min: 1, max: 10, step: .5, value: 2 });
    const part = ctx.control('Component', '구간', { type: 'select', value: '2', options: [{ value: '0', en: 'Environment build', ko: '환경 생성' }, { value: '1', en: 'Reset', ko: '초기화' }, { value: '2', en: 'Interaction', ko: 'interaction' }] });
    const out = ctx.metric('Whole-candidate speedup', '전체 candidate 속도 향상');
    ctx.onChange(() => { const times = [.047, .014, .976], revised = times.map((t, i) => t / (i === val(part) ? val(speed) : 1)), total = revised.reduce((a, b) => a + b, 0); out.textContent = ctx.format(1.037 / total, 3) + '×'; clear(ctx); ctx.view.append(ctx.table([ctx.text('Seconds', '초'), ctx.text('Build', '생성'), ctx.text('Reset', '초기화'), 'Interaction', ctx.text('Total', '합계')], [['Original', ...times, '1.037'], ['Changed', ...revised.map(t => ctx.format(t, 3)), ctx.format(total, 3)]])); ctx.status.textContent = ctx.text('Part 5 local stage timings; the article rounds their sum to 1.04 s. Only the selected cost changes in this arithmetic.', 'Part 5의 로컬 구간별 시간이며 글에서는 합을 1.04초로 반올림합니다. 이 계산에서는 선택한 구간의 비용만 바꿉니다.'); });
  });
  reg('security-stop', ctx => {
    const coef = ctx.control('Replay-cost coefficient', 'replay 비용 계수', { type: 'range', min: .8, max: 1.2, step: .01, value: .95 });
    const out = ctx.metric('Returned candidates', '반환 candidate 수');
    ctx.onChange(() => { const result = stopCount(val(coef)); out.textContent = String(result.count); clear(ctx); ctx.view.append(ctx.plot({ x: [.8, 1.2], y: [7000, 9000], xLabel: 'COEF', yLabel: 'N', series: [{ color: 'blue', points: Array.from({ length: 41 }, (_, i) => { const c = .8 + .01 * i; return [c, stopCount(c).count]; }) }], markers: [{ x: val(coef), y: result.count, color: 'gold', label: ctx.text(result.bound + ' bound', result.bound === 'wall' ? 'wall 조건' : 'replay 조건') }] })); ctx.status.textContent = ctx.text('Part 6 control-flow model: B=9,000 s, fraction=.992, warmup=60 s, candidate=1 s, cushion=1.2 s. Below COEF=1 the wall condition binds first; these times illustrate the proof.', 'Part 6의 제어 흐름 모형: B=9,000초, fraction=.992, warmup=60초, candidate=1초, 여유=1.2초입니다. COEF<1에서는 wall 조건이 먼저 작동합니다. 시간은 증명을 보여 주기 위한 가정입니다.'); });
  });
  reg('security-deadline', ctx => {
    const mode = ctx.control('Evaluator rule', 'evaluator 규칙', { type: 'select', value: 'partial', options: [{ value: 'old', en: 'Before Aug 5: overrun invalidates', ko: '8월 5일 이전: 초과하면 무효' }, { value: 'partial', en: 'Aug 7 source: inner prefix banking', ko: '8월 7일 소스: inner 완료 prefix 보존' }, { value: 'exception', en: 'Candidate exception', ko: 'candidate 예외' }] });
    const n = ctx.control('Returned findings', '반환 finding 수', { type: 'range', min: 500, max: 2000, step: 50, value: 1500 });
    const cost = ctx.control('Seconds per finding', 'finding당 초', { type: 'range', min: 3, max: 12, step: .25, value: 7.5 });
    const out = ctx.metric('Credited one-row score', '반영된 한 row 점수');
    ctx.onChange(() => { const budget = mode.value === 'old' ? 9000 : 8750, r = deadline(val(n), val(cost), budget, mode.value !== 'old', mode.value === 'exception'); out.textContent = r.valid ? ctx.format(.09 * r.credited, 3) : ctx.text('Invalid', '무효'); clear(ctx); ctx.view.append(ctx.table([ctx.text('Phase budget', 'phase 예산'), ctx.text('Completed prefix', '완료 prefix'), ctx.text('Returned', '반환 수')], [[budget + ' s', r.completed, val(n)]])); ctx.status.textContent = ctx.text('One-row accounting with 18 raw per finding. The August source preserves an inner-deadline prefix; the outer RPC breaker and candidate exceptions can still invalidate work. No outer-breaker timing is simulated.', 'finding당 18 raw인 한 row 계산입니다. 8월 소스는 inner deadline의 완료 prefix를 보존하지만 outer RPC 제한과 candidate 예외는 여전히 무효를 만들 수 있습니다. outer 제한의 시간 동작은 계산하지 않습니다.'); });
  });
  reg('security-frontier', ctx => {
    const rows = [['K1', 1, 1, '8/8', 28], ['Exactly K2', 2, 2, '20/20', 52], ['Exactly K3', 3, 3, '20/20', 76], ['Exactly K4', 4, 2, '0/10', 71], ['Always emit', 4, 4, '10/10', 115], ['Exactly K8', 8, 2, '0/8', 72.5]];
    const pick = ctx.control('Recorded local family', '기록된 로컬 family', { type: 'select', value: '0', options: rows.map((r, i) => ({ value: String(i), en: r[0], ko: r[0] })) });
    const out = ctx.metric('Raw per decoded token', 'decode token당 raw');
    ctx.onChange(() => { const r = rows[val(pick)], raw = 16 * r[2] + 2; out.textContent = ctx.format(raw / r[4], 3); clear(ctx); ctx.view.append(ctx.table([ctx.text('Requested K', '요청 K'), ctx.text('Observed posts', '관측 post'), ctx.text('Exact count', '정확한 횟수'), 'Raw', ctx.text('Decode tokens', 'decode token')], [[r[1], r[2], r[3], raw, r[4]]])); ctx.status.textContent = ctx.text('Six frozen Gemma-local rows from Part 9. Token efficiency is not end-to-end replay density. The later hosted K3 result was nearly flat, so these rows cannot predict its score.', 'Part 9의 고정된 Gemma 로컬 6개 결과입니다. token 효율은 end-to-end replay 밀도와 다릅니다. 이후 hosted K3 결과는 거의 같았으므로 이 표로 hosted 점수를 예측할 수 없습니다.'); });
  });
  function prefix(ctx, historical) {
    const leg = ctx.control(historical ? 'Intended selected route' : 'Illustrative ordered bank', historical ? '의도한 selected route' : '가정한 순서형 bank', { type: 'select', value: 'gpt', options: historical ? [{ value: 'gpt', en: 'GPT: 256 prefix + 1,744 tail', ko: 'GPT: prefix 256 + tail 1,744' }, { value: 'gem', en: 'Gemma: 960 prefix + 1,040 tail', ko: 'Gemma: prefix 960 + tail 1,040' }] : [{ value: 'gpt', en: '500 prefix + 1,500 tail', ko: 'prefix 500 + tail 1,500' }, { value: 'gem', en: '1,000 prefix + 1,000 tail', ko: 'prefix 1,000 + tail 1,000' }] });
    const prefixTime = ctx.control('Prefix seconds per candidate', 'prefix candidate당 초', { type: 'range', min: .25, max: 35, step: .25, value: 34 });
    const tailTime = ctx.control('Tail seconds per candidate', 'tail candidate당 초', { type: 'range', min: 1, max: 12, step: .25, value: 7.5 });
    const out = ctx.metric('Tail candidates reached', '도달한 tail candidate 수');
    ctx.onChange(() => { const gpt = leg.value === 'gpt', a = historical ? (gpt ? 256 : 960) : (gpt ? 500 : 1000), b = 2000 - a, r = ordered(a, b, val(prefixTime), val(tailTime), 8750); out.textContent = String(r.second); clear(ctx); ctx.view.append(ctx.plot({ x: [0, 2000], y: [0, 8750], xLabel: ctx.text('Fixed candidate position', '고정된 candidate 순서'), yLabel: ctx.text('Elapsed replay seconds', '누적 replay 초'), bands: [{ x0: 0, x1: a, y0: 0, y1: 8750, color: 'blue' }, { x0: a, x1: 2000, y0: 0, y1: 8750, color: 'gold' }], series: [{ color: 'blue', points: [[0, 0], [r.first, r.first * val(prefixTime)], [r.first + r.second, r.time]] }], markers: [{ x: r.first + r.second, y: r.time, color: 'red', label: ctx.text('Reached', '도달 위치') }] })); ctx.status.textContent = historical ? ctx.text('Part 11 intended route counts; latencies are editable assumptions. Faster rejection can expose a fixed tail. Hosted route identity and private row traces were not logged; this does not reconstruct their execution.', 'Part 11의 의도한 route별 개수이며 시간은 조절 가능한 가정입니다. prefix가 빠르게 거절되면 고정 tail에 도달할 수 있습니다. hosted route와 private row trace가 기록되지 않아 당시 실행을 복원한 것은 아닙니다.') : ctx.text('Illustrative 2,000-candidate bank and assumed latencies, under an 8,750-second replay budget. Only the completed prefix is reached; reducing prefix cost can expose the tail without changing the messages or their order.', '가정한 candidate 2,000개 bank와 시간에 8,750초 replay 예산을 적용합니다. 완료한 prefix까지만 도달하며 메시지나 순서를 바꾸지 않아도 prefix 비용이 줄면 tail에 도달할 수 있습니다.'); });
  }
  reg('security-prefix', ctx => prefix(ctx, true));
  reg('security-order', ctx => prefix(ctx, false));
  function portfolio(ctx, paired) {
    const a = ctx.control('First submission', '첫 번째 제출', { type: 'select', value: '9', options: pairs.map((p, i) => ({ value: String(i), en: p[0], ko: p[0] })) });
    const b = ctx.control('Second submission', '두 번째 제출', { type: 'select', value: paired ? '8' : '5', options: pairs.map((p, i) => ({ value: String(i), en: p[0], ko: p[0] })) });
    const out = ctx.metric('Observed best private score', '관측된 최고 private 점수');
    ctx.onChange(() => { const chosen = [pairs[val(a)], pairs[val(b)]]; out.textContent = ctx.format(Math.max(...chosen.map(p => p[2])), 3); clear(ctx); ctx.view.append(ctx.plot({ x: [0, 1], y: [0, 145], xLabel: ctx.text('0: Public · 1: Private', '0: Public · 1: Private'), yLabel: ctx.text('Same-submission score', '동일 제출 점수'), series: chosen.map((p, i) => ({ label: p[0], color: i ? 'gold' : 'blue', points: [[0, p[1]], [1, p[2]]] })) })); ctx.status.textContent = ctx.text('Exact-reference results disclosed in Part 11. The better of two private scores counted. This is an ex-post comparison of banked options, not a probability model of the unknown defense.', 'Part 11에 공개한 동일 reference의 결과입니다. 두 private 점수 중 높은 값이 반영되었습니다. 이미 제출한 후보의 사후 비교이며 미지의 방어 정책에 대한 확률 모형이 아닙니다.'); });
  }
  reg('security-pairs', ctx => portfolio(ctx, true));
  reg('security-portfolio', ctx => portfolio(ctx, false));
  reg('security-scenarios', ctx => {
    const main = ctx.control('Main family survival (%)', '주력 family 생존 비율 (%)', { type: 'range', min: 0, max: 100, step: 5, value: 20 });
    const alternative = ctx.control('Alternative family survival (%)', '다른 family 생존 비율 (%)', { type: 'range', min: 0, max: 100, step: 5, value: 100 });
    const cost = ctx.control('Held-out replay / public cost', 'held-out replay / public 비용', { type: 'range', min: 1, max: 1.5, step: .01, value: 1.1 });
    const fraction = ctx.control('Replay budget fraction filled', '채운 replay 예산 비율', { type: 'range', min: .5, max: 1, step: .01, value: .8 });
    const out = ctx.metric('Assumed best-of-two value', '가정한 best-of-two 값');
    ctx.onChange(() => { const fits = val(fraction) * val(cost) <= 1, a = fits ? 108 * val(main) / 100 : 0, b = fits ? 24.8 * val(alternative) / 100 : 0; out.textContent = ctx.format(Math.max(a, b), 3); clear(ctx); ctx.view.append(ctx.table([ctx.text('Family', 'family'), ctx.text('Recorded public baseline', '기록된 public 기준'), ctx.text('Assumed scenario value', '가정한 시나리오 값')], [[ctx.text('Single-post main', 'single-post 주력'), 108, ctx.format(a, 3)], [ctx.text('Deputy-only alternative', 'Deputy-only 대안'), 24.8, ctx.format(b, 3)]])); ctx.status.textContent = ctx.text(`Part 7's scenario reasoning, using its public 108.0 and 24.8 scores. Survival percentages are invented inputs, not inferred hidden-policy probabilities. Both banks share the illustrative timing condition f × r ≤ 1: ${fits ? 'fits' : 'overrun'}. Other generation phases are not certified by this check.`, `Part 7의 시나리오 논리에 당시 public 108.0, 24.8을 사용합니다. 생존 비율은 입력한 가정이며 hidden-policy 확률 추정이 아닙니다. 두 bank에 예제 시간 조건 f × r ≤ 1을 적용하면 ${fits ? '예산 안' : '예산 초과'}입니다. 이 검사로 다른 generation phase까지 보장하지는 않습니다.`); });
  });
  reg('security-accounting', ctx => {
    const first = ctx.control('First-stage saving (seconds)', '첫 구간 절감 (초)', { type: 'range', min: 0, max: 20, step: .1, value: 5.8 });
    const close = ctx.control('Closing-stage increase (seconds)', '종료 구간 증가 (초)', { type: 'range', min: 0, max: 20, step: .001, value: 11.942 });
    const out = ctx.metric('Total change (seconds)', '전체 시간 변화 (초)');
    ctx.onChange(() => { const changed = 303.207 - val(first) + val(close) + .121; out.textContent = (changed >= 303.207 ? '+' : '') + ctx.format(changed - 303.207, 3); clear(ctx); ctx.view.append(ctx.table([ctx.text('Run', '실행'), ctx.text('First', '첫 구간'), ctx.text('Closing', '종료 구간'), ctx.text('Residual', '나머지'), ctx.text('Total', '합계')], [['Control', 176.879, 121.671, 4.657, 303.207], ['Variant', ctx.format(176.879 - val(first), 3), ctx.format(121.671 + val(close), 3), 4.778, ctx.format(changed, 3)]])); ctx.status.textContent = ctx.text('The initial values reproduce the fifth-place write-up’s early comparison: 200 findings and 200 cells per run, +6.263 s overall. Moving a control changes hypothetical stage costs only; residual time is not tool execution alone.', '초기값은 5위 write-up의 초기 비교를 재현합니다. 실행마다 finding 200개와 cell 200개를 유지했으며 전체는 6.263초 느려졌습니다. 조절값은 구간별 비용을 가정해 바꾸며 residual은 tool 실행 시간만을 뜻하지 않습니다.'); });
  });
})();
