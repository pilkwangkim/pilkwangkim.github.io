/* Small teaching datasets; no raw OOF predictions or competition execution. */
(function () {
  'use strict';
  const toy = [{ p: .08, y: 0 }, { p: .15, y: 0 }, { p: .22, y: 1 }, { p: .35, y: 0 }, { p: .42, y: 0 }, { p: .48, y: 1 }, { p: .52, y: 0 }, { p: .58, y: 1 }, { p: .65, y: 0 }, { p: .72, y: 1 }, { p: .85, y: 1 }, { p: .92, y: 1 }];
  function confusion(rows, threshold) {
    const r = { tp: 0, fp: 0, fn: 0, tn: 0 };
    for (const { p, y } of rows) r[p >= threshold ? (y ? 'tp' : 'fp') : (y ? 'fn' : 'tn')]++;
    return { ...r, precision: r.tp + r.fp ? r.tp / (r.tp + r.fp) : null, recall: r.tp + r.fn ? r.tp / (r.tp + r.fn) : null, f1: 2 * r.tp + r.fp + r.fn ? 2 * r.tp / (2 * r.tp + r.fp + r.fn) : 0 };
  }
  function auc(rows) {
    const pos = rows.filter(r => r.y === 1), neg = rows.filter(r => r.y === 0);
    return pos.reduce((sum, a) => sum + neg.reduce((s, b) => s + (a.p > b.p ? 1 : a.p === b.p ? .5 : 0), 0), 0) / (pos.length * neg.length);
  }
  function targetEncoding(support, rate, alpha, prior) { return (support * rate + alpha * prior) / (support + alpha); }
  const fluxRatio = color => 10 ** (-.4 * color);
  const medians = { 'g-r': [1.481, .214, .453], 'u-r': [3.061, .586, 1.638], 'u-z': [4.288, .906, 1.875], 'g-i': [2.342, .350, .607] };
  const classNames = ['GALAXY', 'QSO', 'STAR'], selectedMultipliers = [.770524, 1.02, 1.275];
  function classDecision(weights, multipliers) {
    const sum = weights.reduce((a, b) => a + b, 0);
    const probabilities = sum ? weights.map(p => p / sum) : weights.map(() => 1 / weights.length);
    const scores = probabilities.map((p, i) => p * multipliers[i]);
    const argmax = values => values.indexOf(Math.max(...values));
    return { probabilities, scores, before: argmax(probabilities), after: argmax(scores) };
  }
  const alignedCutoff = (men, women) => Math.min(men, women);
  const inSnapshot = (season, day, cutoff) => season < 2026 || (season === 2026 && day <= cutoff);
  const seasonRole = (season, validation) => season < validation ? 'train' : season === validation ? 'validation' : 'excluded';
  const api = { toy, confusion, auc, targetEncoding, fluxRatio, medians, selectedMultipliers, classDecision, alignedCutoff, inSnapshot, seasonRole };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const reg = (id, f) => window.ArticleLabs.register(id, f), v = input => Number(input.value);
  reg('predictive-threshold', ctx => {
    const threshold = ctx.control('Positive decision threshold', '양성 판정 threshold', { type: 'range', min: 0, max: 1, step: .01, value: .5 });
    const out = ctx.metric('Toy precision / recall / F1', '예제 precision / recall / F1'), ranking = ctx.metric('Toy ROC-AUC, unchanged by threshold', 'threshold와 무관한 예제 ROC-AUC');
    ctx.onChange(() => { const r = confusion(toy, v(threshold)); out.textContent = [r.precision, r.recall, r.f1].map(x => x === null ? '—' : ctx.format(x, 3)).join(' / '); ranking.textContent = ctx.format(auc(toy), 4); ctx.view.replaceChildren(); ctx.view.append(ctx.plot({ x: [0, 1], y: [-.15, 1.15], xLabel: ctx.text('Fixed synthetic probability', '고정된 가상 확률'), yLabel: ctx.text('Actual label y', '실제 label y'), series: [{ color: 'muted', dash: true, points: [[v(threshold), -.15], [v(threshold), 1.15]], label: 'threshold' }], markers: toy.map((r, i) => ({ x: r.p, y: r.y, color: r.y ? 'gold' : 'blue', label: `#${i + 1}: p=${r.p}, y=${r.y}, predicted=${Number(r.p >= v(threshold))}` })) })); ctx.view.append(ctx.table([ctx.text('Actual / predicted', '실제 / 예측'), ctx.text('Positive', '양성'), ctx.text('Negative', '음성')], [[ctx.text('Positive', '양성'), r.tp, r.fn], [ctx.text('Negative', '음성'), r.fp, r.tn]])); ctx.status.textContent = ctx.text('Twelve invented examples: six positives and six negatives. Decisions use p ≥ threshold. Moving the threshold changes FP/FN, while the fixed ranking leaves ROC-AUC unchanged. These are not the article’s OOF probabilities or competition metrics.', '가상 예제 12개이며 양성 6개, 음성 6개입니다. p ≥ threshold로 판정합니다. threshold를 바꾸면 FP/FN이 달라지지만 고정된 순위의 ROC-AUC는 그대로입니다. 글의 OOF 확률이나 대회 성능 수치가 아닙니다.'); });
  });
  reg('predictive-target-encoding', ctx => {
    const support = ctx.control('Training-side weighted support', '학습 측 가중 support', { type: 'range', min: 1, max: 500, step: 1, value: 10 });
    const rate = ctx.control('Training category rate', '학습 category 비율', { type: 'range', min: 0, max: 1, step: .01, value: .6 });
    const alpha = ctx.control('Smoothing α', 'smoothing α', { type: 'range', min: 0, max: 300, step: 1, value: 80 });
    const label = ctx.control('Held-out row label (for the leaky comparison)', '제외한 row의 label (누출 비교용)', { type: 'select', value: '1', options: [{ value: '0', en: '0', ko: '0' }, { value: '1', en: '1', ko: '1' }] });
    const out = ctx.metric('Fold-safe TE / label-leaking TE', 'fold-safe TE / label이 누출된 TE');
    ctx.onChange(() => { const n = v(support), p = v(rate), a = v(alpha), prior = .2, safe = targetEncoding(n, p, a, prior), leaky = targetEncoding(n + 1, (n * p + v(label)) / (n + 1), a, prior); out.textContent = `${ctx.format(safe, 4)} / ${ctx.format(leaky, 4)}`; ctx.view.replaceChildren(); ctx.view.append(ctx.plot({ x: [0, 300], y: [0, 1], xLabel: 'α', yLabel: ctx.text('Smoothed target prior', '평활화한 target prior'), series: [{ color: 'blue', label: ctx.text('Fold-safe', 'fold-safe'), points: Array.from({ length: 31 }, (_, i) => [i * 10, targetEncoding(n, p, i * 10, prior)]) }, { color: 'muted', dash: true, label: ctx.text('Global prior 0.20', '전체 prior 0.20'), points: [[0, prior], [300, prior]] }], markers: [{ x: a, y: safe, color: 'gold', label: ctx.format(safe, 4) }] })); ctx.status.textContent = ctx.text('Invented weighted category statistics and a fixed training prior of 0.20. TE=(Σwy+αȳ)/(Σw+α). The fold-safe result ignores the held-out label; the leaky comparison adds that row with weight one. The global prior is also fixed on the training side. No AUC improvement is calculated.', '가상 가중 category 통계와 학습 측 prior 0.20을 사용합니다. TE=(Σwy+αȳ)/(Σw+α)입니다. fold-safe 값은 제외한 label과 무관하며 누출 비교는 그 row를 가중치 1로 더합니다. 전체 prior도 학습 측에서 고정했습니다. AUC 개선을 계산하지 않습니다.'); });
  });
  reg('predictive-colors', ctx => {
    const pair = ctx.control('Band pair', 'band 쌍', { type: 'select', value: 'g-r', options: Object.keys(medians).map(k => ({ value: k, en: k, ko: k })) });
    const cls = ctx.control('Recorded class-median preset', '기록된 class median 초기값', { type: 'select', value: '0', options: classNames.map((c, i) => ({ value: String(i), en: c, ko: c })) });
    const color = ctx.control('Color mₐ − mᵦ', 'color mₐ − mᵦ', { type: 'range', min: -2, max: 6, step: .001, value: 1.481 });
    const shift = ctx.control('Common magnitude shift', '공통 magnitude 이동', { type: 'range', min: -5, max: 5, step: .1, value: 0 });
    pair.addEventListener('change', () => { color.value = String(medians[pair.value][v(cls)]); });
    cls.addEventListener('change', () => { color.value = String(medians[pair.value][v(cls)]); });
    const out = ctx.metric('Flux ratio Fₐ / Fᵦ', 'flux 비율 Fₐ / Fᵦ');
    ctx.onChange(() => { const c = v(color), mb = 18 + v(shift), ma = mb + c; out.textContent = ctx.format(fluxRatio(c), 5); ctx.view.replaceChildren(); ctx.view.append(ctx.plot({ x: [-2, 6], y: [0, 6.4], xLabel: ctx.text('Color mₐ − mᵦ', 'color mₐ − mᵦ'), yLabel: 'Fₐ / Fᵦ', series: [{ color: 'blue', points: Array.from({ length: 81 }, (_, i) => { const x = -2 + i / 10; return [x, fluxRatio(x)]; }) }], markers: [{ x: c, y: fluxRatio(c), color: 'gold', label: `${pair.value}: ${c}` }] })); ctx.view.append(ctx.table(['mₐ', 'mᵦ', 'mₐ − mᵦ', 'Fₐ / Fᵦ'], [[ctx.format(ma, 3), ctx.format(mb, 3), ctx.format(c, 3), ctx.format(fluxRatio(c), 5)]])); ctx.status.textContent = ctx.text('The class presets are the article’s measured median colors. The absolute magnitudes use an invented mᵦ=18 baseline and a common flux reference. Adding the same magnitude shift changes brightness, while color and flux ratio stay fixed. A class median is not a classification rule.', 'class 초기값은 글에서 측정한 median color입니다. 절대 magnitude는 가정한 mᵦ=18과 공통 flux 기준을 사용합니다. 같은 magnitude를 더하면 밝기는 달라져도 color와 flux 비율은 그대로입니다. class median 자체가 분류 규칙은 아닙니다.'); });
  });
  reg('predictive-multipliers', ctx => {
    const probabilities = [48, 12, 40].map((x, i) => ctx.control(`${classNames[i]} probability weight`, `${classNames[i]} 확률 비중`, { type: 'range', min: 0, max: 100, step: 1, value: x }));
    const apply = ctx.control('Apply the recorded class multipliers', '기록된 class multiplier 적용', { type: 'checkbox', value: true });
    const out = ctx.metric('Argmax before → after', 'argmax 적용 전 → 후');
    ctx.onChange(() => { const weights = probabilities.map(v), factors = apply.checked ? selectedMultipliers : [1, 1, 1], r = classDecision(weights, factors); out.textContent = `${classNames[r.before]} → ${classNames[r.after]}`; ctx.view.replaceChildren(); ctx.view.append(ctx.plot({ x: [0, 3], y: [0, 1.3], xLabel: '0: GALAXY · 1: QSO · 2: STAR', yLabel: ctx.text('Decision score pₖ × aₖ', '판정 점수 pₖ × aₖ'), bars: r.scores.map((s, i) => ({ x: i + .3, y: s, width: .4, color: i === r.after ? 'gold' : 'blue', label: classNames[i] + ': ' + ctx.format(s, 4) })) })); ctx.view.append(ctx.table([ctx.text('Class', 'class'), 'pₖ', 'aₖ', 'pₖ × aₖ'], classNames.map((c, i) => [c, ctx.format(r.probabilities[i], 4), factors[i], ctx.format(r.scores[i], 4)]))); ctx.status.textContent = ctx.text('Invented probability weights are normalized; all-zero weights use a uniform vector. Multipliers 0.770524, 1.020000, 1.275000 are the article’s selected values. Weighted scores choose the hard label; they are not calibrated probabilities. Without OOF labels this example computes no balanced-accuracy gain.', '가상 확률 비중을 정규화하며 모두 0이면 균등 벡터를 사용합니다. multiplier 0.770524, 1.020000, 1.275000은 글에서 선택한 값입니다. 가중 점수는 hard label을 고르며 보정된 확률이 아닙니다. OOF label이 없으므로 balanced accuracy 개선을 계산하지 않습니다.'); });
  });
  reg('predictive-cutoff', ctx => {
    const men = ctx.control('Men: latest available 2026 day', '남자: 사용 가능한 2026년 마지막 day', { type: 'range', min: 70, max: 130, step: 1, value: 93 });
    const women = ctx.control('Women: latest available 2026 day', '여자: 사용 가능한 2026년 마지막 day', { type: 'range', min: 70, max: 130, step: 1, value: 97 });
    const asof = ctx.control('Prediction as-of day', '예측 시점의 day', { type: 'range', min: 70, max: 130, step: 1, value: 93 });
    const validation = ctx.control('Rolling validation season', 'rolling 검증 season', { type: 'select', value: '2025', options: [2021, 2022, 2023, 2024, 2025].map(y => ({ value: String(y), en: String(y), ko: String(y) })) });
    const out = ctx.metric('Aligned availability / as-of-safe cutoff', '맞춘 가용 day / 예측 시점까지의 cutoff');
    ctx.onChange(() => { const aligned = alignedCutoff(v(men), v(women)), cutoff = Math.min(aligned, v(asof)); out.textContent = `${aligned} / ${cutoff}`; ctx.view.replaceChildren(); ctx.view.append(ctx.table([ctx.text('Illustrative row', '예제 row'), ctx.text('Available under aligned files', '파일 day를 맞춘 뒤 가용'), ctx.text('Usable at prediction time', '예측 시점에 사용 가능')], [80, 93, 94, 100].map(day => [`2026 · day ${day}`, day <= aligned ? '✓' : '—', inSnapshot(2026, day, cutoff) ? '✓' : '—']))); const years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]; ctx.view.append(ctx.table([ctx.text('Rolling role', 'rolling 역할'), ctx.text('Seasons shown', '표시한 season')], ['train', 'validation', 'excluded'].map(role => [ctx.text(role, { train: '학습', validation: '검증', excluded: '제외' }[role]), years.filter(y => seasonRole(y, v(validation)) === role).join(', ')]))); ctx.status.textContent = ctx.text('The notebook aligned men/women availability to day 93. Here the men=93, women=97 example reproduces that minimum; 97 is an invented example input. An explicit prediction-date cap additionally illustrates the as-of condition. The row days are synthetic; season-forward training uses only seasons before validation, never the future seasons listed as excluded.', '노트북은 남녀 파일의 가용 범위를 day 93으로 맞췄습니다. 여기서는 남자=93, 여자=97인 예제로 그 최솟값을 재현하며 97은 가정한 입력입니다. 예측 시점의 상한도 따로 적용해 as-of 조건을 보여 줍니다. row의 day는 가상이며 season-forward 학습에는 검증 season보다 이전 season만 사용합니다.'); });
  });
})();
