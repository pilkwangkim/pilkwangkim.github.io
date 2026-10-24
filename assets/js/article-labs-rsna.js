(function () {
  'use strict';
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const logit = p => Math.log(clamp(p, 1e-9, 1 - 1e-9) / (1 - clamp(p, 1e-9, 1 - 1e-9)));
  function auc(labels, scores) {
    let total = 0, wins = 0;
    labels.forEach((y, i) => { if (y) labels.forEach((n, j) => {
      if (!n) { total++; wins += scores[i] > scores[j] ? 1 : scores[i] === scores[j] ? 0.5 : 0; }
    }); });
    return total ? wins / total : null;
  }
  function ranks(values) {
    return values.map(v => (values.filter(x => x < v).length + (values.filter(x => x === v).length + 1) / 2) / values.length);
  }
  function bce(q, p) { p = clamp(p, 1e-9, 1 - 1e-9); return -q * Math.log(p) - (1 - q) * Math.log(1 - p); }
  function windowLoss(q, p1, p2) {
    const mean = (p1 + p2) / 2;
    return { separate: (bce(q, p1) + bce(q, p2)) / 2, combined: bce(q, mean),
      separateGrad: [(p1 - q) / 2, (p2 - q) / 2],
      combinedGrad: [p1, p2].map(p => (mean - q) * p * (1 - p) / (2 * mean * (1 - mean))) };
  }
  function roc(labels, scores) {
    const pos = labels.filter(Boolean).length, neg = labels.length - pos;
    const thresholds = [...new Set(scores)].sort((a, b) => b - a);
    return [[0, 0], ...thresholds.map(t => [labels.reduce((n, y, i) => n + (!y && scores[i] >= t), 0) / neg,
      labels.reduce((n, y, i) => n + (y && scores[i] >= t), 0) / pos])];
  }
  function reduce(matrix, weight, method) {
    const columns = matrix[0].map((_, j) => matrix.map(row => row[j]));
    const transformed = method === 'rank' ? columns.map(ranks) : columns;
    return matrix.map((row, i) => {
      const a = method === 'logit' ? sigmoid((logit(row[0]) + logit(row[1])) / 2) : (transformed[0][i] + transformed[1][i]) / 2;
      const b = method === 'logit' ? sigmoid((logit(row[2]) + logit(row[3])) / 2) : (transformed[2][i] + transformed[3][i]) / 2;
      return weight * a + (1 - weight) * b;
    });
  }
  function teacherStatus(evalFold, exampleFold, excluded) {
    if (!excluded.includes(evalFold)) return 'evaluation-path';
    return excluded.includes(exampleFold) ? 'double-exclusion' : 'evaluation-excluded';
  }
  function geometry(crop, pixels, slices, studies = 4407, slots = 6) {
    return { pitch: crop / pixels, bytes: studies * slots * slices * pixels * pixels };
  }
  const budgetRecords = [
    [240, .0068, .0086], [300, .0067, .0057], [360, .0067, .0074],
    [420, .0067, .0090], [480, .0067, .0083], [540, .0067, .0090]
  ];
  function historyRows(rows, recent, includeFailure) {
    const end = Math.max(...rows.map(r => Date.parse(r.date_utc)));
    return rows.filter(r => (!recent || Date.parse(r.date_utc) >= end - 14 * 86400000) &&
      (includeFailure || r.outcome === 'numeric_score'));
  }
  const api = { auc, ranks, bce, windowLoss, roc, reduce, teacherStatus, geometry, budgetRecords, historyRows, sigmoid, logit };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const register = window.ArticleLabs.register;
  const number = el => Number(el.value);
  const clear = ctx => ctx.view.replaceChildren();
  const paragraph = (ctx, en, ko) => ctx.view.append(ctx.el('p', {}, ctx.text(en, ko)));

  register('rsna-ranking', ctx => {
    const labels = [1, 1, 1, 1, 0, 0, 0, 0], base = [.9, .75, .4, .3, .8, .5, .25, .1];
    const which = ctx.control('Study to move', '이동할 검사', { type: 'select', value: '2', options: base.map((_, i) => ({ value: String(i), en: `${labels[i] ? 'Positive' : 'Negative'} ${i % 4 + 1}`, ko: `${labels[i] ? '양성' : '음성'} ${i % 4 + 1}` })) });
    const score = ctx.control('Study score', '검사 점수', { type: 'range', min: 0, max: 1, step: .01, value: .4 });
    const power = ctx.control('Monotonic power', '단조 변환 지수', { type: 'range', min: .2, max: 4, step: .1, value: 1 });
    const threshold = ctx.control('Decision threshold', '판정 임계값', { type: 'range', min: 0, max: 1, step: .01, value: .5 });
    const a = ctx.metric('Pairwise AUC', '쌍별 AUC'), acc = ctx.metric('Threshold accuracy', '임계값 정확도');
    which.addEventListener('change', () => { score.value = base[number(which)]; });
    ctx.onChange(() => {
      const scores = base.slice(); scores[number(which)] = number(score);
      const transformed = scores.map(p => p ** number(power));
      a.textContent = ctx.format(auc(labels, transformed), 3);
      acc.textContent = `${transformed.filter((p, i) => (p >= number(threshold)) === Boolean(labels[i])).length}/8`;
      clear(ctx);
      ctx.view.append(ctx.plot({ x: [0, 1], y: [0, 1], xLabel: 'FPR', yLabel: 'TPR', title: ctx.text('ROC of eight synthetic studies', '가상 검사 8개의 ROC'),
        series: [{ points: [[0, 0], [1, 1]], color: 'muted', dash: true }, { points: roc(labels, transformed), color: 'blue' }] }));
      ctx.view.append(ctx.table([ctx.text('Positive / negative', '양성 / 음성'), 'N1', 'N2', 'N3', 'N4'], transformed.slice(0, 4).map((p, i) => [`P${i + 1}`, ...transformed.slice(4).map(n => p > n ? '1' : p === n ? '½' : '0')])));
      ctx.status.textContent = ctx.text('Each cell is one positive–negative pair. Power changes score scale; the threshold changes binary decisions.', '각 칸은 양성·음성 검사 한 쌍입니다. 지수는 점수 척도를, 임계값은 이진 판정을 바꿉니다.');
    });
  });

  register('rsna-geometry', ctx => {
    const crop = ctx.control('Field width (mm)', '시야 폭 (mm)', { type: 'range', min: 80, max: 200, step: 10, value: 130 });
    const pixels = ctx.control('Output width (pixels)', '출력 폭 (pixel)', { type: 'select', value: '336', options: [224, 336, 448].map(v => ({ value: String(v), en: String(v), ko: String(v) })) });
    const slices = ctx.control('Slices per slot', '슬롯당 slice 수', { type: 'range', min: 4, max: 40, step: 2, value: 16 });
    const pitch = ctx.metric('Nominal output pitch', '명목 출력 pitch'), storage = ctx.metric('Leading uint8 cache term', 'uint8 cache 주요 항');
    ctx.onChange(() => {
      const g = geometry(number(crop), number(pixels), number(slices));
      pitch.textContent = `${ctx.format(g.pitch, 3)} mm/pixel`; storage.textContent = `${ctx.format(g.bytes / 1e9, 1)} GB`;
      clear(ctx);
      const svg = ctx.svg('svg', { class: 'al-chart', viewBox: '0 0 640 220', role: 'img', 'aria-label': ctx.text('Physical field and output sampling grid', '물리 시야와 출력 sampling 격자') });
      const width = number(crop) * 2, left = 320 - width / 2;
      svg.append(ctx.svg('rect', { x: 100, y: 20, width: 440, height: 160, fill: '#87939c', opacity: .12 }));
      svg.append(ctx.svg('rect', { x: left, y: 25, width, height: 150, fill: 'none', stroke: '#357ca4', 'stroke-width': 3 }));
      for (let i = 0; i < number(slices); i++) {
        const x = left + (i + .5) * width / number(slices);
        svg.append(ctx.svg('line', { x1: x, x2: x, y1: 35, y2: 165, stroke: '#357ca4', opacity: .45 }));
      }
      [[200, 95], [320, 100], [455, 80]].forEach(([x, y]) => svg.append(ctx.svg('circle', { cx: x, cy: y, r: 8, fill: x >= left && x <= left + width ? '#bd8d2d' : '#87939c' })));
      svg.append(ctx.svg('text', { x: 320, y: 205, 'text-anchor': 'middle', fill: 'currentColor' }, `${number(crop)} mm / ${number(pixels)} pixels · ${number(slices)} slices`));
      ctx.view.append(svg);
      paragraph(ctx, 'The dots are fixed schematic structures. The crop controls context; the lines show uniformly sampled slice locations. Cache: 4,407 studies × 6 slots × slices × pixels² bytes.', '점은 위치가 고정된 구조의 모식도입니다. Crop은 문맥을, 선은 균일하게 고른 slice 위치를 나타냅니다. Cache: 검사 4,407개 × 슬롯 6개 × slice 수 × pixel 수² byte.');
      ctx.status.textContent = ctx.text('Output pitch describes resizing. Acquisition resolution and model AUC are separate quantities.', '출력 pitch는 resize의 척도입니다. 촬영 해상도와 모델 AUC는 별개의 값입니다.');
    });
  });

  function lossFactory(ctx) {
    const windows = ctx.root.dataset.preset === 'windows';
    const q = ctx.control('Target q', '학습값 q', { type: 'range', min: 0, max: 1, step: .01, value: windows ? 1 : .02 });
    const p1 = ctx.control(windows ? 'Window 1 probability' : 'Prediction p', windows ? 'Window 1 확률' : '예측값 p', { type: 'range', min: .01, max: .99, step: .01, value: windows ? .85 : .2 });
    const p2 = windows ? ctx.control('Window 2 probability', 'Window 2 확률', { type: 'range', min: .01, max: .99, step: .01, value: .1 }) : null;
    const weight = windows ? null : ctx.control('Loss weight w', '손실 가중치 w', { type: 'range', min: 0, max: 1, step: .05, value: 1 });
    const loss = ctx.metric(windows ? 'Mean BCE / BCE of mean' : 'Weighted BCE', windows ? 'BCE 평균 / 평균 확률의 BCE' : '가중 BCE');
    const grad = ctx.metric(windows ? 'Jensen gap' : 'Logit gradient', windows ? 'Jensen gap' : 'Logit gradient');
    ctx.onChange(() => {
      clear(ctx);
      const a = number(p1), target = number(q);
      if (windows) {
        const result = windowLoss(target, a, number(p2));
        loss.textContent = `${ctx.format(result.separate)} / ${ctx.format(result.combined)}`;
        grad.textContent = ctx.format(result.separate - result.combined);
        ctx.view.append(ctx.plot({ x: [.01, .99], y: [0, 5], xLabel: 'p₂', yLabel: 'BCE', title: ctx.text('Move evidence between windows', '두 window 사이의 증거 분배'), series: [
          { points: Array.from({ length: 99 }, (_, i) => { const p = (i + 1) / 100; return [p, windowLoss(target, a, p).separate]; }), color: 'gold', label: 'mean BCE' },
          { points: Array.from({ length: 99 }, (_, i) => { const p = (i + 1) / 100; return [p, windowLoss(target, a, p).combined]; }), color: 'blue', label: 'BCE(mean p)' }
        ] }));
        ctx.view.append(ctx.table([ctx.text('Logit gradient', 'Logit gradient'), 'window 1', 'window 2'], [
          ['mean BCE', ...result.separateGrad.map(x => ctx.format(x))], ['BCE(mean p)', ...result.combinedGrad.map(x => ctx.format(x))]
        ]));
        ctx.status.textContent = ctx.text('The study target is shared; the second objective judges the probability-averaged prediction. The inequality does not determine trained AUC.', '검사 단위 학습값은 같습니다. 두 번째 목적함수는 평균 확률을 평가합니다. 이 부등식이 학습 후 AUC를 결정하지는 않습니다.');
      } else {
        const w = number(weight), g = w * (a - target);
        loss.textContent = ctx.format(w * bce(target, a)); grad.textContent = `${g >= 0 ? '+' : ''}${ctx.format(g)}`;
        ctx.view.append(ctx.plot({ x: [0, 1], y: [-1, 1], xLabel: 'q', yLabel: 'w(p − q)', title: ctx.text('Target scale changes the update direction', '학습값 척도와 갱신 방향'),
          series: [{ points: [[0, w * a], [1, w * (a - 1)]], color: 'blue' }, { points: [[0, 0], [1, 0]], color: 'muted', dash: true }], markers: [{ x: target, y: g, color: 'gold', label: `q=${target}` }] }));
        ctx.view.append(ctx.table(['q', 'w(p − q)', ctx.text('One logit step, η = 0.5', 'Logit 한 step, η = 0.5')], [.02, .6, target].map(t => [ctx.format(t, 2), ctx.format(w * (a - t)), ctx.format(sigmoid(logit(a) - .5 * w * (a - t)))])));
        ctx.status.textContent = ctx.text('A positive gradient lowers the logit under gradient descent; a negative gradient raises it. The example computes one update, without retraining a model.', 'Gradient descent에서 양의 gradient는 logit을 낮추고, 음의 gradient는 높입니다. 이 예제는 모델 재학습 없이 한 번의 갱신을 계산합니다.');
      }
    });
  }
  register('rsna-target-loss', lossFactory);
  register('rsna-window-loss', lossFactory);

  register('rsna-reference-objective', ctx => {
    const condition = ctx.control('Recorded composition', '기록된 조합', { type: 'select', value: '1', options: [{ value: '0', en: 'Starting ensemble', ko: '출발 ensemble' }, { value: '1', en: 'Candidate ensemble', ko: '후보 ensemble' }] });
    const mode = ctx.control('Selection objective', '선택 목적함수', { type: 'select', value: 'minimum', options: [{ value: 'minimum', en: 'Minimum of three references', ko: '세 기준의 최솟값' }, { value: 'primary', en: 'Mean of Opus and GPT', ko: 'Opus와 GPT 평균' }] });
    const score = ctx.metric('Objective', '목적함수'), active = ctx.metric('Binding reference', '목적함수를 결정하는 기준');
    ctx.onChange(() => {
      const t = number(condition), starts = [.8210, .8791, .8835], ends = [.8268, .8743, .8788], names = ['Rules v1', 'Opus', 'GPT'];
      const values = starts.map((x, i) => x + t * (ends[i] - x));
      score.textContent = ctx.format(mode.value === 'minimum' ? Math.min(...values) : (values[1] + values[2]) / 2);
      active.textContent = mode.value === 'minimum' ? names[values.indexOf(Math.min(...values))] : 'Opus + GPT';
      clear(ctx); ctx.view.append(ctx.plot({ x: [0, 1], y: [.81, .9], xLabel: ctx.text('Starting (0) / candidate (1)', '출발 (0) / 후보 (1)'), yLabel: 'AUC', title: ctx.text('Different score levels control the minimum', '서로 다른 점수 수준과 최솟값'), xTicks: [0, 1], markers: starts.flatMap((v, i) => [{ x: 0, y: v, label: names[i], color: ['muted', 'gold', 'blue'][i] }, { x: 1, y: ends[i], label: names[i], color: ['muted', 'gold', 'blue'][i] }]) }));
      ctx.view.append(ctx.table(['Reference', 'Starting', 'Candidate', 'Δ'], starts.map((v, i) => [names[i], ctx.format(v), ctx.format(ends[i]), ctx.format(ends[i] - v)])));
      ctx.status.textContent = ctx.text('Two recorded compositions, scored against the same three references. Rules v1 controls the minimum at both points.', '동일한 세 기준으로 채점한 실제 두 조합입니다. 두 조합 모두 Rules v1이 최솟값을 결정합니다.');
    });
  });

  register('rsna-ensemble', ctx => {
    const method = ctx.control('Within-family reducer', '계열 내부 집계', { type: 'select', value: 'logit', options: [{ value: 'prob', en: 'Mean probability', ko: '확률 평균' }, { value: 'rank', en: 'Mean member percentile', ko: '개별 member percentile 평균' }, { value: 'logit', en: 'Sigmoid of mean logits', ko: 'Logit 평균의 sigmoid' }] });
    const weight = ctx.control('Family A outer weight', '계열 A 최종 가중치', { type: 'range', min: 0, max: 1, step: .05, value: .5 });
    const matrix = [[.99, .2, .75, .7], [.85, .6, .4, .5], [.35, .8, .7, .6], [.8, .4, .4, .3], [.3, .1, .2, .25], [.15, .2, .55, .5]], labels = [1, 1, 1, 0, 0, 0];
    const out = ctx.metric('Combined AUC', '조합 AUC'), shares = ctx.metric('Outer family shares', '계열 최종 비중');
    ctx.onChange(() => {
      const result = reduce(matrix, number(weight), method.value);
      out.textContent = ctx.format(auc(labels, result), 3); shares.textContent = `${ctx.format(number(weight), 2)} / ${ctx.format(1 - number(weight), 2)}`;
      clear(ctx); ctx.view.append(ctx.table([ctx.text('Study / label', '검사 / 라벨'), 'A₁', 'A₂', 'B₁', 'B₂', ctx.text('Combined', '조합값')], matrix.map((r, i) => [`${i + 1} / ${labels[i]}`, ...r.map(x => ctx.format(x, 2)), ctx.format(result[i])] )));
      ctx.view.append(ctx.plot({ x: [1, 6], y: [0, 1], xLabel: ctx.text('Study', '검사'), yLabel: ctx.text('Combined score', '조합 점수'), title: ctx.text('Same members, different operators', '같은 member와 다른 집계 연산'), markers: result.map((y, i) => ({ x: i + 1, y, label: labels[i] ? '+' : '−', color: labels[i] ? 'blue' : 'gold' })) }));
      ctx.status.textContent = ctx.text('Six synthetic studies, two members per family. Family means are combined with fixed outer shares. No operator is universally best.', '가상 검사 6개와 계열당 member 2개입니다. 계열 내부 집계 뒤 최종 비중을 적용합니다. 모든 상황에서 가장 좋은 연산은 없습니다.');
    });
  });

  register('rsna-budget', ctx => {
    const cap = ctx.control('September 23 planning cap (minutes)', '9월 23일 planning cap (분)', { type: 'select', value: '540', options: budgetRecords.map(r => ({ value: String(r[0]), en: String(r[0]), ko: String(r[0]) })) });
    const flat = ctx.metric('Flat selection Δ primary', '공통 구성 선택 Δ primary'), target = ctx.metric('Target-specific Δ primary', '소견별 선택 Δ primary');
    ctx.onChange(() => {
      const r = budgetRecords.find(x => x[0] === number(cap)); flat.textContent = `+${ctx.format(r[1])}`; target.textContent = `+${ctx.format(r[2])}`;
      clear(ctx); ctx.view.append(ctx.plot({ x: [240, 540], y: [.004, .01], xLabel: ctx.text('Planning cap (minutes)', 'Planning cap (분)'), yLabel: 'Δ primary AUC', title: ctx.text('Recorded cross-fitted procedure results', '기록된 cross-fit 선택 절차 결과'),
        series: [{ points: budgetRecords.map(r => [r[0], r[1]]), color: 'blue', label: ctx.text('Flat', '공통 구성') }, { points: budgetRecords.map(r => [r[0], r[2]]), color: 'gold', label: ctx.text('Target-specific', '소견별 구성') }], markers: [{ x: r[0], y: r[2], color: 'gold', label: String(r[0]) }] }));
      ctx.view.append(ctx.table([ctx.text('Executed candidate', '실행한 후보'), ctx.text('Planned minutes', '계획 시간 (분)'), 'Public'], [['Compact rank pair', '207.7', '0.940'], ['Broad', '539.8', '0.935'], ['Broad', '567.8', '0.937'], ['Time limit', '599.8', ctx.text('No AUC', 'AUC 없음')]]));
      ctx.status.textContent = ctx.text('The 300-minute target-specific result includes two fold fallbacks. Planning cost, local procedure AUC and an executed Public outcome answer different questions.', '300분의 소견별 결과에는 fold fallback 두 번이 포함됩니다. 계획 비용, 로컬 선택 절차 AUC, 실제 Public 결과는 서로 다른 질문에 답합니다.');
    });
  });

  register('rsna-teacher-folds', ctx => {
    const foldOptions = [0, 1, 2, 3, 4].map(v => ({ value: String(v), en: String(v), ko: String(v) }));
    const evalFold = ctx.control('Student evaluation fold j', 'Student 평가 fold j', { type: 'select', value: '0', options: foldOptions });
    const sampleFold = ctx.control('Training example fold k', '학습 예제 fold k', { type: 'select', value: '1', options: foldOptions });
    const excludeEval = ctx.control('Teacher excludes j', 'Teacher가 j 제외', { type: 'checkbox', value: true });
    const excludeSample = ctx.control('Teacher excludes k', 'Teacher가 k 제외', { type: 'checkbox', value: true });
    const verdict = ctx.metric('Teacher-target construction', 'Teacher 학습값 구성');
    ctx.onChange(() => {
      const j = number(evalFold), k = number(sampleFold), excluded = [];
      if (excludeEval.checked) excluded.push(j); if (excludeSample.checked) excluded.push(k);
      clear(ctx);
      if (j === k) {
        verdict.textContent = ctx.text('Not a student training example', 'Student 학습 예제가 아님');
        paragraph(ctx, 'An example from evaluation fold j must not enter this student’s training table. Choose k ≠ j.', '평가 fold j의 예제는 이 student의 학습표에 들어갈 수 없습니다. k ≠ j를 고르세요.');
        ctx.status.textContent = ctx.text('Student image holdout is checked before teacher ancestry.', 'Teacher 계보를 확인하기 전에 student의 영상 holdout을 확인합니다.');
        return;
      }
      const status = teacherStatus(j, k, excluded);
      const labels = { 'evaluation-path': ctx.text('Evaluation information path remains', '평가 정보 경로가 남음'), 'evaluation-excluded': ctx.text('Evaluation excluded; example seen by teacher', '평가 제외, teacher는 예제 학습'), 'double-exclusion': ctx.text('B2: both folds excluded', 'B2: 두 fold 모두 제외') };
      verdict.textContent = labels[status];
      ctx.view.append(ctx.table([ctx.text('Fold', 'Fold'), ctx.text('Student training', 'Student 학습'), ctx.text('Teacher training', 'Teacher 학습')], [0, 1, 2, 3, 4].map(f => [f, f === j ? '—' : '✓', excluded.includes(f) ? '—' : '✓'])));
      const svg = ctx.svg('svg', { class: 'al-chart', viewBox: '0 0 640 220', role: 'img', 'aria-label': labels[status] });
      const blocks = [[15, `j = ${j}`], [225, 'Teacher'], [435, `Student (−${j})`]];
      blocks.forEach(([x, text]) => { svg.append(ctx.svg('rect', { x, y: 70, width: 185, height: 65, rx: 8, fill: 'none', stroke: '#357ca4', 'stroke-width': 2 })); svg.append(ctx.svg('text', { x: x + 92, y: 109, 'text-anchor': 'middle', fill: 'currentColor' }, text)); });
      [[200, 225], [410, 435]].forEach(([x1, x2]) => svg.append(ctx.svg('line', { x1, y1: 102, x2, y2: 102, stroke: status === 'evaluation-path' ? '#b34c43' : '#87939c', 'stroke-width': 3, 'stroke-dasharray': status === 'evaluation-path' ? '' : '4 4' })));
      svg.append(ctx.svg('text', { x: 320, y: 180, 'text-anchor': 'middle', fill: 'currentColor' }, status === 'evaluation-path' ? ctx.text('j → teacher → targets for k → student', 'j → teacher → k의 학습값 → student') : ctx.text('j is excluded from teacher training', 'teacher 학습에서 j 제외')));
      ctx.view.append(svg);
      ctx.status.textContent = ctx.text('This checks the declared fold path. Unknown or full-data teacher ancestry cannot certify clean OOF; label-design exposure is a separate question.', '선언한 fold 경로를 확인합니다. 계보가 불명확하거나 전체 데이터로 학습한 teacher는 clean OOF를 보장하지 않습니다. 라벨 설계의 평가 자료 노출은 별도 문제입니다.');
    });
  });

  register('rsna-history', async ctx => {
    const rows = await ctx.fetchCSV('/assets/img/posts/rsna-knee-series/part5-submission-history.csv');
    if (!rows.length || rows.some(r => !Number.isFinite(Number(r.public)) || !Number.isFinite(Number(r.private)))) throw new Error('Invalid paired history');
    const period = ctx.control('Period', '기간', { type: 'select', value: 'recent', options: [{ value: 'all', en: 'All paired submissions', ko: '점수 쌍이 있는 전체 제출' }, { value: 'recent', en: 'Final fourteen days', ko: '마지막 14일' }] });
    const failure = ctx.control('Include known 0.500 deployment failure', '알려진 0.500 배포 실패 포함', { type: 'checkbox', value: false });
    const axis = ctx.control('Horizontal axis', '가로축', { type: 'select', value: 'time', options: [{ value: 'time', en: 'Submission time (UTC)', ko: '제출 시간 (UTC)' }, { value: 'public', en: 'Public score', ko: 'Public 점수' }] });
    const choices = rows.map((r, i) => ({ value: String(i), en: `${r.version ? 'v' + r.version : r.submission} · ${r.date_utc.slice(0, 10)} · ${r.public}/${r.private}`, ko: `${r.version ? 'v' + r.version : r.submission} · ${r.date_utc.slice(0, 10)} · ${r.public}/${r.private}` }));
    const ai = Math.max(0, rows.findIndex(r => r.version === '219')), bi = Math.max(0, rows.findIndex(r => r.version === '229'));
    const a = ctx.control('Compare A', '비교 A', { type: 'select', value: String(ai), options: choices });
    const b = ctx.control('Compare B', '비교 B', { type: 'select', value: String(bi), options: choices });
    const count = ctx.metric('Visible paired rows', '표시한 점수 쌍'), delta = ctx.metric('B − A Public / Private', 'B − A Public / Private');
    ctx.onChange(() => {
      const visible = historyRows(rows, period.value === 'recent', failure.checked), A = rows[number(a)], B = rows[number(b)];
      count.textContent = `${visible.length} / ${rows.length}`; delta.textContent = `${ctx.format(Number(B.public) - Number(A.public), 3)} / ${ctx.format(Number(B.private) - Number(A.private), 3)}`;
      clear(ctx);
      const origin = Date.parse(rows[0].date_utc), time = r => (Date.parse(r.date_utc) - origin) / 86400000, horizontal = r => axis.value === 'time' ? time(r) : Number(r.public);
      const xs = visible.map(horizontal), ys = visible.flatMap(r => [Number(r.public), Number(r.private)]), lo = Math.min(...ys) - .005, hi = Math.max(...ys) + .005;
      ctx.view.append(ctx.plot({ x: [Math.min(...xs) - .001, Math.max(...xs) + .001], y: [lo, hi], xLabel: axis.value === 'time' ? ctx.text('Days since Aug 6, UTC', '8월 6일 이후 일수, UTC') : 'Public', yLabel: 'AUC', title: ctx.text('Exact-submission score pairs', '동일 제출의 점수 쌍'),
        series: axis.value === 'time' ? [{ points: visible.map(r => [horizontal(r), Number(r.public)]), color: 'blue', label: 'Public' }, { points: visible.map(r => [horizontal(r), Number(r.private)]), color: 'gold', label: 'Private' }] : [{ points: [[Math.min(...xs), Math.min(...xs)], [Math.max(...xs), Math.max(...xs)]], color: 'muted', dash: true }],
        markers: axis.value === 'public' ? visible.map(r => ({ x: Number(r.public), y: Number(r.private), color: r.selected === 'True' ? 'gold' : 'blue', label: r.selected === 'True' ? `v${r.version}` : '' })) : visible.filter(r => r.selected === 'True').map(r => ({ x: horizontal(r), y: Number(r.private), color: 'gold', label: `v${r.version}` })) }));
      ctx.view.append(ctx.table([ctx.text('Comparison', '비교'), 'UTC', 'Public', 'Private', ctx.text('Outcome', '결과')], [['A', A], ['B', B]].map(([name, r]) => [name + (r.version ? ` · v${r.version}` : ''), r.date_utc.slice(0, 16).replace('T', ' '), r.public, r.private, r.warning || r.outcome])));
      ctx.status.textContent = ctx.text('October 23 snapshot: 154 paired numeric outcomes; 153 model-performance observations plus one known deployment failure. Private was revealed after the deadline. Unscored attempts are absent from this CSV.', '10월 23일 기록: 수치 점수 쌍 154개로, 모델 성능 관측 153개와 알려진 배포 실패 1개입니다. Private는 마감 뒤 공개됐습니다. 점수 없는 시도는 이 CSV에 없습니다.');
    });
  });
}());
