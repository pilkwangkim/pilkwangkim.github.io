(function () {
  'use strict';
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  function distance(a, b) { return Math.sqrt(a.reduce((s, x, i) => s + ((x - b[i]) * [1.625, .40625, .40625][i]) ** 2, 0)); }
  // Maximum-cardinality matching, then minimum distance; bounded educational graphs only.
  function matchPoints(gt, prediction, gate = 7, metric = distance) {
    let best = { pairs: [], distance: Infinity };
    function visit(i, used, pairs, cost) {
      if (i === gt.length) {
        if (pairs.length > best.pairs.length || (pairs.length === best.pairs.length && cost < best.distance)) best = { pairs: pairs.slice(), distance: cost };
        return;
      }
      visit(i + 1, used, pairs, cost);
      prediction.forEach((p, j) => {
        const d = metric(gt[i], p);
        if (!used.has(j) && d <= gate) { used.add(j); pairs.push([i, j]); visit(i + 1, used, pairs, cost + d); pairs.pop(); used.delete(j); }
      });
    }
    visit(0, new Set(), [], 0); return best;
  }
  function edgeScore(tp, fp, fn, predictedNodes, totalNodes) {
    const j = tp + fp + fn ? tp / (tp + fp + fn) : 0;
    return { j, adjusted: Math.max(0, j * (1 - .1 * (predictedNodes - totalNodes) / totalNodes)) };
  }
  function lineageCounts(targets, predToGt) {
    // Every source in this miniature matches an annotated source with a known successor.
    // Its outgoing prediction is evaluable even when its target has no GT match.
    const tp = targets.filter((target, source) => predToGt.get(target) === source).length;
    return { tp, fp: targets.length - tp, fn: 3 - tp };
  }
  function divisionChange(tp, fp, fn, addedTP, precision) {
    const recovered = Math.min(fn, Math.max(0, addedTP)), addedFP = recovered * (1 - precision) / precision;
    const start = tp / (tp + fp + fn), next = (tp + recovered) / (tp + fp + fn + addedFP);
    return { start, next, addedFP, recovered, precisionBoundary: start / (1 + start), weightedDelta: .1 * (next - start) };
  }
  const graphNodes = ['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2'];
  const truthEdges = [['a0', 'a1'], ['a1', 'b2'], ['a2', 'a3'], ['b0', 'b1'], ['b1', 'a2']];
  function components(nodes, edges) {
    const unseen = new Set(nodes), groups = [];
    while (unseen.size) {
      const stack = [unseen.values().next().value], group = []; unseen.delete(stack[0]);
      while (stack.length) { const n = stack.pop(); group.push(n); edges.forEach(([a, b]) => {
        const next = a === n ? b : b === n ? a : null;
        if (unseen.has(next)) { unseen.delete(next); stack.push(next); }
      }); }
      groups.push(group);
    }
    return groups;
  }
  function graphStages(switchParent, minimum) {
    const edges = [['a0', 'a1'], ['a1', 'a2'], ['a2', 'a3'], ['b0', 'b1'], ['b1', 'b2']];
    const chosen = switchParent ? edges.filter(([a]) => a !== 'a1' && a !== 'b1').concat([['a1', 'b2']]) : edges;
    const kept = components(graphNodes, chosen).filter(c => c.length >= minimum).flat(), keep = new Set(kept);
    const final = chosen.filter(([a, b]) => keep.has(a) && keep.has(b));
    const stats = (nodes, es) => {
      const tp = es.filter(([a, b]) => truthEdges.some(([x, y]) => x === a && y === b)).length;
      const fp = es.length - tp, fn = truthEdges.length - tp;
      return { tp, fp, fn, nodes: nodes.length, edges: es.length, score: edgeScore(tp, fp, fn, nodes.length, graphNodes.length).adjusted };
    };
    return { chosen, kept, final, before: stats(graphNodes, chosen), after: stats(kept, final) };
  }
  const tailRecords = [
    ['Primary, deployed threshold', .1007, -.0198], ['Primary, strict threshold', .0583, -.0169],
    ['Third field, low weight', .0124, -.0073], ['Novel-peak union', .0417, -.0306], ['Hole-filling union', .0448, -.0175]
  ];
  const mixture = (tail, median, share) => share * tail + (1 - share) * median;
  function mean(values) { return values.reduce((s, x) => s + x, 0) / values.length; }
  function std(values) { const m = mean(values); return Math.sqrt(mean(values.map(x => (x - m) ** 2))); }
  function alignFields(primary, secondary, weight, align) {
    const pm = mean(primary), sm = mean(secondary), scale = clamp(std(primary) / Math.max(1e-9, std(secondary)), .5, 2);
    const aligned = secondary.map(x => align ? (x - sm) * scale + pm : x);
    return { aligned, blended: primary.map((x, i) => (1 - weight) * x + weight * aligned[i]), pm, sm, scale };
  }
  function peaks(values, threshold) { return values.reduce((count, x, i) => count + (i > 0 && i < values.length - 1 && x >= threshold && x > values[i - 1] && x >= values[i + 1]), 0); }
  function jitterLinks(shift, registered) {
    const sources = [10, 20, 30, 40], targets = sources.map(x => x + 1 + shift);
    const result = matchPoints(sources.map(x => x + 1 + (registered ? shift : 0)), targets, 7, (a, b) => Math.abs(a - b));
    return { sources, targets, pairs: result.pairs, correct: result.pairs.filter(([i, j]) => i === j).length };
  }
  function foldRows(mode) {
    return mode === 'embryo' ? [[0, 0, 128, 71, 0], [1, 71, 0, 0, 128]] : [18, 18, 18, 17].map((a, i) => [i, 71 - a, 96, a, 32]);
  }
  const api = { distance, matchPoints, edgeScore, lineageCounts, divisionChange, components, graphStages, mixture, tailRecords, alignFields, peaks, jitterLinks, foldRows };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const register = window.ArticleLabs.register, number = el => Number(el.value);
  const clear = ctx => ctx.view.replaceChildren();
  const paragraph = (ctx, en, ko) => ctx.view.append(ctx.el('p', {}, ctx.text(en, ko)));

  register('biohub-lineage', ctx => {
    const dz = ctx.control('Middle target offset Δz (voxels)', '가운데 target 이동 Δz (voxel)', { type: 'range', min: 0, max: 6, step: .25, value: 2 });
    const dx = ctx.control('Middle target offset Δx (voxels)', '가운데 target 이동 Δx (voxel)', { type: 'range', min: -12, max: 12, step: .5, value: 0 });
    const extra = ctx.control('Add one unlinked prediction', '연결 없는 예측점 1개 추가', { type: 'checkbox', value: false });
    const crossing = ctx.control('Swap the two target links', 'Target 연결 두 개 교환', { type: 'checkbox', value: false });
    const dist = ctx.metric('Physical displacement', '물리 이동 거리'), score = ctx.metric('Adjusted edge Jaccard', '보정 edge Jaccard');
    ctx.onChange(() => {
      const gt = [[0, 0, 20], [0, 0, 50], [0, 0, 80]], prediction = gt.map(p => p.slice());
      prediction[1] = [number(dz), 0, 50 + number(dx)]; if (extra.checked) prediction.push([0, 15, 110]);
      const matches = matchPoints(gt, prediction), predToGt = new Map(matches.pairs.map(([i, j]) => [j, i]));
      const targets = crossing.checked ? [1, 0, 2] : [0, 1, 2];
      const { tp, fp, fn } = lineageCounts(targets, predToGt), counts = edgeScore(tp, fp, fn, 3 + prediction.length, 6);
      dist.textContent = `${ctx.format(distance(gt[1], prediction[1]), 2)} µm`; score.textContent = ctx.format(counts.adjusted);
      clear(ctx);
      const svg = ctx.svg('svg', { class: 'al-chart', viewBox: '0 0 640 220', role: 'img', 'aria-label': ctx.text('Annotated links and physical node matching', '주석 연결과 물리 좌표의 node matching') });
      const x = v => 55 + v * 4.5;
      targets.forEach((target, source) => svg.append(ctx.svg('line', { x1: x(gt[source][2]), y1: 40, x2: x(prediction[target][2]), y2: 135 + prediction[target][0] * 8, stroke: predToGt.get(target) === source ? '#357ca4' : '#b34c43', 'stroke-width': 3 })));
      gt.forEach((p, i) => { svg.append(ctx.svg('circle', { cx: x(p[2]), cy: 40, r: 8, fill: '#357ca4' })); svg.append(ctx.svg('circle', { cx: x(p[2]), cy: 135, r: 14, fill: 'none', stroke: '#87939c', 'stroke-dasharray': '4 3' })); svg.append(ctx.svg('text', { x: x(p[2]), y: 20, 'text-anchor': 'middle', fill: 'currentColor' }, `cell ${i + 1}`)); });
      prediction.forEach((p, i) => svg.append(ctx.svg('circle', { cx: x(p[2]), cy: 135 + p[0] * 8, r: 8, fill: predToGt.has(i) ? '#bd8d2d' : '#b34c43' })));
      ctx.view.append(svg);
      paragraph(ctx, 'Dashed circles mark target annotations; display axes are schematic.', '점선 원은 target 주석입니다. 화면 축은 모식도입니다.');
      ctx.view.append(ctx.table(['TP', 'FP', 'FN', ctx.text('Matched targets', '매칭된 target'), 'Npred / Ntotal'], [[tp, fp, fn, `${matches.pairs.length}/3`, `${3 + prediction.length}/6`]]));
      ctx.status.textContent = ctx.text('Three annotated links, 7 µm gate, maximum-cardinality one-to-one matches. An unmatched target still makes a false edge here because its source has an annotated successor. The extra unlinked node affects only the count adjustment; division is zero.', '주석 연결 3개, 7 µm gate, 최대 매칭 수를 우선하는 일대일 대응입니다. 이 예제의 source에는 주석 successor가 있으므로 target이 매칭되지 않은 연결도 FP입니다. 추가 고립점은 node 수 보정에만 영향을 주며 division은 0입니다.');
    });
  });

  register('biohub-folds', ctx => {
    const mode = ctx.control('Evaluation split', '평가 분할', { type: 'select', value: ctx.root.dataset.preset === 'embryo' ? 'balanced' : 'embryo', options: [{ value: 'embryo', en: 'Whole embryo held out', ko: '배아 전체 제외' }, { value: 'balanced', en: 'Four prefix-balanced movie folds', ko: '배아 비율을 맞춘 영상 4-fold' }] });
    const epoch = ctx.control('Epoch selected on evaluation fold', '평가 fold로 epoch 선택', { type: 'checkbox', value: false });
    const layer = ctx.control('Learned stage being checked', '확인할 학습 단계', { type: 'select', value: 'policy', options: [{ value: 'backbone', en: 'Backbone', ko: 'Backbone' }, { value: 'policy', en: 'Repair policy / threshold', ko: '복원 정책 / threshold' }] });
    const domain = ctx.metric('Unseen-embryo separation', '처음 보는 배아의 분리'), selection = ctx.metric('Evaluation used for selection', '평가 자료의 선택 사용');
    ctx.onChange(() => {
      const rows = foldRows(mode.value); domain.textContent = mode.value === 'embryo' ? ctx.text('Separated', '분리됨') : ctx.text('Both embryos remain in training', '학습에 두 배아가 남음');
      selection.textContent = epoch.checked ? ctx.text('Yes: epoch-selection path', '있음: epoch 선택 경로') : ctx.text('Fixed epoch in this example', '이 예제는 epoch 고정');
      clear(ctx); ctx.view.append(ctx.table([ctx.text('Fold', 'Fold'), 'Train A', 'Train B', 'Eval A', 'Eval B'], rows));
      const svg = ctx.svg('svg', { class: 'al-chart', viewBox: '0 0 640 220', role: 'img', 'aria-label': ctx.text('Training and evaluation embryo membership', '학습·평가 배아 구성') });
      rows.forEach((r, i) => {
        const y = 20 + i * 45;
        [[r[1], r[2]], [r[3], r[4]]].forEach((counts, j) => { const left = 90 + j * 300; counts.forEach((n, c) => { if (n) svg.append(ctx.svg('rect', { x: left + (c ? counts[0] : 0), y, width: n, height: 25, fill: c ? '#bd8d2d' : '#357ca4' })); }); });
        svg.append(ctx.svg('text', { x: 5, y: y + 18, fill: 'currentColor' }, `fold ${r[0]}`));
      });
      svg.append(ctx.svg('text', { x: 160, y: 210, 'text-anchor': 'middle', fill: 'currentColor' }, ctx.text('Training movies', '학습 영상')));
      svg.append(ctx.svg('text', { x: 460, y: 210, 'text-anchor': 'middle', fill: 'currentColor' }, ctx.text('Evaluation movies', '평가 영상')));
      ctx.view.append(svg);
      ctx.status.textContent = ctx.text(`Checking ${layer.value}. A backbone’s exclusion does not automatically transfer to a separately fitted policy. Blue: embryo A (71 movies); gold: B (128). Balanced counts illustrate the published approximate construction.`, `${layer.value} 단계 확인입니다. Backbone의 제외 조건은 따로 학습한 정책에 자동으로 적용되지 않습니다. 파랑: 배아 A(71편), 금색: B(128편). 균형 fold의 수는 원문에 제시된 근사 구성을 구체화했습니다.`);
    });
  });

  register('biohub-graph-stages', ctx => {
    const change = ctx.control('Use the locally correct parent switch', '국소적으로 올바른 parent 변경', { type: 'checkbox', value: true });
    const minimum = ctx.control('Minimum component length', '최소 component 길이', { type: 'range', min: 1, max: 4, step: 1, value: 3 });
    const stage = ctx.control('Display stage', '표시 단계', { type: 'select', value: 'after', options: [{ value: 'before', en: 'Before pruning', ko: 'Pruning 전' }, { value: 'after', en: 'After pruning', ko: 'Pruning 후' }] });
    const count = ctx.metric('Retained nodes / edges', '남은 node / edge'), score = ctx.metric('Adjusted edge Jaccard', '보정 edge Jaccard');
    ctx.onChange(() => {
      const r = graphStages(change.checked, number(minimum)), s = r[stage.value], edges = stage.value === 'before' ? r.chosen : r.final, kept = new Set(stage.value === 'before' ? graphNodes : r.kept);
      count.textContent = `${s.nodes} / ${s.edges}`; score.textContent = ctx.format(s.score);
      clear(ctx); const svg = ctx.svg('svg', { class: 'al-chart', viewBox: '0 0 640 220', role: 'img', 'aria-label': ctx.text('Parent choice followed by component pruning', 'Parent 선택 뒤 component pruning') });
      const xy = n => [70 + Number(n[1]) * 155, n[0] === 'a' ? 65 : 155];
      edges.forEach(([a, b]) => { const [x1, y1] = xy(a), [x2, y2] = xy(b), correct = truthEdges.some(([x, y]) => x === a && y === b); svg.append(ctx.svg('line', { x1, y1, x2, y2, stroke: correct ? '#357ca4' : '#b34c43', 'stroke-width': 3 })); });
      graphNodes.forEach(n => { const [cx, cy] = xy(n); svg.append(ctx.svg('circle', { cx, cy, r: 11, fill: kept.has(n) ? '#bd8d2d' : '#87939c', opacity: kept.has(n) ? 1 : .25 })); svg.append(ctx.svg('text', { x: cx, y: cy - 18, 'text-anchor': 'middle', fill: 'currentColor' }, n)); });
      ctx.view.append(svg); ctx.view.append(ctx.table([ctx.text('Stage', '단계'), 'TP', 'FP', 'FN', 'Score'], [['before', r.before], ['after', r.after]].map(([name, s]) => [name, s.tp, s.fp, s.fn, ctx.format(s.score)])));
      ctx.status.textContent = ctx.text('Seven synthetic nodes, five annotated true edges. The correct switch also releases an already claimed target. Pruning can remove other correct edges in the resulting short components.', '가상 node 7개와 주석 edge 5개입니다. 올바른 변경은 기존 parent의 target 점유도 해제합니다. 그 결과 생긴 짧은 component를 pruning하면 다른 올바른 edge도 사라질 수 있습니다.');
    });
  });

  register('biohub-reach', ctx => {
    const pool = ctx.control('Recorded candidate population', '기록된 후보 모집단', { type: 'select', value: 'swaps', options: [{ value: 'swaps', en: 'Reconstructed edge-swap generator', ko: '재구성한 edge-swap generator' }, { value: 'division', en: 'Division geometry inventory', ko: '분열 geometry 목록' }] });
    const step = ctx.control('Inspect a stage', '살펴볼 단계', { type: 'select', value: '1', options: [{ value: '0', en: 'Target error population', ko: '목표 오류 모집단' }, { value: '1', en: 'Candidate reach', ko: '후보가 닿는 범위' }, { value: '2', en: 'Final structural restriction', ko: '마지막 구조 제한' }] });
    const reach = ctx.metric('Count in selected stage', '선택 단계의 수'), coverage = ctx.metric('Fraction of starting population', '출발 모집단 대비 비율');
    ctx.onChange(() => {
      const rows = pool.value === 'swaps' ? [[ctx.text('Missing GT edges', '누락 GT edge'), 38060], [ctx.text('Replacement exists', '대체 후보 있음'), 183], [ctx.text('At most topology-legal swaps', '최대 topology-legal swap'), 115]] : [[ctx.text('Annotated divisions', '주석 분열'), 151], [ctx.text('Parent and daughters matched', 'Parent·두 daughter 매칭'), 107], [ctx.text('One existing parent–daughter edge', '기존 parent–daughter edge 하나'), 101]];
      const selected = rows[number(step)]; reach.textContent = selected[1].toLocaleString(); coverage.textContent = `${ctx.format(100 * selected[1] / rows[0][1], 2)}%`;
      clear(ctx); ctx.view.append(ctx.table([ctx.text('Stage', '단계'), ctx.text('Count', '수'), ctx.text('Starting population', '출발 모집단 대비')], rows.map(([name, n], i) => [`${i === number(step) ? '→ ' : ''}${name}`, n.toLocaleString(), `${ctx.format(100 * n / rows[0][1], 2)}%`])));
      paragraph(ctx, pool.value === 'swaps' ? 'The selected-action filter (3,178 historical swaps, +0.000365) is another population. It does not enumerate the generator or all graph edits.' : 'These are fixed-node geometry diagnostics. A ranker cannot recover events outside its proposal pool, and geometric reach alone does not establish a useful precision.', pool.value === 'swaps' ? '이미 선택된 과거 swap 3,178개를 거른 +0.000365 진단은 다른 모집단입니다. Generator 전체나 가능한 graph 편집 전체를 열거한 결과가 아닙니다.' : '고정 node의 geometry 진단입니다. Ranker는 후보가 없는 사건을 복원할 수 없으며, geometry로 도달할 수 있다는 사실만으로 precision을 알 수 없습니다.');
      ctx.status.textContent = ctx.text('Discrete recorded inventories; no threshold curve or global bound is inferred.', '기록된 목록만 비교합니다. Threshold 곡선이나 전역 상한을 추정하지 않습니다.');
    });
  });

  register('biohub-tail', ctx => {
    const arm = ctx.control('Detector composition', '검출기 구성', { type: 'select', value: '0', options: tailRecords.map((r, i) => ({ value: String(i), en: r[0], ko: ['Primary·기존 threshold', 'Primary·엄격한 threshold', '세 번째 field·작은 비중', '새 peak 합집합', '빈 영역 보충 합집합'][i] })) });
    const share = ctx.control('Assumed tail share', '가정한 tail 비율', { type: 'range', min: 0, max: 1, step: .01, value: .1 });
    const delta = ctx.metric('Movie-count mixture Δ', '영상 수 가중 혼합 Δ'), boundary = ctx.metric('Zero-crossing tail share', '0이 되는 tail 비율');
    ctx.onChange(() => {
      const r = tailRecords[number(arm)], change = mixture(r[1], r[2], number(share)), zero = -r[2] / (r[1] - r[2]);
      delta.textContent = `${change >= 0 ? '+' : ''}${ctx.format(change)}`; boundary.textContent = `${ctx.format(zero * 100, 1)}%`;
      clear(ctx); ctx.view.append(ctx.plot({ x: [0, 1], y: [-.035, .105], xLabel: ctx.text('Assumed tail share π', '가정한 tail 비율 π'), yLabel: ctx.text('Screening mixture Δ', '선별용 혼합 Δ'), title: ctx.text('Tail gain and median loss', 'Tail 개선과 median 손실'), series: [{ points: [[0, r[2]], [1, r[1]]], color: 'blue' }, { points: [[0, 0], [1, 0]], color: 'muted', dash: true }], markers: [{ x: number(share), y: change, color: 'gold', label: `π=${number(share)}` }] }));
      ctx.view.append(ctx.table([ctx.text('Recorded panel', '기록된 패널'), 'Δ'], [[ctx.text('Worst seven movies', '최악 영상 7편'), r[1]], [ctx.text('Near-median eight movies', '중앙값 부근 영상 8편'), r[2]]]));
      ctx.status.textContent = ctx.text('This is the article’s screening heuristic πΔtail + (1−π)Δmedian. The official metric pools edge denominators and division counts, so this is not its aggregate score.', '원문의 선별용 근사식 πΔtail + (1−π)Δmedian입니다. 공식 지표는 edge 분모와 분열 수를 따로 모으므로 공식 종합 점수와 다릅니다.');
    });
  });

  register('biohub-division', ctx => {
    const preset = ctx.control('Starting counts', '출발 사건 수', { type: 'select', value: 'hypothetical', options: [{ value: 'local', en: 'Recorded local v83: 8 / 19 / 143', ko: '기록된 로컬 v83: 8 / 19 / 143' }, { value: 'hypothetical', en: 'Hypothetical strong baseline: 32 / 20 / 48', ko: '가정한 강한 기준선: 32 / 20 / 48' }] });
    const precision = ctx.control('Added-pick precision', '추가 선택 precision', { type: 'range', min: .02, max: 1, step: .01, value: .17 });
    const recovered = ctx.control('Additional true divisions', '추가 true division 수', { type: 'range', min: 0, max: 30, step: 1, value: 8 });
    const change = ctx.metric('Weighted division-score Δ', '가중 division 점수 Δ'), boundary = ctx.metric('Break-even precision', '손익분기 precision');
    ctx.onChange(() => {
      const counts = preset.value === 'local' ? [8, 19, 143] : [32, 20, 48], r = divisionChange(...counts, number(recovered), number(precision));
      change.textContent = `${r.weightedDelta >= 0 ? '+' : ''}${ctx.format(r.weightedDelta)}`; boundary.textContent = ctx.format(r.precisionBoundary, 3);
      clear(ctx); ctx.view.append(ctx.plot({ x: [0, 1], y: [0, 1], xLabel: 'Starting J', yLabel: 'Precision', title: ctx.text('Precision required by the starting Jaccard', '출발 Jaccard에 따른 필요 precision'), series: [{ points: Array.from({ length: 51 }, (_, i) => [i / 50, (i / 50) / (1 + i / 50)]), color: 'blue', label: 'J / (1 + J)' }, { points: [[0, number(precision)], [1, number(precision)]], color: 'gold', dash: true, label: ctx.text('Added precision', '추가 precision') }], markers: [{ x: r.start, y: r.precisionBoundary, color: 'blue', label: `J=${ctx.format(r.start, 3)}` }] }));
      ctx.view.append(ctx.table([ctx.text('Counts', '사건 수'), 'TP', 'FP', 'FN', 'J'], [[ctx.text('Before', '전'), ...counts, ctx.format(r.start)], [ctx.text('After expected additions', '추가 후 기대 사건 수'), counts[0] + r.recovered, ctx.format(counts[1] + r.addedFP, 1), counts[2] - r.recovered, ctx.format(r.next)]]));
      ctx.status.textContent = ctx.text('Additional true events recover existing FN; false-event counts may be fractional expectations at the chosen precision. Existing matches and all edge/node terms stay fixed.', '추가 true event는 기존 FN을 복원합니다. 선택한 precision에 따른 false event 수는 소수인 기대값일 수 있습니다. 기존 사건 매칭과 edge·node 항은 고정했습니다.');
    });
  });

  register('biohub-alignment', ctx => {
    const background = ctx.control('Primary background logit', 'Primary 배경 logit', { type: 'range', min: -16, max: -4, step: .5, value: -14.5 });
    const width = ctx.control('Secondary peak width', 'Secondary peak 폭', { type: 'range', min: 1, max: 5, step: .25, value: 3 });
    const weight = ctx.control('Secondary blend weight', 'Secondary 혼합 비중', { type: 'range', min: 0, max: 1, step: .025, value: .475 });
    const align = ctx.control('Whole-frame mean / standard-deviation alignment', '전체 frame 평균·표준편차 정렬', { type: 'checkbox', value: true });
    const count = ctx.metric('Peaks: primary / blended', 'Peak: primary / 혼합'), means = ctx.metric('Mean logits P / S', '평균 logit P / S');
    ctx.onChange(() => {
      const positions = Array.from({ length: 81 }, (_, i) => i), centers = [15, 40, 65];
      const field = (bg, amp, sigma) => positions.map(x => bg + amp * centers.reduce((s, c) => s + Math.exp(-.5 * ((x - c) / sigma) ** 2), 0));
      const primary = field(number(background), 20, 1.2), secondary = field(-5, 12, number(width)), r = alignFields(primary, secondary, number(weight), align.checked), threshold = Math.log(.97 / .03);
      count.textContent = `${peaks(primary, threshold)} / ${peaks(r.blended, threshold)}`; means.textContent = `${ctx.format(r.pm, 2)} / ${ctx.format(r.sm, 2)}`;
      clear(ctx); const all = primary.concat(secondary, r.blended, r.aligned, [threshold]);
      ctx.view.append(ctx.plot({ x: [0, 80], y: [Math.min(...all) - 1, Math.max(...all) + 1], xLabel: ctx.text('Synthetic voxel index', '가상 voxel index'), yLabel: 'logit', title: ctx.text('Alignment can move genuine peaks too', '정렬은 실제 peak도 이동시킨다'), series: [{ points: primary.map((y, x) => [x, y]), color: 'blue', label: 'P' }, { points: secondary.map((y, x) => [x, y]), color: 'muted', label: 'S' }, { points: r.blended.map((y, x) => [x, y]), color: 'gold', label: 'blend' }, { points: [[0, threshold], [80, threshold]], color: 'red', dash: true, label: 'p = 0.97' }] }));
      ctx.status.textContent = ctx.text(`Three Gaussian peaks in 81 synthetic voxels. Alignment scale clip: [0.5, 2]; current scale ${ctx.format(r.scale, 2)}. Counts are generated here, separate from the article’s recorded 281/644/0/25 frame.`, `가상 voxel 81개에 Gaussian peak 3개를 놓았습니다. 정렬 scale 제한은 [0.5, 2], 현재 값은 ${ctx.format(r.scale, 2)}입니다. 이 화면의 수는 원문에 기록된 한 frame의 281/644/0/25와 별개입니다.`);
    });
  });

  register('biohub-registration', ctx => {
    const shift = ctx.control('Global stage shift (µm)', '전체 stage 이동 (µm)', { type: 'range', min: 0, max: 9, step: .25, value: 9 });
    const registered = ctx.control('Use the global translation in relinking', '재연결에 전체 이동량 사용', { type: 'checkbox', value: false });
    const correct = ctx.metric('Correct links / four cells', '올바른 연결 / 세포 4개'), unmatched = ctx.metric('Unmatched sources', '미연결 source');
    ctx.onChange(() => {
      const r = jitterLinks(number(shift), registered.checked); correct.textContent = `${r.correct}/4`; unmatched.textContent = String(4 - r.pairs.length);
      clear(ctx); const svg = ctx.svg('svg', { class: 'al-chart', viewBox: '0 0 640 220', role: 'img', 'aria-label': ctx.text('Global microscope translation and cell identity', '현미경의 전체 이동과 세포 identity') }), x = v => 40 + v * 10;
      r.pairs.forEach(([i, j]) => svg.append(ctx.svg('line', { x1: x(r.sources[i]), y1: 50, x2: x(r.targets[j]), y2: 150, stroke: i === j ? '#357ca4' : '#b34c43', 'stroke-width': 3 })));
      r.sources.forEach((v, i) => { svg.append(ctx.svg('circle', { cx: x(v), cy: 50, r: 9, fill: '#357ca4' })); svg.append(ctx.svg('text', { x: x(v), y: 25, 'text-anchor': 'middle', fill: 'currentColor' }, String(i + 1))); });
      r.targets.forEach((v, i) => { svg.append(ctx.svg('circle', { cx: x(v), cy: 150, r: 9, fill: '#bd8d2d' })); svg.append(ctx.svg('text', { x: x(v), y: 180, 'text-anchor': 'middle', fill: 'currentColor' }, String(i + 1))); });
      ctx.view.append(svg);
      paragraph(ctx, 'Each cell moves 1 µm biologically; the stage adds the same translation to every cell. A 7 µm admissibility gate and one-to-one distance matching link the two frames.', '각 세포의 생물학적 이동은 1 µm이며, stage는 모든 세포에 같은 이동량을 더합니다. 7 µm gate 안에서 거리 기반 일대일 matching으로 두 frame을 연결합니다.');
      ctx.status.textContent = ctx.text('The example supplies the exact translation. The competition method estimates it from detections; this diagram does not measure that estimation accuracy or an AUC gain.', '이 예제에는 정확한 전체 이동량을 주었습니다. 대회 방법은 검출점으로 그 값을 추정합니다. 이 모식도는 추정 정확도나 점수 개선을 측정하지 않습니다.');
    });
  });

  register('biohub-research-arms', ctx => {
    const arm = ctx.control('Comparison arm', '비교 실험군', { type: 'select', value: 'plugin', options: [{ value: 'base', en: 'P₀ at θ₀', ko: 'P₀와 θ₀' }, { value: 'plugin', en: 'Pᴅ at unchanged θ₀', ko: 'Pᴅ와 기존 θ₀' }, { value: 'adapt', en: 'Pᴅ with training-side θᴅ', ko: 'Pᴅ와 학습 측 θᴅ' }, { value: 'new', en: 'Separately developed P₁', ko: '따로 개발한 P₁' }] });
    const source = ctx.control('Mechanism D', '추가 원리 D', { type: 'select', value: 'proposals', options: [{ value: 'proposals', en: 'New node proposals', ko: '새 node 제안' }, { value: 'features', en: 'New feature scale', ko: '새 feature 척도' }, { value: 'events', en: 'Division evidence', ko: '분열 증거' }] });
    const question = ctx.metric('Question the arm answers', '이 실험군이 답하는 질문');
    const questions = { base: ['What is the complete reference?', '완전한 기준선은 무엇인가?'], plugin: ['Does D help this frozen system?', 'D가 고정한 시스템에 도움이 되는가?'], adapt: ['Does the declared compatibility change help?', '선언한 호환성 조정이 도움이 되는가?'], new: ['Is the alternative complete route worth developing?', '대안의 완전한 예측 경로를 더 개발할 가치가 있는가?'] };
    ctx.onChange(() => {
      clear(ctx); question.textContent = ctx.text(...questions[arm.value]);
      const settings = arm.value === 'adapt' ? 'θᴅ' : arm.value === 'new' ? 'θ₁' : 'θ₀', components = arm.value === 'new' ? ['A₀', 'B₁', 'C₁'] : ['A₀', 'B₀', 'C₀'];
      ctx.view.append(ctx.table([ctx.text('Detector', '검출기'), ctx.text('Association', 'Association'), ctx.text('Graph policy', 'Graph 정책'), ctx.text('Operating choices', '운영 설정'), 'D'], [[...components, settings, arm.value === 'base' ? '—' : source.value]]));
      paragraph(ctx, source.value === 'proposals' ? 'New proposals change the population seen by the linker. Test candidate usefulness separately from a particular frozen linker’s compatibility.' : source.value === 'features' ? 'A new scale changes what downstream thresholds mean. A declared training-side calibration tests a specific compatibility hypothesis.' : 'Division evidence needs a consumer that preserves compatible forks through relinking, pruning and serialization.', source.value === 'proposals' ? '새 제안은 linker가 보는 모집단을 바꿉니다. 후보의 유용성과 특정 고정 linker의 호환성을 나누어 시험합니다.' : source.value === 'features' ? '척도가 달라지면 후속 threshold의 의미도 바뀝니다. 학습 측에서 선언한 calibration은 구체적인 호환성 가설을 시험합니다.' : '분열 증거를 쓰려면 재연결·pruning·직렬화를 거쳐 호환되는 fork를 유지하는 consumer가 필요합니다.');
      ctx.status.textContent = ctx.text('This is an experiment-design schematic. Changing the arm states what is tested; it assigns no score to an unrun alternative. Refitting a learned head changes a component, rather than only θ.', '실험 설계의 모식도입니다. 실험군 변경은 무엇을 시험하는지 정하며, 미실행 대안의 점수를 만들지 않습니다. 학습된 head를 다시 맞추는 것은 θ 조정만이 아니라 component 변경입니다.');
    });
  });
}());
