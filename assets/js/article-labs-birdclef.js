(function () {
  'use strict';
  function ranks(values) { return values.map(v => (values.filter(x => x < v).length + (values.filter(x => x === v).length + 1) / 2) / values.length); }
  function columnRanks(matrix) {
    const columns = matrix[0].map((_, c) => ranks(matrix.map(row => row[c])));
    return matrix.map((row, r) => row.map((_, c) => columns[c][r]));
  }
  function topIndices(values, k) { return values.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k).map(x => x[1]); }
  function overlap(a, b, k) {
    return a.reduce((total, row, r) => { const old = new Set(topIndices(row, k)); return total + topIndices(b[r], k).filter(c => old.has(c)).length / k; }, 0) / a.length;
  }
  function smooth(matrix, groups, alpha) {
    return matrix.map(row => {
      const next = row.slice(); groups.forEach(group => { const average = group.reduce((s, c) => s + row[c], 0) / group.length; group.forEach(c => { next[c] = (1 - alpha) * row[c] + alpha * average; }); }); return next;
    });
  }
  function correction(anchor, side, options) {
    const A = columnRanks(anchor), S = columnRanks(side), { anchorK, sideK, tau, weight, budget, gates, maxActive = .75, minimumOverlap = .9 } = options;
    const mask = A.map((row, r) => { const ak = new Set(topIndices(row, anchorK)), sk = new Set(topIndices(S[r], sideK)); return row.map((v, c) => ak.has(c) || (sk.has(c) && v >= tau)); });
    const raw = A.map((row, r) => row.map((v, c) => mask[r][c] ? v + weight * gates[c] * (S[r][c] - v) : v));
    const movement = matrix => matrix.reduce((s, row, r) => s + row.reduce((ss, v, c) => ss + Math.abs(v - A[r][c]), 0), 0) / (A.length * A[0].length);
    const rawD = movement(raw), scale = rawD > budget && rawD > 0 ? budget / rawD : 1;
    const attempted = A.map((row, r) => row.map((v, c) => v + scale * (raw[r][c] - v))), D = movement(attempted);
    const active = attempted.flat().filter((v, i) => Math.abs(v - A.flat()[i]) > 1e-12).length / attempted.flat().length;
    const top3 = overlap(A, attempted, 3), top10 = overlap(A, attempted, 10);
    const accepted = D > 1e-12 && D <= budget + 1e-12 && active <= maxActive && top3 >= minimumOverlap && top10 >= minimumOverlap;
    return { A, S, mask, attempted, final: accepted ? attempted : A, D, rawD, scale, active, top3, top10, accepted };
  }
  const anchor = Array.from({ length: 6 }, (_, r) => Array.from({ length: 12 }, (_, c) => .08 + (((r * 7 + c * 5 + r * c * 3) % 29) / 29) * .82));
  const side = Array.from({ length: 6 }, (_, r) => Array.from({ length: 12 }, (_, c) => .05 + (((r * 11 + c * 3 + r * c * 7 + 9) % 31) / 31) * .9));
  const api = { ranks, columnRanks, topIndices, overlap, smooth, correction, anchor, side };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  window.ArticleLabs.register('birdclef-gated-ranks', ctx => {
    const anchorK = ctx.control('Anchor top-k per row', '행별 anchor top-k', { type: 'range', min: 0, max: 12, step: 1, value: 6 });
    const sideK = ctx.control('Sidecar top-k per row', '행별 sidecar top-k', { type: 'range', min: 0, max: 12, step: 1, value: 4 });
    const tau = ctx.control('Minimum anchor percentile τ', '최소 anchor percentile τ', { type: 'range', min: 0, max: 1, step: .05, value: .55 });
    const weight = ctx.control('Class weight cap', 'Class 가중치 상한', { type: 'range', min: 0, max: .3, step: .01, value: .03 });
    const budget = ctx.control('Mean movement budget D', '평균 이동 예산 D', { type: 'range', min: 0, max: .05, step: .001, value: .004 });
    const gate = ctx.control('Apply synthetic class gates', '가상 class gate 적용', { type: 'checkbox', value: true });
    const alpha = ctx.control('Genus smoothing α', 'Genus smoothing α', { type: 'range', min: 0, max: .4, step: .05, value: .15 });
    const row = ctx.control('Inspect soundscape row', '살펴볼 soundscape 행', { type: 'select', value: '0', options: anchor.map((_, i) => ({ value: String(i), en: `row ${i + 1}`, ko: `행 ${i + 1}` })) });
    const movement = ctx.metric('Attempted / budget movement', '시도한 이동 / 이동 예산'), head = ctx.metric('Top-3 / top-10 overlap', 'Top-3 / top-10 일치율'), verdict = ctx.metric('Final action', '최종 처리');
    ctx.onChange(() => {
      const smoothed = smooth(smooth(anchor, [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9, 10, 11]], Number(alpha.value)), [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], .05);
      const gates = gate.checked ? [1, 0, .7, 1, 0, .4, 1, 0, .8, 0, 1, .5] : Array(12).fill(1);
      const r = correction(smoothed, side, { anchorK: Number(anchorK.value), sideK: Number(sideK.value), tau: Number(tau.value), weight: Number(weight.value), budget: Number(budget.value), gates }), index = Number(row.value);
      movement.textContent = `${ctx.format(r.D)} / ${ctx.format(Number(budget.value))}`;
      head.textContent = `${ctx.format(100 * r.top3, 1)}% / ${ctx.format(100 * r.top10, 1)}%`;
      verdict.textContent = r.accepted ? ctx.text('Accept correction', '보정 적용') : ctx.text('Keep anchor', 'Anchor 유지');
      ctx.view.replaceChildren();
      ctx.view.append(ctx.plot({ x: [1, 12], y: [0, 1], xLabel: ctx.text('Taxon column', 'Taxon 열'), yLabel: ctx.text('Column percentile', '열 안의 percentile'), title: ctx.text(`Inspect row ${index + 1}: ranks computed across six rows`, `행 ${index + 1}: 여섯 행 사이에서 계산한 rank`),
        series: [{ points: r.A[index].map((v, c) => [c + 1, v]), color: 'blue', label: 'anchor A' }, { points: r.S[index].map((v, c) => [c + 1, v]), color: 'muted', label: 'sidecar S' }, { points: r.final[index].map((v, c) => [c + 1, v]), color: 'gold', label: ctx.text('final', '최종값') }] }));
      ctx.view.append(ctx.table([ctx.text('Taxon', 'Taxon'), 'A', 'S', ctx.text('Mask', 'Mask'), 'W', ctx.text('Final', '최종값')], r.A[index].map((a, c) => [`T${c + 1}`, ctx.format(a, 3), ctx.format(r.S[index][c], 3), r.mask[index][c] ? '✓' : '—', ctx.format(Number(weight.value) * gates[c] * r.scale, 3), ctx.format(r.final[index][c], 3)])));
      ctx.status.textContent = ctx.text(`Six synthetic rows × twelve taxa. Percentiles run down each column; top-k masks run across each row. Class smoothing α=0.05. Budget scale ${ctx.format(r.scale, 3)}; changed cells ${ctx.format(r.active * 100, 1)}%. Acceptance requires ≤75% changed cells and ≥90% top-3/top-10 overlap.`, `가상 행 6개 × taxon 12개입니다. Percentile은 각 열의 행 사이에서, top-k mask는 각 행의 열 사이에서 계산합니다. Class smoothing α=0.05입니다. 예산 scale ${ctx.format(r.scale, 3)}, 변경 칸 ${ctx.format(r.active * 100, 1)}%입니다. 변경 비율 ≤75%, top-3·top-10 일치율 ≥90%여야 적용합니다.`);
    });
  });
}());
