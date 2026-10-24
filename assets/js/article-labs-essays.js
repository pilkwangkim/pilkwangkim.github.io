(function (global) {
  'use strict';
  const assignment = (aMinor, aMajor, bMinor, bMajor) => ({ first: aMinor + bMajor, second: aMajor + bMinor, minorGap: bMinor - aMinor, majorGap: bMajor - aMajor });
  const variance = (w, rho) => w * w + (1 - w) ** 2 + 2 * w * (1 - w) * rho;
  if (typeof module === 'object' && module.exports) module.exports = { assignment, variance };
  if (!global.ArticleLabs) return;
  global.ArticleLabs.register('essays-assignment', ctx => {
    const fields = [['A on α', 'A의 α 성과', 0.2], ['A on β', 'A의 β 성과', 1], ['B on α', 'B의 α 성과', 0.8], ['B on β', 'B의 β 성과', 1.4]].map(([en, ko, value]) => ctx.control(en, ko, { value, min: 0, max: 2, step: 0.05 }));
    const gap = ctx.metric('Difference U₂ − U₁', '차이 U₂ − U₁');
    ctx.onChange(() => {
      const result = assignment(...fields.map(f => Number(f.value))); gap.textContent = ctx.format(result.second - result.first, 3);
      ctx.view.replaceChildren(ctx.table([ctx.text('Assignment', '배치'), ctx.text('Total utility', '총효용')], [['A → α · B → β', ctx.format(result.first)], ['A → β · B → α', ctx.format(result.second)]]),
        ctx.el('p', {}, ctx.text('B’s advantage on α: ', 'α에서 B의 우위: ') + ctx.format(result.minorGap) + ' · ' + ctx.text('on β: ', 'β에서: ') + ctx.format(result.majorGap)));
      ctx.status.textContent = ctx.text('Each person receives exactly one task. These numbers describe the essay’s simplified allocation problem.', '한 사람에게 과업 하나씩 배정합니다. 이 숫자는 에세이의 단순화된 배치 문제를 나타냅니다.');
    });
  });
  global.ArticleLabs.register('essays-correlation', ctx => {
    const weight = ctx.control('Weight of domain A', '영역 A의 비중', { value: 0.5, min: 0, max: 1, step: 0.01 });
    const correlation = ctx.control('Correlation ρ', '상관계수 ρ', { value: 0.2, min: -1, max: 1, step: 0.05 });
    const risk = ctx.metric('Combined standard deviation', '합산 표준편차');
    ctx.onChange(() => {
      const w = Number(weight.value), rho = Number(correlation.value); risk.textContent = ctx.format(Math.sqrt(Math.max(0, variance(w, rho))));
      ctx.view.replaceChildren(ctx.plot({ x: [0, 1], y: [0, 1.05], xLabel: ctx.text('Weight of A', 'A의 비중'), yLabel: ctx.text('Standard deviation, σA = σB = 1', '표준편차, σA = σB = 1'),
        series: [{ points: Array.from({ length: 101 }, (_, i) => [i / 100, Math.sqrt(Math.max(0, variance(i / 100, rho)))]), color: 'blue' }], markers: [{ x: w, y: Math.sqrt(Math.max(0, variance(w, rho))), color: 'gold' }] }));
    });
  });
})(typeof globalThis !== 'undefined' ? globalThis : this);
