/* Public Eawag/MassBank Caffeine examples plus explicitly constructed ranking cases. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else api.install(root.ArticleLabs);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const DATA = '/assets/img/casmi-2026-guide/data/caffeine-massbank.json';
  const ADDUCTS = Object.freeze({ proton: 1.0072764666, deproton: -1.0072764666, sodium: 22.989221, waterloss: -17.003288 });
  const MASSES = Object.freeze([{ name: 'Caffeine', mass: 194.0804 }, { name: 'Eugenol', mass: 164.083730 }, { name: 'Isoeugenol', mass: 164.083730 }]);
  function ppm(a, b) { return 1e6 * (a - b) / b; }
  function massWindow(mz, adduct, tolerance) {
    const neutral = mz - ADDUCTS[adduct], half = neutral * tolerance * 1e-6;
    return { neutral, lower: neutral - half, upper: neutral + half,
      candidates: MASSES.map(c => ({ ...c, ppm: ppm(c.mass, neutral), eligible: Math.abs(c.mass - neutral) <= half })) };
  }
  function reciprocal(rank) { return rank >= 1 && rank <= 25 ? 1 / rank : 0; }
  function rankMetrics(ranks, inPool, insertWrong) {
    const final = ranks.map((r, i) => inPool[i] ? r + (insertWrong ? 1 : 0) : 0);
    const contributions = final.map(reciprocal), count = ranks.length;
    return { final, contributions, mrr: contributions.reduce((a, b) => a + b, 0) / count,
      coverage: inPool.filter(Boolean).length / count, hit25: contributions.filter(v => v > 0).length / count,
      top1: final.filter(v => v === 1).length / count };
  }
  function cosine(a, b, tolerance, squareRoot, removePrecursor) {
    const filter = record => record.peaks.filter(p => !removePrecursor || Math.abs(ppm(p.mz, record.precursor_mz)) > 10);
    const ap = filter(a), bp = filter(b), pairs = [];
    for (let i = 0; i < ap.length; i += 1) for (let j = 0; j < bp.length; j += 1) {
      const error = 1e6 * Math.abs(ap[i].mz - bp[j].mz) / ((ap[i].mz + bp[j].mz) / 2);
      if (error <= tolerance) pairs.push({ i, j, ppm: error });
    }
    // Nearest-mass one-to-one matching. All possible matches in this bundled series
    // are unambiguous for the allowed 0…25 ppm interval.
    pairs.sort((a, b) => a.ppm - b.ppm || a.i - b.i || a.j - b.j);
    const usedA = new Set(), usedB = new Set();
    const matches = pairs.filter(p => {
      if (usedA.has(p.i) || usedB.has(p.j)) return false;
      usedA.add(p.i); usedB.add(p.j); return true;
    });
    const intensity = p => squareRoot ? Math.sqrt(p.intensity_normalized_basepeak_1) : p.intensity_normalized_basepeak_1;
    const normA = Math.hypot(...ap.map(intensity)), normB = Math.hypot(...bp.map(intensity));
    const dot = matches.reduce((total, p) => total + intensity(ap[p.i]) * intensity(bp[p.j]), 0);
    return { value: normA && normB ? dot / (normA * normB) : 0, ap, bp, matches, normA, normB };
  }
  function install(labs) {
    if (!labs) return;
    const n = field => Number(field.value);
    const line = (ctx, en, ko) => ctx.el('p', {}, ctx.text(en, ko));
    const options = records => records.map((r, i) => ({ value: String(i), en: `${r.collision_energy_nominal_percent}% · ${r.accession.split('-').pop()}`, ko: `${r.collision_energy_nominal_percent}% · ${r.accession.split('-').pop()}` }));
    labs.register('casmi-spectra', async ctx => {
      const records = await ctx.fetchJSON(DATA);
      const record = ctx.control('Nominal HCD energy / record', 'nominal HCD energy / record', { type: 'select', value: '2', options: options(records) });
      const peakA = ctx.control('Peak A (m/z)', '피크 A (m/z)', { type: 'select', value: '', options: [] });
      const peakB = ctx.control('Peak B (m/z)', '피크 B (m/z)', { type: 'select', value: '', options: [] });
      const delta = ctx.metric('Absolute peak separation (Da, |z|=1)', '두 피크의 질량 차 (Da, |z|=1)'), intensity = ctx.metric('A / B relative intensities', 'A / B 상대 세기');
      let previous = -1;
      ctx.onReset(() => { previous = -1; });
      function render() {
        const r = records[n(record)];
        if (previous !== n(record)) {
          const opts = () => r.peaks.map((p, i) => ctx.el('option', { value: String(i) }, p.mz.toFixed(4)));
          peakA.replaceChildren(...opts()); peakB.replaceChildren(...opts());
          peakA.value = String(r.peaks.length - 1);
          let index = r.peaks.reduce((best, p, i) => p.intensity_normalized_basepeak_1 > r.peaks[best].intensity_normalized_basepeak_1 ? i : best, 0);
          if (index === r.peaks.length - 1) index = 0;
          peakB.value = String(index); previous = n(record);
        }
        const a = r.peaks[n(peakA)], b = r.peaks[n(peakB)], gap = Math.abs(a.mz - b.mz);
        delta.textContent = ctx.format(gap, 6); intensity.textContent = `${ctx.format(a.intensity_normalized_basepeak_1, 5)} / ${ctx.format(b.intensity_normalized_basepeak_1, 5)}`;
        const neutral = [{ name: 'CO', mass: 27.994915 }, { name: 'C₂H₃NO', mass: 57.021464 }, { name: 'H₂O', mass: 18.010565 }, { name: 'NH₃', mass: 17.026549 }];
        const nearest = neutral.reduce((best, c) => Math.abs(c.mass - gap) < Math.abs(best.mass - gap) ? c : best);
        ctx.view.replaceChildren(ctx.plot({ x: [45, 207], y: [0, 1.15], xLabel: 'm/z', yLabel: ctx.text('Base-peak-normalized intensity', 'base peak로 정규화한 세기'), title: `Caffeine · ${r.accession.split('-').pop()} · ${r.source_collision_energy}`, bars: r.peaks.map((p, i) => ({ x: p.mz, y: p.intensity_normalized_basepeak_1, width: 1.3, color: i === n(peakA) ? 'gold' : i === n(peakB) ? 'red' : 'blue', label: p.mz.toFixed(4) })), markers: [{ x: a.mz, y: a.intensity_normalized_basepeak_1, color: 'gold', label: 'A' }, { x: b.mz, y: b.intensity_normalized_basepeak_1, color: 'red', label: 'B' }] }),
          line(ctx, `${r.peaks.length} listed peaks. Nearest of the four listed neutral examples: ${nearest.name} (${nearest.mass.toFixed(6)} Da); residual ${((gap - nearest.mass) * 1000).toFixed(3)} mDa.`, `기록된 피크 ${r.peaks.length}개입니다. 제시한 중성 분자 네 개 중 가장 가까운 것은 ${nearest.name} (${nearest.mass.toFixed(6)} Da)이며 잔차는 ${((gap - nearest.mass) * 1000).toFixed(3)} mDa입니다.`),
          ctx.table([ctx.text('Selected peak', '선택한 피크'), 'm/z', ctx.text('Normalized intensity', '정규화 세기')], [[ctx.text('A · gold', 'A · 금색'), a.mz.toFixed(4), ctx.format(a.intensity_normalized_basepeak_1, 5)], [ctx.text('B · red', 'B · 빨강'), b.mz.toFixed(4), ctx.format(b.intensity_normalized_basepeak_1, 5)]]));
        ctx.status.textContent = ctx.text('Measured Eawag/MassBank data · CC BY · each record normalized separately. A mass match does not prove a fragmentation pathway.', 'Eawag/MassBank 실측 자료 · CC BY · record마다 독립적으로 정규화했습니다. 질량 일치만으로 분해 경로가 증명되지는 않습니다.');
      }
      ctx.onChange(render);
    });
    labs.register('casmi-cosine', async ctx => {
      const records = await ctx.fetchJSON(DATA);
      const query = ctx.control('Query measurement', 'query 측정', { type: 'select', value: '2', options: options(records) });
      const reference = ctx.control('Reference measurement', 'reference 측정', { type: 'select', value: '1', options: options(records) });
      const tolerance = ctx.control('Peak matching tolerance (ppm)', '피크 matching 허용오차 (ppm)', { type: 'range', min: 0, max: 25, step: 0.5, value: 10 });
      const sqrt = ctx.control('Use √intensity', '√intensity 사용', { type: 'checkbox', value: true });
      const precursor = ctx.control('Remove residual precursor (±10 ppm)', '남은 precursor 제외 (±10 ppm)', { type: 'checkbox', value: false });
      const score = ctx.metric('Spectral cosine', 'spectral cosine'), count = ctx.metric('One-to-one matched peaks', '일대일로 대응한 피크');
      function render() {
        const a = records[n(query)], b = records[n(reference)], r = cosine(a, b, n(tolerance), sqrt.checked, precursor.checked);
        score.textContent = ctx.format(r.value, 4); count.textContent = `${r.matches.length} / ${r.ap.length}, ${r.bp.length}`;
        const ia = new Set(r.matches.map(p => p.i)), ib = new Set(r.matches.map(p => p.j));
        const transform = p => sqrt.checked ? Math.sqrt(p.intensity_normalized_basepeak_1) : p.intensity_normalized_basepeak_1;
        ctx.view.replaceChildren(ctx.plot({ x: [45, 207], y: [-1.15, 1.15], xLabel: 'm/z', yLabel: sqrt.checked ? '±√I' : '±I', title: ctx.text('Query above · reference below · grey peaks are unmatched', '위: query · 아래: reference · 회색: 대응하지 않는 피크'), bars: r.ap.map((p, i) => ({ x: p.mz, y: transform(p), width: 1.3, color: ia.has(i) ? 'blue' : 'muted' })).concat(r.bp.map((p, i) => ({ x: p.mz, y: -transform(p), width: 1.3, color: ib.has(i) ? 'gold' : 'muted' }))) }),
          line(ctx, `Above: query ${a.collision_energy_nominal_percent}%; below: reference ${b.collision_energy_nominal_percent}%. Blue/gold peaks match; grey peaks do not. Nominal energy is not eV.`, `위는 query ${a.collision_energy_nominal_percent}%, 아래는 reference ${b.collision_energy_nominal_percent}%입니다. 파랑·금색 피크는 대응하고 회색은 대응하지 않습니다. nominal energy는 eV가 아닙니다.`),
          ctx.table([ctx.text('Query m/z', 'query m/z'), ctx.text('Reference m/z', 'reference m/z'), ctx.text('Distance (ppm)', '차이 (ppm)')], r.matches.map(p => [r.ap[p.i].mz.toFixed(4), r.bp[p.j].mz.toFixed(4), ctx.format(p.ppm, 3)])));
        ctx.status.textContent = ctx.text('All retained peaks contribute to the norms, including unmatched peaks. These records are the same molecule under different conditions.', '대응하지 않는 피크도 norm에 포함합니다. 같은 분자를 서로 다른 조건에서 측정한 record끼리 비교합니다.');
      }
      ctx.onChange(render);
    });
    labs.register('casmi-ranking', ctx => {
      const rank = [1, 2, 5, 26].map((v, i) => ctx.control(`Query ${i + 1}: correct rank (26 = outside top 25)`, `query ${i + 1}: 정답 순위 (26 = 25위 밖)`, { type: 'range', min: 1, max: 26, step: 1, value: v }));
      const pool = ctx.control('Query 4: correct structure is in the retrieval pool', 'query 4: 검색 pool에 정답이 있음', { type: 'checkbox', value: false });
      const insert = ctx.control('Insert a wrong generated candidate at rank 1', '틀린 생성 후보를 1위에 삽입', { type: 'checkbox', value: false });
      const mrr = ctx.metric('MRR@25', 'MRR@25'), recall = ctx.metric('Candidate-pool recall', '후보 pool 재현율'), hit = ctx.metric('Final Hit@25', '최종 Hit@25');
      function render() {
        const r = rankMetrics(rank.map(n), [true, true, true, pool.checked], insert.checked);
        mrr.textContent = ctx.format(r.mrr, 4); recall.textContent = ctx.format(r.coverage * 100, 0) + '%'; hit.textContent = ctx.format(r.hit25 * 100, 0) + '%';
        ctx.view.replaceChildren(ctx.plot({ x: [0.4, 4.6], y: [0, 1.05], xTicks: [1, 2, 3, 4], xLabel: ctx.text('Constructed query', '구성한 query'), yLabel: '1 / rank', title: ctx.text('Contribution of each molecule to reciprocal rank', '분자별 reciprocal rank'), bars: r.contributions.map((v, i) => ({ x: i + 1, y: v, width: 0.55, color: 'blue' })) }),
          ctx.table([ctx.text('Query', 'query'), ctx.text('Correct structure in pool?', 'pool에 정답이 있는가'), ctx.text('Final correct rank', '최종 정답 순위'), 'RR@25'], r.final.map((v, i) => [i + 1, i < 3 || pool.checked ? ctx.text('Yes', '있음') : ctx.text('No', '없음'), v && v <= 25 ? v : ctx.text('Outside top 25', '25위 밖'), ctx.format(r.contributions[i], 4)])));
        ctx.status.textContent = ctx.text('An absent answer cannot be reranked. A wrong insertion pushes every existing answer down one rank and may displace rank 25.', 'pool에 없는 정답은 재정렬할 수 없습니다. 틀린 후보를 삽입하면 기존 정답은 한 순위씩 밀리고 25위의 정답은 목록 밖으로 나갑니다.');
      }
      ctx.onChange(render);
    });
    labs.register('casmi-mass', ctx => {
      const mz = ctx.control('Observed precursor m/z (|z| = 1)', '관측 precursor m/z (|z| = 1)', { type: 'number', min: 50, max: 500, step: 0.000001, value: 195.0877 });
      const adduct = ctx.control('Assigned adduct', '부여한 adduct', { type: 'select', value: 'proton', options: [{ value: 'proton', en: '[M+H]+', ko: '[M+H]+' }, { value: 'deproton', en: '[M−H]−', ko: '[M−H]−' }, { value: 'sodium', en: '[M+Na]+', ko: '[M+Na]+' }, { value: 'waterloss', en: '[M−H₂O+H]+', ko: '[M−H₂O+H]+' }] });
      const tolerance = ctx.control('Neutral-mass tolerance (ppm)', '중성 질량 허용오차 (ppm)', { type: 'range', min: 0, max: 25, step: 0.5, value: 5 });
      const neutral = ctx.metric('Inferred neutral mass (Da)', '추정한 중성 질량 (Da)'), width = ctx.metric('Half-window (Da)', '질량 창의 반폭 (Da)');
      ctx.button('Caffeine observation', 'Caffeine 관측값', () => { mz.value = '195.0877'; adduct.value = 'proton'; render(); });
      ctx.button('Eugenol [M+H]+ example', 'Eugenol [M+H]+ 예제', () => { mz.value = '165.091006'; adduct.value = 'proton'; render(); });
      function render() {
        if (!Number.isFinite(n(mz)) || n(mz) < 50 || n(mz) > 500) { ctx.status.textContent = ctx.text('Enter m/z between 50 and 500.', 'm/z를 50부터 500 사이로 입력하세요.'); return; }
        const r = massWindow(n(mz), adduct.value, n(tolerance));
        neutral.textContent = ctx.format(r.neutral, 8); width.textContent = ctx.format(r.upper - r.neutral, 7);
        ctx.view.replaceChildren(ctx.plot({ x: [-30, 30], y: [0, 1], xLabel: ctx.text('Candidate mass error (ppm)', '후보 질량 오차 (ppm)'), yLabel: '', title: ctx.text('Mass window and nearby fixed candidates', '질량 창과 가까운 고정 후보'), bands: [{ x0: -n(tolerance), x1: n(tolerance), y0: 0, y1: 1, color: 'blue' }], markers: r.candidates.filter(c => Math.abs(c.ppm) <= 30).map((c, i) => ({ x: c.ppm, y: 0.3 + i * 0.22, label: c.name, color: c.eligible ? 'blue' : 'red' })) }),
          ctx.table([ctx.text('Fixed candidate', '고정 후보'), ctx.text('Neutral mass (Da)', '중성 질량 (Da)'), ctx.text('Error (ppm)', '오차 (ppm)'), ctx.text('Within window?', '질량 창 안인가')], r.candidates.map(c => [c.name, c.mass.toFixed(6), ctx.format(c.ppm, 3), c.eligible ? ctx.text('Yes', '안') : ctx.text('No', '밖')])));
        ctx.status.textContent = ctx.text('Changing the adduct shifts the neutral mass categorically. Eugenol and Isoeugenol always pass or fail together because their exact masses are identical.', 'adduct를 바꾸면 중성 질량의 기준이 크게 달라집니다. exact mass가 같은 Eugenol과 Isoeugenol은 항상 함께 통과하거나 탈락합니다.');
      }
      ctx.onChange(render);
    });
  }
  return { ppm, massWindow, reciprocal, rankMetrics, cosine, install, ADDUCTS, MASSES };
});
