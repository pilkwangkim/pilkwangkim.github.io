/* Exact small quantum examples and explicit resource/rounding models. No verifier runs. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else api.install(root.ArticleLabs);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const TAU = 2 * Math.PI;
  function phase(theta) {
    return { p0: (1 + Math.cos(theta)) / 2, p1: (1 - Math.cos(theta)) / 2,
      before: [[Math.SQRT1_2, 0], [Math.cos(theta) * Math.SQRT1_2, Math.sin(theta) * Math.SQRT1_2]] };
  }
  function cleanup(stage, clean, usePhase) {
    let a = [1, 0, 0, 0]; // index = 2 * data + scratch
    function h() { a = [(a[0] + a[2]) * Math.SQRT1_2, (a[1] + a[3]) * Math.SQRT1_2,
      (a[0] - a[2]) * Math.SQRT1_2, (a[1] - a[3]) * Math.SQRT1_2]; }
    function cx() { [a[2], a[3]] = [a[3], a[2]]; }
    if (stage >= 1) h();
    if (stage >= 2) cx();
    if (stage >= 3 && usePhase) { a[1] *= -1; a[3] *= -1; }
    if (stage >= 4 && clean) cx();
    if (stage >= 5) h();
    const probabilities = a.map(x => x * x);
    return { a, probabilities, p0: probabilities[0] + probabilities[1],
      p1: probabilities[2] + probabilities[3], scratch1: probabilities[1] + probabilities[3] };
  }
  function normalize(x) {
    if (!Number.isInteger(x) || x < 1 || x > 3133452) throw new RangeError('x must be 1…3,133,452');
    const bitLength = Math.floor(Math.log2(x)) + 1;
    const s = Math.floor((22 - bitLength) / 2);
    const normalized = x * 2 ** (2 * s);
    return { s, normalized, m: normalized / 2 ** 20, bits: x.toString(2).padStart(22, '0') };
  }
  function fixedpoint(x, e0, bits) {
    const n = normalize(x), exact = 1 / Math.sqrt(n.m), scale = 2 ** bits;
    const y0 = Math.floor(exact * (1 + e0) * scale) / scale;
    const next = y0 * (3 - n.m * y0 * y0) / 2;
    const rounded = Math.floor(next * scale) / scale;
    return { ...n, exact, y0, next, rounded, target: 1 / Math.sqrt(x),
      errors: [e0, y0 / exact - 1, next / exact - 1, rounded / exact - 1] };
  }
  function gfMul(a, b) {
    let product = 0;
    while (b) { if (b & 1) product ^= a; b >>= 1; a <<= 1; if (a & 4) a ^= 7; }
    return product;
  }
  function gfPow(a, n) { let r = 1; for (let i = 0; i < n; i += 1) r = gfMul(r, a); return r; }
  function gfDiv(a, b) { if (!b) throw new RangeError('Division by zero'); return gfMul(a, gfPow(b, 2)); }
  function decode(s0, s1) {
    if (s0 === 0 && s1 === 0) return { position: -1, value: 0 };
    if (!s0 || !s1) throw new RangeError('Not a single-symbol syndrome');
    const ratio = gfDiv(s1, s0);
    const position = [1, 2, 3].indexOf(ratio);
    return { position, value: gfDiv(s0, gfPow(2, position)) };
  }
  function syndrome(position, value, mask) {
    const s0 = gfMul(value, gfPow(2, position));
    const s1 = gfMul(value, gfPow(2, 2 * position));
    const selected = value & mask;
    const parity = ((selected & 1) + ((selected >> 1) & 1)) % 2;
    return { s0, s1, ...decode(s0, s1), selected, parity, sign: parity ? -1 : 1 };
  }
  function rotate(vector, p, q, theta) {
    const out = vector.slice(), c = Math.cos(theta), s = Math.sin(theta);
    out[p] = c * vector[p] - s * vector[q]; out[q] = s * vector[p] + c * vector[q];
    return out;
  }
  function givens(a, b, reverse, bits) {
    const quantize = angle => ((Math.round(angle / TAU * 2 ** bits) % 2 ** bits) + 2 ** bits) % 2 ** bits;
    const words = [quantize(a), quantize(b)], angles = words.map(w => TAU * w / 2 ** bits);
    function sequence(aa, bb, rev) {
      return rev ? rotate(rotate([1, 0, 0], 0, 2, bb), 0, 1, aa)
        : rotate(rotate([1, 0, 0], 0, 1, aa), 0, 2, bb);
    }
    const ideal = sequence(a, b, reverse), actual = sequence(angles[0], angles[1], reverse);
    return { ideal, actual, other: sequence(a, b, !reverse), words, angles,
      error: Math.hypot(...actual.map((v, i) => v - ideal[i])), norm: Math.hypot(...actual) };
  }
  function resource(task, q, gates, penalty) {
    const ratio = q ** (task === 'dqi' ? 2 : 1) * gates * (task === 'invsqrt' ? 1 : penalty);
    return { ratio, improvement: 1 / ratio, overall: ratio ** (-1 / 3) };
  }
  function install(labs) {
    if (!labs) return;
    const field = ['0', '1', 'α', 'α²'];
    const bit = x => x.toString(2).padStart(2, '0');
    const n = input => Number(input.value);
    const replace = (ctx, nodes) => ctx.view.replaceChildren(...nodes);
    const line = (ctx, en, ko) => ctx.el('p', {}, ctx.text(en, ko));
    labs.register('quantum-phase', ctx => {
      const angle = ctx.control('Relative phase φ (degrees)', '상대 phase φ (도)', { type: 'range', min: 0, max: 360, step: 1, value: 180 });
      const p0 = ctx.metric('After H: P(0)', 'H 적용 후 P(0)'), p1 = ctx.metric('After H: P(1)', 'H 적용 후 P(1)');
      ctx.button('H → H', 'H → H', () => { angle.value = '0'; render(); });
      ctx.button('H → Z → H', 'H → Z → H', () => { angle.value = '180'; render(); });
      function render() {
        const a = n(angle), r = phase(a * Math.PI / 180);
        p0.textContent = ctx.format(r.p0, 4); p1.textContent = ctx.format(r.p1, 4);
        replace(ctx, [ctx.plot({ x: [-0.6, 1.6], y: [0, 1.05], xLabel: ctx.text('Outcome', '측정 결과'), yLabel: ctx.text('Probability', '확률'), title: ctx.text('Before and after the final H', '마지막 H 전후의 측정 확률'), xTicks: [0, 1],
          bars: [0, 1].flatMap((v, i) => [{ x: i - 0.16, y: 0.5, width: 0.25, color: 'muted', label: ctx.text('Before H', 'H 이전') },
            { x: i + 0.16, y: i ? r.p1 : r.p0, width: 0.25, color: 'blue', label: ctx.text('After H', 'H 이후') }]) }),
          line(ctx, `φ = ${a}° · |ψ⟩ = (|0⟩ + exp(iφ)|1⟩)/√2. Grey: before H; blue: after H.`, `φ = ${a}° · |ψ⟩ = (|0⟩ + exp(iφ)|1⟩)/√2. 회색은 H 이전, 파랑은 H 이후입니다.`)]);
        ctx.status.textContent = ctx.text('Measuring before the final H would give 50:50. Applying H instead reveals relative phase in the outcome probabilities.', '마지막 H 전에 측정한다면 50:50입니다. 그때 측정하는 대신 H를 적용하면 상대 phase가 결과 확률에 나타납니다.');
      }
      ctx.onChange(render);
    });
    labs.register('quantum-cleanup', ctx => {
      const stage = ctx.control('Circuit step', '회로 단계', { type: 'range', min: 0, max: 5, step: 1, value: 5 });
      const clean = ctx.control('Uncompute with the second CX', '두 번째 CX로 uncompute', { type: 'checkbox', value: true });
      const phaseOn = ctx.control('Apply Z to scratch', 'scratch에 Z 적용', { type: 'checkbox', value: true });
      const out = ctx.metric('Data qubit: P(1)', 'data qubit의 P(1)'), scratch = ctx.metric('Scratch: P(1)', 'scratch의 P(1)');
      const steps = ['|00⟩', 'H(data)', 'CX(data → scratch)', 'Z(scratch)', 'CX(data → scratch)', 'H(data)'];
      function render() {
        const r = cleanup(n(stage), clean.checked, phaseOn.checked);
        out.textContent = ctx.format(r.p1, 4); scratch.textContent = ctx.format(r.scratch1, 4);
        replace(ctx, [line(ctx, steps.slice(0, n(stage) + 1).map((s, i) => i === 4 && !clean.checked ? 'skip CX' : i === 3 && !phaseOn.checked ? 'skip Z' : s).join(' → '),
          steps.slice(0, n(stage) + 1).map((s, i) => i === 4 && !clean.checked ? 'CX 생략' : i === 3 && !phaseOn.checked ? 'Z 생략' : s).join(' → ')),
        ctx.plot({ x: [-0.6, 3.6], y: [-1.05, 1.05], xLabel: ctx.text('Joint basis: data, scratch (00, 01, 10, 11)', '공동 basis: data, scratch (00, 01, 10, 11)'), yLabel: ctx.text('Real amplitude', '실수 amplitude'), title: ctx.text('Signed amplitudes of the full two-qubit state', '두 qubit 전체 상태의 부호 있는 amplitude'), xTicks: [0, 1, 2, 3], bars: r.a.map((v, i) => ({ x: i, y: v, width: 0.5, color: v < 0 ? 'gold' : 'blue', label: bit(i) })) }),
        ctx.table([ctx.text('Basis', 'basis'), ctx.text('Amplitude', 'amplitude'), ctx.text('Probability', '확률')], r.a.map((a, i) => ['|' + bit(i) + '⟩', ctx.format(a, 4), ctx.format(r.probabilities[i], 4)]))]);
        ctx.status.textContent = n(stage) === 5 ? ctx.text(clean.checked ? 'Clean scratch lets phase become a definite interference outcome.' : 'Dirty scratch retains branch information: the data measurement is 50:50.', clean.checked ? 'scratch를 비우면 phase가 확정적인 interference 결과로 바뀝니다.' : 'scratch에 branch 정보가 남아 data 측정은 50:50이 됩니다.') : ctx.text(`Step ${n(stage)} of 5. Basis order is data then scratch.`, `5단계 중 ${n(stage)}단계입니다. basis의 순서는 data, scratch입니다.`);
      }
      ctx.onChange(render);
    });
    labs.register('quantum-fixedpoint', ctx => {
      const x = ctx.control('Integer x', '정수 x', { type: 'number', min: 1, max: 3133452, step: 1, value: 64 });
      const error = ctx.control('Initial relative error (×10⁻⁴)', '초기 상대 오차 (×10⁻⁴)', { type: 'range', min: -100, max: 100, step: 1, value: 1 });
      const bits = ctx.control('Fractional bits F', '소수부 비트 F', { type: 'range', min: 8, max: 28, step: 1, value: 17 });
      const m = ctx.metric('Normalized M / shift s', '정규화 M / 이동 s'), target = ctx.metric('Ideal phase (turns)', '이상적인 phase (turns)'), err = ctx.metric('Rounded Newton relative error', 'Newton 후 버림한 상대 오차');
      function render() {
        if (!Number.isInteger(n(x)) || n(x) < 1 || n(x) > 3133452) { ctx.status.textContent = ctx.text('Enter an integer from 1 to 3,133,452.', '1부터 3,133,452 사이의 정수를 입력하세요.'); return; }
        const r = fixedpoint(n(x), n(error) * 1e-4, n(bits));
        m.textContent = `${ctx.format(r.m, 6)} / ${r.s}`; target.textContent = ctx.format(r.target, 8); err.textContent = r.errors[3].toExponential(3);
        replace(ctx, [line(ctx, `${r.bits} → left shift ${2 * r.s} bits → M = ${r.m}. Restore scale with 2^(${r.s}−10).`, `${r.bits} → ${2 * r.s}비트 왼쪽 이동 → M = ${r.m}. 2^(${r.s}−10)을 곱해 크기를 복원합니다.`),
          ctx.table([ctx.text('Stage', '단계'), ctx.text('Value on normalized interval', '정규화 구간의 값'), ctx.text('Relative error', '상대 오차')], [
            [ctx.text('Chosen initial estimate', '가정한 초기값'), ctx.format(r.exact * (1 + n(error) * 1e-4), 9), r.errors[0].toExponential(3)],
            [`floor(y₀·2^${n(bits)}) / 2^${n(bits)}`, ctx.format(r.y0, 9), r.errors[1].toExponential(3)],
            ['y₀(3 − My₀²)/2', ctx.format(r.next, 9), r.errors[2].toExponential(3)],
            [ctx.text('Floor once after Newton', 'Newton 후 한 번 버림'), ctx.format(r.rounded, 9), r.errors[3].toExponential(3)]])]);
        ctx.status.textContent = ctx.text('More bits lower the rounding floor. This model floors y₀ and the final Newton value; intermediate arithmetic is exact.', '비트를 늘리면 버림 오차의 바닥이 낮아집니다. 이 모형은 y₀와 최종 Newton 값만 버림하고 중간 연산은 정확하게 계산합니다.');
      }
      ctx.onChange(render);
    });
    labs.register('quantum-gf4', ctx => {
      const position = ctx.control('Single error position', '단일 오류 위치', { type: 'select', value: '1', options: [0, 1, 2].map(v => ({ value: String(v), en: String(v), ko: String(v) })) });
      const value = ctx.control('Nonzero error value', '0이 아닌 오류 값', { type: 'select', value: '3', options: [1, 2, 3].map(v => ({ value: String(v), en: `${field[v]} (${bit(v)})`, ko: `${field[v]} (${bit(v)})` })) });
      const mask = ctx.control('Classical mask at that position', '해당 위치의 classical mask', { type: 'select', value: '2', options: [0, 1, 2, 3].map(v => ({ value: String(v), en: bit(v), ko: bit(v) })) });
      const s = ctx.metric('Received syndromes (S₀, S₁)', '입력 syndrome (S₀, S₁)'), sign = ctx.metric('Required phase', '필요한 phase');
      function render() {
        const r = syndrome(n(position), n(value), n(mask));
        s.textContent = `(${field[r.s0]}, ${field[r.s1]})`; sign.textContent = r.sign > 0 ? '+1' : '−1';
        replace(ctx, [ctx.table([ctx.text('Position j', '위치 j'), ctx.text('Hidden eⱼ', '숨겨진 eⱼ'), 'eⱼ αʲ', 'eⱼ α²ʲ'], [0, 1, 2].map(j => [j, j === n(position) ? field[n(value)] : '0', j === n(position) ? field[r.s0] : '0', j === n(position) ? field[r.s1] : '0'])),
          line(ctx, `S₁/S₀ = ${field[gfDiv(r.s1, r.s0)]} = α^${r.position} → j = ${r.position}; eⱼ = S₀/αʲ = ${field[r.value]}.`, `S₁/S₀ = ${field[gfDiv(r.s1, r.s0)]} = α^${r.position} → j = ${r.position}; eⱼ = S₀/αʲ = ${field[r.value]}.`),
          line(ctx, `${bit(r.value)} AND ${bit(n(mask))} = ${bit(r.selected)} → parity ${r.parity} → phase ${r.sign > 0 ? '+1' : '−1'}. Scratch is cleared; syndromes are preserved.`, `${bit(r.value)} AND ${bit(n(mask))} = ${bit(r.selected)} → parity ${r.parity} → phase ${r.sign > 0 ? '+1' : '−1'}. scratch는 비우고 syndrome은 보존합니다.`)]);
        ctx.status.textContent = ctx.text('Field addition is XOR; multiplication reduces modulo u² + u + 1. Mask parity uses bits, not field multiplication.', '덧셈은 XOR이고 곱셈은 u² + u + 1로 나눈 나머지입니다. mask parity는 field 곱셈이 아닌 비트 연산입니다.');
      }
      ctx.onChange(render);
    });
    labs.register('quantum-givens', ctx => {
      const a = ctx.control('Angle on edge (0,1), degrees', 'edge (0,1)의 각도 (도)', { type: 'range', min: 0, max: 359, step: 1, value: 45 });
      const b = ctx.control('Angle on edge (0,2), degrees', 'edge (0,2)의 각도 (도)', { type: 'range', min: 0, max: 359, step: 1, value: 45 });
      const order = ctx.control('Rotation order', 'rotation 순서', { type: 'select', value: 'forward', options: [{ value: 'forward', en: '(0,1) → (0,2)', ko: '(0,1) → (0,2)' }, { value: 'reverse', en: '(0,2) → (0,1)', ko: '(0,2) → (0,1)' }] });
      const bits = ctx.control('Angle word bits', '각도 word 비트 수', { type: 'range', min: 2, max: 12, step: 1, value: 12 });
      const words = ctx.metric('Loaded words', '로딩한 word'), norm = ctx.metric('Vector length', '벡터 길이'), err = ctx.metric('Vector rounding error ‖Δu‖₂', '벡터 양자화 오차 ‖Δu‖₂');
      function render() {
        const r = givens(n(a) * Math.PI / 180, n(b) * Math.PI / 180, order.value === 'reverse', n(bits));
        words.textContent = r.words.join(', '); norm.textContent = ctx.format(r.norm, 6); err.textContent = r.error.toExponential(3);
        replace(ctx, [ctx.plot({ x: [-0.6, 2.6], y: [-1.1, 1.1], xTicks: [0, 1, 2], xLabel: ctx.text('Vector component', '벡터 성분'), yLabel: ctx.text('Signed value', '부호 있는 값'), title: ctx.text('Blue: ideal · gold: quantized · grey: opposite order', '파랑: 이상적 각도 · 금색: 양자화 · 회색: 반대 순서'), bars: r.ideal.flatMap((v, i) => [{ x: i - 0.23, y: v, width: 0.2, color: 'blue' }, { x: i, y: r.actual[i], width: 0.2, color: 'gold' }, { x: i + 0.23, y: r.other[i], width: 0.2, color: 'muted' }]) }),
          line(ctx, `Blue: ideal angles; gold: quantized; grey: opposite order. Loaded angles: ${r.angles.map(t => ctx.format(t * 180 / Math.PI, 3) + '°').join(', ')}.`, `파랑은 이상적 각도, 금색은 양자화한 각도, 회색은 반대 순서입니다. 로딩한 각도: ${r.angles.map(t => ctx.format(t * 180 / Math.PI, 3) + '°').join(', ')}.`)]);
        ctx.status.textContent = ctx.text('Both operations preserve length. Changing their order changes which component receives the second rotation.', '두 연산 모두 길이를 보존하지만, 순서를 바꾸면 두 번째 rotation이 섞는 성분이 달라집니다.');
      }
      ctx.onChange(render);
    });
    labs.register('quantum-stages', async ctx => {
      const counts = await ctx.fetchJSON('/assets/img/posts/quantum-circuit-compilation/invsqrt-stage-counts.json');
      const stage = ctx.control('Stage to improve', '개선할 단계', { type: 'select', value: '2', options: counts.stages.map((v, i) => ({ value: String(i), en: v, ko: ['정규화', '구간 선택', 'Horner 곱셈 3회', 'y₀ 제곱', 'M × y₀²', '3 − z', '최종 곱셈', '비정규화'][i] })) });
      const saving = ctx.control('Assumed stage reduction (%)', '가정한 단계 절감률 (%)', { type: 'range', min: 0, max: 100, step: 5, value: 25 });
      const removed = ctx.metric('Static Toffolis removed', '줄어드는 정적 Toffoli'), remaining = ctx.metric('Remaining static Toffolis', '남는 정적 Toffoli');
      function render() {
        const index = n(stage), fraction = n(saving) / 100;
        const reduction = Math.floor(counts.forward[index] * fraction) + Math.floor(counts.reverse[index] * fraction);
        removed.textContent = ctx.format(reduction, 0); remaining.textContent = ctx.format(counts.total - reduction, 0);
        replace(ctx, [ctx.plot({ x: [-0.7, 7.7], y: [0, 2300], xTicks: [0, 1, 2, 3, 4, 5, 6, 7], xLabel: ctx.text('Stage index (0–7)', '단계 번호 (0–7)'), yLabel: ctx.text('Static Toffolis', '정적 Toffoli 수'), title: ctx.text('Grey: reference · blue: assumed edited stage', '회색: 기준 회로 · 파랑: 절감을 가정한 단계'), bars: counts.stages.flatMap((_, i) => [{ x: i - 0.15, y: counts.forward[i] + counts.reverse[i], width: 0.26, color: 'muted' }, { x: i + 0.15, y: counts.forward[i] + counts.reverse[i] - (i === index ? reduction : 0), width: 0.26, color: 'blue' }]) }),
          line(ctx, 'Grey: reference counts; blue: the assumed reduction at the selected stage. Other stages stay fixed.', '회색은 기준 수치이고, 파랑은 선택한 단계의 절감을 가정한 수치입니다. 다른 단계의 수치는 고정합니다.'),
          ctx.table([ctx.text('Stage', '단계'), ctx.text('Compute', 'compute'), ctx.text('Uncompute', 'uncompute')], counts.stages.map((v, i) => [`${i} · ${v}`, counts.forward[i], counts.reverse[i]]))]);
        ctx.status.textContent = ctx.text(`Reference total: ${counts.total}. The reduction is a hypothetical arithmetic edit; synthesis and qubit costs are separate.`, `기준 총합은 ${counts.total}개입니다. 절감률은 가정한 연산 변경이며 synthesis와 qubit 비용은 별도로 계산합니다.`);
      }
      ctx.onChange(render);
    });
    labs.register('quantum-cost', ctx => {
      const task = ctx.control('Cost model', '비용 모형', { type: 'select', value: 'dqi', options: [{ value: 'invsqrt', en: 'invsqrt: Q × C', ko: 'invsqrt: Q × C' }, { value: 'dqi', en: 'DQI: R × Q² × T', ko: 'DQI: R × Q² × T' }, { value: 'chem', en: 'Chemistry: λ × T × Q', ko: '분자: λ × T × Q' }] });
      const q = ctx.control('Q relative to reference (%)', '기준 대비 Q (%)', { type: 'range', min: 50, max: 150, step: 1, value: 90 });
      const gate = ctx.control('Gate cost relative to reference (%)', '기준 대비 gate 비용 (%)', { type: 'range', min: 25, max: 175, step: 1, value: 120 });
      const penalty = ctx.control('R or λ relative to reference (%)', '기준 대비 R 또는 λ (%)', { type: 'range', min: 75, max: 200, step: 1, value: 100 });
      const cost = ctx.metric('New / reference task cost', '새 비용 / 기준 task 비용'), score = ctx.metric('Overall score, other tasks at reference', '다른 task가 기준일 때 전체 점수');
      function render() {
        const r = resource(task.value, n(q) / 100, n(gate) / 100, n(penalty) / 100);
        penalty.disabled = task.value === 'invsqrt'; cost.textContent = ctx.format(r.ratio, 4); score.textContent = ctx.format(r.overall, 4);
        const points = Array.from({ length: 101 }, (_, i) => { const v = 50 + i; return [v, resource(task.value, v / 100, n(gate) / 100, n(penalty) / 100).ratio]; });
        const ymax = Math.max(1.2, ...points.map(p => p[1])) * 1.05;
        replace(ctx, [ctx.plot({ x: [50, 150], y: [0, ymax], xLabel: ctx.text('Q (% of reference)', 'Q (기준 대비 %)'), yLabel: ctx.text('Task cost ratio', 'task 비용 비율'), title: ctx.text('Cost tradeoff at the selected gate and penalty ratios', '선택한 gate·penalty 비율에서 Q와 비용의 관계'), series: [{ points, color: 'blue' }, { points: [[50, 1], [150, 1]], color: 'muted', dash: true }], markers: [{ x: n(q), y: r.ratio, color: 'gold', label: ctx.format(r.ratio, 3) }] }),
          line(ctx, task.value === 'invsqrt' ? 'C includes rotation synthesis; the penalty slider is unused.' : 'R or λ is an assumed multiplier. This calculator does not derive it from failures or molecular coefficients.', task.value === 'invsqrt' ? 'C에는 rotation synthesis가 포함됩니다. penalty 슬라이더는 사용하지 않습니다.' : 'R 또는 λ는 가정한 배율입니다. 실패 횟수나 분자 계수에서 배율을 추정하지 않습니다.')]);
        ctx.status.textContent = ctx.text(r.ratio < 1 ? 'This resource combination reduces cost, conditional on passing the task checks.' : 'This resource combination does not reduce task cost.', r.ratio < 1 ? 'task 검사를 통과한다는 조건에서 비용이 줄어드는 조합입니다.' : '이 자원 조합에서는 task 비용이 줄어들지 않습니다.');
      }
      ctx.onChange(render);
    });
  }
  return { phase, cleanup, normalize, fixedpoint, gfMul, gfPow, gfDiv, decode, syndrome, rotate, givens, resource, install };
});
