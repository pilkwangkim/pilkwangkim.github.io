/* Bounded Drude foundations: real-space Bloch factors and occupation redistribution. */
(function () {
  'use strict';
  const E = 1.602176634e-19;
  const inRange = (v, lo, hi) => Number.isFinite(v) && v >= lo && v <= hi;
  const grid = (v, lo, hi, step) => inRange(v, lo, hi) && Math.abs((v - lo) / step - Math.round((v - lo) / step)) < 1e-7;
  const complete = (s, keys) => Boolean(s) && typeof s === 'object' && !Array.isArray(s) && Object.keys(s).length === keys.length && keys.every(k => Object.prototype.hasOwnProperty.call(s, k));
  const clean = v => Math.abs(v) < 1e-14 ? 0 : v;
  function validBlochState(s) { return complete(s, ['mode', 'hopping']) && Number.isInteger(s.mode) && inRange(s.mode, 0, 3) && grid(s.hopping, .1, 3, .1); }
  const BLOCH_A_NM = .5;
  function validBlochProfile(s) { return complete(s, ['phase', 'modulation']) && grid(s.phase, -1, 1, .05) && grid(s.modulation, 0, 1, .05); }
  function blochProfile(xNM, s = { phase: .25, modulation: .75 }) {
    if (!inRange(xNM, -100, 100) || !validBlochProfile(s)) throw new Error('Invalid Bloch profile.');
    // phase=ka/pi; fix the physical lattice spacing, not a moving x scale.
    // This prescribed periodic profile is not a continuum-potential eigenproblem.
    const kPerNM = s.phase * Math.PI / BLOCH_A_NM;
    const u = 1 - s.modulation + s.modulation * Math.cos(Math.PI * xNM / BLOCH_A_NM) ** 8;
    const cosine = Math.cos(kPerNM * xNM), sine = Math.sin(kPerNM * xNM);
    return { u, re: u * cosine, im: u * sine, density: u * u, cosine, sine, kPerNM };
  }
  function validOccupationState(s) { return complete(s, ['delta', 'speed']) && grid(s.delta, -.5, .5, .01) && grid(s.speed, .1, 2, .1); }
  function blochModel(s = { mode: 1, hopping: 1 }) {
    if (!validBlochState(s)) throw new Error('Invalid Bloch settings.');
    const ka = s.mode * Math.PI / 2, energyEV = 2 * s.hopping * (1 - Math.cos(ka));
    const coefficients = Array.from({ length: 4 }, (_, j) => ({ re: clean(Math.cos(ka * j) / 2), im: clean(Math.sin(ka * j) / 2) }));
    // Apply the periodic Hamiltonian independently of the eigenvalue formula.
    const hc = coefficients.map((c, j) => ({
      re: s.hopping * (2 * c.re - coefficients[(j + 3) % 4].re - coefficients[(j + 1) % 4].re),
      im: s.hopping * (2 * c.im - coefficients[(j + 3) % 4].im - coefficients[(j + 1) % 4].im)
    }));
    const probabilities = coefficients.map(c => c.re ** 2 + c.im ** 2);
    return { ka, energyEV: clean(energyEV), coefficients, probabilities, norm: probabilities.reduce((a, b) => a + b, 0),
      residualEV: Math.max(...hc.map((c, j) => Math.hypot(c.re - energyEV * coefficients[j].re, c.im - energyEV * coefficients[j].im))) };
  }
  function occupationModel(s = { delta: .1, speed: 1 }) {
    if (!validOccupationState(s)) throw new Error('Invalid occupation settings.');
    const fPlus = .5 - s.delta, fMinus = .5 + s.delta, velocity = s.speed * 1e5;
    // Each group has g available single-spin states per volume. n=g because
    // f+ + f-=1 here, not because occupations generally sum to one.
    const g = 1e23, density = g * (fPlus + fMinus), velocitySum = velocity * (fPlus - fMinus);
    const currentDensity = -E * g * velocitySum;
    return { fPlus, fMinus, density, densityOverG: density / g, velocity, meanVelocity: velocitySum / (fPlus + fMinus),
      currentDensity, currentNormalized: currentDensity / (E * g * 1e5), currentScale: E * g * 1e5 };
  }
  const api = { E, BLOCH_A_NM, blochModel, blochProfile, occupationModel, validBlochState, validBlochProfile, validOccupationState };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const labs = window.ArticleLabs;
  const setup = ctx => ctx.root.classList.add('drude-lab', 'al-workshop');
  const note = (ctx, en, ko) => ctx.el('p', { class: 'drude-explanation' }, ctx.text(en, ko));
  const control = (ctx, en, ko, min, max, step, value) => ctx.control(en, ko, { type: 'range', min, max, step, value });
  const line = (points, color, label, dash = false) => ({ points, color, label, dash });
  function share(ctx, id, read, valid, restore, render) {
    ctx.stateLink(id, { read, write: s => { if (!valid(s)) return false; restore(s); render(); return true; } });
  }
  labs.register('drude-bloch', ctx => {
    setup(ctx);
    const phase = control(ctx, 'Phase increment ka / π (a = 0.5 nm)', '격자 사이 phase 차이 ka / π (a = 0.5 nm)', -1, 1, .05, .25);
    const modulation = control(ctx, 'Cell-scale modulation η', '격자 안에서의 진폭 변화 η', 0, 1, .05, .75);
    const wavevector = ctx.metric('Wavevector k (nm⁻¹)', 'Wavevector k (nm⁻¹)');
    const residual = ctx.metric('Bloch translation residual', 'Bloch translation 오차');
    const read = () => ({ phase: Number(phase.value), modulation: Number(modulation.value) });
    const restore = s => { phase.value = s.phase; modulation.value = s.modulation; };
    const points = fn => Array.from({ length: 561 }, (_, i) => { const x = -1.75 + 3.5 * i / 560; return [x, fn(x)]; });
    function render() {
      const s = read(), get = x => blochProfile(x, s), p = get(0);
      const c = Math.cos(s.phase * Math.PI), d = Math.sin(s.phase * Math.PI);
      const error = Math.max(...points(x => { const a = get(x), b = get(x + BLOCH_A_NM); return Math.hypot(b.re - c * a.re + d * a.im, b.im - d * a.re - c * a.im); }).map(q => q[1]));
      wavevector.textContent = p.kPerNM.toFixed(3); residual.textContent = error.toExponential(1);
      const domain = { x: [-1.75, 1.75], xTicks: [-1.5, -1, -.5, 0, .5, 1, 1.5], xLabel: 'x (nm); a = 0.5 nm' };
      ctx.view.replaceChildren(ctx.plot({ ...domain, y: [0, 1.1], yTicks: [0, .5, 1],
        title: ctx.text('1. Repeat the same shape at every atomic site', '1. 같은 모양이 격자 간격마다 반복됩니다'), yLabel: 'u(x) = 1 − η + η cos⁸(πx/a)',
        series: [line(points(x => get(x).u), 'blue', 'u(x)')],
        markers: Array.from({ length: 7 }, (_, j) => ({ x: (j - 3) * BLOCH_A_NM, y: 1, color: 'gold', label: 'site ' + (j - 3) })) }),
      ctx.plot({ ...domain, y: [-1.1, 1.1], yTicks: [-1, 0, 1], title: ctx.text('2. The phase factor varies across the chain', '2. 위치에 따라 달라지는 phase factor를 곱합니다'), yLabel: 'exp(ikx): real and imaginary parts',
        series: [line(points(x => get(x).cosine), 'blue', 'cos(kx)'), line(points(x => get(x).sine), 'gold', 'sin(kx)')] }),
      ctx.plot({ ...domain, y: [-1.1, 1.1], yTicks: [-1, 0, 1], title: ctx.text('3. Multiply: ψ(x) = exp(ikx)u(x)', '3. 두 함수를 곱하면 전체 ψ(x)가 됩니다'), yLabel: ctx.text('Relative amplitude', '상대 진폭'),
        series: [line(points(x => get(x).re), 'blue', 'Re ψ'), line(points(x => get(x).im), 'gold', 'Im ψ'), line(points(x => get(x).u), 'muted', '+u', true), line(points(x => -get(x).u), 'muted', '−u', true)] }),
      note(ctx, 'The site spacing a=0.5 nm and all axes stay fixed. Changing k changes the phase factor and the product; changing η changes the periodic profile. Since |exp(ikx)|=1, |ψ|²=u² at fixed η. These are relative amplitudes, not normalized probabilities. The prescribed profile isolates Bloch translation; actual atomic-potential eigenstates generally have k-dependent uₖ.',
        '격자 간격 a=0.5 nm와 모든 축 범위는 고정합니다. k를 바꾸면 phase factor와 전체 ψ가 달라지고, η를 바꾸면 주기적으로 반복되는 모양이 달라집니다. |exp(ikx)|=1이므로 η가 같으면 |ψ|²=u²입니다. 세로축은 상대 진폭이며 전체 확률을 1로 맞춘 값은 아닙니다. 여기서는 Bloch translation을 보기 위해 u를 직접 정했습니다. 실제 atomic potential의 eigenproblem에서 얻는 uₖ는 k에 따라 달라질 수 있습니다.'));
      ctx.status.textContent = ctx.text('Dashed ±u bounds the real and imaginary parts; it is not a localized wave-packet envelope. The functions obey ψ(x+a)=exp(ika)ψ(x).', '점선 ±u는 실수부와 허수부가 놓이는 범위입니다. 전자가 특정 위치에 모인 wave packet의 envelope는 아닙니다. 한 격자 간격을 옮기면 ψ(x+a)=exp(ika)ψ(x)가 성립합니다.');
    }
    ctx.button('Same phase: k = 0', 'Phase 차이가 없는 경우', () => { phase.value = 0; render(); });
    ctx.button('Slow phase change: ka = π/4', 'Phase가 천천히 바뀌는 경우', () => { phase.value = .25; render(); });
    ctx.button('Opposite signs at adjacent sites', '이웃한 격자점에서 부호가 반대인 경우', () => { phase.value = 1; render(); });
    ctx.onChange(render); share(ctx, 'drude-bloch', read, validBlochProfile, restore, render);
  });
  labs.register('drude-occupation', ctx => {
    setup(ctx);
    const delta = control(ctx, 'Redistribution δ (f± = 1/2 ∓ δ)', 'Occupation 변화 δ (f± = 1/2 ∓ δ)', -.5, .5, .01, .1);
    const speed = control(ctx, 'Speed v / vref (vref = 10⁵ m/s)', '속력 v / vref (vref = 10⁵ m/s)', .1, 2, .1, 1);
    const total = ctx.metric('Total density n / g', '전체 밀도 n / g'), current = ctx.metric('J / (e g vref)', 'J / (e g vref)'), drift = ctx.metric('Mean electron velocity / vref', '전자 평균 속도 / vref'), siCurrent = ctx.metric('J (10⁹ A/m²)', 'J (10⁹ A/m²)');
    const read = () => ({ delta: Number(delta.value), speed: Number(speed.value) });
    const restore = s => { delta.value = s.delta; speed.value = s.speed; };
    function render() {
      const s = read(), p = occupationModel(s);
      total.textContent = ctx.format(p.densityOverG, 3); current.textContent = ctx.format(p.currentNormalized, 3); drift.textContent = ctx.format(p.meanVelocity / 1e5, 3); siCurrent.textContent = ctx.format(p.currentDensity / 1e9, 3);
      const points = Array.from({ length: 101 }, (_, i) => { const d = -.5 + i * .01; return [d, occupationModel({ ...s, delta: d }).currentDensity / 1e9]; });
      ctx.view.replaceChildren(ctx.plot({ x: [-.22, .22], y: [0, 1], xTicks: [-.2, -.1, 0, .1, .2], yTicks: [0, .5, 1], title: ctx.text('Redistribution between two velocity groups', '두 velocity group 사이의 occupation 변화'),
        xLabel: 'Velocity (10⁶ m/s)', yLabel: 'Occupation f (0 ≤ f ≤ 1)',
        bars: [{ x: -p.velocity / 1e6, y: p.fMinus, width: .015, color: 'blue', label: `f−=${ctx.format(p.fMinus, 3)}` }, { x: p.velocity / 1e6, y: p.fPlus, width: .015, color: 'gold', label: `f+=${ctx.format(p.fPlus, 3)}` }],
        series: [line([[-.22, .5], [.22, .5]], 'muted', ctx.text('Equilibrium f0 = 1/2', 'Equilibrium f0 = 1/2'), true)] }),
      ctx.plot({ x: [-.5, .5], y: [-3.3, 3.3], xTicks: [-.25, 0, .25, .5], yTicks: [-3, 0, 3], title: ctx.text('Conventional current reverses the electron-velocity sign', 'Conventional current는 전자의 평균 속도와 반대 방향입니다'),
        xLabel: 'δ = f− − 1/2 = 1/2 − f+', yLabel: 'J (10⁹ A/m²)', series: [line(points, 'blue', 'J(δ)'), line([[-.5, 0], [.5, 0]], 'muted', 'J = 0', true)],
        markers: [{ x: s.delta, y: p.currentDensity / 1e9, color: 'blue', label: `δ=${ctx.format(s.delta, 2)}, J=${ctx.format(p.currentDensity / 1e9, 3)}×10⁹ A/m²` }] }),
      note(ctx, 'Each velocity group represents g=10²³ available single-spin states per m³. Its occupied density is g f. We prescribe f+=1/2−δ and f−=1/2+δ, keeping n=g(f++f−)=g fixed. Electron current is J=−eg[(+v)f++(−v)f−]=2egvδ. Occupation is not a one-electron probability distribution over these two groups. No electric field, scattering rate or conductivity is inferred from the chosen δ.', '속도가 각각 +v, −v인 두 group에 single-spin state가 단위 부피당 g=10²³ m⁻³개씩 있다고 가정합니다. 실제로 전자가 차지한 밀도는 각각 g f입니다. f+=1/2−δ, f−=1/2+δ로 바꾸므로 전체 밀도 n=g(f++f−)=g는 일정합니다. 전자의 전하가 −e이므로 J=−eg[(+v)f++(−v)f−]=2egvδ입니다. 여기서 f는 전자 하나가 두 group 중 어디에 있을 확률을 뜻하지 않습니다. δ는 직접 지정한 값이며, 이 값만으로 전기장이나 scattering rate, conductivity를 계산하지는 않습니다.'));
      ctx.status.textContent = ctx.text(`f−=${ctx.format(p.fMinus, 2)}, f+=${ctx.format(p.fPlus, 2)}; their sum stays 1. For δ>0, more electrons occupy −v, so the electron mean velocity is negative and conventional current is positive.`, `f−=${ctx.format(p.fMinus, 2)}, f+=${ctx.format(p.fPlus, 2)}이며 합은 항상 1입니다. δ>0이면 −v 쪽 전자가 늘어납니다. 따라서 전자의 평균 속도는 음수이고 conventional current는 양수입니다.`);
    }
    ctx.button('Equilibrium: no current', 'Equilibrium: 전류가 없는 경우', () => { delta.value = 0; render(); });
    ctx.button('More −v states occupied: J > 0', '−v 쪽 occupation 증가: J > 0', () => { delta.value = .1; render(); });
    ctx.button('More +v states occupied: J < 0', '+v 쪽 occupation 증가: J < 0', () => { delta.value = -.1; render(); });
    ctx.onChange(render); share(ctx, 'drude-occupation', read, validOccupationState, restore, render);
  });
}());
