/* Bounded Drude foundations: Bloch coefficients and occupation redistribution. */
(function () {
  'use strict';
  const E = 1.602176634e-19;
  const inRange = (v, lo, hi) => Number.isFinite(v) && v >= lo && v <= hi;
  const grid = (v, lo, hi, step) => inRange(v, lo, hi) && Math.abs((v - lo) / step - Math.round((v - lo) / step)) < 1e-7;
  const complete = (s, keys) => Boolean(s) && typeof s === 'object' && !Array.isArray(s) && Object.keys(s).length === keys.length && keys.every(k => Object.prototype.hasOwnProperty.call(s, k));
  const clean = v => Math.abs(v) < 1e-14 ? 0 : v;
  function validBlochState(s) { return complete(s, ['mode', 'hopping']) && Number.isInteger(s.mode) && inRange(s.mode, 0, 3) && grid(s.hopping, .1, 3, .1); }
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
  const api = { E, blochModel, occupationModel, validBlochState, validOccupationState };
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
    const mode = ctx.control('Allowed ka on the four-site ring', 'Four-site ring에서 가능한 ka', { type: 'select', value: '1', options: [
      { value: '0', en: '0: same phase', ko: '0: 모든 phase가 같음' }, { value: '1', en: 'π/2: quarter turn per site', ko: 'π/2: site마다 phase가 π/2씩 증가' },
      { value: '2', en: 'π: alternating signs', ko: 'π: 부호가 번갈아 바뀜' }, { value: '3', en: '3π/2: reverse quarter turn', ko: '3π/2: 반대 방향으로 π/2씩 증가' }
    ] });
    const hopping = control(ctx, 'Hopping tₕ (eV)', 'Hopping tₕ (eV)', .1, 3, .1, 1);
    const energy = ctx.metric('Eigenvalue ε (eV)', 'Eigenvalue ε (eV)'), norm = ctx.metric('Σj |cj|²', 'Σj |cj|²'), residual = ctx.metric('maxj |(Hc − εc)j| (eV)', 'maxj |(Hc − εc)j| (eV)');
    const read = () => ({ mode: Number(mode.value), hopping: Number(hopping.value) });
    const restore = s => { mode.value = s.mode; hopping.value = s.hopping; };
    function render() {
      const s = read(), p = blochModel(s);
      energy.textContent = ctx.format(p.energyEV, 3); norm.textContent = ctx.format(p.norm, 4); residual.textContent = p.residualEV.toExponential(1);
      const bars = p.coefficients.flatMap((c, j) => [{ x: j - .13, y: c.re, width: .22, color: 'blue', label: `Re c${j}=${ctx.format(c.re, 3)}` }, { x: j + .13, y: c.im, width: .22, color: 'gold', label: `Im c${j}=${ctx.format(c.im, 3)}` }]);
      ctx.view.replaceChildren(ctx.plot({ x: [-.5, 3.5], y: [-.6, .6], xTicks: [0, 1, 2, 3], yTicks: [-.5, 0, .5],
        title: ctx.text('Real and imaginary Bloch coefficients', 'Bloch coefficient의 실수부와 허수부'), xLabel: ctx.text('Basis site j (discrete labels)', 'Basis site j (site 번호)'), yLabel: ctx.text('cj = exp(i ka j) / 2 · Blue: Re · Gold: Im', 'cj = exp(i ka j) / 2 · Blue: Re · Gold: Im'), bars,
        series: [line([[-.5, 0], [3.5, 0]], 'muted', '', true)] }),
      ctx.plot({ x: [-.5, 3.5], y: [0, .3], xTicks: [0, 1, 2, 3], yTicks: [0, .25], title: ctx.text('All four sites have equal probability', '어느 mode에서도 site별 probability는 같습니다'),
        xLabel: ctx.text('Basis site j', 'Basis site j'), yLabel: '|cj|² = 1/4', bars: p.probabilities.map((q, j) => ({ x: j, y: q, width: .48, color: 'blue', label: `|c${j}|²=${q}` })) }),
      note(ctx, 'The periodic Hamiltonian has diagonal 2tₕ and nearest-neighbour hopping −tₕ, including the 3↔0 bond. Periodicity requires exp(i4ka)=1, so only these four ka values are eigenstates. Hc=εc with ε=2tₕ(1−cos ka). The bars are complex basis coefficients, not electron trajectories. Equal site probabilities do not make the phase patterns equivalent.', 'Hamiltonian의 diagonal은 2tₕ이고, 이웃한 site 사이의 hopping은 −tₕ입니다. Ring이므로 site 3과 0도 연결됩니다. Periodic boundary condition exp(i4ka)=1을 만족하는 ka만 선택할 수 있습니다. 각 mode에서 Hc=εc이며 ε=2tₕ(1−cos ka)입니다. 막대는 complex basis coefficient를 나타내며 전자의 이동 경로가 아닙니다. Site별 probability가 같아도 phase pattern은 서로 다릅니다.'));
      ctx.status.textContent = ctx.text(`ka=${['0', 'π/2', 'π', '3π/2'][s.mode]}; exp(i4ka)=1. Changing tₕ scales the eigenvalue, not the normalized eigenvector. Modes π/2 and 3π/2 have the same energy but complex-conjugate coefficients.`, `ka=${['0', 'π/2', 'π', '3π/2'][s.mode]}이며 exp(i4ka)=1입니다. tₕ를 바꾸면 eigenvalue만 달라지고 normalized eigenvector는 그대로입니다. π/2와 3π/2 mode는 에너지가 같고 coefficient는 서로 complex conjugate입니다.`);
    }
    ctx.button('Same phase: ka = 0', '모든 phase가 같은 mode: ka = 0', () => { mode.value = 0; render(); });
    ctx.button('Compare conjugate modes', 'Complex conjugate mode끼리 비교', () => { mode.value = Number(mode.value) === 1 ? 3 : 1; render(); });
    ctx.button('Alternating signs: ka = π', '부호가 번갈아 바뀌는 mode: ka = π', () => { mode.value = 2; render(); });
    ctx.onChange(render); share(ctx, 'drude-bloch', read, validBlochState, restore, render);
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
      const points = Array.from({ length: 101 }, (_, i) => { const d = -.5 + i * .01; return [d, occupationModel({ ...s, delta: d }).currentNormalized]; });
      ctx.view.replaceChildren(ctx.plot({ x: [-1.6, 1.6], y: [0, 1], xTicks: [-1, 0, 1], yTicks: [0, .5, 1], title: ctx.text('Redistribution between two velocity groups', '두 velocity group 사이의 occupation 변화'),
        xLabel: ctx.text('Velocity / v: −1 is −v; +1 is +v', 'Velocity / v: −1은 −v, +1은 +v'), yLabel: 'Occupation f (0 ≤ f ≤ 1)',
        bars: [{ x: -1, y: p.fMinus, width: .55, color: 'blue', label: `f−=${ctx.format(p.fMinus, 3)}` }, { x: 1, y: p.fPlus, width: .55, color: 'gold', label: `f+=${ctx.format(p.fPlus, 3)}` }],
        series: [line([[-1.6, .5], [1.6, .5]], 'muted', ctx.text('Equilibrium f0 = 1/2', 'Equilibrium f0 = 1/2'), true)] }),
      ctx.plot({ x: [-.5, .5], y: [-2, 2], xTicks: [-.25, 0, .25, .5], yTicks: [-2, -1, 0, 1, 2], title: ctx.text('Conventional current reverses the electron-velocity sign', 'Conventional current는 전자의 평균 속도와 반대 방향입니다'),
        xLabel: 'δ = f− − 1/2 = 1/2 − f+', yLabel: 'J / (e g vref) = 2δ(v/vref)', series: [line(points, 'blue', 'J(δ)'), line([[-.5, 0], [.5, 0]], 'muted', 'J = 0', true)],
        markers: [{ x: s.delta, y: p.currentNormalized, color: 'blue', label: `δ=${ctx.format(s.delta, 2)}, J/(e g vref)=${ctx.format(p.currentNormalized, 3)}` }] }),
      note(ctx, 'Each velocity group represents g=10²³ available single-spin states per m³. Its occupied density is g f. We prescribe f+=1/2−δ and f−=1/2+δ, keeping n=g(f++f−)=g fixed. Electron current is J=−eg[(+v)f++(−v)f−]=2egvδ. Occupation is not a one-electron probability distribution over these two groups. No electric field, scattering rate or conductivity is inferred from the chosen δ.', '속도가 각각 +v, −v인 두 group에 single-spin state가 단위 부피당 g=10²³ m⁻³개씩 있다고 가정합니다. 실제로 전자가 차지한 밀도는 각각 g f입니다. f+=1/2−δ, f−=1/2+δ로 바꾸므로 전체 밀도 n=g(f++f−)=g는 일정합니다. 전자의 전하가 −e이므로 J=−eg[(+v)f++(−v)f−]=2egvδ입니다. 여기서 f는 전자 하나가 두 group 중 어디에 있을 확률을 뜻하지 않습니다. δ는 직접 지정한 값이며, 이 값만으로 전기장이나 scattering rate, conductivity를 계산하지는 않습니다.'));
      ctx.status.textContent = ctx.text(`f−=${ctx.format(p.fMinus, 2)}, f+=${ctx.format(p.fPlus, 2)}; their sum stays 1. For δ>0, more electrons occupy −v, so the electron mean velocity is negative and conventional current is positive.`, `f−=${ctx.format(p.fMinus, 2)}, f+=${ctx.format(p.fPlus, 2)}이며 합은 항상 1입니다. δ>0이면 −v 쪽 전자가 늘어납니다. 따라서 전자의 평균 속도는 음수이고 conventional current는 양수입니다.`);
    }
    ctx.button('Equilibrium: no current', 'Equilibrium: 전류가 없는 경우', () => { delta.value = 0; render(); });
    ctx.button('More −v states occupied: J > 0', '−v 쪽 occupation 증가: J > 0', () => { delta.value = .1; render(); });
    ctx.button('More +v states occupied: J < 0', '+v 쪽 occupation 증가: J < 0', () => { delta.value = -.1; render(); });
    ctx.onChange(render); share(ctx, 'drude-occupation', read, validOccupationState, restore, render);
  });
}());
