/* Bounded Drude teaching models; SI internally, signed conventional current. */
(function () {
  'use strict';
  const E = 1.602176634e-19, M0 = 9.1093837139e-31, HBAR = 6.62607015e-34 / (2 * Math.PI);
  const inRange = (x, lo, hi) => Number.isFinite(x) && x >= lo && x <= hi;
  const grid = (x, lo, hi, step) => inRange(x, lo, hi) && Math.abs((x - lo) / step - Math.round((x - lo) / step)) < 1e-7;
  function stepResponse({ density = 1e23, massRatio = .2, tauFs = 100, field = 1000, s = 2 } = {}) {
    if (!inRange(density, 1e20, 1e28) || !inRange(massRatio, .01, 10) || !inRange(tauFs, 1, 1e4) || !inRange(field, -1e6, 1e6) || !inRange(s, 0, 100)) throw new Error('Invalid Drude settings.');
    const mobility = E * tauFs * 1e-15 / (massRatio * M0), conductivity = E * density * mobility;
    const drift = -mobility * field * -Math.expm1(-s);
    return { mobility, conductivity, drift, currentDensity: -E * density * drift, steadyCurrentDensity: conductivity * field, currentNA: -E * density * drift * 1e-14 * 1e9, voltage: field * 1e-6 };
  }
  function acResponse(omegaTau, phase = 0) {
    if (!inRange(omegaTau, 0, 100) || !inRange(phase, -100, 100)) throw new Error('Invalid AC settings.');
    const real = 1 / (1 + omegaTau ** 2), imaginary = omegaTau * real;
    // Re[(real+i imaginary)exp(-i phase)] = real*cos + imaginary*sin.
    return { real, imaginary, magnitude: 1 / Math.hypot(1, omegaTau), lag: Math.atan(omegaTau), field: Math.cos(phase), current: real * Math.cos(phase) + imaginary * Math.sin(phase) };
  }
  function tightBinding(ka, hoppingEV = 1, latticeNM = .5) {
    if (!inRange(ka, -Math.PI, Math.PI) || !inRange(hoppingEV, .1, 3) || !inRange(latticeNM, .1, 1)) throw new Error('Invalid tight-binding settings.');
    const t = hoppingEV * E, a = latticeNM * 1e-9;
    return { energyEV: 4 * hoppingEV * Math.sin(ka / 2) ** 2, velocity: 2 * t * a / HBAR * Math.sin(ka), inverseMassM0: 2 * t * a ** 2 / HBAR ** 2 * M0 * Math.cos(ka), parabolicEV: hoppingEV * ka ** 2 };
  }
  function screeningIntegral(x) {
    if (!inRange(x, 0, 1e6)) throw new Error('Invalid screening shape.');
    if (x < 1e-3) { let sum = 0; for (let j = 0; j < 9; j++) sum += (-1) ** j * (j + 1) / (j + 2) * x ** j; return sum; }
    return (Math.log1p(x) - x / (1 + x)) / x ** 2;
  }
  function scatteringShape(x, theta = 0) {
    if (!inRange(x, 0, 1e6) || !inRange(theta, 0, Math.PI)) throw new Error('Invalid angular scattering settings.');
    const pdf = .5 * Math.sin(theta) * (1 + x) / (1 + x * Math.sin(theta / 2) ** 2) ** 2;
    const meanLoss = 2 * (1 + x) * screeningIntegral(x);
    return { pdf, transportWeighted: pdf * (1 - Math.cos(theta)), meanLoss, ratio: 1 / meanLoss };
  }
  function chargeMode(s, omegaTau) {
    if (!inRange(s, 0, 100) || !inRange(omegaTau, 0, 10)) throw new Error('Invalid charge-mode settings.');
    let a, derivative, frequency = 0, regime;
    if (omegaTau > .5) {
      frequency = Math.sqrt(omegaTau ** 2 - .25);
      const z = frequency * s, sinc = Math.abs(z) < 1e-4 ? 1 - z ** 2 / 6 + z ** 4 / 120 : Math.sin(z) / z;
      a = Math.exp(-s / 2) * (Math.cos(z) + .5 * s * sinc);
      derivative = -(omegaTau ** 2) * s * Math.exp(-s / 2) * sinc; regime = 'underdamped';
    } else if (omegaTau === .5) {
      a = Math.exp(-s / 2) * (1 + s / 2); derivative = -s / 4 * Math.exp(-s / 2); regime = 'critical';
    } else {
      const b = Math.sqrt(.25 - omegaTau ** 2), slow = -(omegaTau ** 2) / (.5 + b);
      const divided = -Math.expm1(-2 * b * s) / (2 * b);
      a = Math.exp(slow * s) * (1 - slow * divided); derivative = -(omegaTau ** 2) * Math.exp(slow * s) * divided; regime = 'overdamped';
    }
    return { a, derivative, secondDerivative: -derivative - omegaTau ** 2 * a, frequency, regime };
  }
  function chargeProfile(xOverL, s, omegaTau) {
    if (!inRange(xOverL, 0, 1)) throw new Error('Invalid periodic position.');
    const m = chargeMode(s, omegaTau), sn = Math.sin(2 * Math.PI * xOverL), c = Math.cos(2 * Math.PI * xOverL);
    return { densityFraction: .05 * m.a * c, field: -m.a * sn, current: m.derivative * sn, continuityResidual: -m.derivative * c + m.derivative * c };
  }
  const questions = [
    { answer: 1, en: 'A charge-neutral uniform region has ρ=0. Must E be zero?', ko: '균일한 영역의 bulk charge density가 ρ=0입니다. E도 반드시 0일까요?', choices: [['Yes, zero charge forces zero field.', '네. 전하가 없으면 전기장도 없습니다.'], ['No. Poisson fixes the divergence; boundaries may impose a uniform field.', '아니요. Poisson equation은 divergence를 정합니다. 경계 조건에 따라 균일한 전기장이 생길 수 있습니다.'], ['No, but any uniform field violates Poisson.', '아니요. 다만 균일한 전기장은 Poisson equation을 위반합니다.']], explanation: ['In 1D, dE/dx=ρ/ε=0 permits any constant E. Boundary potentials fix that constant; neutrality alone does not.', '1D에서 dE/dx=ρ/ε=0이면 E는 상수입니다. 그 값은 경계에서 지정한 potential로 정해집니다. ρ=0이라는 조건만으로 E=0이 되지는 않습니다.'] },
    { answer: 2, en: 'Near ka=π, the tight-binding band bends downward. What does this establish?', ko: 'Tight-binding band가 ka=π 부근에서 아래로 휘어집니다. 이 사실로 알 수 있는 것은 무엇일까요?', choices: [['Every filled band has negative DC conductivity.', '전자들로 채워진 band는 모두 DC conductivity가 음수입니다.'], ['The group velocity is negative everywhere.', '모든 k에서 group velocity가 음수입니다.'], ['The local inverse curvature mass is negative; a DC conductivity needs a distribution and collision model.', '이 지점의 inverse curvature mass가 음수입니다. DC conductivity를 구하려면 전자의 distribution과 collision model도 알아야 합니다.']], explanation: ['v=(1/ℏ)dε/dk and m_curv⁻¹=(1/ℏ²)d²ε/dk² are different derivatives. Local negative curvature is not a negative scalar DC conductivity.', 'v=(1/ℏ)dε/dk와 m_curv⁻¹=(1/ℏ²)d²ε/dk²는 서로 다른 미분입니다. 특정 지점의 curvature가 음수라고 해서 scalar DC conductivity도 음수인 것은 아닙니다.'] },
    { answer: 0, en: 'Keep the total scattering-event rate fixed. What happens when the 3D angular kernel becomes strongly forward-peaked?', ko: '전체 scattering event rate를 고정합니다. 3D angular kernel이 작은 각도에 강하게 집중되면 어떻게 될까요?', choices: [['Each event removes less directed momentum, so τtr grows relative to τq.', 'Event 하나가 원래 진행 방향의 momentum을 덜 줄이므로 τq에 비해 τtr이 길어집니다.'], ['τtr must equal τq for all angular distributions.', 'Angular distribution과 관계없이 τtr=τq입니다.'], ['Any scattering event completely reverses the electron.', 'Scattering이 일어나면 전자의 방향은 항상 반대로 바뀝니다.']], explanation: ['The transport weight is 1−cosθ. It is small for a small deflection. This is not a universal τtr≥τq theorem: pure reverse scattering has weight 2 and τtr=τq/2.', 'Transport weight는 1−cosθ입니다. 방향이 조금만 바뀌면 이 값도 작습니다. 다만 항상 τtr≥τq인 것은 아닙니다. 반대 방향으로만 scattering하면 weight가 2이므로 τtr=τq/2입니다.'] },
    { answer: 1, en: 'For E=E0 cosωt, which relaxation time sets the Drude phase lag arctan(ωτ)?', ko: 'E=E0 cosωt일 때 Drude 전류의 phase lag arctan(ωτ)를 정하는 τ는 무엇일까요?', choices: [['Only the time for electron energy to reach the lattice temperature.', '전자계가 lattice와 같은 온도가 되는 데 걸리는 시간입니다.'], ['Momentum relaxation time; elastic collisions can relax current without changing electron energy.', 'Momentum relaxation time입니다. Elastic collision도 전자의 에너지를 바꾸지 않으면서 전류를 줄일 수 있습니다.'], ['The current leads the field, so no relaxation time is needed.', '전류가 전기장보다 앞서므로 relaxation time이 필요하지 않습니다.']], explanation: ['With exp(−iωt), σ=σ0/(1−iωτtr), so J=|σ|E0 cos[ωt−arctan(ωτtr)]. Positive Imσ corresponds to a lag in this convention. Energy relaxation is a different process.', 'exp(−iωt) convention에서 σ=σ0/(1−iωτtr)이고 J=|σ|E0 cos[ωt−arctan(ωτtr)]입니다. 이 convention에서는 Imσ>0이면 전류가 전기장보다 늦습니다. Energy relaxation은 별개의 과정입니다.'] },
    { answer: 0, en: 'When is the scalar formula μ=eτtr/m* justified?', ko: 'Scalar 식 μ=eτtr/m*를 사용할 수 있는 조건은 무엇일까요?', choices: [['An isotropic constant parabolic mass with a consistent scalar momentum relaxation time.', '방향에 관계없이 일정한 parabolic mass를 쓰고, 그 모델에 맞는 scalar momentum relaxation time을 쓸 때입니다.'], ['Every anisotropic band at every carrier energy.', '모든 anisotropic band의 모든 에너지에서 성립합니다.'], ['Whenever the curvature vanishes.', 'Curvature가 0이기만 하면 됩니다.']], explanation: ['Anisotropic curvature gives an inverse-mass tensor. Energy-dependent scattering also requires the appropriate conductivity-weighted average, rather than an arbitrary lifetime.', 'Curvature가 방향에 따라 다르면 inverse-mass tensor가 필요합니다. Scattering이 에너지에 따라 달라지면 임의의 lifetime이 아니라 conductivity 계산에 맞게 가중 평균한 값을 써야 합니다.'] }
  ];
  function quizProblem(index) { if (!Number.isInteger(index) || index < 0 || index >= questions.length) throw new Error('Invalid question.'); return questions[index]; }
  const api = { E, M0, HBAR, stepResponse, acResponse, tightBinding, screeningIntegral, scatteringShape, chargeMode, chargeProfile, quizProblem };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const labs = window.ArticleLabs;
  const samples = (lo, hi, count, fn) => Array.from({ length: count }, (_, i) => { const x = lo + (hi - lo) * i / (count - 1); return [x, fn(x)]; });
  function setup(ctx) { ctx.root.classList.add('drude-lab', 'al-workshop'); }
  function note(ctx, en, ko) { return ctx.el('p', { class: 'drude-explanation' }, ctx.text(en, ko)); }
  const control = (ctx, en, ko, min, max, step, value) => ctx.control(en, ko, { type: 'range', min, max, step, value });
  function share(ctx, id, read, valid, write, render) { ctx.stateLink(id, { read, write: s => { if (!s || typeof s !== 'object' || Array.isArray(s) || !valid(s)) return false; write(s); render(); return true; } }); }
  const series = (points, color, label, dash = false) => ({ points, color, label, dash });
  function positiveCeiling(value) {
    const scale = 10 ** Math.floor(Math.log10(value));
    return [1, 2, 5, 10].find(v => v * scale >= value) * scale;
  }
  labs.register('drude-step', ctx => {
    setup(ctx);
    const n = control(ctx, 'Density n (10²³ m⁻³)', '밀도 n (10²³ m⁻³)', .1, 10, .1, 1), mass = control(ctx, 'Mass m*/m0', '질량 m*/m0', .1, 2, .1, .2), tau = control(ctx, 'Momentum time τtr (fs)', 'Momentum relaxation time τtr (fs)', 10, 500, 10, 100), field = control(ctx, 'Electric field E (V/m)', '전기장 E (V/m)', -2000, 2000, 100, 1000), time = control(ctx, 'Time t/τtr', '시간 t/τtr', 0, 6, .1, 2);
    const mu = ctx.metric('Mobility (cm²/V s)', 'Mobility (cm²/V s)'), sigma = ctx.metric('Conductivity (S/m)', 'Conductivity (S/m)'), drift = ctx.metric('Electron drift u (m/s)', '전자 drift u (m/s)'), current = ctx.metric('Current I (nA)', '전류 I (nA)'), steady = ctx.metric('Steady J (A/m²)', '최종 current density J (A/m²)');
    const read = () => ({ n: Number(n.value), mass: Number(mass.value), tau: Number(tau.value), field: Number(field.value), time: Number(time.value) });
    function render() {
      const s = read(), p = { density: s.n * 1e23, massRatio: s.mass, tauFs: s.tau, field: s.field, s: s.time }, result = stepResponse(p);
      // Scale only the plotted values. Engineering powers and integer ticks
      // keep signed tick labels within the shared chart's fixed left margin.
      const exponent = result.steadyCurrentDensity === 0 ? 0 : 3 * Math.floor(Math.log10(Math.abs(result.steadyCurrentDensity) * 1.01) / 3);
      const scale = 10 ** exponent, steadyScaled = result.steadyCurrentDensity / scale, limit = Math.max(2, 2 * Math.ceil(Math.abs(steadyScaled) / 2));
      mu.textContent = ctx.format(result.mobility * 1e4, 2); sigma.textContent = ctx.format(result.conductivity, 2); drift.textContent = ctx.format(result.drift, 3); current.textContent = ctx.format(result.currentNA, 3); steady.textContent = ctx.format(result.steadyCurrentDensity, 2);
      ctx.view.replaceChildren(ctx.plot({ x: [0, 6], y: [-limit, limit], yTicks: [-limit, -limit / 2, 0, limit / 2, limit], title: ctx.text('Current builds after a field step', '전기장을 켜면 전류가 서서히 생깁니다'), xLabel: 's = t / τtr', yLabel: exponent === 0 ? 'J (A/m²)' : `J / (10^${exponent} A/m²)`, series: [series(samples(0, 6, 121, t => stepResponse({ ...p, s: t }).currentDensity / scale), 'blue', 'J(t)'), series([[0, steadyScaled], [6, steadyScaled]], 'gold', 'σ0 E', true)], markers: [{ x: s.time, y: result.currentDensity / scale, color: 'blue' }] }), note(ctx, 'Electrons have charge −e, so u points opposite to E while J=−en u points along E. The wire has L=1 μm and A=(100 nm)²; I=JA and V=EL. The field is prescribed, not solved from contacts.', '전자의 전하는 −e이므로 u는 E와 반대 방향이고, J=−en u는 E와 같은 방향입니다. L=1 μm, A=(100 nm)²인 도선에서 I=JA, V=EL로 계산합니다. 전기장은 contact 조건으로부터 계산하지 않고 직접 지정합니다.'));
      ctx.status.textContent = ctx.text(`J(t)=${ctx.format(result.currentDensity, 2)} A/m²; V=${ctx.format(result.voltage * 1e3, 2)} mV. τtr relaxes momentum, not energy.`, `J(t)=${ctx.format(result.currentDensity, 2)} A/m², V=${ctx.format(result.voltage * 1e3, 2)} mV입니다. τtr은 energy relaxation time이 아니라 momentum relaxation time입니다.`);
    }
    ctx.onChange(render); share(ctx, 'drude-step', read, s => grid(s.n, .1, 10, .1) && grid(s.mass, .1, 2, .1) && grid(s.tau, 10, 500, 10) && grid(s.field, -2000, 2000, 100) && grid(s.time, 0, 6, .1), s => { n.value = s.n; mass.value = s.mass; tau.value = s.tau; field.value = s.field; time.value = s.time; }, render);
  });
  labs.register('drude-ac', ctx => {
    setup(ctx);
    const omega = control(ctx, 'Frequency ωτtr', '주파수 ωτtr', 0, 5, .05, 1), phase = control(ctx, 'Phase ωt / π', '위상 ωt / π', 0, 4, .05, 0);
    const magnitude = ctx.metric('|σ| / σ0', '|σ| / σ0'), lag = ctx.metric('Current lag (degrees)', '전류의 phase lag (degree)'), re = ctx.metric('Re σ / σ0', 'Re σ / σ0'), im = ctx.metric('Im σ / σ0', 'Im σ / σ0');
    function render() {
      const w = Number(omega.value), t = Number(phase.value), p = acResponse(w, w === 0 ? 0 : t * Math.PI);
      phase.disabled = w === 0;
      magnitude.textContent = ctx.format(p.magnitude, 4); lag.textContent = ctx.format(p.lag * 180 / Math.PI, 2); re.textContent = ctx.format(p.real, 4); im.textContent = ctx.format(p.imaginary, 4);
      ctx.view.replaceChildren(ctx.plot({ x: [0, 5], y: [0, 1.05], yTicks: [0, .5, 1], title: ctx.text('Complex conductivity: exp(−iωt) convention', 'Complex conductivity: exp(−iωt) convention'), xLabel: 'ωτtr', yLabel: 'σ / σ0', series: [series(samples(0, 5, 151, x => acResponse(x).real), 'blue', 'Re σ / σ0'), series(samples(0, 5, 151, x => acResponse(x).imaginary), 'gold', 'Im σ / σ0')], markers: [{ x: w, y: p.real, color: 'blue' }, { x: w, y: p.imaginary, color: 'gold' }] }), ctx.plot({ x: [0, 4], y: [-1.1, 1.1], title: w === 0 ? ctx.text('Zero frequency: static field and current', '주파수가 0이면 전기장과 전류가 시간에 따라 변하지 않습니다') : ctx.text('Steady periodic current lags the field', '주기적으로 변하는 전류는 전기장보다 늦습니다'), xLabel: w === 0 ? ctx.text('Display coordinate (phase inactive)', '표시 좌표 (phase 사용 안 함)') : 'ωt / π', yLabel: 'E/E0; J/(σ0 E0)', series: [series(samples(0, 4, 201, x => w === 0 ? 1 : Math.cos(x * Math.PI)), 'muted', 'E/E0', true), series(samples(0, 4, 201, x => w === 0 ? 1 : acResponse(w, x * Math.PI).current), 'blue', 'J/(σ0 E0)')], markers: [{ x: w === 0 ? 0 : t, y: p.current, color: 'blue' }] }), note(ctx, 'J/(σ0E0)=Reσ/σ0 cosωt+Imσ/σ0 sinωt. With exp(−iωt), positive Imσ gives a time lag, not a lead. Reσ≥0 gives nonnegative cycle-averaged Joule power. At ωτ=0 the field and current are static, and the phase control is inactive.', 'J/(σ0E0)=Reσ/σ0 cosωt+Imσ/σ0 sinωt입니다. exp(−iωt) convention에서는 Imσ>0이면 전류가 전기장보다 늦습니다. Reσ≥0이므로 한 주기 평균 전력도 음수가 아닙니다. ωτ=0에서는 전기장과 전류가 시간에 따라 변하지 않으며 phase는 조절하지 않습니다.'));
      ctx.status.textContent = w === 0 ? ctx.text('Static E/E0=J/(σ0E0)=1. Switch-on transients are excluded; the phase setting is retained for nonzero frequency.', 'E/E0=J/(σ0E0)=1로 일정합니다. 전기장을 켠 직후의 transient는 포함하지 않습니다. 주파수를 다시 올리면 저장된 phase 설정을 사용합니다.') : ctx.text(`At ωt/π=${t.toFixed(2)}, normalized current=${p.current.toFixed(4)}. The waveform is the steady periodic solution; switch-on transients are excluded.`, `ωt/π=${t.toFixed(2)}에서 normalized 전류는 ${p.current.toFixed(4)}입니다. 주기적인 변화가 자리 잡은 뒤의 해이며, 전기장을 켠 직후의 transient는 포함하지 않습니다.`);
    }
    ctx.onChange(render); share(ctx, 'drude-ac', () => ({ omega: Number(omega.value), phase: Number(phase.value) }), s => grid(s.omega, 0, 5, .05) && grid(s.phase, 0, 4, .05), s => { omega.value = s.omega; phase.value = s.phase; }, render);
  });
  labs.register('drude-band', ctx => {
    setup(ctx);
    const k = control(ctx, 'Wave vector ka / π', 'Wave vector ka / π', -1, 1, .01, .16), hopping = control(ctx, 'Hopping tₕ (eV)', 'Hopping tₕ (eV)', .1, 3, .1, 1), lattice = control(ctx, 'Lattice spacing a (nm)', 'Lattice spacing a (nm)', .1, 1, .05, .5), parabola = ctx.control('Show band-bottom parabola', 'Band bottom 부근의 parabola와 비교', { type: 'checkbox', value: true });
    const energy = ctx.metric('Energy ε (eV)', '에너지 ε (eV)'), velocity = ctx.metric('Group velocity (m/s)', 'Group velocity (m/s)'), inverse = ctx.metric('m0 / mcurv', 'm0 / mcurv');
    function render() {
      const kp = Number(k.value), t = Number(hopping.value), a = Number(lattice.value), p = tightBinding(kp * Math.PI, t, a), scale = tightBinding(0, t, a).inverseMassM0, vmax = 2 * t * E * a * 1e-9 / HBAR;
      const displayedVelocity = Math.abs(kp) === 1 ? 0 : p.velocity, displayedInverse = Math.abs(kp) === .5 ? 0 : p.inverseMassM0;
      energy.textContent = ctx.format(p.energyEV, 4); velocity.textContent = ctx.format(displayedVelocity, 1); inverse.textContent = ctx.format(displayedInverse, 4);
      const bands = [series(samples(-1, 1, 201, x => tightBinding(x * Math.PI, t, a).energyEV), 'blue', '2tₕ(1−cos ka)')];
      if (parabola.checked) bands.push(series(samples(-.5, .5, 101, x => t * (x * Math.PI) ** 2), 'gold', 'tₕ(ka)²', true));
      ctx.view.replaceChildren(ctx.plot({ x: [-1, 1], y: [0, 4.2 * t], title: ctx.text('Tight-binding band and local parabolic approximation', 'Tight-binding band와 band bottom 부근의 parabola'), xLabel: 'ka / π', yLabel: 'ε (eV)', series: bands, markers: [{ x: kp, y: p.energyEV, color: 'blue' }] }), ctx.plot({ x: [-1, 1], y: [-1.1, 1.1], title: ctx.text('Velocity and curvature are different derivatives', '속도와 curvature는 서로 다른 미분으로 정해집니다'), xLabel: 'ka / π', yLabel: 'v/vmax; (m0/mcurv)/(m0/mbottom)', series: [series(samples(-1, 1, 201, x => Math.sin(x * Math.PI)), 'blue', 'v/vmax'), series(samples(-1, 1, 201, x => Math.cos(x * Math.PI)), 'gold', 'inverse curvature / bottom value')], markers: [{ x: kp, y: p.velocity / vmax, color: 'blue' }, { x: kp, y: p.inverseMassM0 / scale, color: 'gold' }] }), note(ctx, 'The readout is inverse mass m0/mcurv, not mass itself, so there is no artificial pole where curvature vanishes. The local curvature becomes negative near the band top. A conductivity also needs the occupied distribution and collision operator; negative curvature alone does not predict negative DC conductivity.', 'Curvature가 0인 지점에서 발산하는 질량 대신 inverse mass m0/mcurv를 표시합니다. Band top 부근에서는 curvature가 음수입니다. Conductivity를 구하려면 전자의 occupation distribution과 collision operator도 알아야 합니다. 특정 지점의 curvature가 음수라고 해서 DC conductivity도 음수인 것은 아닙니다.'));
      ctx.status.textContent = ctx.text(`vmax=${ctx.format(vmax, 1)} m/s; band-bottom m0/m*=${scale.toFixed(4)}. The parabola is a local approximation, not the full band.`, `vmax=${ctx.format(vmax, 1)} m/s, band bottom의 m0/m*=${scale.toFixed(4)}입니다. Parabola는 band 전체가 아니라 band bottom 부근에서만 쓸 수 있는 근사입니다.`);
    }
    ctx.onChange(render); share(ctx, 'drude-band', () => ({ k: Number(k.value), hopping: Number(hopping.value), lattice: Number(lattice.value), parabola: parabola.checked }), s => grid(s.k, -1, 1, .01) && grid(s.hopping, .1, 3, .1) && grid(s.lattice, .1, 1, .05) && typeof s.parabola === 'boolean', s => { k.value = s.k; hopping.value = s.hopping; lattice.value = s.lattice; parabola.checked = s.parabola; }, render);
  });
  labs.register('drude-scattering', ctx => {
    setup(ctx);
    const mode = ctx.control('Angular model', 'Scattering 각도 분포', { type: 'select', value: 'screened', options: [{ value: 'screened', en: '3D screened kernel', ko: '3D screened kernel' }, { value: 'isotropic', en: '3D isotropic benchmark', ko: '3D isotropic scattering' }, { value: 'reverse', en: 'Two-direction pure reversal', ko: '두 방향: 반대 방향으로만 전환' }] }), shape = control(ctx, 'Shape xₛ=4k²/qs² (fixed event rate)', 'Shape xₛ=4k²/qs² (전체 event rate 고정)', .01, 400, .01, 10);
    const ratio = ctx.metric('τtr / τq', 'τtr / τq'), loss = ctx.metric('Mean momentum loss ⟨1−cosθ⟩', 'Event당 평균 momentum loss ⟨1−cosθ⟩');
    function render() {
      const reverse = mode.value === 'reverse', x = mode.value === 'screened' ? Number(shape.value) : 0, p = reverse ? { ratio: .5, meanLoss: 2 } : scatteringShape(x);
      shape.disabled = mode.value !== 'screened'; ratio.textContent = ctx.format(p.ratio, 4); loss.textContent = ctx.format(p.meanLoss, 4);
      const probability = samples(0, Math.PI, 401, angle => scatteringShape(x, angle).pdf), weighted = samples(0, Math.PI, 401, angle => scatteringShape(x, angle).transportWeighted), ymax = positiveCeiling(Math.max(...probability.map(v => v[1]), ...weighted.map(v => v[1])) * 1.1);
      const chart = reverse ? ctx.plot({ x: [-.1, 1.1], y: [0, 2.2], xTicks: [0, 1], yTicks: [0, 1, 2], title: ctx.text('Discrete benchmark: every event reverses direction', '두 방향 모델: event마다 방향을 뒤집습니다'), xLabel: 'θ / π', yLabel: ctx.text('Probability; weighted probability', 'Probability; weighted probability'), bars: [{ x: .97, y: 1, width: .05, color: 'blue', label: 'P(θ=π)=1' }, { x: 1.03, y: 2, width: .05, color: 'gold', label: 'P(θ=π)(1−cosπ)=2' }] }) : ctx.plot({ x: [0, 1], y: [0, ymax], yTicks: [0, ymax / 2, ymax], title: ctx.text('Angular probability includes the 3D sinθ measure', '3D angular measure sinθ를 포함한 확률밀도'), xLabel: 'θ / π', yLabel: 'Probability density per radian', series: [series(probability.map(p => [p[0] / Math.PI, p[1]]), 'blue', 'pθ; integral over dθ = 1'), series(weighted.map(p => [p[0] / Math.PI, p[1]]), 'gold', 'pθ(1−cosθ)')] });
      ctx.view.replaceChildren(chart, note(ctx, 'For the 3D kernel, pθ=(1+xₛ)sinθ/[2(1+xₛ sin²(θ/2))²]. The blue density integrates to one with dθ (radians); the gold integral is ⟨1−cosθ⟩. The total single-particle event rate 1/τq is fixed while the kernel shape changes. This is not a screening-wavevector sweep at fixed potential strength.', '3D kernel의 확률밀도는 pθ=(1+xₛ)sinθ/[2(1+xₛ sin²(θ/2))²]입니다. 파란 곡선을 dθ(radian)로 적분하면 1이고, 금색 곡선의 적분은 ⟨1−cosθ⟩입니다. 전체 single-particle event rate 1/τq를 고정하고 kernel의 모양만 바꿉니다. Potential strength를 고정한 채 screening wavevector를 바꾸는 계산과는 다릅니다.'), note(ctx, 'Isotropic scattering gives τtr=τq. Forward-peaked scattering gives a longer τtr in this kernel family. The separate two-direction reversal benchmark gives τtr=τq/2: a reversal removes twice the initial directed momentum.', 'Isotropic scattering에서는 τtr=τq입니다. 이 kernel에서는 작은 각도로 scattering할수록 τtr이 길어집니다. 별도로 제시한 두 방향 reversal 모델에서는 τtr=τq/2입니다. 방향을 완전히 뒤집으면 원래 진행 방향의 momentum 성분이 처음 값의 두 배만큼 줄어듭니다.'));
      ctx.status.textContent = ctx.text(`Model: ${mode.value}; τtr/τq=${p.ratio.toFixed(4)}. The shape control is active only for the screened kernel.`, `Model: ${mode.value}, τtr/τq=${p.ratio.toFixed(4)}입니다. Shape는 screened kernel에서만 조절합니다.`);
    }
    ctx.onChange(render); share(ctx, 'drude-scattering', () => ({ mode: mode.value, shape: Number(shape.value) }), s => ['screened', 'isotropic', 'reverse'].includes(s.mode) && grid(s.shape, .01, 400, .01), s => { mode.value = s.mode; shape.value = s.shape; }, render);
  });
  labs.register('drude-charge', ctx => {
    setup(ctx);
    const omega = control(ctx, 'Plasma parameter Ω=ωpτtr', 'Plasma parameter Ω=ωpτtr', .1, 3, .05, 1), time = control(ctx, 'Time s=t/τtr', '시간 s=t/τtr', 0, 12, .1, 2);
    const regime = ctx.metric('Regime', '감쇠 유형'), amplitude = ctx.metric('Density amplitude a', '밀도 변화의 amplitude a'), derivative = ctx.metric('Current amplitude a′', '전류의 amplitude a′'), frequency = ctx.metric('Damped frequency × τtr', '감쇠 진동 주파수 × τtr'), residual = ctx.metric('Scaled continuity residual', 'Normalized continuity residual');
    function render() {
      const w = Number(omega.value), s = Number(time.value), p = chargeMode(s, w);
      regime.textContent = ctx.text(p.regime, ({ underdamped: '진동하며 감쇠', critical: 'Critical damping', overdamped: '진동 없이 감쇠' })[p.regime]); amplitude.textContent = p.a.toFixed(5); derivative.textContent = p.derivative.toFixed(5); frequency.textContent = p.frequency.toFixed(4); residual.textContent = chargeProfile(.23, s, w).continuityResidual.toExponential(1);
      ctx.view.replaceChildren(ctx.plot({ x: [0, 12], y: [-3.1, 3.1], title: ctx.text('Charge amplitude follows a damped oscillator', '밀도 변화의 amplitude는 damped oscillator와 같은 식을 따릅니다'), xLabel: 's=t/τtr', yLabel: 'a(s); a′(s)', series: [series(samples(0, 12, 301, t => chargeMode(t, w).a), 'blue', 'a(s)'), series(samples(0, 12, 301, t => chargeMode(t, w).derivative), 'gold', 'a′(s)')], markers: [{ x: s, y: p.a, color: 'blue' }, { x: s, y: p.derivative, color: 'gold' }] }), ctx.plot({ x: [0, 1], y: [-3.1, 3.1], title: ctx.text('One periodic wavelength: density, field and current', '한 파장에 걸친 밀도·전기장·전류'), xLabel: 'x/L; qL=2π', yLabel: 'δn/A0; E/Escale; J/Jscale', series: [series(samples(0, 1, 201, x => p.a * Math.cos(2 * Math.PI * x)), 'blue', 'δn/A0'), series(samples(0, 1, 201, x => chargeProfile(x, s, w).field), 'red', 'E/Escale'), series(samples(0, 1, 201, x => chargeProfile(x, s, w).current), 'gold', 'J/Jscale')] }), note(ctx, 'A uniform positive background cancels n0. The small perturbation is δn/n0=0.05 a(s)cos(qx), with a(0)=1 and a′(0)=0. Poisson and charge continuity give E/Escale=−a sin(qx), J/Jscale=a′ sin(qx), and a″+a′+Ω²a=0. Escale=eA0/(εq), Jscale=eA0/(qτtr). No contacts, pressure or diffusion are included.', '밀도 n0인 전자의 전하를 균일한 양전하 배경이 상쇄합니다. 작은 밀도 변화는 δn/n0=0.05 a(s)cos(qx)이며 a(0)=1, a′(0)=0입니다. Poisson equation과 charge continuity에 따라 E/Escale=−a sin(qx), J/Jscale=a′ sin(qx), a″+a′+Ω²a=0입니다. Escale=eA0/(εq), Jscale=eA0/(qτtr)입니다. Contact, pressure, diffusion은 포함하지 않습니다.'));
      ctx.status.textContent = ctx.text(`Ω=${w.toFixed(2)}; critical damping is Ω=0.5. The undamped plasma frequency ωp is not the damped oscillation frequency sqrt(ωp²−1/(4τtr²)). Continuity uses charge −eδn.`, `Ω=${w.toFixed(2)}이며 Ω=0.5에서 critical damping입니다. 감쇠가 없을 때의 plasma frequency ωp와 실제 진동 주파수 sqrt(ωp²−1/(4τtr²))는 다릅니다. Continuity equation에는 전하 −eδn을 사용합니다.`);
    }
    ctx.onChange(render); share(ctx, 'drude-charge', () => ({ omega: Number(omega.value), time: Number(time.value) }), s => grid(s.omega, .1, 3, .05) && grid(s.time, 0, 12, .1), s => { omega.value = s.omega; time.value = s.time; }, render);
  });
  function quizDiagram(ctx, index) {
    if (index === 0) return ctx.plot({ x: [0, 1], y: [-.2, 1.2], title: ctx.text('A uniform field has zero divergence', '균일한 전기장의 divergence는 0입니다'), xLabel: 'x/L', yLabel: 'E/E0; ρ/ρscale', series: [series([[0, 1], [1, 1]], 'blue', 'E/E0=1'), series([[0, 0], [1, 0]], 'gold', 'ρ=0')] });
    if (index === 1 || index === 4) return ctx.plot({ x: [-1, 1], y: [-1.1, 4.2], title: ctx.text('Band energy and inverse curvature', 'Band 에너지와 inverse curvature'), xLabel: 'ka/π', yLabel: 'ε/tₕ; inverse curvature / bottom', series: [series(samples(-1, 1, 161, x => 2 * (1 - Math.cos(Math.PI * x))), 'blue', 'ε/tₕ'), series(samples(-1, 1, 161, x => Math.cos(Math.PI * x)), 'gold', 'inverse curvature / bottom')] });
    if (index === 2) return ctx.plot({ x: [0, 1], y: [0, 2.1], title: ctx.text('Momentum loss per event depends on its angle', 'Event당 momentum loss는 scattering 각도에 따라 다릅니다'), xLabel: 'θ/π', yLabel: '1−cosθ', series: [series(samples(0, 1, 101, x => 1 - Math.cos(Math.PI * x)), 'blue', 'Transport weight')], markers: [{ x: 0, y: 0, color: 'blue', label: 'Forward: 0' }, { x: 1, y: 2, color: 'gold', label: 'Reverse: 2' }] });
    return ctx.plot({ x: [0, 2], y: [-1.1, 1.1], title: ctx.text('At ωτ=1, current lags by 45 degrees', 'ωτ=1에서 전류는 전기장보다 45도 늦습니다'), xLabel: 'ωt/π', yLabel: 'Normalized field and current', series: [series(samples(0, 2, 161, x => Math.cos(Math.PI * x)), 'muted', 'E/E0', true), series(samples(0, 2, 161, x => acResponse(1, Math.PI * x).current), 'blue', 'J/(σ0E0)')] });
  }
  labs.register('drude-quiz', ctx => {
    setup(ctx);
    const question = ctx.control('Question', '문제', { type: 'select', value: '0', options: questions.map((q, i) => ({ value: String(i), en: `${i + 1}. ${['Neutrality and field', 'Band curvature', 'Angular scattering', 'AC lag and relaxation', 'Scalar mobility'][i]}`, ko: `${i + 1}. ${['Charge neutrality와 전기장', 'Band curvature', 'Angular scattering', 'AC lag와 relaxation', 'Scalar mobility'][i]}` })) });
    let selected = -1, previous = 0;
    function render() {
      const index = Number(question.value), q = quizProblem(index); if (index !== previous) selected = -1; previous = index;
      const buttons = q.choices.map((choice, j) => { const b = ctx.el('button', { type: 'button', class: 'drude-choice' + (selected === j ? ' drude-chosen' : ''), 'aria-pressed': String(selected === j) }, ctx.text(...choice)); b.addEventListener('click', () => { selected = j; render(); }); return b; });
      ctx.view.replaceChildren(quizDiagram(ctx, index), ctx.el('p', { class: 'drude-question' }, ctx.text(q.en, q.ko)), ctx.el('div', { class: 'drude-choices' }, buttons), selected < 0 ? note(ctx, 'Choose one answer to see the explanation.', '답을 고르면 해설이 나타납니다.') : ctx.el('p', { class: 'drude-answer' + (selected === q.answer ? ' drude-correct' : '') }, ctx.text(selected === q.answer ? 'Correct. ' : 'Try again. ', selected === q.answer ? '맞았습니다. ' : '다시 생각해 보세요. ') + ctx.text(...q.explanation)));
      ctx.status.textContent = ctx.text(`Question ${index + 1} of ${questions.length}. Answers stay in this browser.`, `${questions.length}문제 중 ${index + 1}번입니다. 답은 이 브라우저 안에서만 처리합니다.`);
    }
    ctx.onReset(() => { selected = -1; previous = 0; }); ctx.onChange(render);
    share(ctx, 'drude-quiz', () => ({ question: Number(question.value), selected }), s => Number.isInteger(s.question) && inRange(s.question, 0, 4) && Number.isInteger(s.selected) && inRange(s.selected, -1, 2), s => { question.value = s.question; previous = s.question; selected = s.selected; }, render);
  });
})();
