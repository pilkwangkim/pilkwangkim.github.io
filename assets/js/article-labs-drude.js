/* Bounded Drude teaching models; SI internally, signed conventional current. */
(function () {
  'use strict';
  const E = 1.602176634e-19, M0 = 9.1093837139e-31, HBAR = 6.62607015e-34 / (2 * Math.PI);
  const inRange = (x, lo, hi) => Number.isFinite(x) && x >= lo && x <= hi;
  const grid = (x, lo, hi, step) => inRange(x, lo, hi) && Math.abs((x - lo) / step - Math.round((x - lo) / step)) < 1e-7;
  function stepResponse({ density = 1e23, massRatio = .2, tauFs = 100, field = 1000, s = 2 } = {}) {
    if (!inRange(density, 1e20, 1e28) || !inRange(massRatio, .01, 10) || !inRange(tauFs, 1, 1e4) || !inRange(field, -1e6, 1e6) || !inRange(s, 0, 1e6)) throw new Error('Invalid Drude settings.');
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
  function bandAtK(kNM, hoppingEV, latticeNM) {
    // Plot an extended-zone band on a fixed physical k axis. Folding only
    // the phase passed to tightBinding leaves energy and both derivatives
    // exactly periodic; it never rescales the plotted coordinate.
    if (!inRange(kNM, -32, 32)) throw new Error('Invalid physical wavevector.');
    const ka = kNM * latticeNM;
    const folded = ((ka + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return tightBinding(folded, hoppingEV, latticeNM);
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
  function chargeUnits(omegaTau) {
    if (!inRange(omegaTau, .1, 3)) throw new Error('Invalid physical charge-mode settings.');
    const tau = 100e-15, mass = .2 * M0, permittivity = 11.7 * 8.8541878188e-12, length = 100e-9, q = 2 * Math.PI / length;
    // With mass, permittivity and tau held fixed, changing Ω changes the
    // equilibrium carrier density. Keep that physical amplitude visible.
    const density = permittivity * mass * (omegaTau / tau) ** 2 / E ** 2, initialAmplitude = .05 * density;
    return { tau, mass, permittivity, length, q, density, initialAmplitude,
      fieldScale: E * initialAmplitude / (permittivity * q), currentScale: E * initialAmplitude / (q * tau) };
  }
  const questions = [
    { answer: 1, en: 'A charge-neutral uniform region has ρ=0. Must E be zero?', ko: '균일한 영역의 bulk charge density가 ρ=0입니다. E도 반드시 0일까요?', choices: [['Yes, zero charge forces zero field.', '네. 전하가 없으면 전기장도 없습니다.'], ['No. Poisson fixes the divergence; boundaries may impose a uniform field.', '아니요. Poisson equation은 divergence를 정합니다. 경계 조건에 따라 균일한 전기장이 생길 수 있습니다.'], ['No, but any uniform field violates Poisson.', '아니요. 다만 균일한 전기장은 Poisson equation을 위반합니다.']], explanation: ['In 1D, dE/dx=ρ/ε=0 permits any constant E. Boundary potentials fix that constant; neutrality alone does not.', '1D에서 dE/dx=ρ/ε=0이면 E는 상수입니다. 그 값은 경계에서 지정한 potential로 정해집니다. ρ=0이라는 조건만으로 E=0이 되지는 않습니다.'] },
    { answer: 2, en: 'Near ka=π, the tight-binding band bends downward. What does this establish?', ko: 'Tight-binding band가 ka=π 부근에서 아래로 휘어집니다. 이 사실로 알 수 있는 것은 무엇일까요?', choices: [['Every filled band has negative DC conductivity.', '전자들로 채워진 band는 모두 DC conductivity가 음수입니다.'], ['The group velocity is negative everywhere.', '모든 k에서 group velocity가 음수입니다.'], ['The local inverse curvature mass is negative; a DC conductivity needs a distribution and collision model.', '이 지점의 inverse curvature mass가 음수입니다. DC conductivity를 구하려면 전자의 distribution과 collision model도 알아야 합니다.']], explanation: ['v=(1/ℏ)dε/dk and m_curv⁻¹=(1/ℏ²)d²ε/dk² are different derivatives. Local negative curvature is not a negative scalar DC conductivity.', 'v=(1/ℏ)dε/dk와 m_curv⁻¹=(1/ℏ²)d²ε/dk²는 서로 다른 미분입니다. 특정 지점의 curvature가 음수라고 해서 scalar DC conductivity도 음수인 것은 아닙니다.'] },
    { answer: 0, en: 'Keep the total scattering-event rate fixed. What happens when the 3D angular kernel becomes strongly forward-peaked?', ko: '전체 scattering event rate를 고정합니다. 3D angular kernel이 작은 각도에 강하게 집중되면 어떻게 될까요?', choices: [['Each event removes less directed momentum, so τtr grows relative to τq.', 'Event 하나가 원래 진행 방향의 momentum을 덜 줄이므로 τq에 비해 τtr이 길어집니다.'], ['τtr must equal τq for all angular distributions.', 'Angular distribution과 관계없이 τtr=τq입니다.'], ['Any scattering event completely reverses the electron.', 'Scattering이 일어나면 전자의 방향은 항상 반대로 바뀝니다.']], explanation: ['The transport weight is 1−cosθ. It is small for a small deflection. This is not a universal τtr≥τq theorem: pure reverse scattering has weight 2 and τtr=τq/2.', 'Transport weight는 1−cosθ입니다. 방향이 조금만 바뀌면 이 값도 작습니다. 다만 항상 τtr≥τq인 것은 아닙니다. 반대 방향으로만 scattering하면 weight가 2이므로 τtr=τq/2입니다.'] },
    { answer: 1, en: 'For E=E0 cosωt, which relaxation time sets the Drude phase lag arctan(ωτ)?', ko: 'E=E0 cosωt일 때 Drude 전류의 phase lag arctan(ωτ)를 정하는 τ는 무엇일까요?', choices: [['Only the time for electron energy to reach the lattice temperature.', '전자계가 lattice와 같은 온도가 되는 데 걸리는 시간입니다.'], ['Momentum relaxation time; elastic collisions can relax current without changing electron energy.', 'Momentum relaxation time입니다. Elastic collision도 전자의 에너지를 바꾸지 않으면서 전류를 줄일 수 있습니다.'], ['The current leads the field, so no relaxation time is needed.', '전류가 전기장보다 앞서므로 relaxation time이 필요하지 않습니다.']], explanation: ['With exp(−iωt), σ=σ0/(1−iωτtr), so J=|σ|E0 cos[ωt−arctan(ωτtr)]. Positive Imσ corresponds to a lag in this convention. Energy relaxation is a different process.', 'exp(−iωt) convention에서 σ=σ0/(1−iωτtr)이고 J=|σ|E0 cos[ωt−arctan(ωτtr)]입니다. 이 convention에서는 Imσ>0이면 전류가 전기장보다 늦습니다. Energy relaxation은 별개의 과정입니다.'] },
    { answer: 0, en: 'When is the scalar formula μ=eτtr/m* justified?', ko: 'Scalar 식 μ=eτtr/m*를 사용할 수 있는 조건은 무엇일까요?', choices: [['An isotropic constant parabolic mass with a consistent scalar momentum relaxation time.', '방향에 관계없이 일정한 parabolic mass를 쓰고, 그 모델에 맞는 scalar momentum relaxation time을 쓸 때입니다.'], ['Every anisotropic band at every carrier energy.', '모든 anisotropic band의 모든 에너지에서 성립합니다.'], ['Whenever the curvature vanishes.', 'Curvature가 0이기만 하면 됩니다.']], explanation: ['Anisotropic curvature gives an inverse-mass tensor. Energy-dependent scattering also requires the appropriate conductivity-weighted average, rather than an arbitrary lifetime.', 'Curvature가 방향에 따라 다르면 inverse-mass tensor가 필요합니다. Scattering이 에너지에 따라 달라지면 임의의 lifetime이 아니라 conductivity 계산에 맞게 가중 평균한 값을 써야 합니다.'] }
  ];
  function quizProblem(index) { if (!Number.isInteger(index) || index < 0 || index >= questions.length) throw new Error('Invalid question.'); return questions[index]; }
  const api = { E, M0, HBAR, stepResponse, acResponse, tightBinding, bandAtK, screeningIntegral, scatteringShape, chargeMode, chargeProfile, chargeUnits, quizProblem };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const labs = window.ArticleLabs;
  const samples = (lo, hi, count, fn) => Array.from({ length: count }, (_, i) => { const x = lo + (hi - lo) * i / (count - 1); return [x, fn(x)]; });
  function setup(ctx) { ctx.root.classList.add('drude-lab', 'al-workshop'); }
  function note(ctx, en, ko) { return ctx.el('p', { class: 'drude-explanation' }, ctx.text(en, ko)); }
  const control = (ctx, en, ko, min, max, step, value) => ctx.control(en, ko, { type: 'range', min, max, step, value });
  function share(ctx, id, read, valid, write, render) { ctx.stateLink(id, { read, write: s => { if (!s || typeof s !== 'object' || Array.isArray(s) || !valid(s)) return false; write(s); render(); return true; } }); }
  const series = (points, color, label, dash = false) => ({ points, color, label, dash });
  labs.register('drude-step', ctx => {
    setup(ctx);
    const n = control(ctx, 'Density n (10²³ m⁻³)', '밀도 n (10²³ m⁻³)', .1, 10, .1, 1), mass = control(ctx, 'Mass m*/m0', '질량 m*/m0', .1, 2, .1, .2), tau = control(ctx, 'Momentum time τtr (fs)', 'Momentum relaxation time τtr (fs)', 10, 500, 10, 100), field = control(ctx, 'Electric field E (V/m)', '전기장 E (V/m)', -2000, 2000, 100, 1000), time = control(ctx, 'Probe time t/τtr', '관찰 시각 t/τtr', 0, 6, .1, 2);
    const view = ctx.control('Fixed current view (10⁶ A/m²)', '전류 축 범위 선택 (10⁶ A/m²)', { type: 'select', value: '3', options: [3, 30, 300].map(v => ({ value: String(v), en: `−${v} to +${v}`, ko: `−${v} ~ +${v}` })) });
    const mu = ctx.metric('Mobility (cm²/V s)', 'Mobility (cm²/V s)'), sigma = ctx.metric('Conductivity (S/m)', 'Conductivity (S/m)'), drift = ctx.metric('Electron drift u (m/s)', '전자 drift u (m/s)'), current = ctx.metric('Current I (nA)', '전류 I (nA)'), steady = ctx.metric('Steady J (A/m²)', '최종 current density J (A/m²)');
    const read = () => ({ n: Number(n.value), mass: Number(mass.value), tau: Number(tau.value), field: Number(field.value), time: Number(time.value), view: Number(view.value) });
    function render() {
      const s = read(), p = { density: s.n * 1e23, massRatio: s.mass, tauFs: s.tau, field: s.field, s: s.time }, result = stepResponse(p);
      const scale = 1e6, steadyScaled = result.steadyCurrentDensity / scale, limit = s.view, clipped = Math.abs(steadyScaled) > limit;
      // A monotone step curve intersects a selected fixed y boundary once.
      // Stop at the exact intersection rather than rescaling or flattening
      // the out-of-view portion. Metrics always retain the full SI value.
      const end = clipped ? Math.min(3, -s.tau / 1000 * Math.log1p(-limit / Math.abs(steadyScaled))) : 3;
      const traces = [series(samples(0, end, 301, t => stepResponse({ ...p, s: t * 1000 / s.tau }).currentDensity / scale), 'blue', 'J(t)')];
      if (!clipped) traces.push(series([[0, steadyScaled], [3, steadyScaled]], 'gold', 'σ0 E', true));
      const probePS = s.time * s.tau / 1000;
      traces.push(series([[probePS, -limit], [probePS, limit]], 'muted', ctx.text('Probe time', '관찰 시각'), true));
      const markerOutside = Math.abs(result.currentDensity / scale) > limit;
      mu.textContent = ctx.format(result.mobility * 1e4, 2); sigma.textContent = ctx.format(result.conductivity, 2); drift.textContent = ctx.format(result.drift, 3); current.textContent = ctx.format(result.currentNA, 3); steady.textContent = ctx.format(result.steadyCurrentDensity, 2);
      ctx.view.replaceChildren(ctx.plot({ x: [0, 3], y: [-limit, limit], xTicks: [0, 1, 2, 3], yTicks: [-limit, -limit / 2, 0, limit / 2, limit], title: ctx.text('Current after a field step: fixed physical scales', '전기장을 켠 뒤의 전류: 물리 단위의 고정 축'), xLabel: 't (ps)', yLabel: 'J (10⁶ A/m²)', series: traces, markers: [{ x: probePS, y: markerOutside ? Math.sign(result.currentDensity) * limit : result.currentDensity / scale, color: 'blue', label: markerOutside ? ctx.text('Probe current outside view', '관찰 시각의 전류는 표시 범위 밖') : '' }] }), note(ctx, 'Material sliders never change the axes. The current-view selector alone chooses ±3, ±30 or ±300×10⁶ A/m². Curves stop at the boundary when outside the selected view. The vertical dotted line always marks the physical probe time; a labeled boundary marker does not represent the actual out-of-view current. Metric readouts retain the full current. Changing τ changes the rise time in ps as well as the steady current. L=1 μm, A=(100 nm)²; I=JA and V=EL.', '물질의 매개변수를 바꿔도 축은 움직이지 않습니다. 전류 축 선택 메뉴에서만 ±3, ±30, ±300×10⁶ A/m² 중 하나를 고릅니다. 범위를 벗어난 곡선은 경계에서 끊습니다. 수직 점선은 항상 물리 단위의 관찰 시각을 가리킵니다. 경계의 표식은 실제 전류값을 뜻하지 않으며, 위의 수치는 실제 값을 그대로 보여 줍니다. τ를 바꾸면 ps 단위로 본 상승 시간과 최종 전류가 함께 달라집니다. L=1 μm, A=(100 nm)²이며 I=JA, V=EL입니다.'));
      const signed = (v, scientific = false) => (v < 0 ? '−' : '+') + (scientific ? Math.abs(v).toExponential(3) : Math.abs(v).toFixed(3));
      ctx.status.textContent = ctx.text(`J(t)=${signed(result.currentDensity, true)} A/m²; t=${probePS.toFixed(3)} ps; V=${signed(result.voltage * 1e3)} mV. Steady current: ${clipped ? 'outside view' : 'within view'}.`, `J(t)=${signed(result.currentDensity, true)} A/m², t=${probePS.toFixed(3)} ps, V=${signed(result.voltage * 1e3)} mV입니다. 최종 전류: 표시 범위 ${clipped ? '밖' : '안'}.`);
    }
    ctx.onChange(render); share(ctx, 'drude-step', read, s => grid(s.n, .1, 10, .1) && grid(s.mass, .1, 2, .1) && grid(s.tau, 10, 500, 10) && grid(s.field, -2000, 2000, 100) && grid(s.time, 0, 6, .1) && (s.view === undefined || [3, 30, 300].includes(s.view)), s => { n.value = s.n; mass.value = s.mass; tau.value = s.tau; field.value = s.field; time.value = s.time; view.value = s.view === undefined ? 3 : s.view; }, render);
  });
  labs.register('drude-ac', ctx => {
    setup(ctx);
    const omega = control(ctx, 'Angular frequency ω (10¹³ rad/s)', 'Angular frequency ω (10¹³ rad/s)', 0, 5, .05, 1), time = control(ctx, 'Probe time t (ps)', '관찰 시각 t (ps)', 0, 2, .02, .2);
    // Fixed material and field references: they do not follow the sliders.
    const tauPS = .1, sigmaSI = stepResponse().conductivity, fieldSI = 1000, currentMA = sigmaSI * fieldSI / 1e6;
    const magnitude = ctx.metric('|σ| (S/m)', '|σ| (S/m)'), lag = ctx.metric('Current lag (degrees)', '전류의 phase lag (degree)'), re = ctx.metric('Re σ (S/m)', 'Re σ (S/m)'), im = ctx.metric('Im σ (S/m)', 'Im σ (S/m)');
    function render() {
      const w = Number(omega.value), t = Number(time.value), p = acResponse(w, w * t / tauPS), frequencyTHz = w / (2 * Math.PI * tauPS);
      magnitude.textContent = ctx.format(p.magnitude * sigmaSI, 2); lag.textContent = ctx.format(p.lag * 180 / Math.PI, 2); re.textContent = ctx.format(p.real * sigmaSI, 2); im.textContent = ctx.format(p.imaginary * sigmaSI, 2);
      ctx.view.replaceChildren(ctx.plot({ x: [0, 8], y: [0, 1.5], xTicks: [0, 2, 4, 6, 8], yTicks: [0, .5, 1, 1.5], title: ctx.text('Complex conductivity at fixed τtr = 100 fs', 'τtr=100 fs인 물질의 complex conductivity'), xLabel: 'f = ω/(2π) (THz)', yLabel: 'σ (10³ S/m)', series: [series(samples(0, 8, 161, x => acResponse(2 * Math.PI * x * tauPS).real * sigmaSI / 1e3), 'blue', 'Re σ'), series(samples(0, 8, 161, x => acResponse(2 * Math.PI * x * tauPS).imaginary * sigmaSI / 1e3), 'gold', 'Im σ')], markers: [{ x: frequencyTHz, y: p.real * sigmaSI / 1e3, color: 'blue' }, { x: frequencyTHz, y: p.imaginary * sigmaSI / 1e3, color: 'gold' }] }),
      ctx.plot({ x: [0, 2], y: [-1.6, 1.6], xTicks: [0, .5, 1, 1.5, 2], yTicks: [-1.5, 0, 1.5], title: w === 0 ? ctx.text('Zero frequency: static field and current', '주파수가 0이면 전기장과 전류가 일정합니다') : ctx.text('Higher frequency gives more cycles in the same 2 ps', '같은 2 ps 동안 주파수에 따라 달라지는 진동 횟수'), xLabel: 't (ps)', yLabel: 'E (10³ V/m); J (10⁶ A/m²)', series: [series(samples(0, 2, 801, x => Math.cos(w * x / tauPS)), 'muted', 'E (10³ V/m)', true), series(samples(0, 2, 801, x => currentMA * acResponse(w, w * x / tauPS).current), 'blue', 'J (10⁶ A/m²)')], markers: [{ x: t, y: currentMA * p.current, color: 'blue' }] }),
      note(ctx, 'n=10²³ m⁻³, m*=0.2m0, τtr=100 fs and E0=1000 V/m stay fixed. The frequency axis is in THz and the time window is always 0–2 ps. Blue current uses 10⁶ A/m²; dashed field uses 10³ V/m. They share numeric plotting space but have different units. Positive Imσ gives a current lag with exp(−iωt). Switch-on transients are excluded.', 'n=10²³ m⁻³, m*=0.2m0, τtr=100 fs, E0=1000 V/m를 고정했습니다. 주파수 축은 THz이고 시간 범위는 항상 0–2 ps입니다. 파란 전류는 10⁶ A/m², 점선 전기장은 10³ V/m 단위를 씁니다. 같은 좌표 안에 표시하지만 서로 다른 물리량입니다. exp(−iωt) convention에서 Imσ>0이면 전류가 전기장보다 늦습니다. 전기장을 켠 직후의 transient는 포함하지 않습니다.'));
      ctx.status.textContent = ctx.text(`f=${frequencyTHz.toFixed(3)} THz; at t=${t.toFixed(2)} ps, J=${(currentMA * p.current).toFixed(4)}×10⁶ A/m². This is the steady periodic solution.`, `f=${frequencyTHz.toFixed(3)} THz이며 t=${t.toFixed(2)} ps에서 J=${(currentMA * p.current).toFixed(4)}×10⁶ A/m²입니다. 주기적인 변화가 자리 잡은 뒤의 해입니다.`);
    }
    ctx.onChange(render); share(ctx, 'drude-ac', () => ({ omega: Number(omega.value), timePs: Number(time.value) }), s => grid(s.omega, 0, 5, .05) && grid(s.timePs, 0, 2, .02), s => { omega.value = s.omega; time.value = s.timePs; }, render);
  });
  labs.register('drude-band', ctx => {
    setup(ctx);
    const k = control(ctx, 'Probe translation phase ka / π', '관찰할 translation phase ka / π', -1, 1, .01, .16), hopping = control(ctx, 'Hopping tₕ (eV)', 'Hopping tₕ (eV)', .1, 3, .1, 1), lattice = control(ctx, 'Lattice spacing a (nm)', 'Lattice spacing a (nm)', .1, 1, .05, .5), parabola = ctx.control('Show band-bottom parabola', 'Band bottom 부근의 parabola와 비교', { type: 'checkbox', value: true });
    const energy = ctx.metric('Energy ε (eV)', '에너지 ε (eV)'), velocity = ctx.metric('Group velocity (m/s)', 'Group velocity (m/s)'), inverse = ctx.metric('m0 / mcurv', 'm0 / mcurv');
    function render() {
      const kp = Number(k.value), t = Number(hopping.value), a = Number(lattice.value), physicalK = kp * Math.PI / a, p = tightBinding(kp * Math.PI, t, a), scale = tightBinding(0, t, a).inverseMassM0, vmax = 2 * t * E * a * 1e-9 / HBAR;
      const displayedVelocity = Math.abs(kp) === 1 ? 0 : p.velocity, displayedInverse = Math.abs(kp) === .5 ? 0 : p.inverseMassM0;
      energy.textContent = ctx.format(p.energyEV, 4); velocity.textContent = ctx.format(displayedVelocity, 1); inverse.textContent = ctx.format(displayedInverse, 4);
      const bands = [series(samples(-32, 32, 641, x => bandAtK(x, t, a).energyEV), 'blue', '2tₕ(1−cos ka)')];
      if (parabola.checked) bands.push(series(samples(-Math.PI / (2 * a), Math.PI / (2 * a), 101, x => t * (x * a) ** 2), 'gold', 'tₕ(ka)²; |ka|≤π/2', true));
      const boundaries = (lo, hi) => [-1, 1].map(sign => series([[sign * Math.PI / a, lo], [sign * Math.PI / a, hi]], 'muted', sign === 1 ? 'First-zone edges ±π/a' : '', true));
      ctx.view.replaceChildren(ctx.plot({ x: [-32, 32], y: [0, 12.5], xTicks: [-30, -15, 0, 15, 30], yTicks: [0, 4, 8, 12], title: ctx.text('Band energy on a fixed physical k axis', '고정된 k 축에서 비교하는 band energy'), xLabel: 'k (nm⁻¹)', yLabel: 'ε (eV)', series: [...bands, ...boundaries(0, 12.5)], markers: [{ x: physicalK, y: p.energyEV, color: 'blue' }] }),
      ctx.plot({ x: [-32, 32], y: [-10, 10], xTicks: [-30, -15, 0, 15, 30], yTicks: [-10, -5, 0, 5, 10], title: ctx.text('First derivative: group velocity', '첫 번째 미분: group velocity'), xLabel: 'k (nm⁻¹)', yLabel: 'v (10⁶ m/s)', series: [series(samples(-32, 32, 641, x => bandAtK(x, t, a).velocity / 1e6), 'blue', 'v = (1/ℏ)dε/dk'), ...boundaries(-10, 10)], markers: [{ x: physicalK, y: displayedVelocity / 1e6, color: 'blue' }] }),
      ctx.plot({ x: [-32, 32], y: [-85, 85], xTicks: [-30, -15, 0, 15, 30], yTicks: [-80, -40, 0, 40, 80], title: ctx.text('Second derivative: inverse curvature mass', '두 번째 미분: inverse curvature mass'), xLabel: 'k (nm⁻¹)', yLabel: 'm0 / mcurv', series: [series(samples(-32, 32, 641, x => bandAtK(x, t, a).inverseMassM0), 'gold', '(m0/ℏ²)d²ε/dk²'), ...boundaries(-85, 85)], markers: [{ x: physicalK, y: displayedInverse, color: 'gold' }] }),
      note(ctx, 'Every axis is fixed while tₕ and a change. Increasing tₕ raises the energy and derivative amplitudes; increasing a reduces the reciprocal-space period 2π/a. Repeated periods are the same extended-zone band, not additional bands. Dashed vertical lines mark the moving first-zone edges. The inverse-mass axis uses the constant bare mass m0, not the changing band-bottom value. Negative local curvature alone does not imply negative DC conductivity.', 'tₕ와 a를 바꿔도 축은 그대로입니다. tₕ가 커지면 energy와 미분값의 진폭이 커지고, a가 커지면 k 공간의 주기 2π/a가 짧아집니다. 반복되는 곡선은 서로 다른 band가 아니라 같은 band를 extended-zone으로 그린 것입니다. 수직 점선은 첫 Brillouin zone의 경계입니다. Inverse mass는 변하는 band-bottom 값이 아니라 일정한 bare mass m0를 기준으로 표시합니다. 특정 지점의 curvature가 음수라고 해서 DC conductivity도 음수인 것은 아닙니다.'));
      ctx.status.textContent = ctx.text(`Probe k=${ctx.format(physicalK, 3)} nm⁻¹ (ka/π=${kp.toFixed(2)}); vmax=${ctx.format(vmax, 1)} m/s; bottom m0/m*=${scale.toFixed(4)}. The parabola is shown only for |ka|≤π/2.`, `관찰 지점 k=${ctx.format(physicalK, 3)} nm⁻¹ (ka/π=${kp.toFixed(2)})입니다. vmax=${ctx.format(vmax, 1)} m/s, band bottom의 m0/m*=${scale.toFixed(4)}입니다. Parabola는 |ka|≤π/2에서만 표시합니다.`);
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
      const probability = samples(0, Math.PI, 801, angle => scatteringShape(x, angle).pdf), weighted = samples(0, Math.PI, 801, angle => scatteringShape(x, angle).transportWeighted), ymax = 7;
      const chart = reverse ? ctx.plot({ x: [-.1, 1.1], y: [0, 2.2], xTicks: [0, 1], yTicks: [0, 1, 2], title: ctx.text('Discrete benchmark: every event reverses direction', '두 방향 모델: event마다 방향을 뒤집습니다'), xLabel: 'θ / π', yLabel: ctx.text('Probability; weighted probability', 'Probability; weighted probability'), bars: [{ x: .97, y: 1, width: .05, color: 'blue', label: 'P(θ=π)=1' }, { x: 1.03, y: 2, width: .05, color: 'gold', label: 'P(θ=π)(1−cosπ)=2' }] }) : ctx.plot({ x: [0, 1], y: [0, ymax], yTicks: [0, ymax / 2, ymax], title: ctx.text('Angular probability includes the 3D sinθ measure', '3D angular measure sinθ를 포함한 확률밀도'), xLabel: 'θ / π', yLabel: 'Probability density per radian', series: [series(probability.map(p => [p[0] / Math.PI, p[1]]), 'blue', 'pθ; integral over dθ = 1'), series(weighted.map(p => [p[0] / Math.PI, p[1]]), 'gold', 'pθ(1−cosθ)')] });
      ctx.view.replaceChildren(chart, note(ctx, 'For the 3D kernel, pθ=(1+xₛ)sinθ/[2(1+xₛ sin²(θ/2))²]. The blue density integrates to one with dθ (radians); the gold integral is ⟨1−cosθ⟩. The total single-particle event rate 1/τq is fixed while the kernel shape changes. This is not a screening-wavevector sweep at fixed potential strength.', '3D kernel의 확률밀도는 pθ=(1+xₛ)sinθ/[2(1+xₛ sin²(θ/2))²]입니다. 파란 곡선을 dθ(radian)로 적분하면 1이고, 금색 곡선의 적분은 ⟨1−cosθ⟩입니다. 전체 single-particle event rate 1/τq를 고정하고 kernel의 모양만 바꿉니다. Potential strength를 고정한 채 screening wavevector를 바꾸는 계산과는 다릅니다.'), note(ctx, 'Isotropic scattering gives τtr=τq. Forward-peaked scattering gives a longer τtr in this kernel family. The separate two-direction reversal benchmark gives τtr=τq/2: a reversal removes twice the initial directed momentum.', 'Isotropic scattering에서는 τtr=τq입니다. 이 kernel에서는 작은 각도로 scattering할수록 τtr이 길어집니다. 별도로 제시한 두 방향 reversal 모델에서는 τtr=τq/2입니다. 방향을 완전히 뒤집으면 원래 진행 방향의 momentum 성분이 처음 값의 두 배만큼 줄어듭니다.'));
      ctx.status.textContent = ctx.text(`Model: ${mode.value}; τtr/τq=${p.ratio.toFixed(4)}. The shape control is active only for the screened kernel.`, `Model: ${mode.value}, τtr/τq=${p.ratio.toFixed(4)}입니다. Shape는 screened kernel에서만 조절합니다.`);
    }
    ctx.onChange(render); share(ctx, 'drude-scattering', () => ({ mode: mode.value, shape: Number(shape.value) }), s => ['screened', 'isotropic', 'reverse'].includes(s.mode) && grid(s.shape, .01, 400, .01), s => { mode.value = s.mode; shape.value = s.shape; }, render);
  });
  labs.register('drude-charge', ctx => {
    setup(ctx);
    const omega = control(ctx, 'Plasma parameter Ω=ωpτtr (varies n0)', 'Plasma parameter Ω=ωpτtr (n0 변화)', .1, 3, .05, 1), time = control(ctx, 'Probe time t/τtr (τtr=100 fs)', '관찰 시각 t/τtr (τtr=100 fs)', 0, 12, .1, 2);
    const regime = ctx.metric('Regime', '감쇠 유형'), amplitude = ctx.metric('Density amplitude a', '밀도 변화의 amplitude a'), derivative = ctx.metric('Current amplitude a′', '전류의 amplitude a′'), frequency = ctx.metric('Damped frequency × τtr', '감쇠 진동 주파수 × τtr'), residual = ctx.metric('Scaled continuity residual', 'Normalized continuity residual');
    function render() {
      const w = Number(omega.value), s = Number(time.value), p = chargeMode(s, w), units = chargeUnits(w), densityScale = units.initialAmplitude / 1e22;
      regime.textContent = ctx.text(p.regime, ({ underdamped: '진동하며 감쇠', critical: 'Critical damping', overdamped: '진동 없이 감쇠' })[p.regime]); amplitude.textContent = p.a.toFixed(5); derivative.textContent = p.derivative.toFixed(5); frequency.textContent = p.frequency.toFixed(4); residual.textContent = chargeProfile(.23, s, w).continuityResidual.toExponential(1);
      ctx.view.replaceChildren(ctx.plot({ x: [0, 1.2], y: [-3.5, 3.5], xTicks: [0, .4, .8, 1.2], yTicks: [-3, 0, 3], title: ctx.text('Physical density amplitude at fixed τtr', 'τtr을 고정하고 본 밀도 변화의 진폭'), xLabel: 't (ps)', yLabel: 'A(t) (10²² m⁻³)', series: [series(samples(0, 1.2, 301, ps => densityScale * chargeMode(ps * 10, w).a), 'blue', 'A(t)')], markers: [{ x: s / 10, y: densityScale * p.a, color: 'blue' }] }),
      ctx.plot({ x: [0, 100], y: [-3.5, 3.5], xTicks: [0, 25, 50, 75, 100], yTicks: [-3, 0, 3], title: ctx.text('Density variation over one fixed 100 nm period', '100 nm 주기에 걸친 전자 밀도 변화'), xLabel: 'x (nm)', yLabel: 'δn (10²² m⁻³)', series: [series(samples(0, 100, 201, x => densityScale * p.a * Math.cos(2 * Math.PI * x / 100)), 'blue', 'δn(x,t)')] }),
      ctx.plot({ x: [0, 100], y: [-1, 1], xTicks: [0, 25, 50, 75, 100], yTicks: [-1, 0, 1], title: ctx.text('Poisson field on the same spatial interval', '같은 구간의 Poisson 전기장'), xLabel: 'x (nm)', yLabel: 'E (10⁶ V/m)', series: [series(samples(0, 100, 201, x => units.fieldScale / 1e6 * chargeProfile(x / 100, s, w).field), 'red', 'E(x,t)')] }),
      ctx.plot({ x: [0, 100], y: [-3, 3], xTicks: [0, 25, 50, 75, 100], yTicks: [-3, 0, 3], title: ctx.text('Conventional current from charge continuity', '전하 보존식으로 구한 conventional current'), xLabel: 'x (nm)', yLabel: 'J (10⁹ A/m²)', series: [series(samples(0, 100, 201, x => units.currentScale / 1e9 * chargeProfile(x / 100, s, w).current), 'gold', 'J(x,t)')] }),
      note(ctx, 'Fixed references: τtr=100 fs, m*=0.2m0, εb=11.7, L=100 nm and qL=2π. Changing Ω changes n0=εm*(Ω/τtr)²/e²; the initial density amplitude is A0=0.05n0. Thus changing Ω changes physical amplitudes as well as damping; no curve is divided by its changing peak. Each plot has its own fixed physical units. A uniform positive background cancels n0. Pressure, diffusion and contacts are omitted.', 'τtr=100 fs, m*=0.2m0, εb=11.7, L=100 nm, qL=2π로 고정했습니다. Ω를 바꾸면 n0=εm*(Ω/τtr)²/e²에 따라 기준 밀도가 바뀌고, 초기 밀도 변화의 진폭은 A0=0.05n0입니다. 따라서 Ω가 바뀌면 감쇠뿐 아니라 물리량의 실제 크기도 달라집니다. 변하는 peak 값으로 곡선을 나누지 않으며, 각 그래프는 자기 물리량의 고정 단위를 씁니다. 균일한 양전하가 n0의 전하를 상쇄하고, pressure·diffusion·contact는 포함하지 않습니다.'));
      ctx.status.textContent = ctx.text(`Ω=${w.toFixed(2)}, n0=${units.density.toExponential(3)} m⁻³, t=${(s / 10).toFixed(2)} ps. Critical damping is Ω=0.5; the damped frequency differs from ωp.`, `Ω=${w.toFixed(2)}, n0=${units.density.toExponential(3)} m⁻³, t=${(s / 10).toFixed(2)} ps입니다. Ω=0.5에서 critical damping이며 실제 감쇠 진동 주파수는 ωp와 다릅니다.`);
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
