/* Microscopic Drude teaching models. SI internally; no measured material data. */
(function () {
  'use strict';
  const E = 1.602176634e-19, M0 = 9.1093837139e-31;
  const HBAR = 6.62607015e-34 / (2 * Math.PI), KB = 1.380649e-23;
  const MASS = .2 * M0, AREF = .2e-9, NIREF = 1e25, KREF = .05 * E;
  const VS = 5000;
  const inRange = (x, lo, hi) => Number.isFinite(x) && x >= lo && x <= hi;
  const grid = (x, lo, hi, step) => inRange(x, lo, hi) && Math.abs((x - lo) / step - Math.round((x - lo) / step)) < 1e-7;
  const exactKeys = (s, keys) => s !== null && typeof s === 'object' && !Array.isArray(s) && Object.keys(s).length === keys.length && keys.every(k => Object.prototype.hasOwnProperty.call(s, k));

  function angularChannel(degrees, ratePerPs = 1) {
    if (!inRange(degrees, 0, 180) || !inRange(ratePerPs, .1, 5)) throw new Error('Invalid angular channel.');
    const theta = degrees * Math.PI / 180;
    // 2 sin²(theta/2) avoids cancellation for very small deflections.
    const loss = 2 * Math.sin(theta / 2) ** 2;
    const forward = degrees === 90 ? 0 : Math.cos(theta);
    return { theta, forward, transverse: degrees === 180 ? 0 : Math.sin(theta), loss,
      referenceRate: ratePerPs, transportRate: ratePerPs * loss,
      transportTimePs: loss === 0 ? null : 1 / (ratePerPs * loss),
      timeRatio: loss === 0 ? null : 1 / loss, isIdentity: degrees === 0 };
  }

  function shortRange(energyMeV, densityFactor = 1, amplitudeFactor = 1) {
    if (!inRange(energyMeV, 1, 500) || !inRange(densityFactor, .1, 5) || !inRange(amplitudeFactor, .1, 3)) throw new Error('Invalid short-range inputs.');
    const kinetic = energyMeV * 1e-3 * E, k = Math.sqrt(2 * MASS * kinetic) / HBAR;
    const speed = HBAR * k / MASS, aBorn = AREF * amplitudeFactor;
    const u0 = 2 * Math.PI * HBAR ** 2 * aBorn / MASS;
    const dosOneSpin = MASS * k / (2 * Math.PI ** 2 * HBAR ** 2);
    const crossSection = 4 * Math.PI * aBorn ** 2;
    const rate = NIREF * densityFactor * speed * crossSection;
    const tau = 1 / rate, meanFreePath = speed * tau;
    return { kinetic, k, speed, aBorn, u0, dosOneSpin, crossSection, rate, tau, meanFreePath,
      kMeanFreePath: k * meanFreePath,
      dosRatio: Math.sqrt(energyMeV / 50),
      rateRatio: densityFactor * amplitudeFactor ** 2 * Math.sqrt(energyMeV / 50) };
  }

  function phononMode(qPerNM, temperatureK, energyMeV) {
    if (!inRange(qPerNM, .05, 2) || !inRange(temperatureK, 5, 600) || !inRange(energyMeV, .1, 100)) throw new Error('Invalid phonon inputs.');
    const q = qPerNM * 1e9, phononEnergy = HBAR * VS * q, kinetic = energyMeV * 1e-3 * E;
    const z = phononEnergy / (KB * temperatureK), population = 1 / Math.expm1(z);
    const k = Math.sqrt(2 * MASS * kinetic) / HBAR, speed = HBAR * k / MASS;
    const absorptionCosine = VS / speed - q / (2 * k);
    const emissionCosine = VS / speed + q / (2 * k);
    const absorptionAllowed = Math.abs(absorptionCosine) <= 1 + 1e-12;
    const emissionEnergyAllowed = kinetic >= phononEnergy;
    const emissionAllowed = emissionEnergyAllowed && Math.abs(emissionCosine) <= 1 + 1e-12;
    return { q, k, speed, phononEnergy, z, population, emissionFactor: population + 1,
      equipartitionPopulation: 1 / z, occupationRatio: z * population,
      energyRatio: phononEnergy / kinetic, absorptionCosine, emissionCosine,
      absorptionAllowed, emissionEnergyAllowed, emissionAllowed };
  }

  // Lanczos gamma on positive arguments; the lab uses 0.5 <= argument <= 4.5.
  function gamma(z) {
    if (!(Number.isFinite(z) && z > 0 && z <= 10)) throw new Error('Gamma argument outside this teaching model.');
    const coefficients = [676.5203681218851, -1259.1392167224028, 771.3234287776531,
      -176.6150291621406, 12.507343278686905, -.13857109526572012,
      9.984369578019572e-6, 1.5056327351493116e-7];
    if (z < .5) return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
    const v = z - 1; let a = .99999999999980993;
    coefficients.forEach((coefficient, i) => { a += coefficient / (v + i + 1); });
    const t = v + 7.5;
    return Math.sqrt(2 * Math.PI) * t ** (v + .5) * Math.exp(-t) * a;
  }
  function mbWeight(y, transport = false) {
    if (!inRange(y, 0, 100) || typeof transport !== 'boolean') throw new Error('Invalid MB weight.');
    return y ** (transport ? 1.5 : .5) * Math.exp(-y) / gamma(transport ? 2.5 : 1.5);
  }
  const TAUREF = shortRange(50).tau;
  function thermalAverage(temperatureK, exponent) {
    if (!inRange(temperatureK, 50, 600) || !inRange(exponent, -2, 2)) throw new Error('Invalid thermal average.');
    const scale = (KB * temperatureK / KREF) ** exponent;
    const transportRatio = scale * gamma(exponent + 2.5) / gamma(2.5);
    const particleRatio = exponent <= -1.5 ? null : scale * gamma(exponent + 1.5) / gamma(1.5);
    return { transportRatio, particleRatio, tauEffective: TAUREF * transportRatio,
      tauParticle: particleRatio === null ? null : TAUREF * particleRatio,
      mobility: E * TAUREF * transportRatio / MASS,
      particleToTransport: particleRatio === null ? null : particleRatio / transportRatio };
  }

  const api = { E, M0, HBAR, KB, MASS, AREF, NIREF, KREF, VS, TAUREF,
    angularChannel, shortRange, phononMode, gamma, mbWeight, thermalAverage };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const labs = window.ArticleLabs;
  const samples = (lo, hi, count, fn) => Array.from({ length: count }, (_, i) => { const x = lo + (hi - lo) * i / (count - 1); return [x, fn(x)]; });
  const line = (points, color, label, dash = false) => ({ points, color, label, dash });
  function positiveAxis(maximum) {
    const unit = 10 ** Math.floor(Math.log10(maximum));
    const limit = [1, 2, 5, 10].find(value => value * unit >= maximum) * unit;
    return { y: [0, limit], yTicks: [0, limit / 2, limit] };
  }
  const range = (ctx, en, ko, min, max, step, value) => ctx.control(en, ko, { type: 'range', min, max, step, value });
  const explanation = (ctx, en, ko) => ctx.el('p', { class: 'drude-explanation' }, ctx.text(en, ko));
  function setup(ctx) { ctx.root.classList.add('drude-lab', 'al-workshop'); }
  function share(ctx, id, keys, read, valid, write, render) {
    ctx.stateLink(id, { read, write: s => {
      if (!exactKeys(s, keys) || !valid(s)) return false;
      // Validation of every field precedes any change to the controls.
      write(s); render(); return true;
    } });
  }
  function preset(ctx, en, ko, callback, render) { ctx.button(en, ko, () => { callback(); render(); }); }
  function directionDiagram(ctx, p) {
    const svg = ctx.svg('svg', { viewBox: '0 0 640 260', class: 'al-chart', role: 'img', 'aria-label': ctx.text('Equal-length incident and outgoing wavevectors, with the retained forward projection.', '길이가 같은 incident wavevector와 outgoing wavevector, 원래 방향으로 남는 성분을 비교합니다.') });
    const cx = 320, cy = 143, radius = 82;
    const add = (tag, attrs, text) => { const el = ctx.svg(tag, attrs, text); svg.append(el); return el; };
    const colors = { blue: '#28629a', gold: '#a76d1b', muted: '#607987', light: '#d7e1e5' };
    add('circle', { cx, cy, r: radius, fill: 'none', stroke: colors.light, 'stroke-width': 1.5 });
    add('line', { x1: cx - 115, y1: cy, x2: cx + 125, y2: cy, stroke: colors.muted });
    add('line', { x1: cx, y1: cy + 95, x2: cx, y2: cy - 105, stroke: colors.muted });
    function arrow(x, y, color) {
      add('line', { x1: cx, y1: cy, x2: x, y2: y, stroke: color, 'stroke-width': 3.8 });
      const angle = Math.atan2(y - cy, x - cx), length = 10, width = 5;
      add('polygon', { points: `${x},${y} ${x - length * Math.cos(angle) + width * Math.sin(angle)},${y - length * Math.sin(angle) - width * Math.cos(angle)} ${x - length * Math.cos(angle) - width * Math.sin(angle)},${y - length * Math.sin(angle) + width * Math.cos(angle)}`, fill: color });
    }
    arrow(cx + radius, cy, colors.blue);
    const x = cx + radius * p.forward, y = cy - radius * p.transverse;
    arrow(x, y, colors.gold);
    add('line', { x1: x, y1: y, x2: x, y2: cy, stroke: colors.gold, 'stroke-width': 1.7, 'stroke-dasharray': '5 4' });
    const attrs = { 'font-size': 17, fill: colors.muted, 'font-family': 'system-ui, sans-serif' };
    add('text', { ...attrs, x: cx + 132, y: cy + 6 }, 'kx / k');
    add('text', { ...attrs, x: cx + 10, y: cy - 108 }, 'ky / k');
    add('text', { ...attrs, x: 22, y: 27, fill: colors.blue }, ctx.text('Incoming: +x', 'Incident: +x'));
    add('text', { ...attrs, x: 22, y: 66, fill: colors.gold }, ctx.text(`Outgoing: θ=${(p.theta * 180 / Math.PI).toFixed(0)}°`, `Outgoing: θ=${(p.theta * 180 / Math.PI).toFixed(0)}°`));
    return ctx.el('div', { class: 'al-chart-wrap' }, [svg, explanation(ctx, 'Wavevector space: arrows are not particle paths. The dashed line projects k′ onto the original direction.', 'Wavevector space의 벡터이며 전자가 움직인 경로가 아닙니다. 점선은 k′를 원래 방향으로 projection한 선입니다.')]);
  }

  labs.register('drude-angle', ctx => {
    setup(ctx);
    const angle = range(ctx, 'Deflection θ (degrees)', 'Scattering 각도 θ (degree)', 0, 180, 1, 60);
    const clock = range(ctx, 'Channel reference rate r (ps⁻¹)', 'Channel reference rate r (ps⁻¹)', .1, 5, .1, 1);
    const retained = ctx.metric('Retained forward component k′x/k', '원래 방향으로 남는 성분 k′x/k');
    const lost = ctx.metric('Forward-component loss / k', '원래 방향 성분의 감소량 / k');
    const rate = ctx.metric('Current relaxation rate Γtr (ps⁻¹)', 'Current relaxation rate Γtr (ps⁻¹)');
    const time = ctx.metric('Transport time τtr (ps)', 'Transport time τtr (ps)');
    function render() {
      const p = angularChannel(Number(angle.value), Number(clock.value));
      retained.textContent = ctx.format(p.forward, 4); lost.textContent = ctx.format(p.loss, 4);
      rate.textContent = ctx.format(p.transportRate, 4);
      time.textContent = p.isIdentity ? ctx.text('∞ (no directional decay)', '∞ (방향이 바뀌지 않음)') : ctx.format(p.transportTimePs, 4);
      ctx.view.replaceChildren(directionDiagram(ctx, p), ctx.plot({ x: [0, 180], y: [0, 2.1], xTicks: [0, 90, 180], yTicks: [0, 1, 2],
        title: ctx.text('One counted deflection need not remove one unit of directed momentum', '같은 횟수로 scattering해도 momentum 감소량은 다릅니다'),
        xLabel: 'θ (degrees)', yLabel: 'Γtr / r = 1 − cosθ',
        series: [line(samples(0, 180, 181, t => angularChannel(t).loss), 'blue', '1−cosθ'), line([[0, 1], [180, 1]], 'muted', ctx.text('Reference rate / r = 1', 'Reference rate / r = 1'), true)],
        markers: [{ x: Number(angle.value), y: p.loss, color: 'gold' }] }),
      explanation(ctx, 'The 3D model scatters at one polar angle, uniformly around the incident axis, and keeps speed fixed. Averaging around that axis leaves cosθ of the original directed component. The velocity-weighted occupation imbalance that carries current therefore decays at Γtr=r(1−cosθ). Multiply the drawn wavevectors by ℏ for momentum. No energy is lost in this elastic model.', '3D에서 scattering 각도 하나를 정하고, 원래 방향을 둘러싼 azimuth 방향에는 같은 확률을 줍니다. Speed는 바뀌지 않습니다. Azimuth 방향으로 평균하면 원래 방향의 성분은 cosθ만큼 남고, 전류를 만드는 방향별 occupation 차이는 Γtr=r(1−cosθ)로 감소합니다. 그림의 wavevector에 ℏ를 곱하면 momentum입니다. 이 elastic 모델에서는 에너지가 줄어들지 않습니다.'));
      ctx.status.textContent = p.isIdentity ? ctx.text('At exactly θ=0, the state direction is unchanged: Γtr=0 and τtr is infinite. r is only a reference clock here, not a physical rate for leaving that same state.', 'θ=0에서는 상태의 방향이 그대로이므로 Γtr=0, τtr=∞입니다. 이 끝점의 r은 비교용 reference일 뿐, 같은 상태를 벗어나는 실제 transition rate가 아닙니다.') : ctx.text(`For θ>0, the event lifetime is 1/r=${ctx.format(1 / Number(clock.value), 3)} ps. τtr/(1/r)=${ctx.format(p.timeRatio, 4)}. A reversal removes twice the original forward component.`, `θ>0에서 event lifetime 1/r=${ctx.format(1 / Number(clock.value), 3)} ps, τtr/(1/r)=${ctx.format(p.timeRatio, 4)}입니다. 방향을 완전히 뒤집으면 원래 방향의 성분이 처음 값의 두 배만큼 줄어듭니다.`);
    }
    preset(ctx, 'Small deflection: 10°', '조금만 방향 바꾸기: 10°', () => { angle.value = 10; }, render);
    preset(ctx, 'Right angle: 90°', '직각으로 방향 바꾸기: 90°', () => { angle.value = 90; }, render);
    preset(ctx, 'Reverse: 180°', '반대 방향으로 바꾸기: 180°', () => { angle.value = 180; }, render);
    ctx.onChange(render);
    share(ctx, 'drude-angle', ['angle', 'clock'], () => ({ angle: Number(angle.value), clock: Number(clock.value) }),
      s => grid(s.angle, 0, 180, 1) && grid(s.clock, .1, 5, .1), s => { angle.value = s.angle; clock.value = s.clock; }, render);
  });

  labs.register('drude-impurity', ctx => {
    setup(ctx);
    const energy = range(ctx, 'Kinetic energy K (meV)', 'Kinetic energy K (meV)', 10, 150, 5, 50);
    const density = range(ctx, 'Impurity density ni / 10²⁵ m⁻³', 'Impurity density ni / 10²⁵ m⁻³', .2, 2, .1, 1);
    const amplitude = range(ctx, 'Fourier amplitude |U0| / |U0,ref|', 'Fourier amplitude |U0| / |U0,ref|', .5, 1.5, .1, 1);
    const rate = ctx.metric('Γq = Γtr (ps⁻¹)', 'Γq = Γtr (ps⁻¹)'), tau = ctx.metric('τq = τtr (ps)', 'τq = τtr (ps)');
    const dos = ctx.metric('One-spin DOS / reference DOS', 'One-spin DOS / reference DOS'), path = ctx.metric('Mean free path ℓtr (nm)', 'Mean free path ℓtr (nm)'), kl = ctx.metric('kℓtr (dimensionless)', 'kℓtr (dimensionless)');
    function render() {
      const k = Number(energy.value), n = Number(density.value), u = Number(amplitude.value), p = shortRange(k, n, u);
      rate.textContent = ctx.format(p.rate * 1e-12, 4); tau.textContent = ctx.format(p.tau * 1e12, 4); dos.textContent = ctx.format(p.dosRatio, 4);
      path.textContent = ctx.format(p.meanFreePath * 1e9, 2); kl.textContent = ctx.format(p.kMeanFreePath, 2);
      const rates = samples(10, 150, 141, x => shortRange(x, n, u).rate * 1e-12), times = samples(10, 150, 141, x => shortRange(x, n, u).tau * 1e12);
      const refRates = samples(10, 150, 141, x => shortRange(x).rate * 1e-12), refTimes = samples(10, 150, 141, x => shortRange(x).tau * 1e12);
      ctx.view.replaceChildren(ctx.plot({ x: [10, 150], xTicks: [50, 100, 150], ...positiveAxis(Math.max(rates.at(-1)[1], refRates.at(-1)[1]) * 1.1),
        title: ctx.text('More final states produce a larger fixed-energy rate', '같은 에너지에서 가능한 final state가 많을수록 rate가 커집니다'), xLabel: 'K (meV)', yLabel: 'Γ (ps⁻¹)',
        series: [line(rates, 'blue', ctx.text('Chosen ni, |U0|', '설정한 ni, |U0|')), line(refRates, 'muted', ctx.text('Reference ni, |U0|', 'Reference ni, |U0|'), true)], markers: [{ x: k, y: p.rate * 1e-12, color: 'blue' }] }),
      ctx.plot({ x: [10, 150], xTicks: [50, 100, 150], ...positiveAxis(Math.max(times[0][1], refTimes[0][1]) * 1.1), title: ctx.text('The relaxation time is the reciprocal of that rate', 'Relaxation time은 이 rate의 역수입니다'), xLabel: 'K (meV)', yLabel: 'τ (ps)',
        series: [line(times, 'gold', 'τ=1/Γ'), line(refTimes, 'muted', ctx.text('Reference inputs', 'Reference 설정'), true)], markers: [{ x: k, y: p.tau * 1e12, color: 'gold' }] }),
      explanation(ctx, 'Fixed 3D parabolic mass m*=0.2m0; constant spin-conserving short-range Born amplitude. D1(K)∝√K, Γ=(2π/ℏ)ni|U0|²D1(K)=ni v 4πaB², and τ=1/Γ. The one-spin final-state DOS has no extra spin factor. Reference: Kref=50 meV, ni,ref=10²⁵ m⁻³ and aB,ref=0.2 nm; U0,ref=2πℏ²aB,ref/m*. Doubling |U0| would quadruple Γ, not double it.', 'm*=0.2m0인 3D parabolic band에서, spin을 바꾸지 않는 일정한 short-range Born amplitude를 사용합니다. D1(K)∝√K, Γ=(2π/ℏ)ni|U0|²D1(K)=ni v 4πaB², τ=1/Γ입니다. One-spin final-state DOS에 spin factor 2를 추가하지 않습니다. Reference는 Kref=50 meV, ni,ref=10²⁵ m⁻³, aB,ref=0.2 nm이며 U0,ref=2πℏ²aB,ref/m*입니다. |U0|가 두 배가 되면 Γ는 두 배가 아니라 네 배가 됩니다.'));
      ctx.status.textContent = ctx.text(`At K=${k} meV: v=${p.speed.toExponential(3)} m/s, |U0|=${p.u0.toExponential(3)} J m³. ℓtr=vτtr uses the microscopic speed, not drift. Large kℓ is a weak-disorder check; the unspecified defect range/strength must separately justify Born and short-range approximations.`, `K=${k} meV에서 v=${p.speed.toExponential(3)} m/s, |U0|=${p.u0.toExponential(3)} J m³입니다. ℓtr=vτtr에는 drift가 아니라 이 에너지의 speed를 씁니다. 큰 kℓ은 weak-disorder 조건을 확인하는 지표입니다. Defect의 range와 strength가 Born·short-range 근사에 맞는지는 별도로 확인해야 합니다.`);
    }
    preset(ctx, 'Article reference', '본문의 reference 값', () => { energy.value = 50; density.value = 1; amplitude.value = 1; }, render);
    preset(ctx, 'Double impurity density', 'Impurity density 두 배', () => { energy.value = 50; density.value = 2; amplitude.value = 1; }, render);
    preset(ctx, 'Halve the amplitude', 'Amplitude 절반', () => { energy.value = 50; density.value = 1; amplitude.value = .5; }, render);
    ctx.onChange(render);
    share(ctx, 'drude-impurity', ['energy', 'density', 'amplitude'], () => ({ energy: Number(energy.value), density: Number(density.value), amplitude: Number(amplitude.value) }),
      s => grid(s.energy, 10, 150, 5) && grid(s.density, .2, 2, .1) && grid(s.amplitude, .5, 1.5, .1), s => { energy.value = s.energy; density.value = s.density; amplitude.value = s.amplitude; }, render);
  });

  labs.register('drude-phonon', ctx => {
    setup(ctx);
    const q = range(ctx, 'Phonon wavevector q (nm⁻¹)', 'Phonon wavevector q (nm⁻¹)', .05, 2, .05, 1);
    const temperature = range(ctx, 'Temperature T (K)', '온도 T (K)', 5, 600, 5, 300);
    const energy = range(ctx, 'Electron kinetic energy K (meV)', '전자 kinetic energy K (meV)', .1, 100, .1, 50);
    const phonon = ctx.metric('Phonon energy ℏωq (meV)', 'Phonon energy ℏωq (meV)'), population = ctx.metric('Absorption occupation factor Nq', 'Absorption occupation factor Nq');
    const emission = ctx.metric('Emission occupation factor Nq+1', 'Emission occupation factor Nq+1'), thermal = ctx.metric('ℏωq / kBT', 'ℏωq / kBT'), elastic = ctx.metric('ℏωq / K', 'ℏωq / K');
    const abs = ctx.metric('Absorption at this q', '이 q에서 absorption'), em = ctx.metric('Emission at this q', '이 q에서 emission');
    const allowed = value => value ? ctx.text('Kinematically allowed', 'Energy·momentum 보존 가능') : ctx.text('Kinematically forbidden', 'Energy·momentum 보존 불가');
    function render() {
      const qp = Number(q.value), t = Number(temperature.value), k = Number(energy.value), p = phononMode(qp, t, k);
      phonon.textContent = ctx.format(p.phononEnergy / E * 1e3, 4); population.textContent = ctx.format(p.population, 4); emission.textContent = ctx.format(p.emissionFactor, 4);
      thermal.textContent = ctx.format(p.z, 4); elastic.textContent = ctx.format(p.energyRatio, 4); abs.textContent = allowed(p.absorptionAllowed); em.textContent = allowed(p.emissionAllowed);
      const points = samples(5, 600, 151, x => phononMode(qp, x, k).population), high = samples(5, 600, 151, x => phononMode(qp, x, k).equipartitionPopulation);
      ctx.view.replaceChildren(ctx.plot({ x: [5, 600], ...positiveAxis(Math.max(high.at(-1)[1], points.at(-1)[1] + 1) * 1.08),
        title: ctx.text('Phonon occupation factors are not complete transition rates', 'Phonon occupation factor만으로 transition rate가 정해지지는 않습니다'), xLabel: 'T (K)', yLabel: ctx.text('Occupation factor (dimensionless)', 'Occupation factor (dimensionless)'),
        series: [line(points, 'blue', 'Nq'), line(points.map(([x, y]) => [x, y + 1]), 'gold', 'Nq+1'), line(high, 'muted', 'kBT/(ℏωq)', true)],
        markers: [{ x: t, y: p.population, color: 'blue' }, { x: t, y: p.emissionFactor, color: 'gold' }] }),
      explanation(ctx, 'One longitudinal acoustic mode has ωq=v_s q with v_s=5000 m/s; electrons have a 3D parabolic mass 0.2m0. Nq=1/[exp(ℏωq/kBT)−1] counts mean phonons in a mode. Emission has Nq+1, including spontaneous emission. Complete rates additionally require |Mq|², final-state counting, energy conservation and electron occupations. This plot is a thermal population, not a classical atomic-displacement pattern.', 'Longitudinal acoustic mode 하나에 ωq=v_s q, v_s=5000 m/s를 쓰고, 전자는 m*=0.2m0인 3D parabolic band로 계산합니다. Nq=1/[exp(ℏωq/kBT)−1]은 mode의 평균 phonon 수입니다. Emission의 Nq+1에는 spontaneous emission도 포함됩니다. 실제 rate를 구하려면 |Mq|², final-state 수, energy conservation, 전자의 occupation도 필요합니다. 이 그래프는 thermal population이며 원자의 classical displacement를 그린 것이 아닙니다.'),
      explanation(ctx, 'High occupation needs ℏωq/kBT≪1. Quasielastic scattering separately needs a small electronic energy change on the relevant electronic scale; ℏωq/K is only a simple check here. For this fixed-q parabolic normal process, α is the angle between initial electron k and phonon q: cosα_abs=v_s/v−q/(2k), cosα_em=v_s/v+q/(2k). A real angle requires |cosα|≤1. K≥ℏωq is necessary, but not sufficient, for emission. No Pauli blocking or reciprocal-lattice transfer is included.', 'High occupation에는 ℏωq/kBT≪1이 필요합니다. Quasielastic 근사는 별도로 전자의 관련 에너지 scale에 비해 energy change가 작아야 합니다. 여기서는 간단한 지표로 ℏωq/K를 표시합니다. 고정한 q의 parabolic normal process에서 α는 initial electron k와 phonon q 사이의 각도입니다. cosα_abs=v_s/v−q/(2k), cosα_em=v_s/v+q/(2k)이며 실제 각도가 존재하려면 |cosα|≤1이어야 합니다. Emission에는 K≥ℏωq가 필요하지만, 그 조건만으로 충분하지는 않습니다. Pauli blocking과 reciprocal-lattice transfer는 포함하지 않습니다.'));
      ctx.status.textContent = ctx.text(`Required cosα: absorption ${p.absorptionCosine.toFixed(3)}, emission ${p.emissionCosine.toFixed(3)}. Final electron energies are K+ℏωq=${(k + p.phononEnergy / E * 1e3).toFixed(3)} meV and K−ℏωq=${(k - p.phononEnergy / E * 1e3).toFixed(3)} meV; a negative final energy is unavailable.`, `필요한 cosα는 absorption ${p.absorptionCosine.toFixed(3)}, emission ${p.emissionCosine.toFixed(3)}입니다. 전자의 final energy는 각각 K+ℏωq=${(k + p.phononEnergy / E * 1e3).toFixed(3)} meV, K−ℏωq=${(k - p.phononEnergy / E * 1e3).toFixed(3)} meV입니다. Final energy가 음수인 state는 없습니다.`);
    }
    preset(ctx, 'Small energy change', 'Energy change가 작은 예', () => { q.value = 1; temperature.value = 300; energy.value = 50; }, render);
    preset(ctx, 'Many phonons, large energy change', 'Phonon은 많지만 energy change도 큰 예', () => { q.value = .05; temperature.value = 300; energy.value = .1; }, render);
    preset(ctx, 'Low phonon occupation', 'Phonon occupation이 작은 예', () => { q.value = 2; temperature.value = 5; energy.value = 50; }, render);
    ctx.onChange(render);
    share(ctx, 'drude-phonon', ['q', 'temperature', 'energy'], () => ({ q: Number(q.value), temperature: Number(temperature.value), energy: Number(energy.value) }),
      s => grid(s.q, .05, 2, .05) && grid(s.temperature, 5, 600, 5) && grid(s.energy, .1, 100, .1), s => { q.value = s.q; temperature.value = s.temperature; energy.value = s.energy; }, render);
  });

  labs.register('drude-average', ctx => {
    setup(ctx);
    const temperature = range(ctx, 'Temperature T (K)', '온도 T (K)', 50, 600, 10, 300);
    const exponents = [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2];
    const exponent = ctx.control('Power r in τ(K)=τref(K/Kref)^r', 'τ(K)=τref(K/Kref)^r의 exponent r', { type: 'select', value: '-0.5', options: exponents.map(r => ({ value: String(r), en: String(r), ko: String(r) })) });
    const transport = ctx.metric('Conductivity-weighted τeff (ps)', 'Conductivity-weighted τeff (ps)'), particle = ctx.metric('Particle-average τ (ps)', 'Particle-average τ (ps)');
    const mobility = ctx.metric('μ=eτeff/m* (m²/V s)', 'μ=eτeff/m* (m²/V s)'), ratio = ctx.metric('Particle / conductivity average', 'Particle average / conductivity average');
    function render() {
      const t = Number(temperature.value), r = Number(exponent.value), p = thermalAverage(t, r);
      transport.textContent = ctx.format(p.tauEffective * 1e12, 6); mobility.textContent = ctx.format(p.mobility, 6);
      particle.textContent = p.tauParticle === null ? ctx.text('Diverges at K→0', 'K→0에서 발산') : ctx.format(p.tauParticle * 1e12, 6);
      ratio.textContent = p.particleToTransport === null ? ctx.text('Not finite', '유한하지 않음') : ctx.format(p.particleToTransport, 4);
      const bars = [{ x: 1, y: p.transportRatio, width: .45, color: 'gold', label: 'τeff/τref' }];
      if (p.particleRatio !== null) bars.push({ x: 2, y: p.particleRatio, width: .45, color: 'blue', label: '⟨τ⟩particle/τref' });
      ctx.view.replaceChildren(ctx.plot({ x: [0, 8], y: [0, .55], xTicks: [0, 2, 4, 6, 8], yTicks: [0, .25, .5],
        title: ctx.text('The two normalized energy weights are different', 'Carrier 수를 세는 weight와 전류에 쓰는 weight는 다릅니다'), xLabel: 'y = K/(kBT)', yLabel: ctx.text('Density per unit y', 'y에 대한 확률밀도'),
        series: [line(samples(0, 8, 241, y => mbWeight(y)), 'blue', 'wp=√y exp(−y)/Γ(3/2)'), line(samples(0, 8, 241, y => mbWeight(y, true)), 'gold', 'wtr=y^(3/2) exp(−y)/Γ(5/2)')] }),
      ctx.plot({ x: [.5, 2.5], ...positiveAxis(Math.max(...bars.map(b => b.y)) * 1.12), xTicks: [1, 2],
        title: p.particleRatio === null ? ctx.text('Finite conductivity average; divergent particle average is omitted', 'Conductivity average는 유한하지만 particle average는 발산합니다') : ctx.text('Integrate τ(K) with each weight', '같은 τ(K)를 서로 다른 weight로 적분합니다'),
        xLabel: ctx.text('1: conductivity weight; 2: particle weight', '1: conductivity weight, 2: particle weight'), yLabel: 'Average time / τref', bars }),
      explanation(ctx, 'An isotropic 3D parabolic band and a nondegenerate Maxwell–Boltzmann gas are fixed assumptions. The particle weight is √y exp(−y)/Γ(3/2); the conductivity weight has an extra y from v² and is y^(3/2) exp(−y)/Γ(5/2). Each integrates to one on 0<y<∞. The plot ends at 8, but the displayed averages use the full analytic integrals, not a truncated numerical average.', '방향에 따라 mass가 달라지지 않는 3D parabolic band와 nondegenerate Maxwell–Boltzmann gas를 가정합니다. Particle weight는 √y exp(−y)/Γ(3/2)입니다. Conductivity weight에는 v²에 따른 y가 하나 더 들어가므로 y^(3/2) exp(−y)/Γ(5/2)가 됩니다. 두 weight 모두 0<y<∞에서 적분하면 1입니다. 그림은 y=8까지만 보여주지만, 평균값은 전체 구간의 analytic integral로 계산합니다.'),
      explanation(ctx, 'With τref fixed at the Section 15 value (0.670864 ps at Kref=50 meV), τeff/τref=(kBT/Kref)^r Γ(r+5/2)/Γ(5/2). The particle formula replaces 5/2 by 3/2. Convergence near K=0 requires r>−5/2 for conductivity, but r>−3/2 for the particle average. Thus one can exist while the other diverges. This lab holds τref fixed as T changes; it does not include the additional 1/T scattering-time factor of the acoustic model.', 'τref는 15절의 값인 0.670864 ps(Kref=50 meV)로 고정합니다. τeff/τref=(kBT/Kref)^r Γ(r+5/2)/Γ(5/2)이며, particle average는 5/2 대신 3/2를 씁니다. K=0 부근의 적분이 유한하려면 conductivity에는 r>−5/2, particle average에는 r>−3/2가 필요합니다. 따라서 conductivity average만 유한한 경우도 있습니다. T를 바꿔도 τref는 고정하므로, acoustic 모델의 추가적인 1/T dependence는 포함하지 않습니다.'));
      ctx.status.textContent = ctx.text(`r=${r}, T=${t} K; m*=0.2m0. Γ here is the gamma function, not a scattering rate. For r=0 both averages equal τref. For r=−1/2, the particle average is 1.5 times τeff.`, `r=${r}, T=${t} K, m*=0.2m0입니다. 여기서 Γ는 scattering rate가 아니라 gamma function입니다. r=0이면 두 평균 모두 τref이며, r=−1/2이면 particle average가 τeff의 1.5배입니다.`);
    }
    preset(ctx, 'Constant τ', '에너지와 무관한 τ', () => { exponent.value = 0; }, render);
    preset(ctx, 'Short-range impurity example', 'Short-range impurity 예', () => { exponent.value = -.5; temperature.value = 300; }, render);
    preset(ctx, 'Different convergence bounds', '두 적분의 convergence 비교', () => { exponent.value = -2; }, render);
    ctx.onChange(render);
    share(ctx, 'drude-average', ['temperature', 'exponent'], () => ({ temperature: Number(temperature.value), exponent: Number(exponent.value) }),
      s => grid(s.temperature, 50, 600, 10) && typeof s.exponent === 'number' && exponents.includes(s.exponent), s => { temperature.value = s.temperature; exponent.value = s.exponent; }, render);
  });
})();
