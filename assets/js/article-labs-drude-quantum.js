/* Bounded quantum teaching models: an envelope packet, finite-time transitions,
   and the Fourier transform of a normalized one-dimensional Gaussian potential. */
(function () {
  'use strict';
  const HBAR_EV_FS = 0.6582119569509067;
  const inRange = (x, lo, hi) => Number.isFinite(x) && x >= lo && x <= hi;
  const onGrid = (x, lo, hi, step) => inRange(x, lo, hi) && Math.abs((x - lo) / step - Math.round((x - lo) / step)) < 1e-7;
  const exactKeys = (s, keys) => s !== null && typeof s === 'object' && !Array.isArray(s) && Object.keys(s).length === keys.length && keys.every(key => Object.prototype.hasOwnProperty.call(s, key));

  function packetState(alpha, s) {
    if (!inRange(alpha, -1, 1) || !inRange(s, 0, 3)) throw new Error('Invalid packet settings.');
    const width = Math.hypot(1, s);
    // xi=x/Δx0, s=t/t0, t0=2m*Δx0²/ℏ, alpha=F t0²/(2m*Δx0).
    // The initial real Gaussian has <x>=<v>=0 and variance Δx0².
    return { center: alpha * s * s, width, velocity: 2 * alpha * s, acceleration: 2 * alpha, peak: 1 / (Math.sqrt(2 * Math.PI) * width), allSpaceProbability: 1 };
  }
  function packetDensity(xi, alpha, s) {
    if (!inRange(xi, -1000, 1000)) throw new Error('Invalid normalized position.');
    const p = packetState(alpha, s);
    return p.peak * Math.exp(-.5 * ((xi - p.center) / p.width) ** 2);
  }
  function sinc(z) {
    if (!inRange(z, -1e6, 1e6)) throw new Error('Invalid sinc argument.');
    return Math.abs(z) < 1e-4 ? 1 - z * z / 6 + z ** 4 / 120 : Math.sin(z) / z;
  }
  function transitionProbability(detuningEV, couplingMeV, timeFs) {
    if (!inRange(detuningEV, -100, 100) || !inRange(couplingMeV, 0, 1) || !inRange(timeFs, 0, 50)) throw new Error('Invalid transition settings.');
    const amplitudeScale = couplingMeV * .001 * timeFs / HBAR_EV_FS;
    return amplitudeScale ** 2 * sinc(detuningEV * timeFs / (2 * HBAR_EV_FS)) ** 2;
  }
  function transitionSummary(couplingMeV, timeFs, detuningEV = 0) {
    const probeProbability = transitionProbability(detuningEV, couplingMeV, timeFs);
    return {
      probeProbability,
      peakProbability: transitionProbability(0, couplingMeV, timeFs),
      // This is ∫P(ΔE,t)dΔE, with units of eV, NOT a total probability.
      // Counting final states requires their density and occupation factors.
      energyAreaEV: 2 * Math.PI * (couplingMeV * .001) ** 2 * timeFs / HBAR_EV_FS,
      energyScaleEV: timeFs === 0 ? null : HBAR_EV_FS / timeFs
    };
  }
  function gaussianStrength(width, mode) {
    if (!inRange(width, .5, 3) || !['integral', 'peak'].includes(mode)) throw new Error('Invalid Gaussian settings.');
    // width=ell/ell0; gref>0 is the integrated strength at ell=ell0.
    // Fixing U(0) instead makes g/gref=ell/ell0.
    return mode === 'integral' ? 1 : width;
  }
  function gaussianPotential(xi, width, mode = 'integral') {
    if (!inRange(xi, -1000, 1000)) throw new Error('Invalid normalized position.');
    return gaussianStrength(width, mode) / (Math.sqrt(2 * Math.PI) * width) * Math.exp(-.5 * (xi / width) ** 2);
  }
  function gaussianFourier(q, width, mode = 'integral') {
    if (!inRange(q, -100, 100)) throw new Error('Invalid wavevector transfer.');
    const strength = gaussianStrength(width, mode), amplitude = strength * Math.exp(-.5 * (q * width) ** 2);
    return { strength, amplitude, rateFactor: amplitude * amplitude, peakPotential: strength / (Math.sqrt(2 * Math.PI) * width) };
  }
  const api = { HBAR_EV_FS, packetState, packetDensity, sinc, transitionProbability, transitionSummary, gaussianStrength, gaussianPotential, gaussianFourier };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const labs = window.ArticleLabs;
  const samples = (lo, hi, count, fn) => Array.from({ length: count }, (_, i) => { const x = lo + (hi - lo) * i / (count - 1); return [x, fn(x)]; });
  const line = (points, color, label, dash = false) => ({ points, color, label, dash });
  const range = (ctx, en, ko, min, max, step, value) => ctx.control(en, ko, { type: 'range', min, max, step, value });
  const explanation = (ctx, en, ko) => ctx.el('p', { class: 'drude-explanation' }, ctx.text(en, ko));
  const value = x => x === 0 ? '0' : Math.abs(x) < .001 ? x.toExponential(3) : String(Number(x.toFixed(4)));
  function share(ctx, id, read, valid, write, render) {
    ctx.stateLink(id, { read, write: s => {
      // No control, chart or metric changes until the entire payload is valid.
      if (!valid(s)) return false;
      write(s); render(); return true;
    } });
  }

  labs.register('drude-packet', ctx => {
    ctx.root.classList.add('drude-lab', 'al-workshop');
    const force = range(ctx, 'External force α', '외력 α', -1, 1, .1, 1);
    const time = range(ctx, 'Time s = t/t0', '시간 s = t/t0', 0, 3, .1, 1);
    const center = ctx.metric('Center ⟨x⟩/Δx0', '중심 ⟨x⟩/Δx0');
    const width = ctx.metric('Width Δx(t)/Δx0', '폭 Δx(t)/Δx0');
    const velocity = ctx.metric('Center velocity ⟨v⟩t0/Δx0', '중심의 속도 ⟨v⟩t0/Δx0');
    const normalization = ctx.metric('Probability over all x', '모든 위치에서 적분한 확률');
    const read = () => ({ alpha: Number(force.value), s: Number(time.value) });
    const valid = s => exactKeys(s, ['alpha', 's']) && onGrid(s.alpha, -1, 1, .1) && onGrid(s.s, 0, 3, .1);
    const write = s => { force.value = s.alpha; time.value = s.s; };
    function render() {
      const s = read(), p = packetState(s.alpha, s.s);
      center.textContent = value(p.center); width.textContent = value(p.width); velocity.textContent = value(p.velocity); normalization.textContent = '1';
      const lo = Math.floor(Math.min(-5, p.center - 5 * p.width)), hi = Math.ceil(Math.max(5, p.center + 5 * p.width));
      ctx.view.replaceChildren(
        ctx.plot({ x: [lo, hi], y: [0, .42], xLabel: 'x/Δx0', yLabel: 'Δx0 |ψenv|²', title: ctx.text('The envelope moves and spreads', 'Envelope의 이동과 퍼짐'),
          series: [line(samples(lo, hi, 301, x => packetDensity(x, 0, 0)), 'muted', ctx.text('Initial probability density', '초기 확률밀도'), true), line(samples(lo, hi, 301, x => packetDensity(x, s.alpha, s.s)), 'blue', ctx.text('Probability density at selected time', '선택한 시간의 확률밀도'))],
          markers: [{ x: p.center, y: p.peak, color: 'blue', label: ctx.text('Center', '중심') }], yTicks: [0, .2, .4] }),
        ctx.plot({ x: [0, 3], y: [-10, 10], xLabel: 's = t/t0', yLabel: ctx.text('Length / Δx0', '길이 / Δx0'), title: ctx.text('Force moves the center; dispersion sets the width', '외력은 중심을 움직이고, dispersion은 폭을 넓힙니다'),
          series: [line(samples(0, 3, 121, t => packetState(s.alpha, t).center), 'blue', '⟨x⟩/Δx0 = αs²'), line(samples(0, 3, 121, t => packetState(s.alpha, t).width), 'gold', 'Δx(t)/Δx0 = √(1+s²)')],
          markers: [{ x: s.s, y: p.center, color: 'blue', label: 's' }, { x: s.s, y: p.width, color: 'gold', label: '' }], xTicks: [0, 1, 2, 3], yTicks: [-10, -5, 0, 5, 10] }),
        explanation(ctx, 'Set α=0: the center stays put, but the packet still spreads. Reversing α reverses the center motion without changing the width. These are probability distributions and expectation values, not paths of individual electrons.', 'α=0으로 두면 중심은 그대로 있어도 packet은 퍼집니다. α의 부호를 바꾸면 중심의 이동 방향만 바뀌고 폭은 같습니다. 곡선은 확률분포이고 중심은 expectation value입니다. 전자 하나의 경로를 그린 것이 아닙니다.')
      );
      ctx.status.textContent = ctx.text('Exact within the stated quadratic-envelope Hamiltonian; no scattering.', '아래의 quadratic-envelope Hamiltonian에 대한 정확한 해입니다. Scattering은 포함하지 않습니다.');
    }
    const preset = (en, ko, s) => ctx.button(en, ko, () => { write(s); render(); });
    preset('Spread without force', '외력 없이 퍼짐', { alpha: 0, s: 2 });
    preset('Drive to the right', '오른쪽으로 이동', { alpha: 1, s: 2 });
    preset('Drive to the left', '왼쪽으로 이동', { alpha: -1, s: 2 });
    ctx.note('Henv=−ℏ²∂x²/(2m*)−Fx, m*>0 constant, with a real initial Gaussian ψenv∝exp[−x²/(4Δx0²)], ⟨x⟩=⟨v⟩=0. Δx0 is the initial position standard deviation, not a conductivity. t0=2m*Δx0²/ℏ and α=F t0²/(2m*Δx0). Then ⟨x⟩/Δx0=αs² and Δx(t)/Δx0=√(1+s²). The packet is an ideal parabolic-band envelope, not a resolved lattice wavefunction. No scattering, interband transitions or Berry-curvature terms. The finite plot window is not used to renormalize the density.', 'Henv=−ℏ²∂x²/(2m*)−Fx를 사용하며 m*>0은 상수입니다. 초기 상태는 실수 Gaussian ψenv∝exp[−x²/(4Δx0²)]이고 ⟨x⟩=⟨v⟩=0입니다. Δx0는 초기 위치 분포의 standard deviation입니다. t0=2m*Δx0²/ℏ, α=F t0²/(2m*Δx0)로 두면 ⟨x⟩/Δx0=αs², Δx(t)/Δx0=√(1+s²)입니다. 이상적인 parabolic band의 envelope를 계산하며 lattice 내부의 wavefunction은 표시하지 않습니다. Scattering, interband transition, Berry curvature는 포함하지 않습니다. 그래프에 보이는 범위만으로 확률밀도를 다시 normalize하지 않습니다.');
    ctx.onChange(render);
    share(ctx, 'drude-packet', read, valid, write, render);
  });

  labs.register('drude-transition', ctx => {
    ctx.root.classList.add('drude-lab', 'al-workshop');
    const coupling = range(ctx, 'Matrix-element magnitude |M| (meV)', 'Matrix element의 크기 |M| (meV)', 0, 1, .1, 1);
    const time = range(ctx, 'Time t (fs)', '시간 t (fs)', 0, 50, 1, 25);
    const probe = range(ctx, 'Probe detuning ΔE (meV)', '확인할 detuning ΔE (meV)', -150, 150, 5, 0);
    const reference = ctx.control('Compare with t = 25 fs', 't = 25 fs와 비교', { type: 'checkbox', value: true });
    const peak = ctx.metric('Resonant P (dimensionless)', 'Resonance에서의 P (무차원)');
    const probability = ctx.metric('P at selected ΔE', '선택한 ΔE에서의 P');
    const area = ctx.metric('Full energy area ∫P dΔE (meV)', '전체 energy 적분 ∫P dΔE (meV)');
    const scale = ctx.metric('Energy scale ℏ/t (meV)', 'Energy scale ℏ/t (meV)');
    const read = () => ({ couplingMeV: Number(coupling.value), timeFs: Number(time.value), probeMeV: Number(probe.value), reference: reference.checked });
    const valid = s => exactKeys(s, ['couplingMeV', 'timeFs', 'probeMeV', 'reference']) && onGrid(s.couplingMeV, 0, 1, .1) && onGrid(s.timeFs, 0, 50, 1) && onGrid(s.probeMeV, -150, 150, 5) && typeof s.reference === 'boolean';
    const write = s => { coupling.value = s.couplingMeV; time.value = s.timeFs; probe.value = s.probeMeV; reference.checked = s.reference; };
    function render() {
      const s = read(), p = transitionSummary(s.couplingMeV, s.timeFs, s.probeMeV / 1000);
      peak.textContent = value(p.peakProbability); probability.textContent = value(p.probeProbability); area.textContent = value(p.energyAreaEV * 1000);
      scale.textContent = p.energyScaleEV === null ? ctx.text('Not defined at t=0', 't=0에서는 정의하지 않음') : value(p.energyScaleEV * 1000);
      const curves = [line(samples(-150, 150, 401, e => transitionProbability(e / 1000, s.couplingMeV, s.timeFs) * 1000), 'blue', ctx.text('Selected time', '선택한 시간'))];
      if (s.reference) curves.push(line(samples(-150, 150, 401, e => transitionProbability(e / 1000, s.couplingMeV, 25) * 1000), 'gold', 't = 25 fs', true));
      ctx.view.replaceChildren(
        ctx.plot({ x: [-150, 150], y: [0, 6], xLabel: 'ΔE = Ef−Ei (meV)', yLabel: '10³ P(ΔE,t)', title: ctx.text('A finite observation time gives a finite-width peak', '유한한 시간에는 peak의 폭도 유한합니다'), series: curves,
          markers: [{ x: s.probeMeV, y: p.probeProbability * 1000, color: 'blue', label: ctx.text('Probe', '선택한 ΔE') }], xTicks: [-100, -50, 0, 50, 100, 150], yTicks: [0, 2, 4, 6] }),
        ctx.plot({ x: [0, 50], y: [0, .5], xLabel: 't (fs)', yLabel: '∫P dΔE (meV)', title: ctx.text('The full energy area grows linearly with time', '전체 energy 적분은 시간에 비례합니다'),
          series: [line(samples(0, 50, 101, t => transitionSummary(s.couplingMeV, t).energyAreaEV * 1000), 'blue', '2π|M|²t/ℏ')],
          markers: [{ x: s.timeFs, y: p.energyAreaEV * 1000, color: 'blue', label: 't' }], xTicks: [0, 25, 50], yTicks: [0, .25, .5] }),
        explanation(ctx, 'At ΔE=0, doubling t multiplies P by four. At the same time the peak narrows, so its full energy integral only doubles. That integral has energy units: it is not a total transition probability. A continuum final-state sum needs the density of states. For an electron gas, appropriate occupation and Pauli-blocking factors also enter.', 'ΔE=0에서 시간을 두 배로 늘리면 P는 네 배가 됩니다. 동시에 peak가 좁아지므로 전체 energy 범위에서 적분한 값은 두 배만 됩니다. 이 적분의 단위는 에너지이며 전체 transition probability가 아닙니다. Continuum의 final state를 합하려면 density of states가 필요합니다. Electron gas에서는 occupation과 Pauli blocking도 반영해야 합니다.')
      );
      ctx.status.textContent = ctx.text('First-order perturbation result; every allowed resonant P is below 0.006. The lower plot integrates over all ΔE, not just the displayed window.', 'First-order perturbation 결과입니다. 허용된 모든 설정에서 resonance의 P는 0.006보다 작습니다. 아래 그래프는 위에 보이는 범위만이 아니라 모든 ΔE에서 적분한 값입니다.');
    }
    const preset = (en, ko, s) => ctx.button(en, ko, () => { write(s); render(); });
    preset('Before evolution', 't=0에서 시작', { couplingMeV: 1, timeFs: 0, probeMeV: 0, reference: true });
    preset('Double 25 fs to 50 fs', '25 fs에서 50 fs로', { couplingMeV: 1, timeFs: 50, probeMeV: 0, reference: true });
    preset('Halve |M|', '|M|을 절반으로', { couplingMeV: .5, timeFs: 50, probeMeV: 0, reference: true });
    ctx.note('A time-independent perturbation with constant matrix element M is switched on at t=0. First-order perturbation theory gives P=(|M|t/ℏ)² sinc²(ΔEt/(2ℏ)), sinc z=sin z/z, sinc 0=1, ℏ=0.6582119569509067 eV fs. Here ΔE=Ef−Ei. The exact full energy integral is 2π|M|²t/ℏ, an energy, even though P for an individual state is dimensionless. A total probability needs the final-state sum (or density-of-states integral); an electron gas additionally needs the appropriate occupation factors. Small probabilities for individual states do not by themselves guarantee a small summed probability: first-order theory also requires that sum to remain much less than one. ℏ/t is an energy scale, not the full width at half maximum.', '시간에 따라 변하지 않는 perturbation을 t=0에서 켜고 matrix element M은 일정하게 둡니다. First-order perturbation theory에서 P=(|M|t/ℏ)² sinc²(ΔEt/(2ℏ))이며 sinc z=sin z/z, sinc 0=1, ℏ=0.6582119569509067 eV fs입니다. ΔE=Ef−Ei입니다. 전체 energy 적분은 정확히 2π|M|²t/ℏ입니다. 개별 state의 P는 무차원이지만 이 적분은 에너지 단위를 가집니다. Total probability를 구하려면 final state를 합하거나 density of states를 포함해 적분해야 하며, electron gas에서는 occupation factor도 필요합니다. 각 state로의 확률이 작아도 그 합까지 작다는 보장은 없습니다. First-order theory를 적용하려면 전체 transition probability도 1보다 훨씬 작아야 합니다. ℏ/t는 폭을 나타내는 energy scale이지 FWHM은 아닙니다.');
    ctx.onChange(render);
    share(ctx, 'drude-transition', read, valid, write, render);
  });

  labs.register('drude-gaussian', ctx => {
    ctx.root.classList.add('drude-lab', 'al-workshop');
    const width = range(ctx, 'Width λ = ℓ/ℓ0', '폭 λ = ℓ/ℓ0', .5, 3, .1, 1);
    const transfer = range(ctx, 'Wavevector transfer Q = qℓ0', 'Wavevector transfer Q = qℓ0', 0, 4, .05, 1);
    const mode = ctx.control('What stays fixed?', '무엇을 일정하게 둘까요?', { type: 'select', value: 'integral', options: [
      { value: 'integral', en: 'Integrated strength g', ko: '적분한 세기 g' }, { value: 'peak', en: 'Peak potential U(0)', ko: 'Potential의 peak U(0)' }
    ] });
    const strength = ctx.metric('Integrated strength g/gref', '적분한 세기 g/gref');
    const potential = ctx.metric('Peak U(0)ℓ0/gref', 'Peak U(0)ℓ0/gref');
    const amplitude = ctx.metric('Fourier amplitude Ũ(q)/gref', 'Fourier amplitude Ũ(q)/gref');
    const rate = ctx.metric('Relative rate factor |Ũ(q)/gref|²', '상대적인 rate factor |Ũ(q)/gref|²');
    const read = () => ({ width: Number(width.value), q: Number(transfer.value), mode: mode.value });
    const valid = s => exactKeys(s, ['width', 'q', 'mode']) && onGrid(s.width, .5, 3, .1) && onGrid(s.q, 0, 4, .05) && ['integral', 'peak'].includes(s.mode);
    const write = s => { width.value = s.width; transfer.value = s.q; mode.value = s.mode; };
    function render() {
      const s = read(), p = gaussianFourier(s.q, s.width, s.mode), top = Math.max(1, p.strength, p.strength ** 2);
      strength.textContent = value(p.strength); potential.textContent = value(p.peakPotential); amplitude.textContent = value(p.amplitude); rate.textContent = value(p.rateFactor);
      ctx.view.replaceChildren(
        ctx.plot({ x: [-10, 10], y: [0, .85], xLabel: 'ξ = x/ℓ0', yLabel: 'U(x)ℓ0/gref', title: ctx.text('Real-space potential: choose what stays fixed', '일정하게 유지할 값을 정하고 potential을 비교하세요'),
          series: [line(samples(-10, 10, 301, x => gaussianPotential(x, 1, s.mode)), 'muted', 'ℓ = ℓ0', true), line(samples(-10, 10, 301, x => gaussianPotential(x, s.width, s.mode)), 'blue', ctx.text('Selected width', '선택한 폭'))], xTicks: [-5, 0, 5, 10], yTicks: [0, .4, .8] }),
        ctx.plot({ x: [0, 4], y: [0, top * 1.05], xLabel: 'Q = qℓ0', yLabel: ctx.text('Normalized amplitude / squared amplitude', 'Normalized amplitude / amplitude의 제곱'), title: ctx.text('The golden rule uses the square, not the amplitude', 'Golden rule에는 amplitude의 제곱이 들어갑니다'),
          series: [line(samples(0, 4, 201, q => gaussianFourier(q, s.width, s.mode).amplitude), 'blue', 'Ũ/gref'), line(samples(0, 4, 201, q => gaussianFourier(q, s.width, s.mode).rateFactor), 'gold', '|Ũ/gref|²')],
          markers: [{ x: s.q, y: p.amplitude, color: 'blue', label: 'Q' }, { x: s.q, y: p.rateFactor, color: 'gold', label: '' }], xTicks: [0, 1, 2, 3, 4], yTicks: [0, top / 2, top] }),
        explanation(ctx, s.mode === 'integral' ? 'At fixed g, increasing the width lowers U(0), preserves Ũ(0)=g, and suppresses nonzero wavevector transfers. Compare amplitude and squared amplitude at Q=1 for λ=1 and 2.' : 'At fixed U(0), a wider potential has larger integrated strength: g/gref=λ. Its zero-transfer amplitude therefore grows. This is a different comparison from widening a scatterer at fixed g.', s.mode === 'integral' ? 'g를 고정하고 폭을 늘리면 U(0)는 낮아지지만 Ũ(0)=g는 그대로입니다. 0이 아닌 wavevector transfer는 더 강하게 억제됩니다. Q=1에서 λ=1과 2의 amplitude와 그 제곱을 비교해 보세요.' : 'U(0)를 고정한 채 폭을 늘리면 적분한 세기도 커져 g/gref=λ가 됩니다. 따라서 transfer가 0일 때의 amplitude도 커집니다. g를 고정하고 폭을 늘리는 경우와는 다른 비교입니다.')
      );
      ctx.status.textContent = ctx.text('Rate factor only: normalization length and all final-state/kinematic prefactors are held fixed. q is a wavevector transfer; ℏq is momentum transfer.', 'Rate factor만 비교합니다. Normalization length와 final-state·kinematic prefactor는 모두 고정합니다. q는 wavevector transfer이며 momentum transfer는 ℏq입니다.');
    }
    const preset = (en, ko, s) => ctx.button(en, ko, () => { write(s); render(); });
    preset('Wide, same g', '폭 2배, 같은 g', { width: 2, q: 1, mode: 'integral' });
    preset('Wide, same U(0)', '폭 2배, 같은 U(0)', { width: 2, q: 1, mode: 'peak' });
    preset('Zero transfer, same g', 'Transfer 0, 같은 g', { width: 2, q: 0, mode: 'integral' });
    ctx.note('One-dimensional Uℓ(x)=g exp[−x²/(2ℓ²)]/(√(2π)ℓ), Ũ(q)=∫U(x)e^(−iqx)dx=g exp[−q²ℓ²/2], q=k′−k. ℓ0 is a reference width and gref>0 is the integrated strength at ℓ=ℓ0. ξ=x/ℓ0, λ=ℓ/ℓ0, Q=qℓ0. At fixed g, g=gref. At fixed peak U(0)=gref/(√(2π)ℓ0), g=λgref. For plane waves normalized in length ℒ, M=Ũ/ℒ. With ℒ and the other golden-rule factors fixed, |Ũ/gref|² is a relative rate factor, not a complete rate, transport lifetime, or mobility. The Fourier transform is even, so only Q≥0 is plotted.', '1D potential Uℓ(x)=g exp[−x²/(2ℓ²)]/(√(2π)ℓ), Ũ(q)=∫U(x)e^(−iqx)dx=g exp[−q²ℓ²/2], q=k′−k를 사용합니다. ℓ0는 기준 폭이고 gref>0은 ℓ=ℓ0일 때의 적분한 세기입니다. ξ=x/ℓ0, λ=ℓ/ℓ0, Q=qℓ0입니다. g를 고정하면 g=gref입니다. Peak U(0)=gref/(√(2π)ℓ0)를 고정하면 g=λgref입니다. 길이 ℒ로 normalize한 plane wave의 matrix element는 M=Ũ/ℒ입니다. ℒ와 나머지 golden-rule factor를 고정했을 때 |Ũ/gref|²는 상대적인 rate factor입니다. 완전한 rate, transport lifetime, mobility를 계산한 값은 아닙니다. Fourier transform은 q의 부호에 대칭이므로 Q≥0만 표시합니다.');
    ctx.onChange(render);
    share(ctx, 'drude-gaussian', read, valid, write, render);
  });
})();
