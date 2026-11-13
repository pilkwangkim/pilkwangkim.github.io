/* A deliberately small elastic MS-BTE model; energies meV, lengths nm. */
(function () {
  'use strict';
  const A0 = 38.09982110968584; // hbar^2/(2 m0), meV nm^2; CODATA 2022 m0
  const KB = 0.08617333262145; // meV/K
  const G0 = 77.4809172986; // 2 e^2/h, microampere/volt
  const inRange = (v, lo, hi) => Number.isFinite(v) && v >= lo && v <= hi;
  const integer = (v, lo, hi) => Number.isInteger(v) && v >= lo && v <= hi;
  function packetOccupation(x, t) {
    if (!Number.isFinite(x) || !inRange(t, 0, 4)) throw new Error('Use a finite position and a time between 0 and 4 ps.');
    // Fixed velocity 20 nm/ps; width 12 nm. This is an occupation profile.
    return .1 + .8 * Math.exp(-((x - 20 * t + 30) ** 2) / (2 * 12 ** 2));
  }
  function relaxingPair(t, r) {
    if (!inRange(t, 0, 4) || !inRange(r, 0, 1)) throw new Error('Use 0–4 ps and a reciprocal rate of 0–1 ps^-1.');
    // Equal-speed branches at a common energy, uniform in space.
    const decay = Math.exp(-2 * r * t);
    return { plus: .5 + .3 * decay, minus: .5 - .3 * decay, mean: .5, difference: .6 * decay };
  }
  function fermi(energy, mu, temperature) {
    if (!Number.isFinite(energy) || !Number.isFinite(mu) || !inRange(temperature, 1, 2000)) throw new Error('Finite energies and positive temperature are required.');
    const z = (energy - mu) / (KB * temperature);
    return z >= 0 ? Math.exp(-z) / (1 + Math.exp(-z)) : 1 / (1 + Math.exp(z));
  }
  function softplus(z) { return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)); }
  function subbands(width = 10, height = 8, mass = .19, count = 8) {
    if (!inRange(width, 2, 100) || !inRange(height, 2, 100) || !inRange(mass, .01, 10) || !integer(count, 1, 16)) throw new Error('Invalid hard-wall geometry.');
    const modes = [];
    // The first count modes must occur within 1..count in each direction.
    for (let ny = 1; ny <= count; ny++) for (let nz = 1; nz <= count; nz++) modes.push({ ny, nz, edge: Math.PI ** 2 * A0 / mass * (ny ** 2 / width ** 2 + nz ** 2 / height ** 2) });
    return modes.sort((a, b) => a.edge - b.edge || a.ny - b.ny || a.nz - b.nz).slice(0, count);
  }
  function ballisticCurrent(edges, muL, muR, temperature) {
    if (!Array.isArray(edges) || !edges.length || !Array.from(edges).every(Number.isFinite) || !Number.isFinite(muL) || !Number.isFinite(muR) || !inRange(temperature, 1, 2000)) throw new Error('Invalid ballistic contacts.');
    const kT = KB * temperature;
    // Negative conventional charge current for a positive electron flux.
    return -G0 * kT / 1000 * edges.reduce((sum, edge) => sum + softplus((muL - edge) / kT) - softplus((muR - edge) / kT), 0);
  }
  function solveLinear(matrix, rhs) {
    const n = rhs.length, a = matrix.map((row, i) => [...row, rhs[i]]);
    for (let k = 0; k < n; k++) {
      let p = k;
      for (let i = k + 1; i < n; i++) if (Math.abs(a[i][k]) > Math.abs(a[p][k])) p = i;
      if (Math.abs(a[p][k]) < 1e-13) throw new Error('Singular transport system.');
      [a[k], a[p]] = [a[p], a[k]];
      for (let i = k + 1; i < n; i++) {
        const q = a[i][k] / a[k][k]; a[i][k] = 0;
        for (let j = k + 1; j <= n; j++) a[i][j] -= q * a[k][j];
      }
    }
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let value = a[i][n];
      for (let j = i + 1; j < n; j++) value -= a[i][j] * x[j];
      x[i] = value / a[i][i];
    }
    return x;
  }
  function conductances(modes, beta, gamma) {
    if (!integer(modes, 1, 2) || !inRange(beta, 0, .1) || !inRange(gamma, 0, .1)) throw new Error('Invalid elastic collision graph.');
    const states = 2 * modes;
    return Array.from({ length: states }, (_, a) => Array.from({ length: states }, (_, b) => a === b ? 0 : Math.floor(a / 2) === Math.floor(b / 2) ? beta : gamma / 2));
  }
  function symmetricTransmission(modes, length, beta, gamma, cells) {
    if (!integer(modes, 1, 2) || !inRange(length, 1, 1000) || !inRange(beta, 0, .1) || !inRange(gamma, 0, .1) || !integer(cells, 2, 36)) throw new Error('Invalid symmetric transmission benchmark.');
    // Equal mode occupations reduce the chosen graph to independent copies
    // of one pair with alpha=beta+(modes-1)*gamma/2. No current ratio is used.
    const alpha = beta + (modes - 1) * gamma / 2, dx = length / cells;
    const continuum = 1 / (1 + alpha * length);
    const grid = (1 + alpha * dx) / (1 + alpha * length + alpha * dx);
    return { alpha, continuum, grid, relativeErrorPercent: 100 * (grid / continuum - 1) };
  }
  function solveShell({ modes = 2, length = 100, beta = .005, gamma = .003, fL = .9, fR = .2, cells = 24 } = {}) {
    if (!integer(modes, 1, 2) || !inRange(length, 1, 1000) || !integer(cells, 2, 36)) throw new Error('Invalid shell transport settings.');
    const incoming = value => { const a = Array.isArray(value) ? Array.from(value) : Array(modes).fill(value); if (a.length !== modes || !a.every(v => inRange(v, 0, 1))) throw new Error('Invalid incoming occupations.'); return a; };
    const left = incoming(fL), right = incoming(fR);
    const kappa = conductances(modes, beta, gamma), states = 2 * modes, dx = length / cells, size = states * cells;
    const matrix = Array.from({ length: size }, () => Array(size).fill(0)), rhs = Array(size).fill(0);
    for (let i = 0; i < cells; i++) for (let a = 0; a < states; a++) {
      const row = i * states + a, positive = a % 2 === 0, upstream = i + (positive ? -1 : 1);
      matrix[row][row] += 1 / dx;
      if (upstream < 0 || upstream >= cells) rhs[row] += (positive ? left[Math.floor(a / 2)] : right[Math.floor(a / 2)]) / dx;
      else matrix[row][upstream * states + a] -= 1 / dx;
      for (let b = 0; b < states; b++) if (a !== b) { matrix[row][row] += kappa[a][b]; matrix[row][i * states + b] -= kappa[a][b]; }
    }
    const solution = solveLinear(matrix, rhs), occupation = Array.from({ length: cells }, (_, i) => solution.slice(i * states, (i + 1) * states));
    const modeFlux = Array.from({ length: cells + 1 }, (_, j) => Array.from({ length: modes }, (_, n) => (j === 0 ? left[n] : occupation[j - 1][2 * n]) - (j === cells ? right[n] : occupation[j][2 * n + 1])));
    const flux = modeFlux.map(row => row.reduce((sum, v) => sum + v, 0));
    return { x: occupation.map((_, i) => (i + .5) * dx), occupation, faceX: flux.map((_, i) => i * dx), flux, modeFlux, dx, kappa };
  }
  function collision(occupations, speeds, kappa) {
    if (!Array.isArray(occupations) || !Array.isArray(speeds) || !Array.isArray(kappa) || occupations.length < 2 || occupations.length !== speeds.length || kappa.length !== speeds.length || !Array.from(occupations).every(x => inRange(x, 0, 1)) || !Array.from(speeds).every(x => Number.isFinite(x) && x > 0) || !Array.from(kappa).every(row => Array.isArray(row) && row.length === speeds.length && Array.from(row).every(x => Number.isFinite(x) && x >= 0))) throw new Error('Invalid shell state.');
    if (kappa.some((row, a) => row[a] !== 0)) throw new Error('Elastic pair conductances must have zero diagonal.');
    if (!kappa.every((row, a) => row.every((k, b) => Math.abs(k - kappa[b][a]) < 1e-14))) throw new Error('Elastic pair conductances must be symmetric.');
    return occupations.map((f, a) => speeds[a] * kappa[a].reduce((sum, k, b) => sum + k * (occupations[b] - f), 0));
  }
  const questions = [
    { energy: 90, edges: [50, 110], answer: 1, en: 'An elastic collision keeps the electron at total energy 90 meV. Can it end in subband 2?', ko: '총에너지 90 meV인 전자가 elastic scattering을 겪습니다. Subband 2로 옮겨갈 수 있을까요?', choices: [['Yes: a collision can supply the missing 20 meV.', '네. Scattering이 부족한 20 meV를 공급합니다.'], ['No: subband 2 has no state at this total energy.', '아니요. 이 에너지에서 subband 2로 진행할 수 있는 state는 없습니다.'], ['Yes: the kinetic energy becomes −20 meV.', '네. Kinetic energy가 −20 meV가 됩니다.']], explanation: ['Elastic means no change in total electron energy. Since 90 < 110, the second subband is closed at this energy.', 'Elastic scattering에서는 전자의 총에너지가 바뀌지 않습니다. 90 < 110이므로 이 에너지에서 subband 2로 진행할 수 있는 state는 없습니다.'] },
    { energy: 130, edges: [50, 110], answer: 2, en: 'At total energy 130 meV, an elastic transition goes from subband 1 to subband 2. What is the final longitudinal kinetic energy?', ko: '총에너지 130 meV에서 전자가 elastic scattering으로 subband 1에서 2로 옮겨갑니다. 옮겨 간 뒤 longitudinal kinetic energy는 얼마일까요?', choices: [['80 meV', '80 meV'], ['130 meV', '130 meV'], ['20 meV', '20 meV']], explanation: ['The final kinetic energy is E − ε₂ = 130 − 110 = 20 meV. The original value was 80 meV; total energy, not longitudinal kinetic energy, is conserved.', '옮겨 간 뒤 kinetic energy는 E − ε₂ = 130 − 110 = 20 meV입니다. 처음에는 80 meV였습니다. 보존되는 양은 longitudinal kinetic energy가 아니라 총에너지입니다.'] },
    { energy: 130, edges: [50, 110], answer: 0, en: 'Both reservoirs have the same chemical potential and temperature. Which occupation is an equilibrium solution, even if the two subband speeds differ?', ko: '양쪽 reservoir의 chemical potential과 온도가 같습니다. 두 subband의 speed가 서로 다를 때, equilibrium에서는 어떤 관계가 성립해야 할까요?', choices: [['The same Fermi occupation f(E) in every open direction and subband.', '같은 에너지에서는 subband와 진행 방향에 관계없이 Fermi occupation f(E)가 같습니다.'], ['The same number of electrons in every subband.', '모든 subband의 전자 수가 같습니다.'], ['Only right-moving states are occupied.', '오른쪽으로 가는 state에만 전자가 있습니다.']], explanation: ['Equal occupations make every pair difference zero. Population per energy is weighted by 1/v, so equal occupation does not imply equal populations.', 'Occupation이 같으면 각 state 쌍에서 f_b − f_a = 0입니다. 같은 에너지 구간의 state 수는 1/v에 비례합니다. 따라서 occupation이 같아도 전자 수는 다를 수 있습니다.'] },
    { energy: 130, edges: [50, 110], answer: 1, en: 'For the left contact at x=0, which distribution should the reservoir prescribe?', ko: '왼쪽 contact(x=0)에서는 어느 방향의 occupation을 boundary condition으로 지정해야 할까요?', choices: [['Both incoming f⁺ and outgoing f⁻.', '들어오는 f⁺와 나가는 f⁻를 모두 지정합니다.'], ['Only incoming f⁺; transport determines outgoing f⁻.', '들어오는 f⁺만 지정합니다. 나가는 f⁻는 transport 계산으로 구합니다.'], ['Only outgoing f⁻.', '나가는 f⁻만 지정합니다.']], explanation: ['The left reservoir supplies right-moving electrons. Left-moving electrons reaching it came from the device, so their occupation must be solved rather than independently fixed.', '왼쪽 reservoir에서 오른쪽으로 움직이는 전자가 소자로 들어옵니다. 왼쪽 contact로 빠져나오는 전자는 소자 내부를 거쳐 왔으므로, 그 occupation은 따로 고정하지 않고 transport 계산으로 구해야 합니다.'] }
  ];
  function quizProblem(index) { if (!integer(index, 0, questions.length - 1)) throw new Error('Invalid question.'); return questions[index]; }
  const api = { A0, KB, G0, packetOccupation, relaxingPair, fermi, subbands, ballisticCurrent, solveShell, conductances, symmetricTransmission, collision, quizProblem };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const labs = window.ArticleLabs;
  function setup(ctx) { ctx.root.classList.add('msbte-lab', 'al-workshop'); }
  function paragraph(ctx, en, ko) { return ctx.el('p', { class: 'msbte-explanation' }, ctx.text(en, ko)); }
  function modelLink(ctx, id, read, valid, write, render) { ctx.stateLink(id, { read, write: s => { const keys = Object.keys(read()); if (!s || typeof s !== 'object' || Array.isArray(s) || Object.keys(s).length !== keys.length || !keys.every(key => Object.prototype.hasOwnProperty.call(s, key)) || !valid(s)) return false; write(s); render(); return true; } }); }
  function numberControl(ctx, en, ko, min, max, step, value) { return ctx.control(en, ko, { type: 'range', min, max, step, value }); }
  const sampled = (lo, hi, count, fn) => Array.from({ length: count }, (_, j) => { const x = lo + (hi - lo) * j / (count - 1); return [x, fn(x)]; });
  const colors = ['blue', 'gold', 'red'];
  const stableExp = (value, digits = 2) => value.toExponential(digits).replace(/e([+-])(\d)$/, 'e$10$2');
  labs.register('msbte-basics', ctx => {
    setup(ctx);
    const mode = ctx.control('Separate process', '계산할 예제', { type: 'select', value: 'flight', options: [{ value: 'flight', en: 'Collisionless transport', ko: 'Free flight' }, { value: 'relaxation', en: 'Uniform elastic relaxation', ko: 'Relaxation' }] });
    const time = numberControl(ctx, 'Time t (ps)', '시간 t (ps)', 0, 4, .1, 0);
    const rate = numberControl(ctx, 'Reversal rate r (ps⁻¹)', '방향 전환 rate r (ps⁻¹)', 0, 1, .05, .4);
    // The two initial-value problems have different observables. Keep their
    // native definition-list rows separate and hide the inactive set.
    function readout(key, en, ko) {
      const output = ctx.el('output');
      const row = ctx.el('div', { 'data-basics-metric': key }, [ctx.el('dt', {}, ctx.text(en, ko)), ctx.el('dd', {}, output)]);
      ctx.metrics.append(row); return { row, output };
    }
    const centre = readout('centre', 'Profile centre (nm)', 'Profile 중심 (nm)'), peak = readout('peak', 'Peak occupation', '최대 occupation');
    const plus = readout('plus', 'Occupation f⁺(t)', 'Occupation f⁺(t)'), minus = readout('minus', 'Occupation f⁻(t)', 'Occupation f⁻(t)');
    const mean = readout('mean', 'Mean (f⁺ + f⁻)/2', '평균 (f⁺ + f⁻)/2'), difference = readout('difference', 'Difference f⁺ − f⁻', '차이 f⁺ − f⁻');
    function render() {
      const t = Number(time.value), r = Number(rate.value), flight = mode.value === 'flight';
      rate.disabled = flight;
      [centre, peak].forEach(item => { item.row.hidden = !flight; });
      [plus, minus, mean, difference].forEach(item => { item.row.hidden = flight; });
      centre.output.textContent = (-30 + 20 * t).toFixed(1); peak.output.textContent = '0.9000';
      const pair = relaxingPair(t, r);
      plus.output.textContent = pair.plus.toFixed(4); minus.output.textContent = pair.minus.toFixed(4);
      mean.output.textContent = pair.mean.toFixed(4); difference.output.textContent = pair.difference.toFixed(4);
      if (flight) {
        ctx.view.replaceChildren(
          ctx.plot({ x: [-100, 140], y: [0, 1], xTicks: [-100, -40, 20, 80, 140], title: ctx.text('Collisionless transport: the occupation profile shifts without changing shape', 'Collision이 없으면 occupation profile은 모양을 유지하며 이동합니다'), xLabel: 'Position x (nm)', yLabel: 'Occupation f', series: [{ points: sampled(-100, 140, 241, x => packetOccupation(x, 0)), color: 'muted', dash: true, label: ctx.text('Initial: t = 0 ps', '처음: t = 0 ps') }, { points: sampled(-100, 140, 241, x => packetOccupation(x, t)), color: 'blue', label: `t = ${t.toFixed(1)} ps` }] }),
          paragraph(ctx, 'The velocity is fixed at +20 nm/ps, with no force or collisions. The solution f(x,t)=0.1+0.8 exp[−(x−20t+30)²/(2·12²)] carries the initial occupation profile to the right. Its peak moves from −30 nm to −30+20t nm without changing its height or width. This is an occupation profile, not a single-electron wavefunction or a probability density normalized to one.', 'Velocity는 +20 nm/ps이고 force와 collision은 없습니다. 해는 f(x,t)=0.1+0.8 exp[−(x−20t+30)²/(2·12²)]입니다. 이 식에 따라 처음의 occupation profile이 오른쪽으로 이동합니다. 최대값과 폭은 그대로이고, 중심만 −30 nm에서 −30+20t nm로 이동합니다. 이 곡선은 전자 하나의 wavefunction이나 적분하면 1이 되는 probability density가 아니라 occupation profile입니다.')
        );
        ctx.status.textContent = ctx.text(`t=${t.toFixed(1)} ps; centre=${(-30 + 20 * t).toFixed(1)} nm. Here ∂t f+v∂x f=0: time variation at a fixed x balances spatial streaming. The rate control is inactive.`, `t=${t.toFixed(1)} ps, 중심=${(-30 + 20 * t).toFixed(1)} nm입니다. ∂t f+v∂x f=0이므로 시간 변화 항과 streaming 항이 서로 상쇄됩니다. 이 모드에서는 rate를 사용하지 않습니다.`);
      } else {
        ctx.view.replaceChildren(
          ctx.plot({ x: [0, 4], y: [0, 1], title: ctx.text('Uniform elastic relaxation: two occupations approach their conserved mean', 'Elastic relaxation: 평균은 일정하고 두 occupation의 차이는 줄어듭니다'), xLabel: 'Time (ps)', yLabel: 'Occupation f⁺, f⁻', series: [{ points: sampled(0, 4, 101, s => relaxingPair(s, r).plus), color: 'blue', label: 'f⁺' }, { points: sampled(0, 4, 101, s => relaxingPair(s, r).minus), color: 'gold', label: 'f⁻' }], markers: [{ x: t, y: pair.plus, color: 'blue', label: 'f⁺(t)' }, { x: t, y: pair.minus, color: 'gold', label: 'f⁻(t)' }] }),
          paragraph(ctx, 'This is a different initial-value problem: occupations are uniform in space, so there are no streaming gradients. Two branches at one energy have opposite velocities of equal magnitude and reciprocal elastic reversal rate r. Starting from f⁺(0)=0.8 and f⁻(0)=0.2, the exact solution is f±(t)=0.5±0.3 exp(−2rt). The mean stays 0.5, while the directional difference is 0.6 exp(−2rt). No contacts or wire boundary-value problem are being solved.', '이번에는 분포가 어느 위치에서나 같다고 두고, 방향별 occupation의 시간 변화만 계산합니다. f⁺와 f⁻가 각각 위치에 따라 변하지 않으므로 공간 미분 항은 0입니다. 두 branch는 에너지와 speed가 같고 진행 방향만 반대입니다. 두 방향의 elastic reversal rate를 모두 r로 둡니다. f⁺(0)=0.8, f⁻(0)=0.2에서 출발하면 해는 f±(t)=0.5±0.3 exp(−2rt)입니다. 평균은 0.5로 유지되고, 방향별 차이는 0.6 exp(−2rt)로 줄어듭니다. Contact 조건을 지정해 wire 내부의 분포를 구하는 boundary-value problem은 아닙니다.'),
          paragraph(ctx, 'r is an inverse-time rate in ps⁻¹. It is not the later inverse-length coefficient β in nm⁻¹. These two examples isolate transport and relaxation; the interface does not combine them into one simulation.', 'r의 단위는 시간의 역수인 ps⁻¹입니다. 뒤에서 사용할 inverse-length 계수 β(nm⁻¹)와는 다릅니다. 하나는 이동만, 다른 하나는 relaxation만 계산합니다. 두 과정을 동시에 계산하지는 않습니다.')
        );
        ctx.status.textContent = ctx.text(`t=${t.toFixed(1)} ps, r=${r.toFixed(2)} ps⁻¹. The sum f⁺+f⁻ stays 1; their difference relaxes at rate 2r. Set r=0 to keep the initial occupations unchanged.`, `t=${t.toFixed(1)} ps, r=${r.toFixed(2)} ps⁻¹입니다. 합 f⁺+f⁻는 1로 유지되고, 차이의 relaxation rate는 2r입니다. r=0이면 처음의 occupation이 그대로 유지됩니다.`);
      }
    }
    ctx.onChange(render);
    modelLink(ctx, 'msbte-basics', () => ({ mode: mode.value, time: Number(time.value), rate: Number(rate.value) }), s => ['flight', 'relaxation'].includes(s.mode) && inRange(s.time, 0, 4) && Math.abs(s.time * 10 - Math.round(s.time * 10)) < 1e-8 && inRange(s.rate, 0, 1) && Math.abs(s.rate * 20 - Math.round(s.rate * 20)) < 1e-8, s => { mode.value = s.mode; time.value = String(s.time); rate.value = String(s.rate); }, render);
  });
  function quizDiagram(ctx, index, q) {
    if (index === 2) return ctx.plot({ x: [-.5, 3.5], y: [0, 1], xTicks: [0, 1, 2, 3], title: ctx.text('Equilibrium at one total energy E', '총에너지 E를 고정했을 때의 equilibrium'), xLabel: ctx.text('States: 0=(1,+), 1=(1,−), 2=(2,+), 3=(2,−)', 'State: 0=(1,+), 1=(1,−), 2=(2,+), 3=(2,−)'), yLabel: ctx.text('Same occupation f(E), illustrated with f=0.4', '같은 occupation f(E), 예: f=0.4'), bars: [0, 1, 2, 3].map(x => ({ x, y: .4, width: .55, color: x < 2 ? 'blue' : 'gold' })), series: [] });
    if (index === 3) {
      const svg = ctx.svg('svg', { viewBox: '0 0 480 190', class: 'msbte-contact-diagram', role: 'img', 'aria-label': ctx.text('At the left contact, f plus enters the device from the reservoir and f minus exits toward the reservoir.', '왼쪽 contact에서 f plus는 reservoir에서 소자로 들어오고, f minus는 소자에서 reservoir로 나갑니다.') });
      svg.append(ctx.svg('title', {}, ctx.text('Prescribe inflow, solve outflow', '들어오는 값만 지정하고 나가는 값은 계산합니다')));
      svg.append(ctx.svg('rect', { x: 10, y: 20, width: 120, height: 140, rx: 4, class: 'msbte-reservoir' }), ctx.svg('rect', { x: 260, y: 20, width: 205, height: 140, rx: 4, class: 'msbte-device' }));
      svg.append(ctx.svg('text', { x: 70, y: 85, 'text-anchor': 'middle' }, 'L'), ctx.svg('text', { x: 70, y: 112, 'text-anchor': 'middle' }, 'reservoir'), ctx.svg('text', { x: 362, y: 98, 'text-anchor': 'middle' }, 'device'));
      svg.append(ctx.svg('path', { d: 'M130 65 H251 M250 58 L262 65 L250 72 Z', class: 'msbte-arrow-in' }), ctx.svg('path', { d: 'M260 125 H141 M142 118 L130 125 L142 132 Z', class: 'msbte-arrow-out' }));
      svg.append(ctx.svg('text', { x: 195, y: 48, 'text-anchor': 'middle' }, 'f⁺ →'), ctx.svg('text', { x: 195, y: 153, 'text-anchor': 'middle' }, '← f⁻'), ctx.svg('text', { x: 260, y: 184, 'text-anchor': 'middle' }, 'x = 0'));
      return svg;
    }
    const y = e => 230 - 1.25 * e;
    const description = ctx.text(`Subband edges are ${q.edges[0]} and ${q.edges[1]} meV. The dashed line is total energy ${q.energy} meV. Shaded columns contain propagating states only above each edge. The columns label subbands, not positions.`, `Subband edge는 ${q.edges[0]}, ${q.edges[1]} meV이고 점선은 총에너지 ${q.energy} meV입니다. 음영은 각 edge보다 에너지가 높은 propagating state를 나타냅니다. 두 열은 위치가 아니라 subband를 구분합니다.`);
    const svg = ctx.svg('svg', { viewBox: '0 0 400 300', class: 'msbte-energy-diagram', role: 'img', 'aria-label': description });
    svg.append(ctx.svg('title', {}, ctx.text('Available subbands at a fixed total energy', '이 총에너지에서 전자가 진행할 수 있는 subband')), ctx.svg('desc', {}, description));
    svg.append(ctx.svg('text', { x: 10, y: 18 }, 'Energy (meV)'), ctx.svg('line', { x1: 60, x2: 60, y1: 30, y2: 230, stroke: 'var(--al-muted)', 'stroke-width': 1.5 }));
    for (const tick of [0, 80, 160]) {
      svg.append(ctx.svg('line', { x1: 54, x2: 60, y1: y(tick), y2: y(tick), stroke: 'var(--al-muted)', 'stroke-width': 1.5 }), ctx.svg('text', { x: 47, y: y(tick) + 6, 'text-anchor': 'end', class: 'msbte-energy-tick' }, String(tick)));
    }
    q.edges.forEach((edge, n) => {
      const centre = n ? 300 : 135, color = n ? 'var(--al-gold)' : 'var(--al-blue)';
      svg.append(ctx.svg('rect', { x: centre - 34, y: 30, width: 68, height: y(edge) - 30, fill: color, opacity: .13 }));
      svg.append(ctx.svg('line', { x1: centre - 48, x2: centre + 48, y1: y(edge), y2: y(edge), stroke: color, 'stroke-width': 3 }));
      svg.append(ctx.svg('text', { x: centre, y: y(edge) - 6, 'text-anchor': 'middle', class: 'msbte-energy-edge' }, `ε${n ? '₂' : '₁'} = ${edge}`), ctx.svg('text', { x: centre, y: 260, 'text-anchor': 'middle' }, `Subband ${n + 1}`));
    });
    svg.append(ctx.svg('line', { x1: 72, x2: 382, y1: y(q.energy), y2: y(q.energy), stroke: 'var(--al-red)', 'stroke-width': 2.5, 'stroke-dasharray': '7 5' }), ctx.svg('text', { x: 384, y: y(q.energy) - 10, 'text-anchor': 'end', class: 'msbte-energy-total' }, `E = ${q.energy}`));
    q.edges.forEach((edge, n) => { if (q.energy > edge) svg.append(ctx.svg('circle', { cx: n ? 300 : 135, cy: y(q.energy), r: 5.5, fill: 'var(--al-red)' })); });
    svg.append(ctx.svg('rect', { x: 80, y: 280, width: 17, height: 13, fill: 'var(--al-blue)', opacity: .2 }), ctx.svg('text', { x: 106, y: 292, class: 'msbte-energy-key' }, 'Open states: E > εₙ'));
    return svg;
  }
  labs.register('msbte-confinement', ctx => {
    setup(ctx);
    const width = numberControl(ctx, 'Width W (nm)', 'Width W (nm)', 6, 20, .5, 10), level = ctx.control('Selected transverse mode', '표시할 transverse mode', { type: 'select', value: '0', options: [0, 1, 2].map(i => ({ value: String(i), en: 'Mode ' + (i + 1), ko: 'Mode ' + (i + 1) })) });
    const energy = ctx.metric('First three edges (meV)', '에너지가 낮은 세 subband edge (meV)'), geometry = ctx.metric('Cross section / effective mass', '단면 / effective mass');
    function render() {
      const w = Number(width.value), modes = subbands(w, 8, .19, 3), selected = modes[Number(level.value)];
      energy.textContent = modes.map((m, i) => (i + 1) + ': ' + m.edge.toFixed(2)).join('\n'); geometry.textContent = w.toFixed(1) + ' × 8.0 nm²; 0.19 m₀';
      ctx.view.replaceChildren(
        ctx.plot({ x: [-.5, .5], y: [0, 320], xTicks: [-.5, 0, .5], yTicks: [0, 80, 160, 240, 320], title: ctx.text('Subband energies on fixed physical axes', 'Wire 폭에 따른 subband dispersion'), xLabel: 'Longitudinal k (nm⁻¹)', yLabel: 'Total energy E (meV)', series: modes.map((m, i) => ({ points: sampled(-.5, .5, 81, k => m.edge + A0 / .19 * k * k), color: colors[i], label: '(' + m.ny + ',' + m.nz + ')' })),
          markers: [{ x: 0, y: selected.edge, color: colors[Number(level.value)], label: ctx.text('Selected mode edge', '선택한 mode의 edge') }] }),
        ctx.plot({ x: [0, 20], y: [0, .36], xTicks: [0, 5, 10, 15, 20], yTicks: [0, .1, .2, .3], title: ctx.text('Selected mode: marginal density along y', '선택한 mode의 y 방향 marginal density'), xLabel: 'y (nm)', yLabel: '∫ |φ(y,z)|² dz (nm⁻¹)', series: [{ points: sampled(0, 20, 801, y => y <= w ? 2 / w * Math.sin(selected.ny * Math.PI * y / w) ** 2 : 0), color: colors[Number(level.value)], label: 'nᵧ=' + selected.ny }, { points: [[w, 0], [w, .36]], color: 'muted', dash: true, label: 'Hard wall: y=W' }] }),
        ctx.plot({ x: [0, 8], y: [0, .36], xTicks: [0, 2, 4, 6, 8], yTicks: [0, .1, .2, .3], title: ctx.text('The z density distinguishes modes with the same nᵧ', 'nᵧ가 같아도 z 방향 분포는 다를 수 있습니다'), xLabel: 'z (nm)', yLabel: '∫ |φ(y,z)|² dy (nm⁻¹)', series: [{ points: sampled(0, 8, 321, z => 2 / 8 * Math.sin(selected.nz * Math.PI * z / 8) ** 2), color: colors[Number(level.value)], label: 'n_z=' + selected.nz }] }),
        paragraph(ctx, 'Height H=8 nm and m*=0.19m₀ stay fixed. All axis limits remain unchanged while W varies. Narrowing the wire raises its energies, compresses the y density and raises its peak; the z shape changes only when the selected n_z changes. Each marginal integrates to one. Outside 0<y<W the density is zero. Modes are ranked by energy, so the selected rank can change its (nᵧ,n_z) pair as W changes.', '높이 H=8 nm와 m*=0.19m₀는 고정합니다. W를 바꿔도 모든 축 범위는 그대로입니다. Wire의 폭을 줄이면 subband energy가 높아집니다. y 방향 probability density가 분포하는 범위도 좁아지면서 peak가 높아집니다. z 방향 모양은 선택한 n_z가 바뀔 때만 달라집니다. 두 marginal density는 각각 적분하면 1이며, 0<y<W 바깥의 density는 0입니다. Mode 번호는 에너지가 낮은 순서이므로 W에 따라 같은 번호의 (nᵧ,n_z)가 바뀔 수 있습니다.')
      );
      ctx.status.textContent = ctx.text('Selected (' + selected.ny + ',' + selected.nz + '), edge=' + selected.edge.toFixed(2) + ' meV. A selected edge is marked above; both transverse densities are shown even at an energy degeneracy.', '선택한 mode는 (' + selected.ny + ',' + selected.nz + ')이고 edge=' + selected.edge.toFixed(2) + ' meV입니다. 위 그래프의 점은 선택한 mode의 subband edge를 가리킵니다. 두 mode의 energy가 같더라도 y·z 방향 분포를 비교하면 구별할 수 있습니다.');
    }
    ctx.onChange(render);
    modelLink(ctx, 'msbte-confinement', () => ({ width: Number(width.value), level: Number(level.value) }), s => inRange(s.width, 6, 20) && Number.isInteger(2 * s.width) && integer(s.level, 0, 2), s => { width.value = String(s.width); level.value = String(s.level); }, render);
  });
  labs.register('msbte-injection', ctx => {
    setup(ctx);
    const muL = numberControl(ctx, 'Left chemical potential μL (meV)', '왼쪽 chemical potential μL (meV)', 50, 180, 5, 115), muR = numberControl(ctx, 'Right chemical potential μR (meV)', '오른쪽 chemical potential μR (meV)', 50, 180, 5, 95), temperature = numberControl(ctx, 'Temperature T (K)', '온도 T (K)', 50, 500, 25, 300);
    const count = ctx.control('Subbands in current sum', '전류 합에 포함할 subband 수', { type: 'select', value: '8', options: [2, 4, 8].map(n => ({ value: String(n), en: String(n), ko: String(n) })) });
    const electronCurrent = ctx.metric('Electron-flow current Ie (μA)', 'Electron-flow current Ie (μA)'), current = ctx.metric('Charge current Icharge = −Ie (μA)', 'Charge current Icharge = −Ie (μA)'), cutoff = ctx.metric('Change: 4 → 8 subbands (μA)', 'Subband 4개 → 8개 전류 차이 (μA)');
    function render() {
      const modes = subbands(10, 8, .19, 8), edges = modes.map(m => m.edge), l = Number(muL.value), r = Number(muR.value), t = Number(temperature.value), n = Number(count.value);
      const value = ballisticCurrent(edges.slice(0, n), l, r, t);
      current.textContent = value.toFixed(5); electronCurrent.textContent = (-value).toFixed(5); cutoff.textContent = stableExp(Math.abs(ballisticCurrent(edges, l, r, t) - ballisticCurrent(edges.slice(0, 4), l, r, t)));
      const cumulative = Array.from({ length: 8 }, (_, j) => [j + 1, -ballisticCurrent(edges.slice(0, j + 1), l, r, t)]);
      const series = [];
      modes.slice(0, 2).forEach((m, i) => {
        series.push({ points: sampled(-.8, -.00001, 100, k => fermi(m.edge + A0 / .19 * k * k, r, t)), color: colors[i], dash: true, label: `n=${i + 1}, k<0 ← R` });
        series.push({ points: sampled(.00001, .8, 100, k => fermi(m.edge + A0 / .19 * k * k, l, t)), color: colors[i], label: `n=${i + 1}, k>0 ← L` });
      });
      ctx.view.replaceChildren(ctx.plot({ x: [-.8, .8], y: [0, 1], title: ctx.text('Ballistic occupation: first two subbands', 'Ballistic occupation: 첫 두 subband'), xLabel: 'Longitudinal k (nm⁻¹)', yLabel: 'Occupation f (0 to 1)', series }), ctx.plot({ x: [20, 260], y: [0, 1], title: ctx.text('The reservoirs supply occupations at the same total energy', '총에너지에 따른 두 reservoir의 occupation'), xLabel: 'Total energy E (meV)', yLabel: 'Fermi occupation', series: [{ points: sampled(20, 260, 121, e => fermi(e, l, t)), color: 'blue', label: `L: μ=${l} meV` }, { points: sampled(20, 260, 121, e => fermi(e, r, t)), color: 'gold', label: `R: μ=${r} meV` }] }), ctx.plot({ x: [.5, 8.5], y: [-22, 22], xTicks: [1, 2, 4, 6, 8], yTicks: [-20, -10, 0, 10, 20], title: ctx.text('Cumulative current from the retained subbands', '계산에 포함한 subband 수에 따른 누적 전류'), xLabel: ctx.text('Number of retained subbands', '전류 합에 포함한 subband 수'), yLabel: 'Electron-flow current Ie (μA)',
        bands: [{ x0: .5, x1: n + .5, color: 'blue' }],
        series: [{ points: cumulative, color: 'muted', dash: true, label: ctx.text('Cumulative current, up to 8', '누적 전류: 최대 8개') }, { points: cumulative.slice(0, n), color: 'blue', label: ctx.text('Included in the selected sum', '현재 계산에 포함한 전류') }, { points: [[n, -22], [n, 22]], color: 'gold', dash: true, label: ctx.text('Selected cutoff', '선택한 subband 수') }],
        markers: [{ x: n, y: -value, color: 'blue', label: ctx.text('Selected current', '선택한 전류') }] }), paragraph(ctx, 'Uniform hard-wall wire: W=10 nm, H=8 nm, m*=0.19m₀, spin factor 2, one valley, no scattering and unit transmission. The occupation plot shows the first two subbands. The fixed-axis current plot shows cumulative contributions through all eight, with the selected cutoff shaded and marked. The current includes the selected count and integrates energy to infinity analytically. Ie=eΦ is positive when electrons flow toward +x; the charge current is Icharge=−Ie. The value at k=0 is irrelevant to the integral.', 'W=10 nm, H=8 nm, m*=0.19m₀인 균일한 hard-wall wire입니다. Spin factor는 2, valley는 하나이며 scattering이 없고 transmission은 1입니다. Occupation 그래프는 첫 두 subband를 보여 줍니다. 전류 그래프에서는 첫 번째 subband부터 여덟 번째까지 순서대로 더한 값을 비교합니다. 축의 범위는 고정되어 있고, 현재 계산에 포함한 subband 수를 음영과 점선으로 표시합니다. 전류는 선택한 개수의 subband를 합해 계산합니다. 에너지 적분은 무한대까지 해석적으로 계산합니다. 전자가 +x로 흐르면 Ie=eΦ는 양수이고 charge current Icharge=−Ie는 음수입니다. k=0 한 점의 값은 적분 결과에 영향을 주지 않습니다.'));
      ctx.status.textContent = ctx.text(`μL=${l}, μR=${r} meV; T=${t} K. ${n} subbands included. Equalize μL and μR to check zero current.`, `μL=${l}, μR=${r} meV, T=${t} K입니다. Subband ${n}개를 포함했습니다. μL과 μR을 같게 맞춰 전류가 0이 되는지 확인해 보세요.`);
    }
    ctx.onChange(render);
    modelLink(ctx, 'msbte-injection', () => ({ muL: Number(muL.value), muR: Number(muR.value), temperature: Number(temperature.value), count: Number(count.value) }), s => [s.muL, s.muR].every(v => inRange(v, 50, 180) && v % 5 === 0) && inRange(s.temperature, 50, 500) && s.temperature % 25 === 0 && [2, 4, 8].includes(s.count), s => { muL.value = String(s.muL); muR.value = String(s.muR); temperature.value = String(s.temperature); count.value = String(s.count); }, render);
  });
  labs.register('msbte-scattering', ctx => {
    setup(ctx);
    const energy = numberControl(ctx, 'Total energy E (meV)', '총에너지 E (meV)', 60, 140, 5, 130), beta = numberControl(ctx, 'Intraband β (nm⁻¹)', 'Intraband β (nm⁻¹)', 0, .04, .001, .005), gamma = numberControl(ctx, 'Inter-subband γ (nm⁻¹)', 'Inter-subband γ (nm⁻¹)', 0, .04, .001, .003);
    const fL = numberControl(ctx, 'Incoming occupation fL(E)', '들어오는 occupation fL(E)', 0, 1, .05, .9), fR = numberControl(ctx, 'Incoming occupation fR(E)', '들어오는 occupation fR(E)', 0, 1, .05, .2);
    const filter = ctx.control('Mode-selective left injector (nonthermal)', '왼쪽에서 mode별로 다르게 주입 (nonthermal)', { type: 'checkbox', value: false });
    const fL2 = numberControl(ctx, 'Mode-2 incoming fL(E)', 'Mode 2에 주입할 fL(E)', 0, 1, .05, .3);
    const cells = ctx.control('Spatial cells N', '공간 cell 수 N', { type: 'select', value: '24', options: [12, 24, 36].map(n => ({ value: String(n), en: String(n), ko: String(n) })) });
    const flux = ctx.metric('Spectral face flux q', 'Spectral face flux q'), residual = ctx.metric('Max face-flux variation', 'Face flux의 최대 차이'), open = ctx.metric('Open subbands / E−εn (meV)', '진행 가능한 subband / E−εn (meV)');
    const gridError = ctx.metric('Transmission grid error (%)', 'Transmission grid 오차 (%)');
    function render() {
      const e = Number(energy.value), bands = subbands(10, 8, .19, 2), active = bands.filter(m => e > m.edge), n = active.length, b = Number(beta.value), g = Number(gamma.value), l = Number(fL.value), r = Number(fR.value), count = Number(cells.value), left = Array.from({ length: n }, (_, i) => filter.checked && i === 1 ? Number(fL2.value) : l), result = solveShell({ modes: n, length: 100, beta: b, gamma: g, fL: left, fR: r, cells: count });
      fL2.disabled = !filter.checked || n < 2; gamma.disabled = n < 2; filter.disabled = n < 2;
      flux.textContent = Number(result.flux[0].toFixed(6)).toFixed(6); residual.textContent = stableExp(Math.max(...result.flux) - Math.min(...result.flux)); open.textContent = bands.map((m, i) => (i + 1) + ': ' + (e > m.edge ? (e - m.edge).toFixed(1) : '—')).join('\n');
      const symmetricInflow = n === 1 || left.every(value => Math.abs(value - left[0]) < 1e-12);
      const benchmark = symmetricInflow ? symmetricTransmission(n, 100, b, g, count) : null;
      gridError.textContent = benchmark ? benchmark.relativeErrorPercent.toFixed(2) : '—';
      const series = [];
      active.forEach((_, i) => {
        series.push({ points: [[0, left[i]], ...result.x.map((x, j) => [x, result.occupation[j][2 * i]])], color: colors[i], label: `n=${i + 1}, f⁺` });
        series.push({ points: [...result.x.map((x, j) => [x, result.occupation[j][2 * i + 1]]), [100, r]], color: colors[i], dash: true, label: `n=${i + 1}, f⁻` });
      });
      const branchMarkers = active.flatMap((band, i) => { const root = Math.sqrt((e - band.edge) * .19 / A0); return [-root, root].map(k => ({ x: k, y: e, color: colors[i], label: 'n=' + (i + 1) + ', k=' + k.toFixed(3) + ' nm⁻¹' })); });
      const fluxSeries = active.map((_, i) => ({ points: result.faceX.map((x, j) => [x, result.modeFlux[j][i]]), color: colors[i], dash: true, label: `n=${i + 1}` }));
      fluxSeries.push({ points: result.faceX.map((x, i) => [x, result.flux[i]]), color: 'red', label: ctx.text('Total', '전체') });
      ctx.view.replaceChildren(
        ctx.plot({ x: [0, 100], y: [0, 1], title: ctx.text(`At E=${e} meV: ${n} open ${n === 1 ? 'subband' : 'subbands'}`, `E=${e} meV: 전자가 진행할 수 있는 subband ${n}개`), xLabel: 'Position x (nm)', yLabel: 'Occupation f', series, markers: result.x.map((x, j) => ({ x, y: result.occupation[j][0], color: colors[0], label: 'Cell ' + (j + 1) + '/' + count })) }),
        ctx.plot({ x: [0, 100], y: [-2.1, 2.1], yTicks: [-2, -1, 0, 1, 2], title: ctx.text('Face flux: redistribution need not change the total', 'Face flux: subband별 값이 바뀌어도 전체는 보존됩니다'), xLabel: 'Face position x (nm)', yLabel: 'q(E,x) (dimensionless)', series: fluxSeries }),
        ctx.plot({ x: [-.8, .8], y: [0, 280], xTicks: [-.8, -.4, 0, .4, .8], yTicks: [0, 70, 140, 210, 280], title: ctx.text('A selected energy fixes the available wavevectors', '총에너지를 바꾸면 가능한 wavevector가 달라집니다'), xLabel: 'Longitudinal k (nm⁻¹)', yLabel: 'Total energy E (meV)',
          series: [...bands.map((band, i) => ({ points: sampled(-.8, .8, 161, k => band.edge + A0 / .19 * k * k), color: colors[i], label: 'Subband ' + (i + 1) })), { points: [[-.8, e], [.8, e]], color: 'red', dash: true, label: 'Selected E' }], markers: branchMarkers }),
        paragraph(ctx, 'Length is 100 nm. β connects opposite directions within a subband; γ/2 connects every directional pair between subbands. These are chosen graph coefficients, not fitted material rates. With identical contact occupations, the mode curves coincide in this symmetric model. Enable the prescribed nonthermal mode-selective injector to set mode 2\'s incoming occupation independently and see inter-subband redistribution. Its default 0.3 is below mode 1\'s 0.9, but it can also be raised above mode 1. This is not an equilibrium reservoir. At one fixed energy, these are occupations, not energy-integrated electron counts. In this particular inverse-length model, speed cancels from the shell transport equation. Changing E while the same subbands remain available therefore leaves the occupation profiles unchanged; the dispersion plot explicitly shows the changed physical k roots instead. γ and the mode-selective option are disabled when only one subband has propagating states.', '길이는 100 nm입니다. 같은 subband에서 방향이 바뀌는 scattering에는 β를, 다른 subband로 옮겨 가는 scattering에는 각 방향 쌍마다 γ/2를 씁니다. 실제 재료에 맞춘 scattering rate가 아니라 이 예제를 위해 정한 계수입니다. 왼쪽에서 두 mode에 같은 occupation을 주입하고 오른쪽에서도 그렇게 지정하면, 같은 진행 방향끼리는 두 mode의 곡선이 겹칩니다. Mode별 주입을 켜면 mode 2의 incoming occupation을 따로 정할 수 있습니다. 전자들이 subband 사이를 오가면서 mode별 occupation이 어떻게 달라지는지 비교해 보세요. 기본값 0.3은 mode 1의 0.9보다 작지만, 첫 번째 mode의 occupation보다 크게 지정해도 됩니다. 이 옵션에서는 nonthermal mode-selective injector의 주입 조건을 직접 지정합니다. Equilibrium reservoir에서 주입하는 경우와는 다릅니다. Occupation 그래프는 한 에너지에서 각 state가 얼마나 채워져 있는지를 보여 주며, 에너지 전체를 합한 전자 수가 아닙니다. 여기서 정한 inverse-length 계수를 고정하면 shell 방정식의 양변에서 speed가 약분됩니다. 따라서 진행 가능한 subband가 그대로이면 E를 바꿔도 occupation profile은 같습니다. 반면 전자의 kinetic energy와 speed는 달라지므로, dispersion 그래프에서 선택한 에너지의 수평선과 가능한 k 위치는 바뀝니다. 진행 가능한 subband가 하나뿐이면 γ와 mode별 주입 옵션은 비활성화합니다.'),
        paragraph(ctx, `The first-order upwind grid conserves FACE flux exactly to solver precision but has finite-grid error. For one subband, its transmission is (1+βΔx)/(1+βL+βΔx), not exactly the continuum 1/(1+βL). Dots on the first positive-direction curve mark cell centres, so changing N also changes the visible sampling. Increase N to see convergence. Subband 3 remains closed because E≤140 meV < ε₃≈${subbands(10, 8, .19, 3)[2].edge.toFixed(1)} meV.`, `First-order upwind 방식으로 계산한 전체 flux는 각 cell 경계에서 수치오차 범위 안에서 같은 값이 나옵니다. 그래도 계산 결과 자체는 grid 간격에 따라 달라질 수 있습니다. Subband가 하나일 때 grid로 계산한 transmission은 (1+βΔx)/(1+βL+βΔx)입니다. 연속 방정식의 해인 transmission 1/(1+βL)과는 차이가 있습니다. Subband 1의 f⁺ 곡선에 찍은 점은 각 cell의 중심에서 계산한 값입니다. N을 바꾸면 점의 간격도 달라집니다. N을 늘려 수렴을 확인하세요. E≤140 meV이고 ε₃≈${subbands(10, 8, .19, 3)[2].edge.toFixed(1)} meV이므로, 세 번째 subband에는 이 에너지로 진행할 수 있는 state가 없습니다.`)
      );
      ctx.status.textContent = ctx.text(`Only incoming f⁺(0)=[${left.join(', ')}], f⁻(L)=${r} are fixed. Outgoing occupations are solved. γ has no effect when only one subband is open.`, `들어오는 f⁺(0)=[${left.join(', ')}], f⁻(L)=${r}만 지정했습니다. 나가는 occupation은 계산으로 구합니다. 이 에너지에서 진행할 수 있는 subband가 하나뿐이면 γ는 결과에 영향을 주지 않습니다.`);
      ctx.status.textContent += ' ' + (benchmark
        ? ctx.text(`Equal-mode inflow benchmark: T_FV=${benchmark.grid.toFixed(6)}, T_cont=${benchmark.continuum.toFixed(6)}. The percentage compares transmission, not the error of a possibly zero equilibrium current.`, `각 contact에서 mode별 inflow가 같을 때의 기준값: T_FV=${benchmark.grid.toFixed(6)}, T_cont=${benchmark.continuum.toFixed(6)}입니다. 이 백분율은 transmission의 오차이며, equilibrium에서 0이 될 수 있는 전류의 상대오차가 아닙니다.`)
        : ctx.text('Mode-dependent inflow: the symmetric transmission benchmark does not apply. Increase N and compare the profiles and flux.', 'Mode별 inflow가 서로 다르므로 이 transmission 기준값을 사용할 수 없습니다. N을 늘려 profile과 flux가 얼마나 달라지는지 비교하세요.'));
    }
    ctx.onChange(render);
    modelLink(ctx, 'msbte-scattering', () => ({ energy: Number(energy.value), beta: Number(beta.value), gamma: Number(gamma.value), fL: Number(fL.value), fR: Number(fR.value), cells: Number(cells.value), filter: filter.checked, fL2: Number(fL2.value) }), s => inRange(s.energy, 60, 140) && s.energy % 5 === 0 && [s.beta, s.gamma].every(v => inRange(v, 0, .04) && Math.abs(v * 1000 - Math.round(v * 1000)) < 1e-8) && [s.fL, s.fR, s.fL2].every(v => inRange(v, 0, 1) && Math.abs(v * 20 - Math.round(v * 20)) < 1e-8) && [12, 24, 36].includes(s.cells) && typeof s.filter === 'boolean', s => { energy.value = String(s.energy); beta.value = String(s.beta); gamma.value = String(s.gamma); fL.value = String(s.fL); fR.value = String(s.fR); cells.value = String(s.cells); filter.checked = s.filter; fL2.value = String(s.fL2); }, render);
  });
  labs.register('msbte-quiz', ctx => {
    setup(ctx);
    const index = ctx.control('Question', '문제', { type: 'select', value: '0', options: questions.map((_, i) => ({ value: String(i), en: `${i + 1} / ${questions.length}`, ko: `${i + 1} / ${questions.length}` })) });
    let selected = null, checked = false, lastIndex = 0;
    ctx.button('Check answer', '답 확인', () => { if (selected !== null) { checked = true; render(); } else ctx.status.textContent = ctx.text('Choose an answer first.', '답을 먼저 골라 주세요.'); });
    function render() {
      const id = Number(index.value), q = quizProblem(id);
      if (id !== lastIndex) { selected = null; checked = false; lastIndex = id; }
      const choices = ctx.el('div', { class: 'msbte-choices', role: 'group', 'aria-label': ctx.text('Answer choices', '답 선택') });
      q.choices.forEach((choice, j) => { const button = ctx.el('button', { type: 'button', class: 'msbte-choice' + (selected === j ? ' msbte-chosen' : ''), 'aria-pressed': String(selected === j) }, ctx.text(...choice)); button.addEventListener('click', () => { selected = j; checked = false; render(); const replacement = ctx.root.querySelectorAll('.msbte-choice')[j]; if (replacement) replacement.focus({ preventScroll: true }); }); choices.append(button); });
      const nodes = [paragraph(ctx, q.en, q.ko), quizDiagram(ctx, id, q), choices];
      if (checked) nodes.push(ctx.el('p', { class: 'msbte-answer' + (selected === q.answer ? ' msbte-correct' : '') }, [ctx.el('strong', {}, ctx.text(selected === q.answer ? 'Correct. ' : 'Try again. ', selected === q.answer ? '맞았습니다. ' : '다시 확인해 보세요. ')), ctx.text(...q.explanation)]));
      ctx.view.replaceChildren(...nodes);
      ctx.status.textContent = ctx.text('Each diagram shows the energy, occupation or contact condition used in that question. No user-supplied code is executed.', '각 그림은 해당 문제의 에너지, occupation, contact 조건을 보여 줍니다. 사용자가 입력한 코드를 실행하지 않습니다.');
    }
    ctx.onChange(render); ctx.onReset(() => { selected = null; checked = false; lastIndex = 0; });
    modelLink(ctx, 'msbte-quiz', () => ({ index: Number(index.value), selected, checked }), s => integer(s.index, 0, questions.length - 1) && (s.selected === null || integer(s.selected, 0, 2)) && typeof s.checked === 'boolean' && (!s.checked || s.selected !== null), s => { index.value = String(s.index); lastIndex = s.index; selected = s.selected; checked = s.checked; }, render);
  });
})();
