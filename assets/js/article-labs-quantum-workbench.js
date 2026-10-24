/* Small exact complex-state circuit puzzles. All qubits are simulated, without measurement. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else api.install(root.ArticleLabs);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const LIMIT = 12, EPS = 1e-9;
  const gate = (op, target, controls = []) => ({ op, target, controls });
  const PUZZLES = [
    { id: 'shorten', n: 2, data: 2, budget: 1, en: 'Shorten H → Z → H', ko: 'H → Z → H 줄이기',
      instruction: ['Replace three gates by one, preserving the action on both qubits.', '두 qubit에 대한 동작을 보존하면서 세 gate를 하나로 줄이세요.'],
      target: [gate('X', 0)], initial: [gate('H', 0), gate('Z', 0), gate('H', 0)] },
    { id: 'phase', n: 2, data: 2, budget: 1, en: 'Recover the missing phase', ko: '빠진 phase 복원하기',
      instruction: ['The current identity circuit has the right basis probabilities. Restore the relative sign of the target Z.', '현재 identity 회로는 basis 측정 확률이 맞습니다. 목표 Z의 상대 부호를 복원하세요.'],
      target: [gate('Z', 0)], initial: [] },
    { id: 'scratch', n: 3, data: 2, budget: 3, en: 'Erase the AND scratch', ko: 'AND scratch 비우기',
      instruction: ['The oracle changes the sign of |11⟩. Return q2 to |0⟩ for every data input; add the missing uncomputation or find a shorter circuit.', 'oracle은 |11⟩의 부호를 바꿉니다. 모든 data 입력에서 q2를 |0⟩로 돌려놓도록 uncomputation을 추가하거나 더 짧은 회로를 찾으세요.'],
      target: [gate('CCX', 2, [0, 1]), gate('Z', 2), gate('CCX', 2, [0, 1])],
      initial: [gate('CCX', 2, [0, 1]), gate('Z', 2)] }
  ];
  const copy = value => JSON.parse(JSON.stringify(value));
  const complex = (r = 0, i = 0) => [r, i];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const scale = (a, b) => [a[0] * b, a[1] * b];
  const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  const abs2 = a => a[0] ** 2 + a[1] ** 2;
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  function validGate(g, n) {
    if (!g || typeof g !== 'object' || !['X', 'H', 'Z', 'S', 'T', 'CX', 'CZ', 'CCX'].includes(g.op)) return false;
    if (!Number.isInteger(g.target) || g.target < 0 || g.target >= n || !Array.isArray(g.controls)) return false;
    const count = g.op === 'CCX' ? 2 : ['CX', 'CZ'].includes(g.op) ? 1 : 0;
    return g.controls.length === count && new Set([g.target, ...g.controls]).size === count + 1 &&
      g.controls.every(q => Number.isInteger(q) && q >= 0 && q < n);
  }
  function validateCircuit(circuit, n) {
    if (![2, 3].includes(n) || !Array.isArray(circuit) || circuit.length > LIMIT || !circuit.every(g => validGate(g, n))) throw new RangeError('Invalid bounded circuit');
    return copy(circuit);
  }
  function basis(n, index) {
    if (!Number.isInteger(index) || index < 0 || index >= 2 ** n) throw new RangeError('Invalid basis input');
    return Array.from({ length: 2 ** n }, (_, i) => complex(i === index ? 1 : 0));
  }
  function applyGate(state, g, n) {
    if (!validGate(g, n) || state.length !== 2 ** n) throw new RangeError('Invalid gate or state');
    const out = state.map(a => a.slice()), mask = 1 << (n - 1 - g.target);
    const enabled = i => g.controls.every(q => (i & (1 << (n - 1 - q))) !== 0);
    for (let i = 0; i < state.length; i += 1) {
      if ((i & mask) || !enabled(i)) continue;
      const j = i | mask;
      if (['X', 'CX', 'CCX'].includes(g.op)) { out[i] = state[j].slice(); out[j] = state[i].slice(); }
      else if (g.op === 'H') { out[i] = scale(add(state[i], state[j]), Math.SQRT1_2); out[j] = scale(add(state[i], scale(state[j], -1)), Math.SQRT1_2); }
      else {
        const angle = ['Z', 'CZ'].includes(g.op) ? Math.PI : g.op === 'S' ? Math.PI / 2 : Math.PI / 4;
        out[j] = mul(state[j], [Math.cos(angle), Math.sin(angle)]);
      }
    }
    return out;
  }
  function simulate(circuit, n, input, steps = circuit.length) {
    validateCircuit(circuit, n);
    let state = Array.isArray(input) ? input.map(a => a.slice()) : basis(n, input);
    if (state.length !== 2 ** n || state.some(a => !Array.isArray(a) || a.length !== 2 || a.some(v => !Number.isFinite(v)))) throw new RangeError('Invalid state');
    const count = Math.max(0, Math.min(circuit.length, Math.trunc(steps)));
    for (let i = 0; i < count; i += 1) state = applyGate(state, circuit[i], n);
    return state;
  }
  function compare(candidate, reference, n, data = n) {
    validateCircuit(candidate, n); validateCircuit(reference, n);
    if (!Number.isInteger(data) || data < 1 || data > n) throw new RangeError('Invalid data register');
    const scratch = n - data, mask = (1 << scratch) - 1;
    const outputs = [], targets = [];
    for (let i = 0; i < 2 ** data; i += 1) { outputs.push(simulate(candidate, n, i << scratch)); targets.push(simulate(reference, n, i << scratch)); }
    let sharedPhase = [1, 0];
    outer: for (let i = 0; i < targets.length; i += 1) for (let j = 0; j < targets[i].length; j += 1) {
      if (abs2(targets[i][j]) > EPS && abs2(outputs[i][j]) > EPS) {
        const ratio = mul(outputs[i][j], [targets[i][j][0], -targets[i][j][1]]), magnitude = Math.sqrt(abs2(ratio));
        sharedPhase = scale(ratio, 1 / magnitude); break outer;
      }
    }
    let error = 0, probabilityError = 0, leakage = 0, witness = null;
    for (let i = 0; i < outputs.length; i += 1) {
      let columnError = 0, leak = 0;
      for (let j = 0; j < outputs[i].length; j += 1) {
        columnError = Math.max(columnError, distance(outputs[i][j], mul(sharedPhase, targets[i][j])));
        probabilityError = Math.max(probabilityError, Math.abs(abs2(outputs[i][j]) - abs2(targets[i][j])));
        if (j & mask) leak += abs2(outputs[i][j]);
      }
      if (columnError > EPS && witness === null) witness = i;
      error = Math.max(error, columnError); leakage = Math.max(leakage, leak);
    }
    // A relative phase between columns needs a coherent input to make its effect observable.
    let coherent = null;
    if (error > EPS && probabilityError <= EPS && witness !== null && witness !== 0) {
      const input = basis(n, 0).map((a, i) => complex(i === 0 || i === (witness << scratch) ? Math.SQRT1_2 : 0));
      coherent = { first: 0, second: witness, input, actual: simulate(candidate, n, input), expected: simulate(reference, n, input) };
    }
    return { equivalent: error < EPS && leakage < EPS, error, probabilityError, leakage, sharedPhase, witness, coherent, outputs, targets };
  }
  function edit(circuit, operation, index, selectedGate, n) {
    const out = validateCircuit(circuit, n);
    if (!Number.isInteger(index) || index < 0 || index > out.length) throw new RangeError('Invalid slot');
    if (operation === 'insert') { if (out.length >= LIMIT || !validGate(selectedGate, n)) throw new RangeError('Gate limit or invalid gate'); out.splice(index, 0, copy(selectedGate)); }
    else if (operation === 'replace') { if (index >= out.length || !validGate(selectedGate, n)) throw new RangeError('Select an existing gate'); out[index] = copy(selectedGate); }
    else if (operation === 'remove') { if (index >= out.length) throw new RangeError('Select an existing gate'); out.splice(index, 1); }
    else if (operation === 'left') { if (index <= 0 || index >= out.length) throw new RangeError('Cannot move left'); [out[index - 1], out[index]] = [out[index], out[index - 1]]; }
    else if (operation === 'right') { if (index >= out.length - 1) throw new RangeError('Cannot move right'); [out[index], out[index + 1]] = [out[index + 1], out[index]]; }
    else throw new RangeError('Unknown edit');
    return out;
  }
  function palette(n) {
    const gates = [];
    for (const op of ['X', 'H', 'Z', 'S', 'T']) for (let target = 0; target < n; target += 1) gates.push(gate(op, target));
    for (const op of ['CX', 'CZ']) for (let control = 0; control < n; control += 1) for (let target = 0; target < n; target += 1) if (target !== control) gates.push(gate(op, target, [control]));
    if (n === 3) for (let target = 0; target < n; target += 1) gates.push(gate('CCX', target, [0, 1, 2].filter(q => q !== target)));
    return gates;
  }
  function gateLabel(g) { return g.controls.length ? `${g.op} ${g.controls.map(q => 'q' + q).join(',')} → q${g.target}` : `${g.op} q${g.target}`; }
  function restore(value) {
    if (!value || typeof value !== 'object') throw new RangeError('Invalid saved circuit');
    const p = PUZZLES.find(puzzle => puzzle.id === value.puzzle);
    if (!p) throw new RangeError('Unknown puzzle');
    const circuit = validateCircuit(value.circuit, p.n);
    const selected = Number.isInteger(value.selected) ? Math.max(0, Math.min(circuit.length, value.selected)) : 0;
    const step = Number.isInteger(value.step) ? Math.max(0, Math.min(circuit.length, value.step)) : circuit.length;
    const input = value.input === 'plus' ? 'plus' : String(Math.max(0, Math.min(2 ** p.data - 1, Number.isInteger(Number(value.input)) ? Number(value.input) : 0)));
    return { puzzle: p.id, circuit, selected, step, input };
  }
  function install(labs) {
    if (!labs) return;
    labs.register('quantum-circuit-puzzle', ctx => {
      ctx.root.classList.add('al-quantum-workbench');
      const challenge = ctx.control('Challenge', '퍼즐', { type: 'select', value: 'shorten', options: PUZZLES.map(p => ({ value: p.id, en: p.en, ko: p.ko })) });
      const choice = ctx.control('Gate palette', '추가할 gate', { type: 'select', value: '0', options: [] });
      const selected = ctx.control('Selected slot (0 = first)', '선택 위치 (0 = 처음)', { type: 'number', min: 0, max: 3, step: 1, value: 0 });
      const input = ctx.control('Probe input (data)', '확인할 입력 (data)', { type: 'select', value: 'plus', options: [] });
      const step = ctx.control('State after gate', '상태를 볼 gate 단계', { type: 'range', min: 0, max: 3, step: 1, value: 3 });
      const reveal = ctx.control('Show a solution', '풀이 보기', { type: 'checkbox', value: false });
      const count = ctx.metric('Gates / goal', 'gate 수 / 목표'), verdict = ctx.metric('Full action check', '전체 동작 검사'), scratch = ctx.metric('Worst scratch P(≠0)', 'scratch P(≠0) 최댓값');
      let puzzle = PUZZLES[0], circuit = copy(puzzle.initial), message = '';
      function options() {
        choice.replaceChildren(...palette(puzzle.n).map((g, i) => ctx.el('option', { value: i }, gateLabel(g)))); choice.value = '0';
        input.replaceChildren(ctx.el('option', { value: 'plus' }, '|+⟩⊗' + puzzle.data), ...Array.from({ length: 2 ** puzzle.data }, (_, i) => ctx.el('option', { value: i }, '|' + i.toString(2).padStart(puzzle.data, '0') + '⟩'))); input.value = 'plus';
      }
      function load(id) { puzzle = PUZZLES.find(p => p.id === id) || PUZZLES[0]; circuit = copy(puzzle.initial); selected.value = '0'; step.max = String(circuit.length); step.value = String(circuit.length); message = ''; options(); }
      function change(operation) {
        try {
          const raw = selected.value.trim(), index = Number(raw);
          if (!raw || !Number.isInteger(index) || index < 0 || index > circuit.length) throw new RangeError('Invalid slot');
          circuit = edit(circuit, operation, index, palette(puzzle.n)[Number(choice.value)], puzzle.n); selected.value = String(operation === 'left' ? index - 1 : operation === 'right' ? index + 1 : Math.min(index, circuit.length)); step.max = String(circuit.length); step.value = String(circuit.length); message = '';
        }
        catch (error) { message = ctx.text('Choose a valid gate slot; at most 12 gates are allowed.', 'gate 위치를 확인하세요. gate는 최대 12개까지 둘 수 있습니다.'); }
        render();
      }
      ctx.button('Insert before slot', '선택 위치에 삽입', () => change('insert'));
      ctx.button('Replace selected', '선택 gate 교체', () => change('replace'));
      ctx.button('Delete selected', '선택 gate 삭제', () => change('remove'));
      const left = ctx.button('Move ←', '왼쪽 이동', () => change('left')), right = ctx.button('Move →', '오른쪽 이동', () => change('right'));
      ctx.button('Clear circuit', '회로 비우기', () => { circuit = []; selected.value = '0'; step.value = '0'; message = ''; render(); });
      const actions = ctx.el('div', { class: 'qw-actions', role: 'group', 'aria-label': ctx.text('Edit selected gate', '선택 gate 편집') });
      ctx.ui.querySelectorAll(':scope > .al-button').forEach(button => actions.append(button));
      const probeControls = ctx.el('details', { class: 'qw-probe' }, ctx.el('summary', {}, ctx.text('Options, state & solution', '추가 설정 · 상태 · 풀이')));
      [selected, input, step, reveal].forEach(control => probeControls.append(control.closest('.al-control')));
      ctx.ui.append(actions, probeControls);
      function amplitude(a) { const clean = v => Math.abs(v) < 1e-10 ? 0 : v; return `${ctx.format(clean(a[0]), 3)} ${clean(a[1]) < 0 ? '−' : '+'} ${ctx.format(Math.abs(clean(a[1])), 3)}i`; }
      function render() {
        if (challenge.value !== puzzle.id) load(challenge.value);
        selected.max = String(circuit.length); step.max = String(circuit.length);
        selected.value = String(Math.max(0, Math.min(circuit.length, Number(selected.value) || 0)));
        step.value = String(Math.max(0, Math.min(circuit.length, Number(step.value) || 0)));
        const slot = Number(selected.value); left.disabled = slot <= 0 || slot >= circuit.length; right.disabled = slot >= circuit.length - 1;
        const r = compare(circuit, puzzle.target, puzzle.n, puzzle.data), solved = r.equivalent && circuit.length <= puzzle.budget;
        count.textContent = `${circuit.length} / ≤${puzzle.budget}`;
        verdict.textContent = ctx.text(r.equivalent ? (solved ? 'Solved' : 'Equivalent; shorten it') : 'Counterexample found', r.equivalent ? (solved ? '성공' : '동일한 동작 · 더 줄이기') : '반례 있음'); scratch.textContent = ctx.format(r.leakage, 4);
        ctx.root.dataset.puzzle = puzzle.id; ctx.root.dataset.equivalent = String(r.equivalent); ctx.root.dataset.solved = String(solved); ctx.root.dataset.gateCount = String(circuit.length);
        const heading = ctx.el('p', { class: 'qw-goal' }, ctx.text(...puzzle.instruction));
        const slots = ctx.el('div', { class: 'qw-circuit', role: 'group', 'aria-label': ctx.text('Editable circuit, left to right', '왼쪽에서 오른쪽으로 적용하는 편집 회로') });
        circuit.forEach((g, i) => {
          const button = ctx.el('button', { type: 'button', class: 'qw-gate', 'aria-pressed': String(i === slot), 'aria-label': `${i}: ${gateLabel(g)}` }, [ctx.el('span', { class: 'qw-step' }, String(i)), ...Array.from({ length: puzzle.n }, (_, q) => ctx.el('span', { class: 'qw-wire' }, `q${q} ${q === g.target ? (g.op === 'CX' || g.op === 'CCX' ? '⊕' : g.op === 'CZ' ? 'Z' : g.op) : g.controls.includes(q) ? '●' : '─'}`))]);
          button.addEventListener('click', () => { selected.value = String(i); selected.dispatchEvent(new Event('input', { bubbles: true })); const replacement = ctx.view.querySelector(`.qw-gate[data-slot="${i}"]`); if (replacement) replacement.focus({ preventScroll: true }); }); button.dataset.slot = String(i); slots.append(button);
        });
        const append = ctx.el('button', { type: 'button', class: 'qw-gate qw-end', 'aria-pressed': String(slot === circuit.length) }, ctx.text(`${circuit.length}: + end`, `${circuit.length}: + 끝`)); append.addEventListener('click', () => { selected.value = String(circuit.length); selected.dispatchEvent(new Event('input', { bubbles: true })); const replacement = ctx.view.querySelector('.qw-end'); if (replacement) replacement.focus({ preventScroll: true }); }); slots.append(append);
        let probe;
        if (input.value === 'plus') probe = basis(puzzle.n, 0).map((a, i) => complex((i & ((1 << (puzzle.n - puzzle.data)) - 1)) === 0 ? 1 / Math.sqrt(2 ** puzzle.data) : 0));
        else probe = basis(puzzle.n, Number(input.value) << (puzzle.n - puzzle.data));
        const actual = simulate(circuit, puzzle.n, probe, Number(step.value)), expected = simulate(puzzle.target, puzzle.n, probe);
        const inspect = ctx.el('details', { class: 'qw-state' }, [ctx.el('summary', {}, ctx.text(`Complex amplitudes after gate ${step.value}`, `${step.value}단계 후 복소 amplitude`)),
          ctx.table([ctx.text('Basis q0…', 'basis q0…'), ctx.text('Your circuit (step)', '현재 회로 (단계)'), ctx.text('Target (final)', '목표 (최종)'), 'P (step)'], actual.map((a, i) => ['|' + i.toString(2).padStart(puzzle.n, '0') + '⟩', amplitude(a), amplitude(expected[i]), ctx.format(abs2(a), 3)]))]);
        const summary = ctx.el('div', { class: 'qw-verdict', 'data-qw-check': '' }, r.equivalent ? ctx.text(`All ${2 ** puzzle.data} basis inputs agree under one shared global phase.`, `모든 ${2 ** puzzle.data}개 basis 입력이 하나의 공통 global phase 아래 일치합니다.`) : r.coherent ? ctx.text(`Basis probabilities match, but (|${r.coherent.first.toString(2).padStart(puzzle.data, '0')}⟩ + |${r.coherent.second.toString(2).padStart(puzzle.data, '0')}⟩)/√2 has the wrong relative phase.`, `basis 측정 확률은 같지만 (|${r.coherent.first.toString(2).padStart(puzzle.data, '0')}⟩ + |${r.coherent.second.toString(2).padStart(puzzle.data, '0')}⟩)/√2의 상대 phase가 다릅니다.`) : ctx.text(`Counterexample data input: |${r.witness.toString(2).padStart(puzzle.data, '0')}⟩; amplitude error ${ctx.format(r.error, 3)}.`, `반례 data 입력: |${r.witness.toString(2).padStart(puzzle.data, '0')}⟩; amplitude 오차 ${ctx.format(r.error, 3)}.`));
        if (!r.equivalent) {
          const actualWitness = r.coherent ? r.coherent.actual : r.outputs[r.witness], targetWitness = r.coherent ? r.coherent.expected : r.targets[r.witness];
          const evidence = ctx.el('details', { class: 'qw-state' }, [ctx.el('summary', {}, ctx.text('Inspect the counterexample', '반례의 상태 비교')), ctx.table(['Basis', ctx.text('Your output', '현재 출력'), ctx.text('Target output', '목표 출력')], actualWitness.map((a, i) => ['|' + i.toString(2).padStart(puzzle.n, '0') + '⟩', amplitude(a), amplitude(targetWitness[i])]))]);
          summary.append(evidence);
        }
        const nodes = [heading, ctx.el('p', { class: 'qw-target' }, ctx.text('Target: ', '목표: ') + (puzzle.id === 'shorten' ? 'X q0' : puzzle.id === 'phase' ? 'Z q0' : 'CZ q0 → q1; q2 = |0⟩')), slots, summary, inspect];
        if (reveal.checked) nodes.push(ctx.el('p', { class: 'qw-solution' }, ctx.text('One solution: ', '풀이 예: ') + puzzle.target.map(gateLabel).join(' → ')));
        ctx.view.replaceChildren(ctx.metrics, ...nodes);
        ctx.status.textContent = message || ctx.text(`Exact ${puzzle.n}-qubit state simulation; q0 is the leftmost bit. All ${2 ** puzzle.data} data-basis inputs share one comparison phase; ${puzzle.n > puzzle.data ? 'q2 starts at 0 and must finish at 0.' : 'both qubits are data.'}`, `${puzzle.n} qubit의 전체 복소 상태를 계산하며 q0가 가장 왼쪽 비트입니다. ${2 ** puzzle.data}개 data-basis 입력에 공통 phase 하나를 사용합니다. ${puzzle.n > puzzle.data ? 'q2는 0으로 시작하고 0으로 끝나야 합니다.' : '두 qubit 모두 data입니다.'}`);
      }
      options();
      ctx.onReset(() => { challenge.value = PUZZLES[0].id; load(PUZZLES[0].id); reveal.checked = false; });
      ctx.onChange(render);
      if (ctx.stateLink) ctx.stateLink('quantum-circuit-puzzle', {
        read: () => ({ puzzle: puzzle.id, circuit: copy(circuit), selected: Number(selected.value), step: Number(step.value), input: input.value }),
        write: value => { const state = restore(value); challenge.value = state.puzzle; load(state.puzzle); circuit = state.circuit; selected.value = String(state.selected); step.max = String(circuit.length); step.value = String(state.step); input.value = state.input; render(); }
      });
    });
  }
  return { LIMIT, PUZZLES, gate, validGate, validateCircuit, basis, applyGate, simulate, compare, edit, palette, restore, install };
});
