/* Finite workshop models. Saved pytest records are read, never executed here. */
(function () {
  'use strict';
  const ids = ['A', 'B', 'C', 'D', 'E', 'F'];
  const clone = state => ({ capacity: state.capacity, confirmed: state.confirmed.slice(), waiting: state.waiting.slice() });
  const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  function validateId(id) { if (typeof id !== 'string' || id.length === 0) throw new Error('Participant ID must be a nonempty string'); }
  function validateState(state) {
    if (!state || typeof state !== 'object' || Array.isArray(state) || Object.keys(state).sort().join(',') !== 'capacity,confirmed,waiting' || !Number.isSafeInteger(state.capacity) || state.capacity < 1 || !Array.isArray(state.confirmed) || !Array.isArray(state.waiting)) throw new Error('Invalid workshop state');
    const participants = state.confirmed.concat(state.waiting);
    for (const id of participants) validateId(id);
    if (new Set(participants).size !== participants.length || state.confirmed.length > state.capacity || state.waiting.length && state.confirmed.length !== state.capacity) throw new Error('State violates the workshop rules');
    return true;
  }
  function createState(capacity) { const state = { capacity, confirmed: [], waiting: [] }; validateState(state); return state; }
  function register(state, id) {
    validateState(state); validateId(id);
    const next = clone(state);
    if (next.confirmed.includes(id) || next.waiting.includes(id)) return next;
    (next.confirmed.length >= next.capacity ? next.waiting : next.confirmed).push(id);
    return next;
  }
  function cancel(state, id) {
    validateState(state); validateId(id);
    const next = clone(state), place = next.confirmed.indexOf(id);
    if (place !== -1) {
      next.confirmed.splice(place, 1);
      if (next.waiting.length) next.confirmed.push(next.waiting.shift());
    } else next.waiting = next.waiting.filter(value => value !== id);
    return next;
  }
  function operation(state, action) {
    if (!action || !['register', 'cancel'].includes(action.operation)) throw new Error('Unknown operation');
    return (action.operation === 'register' ? register : cancel)(state, action.participant_id);
  }
  function replay(capacity, history) {
    if (!Number.isInteger(capacity) || capacity < 1 || capacity > 4 || !Array.isArray(history) || history.length > 30) throw new Error('Invalid bounded history');
    let current = createState(capacity), previous = clone(current);
    for (const action of history) {
      if (!action || Object.keys(action).sort().join(',') !== 'operation,participant_id' || !ids.includes(action.participant_id)) throw new Error('Invalid participant or action');
      previous = current; current = operation(current, action);
    }
    return { previous, current };
  }
  function rule(state, action) {
    const id = action.participant_id;
    if (action.operation === 'register') return state.confirmed.includes(id) || state.waiting.includes(id) ? 'duplicate' : state.confirmed.length < state.capacity ? 'confirm' : 'wait';
    return state.confirmed.includes(id) ? state.waiting.length ? 'promote' : 'cancel-confirmed' : state.waiting.includes(id) ? 'cancel-waiting' : 'missing';
  }
  const ruleText = {
    start: ['No action yet. Registration fills confirmed seats before the FIFO waiting list.', '아직 신청이나 취소를 하지 않았습니다. 빈자리를 먼저 채우고, 정원이 차면 신청 순서대로 대기합니다.'],
    confirm: ['A seat is free: append the participant to confirmed.', '빈자리가 있으므로 참가자를 확정 명단 끝에 추가합니다.'],
    wait: ['All seats are taken: append the participant to the end of waiting.', '정원이 찼으므로 참가자를 대기 명단 끝에 추가합니다.'],
    duplicate: ['This exact ID is already registered: keep the same state value.', '같은 ID가 이미 신청했으므로 상태 값은 바뀌지 않습니다.'],
    promote: ['Remove the confirmed participant and promote the first waiting participant (FIFO).', '확정 참가자를 지우고 가장 먼저 대기한 참가자를 확정합니다(FIFO).'],
    'cancel-confirmed': ['Remove the confirmed participant. Nobody is waiting.', '확정 참가자를 지웁니다. 대기자는 없습니다.'],
    'cancel-waiting': ['Remove only this waiting ID; preserve the order of everyone else.', '해당 대기자만 지우고 나머지 대기 순서는 유지합니다.'],
    missing: ['This ID is absent: cancellation leaves the state value unchanged.', '명단에 없는 ID이므로 취소해도 상태 값은 바뀌지 않습니다.']
  };
  const bugIds = ['boundary', 'duplicate', 'lifo', 'mutation'];
  const bugLabels = {
    boundary: ['Wrong boundary: >= becomes >', '경계 오류: >=를 >로 변경'],
    duplicate: ['Duplicate registration allowed', '중복 신청 허용'],
    lifo: ['Last waiting participant promoted', '마지막 대기자부터 확정'],
    mutation: ['Input state is mutated', '입력 상태를 직접 변경']
  };
  // Deliberately faulty, finite implementations used by the test-selection lab.
  function implementation(state, action, bug = 'correct') {
    validateState(state); validateId(action.participant_id);
    if (!['correct', ...bugIds].includes(bug) || !['register', 'cancel'].includes(action.operation)) throw new Error('Unknown finite implementation');
    const next = bug === 'mutation' ? state : clone(state), id = action.participant_id;
    if (action.operation === 'register') {
      if (bug !== 'duplicate' && (next.confirmed.includes(id) || next.waiting.includes(id))) return next;
      const full = bug === 'boundary' ? next.confirmed.length > next.capacity : next.confirmed.length >= next.capacity;
      (full ? next.waiting : next.confirmed).push(id);
    } else {
      const index = next.confirmed.indexOf(id);
      if (index !== -1) {
        next.confirmed.splice(index, 1);
        if (next.waiting.length) next.confirmed.push(bug === 'lifo' ? next.waiting.pop() : next.waiting.shift());
      } else next.waiting = next.waiting.filter(value => value !== id);
    }
    return next;
  }
  const testCases = [
    { id: 'free-seat', name: ['A free seat confirms A', '빈자리가 있으면 A 확정'], initial: { capacity: 2, confirmed: [], waiting: [] }, action: { operation: 'register', participant_id: 'A' }, expected: { capacity: 2, confirmed: ['A'], waiting: [] }, assertion: 'result' },
    { id: 'at-capacity', name: ['At capacity, B must wait', '정원이 차면 B는 대기'], initial: { capacity: 1, confirmed: ['A'], waiting: [] }, action: { operation: 'register', participant_id: 'B' }, expected: { capacity: 1, confirmed: ['A'], waiting: ['B'] }, assertion: 'result' },
    { id: 'duplicate-waiting', name: ['Registering waiting B is a no-op', '대기 중인 B가 다시 신청해도 동일'], initial: { capacity: 1, confirmed: ['A'], waiting: ['B'] }, action: { operation: 'register', participant_id: 'B' }, expected: { capacity: 1, confirmed: ['A'], waiting: ['B'] }, assertion: 'result' },
    { id: 'fifo', name: ['Cancel A: promote B before C', 'A 취소: C보다 먼저 대기한 B 확정'], initial: { capacity: 1, confirmed: ['A'], waiting: ['B', 'C'] }, action: { operation: 'cancel', participant_id: 'A' }, expected: { capacity: 1, confirmed: ['B'], waiting: ['C'] }, assertion: 'result' },
    { id: 'waiting-cancel', name: ['Cancel B: leave C waiting', 'B 취소: C는 계속 대기'], initial: { capacity: 1, confirmed: ['A'], waiting: ['B', 'C'] }, action: { operation: 'cancel', participant_id: 'B' }, expected: { capacity: 1, confirmed: ['A'], waiting: ['C'] }, assertion: 'result' },
    { id: 'nonmutation', name: ['Registration preserves the source', '신청해도 원본 상태는 유지'], initial: { capacity: 2, confirmed: [], waiting: [] }, action: { operation: 'register', participant_id: 'A' }, expected: { capacity: 2, confirmed: [], waiting: [] }, assertion: 'source' }
  ];
  function runTest(testId, bug = 'correct') {
    const test = testCases.find(value => value.id === testId);
    if (!test) throw new Error('Unknown test');
    const source = clone(test.initial), result = implementation(source, test.action, bug), actual = test.assertion === 'source' ? source : result;
    return { test: testId, bug, passed: equal(actual, test.expected), initial: clone(test.initial), action: { ...test.action }, assertion: test.assertion, expected: clone(test.expected), actual: clone(actual) };
  }
  function testMatrix(selected) {
    if (!Array.isArray(selected) || selected.some(id => !testCases.some(test => test.id === id)) || new Set(selected).size !== selected.length) throw new Error('Invalid test selection');
    return selected.map(test => ({ test, results: bugIds.map(bug => runTest(test, bug)) }));
  }
  const refactorCases = testCases.slice(0, 3).map(test => ({ id: test.id, name: test.name, initial: clone(test.initial), participant_id: test.action.participant_id })).concat([{ id: 'duplicate-confirmed', name: ['A is already confirmed', 'A는 이미 확정'], initial: { capacity: 2, confirmed: ['A'], waiting: [] }, participant_id: 'A' }]);
  const refactorVariants = {
    before: `function register(state, id) {\n  const next = copyState(state);\n  if (!next.confirmed.includes(id)) {\n    if (!next.waiting.includes(id)) {\n      if (next.confirmed.length >= next.capacity) {\n        next.waiting.push(id);\n      } else {\n        next.confirmed.push(id);\n      }\n    }\n  }\n  return next;\n}`,
    after: `function register(state, id) {\n  const next = copyState(state);\n  if (next.confirmed.includes(id) ||\n      next.waiting.includes(id)) return next;\n  if (next.confirmed.length >= next.capacity) {\n    next.waiting.push(id);\n    return next;\n  }\n  next.confirmed.push(id);\n  return next;\n}`,
    faulty: `function register(state, id) {\n  const next = copyState(state);\n  if (next.confirmed.includes(id) ||\n      next.waiting.includes(id)) return next;\n  // BUG: an exactly full workshop is not caught.\n  if (next.confirmed.length > next.capacity) {\n    next.waiting.push(id);\n    return next;\n  }\n  next.confirmed.push(id);\n  return next;\n}`
  };
  function beforeRegister(state, id) {
    const next = clone(state);
    if (!next.confirmed.includes(id)) { if (!next.waiting.includes(id)) { if (next.confirmed.length >= next.capacity) next.waiting.push(id); else next.confirmed.push(id); } }
    return next;
  }
  function afterRegister(state, id) {
    const next = clone(state);
    if (next.confirmed.includes(id) || next.waiting.includes(id)) return next;
    if (next.confirmed.length >= next.capacity) { next.waiting.push(id); return next; }
    next.confirmed.push(id); return next;
  }
  function compareRefactor(caseId, variant) {
    const example = refactorCases.find(value => value.id === caseId);
    if (!example || !['after', 'faulty'].includes(variant)) throw new Error('Unknown refactor preset');
    const expected = beforeRegister(example.initial, example.participant_id);
    const actual = variant === 'after' ? afterRegister(example.initial, example.participant_id) : implementation(example.initial, { operation: 'register', participant_id: example.participant_id }, 'boundary');
    return { initial: clone(example.initial), participant_id: example.participant_id, expected, actual, same: equal(expected, actual) };
  }
  const quizQuestions = [
    { id: 'expected-actual', prompt: ['At capacity 1, A is confirmed. What should registering B return?', '정원 1명에 A가 확정 상태입니다. B가 신청하면 어떤 상태가 되어야 할까요?'], illustration: { kind: 'state', state: { capacity: 1, confirmed: ['A'], waiting: [] }, action: 'register(state, "B")' }, options: [
      { id: 'wait', label: ['confirmed=[A], waiting=[B]', 'confirmed=[A], waiting=[B]'], correct: true, feedback: ['Correct. The expected state follows the capacity rule, not the current implementation.', '맞습니다. 기대값은 현재 구현이 아니라 정원 규칙에서 정합니다.'] },
      { id: 'overfill', label: ['confirmed=[A,B], waiting=[]', 'confirmed=[A,B], waiting=[]'], feedback: ['That exceeds capacity. If the implementation returns this, it is the actual value to fix, not the expected value.', '정원을 초과합니다. 구현이 이 값을 반환한다면 기대값을 바꿀 것이 아니라 실제 반환값을 고쳐야 합니다.'] },
      { id: 'drop', label: ['confirmed=[A], waiting=[]', 'confirmed=[A], waiting=[]'], feedback: ['B is new, not a duplicate. Registration must retain B on the waiting list.', 'B는 처음 신청했습니다. B를 대기 명단에 남겨야 합니다.'] }
    ] },
    { id: 'meaningful-red', prompt: ['Which failure is meaningful Red for the waiting-list rule?', '대기 규칙이 아직 구현되지 않았음을 보여 주는 실패는 무엇일까요?'], illustration: { kind: 'code', code: '# A: the intended test ran\nassert actual["waiting"] == ["B"]\n# AssertionError: [] != ["B"]\n\n# B: the test could not start\nModuleNotFoundError: No module named "pytest"' }, options: [
      { id: 'environment', label: ['pytest cannot be imported', 'pytest를 불러오지 못함'], feedback: ['An environment error does not test the waiting rule. Repair the environment, then obtain an assertion failure.', '환경 오류로는 대기 규칙을 검사하지 못합니다. 환경을 고친 다음 assertion 실패를 확인해야 합니다.'] },
      { id: 'assertion', label: ['The collected test reaches the assertion and finds waiting=[]', '테스트가 assertion까지 실행되어 대기 명단이 비어 있음을 확인'], correct: true, feedback: ['Correct. The test ran and exposed missing behavior. The saved TDD records show this kind of failure.', '맞습니다. 테스트가 실행되어 아직 없는 동작을 드러냈습니다. 저장된 TDD 실행 기록도 이런 실패를 보여 줍니다.'] },
      { id: 'any', label: ['Any red terminal message is enough', '터미널에 빨간 글자만 나오면 충분'], feedback: ['Color does not identify the cause. Check whether the intended test ran and failed for the missing behavior.', '색만으로는 실패 원인을 알 수 없습니다. 의도한 테스트가 실행되어 아직 없는 동작 때문에 실패했는지 확인하세요.'] }
    ] },
    { id: 'refactor-rule', prompt: ['Which change is a refactor under the FIFO contract?', 'FIFO 규칙을 유지하는 refactor는 어느 것일까요?'], illustration: { kind: 'code', code: '# waiting = ["B", "C"]\n# cancel confirmed A\n# contract: promote B, keep C waiting' }, options: [
      { id: 'lifo', label: ['Promote C because removing the last element is shorter', '코드가 짧아지도록 마지막 대기자 C를 확정'], feedback: ['That changes FIFO to LIFO. Shorter code does not make a rule change a refactor.', 'FIFO를 LIFO로 바꾼 것입니다. 코드가 짧아져도 규칙 변경이 refactor가 되지는 않습니다.'] },
      { id: 'capacity', label: ['Increase capacity so nobody waits', '아무도 기다리지 않도록 정원을 늘림'], feedback: ['Capacity is part of the input contract. Altering it changes behavior rather than structure alone.', '정원은 입력 계약의 일부입니다. 정원을 바꾸면 구조뿐 아니라 동작도 바뀝니다.'] },
      { id: 'guard', label: ['Replace nested branches with guard clauses; keep the same state results', '중첩 분기를 guard clause로 바꾸고 상태 결과는 유지'], correct: true, feedback: ['Correct. Internal structure changes while the public behavior stays the same.', '맞습니다. 내부 구조를 바꾸면서 공개 동작은 유지합니다.'] }
    ] },
    { id: 'coverage-assertion', prompt: ['This test executes register. What is missing?', '이 테스트는 register를 실행합니다. 무엇이 빠졌을까요?'], illustration: { kind: 'code', code: 'def test_full_workshop():\n    state = {"capacity": 1,\n             "confirmed": ["A"],\n             "waiting": []}\n    register(state, "B")\n    # no assertion' }, options: [
      { id: 'assert', label: ['Assert the returned waiting list and unchanged source', '반환된 대기 명단을 확인하고 원본 상태가 유지됐는지 검사'], correct: true, feedback: ['Correct. Executing a line alone cannot reject a wrong returned state or source mutation.', '맞습니다. 한 줄을 실행했다는 사실만으로 잘못된 반환값이나 원본 변경을 잡아내지는 못합니다.'] },
      { id: 'coverage', label: ['Nothing: reaching the line proves correctness', '없음: 해당 줄을 실행했으므로 정확함'], feedback: ['Coverage records execution, not whether the result met the rule. An assertion must compare behavior with an expectation.', 'Coverage는 실행 여부를 나타냅니다. 결과가 규칙을 만족하는지는 assertion으로 기대값과 비교해야 합니다.'] },
      { id: 'rename', label: ['Rename the test to test_everything', '이름을 test_everything으로 변경'], feedback: ['A broader name adds no check. Add a specific expected result.', '테스트 이름을 바꿔도 검사 내용은 늘어나지 않습니다. 구체적인 기대 결과를 검사해야 합니다.'] }
    ] },
    { id: 'public-helper', prompt: ['Which test can survive replacing a private helper?', '비공개 helper를 교체해도 유지할 수 있는 테스트는 어느 것일까요?'], illustration: { kind: 'state', state: { capacity: 1, confirmed: ['A'], waiting: ['B', 'C'] }, action: 'cancel(state, "A")' }, options: [
      { id: 'helper', label: ['Require the private _promote helper to be called exactly once', '비공개 _promote가 정확히 한 번 호출되도록 요구'], feedback: ['A correct implementation may inline or replace that helper. This assertion binds the test to the current structure.', '올바른 구현도 helper를 없애거나 바꿀 수 있습니다. 이 검사는 현재 내부 구조에 묶여 있습니다.'] },
      { id: 'public', label: ['Assert confirmed=[B], waiting=[C] through cancel', 'cancel의 결과가 confirmed=[B], waiting=[C]인지 검사'], correct: true, feedback: ['Correct. The assertion names the public FIFO behavior without requiring a private call sequence.', '맞습니다. 비공개 함수의 호출 순서를 요구하지 않고 공개된 FIFO 동작을 검사합니다.'] },
      { id: 'private-name', label: ['Assert the private helper has the same name', '비공개 helper 이름이 그대로인지 검사'], feedback: ['The helper name is not the user-visible result. Check the state returned by cancel.', 'helper 이름은 사용자가 관찰하는 결과가 아닙니다. cancel이 반환하는 상태를 검사하세요.'] }
    ] }
  ];
  function answerQuiz(questionId, optionId) {
    const question = quizQuestions.find(item => item.id === questionId), option = question?.options.find(item => item.id === optionId);
    if (!option) throw new Error('Unknown quiz answer');
    return { correct: option.correct === true, feedback: option.feedback.slice() };
  }
  const api = { ids, validateState, createState, register, cancel, operation, replay, rule, bugIds, bugLabels, implementation, testCases, runTest, testMatrix, refactorVariants, refactorCases, beforeRegister, afterRegister, compareRefactor, quizQuestions, answerQuiz };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const registerLab = window.ArticleLabs.register;
  function setup(ctx) { ctx.root.classList.add('cc-lab', 'cleancode-lab', 'al-workshop'); }
  const translated = (ctx, pair) => ctx.text(pair[0], pair[1]);
  const paragraph = (ctx, en, ko, attrs = {}) => ctx.el('p', attrs, ctx.text(en, ko));
  function codeBlock(ctx, code, title) { return ctx.el('div', { class: 'cc-code' }, [title ? ctx.el('strong', {}, title) : null, ctx.el('pre', { tabindex: '0' }, ctx.el('code', {}, code))]); }
  function stateCard(ctx, state, title) {
    return ctx.el('section', { class: 'cc-state-card', 'aria-label': title }, [ctx.el('strong', {}, title), ctx.el('p', {}, `${ctx.text('Capacity', '정원')}: ${state.capacity}`), ...[['confirmed', state.confirmed, 'Confirmed', '확정'], ['waiting', state.waiting, 'Waiting · FIFO →', '대기 · FIFO →']].map(([key, values, en, ko]) => ctx.el('div', { class: 'cc-lane' }, [ctx.el('span', { class: 'cc-lane-label' }, ctx.text(en, ko)), ctx.el('div', { class: 'cc-tokens', 'data-list': key }, values.length ? values.map((id, i) => ctx.el('span', { class: 'cc-token', 'aria-label': `${id}, ${i + 1}` }, id)) : ctx.el('span', { class: 'cc-empty' }, '∅'))]))]);
  }
  function validKeys(state, keys) { return state && typeof state === 'object' && !Array.isArray(state) && Object.keys(state).sort().join(',') === keys.slice().sort().join(','); }
  function boundedIndex(value, length) { return Number.isInteger(value) && value >= 0 && value < length; }
  function selectOptions(items) { return items.map(item => ({ value: item.id, en: item.name[0], ko: item.name[1] })); }
  registerLab('cleancode-state', ctx => {
    setup(ctx);
    const capacity = ctx.control('Capacity · fixed until Reset', '정원 · 초기화 전까지 고정', { type: 'select', value: '2', options: [1, 2, 3, 4].map(n => ({ value: String(n), en: `${n} ${n === 1 ? 'seat' : 'seats'}`, ko: `${n}명` })) });
    const participant = ctx.control('Participant ID', '참가자 ID', { type: 'select', value: 'A', options: ids.map(value => ({ value, en: value, ko: value })) });
    const count = ctx.metric('Actions / limit', '실행한 동작 / 최대'), occupancy = ctx.metric('Confirmed / capacity', '확정 / 정원');
    let history = [], fixedCapacity = 2;
    const board = ctx.el('div', { class: 'cc-state-pair' }), explanation = ctx.el('p', { class: 'cc-feedback', role: 'status' }), historyView = ctx.el('details', { class: 'cc-history' });
    ctx.view.append(board, explanation, historyView);
    function paint() {
      if (!history.length) fixedCapacity = Number(capacity.value);
      const result = replay(fixedCapacity, history);
      capacity.value = String(fixedCapacity); capacity.disabled = history.length > 0;
      board.replaceChildren(stateCard(ctx, result.previous, ctx.text('Previous state', '이전 상태')), stateCard(ctx, result.current, ctx.text('New state', '새 상태')));
      explanation.textContent = translated(ctx, ruleText[history.length ? rule(result.previous, history[history.length - 1]) : 'start']);
      historyView.replaceChildren(ctx.el('summary', {}, ctx.text(`Action history (${history.length})`, `실행 기록 (${history.length})`)), ctx.el('ol', {}, history.map(action => ctx.el('li', {}, `${action.operation}("${action.participant_id}")`))));
      count.textContent = `${history.length} / 30`; occupancy.textContent = `${result.current.confirmed.length} / ${fixedCapacity}`;
      ctx.status.textContent = ctx.text('A–F · capacity 1–4 · source state is preserved · Reset starts a new history.', 'A–F · 정원 1–4명 · 원본 상태 유지 · ‘초기값으로’를 누르면 새 기록을 시작합니다.');
      registerButton.disabled = cancelButton.disabled = history.length >= 30;
    }
    function act(operation) {
      if (history.length >= 30) { ctx.status.textContent = ctx.text('30 actions reached. Reset to begin again.', '동작 30회에 도달했습니다. ‘초기값으로’를 눌러 다시 시작하세요.'); return; }
      if (!ids.includes(participant.value)) return;
      if (!history.length) fixedCapacity = Number(capacity.value);
      history.push({ operation, participant_id: participant.value }); paint();
    }
    const registerButton = ctx.button('Register', '신청', () => act('register')), cancelButton = ctx.button('Cancel', '취소', () => act('cancel'));
    ctx.onReset(() => { history = []; fixedCapacity = Number(capacity.value); });
    ctx.onChange(paint);
    ctx.stateLink('cleancode-state', { read: () => ({ capacity: fixedCapacity, participant: participant.value, history: history.map(action => ({ ...action })) }), write: state => {
      if (!validKeys(state, ['capacity', 'participant', 'history']) || !ids.includes(state.participant)) return false;
      replay(state.capacity, state.history); const nextHistory = state.history.map(action => ({ ...action }));
      fixedCapacity = state.capacity; capacity.value = String(state.capacity); participant.value = state.participant; history = nextHistory; paint(); return true;
    } });
  });
  registerLab('cleancode-tdd', async ctx => {
    setup(ctx);
    const records = await ctx.fetchJSON('/assets/data/cleancode/tdd-runs.json'), snapshots = records.snapshots;
    if (!Array.isArray(snapshots) || !snapshots.length || snapshots.some(item => !['red', 'green', 'refactor'].includes(item.stage) || typeof item.code !== 'string' || typeof item.test !== 'string' || !['failed', 'passed', 'error'].includes(item.outcome) || !Number.isInteger(item.exit_code) || !item.failure_reason?.en || !item.failure_reason?.ko)) throw new Error('Invalid saved Python execution records');
    const step = ctx.control('Saved execution', '저장된 실행', { type: 'select', value: snapshots[0].id, options: snapshots.map((item, i) => ({ value: item.id, en: `${i + 1}. ${item.title.en}`, ko: `${i + 1}. ${item.title.ko}` })) });
    const position = ctx.metric('Record', '실행 기록'), outcome = ctx.metric('pytest outcome', 'pytest 결과');
    const body = ctx.el('div', { class: 'cc-tdd' }); ctx.view.append(body);
    function paint() {
      const index = snapshots.findIndex(item => item.id === step.value), item = snapshots[index];
      if (!item) return;
      position.textContent = `${index + 1} / ${snapshots.length}`; outcome.textContent = `${item.outcome.toUpperCase()} · exit ${item.exit_code}`;
      body.replaceChildren(paragraph(ctx, 'Saved Python execution record · no Python or pytest runs in this browser.', '저장된 Python 실행 기록 · 브라우저에서 Python이나 pytest를 실행하지 않습니다.', { class: 'cc-record-label' }), ctx.el('h4', {}, ctx.ko ? item.title.ko : item.title.en), codeBlock(ctx, item.code, ctx.text('Implementation snapshot', '구현 스냅샷')), codeBlock(ctx, item.test, ctx.text('Test snapshot', '테스트 스냅샷')), codeBlock(ctx, `${item.command}\n\n${item.output_excerpt || item.stdout}`, ctx.text('Recorded command and output', '실제 실행 명령과 출력')), paragraph(ctx, item.failure_reason.en, item.failure_reason.ko, { class: 'cc-feedback', role: 'status' }));
      ctx.status.textContent = `Python ${records.python_version} · pytest ${records.pytest_version} · ${item.summary}`;
      previous.disabled = index === 0; next.disabled = index === snapshots.length - 1;
    }
    const previous = ctx.button('Previous record', '이전 기록', () => { const i = snapshots.findIndex(item => item.id === step.value); if (i > 0) { step.value = snapshots[i - 1].id; paint(); } });
    const next = ctx.button('Next record', '다음 기록', () => { const i = snapshots.findIndex(item => item.id === step.value); if (i < snapshots.length - 1) { step.value = snapshots[i + 1].id; paint(); } });
    ctx.onChange(paint);
    ctx.stateLink('cleancode-tdd', { read: () => ({ snapshot: step.value }), write: state => { if (!validKeys(state, ['snapshot']) || !snapshots.some(item => item.id === state.snapshot)) return false; step.value = state.snapshot; paint(); return true; } });
  });
  registerLab('cleancode-refactor', ctx => {
    setup(ctx);
    const selected = ctx.control('Registration input', '신청 입력', { type: 'select', value: 'at-capacity', options: selectOptions(refactorCases) });
    const variant = ctx.control('Compare with before', '변경 전과 비교', { type: 'select', value: 'after', options: [{ value: 'after', en: 'After · guard clauses', ko: '변경 후 · guard clause' }, { value: 'faulty', en: 'Faulty · wrong boundary', ko: '오류 버전 · 잘못된 경계' }] });
    const result = ctx.metric('Selected input', '선택한 입력');
    const body = ctx.el('div', { class: 'cc-refactor' }); ctx.view.append(body);
    function paint() {
      const comparison = compareRefactor(selected.value, variant.value);
      result.textContent = comparison.same ? ctx.text('Same returned state', '반환 상태가 같음') : ctx.text('Counterexample found', '반례 발견');
      const codes = ctx.el('details', { class: 'cc-variants' }, [ctx.el('summary', {}, ctx.text('Compare finite JavaScript implementations', '고정된 JavaScript 구현 비교')), paragraph(ctx, 'All variants use copyState = state => ({capacity: state.capacity, confirmed: [...state.confirmed], waiting: [...state.waiting]}). Inputs have already passed the same state and ID validation.', '모든 버전은 copyState = state => ({capacity: state.capacity, confirmed: [...state.confirmed], waiting: [...state.waiting]})를 사용합니다. 입력에는 동일한 상태·ID 검증을 미리 적용합니다.'), codeBlock(ctx, refactorVariants.before, ctx.text('Before · nested branches', '변경 전 · 중첩 분기')), codeBlock(ctx, refactorVariants[variant.value], ctx.text('Selected variant', '선택한 버전'))]);
      body.replaceChildren(stateCard(ctx, comparison.initial, `register(state, "${comparison.participant_id}")`), ctx.el('div', { class: 'cc-state-pair' }, [stateCard(ctx, comparison.expected, ctx.text('Before result', '변경 전 결과')), stateCard(ctx, comparison.actual, ctx.text('Selected result', '선택한 버전의 결과'))]), paragraph(ctx, comparison.same ? 'These implementations return the same state for this input. This finite comparison is not a proof for every input.' : 'The workshop is exactly full. Changing >= to > confirms one extra participant: this changes the rule, so it is not a refactor.', comparison.same ? '이 입력에서는 두 구현의 반환 상태가 같습니다. 유한한 입력 비교가 모든 입력에 대한 증명은 아닙니다.' : '정원이 정확히 찬 상태입니다. >=를 >로 바꾸면 한 명을 더 확정합니다. 규칙이 달라졌으므로 refactor가 아닙니다.', { class: 'cc-feedback', role: 'status' }), codes);
      ctx.status.textContent = ctx.text('Observable result: capacity and ordered confirmed/waiting IDs. No cleanliness score.', '관찰하는 결과: 정원과 순서가 있는 확정·대기 ID.');
    }
    ctx.onChange(paint);
    ctx.stateLink('cleancode-refactor', { read: () => ({ input: selected.value, variant: variant.value }), write: state => { if (!validKeys(state, ['input', 'variant']) || !refactorCases.some(item => item.id === state.input) || !['after', 'faulty'].includes(state.variant)) return false; selected.value = state.input; variant.value = state.variant; paint(); return true; } });
  });
  registerLab('cleancode-tests', ctx => {
    setup(ctx);
    const selected = testCases.map(test => ctx.control(test.name[0], test.name[1], { type: 'checkbox', value: test.id === 'free-seat' }));
    const caught = ctx.metric('Faults detected / four', '발견한 오류 / 4개');
    const body = ctx.el('div', { class: 'cc-test-results' }); ctx.view.append(body);
    function paint() {
      const chosen = testCases.filter((_, i) => selected[i].checked), matrix = testMatrix(chosen.map(test => test.id));
      caught.textContent = `${bugIds.filter(bug => chosen.some(test => !runTest(test.id, bug).passed)).length} / 4`;
      body.replaceChildren(paragraph(ctx, 'PASS means this test did not detect that fault. FAIL means its assertion found a counterexample. These are finite JavaScript implementations, not a coverage measurement.', 'PASS는 해당 테스트가 그 오류를 잡지 못했다는 뜻이고, FAIL은 assertion이 반례를 찾았다는 뜻입니다. 고정된 JavaScript 구현의 실행 결과이며 coverage 측정값은 아닙니다.'));
      if (!chosen.length) body.append(paragraph(ctx, 'No tests selected: none of these faults is checked.', '테스트를 선택하지 않아 어떤 오류도 검사하지 않습니다.'));
      for (const bug of bugIds) {
        const section = ctx.el('section', { class: 'cc-bug-card' }, ctx.el('h4', {}, translated(ctx, bugLabels[bug])));
        section.append(ctx.el('ul', { class: 'cc-matrix-list' }, matrix.map(row => { const item = row.results.find(item => item.bug === bug), test = testCases.find(test => test.id === row.test); return ctx.el('li', { class: item.passed ? 'cc-pass' : 'cc-fail' }, `${item.passed ? 'PASS' : 'FAIL'} · ${translated(ctx, test.name)}`); })));
        const failure = matrix.map(row => row.results.find(item => item.bug === bug)).find(item => !item.passed);
        if (failure) section.append(ctx.el('details', {}, [ctx.el('summary', {}, ctx.text('Exact failing counterexample', '실패한 사례 자세히 보기')), codeBlock(ctx, `${JSON.stringify(failure.initial)}\n${failure.action.operation}(state, "${failure.action.participant_id}")\n${ctx.text('Assert', '검사 대상')}: ${failure.assertion === 'source' ? ctx.text('unchanged source', '바뀌지 않은 원본') : ctx.text('returned state', '반환 상태')}\n${ctx.text(failure.assertion === 'source' ? 'Expected source state' : 'Expected returned state', failure.assertion === 'source' ? '기대 원본 상태' : '기대 반환 상태')}: ${JSON.stringify(failure.expected)}\n${ctx.text(failure.assertion === 'source' ? 'Actual source state' : 'Actual returned state', failure.assertion === 'source' ? '실제 원본 상태' : '실제 반환 상태')}:   ${JSON.stringify(failure.actual)}`)]));
        body.append(section);
      }
      ctx.status.textContent = ctx.text(`${chosen.length} selected tests × 4 fault implementations. Only the stated assertions are checked.`, `선택한 테스트 ${chosen.length}개 × 오류 구현 4개. 각 테스트에 명시된 assertion만 검사합니다.`);
    }
    ctx.onChange(paint);
    ctx.stateLink('cleancode-tests', { read: () => ({ tests: testCases.filter((_, i) => selected[i].checked).map(test => test.id) }), write: state => { if (!validKeys(state, ['tests'])) return false; testMatrix(state.tests); selected.forEach((control, i) => { control.checked = state.tests.includes(testCases[i].id); }); paint(); return true; } });
  });
  registerLab('cleancode-quiz', ctx => {
    setup(ctx);
    let index = 0, answers = Array(quizQuestions.length).fill(null);
    const score = ctx.metric('Correct / answered', '정답 / 응답'), progress = ctx.metric('Question', '문항');
    const body = ctx.el('div', { class: 'cc-quiz' }); ctx.view.append(body);
    function paint(focusChoice = -1) {
      const question = quizQuestions[index], selected = answers[index], answer = selected === null ? null : answerQuiz(question.id, selected);
      progress.textContent = `${index + 1} / ${quizQuestions.length}`; score.textContent = `${answers.filter((id, i) => id !== null && answerQuiz(quizQuestions[i].id, id).correct).length} / ${answers.filter(id => id !== null).length}`;
      const illustration = question.illustration.kind === 'state'
        ? stateCard(ctx, question.illustration.state, question.illustration.action)
        : codeBlock(ctx, question.illustration.code);
      const choices = question.options.map((option, choiceIndex) => {
        const pressed = selected === option.id;
        const button = ctx.el('button', {
          type: 'button', class: pressed ? 'cc-answer cc-selected' : 'cc-answer', 'aria-pressed': String(pressed)
        }, translated(ctx, option.label));
        button.addEventListener('click', () => {
          answers[index] = option.id;
          paint(choiceIndex);
        });
        return button;
      });
      const feedback = answer
        ? paragraph(ctx, answer.feedback[0], answer.feedback[1], { class: 'cc-feedback', role: 'status' })
        : paragraph(ctx, 'Choose an answer to see feedback for that option.', '답을 선택하면 해당 선택지의 설명이 나타납니다.', { class: 'cc-feedback' });
      body.replaceChildren(
        ctx.el('h4', {}, translated(ctx, question.prompt)),
        illustration,
        ctx.el('div', { class: 'cc-answers', role: 'group', 'aria-label': ctx.text('Answer choices', '답안 선택') }, choices),
        feedback
      );
      if (focusChoice >= 0) choices[focusChoice].focus({ preventScroll: true });
      previous.disabled = index === 0; next.disabled = index === quizQuestions.length - 1;
      ctx.status.textContent = ctx.text('Five rule and testing questions. You may revise any answer.', '규칙과 테스트에 관한 다섯 문항입니다. 답은 언제든 바꿀 수 있습니다.');
    }
    const previous = ctx.button('Previous question', '이전 문항', () => { if (index > 0) { index--; paint(); } }), next = ctx.button('Next question', '다음 문항', () => { if (index < quizQuestions.length - 1) { index++; paint(); } });
    ctx.onReset(() => { index = 0; answers = Array(quizQuestions.length).fill(null); }); ctx.onChange(paint);
    ctx.stateLink('cleancode-quiz', { read: () => ({ index, answers: answers.slice() }), write: state => {
      if (!validKeys(state, ['index', 'answers']) || !boundedIndex(state.index, quizQuestions.length) || !Array.isArray(state.answers) || state.answers.length !== quizQuestions.length || Array.from(state.answers).some((answer, i) => answer !== null && !quizQuestions[i].options.some(option => option.id === answer))) return false;
      index = state.index; answers = state.answers.slice(); paint(); return true;
    } });
  });
}());
