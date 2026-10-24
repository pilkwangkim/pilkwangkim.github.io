/* Local teaching examples and explicitly scoped arithmetic from the Gemma series. */
(function () {
  'use strict';
  function normalizeLimit(value, repaired) { return repaired ? (value === null ? 10 : value) : (value || 10); }
  function admittedCalls(limit, allowance, initial, growth) {
    return initial + allowance > limit ? 0 : 1 + Math.floor((limit - allowance - initial) / growth);
  }
  function parsedPath(text) {
    const token = '<|"|>', start = text.indexOf(token), end = text.indexOf(token, start + token.length);
    return start < 0 || end < 0 ? null : text.slice(start + token.length, end);
  }
  function phi(z) {
    const sign = z < 0 ? -1 : 1, x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + .3275911 * x);
    const erf = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - .284496736) * t + .254829592) * t) * Math.exp(-x * x);
    return .5 * (1 + sign * erf);
  }
  function finishChance(tasks, startup, mean, sd, multiplier = 1, limit = 720) {
    const total = startup + tasks * mean * multiplier;
    return sd > 0 ? phi((limit - total) / (sd * multiplier * Math.sqrt(tasks))) : Number(total <= limit);
  }
  function paired(both, loss, gain, neither) {
    const n = both + loss + gain + neither;
    return { n, baseline: both + loss, candidate: both + gain, delta: n ? (gain - loss) / n : 0, changed: loss + gain };
  }
  const api = { normalizeLimit, admittedCalls, parsedPath, phi, finishChance, paired };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const reg = (id, fn) => window.ArticleLabs.register(id, fn);
  const v = el => Number(el.value);
  const clear = ctx => ctx.view.replaceChildren();
  reg('gemma-bug', ctx => {
    const input = ctx.control('Parsed input', '변환된 입력', { type: 'select', value: '0', options: [{ value: 'none', en: 'Omitted → None', ko: '생략 → None' }, { value: '0', en: 'Explicit zero', ko: '명시한 0' }, { value: '5', en: 'Ordinary value: five', ko: '보통 값: 5' }] });
    const repair = ctx.control('Use an explicit None check', 'None을 명시적으로 검사', { type: 'checkbox', value: false });
    const out = ctx.metric('Returned limit', '반환 limit');
    ctx.onChange(() => {
      const x = input.value === 'none' ? null : v(input), result = normalizeLimit(x, repair.checked);
      out.textContent = String(result); clear(ctx);
      const labels = ['None', '0', '5'], xs = [null, 0, 5];
      ctx.view.append(ctx.el('pre', {}, ctx.el('code', {}, repair.checked ? 'return default if value is None else value' : 'return value or default')));
      ctx.view.append(ctx.table([ctx.text('Input', '입력'), ctx.text('Helper result', 'helper 결과'), ctx.text('Required', '요구값'), ctx.text('Check', '검사')], xs.map((x, i) => [labels[i], normalizeLimit(x, repair.checked), i === 0 ? 10 : x, normalizeLimit(x, repair.checked) === (i === 0 ? 10 : x) ? 'PASS' : 'FAIL'])));
      ctx.status.textContent = ctx.text(`Request → options → shared helper → limit ${result}. This invented API has at least ten items and defines an explicit zero as an empty result. The three checks do not define other input types.`, `요청 → options → 공용 helper → limit ${result}. 이 예제 API는 item이 10개 이상이며 명시한 0은 빈 결과를 뜻합니다. 세 검사가 다른 입력형의 계약까지 정하지는 않습니다.`);
    });
  });
  reg('gemma-patch', ctx => {
    const edit = ctx.control('Repair the tracked implementation', '추적 중인 구현 파일 수정', { type: 'checkbox', value: true });
    const scratch = ctx.control('Leave a new scratch script in the repository', '저장소에 새 scratch script 남기기', { type: 'checkbox', value: true });
    const process = ctx.control('Fix only the running process', '실행 중인 process만 수정', { type: 'checkbox', value: false });
    const out = ctx.metric('Implementation repair reaches the verifier', '구현 수정이 verifier에 전달됨');
    ctx.onChange(() => { out.textContent = ctx.text(edit.checked ? 'Yes' : 'No', edit.checked ? '예' : '아니요'); clear(ctx); ctx.view.append(ctx.table([ctx.text('Object', '대상'), ctx.text('Attempt state', '시도 중 상태'), ctx.text('Extracted patch', '추출 patch'), ctx.text('Fresh verifier', '새 verifier')], [
      ['query/limits.py', edit.checked ? 'edited' : 'baseline', edit.checked ? 'diff included' : 'no diff', edit.checked ? 'repair applied' : 'old helper'],
      ['repro_limit.py', scratch.checked ? 'new, untracked' : 'absent', scratch.checked ? 'included by git add -N .' : 'absent', scratch.checked ? 'extra file applied' : 'absent'],
      [ctx.text('Running Python state', '실행 중인 Python 상태'), process.checked ? 'changed' : 'unchanged', ctx.text('No transfer', '전달 안 됨'), ctx.text('Fresh process', '새 process')]
    ])); ctx.status.textContent = ctx.text('A bounded extraction model of the inspected swegemma 0.2.7: git add -N . followed by git diff --binary. Non-ignored new files can enter the diff. Patch delivery does not itself establish that verification passes.', '확인한 swegemma 0.2.7의 추출 경계를 보여 줍니다. git add -N . 뒤에 git diff --binary를 실행하여 ignore하지 않은 새 파일도 diff에 들어갈 수 있습니다. patch 전달만으로 검증 통과가 성립하지는 않습니다.'); });
  });
  reg('gemma-parser', ctx => {
    const mode = ctx.control('String termination', '문자열 종료 방식', { type: 'select', value: 'bad', options: [{ value: 'good', en: 'Special delimiter closes the path', ko: '특수 delimiter로 경로 종료' }, { value: 'bad', en: 'Ordinary escaped quote leaves it open', ko: '일반 escaped quote로 문자열이 열린 상태' }] });
    const out = ctx.metric('Parsed filepath', '변환된 filepath');
    ctx.onChange(() => { const good = 'read_file{filepath:<|"|>fastapi/openapi/utils.py<|"|>,start_line:400,end_line:568}', bad = 'read_file{filepath:<|"|>fastapi/openapi/utils.py\\",start_line\\":400<|"|>,end_line:568}', text = mode.value === 'good' ? good : bad; out.textContent = parsedPath(text); clear(ctx); ctx.view.append(ctx.el('pre', {}, ctx.el('code', {}, text))); ctx.view.append(ctx.table([ctx.text('Field', '필드'), ctx.text('What reaches the tool', 'tool에 전달된 값')], [['filepath', parsedPath(text)], ['start_line', mode.value === 'good' ? '400' : ctx.text('absorbed into filepath', 'filepath에 흡수됨')], ['end_line', '568']])); ctx.status.textContent = ctx.text('Teaching reconstruction from recorded arguments in Part 2; the original generated tokens were not retained. A normal quote is not the <|"|> delimiter. Repeating this request cannot correct the constructed filename.', 'Part 2에 기록된 인자에서 재구성한 학습용 예입니다. 원래 생성 token은 보존되지 않았습니다. 일반 따옴표는 <|"|> delimiter가 아니며 같은 요청을 반복해도 잘못 구성된 파일명은 고쳐지지 않습니다.'); });
  });
  reg('gemma-context', ctx => {
    const allowance = ctx.control('Requested maximum output tokens', '요청한 최대 output token', { type: 'range', min: 1024, max: 16384, step: 1024, value: 8192 });
    const initial = ctx.control('First prompt tokens', '첫 prompt token', { type: 'range', min: 1000, max: 24000, step: 1, value: 18033 });
    const growth = ctx.control('Tokens added per completed call', '완료 call당 추가 token', { type: 'range', min: 20, max: 500, step: 1, value: 119 });
    const out = ctx.metric('Admitted calls, including the first', '첫 call을 포함한 허용 call 수');
    ctx.onChange(() => { const n = admittedCalls(32768, v(allowance), v(initial), v(growth)); out.textContent = String(n); clear(ctx); const points = Array.from({ length: Math.min(100, n + 3) }, (_, i) => [i + 1, v(initial) + i * v(growth) + v(allowance)]); const ymax = Math.max(32768, ...points.map(p => p[1])); ctx.view.append(ctx.plot({ x: [1, Math.max(3, points.length)], y: [0, ymax * 1.04], xLabel: ctx.text('Request number', '요청 번호'), yLabel: ctx.text('Prompt + reserved output tokens', 'prompt + 예약 output token'), series: [{ color: 'blue', label: ctx.text('Requested capacity', '요청 용량'), points }, { color: 'red', label: '32,768', points: [[1, 32768], [Math.max(3, points.length), 32768]], dash: true }] })); ctx.status.textContent = ctx.text('The recorded diagnostic starts at 18,033 prompt tokens, reserves 8,192, and adds 119 per call: 55 calls fit and request 56 is rejected. The slider model assumes constant growth and no compaction; output reservation is checked before generation.', '기록된 진단의 시작 prompt는 18,033 token, 예약 output은 8,192, call당 증가는 119입니다. 55개 call이 들어가고 56번째 요청은 거절됩니다. 조절 모형은 일정한 증가량과 compaction이 없음을 가정하며 output 예약량은 생성 전에 검사합니다.'); });
  });
  reg('gemma-timing', ctx => {
    const mean = ctx.control('Reference mean per task (minutes)', '기준 task당 평균 (분)', { type: 'range', min: 3, max: 6, step: .1, value: 5 });
    const multiplier = ctx.control('Task-time multiplier', 'task 시간 배율', { type: 'range', min: .8, max: 1.4, step: .01, value: 1.18 });
    const solve = ctx.control('Candidate solves, assuming 58 Public tasks', 'Public 58개를 가정한 candidate 해결 수', { type: 'range', min: 1, max: 20, step: 1, value: 12 });
    const out = ctx.metric('Completion chance / expected utility', '완료 확률 / 기대 utility');
    ctx.onChange(() => { const p = finishChance(120, 20, v(mean), 2.5, v(multiplier)), u = p * v(solve) / 58; out.textContent = `${ctx.format(100 * p, 1)}% / ${ctx.format(u, 3)}`; clear(ctx); ctx.view.append(ctx.plot({ x: [.8, 1.4], y: [0, 1], xLabel: ctx.text('Task-time multiplier', 'task 시간 배율'), yLabel: ctx.text('P(complete within 12 h)', 'P(12시간 내 완료)'), series: [{ color: 'blue', points: Array.from({ length: 61 }, (_, i) => { const x = .8 + .01 * i; return [x, finishChance(120, 20, v(mean), 2.5, x)]; }) }], markers: [{ x: v(multiplier), y: p, color: 'gold', label: ctx.format(p, 3) }] })); ctx.status.textContent = ctx.text('Hypothetical normal-sum model: 120 independent tasks, 20-minute startup, reference SD 2.5 minutes; multiplier scales mean and SD together. Utility is completion probability × raw solve fraction. A timeout is recorded as an error by Kaggle; zero is our planning utility. These are not hosted timing estimates.', '가정한 정규합 모형입니다. 독립 task 120개, 공통 시작 시간 20분, 기준 SD 2.5분이며 배율은 평균과 SD를 함께 바꿉니다. utility는 완료 확률 × 반올림 전 해결 비율입니다. Kaggle은 timeout을 오류로 기록하며 여기서 0은 계획용 utility입니다. hosted 시간 추정값이 아닙니다.'); });
  });
  reg('gemma-paired', ctx => {
    const both = ctx.control('Both solve', '둘 다 해결', { type: 'range', min: 0, max: 12, step: 1, value: 2 });
    const loss = ctx.control('Baseline only', 'baseline만 해결', { type: 'range', min: 0, max: 12, step: 1, value: 2 });
    const gain = ctx.control('Candidate only', 'candidate만 해결', { type: 'range', min: 0, max: 12, step: 1, value: 3 });
    const neither = ctx.control('Neither solves', '둘 다 미해결', { type: 'range', min: 0, max: 12, step: 1, value: 5 });
    const out = ctx.metric('Paired difference', 'paired 차이');
    ctx.onChange(() => { const r = paired(v(both), v(loss), v(gain), v(neither)); out.textContent = r.n ? `${ctx.format(r.delta * 100, 2)} pp` : '—'; clear(ctx); ctx.view.append(ctx.table([ctx.text('Same issues', '동일 issue'), ctx.text('Candidate solves', 'candidate 해결'), ctx.text('Candidate unresolved', 'candidate 미해결')], [[ctx.text('Baseline solves', 'baseline 해결'), v(both), v(loss)], [ctx.text('Baseline unresolved', 'baseline 미해결'), v(gain), v(neither)]])); ctx.status.textContent = ctx.text(`Invented paired outcomes: baseline ${r.baseline}/${r.n}, candidate ${r.candidate}/${r.n}; inspect all ${r.changed} changed outcomes. Gains and regressions cancel in the mean, not in their failure mechanisms.`, `가정한 paired 결과: baseline ${r.baseline}/${r.n}, candidate ${r.candidate}/${r.n}이며 달라진 ${r.changed}개 결과를 확인해야 합니다. 평균에서 상쇄된 gain과 regression도 실패 원인은 다를 수 있습니다.`); });
  });
})();
