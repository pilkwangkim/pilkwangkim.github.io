/* ARC article models use local illustrative grids and frozen historical constants. */
(function () {
  'use strict';
  function rhae(human, agent, completed) { return completed && agent > 0 ? Math.min((human / agent) ** 2, 1.15) : 0; }
  function gameScore(levels) {
    const denom = levels.reduce((s, _, i) => s + i + 1, 0);
    const weighted = levels.reduce((s, l, i) => s + (i + 1) * rhae(l.h, l.a, l.done), 0);
    const cap = levels.reduce((s, l, i) => s + (l.done ? i + 1 : 0), 0);
    return Math.min(cap, weighted) / denom;
  }
  function tokenBudget(runs, concurrency, cap, throughput) {
    return { waves: Math.ceil(runs / concurrency), tokens: throughput * cap / concurrency, wall: Math.ceil(runs / concurrency) * cap };
  }
  function gatedPlan(plan, actual) {
    const executed = [];
    for (let i = 0; i < plan.length; i++) { executed.push(actual[i]); if (plan[i] !== actual[i]) return { executed, mismatch: i, dropped: plan.length - i - 1 }; }
    return { executed, mismatch: -1, dropped: 0 };
  }
  const api = { rhae, gameScore, tokenBudget, gatedPlan };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const reg = (id, fn) => window.ArticleLabs.register(id, fn), v = el => Number(el.value);
  reg('arc-rhae', ctx => {
    const human = ctx.control('Human actions per level', 'level당 human action', { type: 'range', min: 5, max: 50, step: 1, value: 10 });
    const agent = ctx.control('Agent actions per completed level', '완료 level당 agent action', { type: 'range', min: 1, max: 100, step: 1, value: 20 });
    const completed = ctx.control('Sequential levels completed (of 5)', '순서대로 완료한 level (총 5개)', { type: 'range', min: 0, max: 5, step: 1, value: 2 });
    const out = ctx.metric('Game score', 'game 점수');
    ctx.onChange(() => { const levels = Array.from({ length: 5 }, (_, i) => ({ h: v(human), a: v(agent), done: i < v(completed) })), score = gameScore(levels); out.textContent = `${ctx.format(score * 100, 2)}%`; ctx.view.replaceChildren(); ctx.view.append(ctx.plot({ x: [1, 100], y: [0, 1.2], xLabel: ctx.text('Agent actions', 'agent action'), yLabel: ctx.text('Completed-level score', '완료 level 점수'), series: [{ color: 'blue', points: Array.from({ length: 100 }, (_, i) => [i + 1, rhae(v(human), i + 1, true)]) }], markers: [{ x: v(agent), y: rhae(v(human), v(agent), true), color: 'gold', label: ctx.format(rhae(v(human), v(agent), true), 3) }] })); ctx.view.append(ctx.table([ctx.text('Level', 'level'), ctx.text('Weight', '가중치'), ctx.text('Contribution before game cap', 'game cap 적용 전 기여')], levels.map((l, i) => [i + 1, `${i + 1}/15`, ctx.format((i + 1) * rhae(l.h, l.a, l.done) / 15, 3)]))); ctx.status.textContent = ctx.text('Article scoring convention: min((h/a)²,1.15), later-level weights, then the completed-level cap. This five-level example assigns the same h and a to every completed level; uncompleted levels score zero. No cutoff at five times human actions.', '글에서 확인한 점수식은 min((h/a)²,1.15), 뒤 level에 더 큰 가중치, 완료 level cap의 순서입니다. 이 5-level 예제는 완료 level의 h와 a가 같다고 가정하며 미완료 level은 0점입니다. human action의 5배에서 끊는 규칙은 없습니다.'); });
  });
  reg('arc-token-budget', ctx => {
    const concurrency = ctx.control('Concurrent runs', '동시 run 수', { type: 'range', min: 4, max: 40, step: 1, value: 28 });
    const rate = ctx.control('Assumed aggregate tokens / second', '가정한 전체 token / 초', { type: 'range', min: 100, max: 400, step: 1, value: 204 });
    const tokensPerAction = ctx.control('Tokens per action', 'action당 token', { type: 'range', min: 100, max: 1000, step: 10, value: 600 });
    const out = ctx.metric('Tokens per run / approximate actions', 'run당 token / 대략적인 action');
    ctx.onChange(() => { const r = tokenBudget(110, v(concurrency), 7920, v(rate)); out.textContent = `${Math.round(r.tokens).toLocaleString()} / ${Math.floor(r.tokens / v(tokensPerAction))}`; ctx.view.replaceChildren(); ctx.view.append(ctx.plot({ x: [4, 40], y: [0, v(rate) * 7920 / 4], xLabel: ctx.text('Concurrent runs', '동시 run'), yLabel: ctx.text('Tokens per run', 'run당 token'), series: [{ color: 'blue', points: Array.from({ length: 37 }, (_, i) => [i + 4, tokenBudget(110, i + 4, 7920, v(rate)).tokens]) }], markers: [{ x: v(concurrency), y: r.tokens, color: 'gold', label: `${r.waves} waves` }] })); ctx.status.textContent = ctx.text(`Historical Part 2 preset: 110 runs, 7,920 s per run, concurrency 28, aggregate 204 tokens/s. At this setting four waves use 31,680 s. The selected setting gives ${r.waves} waves and ${(r.wall / 3600).toFixed(2)} h before extra overhead; throughput is held fixed as an assumption, not a hardware scaling prediction.`, `Part 2의 당시 기준은 run 110개, run당 7,920초, 동시 28개, 전체 204 token/초입니다. 기준의 4개 wave는 31,680초를 사용합니다. 현재 선택은 ${r.waves}개 wave, 추가 overhead 전 ${(r.wall / 3600).toFixed(2)}시간입니다. 처리량을 고정한 가정이며 하드웨어 scaling 예측이 아닙니다.`); });
  });
  reg('arc-mismatch', ctx => {
    const mismatch = ctx.control('Observed mismatch after action', 'action 뒤 관측 mismatch', { type: 'select', value: '2', options: [{ value: '0', en: 'None', ko: '없음' }, ...[1, 2, 3, 4].map(i => ({ value: String(i), en: `Action ${i}`, ko: `action ${i}` }))] });
    const step = ctx.control('Inspect executed step', '실행 step 확인', { type: 'range', min: 0, max: 4, step: 1, value: 2 });
    const out = ctx.metric('Executed / dropped actions', '실행 / 버린 action 수');
    ctx.onChange(() => { const plan = [1, 2, 3, 4], actual = plan.slice(); if (v(mismatch)) actual[v(mismatch) - 1] = 0; const r = gatedPlan(plan, actual), at = Math.min(v(step), r.executed.length), predicted = at ? plan[at - 1] : 0, observed = at ? actual[at - 1] : 0; out.textContent = `${r.executed.length} / ${r.dropped}`; ctx.view.replaceChildren(); const boards = ctx.el('div', { class: 'article-lab-grid-pair' }); for (const [label, pos] of [[ctx.text('Predicted frame', '예측 frame'), predicted], [ctx.text('Observed frame', '관측 frame'), observed]]) { const block = ctx.el('div'); block.append(ctx.el('p', {}, label)); const svg = ctx.svg('svg', { viewBox: '0 0 250 65', role: 'img', 'aria-label': label + ': ' + pos }); for (let i = 0; i < 5; i++) svg.append(ctx.svg('rect', { x: i * 48 + 2, y: 5, width: 43, height: 43, fill: i === pos ? '#367bac' : '#dce4e8', stroke: '#6d7e89' })); block.append(svg); boards.append(block); } ctx.view.append(boards); ctx.view.append(ctx.table([ctx.text('Plan step', '계획 step'), ctx.text('Expected cell', '예상 cell'), ctx.text('Gate result', 'gate 결과')], plan.map((p, i) => [i + 1, p, i > r.mismatch && r.mismatch >= 0 ? ctx.text('Dropped', '버림') : i === r.mismatch ? ctx.text('Mismatch: stop', '불일치: 중단') : ctx.text('Match', '일치')]))); ctx.status.textContent = ctx.text(`Illustrative five-cell deterministic model, not an ARC game trace. Inspected step ${at}; the executed mismatch action is retained, and only the unexecuted plan tail is dropped. A mismatch sends control back to observation/model revision.`, `실제 ARC game trace가 아닌 5-cell 결정론적 예제입니다. 확인 중인 step은 ${at}이며 mismatch가 난 action은 이미 실행되었고 아직 실행하지 않은 계획의 뒷부분만 버립니다. 이후 관측과 모형 수정으로 돌아갑니다.`); });
  });
})();
