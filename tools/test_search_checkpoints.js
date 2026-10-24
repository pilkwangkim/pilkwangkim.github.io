'use strict';
/* Editor-reload contracts for the real bilingual search factories; no layout claims. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(root, 'assets/data/astar-exercises.json'), 'utf8'));
const factories = new Map(); let checks = 0;
const eq = (a, b, message) => { assert.deepEqual(a, b, message); checks++; };
class Node {
  constructor(tag = 'div', attrs = {}, children = []) {
    this.tagName = tag.toUpperCase(); this.attributes = {}; this.dataset = {}; this.children = []; this.listeners = {}; this._text = ''; this.value = ''; this.checked = false; this.disabled = false; this.isConnected = true; this.tabIndex = -1;
    this.classList = { add: (...classes) => { this.className = [this.className || '', ...classes].join(' ').trim(); } };
    this.style = { setProperty() {} }; Object.entries(attrs || {}).forEach(([k, v]) => this.setAttribute(k, v)); this.append(...(Array.isArray(children) ? children : [children]));
  }
  append(...nodes) { this.children.push(...nodes.filter(n => n !== null && n !== undefined)); }
  replaceChildren(...nodes) { this.children = []; this._text = ''; this.append(...nodes); }
  get textContent() { return this._text + this.children.map(n => n instanceof Node ? n.textContent : String(n)).join(''); }
  set textContent(value) { this.replaceChildren(); this._text = String(value); }
  setAttribute(k, v) { this.attributes[k] = String(v); if (k === 'class') this.className = String(v); else if (k === 'value') this.value = String(v); else if (k.startsWith('data-')) this.dataset[k.slice(5).replace(/-([a-z])/g, (_m, c) => c.toUpperCase())] = String(v); else if (['disabled', 'hidden', 'checked'].includes(k)) this[k] = Boolean(v); else this[k] = String(v); }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  fire(type, payload = {}) { if (!this.disabled) for (const fn of this.listeners[type] || []) fn({ preventDefault() {}, ...payload }); }
  focus() { this.focused = true; }
  all(predicate) { return this.children.flatMap(n => n instanceof Node ? [...(predicate(n) ? [n] : []), ...n.all(predicate)] : []); }
}
const win = { ArticleLabs: { register: (id, fn) => factories.set(id, fn) }, location: { hash: '' }, addEventListener() {} };
for (const bundle of ['pathfinding', 'search-unweighted', 'search-dijkstra', 'search-astar']) vm.runInNewContext(fs.readFileSync(path.join(root, `assets/js/article-labs-${bundle}.js`), 'utf8'), { window: win, console });
async function mount(id, ko) {
  const controls = [], buttons = [], metrics = []; let render, reset, provider;
  const configure = hooks => { assert(!provider, 'One state provider per widget'); provider = hooks; };
  const ctx = {
    root: new Node('section'), ui: new Node(), view: new Node(), status: new Node(), ko,
    text: (en, kr) => ko ? kr : en, el: (tag, attrs, children) => new Node(tag, attrs, children), svg: (tag, attrs, children) => new Node(tag, attrs, children), asset: p => p,
    format: (v, digits = 3) => Number.isFinite(v) ? Number(v.toPrecision(digits)).toString() : '∞',
    metric: () => { const node = new Node('output'); metrics.push(node); return node; },
    control: (_en, _ko, options) => { const node = new Node(options.type === 'select' ? 'select' : 'input', { type: options.type, value: String(options.value) }); node.checked = Boolean(options.value); controls.push(node); ctx.ui.append(node); return node; },
    button: (en, _ko, handler) => { const node = new Node('button'); node.addEventListener('click', handler); buttons.push({ en, node }); ctx.ui.append(node); return node; },
    table: (headers, rows) => new Node('table', {}, JSON.stringify({ headers, rows })), fetchJSON: async () => data,
    onChange: fn => { render = fn; fn(); }, onReset: fn => { reset = fn; }, checkpointState: configure, stateLink: (_id, hooks) => configure(hooks)
  };
  await factories.get(id)(ctx);
  return { id, ko, ctx, controls, metrics, render: () => render(), reset, get provider() { return provider; }, click: name => { const button = buttons.find(b => b.en === name); assert(button, name); button.node.fire('click'); }, nodes: cls => ctx.view.all(n => (n.className || '').split(' ').includes(cls)) };
}
const clone = value => JSON.parse(JSON.stringify(value));
const semantic = lab => ({ metrics: lab.metrics.map(n => n.textContent), view: lab.ctx.view.textContent, status: lab.ctx.status.textContent, controls: lab.controls.map(n => [n.value, n.checked]), cells: lab.ctx.view.all(n => n.role === 'gridcell').map(n => [n.textContent, n.className]) });
async function roundTrip(lab) {
  assert(lab.provider, lab.id + ': checkpoint provider exists'); checks++;
  const saved = clone(lab.provider.read()), before = semantic(lab), fresh = await mount(lab.id, lab.ko);
  // Native field values are also restored by ArticleLabs before the provider.
  fresh.controls.forEach((field, i) => { field.value = lab.controls[i].value; field.checked = lab.controls[i].checked; });
  eq(fresh.provider.write(saved), true, lab.id + ': accepts a serialized checkpoint'); fresh.render();
  eq(clone(fresh.provider.read()), saved, lab.id + ': preserves every state field');
  eq(semantic(fresh), before, lab.id + ': preserves visible output and controls');
  for (const invalid of [null, { ...saved, steps: -1 }, { ...saved, steps: 1000001 }].filter((_, i) => i === 0 || Object.hasOwn(saved, 'steps'))) {
    eq(fresh.provider.write(invalid), false, lab.id + ': rejects invalid checkpoint');
    eq(clone(fresh.provider.read()), saved, lab.id + ': rejected checkpoint is atomic');
  }
  return fresh;
}
async function main() {
  for (const ko of [false, true]) {
    const maze = await mount('search-dfs-maze', ko); maze.click('Step DFS'); maze.click('Step DFS'); await roundTrip(maze);
    maze.click('Show route'); await roundTrip(maze);
    const shown = maze.provider.read().player; maze.nodes('su-maze-board')[0].children[shown.at(-2)].fire('click');
    eq(maze.provider.read().revealed, false, 'Clicking the previous cell resumes manual play after Show route');
    await roundTrip(maze); maze.click('Undo route'); await roundTrip(maze);
    const queens = await mount('search-dfs-queens', ko); queens.nodes('su-queen-board')[0].children[4].fire('click'); await roundTrip(queens);
    queens.click('Step DFS'); queens.click('Step DFS'); await roundTrip(queens);
    queens.click('Find all solutions'); queens.nodes('su-solution-grid')[0].children[1].fire('click'); await roundTrip(queens);
    const words = await mount('search-bfs-words', ko); words.nodes('su-word-options')[0].children.find(n => n.textContent === 'CORD').fire('click'); words.click('Step BFS'); await roundTrip(words);
    words.click('Reveal shortest chain'); await roundTrip(words);
    const jugs = await mount('search-bfs-jugs', ko); jugs.nodes('su-jug-actions')[0].children[1].fire('click'); jugs.nodes('su-jug-actions')[0].children[5].fire('click'); jugs.click('Step BFS'); await roundTrip(jugs);
    jugs.controls[0].value = '2-4-3'; jugs.render(); jugs.click('Reveal shortest plan'); await roundTrip(jugs);
    for (const id of ['search-dijkstra-delivery', 'search-dijkstra-dispatch']) {
      const lab = await mount(id, ko); lab.click('Step: one queue removal'); await roundTrip(lab); lab.click('Finish Dijkstra'); await roundTrip(lab);
      const stop = lab.ctx.view.all(n => n.role === 'button' && (n.attributes['aria-label'] || '').startsWith('D,'))[0];
      stop.fire('click'); if (id.endsWith('delivery')) lab.click('Check my delivery'); await roundTrip(lab);
    }
    const tiles = await mount('search-astar-tiles', ko); tiles.click('Step search'); await roundTrip(tiles); tiles.click('Hint: one move'); await roundTrip(tiles); tiles.click('Play solver solution'); await roundTrip(tiles);
    const detective = await mount('search-astar-detective', ko); detective.ctx.view.all(n => n.tagName === 'BUTTON')[0].fire('click'); await roundTrip(detective); detective.click('Finish'); await roundTrip(detective);
    for (const id of ['astar-traversal', 'astar-weighted']) { const lab = await mount(id, ko); lab.click('Step'); lab.click('Step'); await roundTrip(lab); lab.click('Finish now'); await roundTrip(lab); }
    const blocked = await mount('astar-weighted', ko); blocked.controls[0].value = 'blocked'; blocked.render(); blocked.click('Finish now'); await roundTrip(blocked);
    const priority = await mount('astar-priority-lab', ko); priority.ctx.ui.all(n => n.tagName === 'INPUT')[0].value = 'g + 0.3 * h'; priority.render(); await roundTrip(priority);
    const quiz = await mount('astar-quiz', ko); quiz.ctx.view.all(n => n.role === 'radio')[1].fire('click'); quiz.click('Show / hide hint'); quiz.click('Next exercise'); quiz.ctx.view.all(n => n.role === 'radio')[0].fire('click'); await roundTrip(quiz);
    eq(quiz.provider.write({ ...clone(quiz.provider.read()), answers: [99, -1, -1, -1, -1] }), false, 'Invalid quiz answer rejected');
    const code = await mount('astar-code-test', ko); code.click('Load reference solution'); code.click('Show / hide hint'); await roundTrip(code);
    code.ctx.ui.all(n => n.tagName === 'INPUT')[0].value = 'g'; code.render(); await roundTrip(code); code.click('Run coding tests'); await roundTrip(code);
  }
  console.log(JSON.stringify({ status: 'passed', checks, factories: factories.size, languages: ['en', 'ko'], scope: 'JSON checkpoint across a fresh factory; partial/completed solver, manual route, quiz, hints, draft/results, invalid-state atomicity. Browser save/refresh remains separate.' }, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
