'use strict';
const assert = require('node:assert/strict');
const m = require('../assets/js/article-labs-farm-game.js');
const apply = (state, action) => { const step = m.transition(state, action); assert.equal(step.ok, true, step.error); return step.state; };
const end = state => apply(state, { type: 'end' });
const plant = (state, plot = 0, crop = 'wheat') => apply(state, { type: 'plant', plot, crop });
function careUntil(state, predicate) {
  while (!predicate(state) && state.day < m.DAYS) {
    if (!m.world(state.seed, state.day).rain) {
      for (let i = 0; i < state.plots.length; i++) if (state.plots[i] && !state.plots[i].watered) state = apply(state, { type: 'water', plot: i });
    }
    state = end(state);
  }
  return state;
}
function immutableFailure(state, action, error) {
  const before = JSON.stringify(state), step = m.transition(state, action);
  assert.equal(step.ok, false); assert.equal(step.error, error); assert.equal(step.state, state); assert.equal(JSON.stringify(state), before);
}
assert.deepEqual(m.world(731, 0), m.world(731, 0));
assert.notDeepEqual(Array.from({ length: 10 }, (_, d) => m.world(731, d)), Array.from({ length: 10 }, (_, d) => m.world(732, d)));
assert.throws(() => m.initial(0)); assert.throws(() => m.world(731, 10));
for (const seed of [1, 143, 731, 991, 99999]) for (let day = 0; day < 10; day++) {
  const world = m.world(seed, day);
  for (const [name, price] of Object.entries(world.prices)) assert.ok(price >= Math.round(m.CROPS[name].price * 0.8) && price <= Math.round(m.CROPS[name].price * 1.2));
}
let state = m.initial();
immutableFailure(state, { type: 'harvest', plot: 0 }, 'unripe');
immutableFailure(state, { type: 'water', plot: 0 }, 'empty');
immutableFailure(state, { type: 'plant', plot: 5, crop: 'wheat' }, 'action');
immutableFailure(state, { type: 'plant', plot: 0, crop: '__proto__' }, 'action');
const original = JSON.stringify(state); state = plant(state);
assert.equal(JSON.stringify(m.initial()), original); assert.equal(state.cash, 50); assert.equal(state.labor, 1);
immutableFailure(state, { type: 'plant', plot: 0, crop: 'wheat' }, 'occupied');
immutableFailure(state, { type: 'water', plot: 0 }, 'watered');
immutableFailure(state, { type: 'harvest', plot: 0 }, 'unripe');
state = careUntil(state, s => s.plots[0].progress >= 2);
const price = m.world(state.seed, state.day).prices.wheat;
state = apply(state, { type: 'harvest', plot: 0 });
assert.equal(state.cash, 50 + price); assert.equal(state.plots[0], null);
immutableFailure(state, { type: 'harvest', plot: 0 }, 'unripe');
assert.equal(state.cash, m.INITIAL_CASH + state.ledger.reduce((sum, row) => sum + row.change, 0));
state = plant(m.initial(), 0, 'tomato');
immutableFailure(state, { type: 'plant', plot: 1, crop: 'carrot' }, 'cash');
state = careUntil(state, s => s.plots[0].progress >= 8);
assert.equal(state.day, 8); state = apply(state, { type: 'harvest', plot: 0 });
immutableFailure(state, { type: 'harvest', plot: 0 }, 'unripe');
state = careUntil(state, s => s.plots[0].progress >= s.plots[0].nextYield);
assert.equal(state.day, 9); state = apply(state, { type: 'harvest', plot: 0 });
assert.equal(state.ledger.filter(e => e.type === 'harvest').length, 2);
state = end(state); assert.equal(state.day, 10); assert.equal(state.labor, 0);
immutableFailure(state, { type: 'end' }, 'finished');
immutableFailure(state, { type: 'harvest', plot: 0 }, 'finished');
let cash = state.cash; assert.equal(m.replay(state.seed, state.ledger.map(m.cleanAction)).cash, cash);
state = m.initial();
for (let hire = 0; hire < 3; hire++) state = apply(state, { type: 'hire' });
assert.equal(state.cash, 56); assert.equal(state.labor, 5); immutableFailure(state, { type: 'hire' }, 'hires');
state = end(state); assert.equal(state.hires, 0); assert.equal(state.labor, 2);
state = plant(plant(m.initial(), 0), 1);
immutableFailure(state, { type: 'plant', plot: 2, crop: 'wheat' }, 'labor');
state = apply(state, { type: 'hire' }); state = plant(state, 2); assert.equal(state.labor, 0);
// Dry-day neglect pauses growth; extra cash cannot create time or a free harvest.
let drySeed = 1;
while (m.world(drySeed, 1).rain) drySeed += 1;
state = end(plant(m.initial(drySeed))); assert.equal(state.plots[0].progress, 1);
state = end(state); assert.equal(state.plots[0].progress, 1);
// Late planting consumes money but gives no salvage at the horizon.
state = m.initial(); for (let day = 0; day < 9; day++) state = end(state);
state = end(plant(state)); assert.equal(state.cash, 50); assert.equal(state.plots[0].progress, 1);
// Prefix replay is an exact undo, then an alternative branch keeps the same exogenous world.
const actions = [{ type: 'plant', plot: 0, crop: 'wheat' }, { type: 'plant', plot: 1, crop: 'wheat' }, { type: 'end' }];
const full = m.replay(731, actions), prefix = m.replay(731, actions, 1);
assert.deepEqual(prefix, plant(m.initial()));
const branch = m.replay(731, actions.slice(0, 1).concat({ type: 'plant', plot: 1, crop: 'carrot' }, { type: 'end' }));
assert.equal(branch.day, full.day); assert.notEqual(branch.cash, full.cash); assert.deepEqual(m.world(branch.seed, branch.day), m.world(full.seed, full.day));
assert.deepEqual(m.replay(731, actions), full); assert.throws(() => m.replay(731, actions, 4));
assert.throws(() => m.replay(731, [{ type: 'harvest', plot: 0 }]));
assert.throws(() => m.replay(731, Array(101).fill({ type: 'end' })));
// Diverse legal paths conserve cash, obey work and wage constraints, and terminate in ten nights.
let transitions = 0;
for (let seed = 1; seed <= 50; seed++) {
  state = m.initial(seed);
  for (let step = 0; step < 100 && state.day < 10; step++) {
    const choices = [{ type: 'hire' }, ...Array.from({ length: 4 }, (_, plot) => [
      { type: 'plant', plot, crop: ['wheat', 'carrot', 'tomato'][(seed + step + plot) % 3] }, { type: 'water', plot }, { type: 'harvest', plot }
    ]).flat(), { type: 'end' }];
    const viable = choices.filter(a => m.transition(state, a).ok);
    const choice = step % 6 === 5 ? { type: 'end' } : viable[(seed * 7 + step * 11) % viable.length];
    const before = JSON.stringify(state), next = apply(state, choice); assert.equal(JSON.stringify(state), before);
    state = next; transitions += 1;
    assert.ok(state.cash >= 0 && state.labor >= 0 && state.labor <= 5 && state.hires <= 3);
    assert.equal(state.cash, 60 + state.ledger.reduce((sum, row) => sum + row.change, 0));
    assert.deepEqual(m.replay(seed, state.ledger.map(m.cleanAction)), state);
  }
  while (state.day < 10) state = end(state);
  assert.equal(m.cashLine(state).at(-1)[0], 10);
}
// Exercise the actual factory controller as well as the pure model. This stub
// checks actions, reset and share-state restore; browser geometry is separate.
const vm = require('node:vm'), fs = require('node:fs');
let factory;
vm.runInNewContext(fs.readFileSync(require.resolve('../assets/js/article-labs-farm-game.js'), 'utf8'), {
  window: { ArticleLabs: { register(id, fn) { assert.equal(id, 'kaggriculture-farm-game'); factory = fn; } } }
});
class Node {
  constructor(attrs = {}, children = []) { this.dataset = {}; this.listeners = {}; this.children = [children].flat(); this.classList = { add() {} }; Object.assign(this, attrs); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  emit(type) { (this.listeners[type] || []).forEach(fn => fn()); }
}
for (const ko of [false, true]) {
  const controls = [], buttons = []; let render, reset, shared;
  const ctx = {
    ko, root: new Node(), ui: new Node(), view: new Node(), metrics: new Node(), status: new Node(), text: (en, kr) => ko ? kr : en,
    el: (_tag, attrs, children) => new Node(attrs, children),
    metric: () => new Node(), table: () => new Node(), plot: spec => { assert(spec.y[1] > spec.y[0]); return new Node(); },
    control: (_en, _kr, options) => { const node = new Node({ ...options, value: String(options.value) }); controls.push(node); return node; },
    button: (_en, _kr, fn) => { buttons.push(fn); return new Node(); },
    onReset: fn => { reset = fn; }, onChange: fn => { render = fn; fn(); }, stateLink: (_key, config) => { shared = config; }
  };
  factory(ctx);
  buttons[0](); buttons[4]();
  if (!m.world(731, 1).rain) buttons[1]();
  buttons[4](); buttons[2]();
  let snapshot = JSON.parse(JSON.stringify(shared.read()));
  assert.ok(m.replay(snapshot.seed, snapshot.actions, snapshot.cursor).cash > 60);
  reset(); render(); assert.equal(shared.read().actions.length, 0);
  shared.write(snapshot); assert.deepEqual(JSON.parse(JSON.stringify(shared.read())), snapshot);
  controls[2].value = '0'; controls[2].emit('input');
  controls[1].value = 'carrot'; buttons[0]();
  snapshot = JSON.parse(JSON.stringify(shared.read()));
  assert.equal(snapshot.actions[0].crop, 'carrot'); assert.ok(snapshot.reference.length > 0);
  const unchanged = JSON.stringify(shared.read());
  assert.throws(() => shared.write({ ...snapshot, actions: [{ type: 'harvest', plot: 0 }], cursor: 1 }));
  assert.equal(JSON.stringify(shared.read()), unchanged);
  buttons[5](); assert.equal(shared.read().cursor, 0);
}
console.log('PASS Farm game: crop timing, no duplicate yields/salvage, cash and labor conservation, invalid-action immutability, finite horizon, deterministic undo/branch; ' + transitions + ' legal transitions across 50 seeds; EN/KO controller action, reset, sharing and branch contracts');
