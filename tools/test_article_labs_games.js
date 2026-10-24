'use strict';
const assert = require('node:assert/strict');
const lab = require('../assets/js/article-labs-games.js');
const near = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
near(lab.fleetSpeed(1), 1); near(lab.fleetSpeed(1000), 6); near(lab.fleetSpeed(5000), 6);
for (let n = 1; n < 1000; n++) assert.ok(lab.fleetSpeed(n + 1) >= lab.fleetSpeed(n));
near(lab.segmentDistance([5, 5], [0, 0], [10, 0]), 5);
near(lab.segmentDistance([5, 5], [0, 0], [0, 0]), Math.sqrt(50));
assert.equal(lab.routeGeometry([12, 50], [80, 50], 100).safe, false);
assert.equal(lab.routeGeometry([12, 10], [80, 10], 100).safe, true);
const source = [12, 25]; let found = 0;
for (const n of [1, 30, 100, 1000]) for (const phase of [0, 1, 2, 3, 4, 5]) for (const omega of [0, .04, .08]) {
  const hit = lab.intercept(source, n, phase, omega);
  if (hit) { found++; assert.equal(hit.turns, Math.ceil(hit.distance / lab.fleetSpeed(n))); assert.ok(hit.safe); assert.ok(hit.turns <= 80); assert.deepEqual(hit.target, lab.orbitPosition(phase, omega, hit.turns)); }
}
assert.ok(found > 30); assert.ok(lab.intercept(source, 30, 120 * Math.PI / 180, .04));
const original = lab.initialMaze(), transfer = lab.mazeStep(original, { actor: 'scout', type: 'TRANSFER', amount: 99 });
assert.equal(original.turn, 0); assert.equal(original.scout.energy, 8);
assert.equal(transfer.scout.energy, 0); assert.deepEqual(transfer.scout.pos, original.scout.pos);
assert.equal(transfer.factory.energy + transfer.scout.energy, original.factory.energy + original.scout.energy - 2);
assert.equal(lab.mazeStep(transfer, { actor: 'scout', type: 'MOVE_E' }).message, 'energy');
assert.equal(lab.mazeStep(original, { actor: 'factory', type: 'MOVE_N' }).message, 'occupied');
assert.equal(lab.mazeStep(original, { actor: 'scout', type: 'MOVE_N' }).message, 'wall');
let jumped = lab.mazeStep(original, { actor: 'factory', type: 'JUMP_N' });
assert.equal(jumped.jumpCooldown, 3); assert.deepEqual(jumped.factory.pos, [2, 2]);
assert.equal(lab.mazeStep(jumped, { actor: 'factory', type: 'JUMP_N' }).message, 'cooldown');
let idle = original;
for (let i = 0; i < 4; i++) idle = lab.mazeStep(idle, { actor: 'factory', type: 'IDLE' });
assert.ok(idle.lost);
const freed = lab.mazeStep(original, { actor: 'scout', type: 'MOVE_E' });
assert.equal(lab.mazeStep(freed, { actor: 'factory', type: 'MOVE_N' }).message, 'move');
assert.deepEqual(lab.legalTopK([.4, .9, .5], [2, 5, 8], 1, 2, 3), [5, 8]);
assert.deepEqual(lab.legalTopK([.4, .9, .5], [2, 5, 8], 0, 2, 0), []);
assert.throws(() => lab.legalTopK([.4, .9], [2, 2], 1, 2, 1));
for (const weights of [[.5, .2, .3], [0, 0, 0], [1, 0, 0]]) for (const coverage of [0, .8]) for (const marginFloor of [0, .5]) for (const requested of [0, 1, 2, 3]) {
  const r = lab.hybridPolicy([.9, .3, .55], [.45, .8, .6], [.25, .65, .95], weights, coverage, .6, marginFloor, [2, 5, 8], 1, 2, requested);
  near(r.normalized.reduce((s, x) => s + x, 0), 1);
  assert.ok(r.selected.length >= 1 && r.selected.length <= 2); assert.equal(new Set(r.selected).size, r.selected.length);
  assert.ok(r.selected.every(x => [2, 5, 8].includes(x)));
  if (coverage < .6) assert.notEqual(r.reason, 'blend');
}
const factories = {};
global.window = { ArticleLabs: { register: (id, factory) => { factories[id] = factory; } } };
delete require.cache[require.resolve('../assets/js/article-labs-games.js')]; require('../assets/js/article-labs-games.js'); delete global.window;
function node() { return { append() {}, replaceChildren() {}, addEventListener() {}, textContent: '' }; }
let renders = 0, actions = 0;
for (const [id, factory] of Object.entries(factories)) for (const ko of [false, true]) {
  const controls = [], buttons = [], ctx = { root: { dataset: {} }, ko, text: (en, kr) => ko ? kr : en, el: node, svg: node, view: node(), status: node(), metric: node, format: (v, n = 4) => Number(v).toFixed(n), table: () => node(),
    control(en, kr, opts) { const input = { ...node(), value: String(opts.value), checked: Boolean(opts.value), opts }; controls.push(input); return input; }, onChange(fn) { ctx.render = fn; fn(); },
    button(en, kr, fn) { buttons.push(fn); }, onReset(fn) { ctx.reset = fn; },
    plot(spec) { for (const s of spec.series || []) for (const [x, y] of s.points) { assert.ok(Number.isFinite(x) && Number.isFinite(y)); assert.ok(x >= spec.x[0] - 1e-8 && x <= spec.x[1] + 1e-8); assert.ok(y >= spec.y[0] - 1e-8 && y <= spec.y[1] + 1e-8); } return node(); }
  };
  factory(ctx); renders++;
  for (const input of controls) {
    const original = input.value;
    const values = input.opts.type === 'select' ? input.opts.options.map(x => x.value) : [input.opts.min, input.opts.max];
    for (const v of values) { input.value = String(v); ctx.render(); renders++; }
    input.value = original;
  }
  for (const button of buttons) { button(); actions++; }
  if (ctx.reset) { ctx.reset(); ctx.render(); renders++; }
}
console.log(`Games physics, turn-order, cooldown, transfer and legal-menu invariants; ${renders} bilingual renders and ${actions} button actions passed (3 factories).`);
