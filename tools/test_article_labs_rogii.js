'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const r = require('../assets/js/article-labs-rogii.js');
let checks = 0;
function near(actual, expected, eps = 1e-10) { assert.ok(Math.abs(actual - expected) <= eps, `${actual} != ${expected}`); checks += 1; }
for (let t = -30; t <= 60; t += 0.25) near(r.gamma(t, false), r.gamma(t + 20, false));
const repeat = r.observation('repeat'); assert.equal(repeat.length, 49); checks += 1;
assert.equal(r.observation('gap').filter(row => row.present).length, 28); checks += 1;
for (const mode of ['repeat', 'distinct', 'gap']) for (const tau of [1, 5, 15, 50, 100]) {
  const a = r.alignment(mode, tau);
  near(a.q.reduce((x, y) => x + y), 1);
  assert.ok(a.mean >= -5 && a.mean <= 35); assert.ok(a.q.every(x => Number.isFinite(x) && x >= 0));
  near(a.best.cost, Math.min(...a.costs.map(x => x.cost))); checks += 2;
}
for (let shift = -5; shift <= 15; shift += 0.25) near(r.shiftCost(shift, 'repeat'), r.shiftCost(shift + 20, 'repeat'));
assert.ok(r.shiftCost(5, 'distinct') < r.shiftCost(25, 'distinct')); checks += 1;
assert.ok(r.shiftCost(25, 'gap') - r.shiftCost(5, 'gap') < r.shiftCost(25, 'distinct') - r.shiftCost(5, 'distinct')); checks += 1;
assert.throws(() => r.alignment('repeat', 0)); checks += 1;
for (let p = 0; p <= 100; p += 1) for (let separation = 0; separation <= 40; separation += 1) {
  const probability = p / 100, a = r.hedge(probability, separation);
  near(a.meanRisk, probability * (1 - probability) * separation ** 2);
  near(a.hardRisk, Math.min(probability, 1 - probability) * separation ** 2);
  assert.ok(a.meanRisk <= a.hardRisk + 1e-10);
  near(a.hardRisk - a.meanRisk, Math.min(probability, 1 - probability) ** 2 * separation ** 2);
  // Directly search all possible convex-combination estimates as an independent minimum check.
  for (const fraction of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) assert.ok(r.risk(probability, 0, separation, fraction * separation) >= a.meanRisk - 1e-10);
  checks += 8;
}
assert.deepEqual(r.standings('public').map(x => x.id), ['PS1', 'PS3', 'M3', 'M5', 'M4']); checks += 1;
assert.deepEqual(r.standings('private').map(x => x.id), ['M5', 'M3', 'PS3', 'M4', 'PS1']); checks += 1;
assert.equal(r.rank('PS1', 'public'), 1); assert.equal(r.rank('PS1', 'private'), 5); checks += 2;
const source = fs.readFileSync(path.join(__dirname, '../_posts/2026-3Q/2026-08-06-ROGII-Competition-Retrospective-Public-Private-Shake-Up.md'), 'utf8');
for (const item of r.PORTFOLIO) {
  const row = source.split('\n').find(line => line.startsWith('|') && line.toLowerCase().includes(item.name.toLowerCase()) && line.includes(item.public.toFixed(3)));
  assert.ok(row, `Missing source row for ${item.id}`); assert.ok(row.includes(item.private.toFixed(3))); checks += 2;
}
console.log(`ROGII labs: ${checks} checks passed (periodic/missing GR evidence, normalized shift weights, two-mode squared risk, exact published score pairs).`);
