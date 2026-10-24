'use strict';
const assert = require('node:assert/strict');
const m = require('../assets/js/article-labs-kaggriculture.js');
const close = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) <= tolerance, `${a} differs from ${b}`);
close(m.discounted(0.9, 0), 6.2); close(m.discounted(1, 0), 10); close(m.discounted(0, 0), -10);
close(m.bcStep(0, 0, 20).probabilities[0], 0.1);
close(m.bcStep(0.5, 0, 1).probabilities[0], 0.1887, 1e-4);
for (let target = 0; target < 3; target++) { const result = m.bcStep(0.5, target, 20); close(result.probabilities.reduce((a, b) => a + b), 1); assert.ok(result.probabilities[target] > 0.9); }
close(m.ppo(1.5, 1, 0.2).objective, 1.2); close(m.ppo(0.5, -1, 0.2).objective, -0.8); close(m.ppo(1.5, -1, 0.2).objective, -1.5);
assert.deepEqual(m.wages(12), { total: 376, next: 233 }); assert.deepEqual(m.wages(0), { total: 0, next: 1 });
assert.equal(m.harvest('tomato', 0, 7).units, 0); assert.equal(m.harvest('tomato', 0, 8).units, 1); assert.equal(m.harvest('tomato', 0, 29).units, 22);
assert.deepEqual(m.harvest('tomato', 20, 10), { units: 0, revenue: 0, net: 0 });
assert.equal(m.harvest('strawberry', 0, 11).units, 1); assert.equal(m.harvest('strawberry', 0, 12).units, 2); assert.equal(m.harvest('melon', 0, 29).units, 1);
for (let i = 0; i <= 20; i++) { const w = i / 20, r = m.paired(w); close(r.mean, 0.1 * w); close(r.distribution.reduce((a, b) => a + b.probability, 0), 1); close(r.distribution.reduce((a, b) => a + b.value * b.probability, 0), r.mean); assert.ok(r.low <= r.mean && r.high >= r.mean); }
const plans = m.plans(4000, 0.25, 1); close(plans[0].score, 0); close(plans[1].cost, 1458); close(plans[1].mean, 1991.52); close(plans[1].score, 525.2529721, 1e-6);
assert.equal(plans.slice().sort((a, b) => b.score - a.score)[0].name, 'B'); assert.ok(m.plans(100, 0.25, 1).slice(1).every(p => !p.feasible));
console.log('PASS Kaggriculture: BC gradient, signed PPO clipping, delayed return, crop timing, Fibonacci wages and exact clustered bootstrap');
