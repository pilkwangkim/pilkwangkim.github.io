'use strict';
const assert = require('node:assert/strict'), m = require('../assets/js/article-labs-essays.js');
assert.deepEqual(m.assignment(0.2, 1, 0.8, 1.4), { first: 1.5999999999999999, second: 1.8, minorGap: 0.6000000000000001, majorGap: 0.3999999999999999 });
for (let i = 0; i <= 100; i++) for (let j = -20; j <= 20; j++) {
  const w = i / 100, rho = j / 20, v = m.variance(w, rho); assert.ok(v >= -1e-15 && v <= 1 + 1e-15); assert.ok(Math.abs(m.variance(1 - w, rho) - v) < 1e-14);
}
assert.equal(m.variance(0.5, -1), 0); assert.equal(m.variance(0.5, 1), 1); assert.equal(m.variance(0.5, 0), 0.5);
console.log('PASS essay assignment and covariance identities');
