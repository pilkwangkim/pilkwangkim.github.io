'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const c = require('../assets/js/article-labs-casmi.js');
const records = JSON.parse(fs.readFileSync(path.join(__dirname, '../assets/img/casmi-2026-guide/data/caffeine-massbank.json')));
let checks = 0;
function near(actual, expected, tolerance = 1e-12) { assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`); checks += 1; }
assert.deepEqual(records.map(r => r.peaks.length), [2, 9, 11]); checks += 1;
near(c.cosine(records[2], records[1], 10, true, false).value, 0.9267618308311408);
near(c.cosine(records[2], records[0], 10, true, false).value, 0.39246457921873074);
for (const a of records) for (const b of records) for (let tolerance = 0; tolerance <= 25; tolerance += 0.5) for (const sqrt of [false, true]) for (const remove of [false, true]) {
  const r = c.cosine(a, b, tolerance, sqrt, remove);
  assert.ok(r.value >= 0 && r.value <= 1 + 1e-14);
  assert.equal(new Set(r.matches.map(p => p.i)).size, r.matches.length);
  assert.equal(new Set(r.matches.map(p => p.j)).size, r.matches.length);
  near(r.value, c.cosine(b, a, tolerance, sqrt, remove).value);
  if (a === b) near(r.value, 1);
  if (remove) assert.ok(r.ap.every(p => Math.abs(c.ppm(p.mz, a.precursor_mz)) > 10));
  checks += 4;
}
near(138.0662 - 110.0713, 27.9949); near(195.0878 - 138.0662, 57.0216);
const neutral = c.massWindow(195.0877, 'proton', 5); near(neutral.neutral, 194.0804235334); near(neutral.upper - neutral.neutral, 0.000970402117667, 1e-13);
assert.equal(neutral.candidates.filter(x => x.eligible).length, 1); checks += 1;
assert.equal(c.massWindow(195.0877, 'sodium', 25).candidates.filter(x => x.eligible).length, 0); checks += 1;
const eugenol = c.massWindow(165.091006, 'proton', 5);
assert.deepEqual(eugenol.candidates.filter(x => x.eligible).map(x => x.name), ['Eugenol', 'Isoeugenol']); checks += 1;
const initial = c.rankMetrics([1, 2, 5, 26], [true, true, true, false], false);
near(initial.mrr, 0.425); near(initial.coverage, 0.75); near(initial.hit25, 0.75);
near(c.rankMetrics([1, 2, 5, 26], [true, true, true, false], true).mrr, 0.25);
near(c.rankMetrics([25], [true], false).mrr, 0.04); near(c.rankMetrics([25], [true], true).mrr, 0);
near(c.rankMetrics([1], [false], false).mrr, 0); near(c.rankMetrics([26], [true], false).coverage, 1);
console.log(`CASMI labs: ${checks} checks passed (public spectra, one-to-one matching, cosine, mass windows, candidate coverage and MRR).`);
