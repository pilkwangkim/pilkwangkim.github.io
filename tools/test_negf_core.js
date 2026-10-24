#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const core = require('../assets/js/negf-core.js');
const t = core.constants.hoppingEV;
let checks = 0;
function check(name, fn) {
  fn();
  checks += 1;
  console.log('PASS ' + name);
}
function close(actual, expected, tolerance, label) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} differs from ${expected} by ${Math.abs(actual - expected)}`);
}

// Independent, pivoted dense complex Gaussian elimination of A x = e_N.
// It does not use the core's determinant recurrence or its self-energy.
function denseTransmission(energy, potential) {
  const n = potential.length;
  const k = Math.acos(1 - energy / (2 * t));
  const sigma = [-t * Math.cos(k), -t * Math.sin(k)];
  const gamma = 2 * t * Math.sin(k);
  const matrix = Array.from({ length: n }, (_, i) => Array.from({ length: n + 1 }, (_, j) => {
    if (j === n) return [i === n - 1 ? 1 : 0, 0];
    if (i === j) {
      const contacts = Number(i === 0) + Number(i === n - 1);
      return [energy - 2 * t - potential[i] - contacts * sigma[0], -contacts * sigma[1]];
    }
    return [Math.abs(i - j) === 1 ? t : 0, 0];
  }));
  const absoluteSquare = z => z[0] ** 2 + z[1] ** 2;
  const multiply = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  function divide(a, b) {
    const denominator = absoluteSquare(b);
    return [(a[0] * b[0] + a[1] * b[1]) / denominator, (a[1] * b[0] - a[0] * b[1]) / denominator];
  }
  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (absoluteSquare(matrix[row][column]) > absoluteSquare(matrix[pivot][column])) pivot = row;
    }
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    for (let row = column + 1; row < n; row += 1) {
      const factor = divide(matrix[row][column], matrix[column][column]);
      for (let j = column + 1; j <= n; j += 1) {
        const product = multiply(factor, matrix[column][j]);
        matrix[row][j][0] -= product[0];
        matrix[row][j][1] -= product[1];
      }
    }
  }
  const solution = Array.from({ length: n }, () => [0, 0]);
  for (let row = n - 1; row >= 0; row -= 1) {
    const rhs = matrix[row][n].slice();
    for (let j = row + 1; j < n; j += 1) {
      const product = multiply(matrix[row][j], solution[j]);
      rhs[0] -= product[0]; rhs[1] -= product[1];
    }
    solution[row] = divide(rhs, matrix[row][row]);
  }
  return gamma * gamma * absoluteSquare(solution[0]);
}

check('physical constants and exact Example A geometry', () => {
  close(t, 0.5686540464904419, 1e-15, 'hopping');
  const device = core.profile();
  assert.equal(device.sites, 100);
  assert.equal(device.wellStart, 42);
  assert.equal(device.wellEnd, 58);
  assert.deepEqual(device.barriers, [[39, 42], [58, 61]]);
  assert.equal(Array.from(device.potential).filter(v => v === 0.4).length, 6);
  assert.equal(device.xNM[99], 99);
  for (let width = 6; width <= 30; width += 1) {
    const p = core.profile(0.65, width);
    assert.equal(p.wellEnd - p.wellStart, width);
    assert.equal(p.barriers[1][0] - p.barriers[0][1], width);
  }
});

check('retarded lead branch, surface recursion and closed channels', () => {
  for (const energy of [-100, -1, 0, 0.001, 0.1, 2 * t, 4 * t - 0.001, 4 * t, 4 * t + 1, 100]) {
    const s = core.leadSelfEnergy(energy);
    assert.ok(s.imag <= 0);
    close(s.gamma, -2 * s.imag, 0, 'broadening');
    const b = energy - 2 * t;
    close(s.real * s.real - s.imag * s.imag - b * s.real + t * t, 0, 2e-13, 'self-energy quadratic real');
    close(2 * s.real * s.imag - b * s.imag, 0, 1e-14, 'self-energy quadratic imaginary');
    if (energy <= 0 || energy >= 4 * t) {
      assert.equal(s.gamma, 0);
      assert.equal(core.transmission(energy, core.profile()), 0);
      assert.ok(Math.abs(s.real / t) <= 1 + 1e-14, 'evanescent root must decay');
    }
  }
});

check('matched wires transmit one across the entire propagating band', () => {
  for (const length of [1, 2, 3, 17, 100]) {
    for (let i = 0; i < 201; i += 1) {
      const energy = 1e-5 + (4 * t - 2e-5) * i / 200;
      close(core.transmission(energy, new Float64Array(length)), 1, 8e-12, 'matched wire');
    }
  }
});

check('uniform finite barriers agree with the analytic Chebyshev solution', () => {
  for (const height of [0.08, 0.4, 0.65]) {
    for (const length of [1, 2, 3, 7, 20]) {
      for (const energy of [0.001, 0.05, 0.15, height, 0.5, 0.9, 1.7]) {
        // U_(n-1)(cos q), valid for propagating and evanescent q alike.
        const x = (2 * t + height - energy) / (2 * t);
        let earlier = 0, chebyshev = 1;
        for (let n = 1; n < length; n += 1) {
          const next = 2 * x * chebyshev - earlier;
          earlier = chebyshev; chebyshev = next;
        }
        const gammaSquared = energy * (4 * t - energy);
        const analytic = gammaSquared / (gammaSquared + height * height * chebyshev * chebyshev);
        close(core.transmission(energy, Array(length).fill(height)), analytic, 2e-12, 'analytic finite barrier');
      }
    }
  }
});

check('asymmetric potentials agree with an independent dense complex solve and reciprocity', () => {
  const potentials = [
    [0.2],
    [0, 0.12, 0.5, 0, 0.05, 0.3, 0, 0],
    Array.from({ length: 13 }, (_, i) => (i * 17 % 11) * 0.041),
    Array.from(core.profile().potential)
  ];
  for (const potential of potentials) {
    for (const energy of [0.00543183185, 0.016126494083, 0.063869428566, 0.141312760935, 0.333, 0.59, 1.7]) {
      const actual = core.transmission(energy, potential);
      close(actual, denseTransmission(energy, potential), 2e-10, 'dense inverse');
      close(actual, core.transmission(energy, potential.slice().reverse()), 2e-10, 'reciprocity');
      assert.ok(actual >= 0 && actual <= 1 + 2e-10);
    }
  }
});

check('adaptive default spectrum resolves all six real resonances and their sides', () => {
  const result = core.spectrum(core.profile());
  const expected = [0.016126494083, 0.063869428566, 0.141312760935, 0.245260414529, 0.371348237254, 0.514602857252];
  assert.equal(result.peaks.length, expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    const peak = result.peaks[i];
    close(peak.energyEV, expected[i], 1e-11, 'refined resonance');
    close(peak.transmission, 1, 2e-10, 'unit resonance');
    assert.ok(peak.widthEV > 0);
    const index = result.energies.indexOf(peak.energyEV);
    assert.ok(index > 0 && index < result.energies.length - 1);
    assert.ok(result.transmissions[index - 1] < peak.transmission);
    assert.ok(result.transmissions[index + 1] < peak.transmission);
    assert.ok(peak.energyEV - result.energies[index - 1] < peak.widthEV / 4);
    assert.ok(result.energies[index + 1] - peak.energyEV < peak.widthEV / 4);
  }
  for (let i = 1; i < result.energies.length; i += 1) assert.ok(result.energies[i] > result.energies[i - 1]);
  close(result.energies[0], 0.001, 1e-16, 'minimum energy');
  close(result.energies[result.energies.length - 1], 0.6, 1e-15, 'maximum energy');
  assert.ok(result.evaluations < 4000);
});

check('higher barriers narrow the first resonance and wider wells lower it', () => {
  const low = core.spectrum(core.profile(0.2, 16)).peaks[0];
  const high = core.spectrum(core.profile(0.65, 16)).peaks[0];
  const wide = core.spectrum(core.profile(0.65, 30)).peaks[0];
  const narrow = core.spectrum(core.profile(0.65, 6)).peaks[0];
  assert.ok(high.widthEV < low.widthEV);
  assert.ok(wide.energyEV < narrow.energyEV);
  assert.ok(wide.widthEV < 2e-6, 'micronarrow first peak must be resolved');
  close(wide.transmission, 1, 2e-10, 'narrow peak height');
  close(wide.transmission, denseTransmission(wide.energyEV, Array.from(core.profile(0.65, 30).potential)), 2e-10, 'narrow peak dense agreement');
});

check('all 1650 slider configurations preserve unitarity and resolved peaks', () => {
  let configurations = 0;
  for (let h = 0; h <= 65; h += 1) {
    for (let width = 6; width <= 30; width += 1) {
      const result = core.spectrum(core.profile(h / 100, width));
      for (const value of result.transmissions) {
        assert.ok(Number.isFinite(value) && value >= 0 && value <= 1 + 2e-9, `unitarity at height=${h / 100}, width=${width}: ${value}`);
      }
      if (h === 0) assert.equal(result.peaks.length, 0);
      else {
        assert.ok(result.peaks.length > 0);
        for (const peak of result.peaks) close(peak.transmission, 1, 2e-9, 'slider resonance');
      }
      configurations += 1;
    }
  }
  assert.equal(configurations, 1650);
});

check('invalid inputs fail explicitly', () => {
  for (const value of [NaN, Infinity, -0.01, 0.66, '0.4']) assert.throws(() => core.profile(value, 16));
  for (const value of [5, 31, 16.5, '16', NaN]) assert.throws(() => core.profile(0.4, value));
  assert.throws(() => core.transmission(NaN, [0]));
  assert.throws(() => core.transmission(0.1, []));
  assert.throws(() => core.transmission(0.1, [Infinity]));
  assert.throws(() => core.spectrum([0, 1]));
  assert.throws(() => core.spectrum(core.profile(), { minEV: 0 }));
  assert.throws(() => core.spectrum(core.profile(), { minEV: 0.2, maxEV: 0.1 }));
  assert.throws(() => core.spectrum(core.profile(), { baseIntervals: 4 }));
});

console.log(`${checks} NEGF numerical checks passed.`);
