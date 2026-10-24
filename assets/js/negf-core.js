/* Coherent, zero-bias, single-channel transport for the NEGF article.
 * Energies are eV; x is nm. This is the finite-difference lattice model,
 * not a continuum transfer-matrix approximation or a fitted peak model.
 */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.NEGFCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const HBAR = 1.054571817e-34;
  const ELECTRON_MASS = 9.1093837015e-31;
  const JOULES_PER_EV = 1.602176634e-19;
  const HOPPING = HBAR * HBAR / (2 * 0.067 * ELECTRON_MASS * 1e-18) / JOULES_PER_EV;
  const constants = Object.freeze({
    hoppingEV: HOPPING,
    spacingNM: 1,
    effectiveMassRatio: 0.067,
    deviceSites: 100,
    barrierSites: 3,
    defaultHeightEV: 0.4,
    defaultWellSites: 16,
    energyMinEV: 0.001,
    energyMaxEV: 0.6,
    minHeightEV: 0,
    maxHeightEV: 0.65,
    minWellSites: 6,
    maxWellSites: 30
  });

  function finite(value, name) {
    if (!Number.isFinite(value)) throw new TypeError(name + ' must be finite');
    return value;
  }

  function profile(heightEV, wellSites) {
    heightEV = finite(heightEV === undefined ? constants.defaultHeightEV : heightEV, 'Barrier height');
    wellSites = wellSites === undefined ? constants.defaultWellSites : wellSites;
    if (heightEV < constants.minHeightEV || heightEV > constants.maxHeightEV) {
      throw new RangeError('Barrier height must be between 0 and 0.65 eV');
    }
    if (!Number.isInteger(wellSites) || wellSites < constants.minWellSites || wellSites > constants.maxWellSites) {
      throw new RangeError('Well width must be an integer from 6 to 30 sites');
    }
    const sites = constants.deviceSites;
    const wellStart = Math.floor((sites - wellSites) / 2);
    const wellEnd = wellStart + wellSites;
    const barriers = [[wellStart - constants.barrierSites, wellStart], [wellEnd, wellEnd + constants.barrierSites]];
    const potential = new Float64Array(sites);
    for (const [lo, hi] of barriers) potential.fill(heightEV, lo, hi);
    return {
      potential,
      xNM: Float64Array.from({ length: sites }, (_, i) => i * constants.spacingNM),
      heightEV, wellSites, wellStart, wellEnd, barriers, sites
    };
  }

  // The real-axis retarded boundary value is evaluated analytically: no
  // finite eta is added, so artificial absorption cannot wash out a peak.
  function leadSelfEnergy(energyEV) {
    finite(energyEV, 'Energy');
    const b = energyEV - 2 * HOPPING;
    if (energyEV > 0 && energyEV < 4 * HOPPING) {
      const gamma = Math.sqrt(energyEV * (4 * HOPPING - energyEV));
      return { real: b / 2, imag: -gamma / 2, gamma };
    }
    const radical = Math.sqrt(Math.max(0, b * b - 4 * HOPPING * HOPPING));
    // Rationalized small root, continuous from the decaying lead solution.
    const real = 2 * HOPPING * HOPPING / (b + (b < 0 ? -radical : radical));
    return { real, imag: 0, gamma: 0 };
  }

  function potentialOf(device) {
    const potential = device && device.potential ? device.potential : device;
    if (!(Array.isArray(potential) || potential instanceof Float64Array) || potential.length < 1) {
      throw new TypeError('A nonempty potential array is required');
    }
    return potential;
  }

  function transmission(energyEV, device) {
    const potential = potentialOf(device);
    const sigma = leadSelfEnergy(energyEV);
    if (sigma.gamma === 0) return 0;

    // det(A/t): D_j = (A_jj/t) D_{j-1} - D_{j-2};
    // |G_1N|^2 = 1 / (t^2 |D_N|^2). Retain complex determinants
    // throughout instead of multiplying exponentially growing wave states.
    let previousReal = 1;
    let previousImag = 0;
    let earlierReal = 0;
    let earlierImag = 0;
    for (let j = 0; j < potential.length; j += 1) {
      finite(potential[j], 'Potential');
      let diagonalReal = (energyEV - 2 * HOPPING - potential[j]) / HOPPING;
      let diagonalImag = 0;
      if (j === 0) {
        diagonalReal -= sigma.real / HOPPING;
        diagonalImag -= sigma.imag / HOPPING;
      }
      if (j === potential.length - 1) {
        diagonalReal -= sigma.real / HOPPING;
        diagonalImag -= sigma.imag / HOPPING;
      }
      const nextReal = diagonalReal * previousReal - diagonalImag * previousImag - earlierReal;
      const nextImag = diagonalReal * previousImag + diagonalImag * previousReal - earlierImag;
      earlierReal = previousReal;
      earlierImag = previousImag;
      previousReal = nextReal;
      previousImag = nextImag;
    }
    const denominator = previousReal * previousReal + previousImag * previousImag;
    const result = (sigma.gamma / HOPPING) ** 2 / denominator;
    if (!Number.isFinite(result)) throw new RangeError('Transmission exceeded numerical range');
    // No clipping: a violation of unitarity must remain visible to tests.
    return result;
  }

  // For equal barriers B and a clean well W, det(B)=det(W)=1 gives
  // B W B = tr(B W) B - W^-1. The reflection numerator therefore
  // factors into the single-barrier numerator and tr(B W). Locating
  // zeros of this real trace finds narrow unit-transmission resonances
  // even when a uniform energy mesh has no sample near their maxima.
  function cavityCondition(energyEV, device) {
    let a = 1, b = 0, c = 0, d = 1;
    const start = device.barriers[0][0];
    for (let j = start; j < device.wellEnd; j += 1) {
      const diagonal = (2 * HOPPING + device.potential[j] - energyEV) / HOPPING;
      const nextA = diagonal * a - c;
      const nextB = diagonal * b - d;
      c = a; d = b; a = nextA; b = nextB;
    }
    return a + d;
  }

  function bisectZero(fn, left, right) {
    let leftValue = fn(left);
    for (let i = 0; i < 60 && right - left > 2e-14; i += 1) {
      const middle = (left + right) / 2;
      const value = fn(middle);
      if (value === 0) return middle;
      if ((value > 0) === (leftValue > 0)) { left = middle; leftValue = value; }
      else right = middle;
    }
    return (left + right) / 2;
  }

  function spectrum(device, options) {
    options = options || {};
    // Resonance seeding uses the specific symmetric double-barrier
    // geometry returned by profile(); transmission() also accepts arrays.
    if (!device || !Number.isInteger(device.wellSites) || !device.barriers) {
      throw new TypeError('spectrum() requires a profile() device');
    }
    const minEV = options.minEV === undefined ? constants.energyMinEV : finite(options.minEV, 'Minimum energy');
    const maxEV = options.maxEV === undefined ? constants.energyMaxEV : finite(options.maxEV, 'Maximum energy');
    if (!(minEV > 0 && maxEV > minEV && maxEV < 4 * HOPPING)) throw new RangeError('Spectrum must lie inside the lead band');
    const baseIntervals = options.baseIntervals === undefined ? 600 : options.baseIntervals;
    if (!Number.isInteger(baseIntervals) || baseIntervals < 100 || baseIntervals > 10000) throw new RangeError('Invalid spectrum grid size');
    const cache = new Map();
    function sample(energy) {
      if (!cache.has(energy)) cache.set(energy, transmission(energy, device));
      return cache.get(energy);
    }
    for (let i = 0; i <= baseIntervals; i += 1) sample(minEV + (maxEV - minEV) * i / baseIntervals);

    const seeds = [];
    if (device.heightEV > 0) {
      const condition = energy => cavityCondition(energy, device);
      const rootIntervals = Math.max(1200, baseIntervals);
      let previousEnergy = minEV;
      let previousValue = condition(previousEnergy);
      for (let i = 1; i <= rootIntervals; i += 1) {
        const energy = minEV + (maxEV - minEV) * i / rootIntervals;
        const value = condition(energy);
        if (previousValue === 0) seeds.push(previousEnergy);
        else if (value === 0) seeds.push(energy);
        else if ((value > 0) !== (previousValue > 0)) seeds.push(bisectZero(condition, previousEnergy, energy));
        previousEnergy = energy;
        previousValue = value;
      }
      // Three-site single-barrier transparency: U+2t-2t cos(n*pi/3).
      for (const energy of [device.heightEV + HOPPING, device.heightEV + 3 * HOPPING]) {
        if (energy > minEV && energy < maxEV) seeds.push(energy);
      }
    }
    seeds.sort((a, b) => a - b);
    const distinctSeeds = seeds.filter((energy, i) => i === 0 || energy - seeds[i - 1] > 1e-10);
    const peaks = distinctSeeds.map((energyEV, i) => {
      const peakTransmission = sample(energyEV);
      const target = peakTransmission / 2;
      const leftLimit = i === 0 ? minEV : (distinctSeeds[i - 1] + energyEV) / 2;
      const rightLimit = i === distinctSeeds.length - 1 ? maxEV : (energyEV + distinctSeeds[i + 1]) / 2;
      function halfMaximum(direction, limit) {
        let distance = 1e-9;
        let edge = energyEV;
        while (true) {
          edge = direction < 0 ? Math.max(limit, energyEV - distance) : Math.min(limit, energyEV + distance);
          const value = sample(edge);
          if (value <= target) {
            return direction < 0
              ? bisectZero(e => sample(e) - target, edge, energyEV)
              : bisectZero(e => sample(e) - target, energyEV, edge);
          }
          if (edge === limit) return null;
          distance *= 2;
        }
      }
      const leftHalf = halfMaximum(-1, leftLimit);
      const rightHalf = halfMaximum(1, rightLimit);
      const widthEV = leftHalf !== null && rightHalf !== null ? rightHalf - leftHalf : null;
      if (leftHalf !== null || rightHalf !== null) {
        const leftScale = leftHalf === null ? rightHalf - energyEV : energyEV - leftHalf;
        const rightScale = rightHalf === null ? energyEV - leftHalf : rightHalf - energyEV;
        for (const factor of [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]) {
          const left = energyEV - factor * leftScale;
          const right = energyEV + factor * rightScale;
          if (left > leftLimit) sample(left);
          if (right < rightLimit) sample(right);
        }
      }
      return { energyEV, transmission: peakTransmission, widthEV };
    });

    const sorted = Array.from(cache.keys()).sort((a, b) => a - b);
    return {
      energies: Float64Array.from(sorted),
      transmissions: Float64Array.from(sorted, energy => cache.get(energy)),
      peaks,
      evaluations: cache.size,
      minEV,
      maxEV
    };
  }

  return Object.freeze({ constants, profile, leadSelfEnergy, transmission, spectrum });
});
