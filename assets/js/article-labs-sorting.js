/* Local sorting workbenches. Pure, bounded engines are also exported for Node. */
(function () {
  'use strict';
  const MAX_N = 24, MAX_KEY = 99, MAX_TRACE = 12000;
  const ALGORITHMS = Object.freeze(['selection', 'bubble', 'insertion', 'binary-insertion', 'merge', 'quick3', 'heap']);
  const PRESETS = Object.freeze({ random: [7, 2, 8, 3, 2, 6, 1, 5], sorted: [1, 2, 2, 3, 5, 6, 7, 8], reverse: [8, 7, 6, 5, 3, 2, 2, 1], nearly: [1, 2, 3, 5, 4, 6, 7, 8], duplicates: [3, 1, 3, 2, 3, 1, 2, 3] });
  function validValues(values, min = 0, maxN = MAX_N, maxKey = MAX_KEY) {
    if (!Array.isArray(values) || values.length < min || values.length > maxN || !values.every(x => Number.isInteger(x) && x >= 0 && x <= maxKey)) throw new Error(`Expected ${min}–${maxN} integers in 0–${maxKey}.`);
    return values.slice();
  }
  function parseInput(text, min = 2, maxN = MAX_N, maxKey = MAX_KEY) {
    if (typeof text !== 'string' || text.length > 300 || !/^\s*\d+\s*(?:,\s*\d+\s*)*$/.test(text)) throw new Error('Use comma-separated nonnegative integers.');
    return validValues(text.split(',').map(x => Number(x.trim())), min, maxN, maxKey);
  }
  function records(values) {
    validValues(values, 0, MAX_N, 1000000); const seen = new Map(), totals = new Map();
    values.forEach(k => totals.set(k, (totals.get(k) || 0) + 1));
    return values.map((key, id) => { const occurrence = seen.get(key) || 0; seen.set(key, occurrence + 1); return { key, id, label: `${key}${totals.get(key) > 1 ? String.fromCharCode(97 + occurrence) : ''}` }; });
  }
  function isSorted(a) { return a.every((x, i) => !i || a[i - 1].key <= x.key); }
  function isStable(a) { const last = new Map(); for (const r of a) { if (last.has(r.key) && last.get(r.key) >= r.id) return false; last.set(r.key, r.id); } return true; }
  function inversionCount(values) { let count = 0; for (let i = 0; i < values.length; i++) for (let j = i + 1; j < values.length; j++) if (values[i] > values[j]) count++; return count; }
  function seeded(seed) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw new Error('Invalid seed.');
    let state = seed >>> 0;
    return () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; };
  }
  function shuffled(values, rng) { const a = values.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
  function factorial(n) { if (!Number.isInteger(n) || n < 0 || n > 12) throw new Error('Invalid factorial argument.'); let v = 1; for (let k = 2; k <= n; k++) v *= k; return v; }
  function bogoExperiment(n, seed = 17, limit = 2000) {
    if (!Number.isInteger(n) || n < 3 || n > 7 || !Number.isInteger(limit) || limit < 1 || limit > 2000) throw new Error('Invalid random-sort setting.');
    const rng = seeded(seed), original = Array.from({ length: n }, (_, i) => n - i), attempts = [];
    for (let i = 0; i < limit; i++) { const a = shuffled(original, rng); attempts.push(a); if (a.every((v, j) => !j || a[j - 1] <= v)) break; }
    return { original, attempts, success: attempts.at(-1).every((v, j, a) => !j || a[j - 1] <= v), expected: factorial(n), probability: 1 / factorial(n), limit };
  }
  function sortingTrace(values, algorithm = 'insertion') {
    validValues(values); if (!ALGORITHMS.includes(algorithm)) throw new Error('Unknown sorting algorithm.');
    const a = records(values), steps = []; let comparisons = 0, writes = 0, swaps = 0, aux = null, held = null;
    function emit(en, ko, active = [], extra = {}) {
      if (steps.length >= MAX_TRACE) throw new Error('Trace bound exceeded.');
      steps.push({ array: a.slice(), aux: aux ? aux.slice() : null, held, comparisons, writes, swaps, active: active.slice(), en, ko, ...extra });
    }
    function cmp(x, y, active = []) { comparisons++; emit(`Compare ${x.label} and ${y.label}.`, `${x.label}와 ${y.label}의 key를 비교합니다.`, active); return x.key - y.key; }
    function put(i, record) { a[i] = record; writes++; emit(`Write ${record.label} to index ${i}.`, `Index ${i}에 ${record.label}를 씁니다.`, [i]); }
    function swap(i, j) { if (i === j) return; [a[i], a[j]] = [a[j], a[i]]; writes += 2; swaps++; emit(`Swap indices ${i} and ${j}: two array writes.`, `Index ${i}와 ${j}를 바꿉니다. Array slot에 두 번 씁니다.`, [i, j]); }
    emit('Input records. Letter suffixes identify equal-key records.', '입력 record입니다. 같은 key에는 원래 순서를 나타내는 a, b, …를 붙였습니다.');
    const n = a.length;
    if (algorithm === 'selection') {
      for (let i = 0; i + 1 < n; i++) { let best = i; for (let j = i + 1; j < n; j++) if (cmp(a[j], a[best], [j, best]) < 0) best = j; swap(i, best); emit(`Minimum of the remaining suffix is fixed at ${i}.`, `남은 구간의 최솟값을 index ${i}에 놓았습니다.`, [i]); }
    } else if (algorithm === 'bubble') {
      for (let end = n - 1; end > 0; end--) { let changed = false; for (let j = 0; j < end; j++) if (cmp(a[j], a[j + 1], [j, j + 1]) > 0) { swap(j, j + 1); changed = true; } if (!changed) { emit('No swap in this pass: the entire array is sorted.', '이번 pass에서 한 번도 바꾸지 않았으므로 정렬이 끝났습니다.'); break; } }
    } else if (algorithm === 'insertion' || algorithm === 'binary-insertion') {
      for (let i = 1; i < n; i++) {
        const key = a[i]; let position = i; held = key;
        emit(`Hold ${key.label}; the prefix [0, ${i}) is sorted.`, `${key.label}를 임시로 보관하고, 정렬된 prefix [0, ${i})에서 자리를 찾습니다.`, [i], { held: key });
        if (algorithm === 'binary-insertion') {
          let lo = 0, hi = i;
          while (lo < hi) { const mid = (lo + hi) >> 1; if (cmp(a[mid], key, [mid, i]) <= 0) lo = mid + 1; else hi = mid; }
          position = lo;
          emit(`Upper bound is ${position}: insert after existing equal keys.`, `Upper bound는 ${position}입니다. 같은 key가 있으면 그 뒤에 넣습니다.`, [position], { held: key });
          for (let j = i; j > position; j--) put(j, a[j - 1]);
        } else {
          let j = i - 1;
          while (j >= 0 && cmp(a[j], key, [j, i]) > 0) { put(j + 1, a[j]); j--; }
          position = j + 1;
        }
        put(position, key); held = null;
        emit(`Prefix [0, ${i + 1}) is sorted.`, `Prefix [0, ${i + 1})의 정렬이 끝났습니다.`);
      }
    } else if (algorithm === 'merge') {
      const buffer = Array(n).fill(null);
      function merge(lo, hi) {
        if (hi - lo < 2) return;
        const mid = (lo + hi) >> 1; merge(lo, mid); merge(mid, hi);
        aux = buffer;
        for (let k = lo; k < hi; k++) { aux[k] = a[k]; writes++; emit(`Copy ${a[k].label} into the auxiliary run.`, `${a[k].label}를 auxiliary array로 복사합니다.`, [k]); }
        let left = lo, right = mid;
        for (let k = lo; k < hi; k++) {
          const takeLeft = right >= hi || left < mid && cmp(aux[left], aux[right], [left, right]) <= 0;
          put(k, takeLeft ? aux[left++] : aux[right++]);
        }
        aux = null; emit(`Merged sorted runs [${lo}, ${mid}) and [${mid}, ${hi}).`, `정렬된 두 run [${lo}, ${mid})와 [${mid}, ${hi})를 합쳤습니다.`);
      }
      merge(0, n);
    } else if (algorithm === 'quick3') {
      function quick(lo, hi) {
        if (lo >= hi) return;
        const pivot = a[lo]; let lt = lo, mid = lo + 1, gt = hi;
        emit(`First-record pivot ${pivot.label}; split into <, = and > regions.`, `첫 record ${pivot.label}를 pivot으로 골라 <, =, > 구간으로 나눕니다.`, [lo], { pivot: pivot.key });
        while (mid <= gt) {
          if (cmp(a[mid], pivot, [mid, lt, gt]) < 0) { swap(lt, mid); lt++; mid++; }
          else if (cmp(a[mid], pivot, [mid, lt, gt]) > 0) { swap(mid, gt); gt--; }
          else mid++;
        }
        emit(`Equal region [${lt}, ${gt}] needs no recursive sort.`, `Pivot과 같은 구간 [${lt}, ${gt}]는 다시 정렬하지 않습니다.`, [], { pivot: pivot.key });
        quick(lo, lt - 1); quick(gt + 1, hi);
      }
      quick(0, n - 1);
    } else if (algorithm === 'heap') {
      function sift(root, size) {
        while (2 * root + 1 < size) {
          let child = 2 * root + 1;
          if (child + 1 < size && cmp(a[child], a[child + 1], [child, child + 1]) < 0) child++;
          if (cmp(a[root], a[child], [root, child]) >= 0) return;
          swap(root, child); root = child;
        }
      }
      for (let root = Math.floor(n / 2) - 1; root >= 0; root--) sift(root, n);
      emit('Bottom-up heap construction is complete; root holds the maximum.', 'Bottom-up heap 구성이 끝났습니다. Root에 최댓값이 있습니다.', n ? [0] : []);
      for (let size = n; size > 1; size--) { swap(0, size - 1); sift(0, size - 1); }
    }
    emit('Sorted. Compare duplicate labels with their input order.', '정렬이 끝났습니다. 같은 key의 a, b, … 순서가 유지됐는지 확인하세요.', [], { done: true });
    return steps;
  }
  function distributionTrace(values, method = 'radix', base = 10) {
    validValues(values, 2, 16); if (!['radix', 'counting'].includes(method) || ![2, 10].includes(base)) throw new Error('Invalid distribution setting.');
    let a = records(values), writes = 0; const steps = [], max = Math.max(...values), radix = method === 'counting' ? max + 1 : base;
    const emit = (en, ko, buckets, extra = {}) => steps.push({ array: a.slice(), buckets: buckets.map(b => b.slice()), writes, en, ko, ...extra });
    emit('Input records; every bucket preserves arrival order.', '입력 record입니다. 각 bucket에서는 들어온 순서를 유지합니다.', []);
    for (let place = 1; place <= Math.max(1, max); place *= base) {
      const buckets = Array.from({ length: radix }, () => []);
      emit(method === 'counting' ? `K = max key + 1 = ${radix}; empty buckets still require space.` : `Read digit floor(key / ${place}) mod ${base}.`, method === 'counting' ? `K = 최대 key + 1 = ${radix}입니다. 빈 bucket도 공간이 필요합니다.` : `Digit = floor(key / ${place}) mod ${base}를 읽습니다.`, buckets, { place, radix });
      for (let i = 0; i < a.length; i++) { const digit = method === 'counting' ? a[i].key : Math.floor(a[i].key / place) % base; buckets[digit].push(a[i]); writes++; emit(`Append ${a[i].label} to bucket ${digit}; never prepend.`, `${a[i].label}를 bucket ${digit}의 맨 뒤에 넣습니다.`, buckets, { active: [i], place, radix }); }
      const output = buckets.flat();
      for (let i = 0; i < output.length; i++) { a[i] = output[i]; writes++; emit(`Read buckets in order: write ${a[i].label} to output ${i}.`, `Bucket 번호 순서로 꺼내 output ${i}에 ${a[i].label}를 씁니다.`, buckets, { active: [i], place, radix }); }
      if (method === 'counting' || place > Math.floor(max / base)) break;
    }
    emit('Sorted. Stable digit passes retain the earlier lower-digit order.', '정렬이 끝났습니다. Stable한 digit pass가 앞서 맞춘 낮은 자릿수 순서를 보존합니다.', [], { done: true });
    return steps;
  }
  const CODE_CASES = Object.freeze([
    { id: 'empty', input: [] }, { id: 'singleton', input: [2] }, { id: 'sorted', input: [0, 0, 1, 1, 2, 2] }, { id: 'reverse', input: [2, 2, 1, 1, 0, 0] },
    { id: 'recheck', input: [1, 2, 0] }, { id: 'all-twos', input: [2, 2, 2, 2] }, { id: 'mixed', input: [2, 0, 2, 1, 1, 0] },
    { id: 'long-cycle', input: [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1] }
  ].map(x => Object.freeze({ ...x, input: Object.freeze(x.input) })));
  const CODE_CASE_PURPOSES = Object.freeze({
    empty: ['Empty unknown interval', '비어 있는 unknown 구간'],
    singleton: ['Classify the final record', '마지막 record까지 분류하기'],
    sorted: ['Verify the partition contract', '정렬돼 있어도 분류는 끝내기'],
    reverse: ['Preserve the three regions', '세 구간의 조건 유지하기'],
    recheck: ['Inspect the incoming value', 'Swap으로 들어온 값 다시 확인하기'],
    'all-twos': ['Shrink high without skipping', '값을 건너뛰지 않고 high 줄이기'],
    mixed: ['Coordinate all three branches', '세 branch의 pointer 갱신 확인하기'],
    'long-cycle': ['Meet the linear-work budgets', 'Linear time의 작업량 제한 지키기']
  });
  function colorsTrace(values, choices = { condition: '<=', zero: 'advance', two: 'stay' }) {
    validValues(values, 0, MAX_N, 2);
    if (!choices || !['<=', '<'].includes(choices.condition) || !['advance', 'stay'].includes(choices.zero) || !['advance', 'stay'].includes(choices.two)) throw new Error('Invalid guided code choices.');
    const a = records(values), steps = []; let low = 0, mid = 0, high = a.length - 1, iterations = 0, writes = 0, limit = false, error = false;
    function emit(en, ko, active = []) { steps.push({ array: a.slice(), low, mid, high, iterations, writes, comparisons: iterations, en, ko, active }); }
    function swap(i, j) { if (i !== j) { [a[i], a[j]] = [a[j], a[i]]; writes += 2; } }
    emit('Unknown region is [mid, high]; classify one record per iteration.', '아직 확인하지 않은 구간은 [mid, high]입니다. 한 번에 record 하나를 분류합니다.');
    while (choices.condition === '<=' ? mid <= high : mid < high) {
      if (iterations >= a.length) { limit = true; break; }
      if (low > mid || mid < 0 || mid >= a.length || high >= a.length) { error = true; break; }
      const key = a[mid].key; iterations++;
      if (key === 0) { swap(low, mid); low++; if (choices.zero === 'advance') mid++; emit(choices.zero === 'advance' ? '0: swap with low; advance low and mid.' : '0: this choice advances low but leaves mid, breaking the classified-region rule.', choices.zero === 'advance' ? '0: low와 바꾸고 low와 mid를 한 칸 옮깁니다.' : '0: low만 옮기고 mid를 그대로 두면, 분류를 마친 구간의 invariant가 깨집니다.', [low - 1]); }
      else if (key === 1) { mid++; emit('1: already in the middle region; advance mid.', '1: 가운데 구간에 둘 값이므로 mid만 옮깁니다.', [mid - 1]); }
      else { swap(mid, high); high--; if (choices.two === 'advance') mid++; emit(choices.two === 'stay' ? '2: swap with high and shrink high; recheck the incoming record.' : '2: this choice also advances mid, skipping the incoming unclassified record.', choices.two === 'stay' ? '2: high와 바꾸고 high를 줄입니다. 새로 들어온 record는 다시 확인해야 합니다.' : '2: 선택한 규칙은 mid까지 옮겨 새로 들어온 record를 확인하지 않고 건너뜁니다.', [mid, high + 1].filter(i => i < a.length)); }
    }
    const done = !limit && !error && mid > high;
    emit(limit ? 'Stopped at the n-iteration safety budget.' : error ? 'Pointer invariant low ≤ mid failed.' : done ? 'Unknown region is empty.' : 'Loop stopped with an unclassified record.', limit ? 'Iteration이 n번에 도달해 안전장치가 실행을 멈췄습니다.' : error ? 'Pointer invariant low ≤ mid가 깨졌습니다.' : done ? 'Unknown 구간이 비었습니다. 모든 record를 분류했습니다.' : '아직 분류하지 않은 record가 있는데 반복을 끝냈습니다.');
    const sorted = isSorted(a), passed = done && sorted && iterations <= a.length && writes <= 2 * a.length;
    let diagnosticEn, diagnosticKo;
    if (limit) { diagnosticEn = 'LIMIT: n iterations used before the unknown interval emptied. Check which pointer makes progress.'; diagnosticKo = 'LIMIT: n번 실행했지만 unknown 구간이 남았습니다. 어떤 pointer가 구간을 줄이는지 확인하세요.'; }
    else if (error) { diagnosticEn = 'INVARIANT: low passed mid. A 0-swap must preserve the classified 1 region.'; diagnosticKo = 'INVARIANT: low가 mid를 넘어섰습니다. 0을 바꾼 뒤에도 분류된 1 구간의 조건이 유지돼야 합니다.'; }
    else if (!done) { diagnosticEn = `FAIL: mid = ${mid}, high = ${high}; the final unknown record was never classified.`; diagnosticKo = `FAIL: mid = ${mid}, high = ${high}입니다. 마지막 unknown record를 분류하지 않고 끝났습니다.`; }
    else if (!sorted) { diagnosticEn = 'FAIL: pointers crossed, but the output is unsorted. A 2-swap skipped an incoming unknown value.'; diagnosticKo = 'FAIL: pointer는 교차했지만 결과가 정렬되지 않았습니다. 2를 바꾼 뒤 들어온 unknown 값을 건너뛰었습니다.'; }
    else { diagnosticEn = 'PASS: sorted output, empty unknown interval, and both work budgets met.'; diagnosticKo = 'PASS: 결과가 정렬됐고 unknown 구간이 비었으며 iteration과 쓰기 횟수 제한도 지켰습니다.'; }
    return { array: a, steps, iterations, writes, limit, error, done, sorted, low, mid, high, diagnosticEn, diagnosticKo, passed };
  }
  function judgeColors(choices) { return CODE_CASES.map(c => ({ id: c.id, input: c.input.slice(), purposeEn: CODE_CASE_PURPOSES[c.id][0], purposeKo: CODE_CASE_PURPOSES[c.id][1], expected: c.input.slice().sort((a, b) => a - b), ...colorsTrace(c.input, choices) })); }
  const QUIZ = Object.freeze([
    { en: 'Stability: keys versus record identity', ko: 'Stability: key와 record 순서 구분하기', skillEn: 'Contract', skillKo: '정렬 조건', input: [2, 1, 2], labels: ['2a', '1a', '2b'], promptEn: 'Sort [2a, 1a, 2b] by the numeric key in ascending order. The letters identify different records, not a second sort key. Which result is both sorted and stable?', promptKo: '[2a, 1a, 2b]를 숫자 key의 오름차순으로 정렬합니다. a, b는 record를 구별하는 이름일 뿐, 추가 정렬 기준은 아닙니다. 정렬 조건과 stability를 모두 만족하는 결과는 무엇인가요?', options: [['1a, 2a, 2b', '1a, 2a, 2b'], ['1a, 2b, 2a', '1a, 2b, 2a'], ['2a, 1a, 2b', '2a, 1a, 2b']], answer: 0, hintEn: 'Check two conditions separately: are the numeric keys sorted, and did equal-key records keep their input order?', hintKo: '숫자가 오름차순인지, 같은 key끼리는 입력 순서를 지켰는지 따로 확인하세요.', explainEn: 'Stability constrains only equal keys. Since 2a preceded 2b, a stable sorted output must retain that order while moving key 1 first.', explainKo: 'Stability는 같은 key끼리의 순서에 관한 조건입니다. 1을 맨 앞으로 옮긴 뒤에도, 2a가 2b보다 앞에 있어야 합니다.', optionReasons: [['Both requirements hold: 1 < 2, and 2a remains before 2b.', '1 < 2이므로 정렬돼 있고, 2a가 2b보다 앞에 있어 원래 순서도 지켰습니다.'], ['The keys are sorted, but reversing 2a and 2b violates stability.', '숫자는 정렬됐지만 2a와 2b의 순서가 바뀌어 stability를 만족하지 않습니다.'], ['The equal-key order survives, but 2 before 1 violates the sorting condition.', '같은 key의 순서는 지켰지만 2가 1보다 앞에 있으므로 정렬 결과가 아닙니다.']] },
    { en: 'Inversions: a lower bound on adjacent swaps', ko: 'Inversion: adjacent swap은 최소 몇 번일까?', skillEn: 'Work lower bound', skillKo: '최소 작업량', input: [4, 3, 2, 1], promptEn: 'Only neighboring positions may be swapped; swapping distant positions is not allowed. What is the minimum number of adjacent swaps needed to sort [4, 3, 2, 1]?', promptKo: '서로 붙어 있는 두 원소만 바꿀 수 있습니다. 멀리 떨어진 원소끼리 바로 바꿀 수는 없습니다. [4, 3, 2, 1]을 정렬하는 데 필요한 adjacent swap의 최소 횟수는 얼마인가요?', options: [['3', '3'], ['6', '6'], ['4 log₂4 = 8', '4 log₂4 = 8']], answer: 1, hintEn: 'Count reversed pairs, including non-neighboring pairs. How much can one adjacent swap reduce that count?', hintKo: '현재 붙어 있지 않은 pair도 포함해 순서가 뒤집힌 pair를 세어 보세요. Adjacent swap 한 번으로 이 수를 얼마나 줄일 수 있을까요?', explainEn: 'The six inversions are (4,3), (4,2), (4,1), (3,2), (3,1), and (2,1). An adjacent inverted swap removes exactly one; repeatedly doing so reaches zero in six swaps.', explainKo: 'Inversion은 (4,3), (4,2), (4,1), (3,2), (3,1), (2,1)의 여섯 개입니다. 붙어 있는 두 값을 올바른 순서로 바꾸면 정확히 하나가 줄어듭니다. 이를 반복하면 여섯 번 만에 정렬할 수 있으므로 최소 횟수도 여섯 번입니다.', optionReasons: [['Three neighboring boundaries are not the same as all reversed pairs. This input has six inversions.', '현재 이웃한 세 쌍만 세면 안 됩니다. 이 입력에는 순서가 뒤집힌 pair가 여섯 개 있습니다.'], ['Six is both necessary and attainable, one removed inversion per adjacent swap.', 'Swap 한 번에 inversion을 하나씩만 없앨 수 있으므로 최소 여섯 번이 필요합니다. 실제로 여섯 번에 정렬할 수도 있습니다.'], ['n log₂n is an asymptotic comparison scale, not this input’s exact adjacent-swap count.', 'n log₂n은 입력이 커질 때 비교 횟수의 증가율을 분석하는 식이지, 이 배열의 adjacent swap 횟수를 구하는 식이 아닙니다.']] },
    { en: 'Three-way quicksort: duplicate keys', ko: '3-way quicksort: 같은 key를 한꺼번에 처리하기', skillEn: 'Partition and recursion', skillKo: 'Partition과 recursion', input: [4, 4, 4, 4], promptEn: 'The displayed keys are all 4; the algorithm must discover this by inspecting them. With pivot 4, what work does 3-way quicksort perform on an all-equal input of length n?', promptKo: '그림의 key는 모두 4입니다. 알고리즘도 각 값을 확인해야 모두 같다는 것을 알 수 있습니다. Pivot이 4일 때, 길이 n이고 key가 모두 같은 입력을 3-way quicksort는 어떻게 처리하나요?', options: [['Remove one pivot per recursive call: Θ(n²)', '호출마다 pivot 하나만 제외하고 recursion: Θ(n²)'], ['One Θ(n) partition; neither outer region needs recursion', 'Partition 한 번으로 종료: Θ(n). 양쪽 구간은 비어 있음'], ['Skip all key inspections: Θ(1)', 'Key를 하나도 확인하지 않고 종료: Θ(1)']], answer: 1, hintEn: 'After partitioning, which of the < pivot, = pivot, and > pivot regions contain records? Which regions are recursive?', hintKo: 'Partition이 끝난 뒤 < pivot, = pivot, > pivot 중 어디에 원소가 남을까요? Recursion이 필요한 구간은 어디인가요?', explainEn: 'One scan places all n records in the equal region. The < and > regions are empty, so the recursive work stops. The linear bound is for this all-equal input family.', explainKo: '한 번 훑으면 n개가 모두 equal 구간에 들어갑니다. < 구간과 > 구간은 비어 있으므로 추가 recursion은 없습니다. 이 Θ(n) 결과는 key가 모두 같은 입력에 대한 분석입니다.', optionReasons: [['That can describe a poorly chosen two-way partition, but a three-way partition removes the entire equal region from recursion.', '잘못 구성한 2-way partition에서는 그럴 수 있지만, 3-way partition은 같은 key 전체를 recursion 대상에서 제외합니다.'], ['The scan costs Θ(n); both recursive outer regions have length zero.', '전체를 확인하는 데 Θ(n)이 들고, recursion할 양쪽 구간의 길이는 모두 0입니다.'], ['The picture tells you the keys are equal; the algorithm still needs to inspect them.', '그림에서는 모두 같아 보여도, 알고리즘은 각 값을 확인해야 합니다.']] },
    { en: 'Counting sort: pay for the key range', ko: 'Counting sort: 원소 수만 보면 안 되는 이유', skillEn: 'Cost-model assumptions', skillKo: 'Cost model의 가정', input: [0, 2, 7, 9, 1000000], promptEn: 'Use a plain counter array indexed directly by keys 0 through 1,000,000, including unused keys. Why is that counting-sort design unattractive for these five records?', promptKo: '0부터 1,000,000까지 key를 그대로 index로 쓰는 counter array를 만듭니다. 입력에 없는 key의 자리도 만듭니다. 원소가 다섯 개뿐인 이 경우, 이 방식의 counting sort가 비효율적인 이유는 무엇인가요?', options: [['It allocates and scans K = 1,000,001 counters for n = 5', 'n은 5인데 K = 1,000,001개의 counter를 만들고 훑어야 함'], ['It must compare all n(n−1)/2 pairs', 'n(n−1)/2개의 모든 pair를 비교해야 함'], ['It cannot represent repeated keys', '같은 key가 여러 번 나오면 표현할 수 없음']], answer: 0, hintEn: 'A direct-address array includes unused indexes. Compare its length K with the number of records n.', hintKo: 'Array에는 사용하지 않는 index도 포함됩니다. Counter 수 K와 원소 수 n을 비교해 보세요.', explainEn: 'For this direct-address design, both the time bound and total auxiliary space are Θ(n + K). K dominates n here; a small n alone does not make counting sort cheap.', explainKo: '이 구현의 시간과 전체 auxiliary space는 Θ(n + K)입니다. 여기서는 K가 n보다 훨씬 큽니다. 원소가 적더라도 key 범위가 크면 그만큼 비용이 듭니다.', optionReasons: [['Even zero-frequency keys occupy counters, so the range costs more than handling the five records.', '빈도 0인 key도 counter 한 칸을 차지합니다. 다섯 원소를 처리하는 일보다 counter array를 만들고 훑는 데 더 많은 작업이 필요합니다.'], ['Counting sort counts by key index; it does not learn the order by comparing every record pair.', 'Counting sort는 key index별로 개수를 셉니다. 모든 record pair를 비교해서 순서를 알아내는 방법이 아닙니다.'], ['Repeated keys simply increment the same counter; duplicates are not the problem.', '같은 key가 또 나오면 해당 counter를 하나 늘리면 됩니다. 중복 key도 처리할 수 있습니다.']] },
    { en: 'Dutch flag: recheck the swapped-in value', ko: 'Dutch flag: swap으로 들어온 값 다시 확인하기', skillEn: 'Loop invariant', skillKo: 'Loop invariant', input: [1, 2, 0], promptEn: 'In [1, 2, 0], low = 0, mid = 1, high = 2: index 0 is a classified 1, and indexes 1–2 are unknown. After swapping the 2 at mid with the 0 at high and decrementing high, what should happen to mid?', promptKo: '[1, 2, 0]에서 low = 0, mid = 1, high = 2입니다. Index 0의 1은 분류를 마쳤고, index 1–2는 unknown 구간입니다. mid 위치의 2와 high 위치의 0을 바꾸고 high를 1 줄였습니다. mid는 어떻게 해야 하나요?', options: [['Advance mid immediately', 'mid를 바로 한 칸 옮김'], ['Keep mid and inspect the incoming 0 next', 'mid는 그대로 두고 들어온 0을 다음에 확인함'], ['Reset low and mid to 0', 'low와 mid를 모두 0으로 되돌림']], answer: 1, hintEn: 'Track where the incoming value came from. Was its source position classified before the swap?', hintKo: '새로 들어온 값이 어디에서 왔는지 보세요. 바꾸기 전에 그 자리는 분류된 구간이었나요?', explainEn: 'The array becomes [1,0,2], with mid = high = 1. The 0 still needs a swap with low. Keeping mid permits that step; the unknown interval then empties.', explainKo: '배열은 [1,0,2]가 되고 mid = high = 1입니다. 이 0을 low의 1과 바꿔야 정렬이 끝납니다. mid를 그대로 두어야 다음 iteration에서 이 0을 처리할 수 있습니다. 그 작업까지 마치면 unknown 구간이 비게 됩니다.', optionReasons: [['Advancing mid makes it exceed high and exits with [1,0,2], an unsorted output.', 'mid를 옮기면 high를 넘어 반복이 끝납니다. 결과가 [1,0,2]라서 정렬되지 않았습니다.'], ['The incoming 0 was unknown, so it must be classified before mid advances.', '들어온 0은 아직 분류하지 않은 값입니다. 이 값을 먼저 처리한 뒤 mid를 옮겨야 합니다.'], ['Restarting discards a valid classified prefix. The invariant already tells you exactly which position to inspect.', '이미 올바르게 분류한 앞부분을 다시 확인할 필요는 없습니다. Invariant에 따라 현재 mid만 확인하면 됩니다.']] }
  ]);
  const api = { MAX_N, MAX_KEY, MAX_TRACE, ALGORITHMS, PRESETS, validValues, parseInput, records, isSorted, isStable, inversionCount, seeded, shuffled, factorial, bogoExperiment, sortingTrace, distributionTrace, CODE_CASES, CODE_CASE_PURPOSES, colorsTrace, judgeColors, QUIZ };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const register = window.ArticleLabs.register;
  const timersAvailable = typeof window.setTimeout === 'function' && typeof window.clearTimeout === 'function';
  function init(ctx) { ctx.root.classList.add('sort-lab', 'al-workshop'); }
  function textInput(ctx, en, ko, value) {
    const node = ctx.el('input', { type: 'text', value, maxlength: '300', spellcheck: 'false', class: 'sort-input', 'aria-label': ctx.text(en, ko) }); node.value = value;
    ctx.ui.append(ctx.el('label', { class: 'al-control sort-full-control' }, [ctx.el('span', {}, ctx.text(en, ko)), node])); return node;
  }
  function clear(node) { node.replaceChildren(); }
  function drawRecords(ctx, node, a, active = [], regions = null) {
    clear(node); const row = ctx.el('div', { class: 'sort-record-row', role: 'list', 'aria-label': ctx.text('Records in array order', '배열에 놓인 순서대로 표시한 record') });
    a.forEach((r, i) => {
      if (!r) { row.append(ctx.el('span', { class: 'sort-record sort-empty', role: 'listitem' }, '·')); return; }
      const region = regions ? i < regions.low ? ' zero' : i < regions.mid ? ' one' : i > regions.high ? ' two' : ' unknown' : '';
      row.append(ctx.el('span', { class: `sort-record${active.includes(i) ? ' active' : ''}${region}`, role: 'listitem', 'aria-label': `${i}: ${r.label}` }, [ctx.el('span', { class: 'sort-record-value' }, r.label), ctx.el('small', {}, i)]));
    }); node.append(row);
  }
  function drawBars(ctx, node, state) {
    clear(node); const a = state.array, max = Math.max(1, ...a.map(r => r.key));
    const chart = ctx.el('div', { class: 'sort-bars', role: 'img', 'aria-label': a.map(r => r.label).join(', ') });
    a.forEach((r, i) => { const bar = ctx.el('div', { class: `sort-bar${state.active && state.active.includes(i) ? ' active' : ''}` });
      bar.append(ctx.el('span', { class: 'sort-bar-label' }, r.label), ctx.el('span', { class: 'sort-bar-fill', style: `height:${16 + 85 * r.key / max}px` }), ctx.el('small', {}, i)); chart.append(bar); });
    node.append(chart);
    if (state.held) node.append(ctx.el('p', { class: 'sort-caption' }, ctx.text(`Held outside the array: ${state.held.label}`, `임시로 보관 중인 record: ${state.held.label}`)));
    if (state.aux) { node.append(ctx.el('p', { class: 'sort-caption' }, ctx.text('Auxiliary array (its writes count too)', 'Auxiliary array · 여기에 쓰는 횟수도 셉니다.'))); const buffer = ctx.el('div'); drawRecords(ctx, buffer, state.aux); node.append(buffer); }
  }
  function traceController(ctx, draw, getDelay = () => 300) {
    let trace = [], cursor = 0, timer = null, runButton;
    function stop() { if (timer !== null && timersAvailable) window.clearTimeout(timer); timer = null; if (runButton) runButton.textContent = ctx.text('Run', '실행'); }
    function paint() { if (trace.length) draw(trace[cursor], cursor, trace.length); }
    function step(delta = 1) { stop(); cursor = Math.max(0, Math.min(trace.length - 1, cursor + delta)); paint(); }
    ctx.button('Back', '이전 단계', () => step(-1)); ctx.button('Step', '한 단계', () => step());
    runButton = ctx.button('Run', '실행', () => {
      if (timer !== null) { stop(); return; }
      if (!timersAvailable) { cursor = trace.length - 1; paint(); return; }
      if (cursor >= trace.length - 1) cursor = 0;
      runButton.textContent = ctx.text('Pause', '일시정지');
      function tick() { if (ctx.root.isConnected === false || cursor >= trace.length - 1) { stop(); return; } cursor++; paint(); if (cursor >= trace.length - 1) stop(); else timer = window.setTimeout(tick, getDelay()); }
      timer = window.setTimeout(tick, getDelay());
    });
    ctx.button('Finish', '끝까지 보기', () => { stop(); cursor = trace.length - 1; paint(); });
    return { stop, step, paint, set(t, at = 0) { stop(); trace = t; cursor = at; paint(); }, cursor: () => cursor, length: () => trace.length };
  }
  function errorText(ctx, message) { ctx.status.textContent = ctx.text(`Input unchanged: ${message} Use 2–24 comma-separated integers from 0 to 99.`, `기존 입력을 유지합니다. 0–99 사이 정수 2–24개를 쉼표로 구분해 입력하세요. ${message}`); }
  register('sorting-playground', ctx => {
    init(ctx);
    const algorithm = ctx.control('Algorithm', 'Algorithm', { type: 'select', value: 'insertion', options: ALGORITHMS.map(value => ({ value, en: value === 'quick3' ? 'Quick3 · first pivot' : value, ko: value === 'quick3' ? 'Quick3 · 첫 원소 pivot' : value })) });
    const preset = ctx.control('Input shape', '입력 형태', { type: 'select', value: 'random', options: [['random', 'Mixed', '섞인 입력'], ['sorted', 'Sorted', '정렬된 입력'], ['reverse', 'Reverse', '역순'], ['nearly', 'Nearly sorted', '거의 정렬됨'], ['duplicates', 'Many duplicates', '같은 key가 많음']].map(([value, en, ko]) => ({ value, en, ko })) });
    const speed = ctx.control('Delay per event (ms)', '한 단계 간격 (ms)', { type: 'range', min: 80, max: 800, step: 40, value: 320 });
    const input = textInput(ctx, 'Edit keys (2–24)', 'Key 직접 입력 (2–24개)', PRESETS.random.join(', '));
    let values = PRESETS.random.slice(), signature = '', loaded = preset.value;
    const comparisons = ctx.metric('Key comparisons', 'Key 비교'), writes = ctx.metric('Record-array writes', 'Record array 쓰기'), position = ctx.metric('Event / total', '단계 / 전체');
    const chart = ctx.el('div'), annotation = ctx.el('p', { class: 'sort-explanation', 'aria-live': 'polite' }), stable = ctx.el('p', { class: 'sort-caption' });
    ctx.view.append(chart, annotation, stable);
    ctx.button('Apply array', '입력 적용', () => { try { const next = parseInput(input.value); values = next; rebuild(); } catch (e) { errorText(ctx, e.message); } });
    const controller = traceController(ctx, (state, index, length) => { drawBars(ctx, chart, state); comparisons.textContent = state.comparisons; writes.textContent = state.writes; position.textContent = `${index} / ${length - 1}`; annotation.textContent = ctx.text(state.en, state.ko); stable.textContent = state.done ? ctx.text(`Equal-key order preserved on this input: ${isStable(state.array) ? 'yes' : 'no'}. A single stable result is not a guarantee for the algorithm.`, `이 입력에서 같은 key의 순서 유지: ${isStable(state.array) ? '예' : '아니요'}. 이 입력에서 순서가 유지됐다고 알고리즘이 항상 stable한 것은 아닙니다.`) : ctx.text('Letters travel with their original records; only the numeric key is compared.', 'a, b, …는 원래 record를 따라 움직입니다. 비교에는 숫자 key만 씁니다.'); }, () => Number(speed.value));
    function rebuild() { controller.set(sortingTrace(values, algorithm.value)); signature = `${algorithm.value}:${values}`; ctx.status.textContent = ctx.text('Choose an input shape, then compare work—not animation speed. Quick3 always uses the first record as pivot here.', '입력 형태를 바꾸어 비교·쓰기 횟수를 살펴보세요. Animation 재생 속도가 아니라 두 연산 횟수를 비교하세요. 여기서 Quick3는 항상 첫 record를 pivot으로 고릅니다.'); }
    ctx.onReset(() => { controller.stop(); values = PRESETS.random.slice(); loaded = 'random'; input.value = values.join(', '); signature = ''; });
    ctx.onChange(() => { if (preset.value !== loaded) { loaded = preset.value; values = PRESETS[loaded].slice(); input.value = values.join(', '); } if (`${algorithm.value}:${values}` !== signature) rebuild(); });
    ctx.stateLink('sorting-playground', { read: () => ({ algorithm: algorithm.value, preset: preset.value, values: values.slice(), cursor: controller.cursor(), speed: Number(speed.value) }), write: s => { if (!s || !ALGORITHMS.includes(s.algorithm) || !Object.hasOwn(PRESETS, s.preset) || !Number.isInteger(s.cursor) || s.cursor < 0 || !Number.isInteger(s.speed) || s.speed < 80 || s.speed > 800 || (s.speed - 80) % 40) return false; let a, t; try { a = validValues(s.values, 2); t = sortingTrace(a, s.algorithm); } catch (_) { return false; } if (s.cursor >= t.length) return false; controller.stop(); algorithm.value = s.algorithm; loaded = preset.value = s.preset; values = a; input.value = a.join(', '); speed.value = s.speed; signature = `${algorithm.value}:${values}`; controller.set(t, s.cursor); return true; } });
  });
  register('sorting-random', ctx => {
    init(ctx); const size = ctx.control('Distinct keys n', '서로 다른 key 개수 n', { type: 'range', min: 3, max: 7, step: 1, value: 5 }), seed = ctx.control('Seed', 'Seed', { type: 'number', min: 0, max: 9999, step: 1, value: 17 });
    const attempts = ctx.metric('Shuffles tried', '시도한 shuffle'), chance = ctx.metric('Success probability per ideal shuffle', '이상적인 shuffle 1회 성공 확률'), expected = ctx.metric('Expected shuffles', 'Shuffle 횟수의 expected value');
    const row = ctx.el('div'), explanation = ctx.el('p', { class: 'sort-explanation' }); ctx.view.append(row, explanation); let signature = '', experiment;
    const controller = traceController(ctx, (state, i) => { drawRecords(ctx, row, records(state.array)); attempts.textContent = i; chance.textContent = `1 / ${experiment.expected}`; expected.textContent = `${experiment.expected} = ${size.value}!`; explanation.textContent = i === 0 ? ctx.text('The reverse input is not sorted. Each next step tries a complete Fisher–Yates shuffle.', '역순 입력에서 시작합니다. 다음 단계마다 Fisher–Yates shuffle을 한 번 시도합니다.') : ctx.text(`Shuffle ${i}: ${state.sorted ? 'sorted!' : 'not sorted yet.'}`, `Shuffle ${i}회: ${state.sorted ? '정렬됐습니다!' : '아직 정렬되지 않았습니다.'}`); ctx.status.textContent = i === experiment.attempts.length ? experiment.success ? ctx.text('This seeded run succeeded. n! is an expectation under independent uniform shuffles, not a promise for any run.', '이 seed에서는 정렬에 성공했습니다. n!은 매번 독립적으로 모든 순서를 같은 확률로 섞는다고 가정했을 때의 expected value입니다. 반드시 n!번 안에 끝난다는 뜻은 아닙니다.') : ctx.text('Stopped after 2,000 attempts. An unfinished bounded demo is not a failed sorting proof; random sort has no finite worst-case shuffle count.', '2,000회에서 멈췄습니다. 시연을 위해 정한 제한에 도달한 것입니다. Bogosort에는 유한한 worst-case shuffle 횟수가 없습니다.') : ctx.text('At most 2,000 shuffles are generated. The seed makes this finite demonstration reproducible.', '최대 2,000회까지만 시도합니다. 같은 seed를 고르면 같은 과정을 다시 볼 수 있습니다.'); }, () => 160);
    function rebuild() { experiment = bogoExperiment(Number(size.value), Number(seed.value)); controller.set([{ array: experiment.original, sorted: false }, ...experiment.attempts.map(a => ({ array: a, sorted: a.every((x, i) => !i || a[i - 1] <= x) }))]); signature = `${size.value}:${seed.value}`; }
    ctx.onReset(() => { controller.stop(); signature = ''; }); ctx.onChange(() => { if (signature !== `${size.value}:${seed.value}`) rebuild(); });
    ctx.stateLink('sorting-random', { read: () => ({ n: Number(size.value), seed: Number(seed.value), cursor: controller.cursor() }), write: s => { if (!s || !Number.isInteger(s.n) || s.n < 3 || s.n > 7 || !Number.isInteger(s.seed) || s.seed < 0 || s.seed > 9999 || !Number.isInteger(s.cursor) || s.cursor < 0) return false; const next = bogoExperiment(s.n, s.seed); if (s.cursor > next.attempts.length) return false; controller.stop(); size.value = s.n; seed.value = s.seed; experiment = next; signature = `${size.value}:${seed.value}`; controller.set([{ array: next.original, sorted: false }, ...next.attempts.map(a => ({ array: a, sorted: a.every((x, i) => !i || a[i - 1] <= x) }))], s.cursor); return true; } });
  });
  register('sorting-inversions', ctx => {
    init(ctx); const preset = ctx.control('Input', '입력', { type: 'select', value: 'reverse', options: ['reverse', 'nearly', 'duplicates'].map(value => ({ value, en: value, ko: value === 'reverse' ? '역순' : value === 'nearly' ? '거의 정렬됨' : '중복 key 포함' })) }), mode = ctx.control('Insertion trace', 'Insertion trace', { type: 'select', value: 'insertion', options: [{ value: 'insertion', en: 'Linear search', ko: 'Linear search로 찾기' }, { value: 'binary-insertion', en: 'Binary search', ko: 'Binary search로 찾기' }] });
    let values = [4, 3, 2, 1], manualMoves = 0, loaded = 'reverse', signature = '', feedback = '';
    const inversions = ctx.metric('Current inversions I', '현재 inversion I'), moves = ctx.metric('Manual adjacent swaps', '직접 실행한 adjacent swap'), work = ctx.metric('Trace comparisons / writes', 'Trace 비교 / 쓰기');
    const manual = ctx.el('div'), swaps = ctx.el('div', { class: 'sort-swap-buttons' }), trace = ctx.el('div'), note = ctx.el('p', { class: 'sort-explanation', 'aria-live': 'polite' });
    ctx.view.append(ctx.el('p', { class: 'sort-caption' }, ctx.text('Manual board: swap neighbors, then watch I change.', '직접 이웃을 바꾸고 inversion 수가 어떻게 달라지는지 확인하세요.')), manual, swaps, ctx.el('p', { class: 'sort-caption' }, ctx.text('Independent insertion trace from your current board', '현재 입력에서 따로 실행하는 insertion trace')), trace, note);
    const controller = traceController(ctx, state => { drawBars(ctx, trace, state); work.textContent = `${state.comparisons} / ${state.writes}`; note.textContent = feedback || ctx.text(state.en, state.ko); }, () => 220);
    function paintManual() { drawRecords(ctx, manual, records(values)); inversions.textContent = inversionCount(values); moves.textContent = manualMoves; swaps.replaceChildren(); for (let i = 0; i + 1 < values.length; i++) { const button = ctx.el('button', { type: 'button', class: 'al-button', 'aria-label': ctx.text(`Swap indices ${i} and ${i + 1}`, `Index ${i}와 ${i + 1} 바꾸기`) }, `${i} ↔ ${i + 1}`); button.addEventListener('click', () => { const before = inversionCount(values); [values[i], values[i + 1]] = [values[i + 1], values[i]]; manualMoves++; const after = inversionCount(values); feedback = after < before ? ctx.text(`I: ${before} → ${after}. Swapping this inverted pair removes exactly one inversion.`, `I: ${before} → ${after}. 거꾸로 놓인 이웃을 바꿔 inversion이 정확히 하나 줄었습니다.`) : after > before ? ctx.text(`I: ${before} → ${after}. Swapping this ordered pair creates exactly one inversion.`, `I: ${before} → ${after}. 순서가 맞는 이웃을 바꿔 inversion이 정확히 하나 늘었습니다.`) : ctx.text(`I: ${before} → ${after}. Equal keys do not change the inversion count.`, `I: ${before} → ${after}. 같은 key끼리 바꾸면 inversion 수는 변하지 않습니다.`); rebuild(); }); swaps.append(button); } }
    function rebuild() { controller.set(sortingTrace(values, mode.value)); signature = `${mode.value}:${values}`; paintManual(); ctx.status.textContent = ctx.text('Binary search reduces position-finding comparisons, not the number of records that must shift. Final placement writes are counted in addition to shifts.', 'Binary search는 넣을 자리를 찾는 비교를 줄입니다. 뒤로 밀어야 하는 record 수까지 줄이지는 못합니다. 쓰기 횟수에는 shift뿐 아니라 마지막 자리에 넣는 연산도 포함합니다.'); }
    ctx.button('Swap one inverted pair', '거꾸로 놓인 이웃 한 쌍 바꾸기', () => { const i = values.findIndex((v, j) => j + 1 < values.length && v > values[j + 1]); if (i < 0) { feedback = ctx.text('No inversions remain.', 'Inversion이 남아 있지 않습니다.'); note.textContent = feedback; return; } [values[i], values[i + 1]] = [values[i + 1], values[i]]; manualMoves++; feedback = ''; rebuild(); });
    ctx.onReset(() => { controller.stop(); values = [4, 3, 2, 1]; loaded = 'reverse'; manualMoves = 0; signature = ''; feedback = ''; });
    ctx.onChange(() => { if (preset.value !== loaded) { loaded = preset.value; values = loaded === 'reverse' ? [4, 3, 2, 1] : loaded === 'nearly' ? [1, 2, 4, 3, 5] : [2, 1, 2, 1]; manualMoves = 0; feedback = ''; } if (signature !== `${mode.value}:${values}`) rebuild(); });
    ctx.checkpointState({ read: () => ({ preset: preset.value, mode: mode.value, values: values.slice(), swaps: manualMoves, cursor: controller.cursor(), feedback }), write: s => { if (!s || !['reverse', 'nearly', 'duplicates'].includes(s.preset) || !['insertion', 'binary-insertion'].includes(s.mode) || !Number.isInteger(s.swaps) || s.swaps < 0 || s.swaps > 100000 || !Number.isInteger(s.cursor) || s.cursor < 0 || typeof s.feedback !== 'string' || s.feedback.length > 500) return false; let a, t; try { a = validValues(s.values, 2, 8); t = sortingTrace(a, s.mode); } catch (_) { return false; } if (s.cursor >= t.length) return false; controller.stop(); preset.value = loaded = s.preset; mode.value = s.mode; values = a; manualMoves = s.swaps; feedback = s.feedback; signature = `${mode.value}:${values}`; paintManual(); controller.set(t, s.cursor); return true; } });
  });
  register('sorting-merge', ctx => {
    init(ctx); const original = records([2, 4, 6, 2, 3, 6]), left = original.slice(0, 3), right = original.slice(3); let picks = [], feedback = '';
    const remaining = ctx.metric('Left / right remaining', '왼쪽 / 오른쪽 남은 수'), result = ctx.metric('Sorted / stable so far', 'Sorted / stable 여부');
    const leftNode = ctx.el('div'), rightNode = ctx.el('div'), outputNode = ctx.el('div'), feedbackNode = ctx.el('p', { class: 'sort-explanation', 'aria-live': 'polite' });
    const pickButtons = ctx.el('div', { class: 'sort-choice-row' });
    function state() { const li = picks.filter(x => x === 'L').length, ri = picks.length - li; let l = 0, r = 0; const output = picks.map(p => p === 'L' ? left[l++] : right[r++]); return { li, ri, output }; }
    function pick(side) { const s = state(); if (side === 'L' ? s.li >= left.length : s.ri >= right.length) return; const a = left[s.li], b = right[s.ri], chosen = side === 'L' ? a : b; const tie = a && b && a.key === b.key; feedback = tie ? side === 'L' ? ctx.text('Equal keys: choosing left keeps the older record first.', 'Key가 같을 때 왼쪽을 골라야 원래 앞에 있던 record가 먼저 나옵니다.') : ctx.text('Equal keys: choosing right reverses these records, even if numeric keys remain sorted.', 'Key가 같을 때 오른쪽을 고르면 숫자 순서는 맞아도 두 record의 원래 순서가 뒤집힙니다.') : ctx.text(`Append ${chosen.label}. Always compare the two run heads while both exist.`, `${chosen.label}를 output에 넣었습니다. 두 run에 값이 남아 있으면 맨 앞 key끼리 비교합니다.`); picks.push(side); paint(); }
    const lButton = ctx.el('button', { type: 'button', class: 'al-button' }, ctx.text('Take left head', '왼쪽 맨 앞 선택')), rButton = ctx.el('button', { type: 'button', class: 'al-button' }, ctx.text('Take right head', '오른쪽 맨 앞 선택')); lButton.addEventListener('click', () => pick('L')); rButton.addEventListener('click', () => pick('R')); pickButtons.append(lButton, rButton);
    ctx.view.append(ctx.el('p', { class: 'sort-caption' }, 'Left run'), leftNode, ctx.el('p', { class: 'sort-caption' }, 'Right run'), rightNode, pickButtons, ctx.el('p', { class: 'sort-caption' }, 'Output'), outputNode, feedbackNode);
    function paint() { const s = state(); drawRecords(ctx, leftNode, left.slice(s.li), s.li < left.length ? [0] : []); drawRecords(ctx, rightNode, right.slice(s.ri), s.ri < right.length ? [0] : []); drawRecords(ctx, outputNode, s.output); lButton.disabled = s.li >= left.length; rButton.disabled = s.ri >= right.length; remaining.textContent = `${left.length - s.li} / ${right.length - s.ri}`; result.textContent = `${isSorted(s.output) ? '✓' : '✗'} / ${isStable(s.output) ? '✓' : '✗'}`; feedbackNode.textContent = feedback || ctx.text('Two sorted runs are ready. Choose their next output record; try the right-hand 2 first.', '정렬된 두 run이 있습니다. 다음에 내보낼 record를 골라 보세요. 처음에 오른쪽 2를 고르면 어떻게 될까요?'); ctx.status.textContent = picks.length === 6 ? ctx.text(`Finished: sorted=${isSorted(s.output)}, stable=${isStable(s.output)}. Reset to try a different tie rule.`, `완료: sorted=${isSorted(s.output)}, stable=${isStable(s.output)}. Reset한 뒤 key가 같을 때 고르는 쪽을 바꿔 보세요.`) : ctx.text('Stability refers to the original sequence: every left-run record preceded every right-run record.', '원래 입력에서는 왼쪽 run의 모든 record가 오른쪽 run보다 앞에 있었습니다. Stability는 그 원래 순서를 기준으로 판단합니다.'); }
    ctx.button('Undo pick', '선택 되돌리기', () => { picks.pop(); feedback = ''; paint(); });
    ctx.button('Stable next pick', 'Stability를 지키며 다음 값 선택', () => { const s = state(); if (picks.length < 6) pick(s.ri >= right.length || s.li < left.length && left[s.li].key <= right[s.ri].key ? 'L' : 'R'); });
    ctx.button('Complete stable merge', 'Stable merge 끝까지', () => { while (picks.length < 6) { const s = state(); pick(s.ri >= right.length || s.li < left.length && left[s.li].key <= right[s.ri].key ? 'L' : 'R'); } });
    ctx.onReset(() => { picks = []; feedback = ''; }); ctx.onChange(paint);
    ctx.checkpointState({ read: () => ({ picks: picks.slice(), feedback }), write: s => { if (!s || !Array.isArray(s.picks) || s.picks.length > 6 || !s.picks.every(p => p === 'L' || p === 'R') || s.picks.filter(p => p === 'L').length > 3 || s.picks.filter(p => p === 'R').length > 3 || typeof s.feedback !== 'string' || s.feedback.length > 500) return false; picks = s.picks.slice(); feedback = s.feedback; paint(); return true; } });
  });
  register('sorting-partition', ctx => {
    init(ctx); const preset = ctx.control('Input', '입력', { type: 'select', value: 'mixed', options: [{ value: 'mixed', en: 'Mixed colors', ko: '섞인 색' }, { value: 'recheck', en: 'Recheck trap: 1,2,0', ko: '다시 확인할 값: 1,2,0' }, { value: 'twos', en: 'All 2s', ko: '모두 2' }] });
    const input = textInput(ctx, 'Edit 0,1,2 keys (2–24)', '0,1,2 직접 입력 (2–24개)', '2, 0, 2, 1, 1, 0'); let values = [2, 0, 2, 1, 1, 0], loaded = 'mixed', signature = '';
    const pointer = ctx.metric('low / mid / high', 'low / mid / high'), unknown = ctx.metric('Unknown records', '아직 확인하지 않은 수'), iterations = ctx.metric('Iterations / n', 'Iteration / n');
    const board = ctx.el('div'), bands = ctx.el('div', { class: 'sort-regions' }, [ctx.el('span', { class: 'zero' }, '[0, low): 0'), ctx.el('span', { class: 'one' }, '[low, mid): 1'), ctx.el('span', { class: 'unknown' }, '[mid, high]: ?'), ctx.el('span', { class: 'two' }, '(high, n): 2')]), note = ctx.el('p', { class: 'sort-explanation', 'aria-live': 'polite' }); ctx.view.append(bands, board, note);
    ctx.button('Apply colors', '입력 적용', () => { try { values = parseInput(input.value, 2, 24, 2); rebuild(); } catch (_) { ctx.status.textContent = ctx.text('Input unchanged. Enter 2–24 comma-separated keys, each 0, 1 or 2.', '기존 입력을 유지합니다. 0, 1, 2 중에서 2–24개를 쉼표로 구분해 입력하세요.'); } });
    const controller = traceController(ctx, s => { drawRecords(ctx, board, s.array, s.active, s); pointer.textContent = `${s.low} / ${s.mid} / ${s.high}`; unknown.textContent = Math.max(0, s.high - s.mid + 1); iterations.textContent = `${s.iterations} / ${s.array.length}`; note.textContent = ctx.text(s.en, s.ko); });
    function rebuild() { controller.set(colorsTrace(values).steps); signature = `${preset.value}:${values}`; ctx.status.textContent = ctx.text('Each iteration shrinks the unknown region by exactly one. A 2-swap moves an uninspected value to mid: high decreases but mid stays.', 'Iteration마다 unknown 구간이 정확히 하나 줄어듭니다. 2를 high와 바꾸면 아직 확인하지 않은 값이 mid로 오므로 high만 줄이고 mid는 그대로 둡니다.'); }
    ctx.onReset(() => { controller.stop(); values = [2, 0, 2, 1, 1, 0]; loaded = 'mixed'; signature = ''; input.value = values.join(', '); }); ctx.onChange(() => { if (preset.value !== loaded) { loaded = preset.value; values = loaded === 'recheck' ? [1, 2, 0] : loaded === 'twos' ? [2, 2, 2, 2] : [2, 0, 2, 1, 1, 0]; input.value = values.join(', '); } if (signature !== `${preset.value}:${values}`) rebuild(); });
    ctx.checkpointState({ read: () => ({ preset: preset.value, values: values.slice(), cursor: controller.cursor() }), write: s => { if (!s || !['mixed', 'recheck', 'twos'].includes(s.preset) || !Number.isInteger(s.cursor) || s.cursor < 0) return false; let a, t; try { a = validValues(s.values, 2, 24, 2); t = colorsTrace(a).steps; } catch (_) { return false; } if (s.cursor >= t.length) return false; controller.stop(); loaded = preset.value = s.preset; values = a; input.value = a.join(', '); signature = `${preset.value}:${values}`; controller.set(t, s.cursor); return true; } });
  });
  register('sorting-distribution', ctx => {
    init(ctx); const method = ctx.control('Method', '방법', { type: 'select', value: 'radix', options: [{ value: 'radix', en: 'Stable LSD radix', ko: 'Stable LSD radix' }, { value: 'counting', en: 'Stable counting buckets', ko: 'Stable counting bucket' }] }), base = ctx.control('Radix base', 'Radix base', { type: 'select', value: '10', options: [{ value: '10', en: 'Base 10', ko: 'Base 10' }, { value: '2', en: 'Base 2', ko: 'Base 2' }] });
    const input = textInput(ctx, 'Edit keys (2–16; 0–99)', 'Key 직접 입력 (2–16개 · 0–99)', '21, 11, 12, 22, 11'); let values = [21, 11, 12, 22, 11], signature = '';
    const place = ctx.metric('Digit place / bucket count', 'Digit 위치 / bucket 수'), writes = ctx.metric('Record-array writes', 'Record array 쓰기'); const board = ctx.el('div'), buckets = ctx.el('div', { class: 'sort-buckets' }), note = ctx.el('p', { class: 'sort-explanation', 'aria-live': 'polite' }); ctx.view.append(board, buckets, note);
    ctx.button('Apply keys', '입력 적용', () => { try { values = parseInput(input.value, 2, 16); rebuild(); } catch (_) { ctx.status.textContent = ctx.text('Input unchanged. Use 2–16 comma-separated integers in 0–99.', '기존 입력을 유지합니다. 0–99 사이 정수 2–16개를 쉼표로 구분해 입력하세요.'); } });
    const controller = traceController(ctx, s => { drawRecords(ctx, board, s.array, s.active); buckets.replaceChildren(); s.buckets.forEach((bucket, i) => { if (!bucket.length) return; const row = ctx.el('div', { class: 'sort-bucket' }, [ctx.el('strong', {}, `${i}`)]), content = ctx.el('div'); drawRecords(ctx, content, bucket); row.append(content); buckets.append(row); }); place.textContent = `${s.place || '—'} / ${s.radix || '—'}`; writes.textContent = s.writes; note.textContent = ctx.text(s.en, s.ko); });
    function rebuild() { controller.set(distributionTrace(values, method.value, Number(base.value))); signature = `${method.value}:${base.value}:${values}`; ctx.status.textContent = ctx.text('Records enter each bucket left to right and leave in the same order. Empty buckets are omitted from the drawing, not from K or the cost model. Base affects radix only.', '입력을 왼쪽부터 읽어 해당 bucket의 맨 뒤에 넣고, 넣은 순서대로 꺼냅니다. 빈 bucket은 그림에서만 생략하며, K와 비용 계산에서는 빠지지 않습니다. Base는 radix에만 적용됩니다.'); }
    ctx.onReset(() => { controller.stop(); values = [21, 11, 12, 22, 11]; input.value = values.join(', '); signature = ''; }); ctx.onChange(() => { if (signature !== `${method.value}:${base.value}:${values}`) rebuild(); });
    ctx.stateLink('sorting-distribution', { read: () => ({ method: method.value, base: Number(base.value), values: values.slice(), cursor: controller.cursor() }), write: s => { if (!s || !['radix', 'counting'].includes(s.method) || ![2, 10].includes(s.base) || !Number.isInteger(s.cursor) || s.cursor < 0) return false; let a, t; try { a = validValues(s.values, 2, 16); t = distributionTrace(a, s.method, s.base); } catch (_) { return false; } if (s.cursor >= t.length) return false; controller.stop(); method.value = s.method; base.value = s.base; values = a; input.value = a.join(', '); signature = `${method.value}:${base.value}:${values}`; controller.set(t, s.cursor); return true; } });
  });
  register('sorting-quiz', ctx => {
    init(ctx); const select = ctx.control('Question', '문제', { type: 'select', value: '0', options: QUIZ.map((q, i) => ({ value: String(i), en: `${i + 1}. ${q.en}`, ko: `${i + 1}. ${q.ko}` })) }); let answers = Array(5).fill(null), hints = Array(5).fill(false);
    const progress = ctx.metric('Correct / answered', '정답 수 / 푼 문제 수'); const skill = ctx.el('p', { class: 'sort-skill' }), title = ctx.el('div', { class: 'sort-question-title', role: 'heading', 'aria-level': '3' }), diagram = ctx.el('div', { class: 'sort-quiz-diagram' }), prompt = ctx.el('p', { class: 'sort-question-prompt' }), choices = ctx.el('div', { class: 'sort-answers' }), hint = ctx.el('p', { class: 'sort-hint' }), feedback = ctx.el('div', { class: 'sort-feedback', 'aria-live': 'polite' }); ctx.view.append(skill, title, diagram, prompt, choices, hint, feedback);
    function paint() {
      const i = Number(select.value), q = QUIZ[i];
      skill.textContent = `${i + 1} / ${QUIZ.length} · ${ctx.text(q.skillEn, q.skillKo)}`;
      title.textContent = ctx.text(q.en, q.ko);
      drawRecords(ctx, diagram, records(q.input).map((r, j) => ({ ...r, label: q.labels ? q.labels[j] : r.label })), i === 4 ? [1, 2] : []);
      if (i === 4) diagram.append(ctx.el('p', { class: 'sort-caption' }, 'low = 0 · mid = 1 · high = 2'));
      if (i === 3) diagram.append(ctx.el('p', { class: 'sort-caption' }, 'n = 5 · key range 0 … 1,000,000'));
      prompt.textContent = ctx.text(q.promptEn, q.promptKo); choices.replaceChildren();
      q.options.forEach(([en, ko], j) => {
        const button = ctx.el('button', { type: 'button', class: `sort-answer${answers[i] === j ? ' selected' : ''}`, role: 'radio', 'aria-checked': String(answers[i] === j) }, ctx.text(en, ko));
        button.addEventListener('click', () => { answers[i] = j; paint(); }); choices.append(button);
      });
      choices.setAttribute('role', 'radiogroup'); choices.setAttribute('aria-label', ctx.text('Answer choices', '답 선택'));
      hint.textContent = hints[i] ? `${ctx.text('Hint: ', '힌트: ')}${ctx.text(q.hintEn, q.hintKo)}` : '';
      feedback.replaceChildren();
      if (answers[i] !== null) {
        const chosen = q.optionReasons[answers[i]];
        feedback.append(ctx.el('p', {}, `${answers[i] === q.answer ? ctx.text('Correct. ', '맞았습니다. ') : ctx.text('Not quite. ', '다시 생각해 보세요. ')}${ctx.text(chosen[0], chosen[1])}`), ctx.el('p', {}, ctx.text(q.explainEn, q.explainKo)));
      }
      feedback.className = `sort-feedback${answers[i] === null ? '' : answers[i] === q.answer ? ' correct' : ' incorrect'}`;
      progress.textContent = `${answers.filter((v, j) => v === QUIZ[j].answer).length} / ${answers.filter(v => v !== null).length}`;
      ctx.status.textContent = ctx.text('Choose an answer for feedback on that choice and the complete reasoning. Previous/Next keeps your selections; Reset clears all five.', '답을 고르면 선택한 보기가 맞거나 틀린 이유와 전체 해설이 나옵니다. 이전·다음으로 이동해도 선택은 남고, Reset하면 다섯 문제를 모두 초기화합니다.');
    }
    ctx.button('Previous question', '이전 문제', () => { select.value = Math.max(0, Number(select.value) - 1); paint(); }); ctx.button('Next question', '다음 문제', () => { select.value = Math.min(4, Number(select.value) + 1); paint(); }); ctx.button('Hint', '힌트', () => { hints[Number(select.value)] = !hints[Number(select.value)]; paint(); });
    ctx.onReset(() => { answers = Array(5).fill(null); hints = Array(5).fill(false); }); ctx.onChange(paint);
    ctx.checkpointState({ read: () => ({ question: Number(select.value), answers: answers.slice(), hints: hints.slice() }), write: s => { if (!s || !Number.isInteger(s.question) || s.question < 0 || s.question > 4 || !Array.isArray(s.answers) || s.answers.length !== 5 || !s.answers.every((x, i) => x === null || Number.isInteger(x) && x >= 0 && x < QUIZ[i].options.length) || !Array.isArray(s.hints) || s.hints.length !== 5 || !s.hints.every(x => typeof x === 'boolean')) return false; select.value = s.question; answers = s.answers.slice(); hints = s.hints.slice(); paint(); return true; } });
  });
  register('sorting-code', ctx => {
    init(ctx); const condition = ctx.control('Loop while', '반복 조건', { type: 'select', value: '<', options: [{ value: '<', en: 'mid < high', ko: 'mid < high' }, { value: '<=', en: 'mid <= high', ko: 'mid <= high' }] }), zero = ctx.control('After swapping 0 with low', '0을 a[low]와 바꾼 뒤', { type: 'select', value: 'advance', options: [{ value: 'advance', en: 'low += 1; mid += 1', ko: 'low += 1; mid += 1' }, { value: 'stay', en: 'low += 1 only', ko: 'low += 1만 실행' }] }), two = ctx.control('After swapping 2 with high', '2를 a[high]와 바꾼 뒤', { type: 'select', value: 'advance', options: [{ value: 'advance', en: 'high -= 1; mid += 1', ko: 'high -= 1; mid += 1' }, { value: 'stay', en: 'high -= 1 only', ko: 'high -= 1만 실행' }] });
    const result = ctx.metric('Passed / cases', '통과 / case'); const code = ctx.el('pre', { class: 'sort-code' }), diagram = ctx.el('div'), tests = ctx.el('div', { class: 'sort-test-results' }), feedback = ctx.el('p', { class: 'sort-explanation', 'aria-live': 'polite' }); ctx.view.append(code, ctx.el('p', { class: 'sort-caption' }, ctx.text('The recheck case: unknown region [mid, high]', '다시 확인할 값이 들어오는 예제 · unknown 구간 [mid, high]')), diagram, tests, feedback); let results = null, signature = '';
    function settings() { return { condition: condition.value, zero: zero.value, two: two.value }; }
    function paint() {
      const s = settings(); signature = JSON.stringify(s);
      code.textContent = `def sort_colors(a):\n    low = mid = 0\n    high = len(a) - 1\n    while mid ${s.condition} high:\n        if a[mid] == 0:\n            a[low], a[mid] = a[mid], a[low]\n            low += 1${s.zero === 'advance' ? '\n            mid += 1' : ''}\n        elif a[mid] == 1:\n            mid += 1\n        else:\n            a[mid], a[high] = a[high], a[mid]\n            high -= 1${s.two === 'advance' ? '\n            mid += 1' : ''}\n    return a`;
      drawRecords(ctx, diagram, records([1, 2, 0]), [1, 2], { low: 0, mid: 1, high: 2 }); tests.replaceChildren();
      result.textContent = results ? `${results.filter(r => r.passed).length} / 8` : '— / 8';
      if (results) {
        const firstFailed = results.findIndex(r => !r.passed);
        results.forEach((r, index) => {
          const card = ctx.el('details', { class: `sort-test-case ${r.passed ? 'passed' : 'failed'}`, ...(index === firstFailed ? { open: '' } : {}) });
          card.append(ctx.el('summary', {}, `${r.passed ? 'PASS' : r.limit ? 'LIMIT' : r.error ? 'INVARIANT' : 'FAIL'} · ${r.id} — ${ctx.text(r.purposeEn, r.purposeKo)}`),
            ctx.el('p', { class: 'sort-case-arrays' }, `${ctx.text('Input', '입력')}: [${r.input.join(', ')}]\n${ctx.text('Actual', '결과')}: [${r.array.map(x => x.key).join(', ')}]\n${ctx.text('Expected', '정답')}: [${r.expected.join(', ')}]`),
            ctx.el('p', { class: 'sort-caption' }, `${ctx.text('Iterations', 'Iteration')}: ${r.iterations} / ${r.input.length} · ${ctx.text('Writes', '쓰기')}: ${r.writes} / ${2 * r.input.length} · low / mid / high = ${r.low} / ${r.mid} / ${r.high}`),
            ctx.el('p', {}, ctx.text(r.diagnosticEn, r.diagnosticKo)));
          tests.append(card);
        });
        feedback.textContent = results.every(r => r.passed) ? ctx.text('8/8: the unknown interval shrinks by exactly one per iteration. The invariant explains why the rules work for every legal input; eight examples alone are not that proof.', '8/8: iteration마다 unknown 구간이 정확히 하나씩 줄었습니다. 이 규칙이 다른 입력에서도 맞는 이유는 invariant로 설명할 수 있어야 합니다. 여덟 예제를 통과한 것만으로 증명한 것은 아닙니다.') : ctx.text('Open a case to compare the actual and expected arrays. FAIL can mean an unclassified record remains even when the numbers already look sorted; LIMIT and INVARIANT name different failures.', '각 case에서 결과와 정답을 비교해 보세요. 숫자가 정렬돼 보여도 분류하지 않은 record가 남으면 FAIL입니다. LIMIT는 작업량 제한, INVARIANT는 pointer 조건의 위반을 뜻합니다.');
      } else feedback.textContent = ctx.text('Choose the loop boundary and the two pointer updates, then run the eight cases. The displayed code is a compact outline; the bounded browser runner adds safety checks and counters. Download Python to edit the full implementation.', '반복 조건과 두 pointer 갱신 규칙을 고른 뒤 여덟 case를 실행하세요. 화면의 코드는 핵심 동작만 보여 줍니다. 브라우저에서 실행할 때는 안전장치와 counter도 함께 사용합니다. 전체 코드를 직접 고치려면 Python 파일을 내려받으세요.');
      ctx.status.textContent = ctx.text('Contract: mutate and return the original list of keys 0/1/2 in one pass. At most n iterations and 2n record-array writes; self-swaps cost zero. Trace snapshots are visualization copies, not algorithm workspace.', '조건: 0/1/2로 이뤄진 원래 list를 한 pass에 직접 바꾸고 같은 list를 반환합니다. Iteration은 n번 이하, record array 쓰기는 2n번 이하이며 self-swap은 0번으로 셉니다. 시각화를 위해 저장한 snapshot은 알고리즘의 workspace에 포함하지 않습니다.');
    }
    ctx.button('Run eight tests', '여덟 case 실행', () => { results = judgeColors(settings()); paint(); }); ctx.button('Fill reference solution', '정답 규칙 채우기', () => { condition.value = '<='; zero.value = 'advance'; two.value = 'stay'; results = judgeColors(settings()); paint(); }); ctx.onReset(() => { results = null; }); ctx.onChange(() => { if (signature !== JSON.stringify(settings())) results = null; paint(); });
    ctx.checkpointState({ read: () => ({ ...settings(), tested: results !== null }), write: s => { if (!s || !['<', '<='].includes(s.condition) || !['advance', 'stay'].includes(s.zero) || !['advance', 'stay'].includes(s.two) || typeof s.tested !== 'boolean') return false; condition.value = s.condition; zero.value = s.zero; two.value = s.two; results = s.tested ? judgeColors(settings()) : null; paint(); return true; } });
  });
})();
