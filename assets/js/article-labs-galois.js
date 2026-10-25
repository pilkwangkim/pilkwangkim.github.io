/* Exact finite Galois examples; numerical bisection is kept separate. */
(function () {
  'use strict';
  const ROOT_LABELS = ['√2', '−√2', '√3', '−√3'];
  const V4_LABELS = ['e', 'σ', 'τ', 'στ'];
  function integer(value, lo, hi) { return Number.isInteger(value) && value >= lo && value <= hi; }
  function checkedPermutation(p, n = p && p.length) {
    if (!Array.isArray(p) || !integer(n, 1, 5) || p.length !== n || new Set(p).size !== n || !p.every(x => integer(x, 0, n - 1))) throw new Error('Expected a permutation of 0,...,n−1, n≤5.');
    return p;
  }
  function identity(n) { if (!integer(n, 1, 5)) throw new Error('Invalid degree.'); return Array.from({ length: n }, (_, i) => i); }
  function compose(p, q) { return q.map(x => p[x]); } // Apply q, then p.
  function inverse(p) { const out = Array(p.length); p.forEach((x, i) => { out[x] = i; }); return out; }
  function commutator(p, q) { return compose(compose(compose(p, q), inverse(p)), inverse(q)); }
  function permutations(n) {
    const out = [], visit = (prefix, rest) => { if (!rest.length) out.push(prefix); else rest.forEach((x, i) => visit([...prefix, x], rest.filter((_, j) => j !== i))); };
    visit([], identity(n)); return out;
  }
  function parity(p) { let count = 0; for (let i = 0; i < p.length; i++) for (let j = i + 1; j < p.length; j++) if (p[i] > p[j]) count++; return count % 2; }
  function groupClosure(generators, n) {
    if (!Array.isArray(generators) || generators.length > 120) throw new Error('Invalid generators.');
    generators.forEach(p => checkedPermutation(p, n));
    const unique = new Map(); [...generators, ...generators.map(inverse)].forEach(p => unique.set(p.join(','), p));
    const start = identity(n), group = [start], seen = new Set([start.join(',')]);
    for (let i = 0; i < group.length; i++) for (const generator of unique.values()) {
      const p = compose(generator, group[i]), key = p.join(',');
      if (!seen.has(key)) { seen.add(key); group.push(p); if (group.length > 120) throw new Error('Finite group bound exceeded.'); }
    }
    return group.sort((a, b) => a.join(',').localeCompare(b.join(',')));
  }
  function derivedSubgroup(group) {
    if (!Array.isArray(group) || !group.length || group.length > 120) throw new Error('Invalid finite group.');
    const generators = new Map();
    for (const p of group) for (const q of group) { const c = commutator(p, q); generators.set(c.join(','), c); }
    return groupClosure([...generators.values()], group[0].length);
  }
  function derivedSeries(group) {
    const series = [group];
    for (let i = 0; i < 6; i++) { if (group.length === 1) break; const next = derivedSubgroup(group); series.push(next); if (next.length === group.length) break; group = next; }
    return series;
  }
  const groupCache = new Map();
  function namedGroup(name) {
    if (!['C5', 'S3', 'S4', 'A5', 'S5'].includes(name)) throw new Error('Unknown group.');
    if (!groupCache.has(name)) {
      let group;
      if (name === 'C5') group = groupClosure([[1, 2, 3, 4, 0]], 5);
      else { const n = Number(name[1]); group = permutations(n); if (name[0] === 'A') group = group.filter(p => !parity(p)); }
      groupCache.set(name, derivedSeries(group));
    }
    return groupCache.get(name);
  }
  function pair(a, b) { if (![a, b].every(x => integer(x, -9, 9))) throw new Error('Integer coefficients −9,...,9 required.'); return [a, b]; }
  function addPair(x, y) { return [x[0] + y[0] || 0, x[1] + y[1] || 0]; }
  function mulPair(x, y) { return [x[0] * y[0] + 2 * x[1] * y[1] || 0, x[0] * y[1] + x[1] * y[0] || 0]; }
  function conjugatePair(x) { return [x[0], -x[1] || 0]; }
  function gcd(a, b) { a = Math.abs(a); b = Math.abs(b); while (b) [a, b] = [b, a % b]; return a; }
  function fraction(a, b) { if (!b) throw new Error('Division by zero.'); const d = gcd(a, b), sign = b < 0 ? -1 : 1; return [sign * a / d, sign * b / d]; }
  function inversePair(x) { const norm = x[0] ** 2 - 2 * x[1] ** 2; return norm ? [fraction(x[0], norm), fraction(-x[1], norm)] : null; }
  function pairLabel(x) { const [a, b] = x; if (!b) return String(a); const term = (Math.abs(b) === 1 ? '' : Math.abs(b)) + '√2'; return a ? `${a} ${b < 0 ? '−' : '+'} ${term}` : `${b < 0 ? '−' : ''}${term}`; }
  function fractionLabel(q) { return q[1] === 1 ? String(q[0]) : `${q[0]}/${q[1]}`; }
  function inverseLabel(q) { return q ? `(${fractionLabel(q[0])}) + (${fractionLabel(q[1])})√2` : 'undefined'; }
  function rootPermutationChecks(p) {
    checkedPermutation(p, 4);
    const square = [2, 2, 3, 3], negative = [1, 0, 3, 2];
    const squares = p.every((image, i) => square[image] === square[i]);
    const negatives = p.every((image, i) => p[negative[i]] === negative[image]);
    return { squares, negatives, valid: squares && negatives, allRemainRoots: true };
  }
  function v4Closure(generators) {
    if (!Array.isArray(generators) || !generators.every(x => integer(x, 0, 3))) throw new Error('Invalid V4 elements.');
    const result = new Set([0]); for (let changed = true; changed;) { changed = false; for (const a of [...result]) for (const b of generators) if (!result.has(a ^ b)) { result.add(a ^ b); changed = true; } }
    return [...result].sort();
  }
  function fixedBasis(subgroup) { return [0, 1, 2, 3].filter(basis => subgroup.every(g => parityBits(g & basis) === 0)); }
  function parityBits(n) { return (n & 1) ^ ((n >> 1) & 1); }
  const QUINTICS = {
    obstruction: { label: 'x⁵ − 5x + 1', f: x => x ** 5 - 5 * x + 1, brackets: [[-3, -1], [-1, 1], [1, 3]], order: 120 },
    radical: { label: 'x⁵ − 2', f: x => x ** 5 - 2, brackets: [[1, 3]], order: 20 }
  };
  function bisection(name, rootIndex, count) {
    if (!(name in QUINTICS) || !integer(rootIndex, 0, QUINTICS[name].brackets.length - 1) || !integer(count, 0, 40)) throw new Error('Invalid bisection settings.');
    const model = QUINTICS[name]; let [lo, hi] = model.brackets[rootIndex]; const initial = [lo, hi];
    for (let i = 0; i < count; i++) { const mid = (lo + hi) / 2, value = model.f(mid); if (!value) { lo = hi = mid; break; } if (model.f(lo) * value < 0) hi = mid; else lo = mid; }
    const middle = (lo + hi) / 2; return { lo, hi, middle, residual: model.f(middle), width: hi - lo, initial, count };
  }
  function formatBisectionBracket(lo, hi) {
    // Seventeen significant digits distinguish binary64 endpoints after 40 steps.
    return `[${lo.toPrecision(17)}, ${hi.toPrecision(17)}]`;
  }
  const QUIZ = [
    { en: 'What counts as a radical formula?', ko: '어떤 표현을 radical 공식이라고 할까?', diagram: ['coefficients', '+ − × ÷', 'm-th roots', 'finite expression'], promptEn: 'For an irrational root of a rational-coefficient polynomial, which step is permitted when building an exact radical expression?', promptKo: '유리계수 polynomial의 무리수인 근을 정확한 radical expression으로 나타내려 합니다. 다음 중 허용되는 단계는 무엇인가요?', options: [['Take a cube root of an earlier expression.', '앞서 만든 표현의 세제곱근을 구한다.'], ['Use “the smallest real root” as a new operation.', '“가장 작은 실근”을 구하는 연산을 새로 추가한다.'], ['Treat a 12-decimal bisection approximation as the exact root.', '소수점 아래 12자리의 bisection 근삿값을 그 근의 정확한 값으로 쓴다.']], answer: 0, hintEn: 'An exact expression must equal the target root, not merely approximate it.', hintKo: '정확한 표현은 목표 근과 같은 값이어야 합니다. 가까운 값을 구하는 것과는 다릅니다.', optionReasons: [['Radicals and field operations are exactly the allowed finite operations.', '사칙연산과 radical 추출은 모두 정의에서 허용한 연산입니다.'], ['An implicit root selector names a root but does not express it using the allowed operations.', '어느 근인지 지정했을 뿐, 허용된 연산으로 그 근을 표현한 것은 아닙니다.'], ['The finite decimal is rational and is itself an allowed constant, but it is not equal to the target irrational root. An approximation is not an exact formula for that root.', '그 유한소수 자체는 유리수이므로 허용되는 상수입니다. 하지만 목표인 무리수 근과 같지는 않습니다. 근삿값을 구했다고 그 근의 정확한 공식을 얻은 것은 아닙니다.']] },
    { en: 'Linear map or field automorphism?', ko: 'Linear map이면 field automorphism일까?', diagram: ['basis: (1, √2)', 'T(a + b√2) = a + 2b√2', 'T(√2)²  ?  T(2)'], promptEn: 'T is Q-linear and invertible. Is it a Q-automorphism of Q(√2)?', promptKo: 'T는 Q-linear이며 invertible한 변환입니다. Q(√2)의 Q-automorphism이기도 할까요?', options: [['Yes: invertible linear maps preserve multiplication.', '그렇다. Invertible linear map은 곱셈도 보존한다.'], ['No: T(√2)² = 8 but T(2) = 2.', '아니다. T(√2)² = 8인데 T(2) = 2다.'], ['No: no nonidentity automorphism is possible.', '아니다. Identity 이외의 automorphism은 존재하지 않는다.']], answer: 1, hintEn: 'Apply T to the relation (√2)² = 2.', hintKo: '(√2)² = 2라는 관계에 T를 적용해 보세요.', optionReasons: [['Linearity preserves addition and rational scaling, not arbitrary products.', 'Linearity는 덧셈과 유리수배를 보존합니다. 곱셈도 보존하는지는 별도로 확인해야 합니다.'], ['The multiplication relation fails, so T is not a field automorphism.', '곱셈 관계를 보존하지 못하므로 field automorphism이 아닙니다.'], ['Conjugation a+b√2 ↦ a−b√2 is a nonidentity automorphism.', 'a+b√2를 a−b√2로 보내는 conjugation은 identity가 아닌 automorphism입니다.']] },
    { en: 'Can two square-root pairs be exchanged?', ko: '두 square-root 쌍을 서로 바꿀 수 있을까?', diagram: ['√2 ↔ √3', '−√2 ↔ −√3', '2 and 3 stay fixed'], promptEn: 'The four roots of (x²−2)(x²−3) are permuted bijectively as shown. Does this extend to a Q-automorphism?', promptKo: '그림처럼 (x²−2)(x²−3)의 네 근을 일대일로 바꾸었습니다. 이 permutation을 Q-automorphism으로 확장할 수 있나요?', options: [['Yes: every root is still a root of the same quartic.', '그렇다. 바꾼 뒤에도 같은 quartic의 근이다.'], ['Yes: the permutation has an inverse.', '그렇다. Inverse permutation이 존재한다.'], ['No: the equation α² = 2 would become 3 = 2.', '아니다. α² = 2라는 관계가 3 = 2로 바뀐다.']], answer: 2, hintEn: 'A Q-automorphism must preserve every rational polynomial relation, not just the product polynomial.', hintKo: '곱으로 만든 다항식 하나만 만족해서는 부족합니다. 유리계수의 모든 polynomial relation을 보존해야 합니다.', optionReasons: [['Remaining in the root set is necessary but not sufficient.', '근들의 집합 안에 남는 것은 필요조건일 뿐, 충분조건은 아닙니다.'], ['An inverse makes this a permutation, but does not ensure that it respects multiplication.', 'Inverse가 있으므로 permutation은 맞습니다. 하지만 곱셈 관계도 보존한다는 뜻은 아닙니다.'], ['Rational constants are fixed, so √2 cannot be sent to an element whose square is 3.', '유리수는 고정해야 하므로 √2를 제곱값이 3인 수로 보낼 수는 없습니다.']] },
    { en: 'Fix more numbers, allow fewer symmetries', ko: '고정해야 할 수가 늘어나면 symmetry는 어떻게 될까?', diagram: ['L = Q(√2, √3)', 'σ: √2 ↦ −√2, √3 ↦ √3', 'E = Q(√3)'], promptEn: 'Which subgroup fixes every element of E=Q(√3)?', promptKo: 'E=Q(√3)의 모든 원소를 그대로 두는 subgroup은 무엇인가요?', options: [['{e, σ}', '{e, σ}'], ['The full four-element group.', '원소 네 개로 이루어진 전체 group'], ['Only e, because √3 is irrational.', '√3가 무리수이므로 {e}만 가능하다.']], answer: 0, hintEn: 'The image of √2 is unrestricted by fixing √3.', hintKo: '√3를 고정해도 √2의 부호까지 고정할 필요는 없습니다.', optionReasons: [['Both e and σ fix √3 and every rational combination of it.', 'e와 σ는 √3과 그 유리계수 표현을 모두 고정합니다.'], ['τ flips √3, so the full group does not fix E pointwise.', 'τ는 √3의 부호를 바꿉니다. 따라서 전체 group은 E를 원소별로 고정하지 않습니다.'], ['Fixing √3 still allows √2 to change sign.', '√3을 그대로 두면서 √2의 부호만 바꿀 수 있습니다.']] },
    { en: 'Nonabelian does not mean nonsolvable', ko: 'Nonabelian이면 nonsolvable일까?', diagram: ['S₃: 6 → 3 → 1', 'S₅: 120 → 60 → 60', 'orders of derived subgroups'], promptEn: 'Which conclusion follows from these derived-series orders?', promptKo: '그림에 나온 derived series를 보고 어떤 결론을 내릴 수 있나요?', options: [['Neither group is solvable because both are nonabelian.', '둘 다 nonabelian이므로 어느 쪽도 solvable하지 않다.'], ['S₃ is solvable; S₅ is not.', 'S₃는 solvable이고 S₅는 그렇지 않다.'], ['S₅ is solvable because its order decreases once.', 'S₅는 order가 한 번 줄었으므로 solvable하다.']], answer: 1, hintEn: 'Solvability requires reaching the identity group, not just shrinking once.', hintKo: '한 번 줄어드는 것으로는 부족합니다. Identity group에 도달하는지 확인하세요.', optionReasons: [['A nonabelian group may still have a derived series ending at the identity.', 'Nonabelian group이라도 derived series가 identity group에서 끝날 수 있습니다.'], ['S₃ reaches order 1. S₅ reaches A₅, whose 60 elements also form its derived subgroup, so the next step gives A₅ again.', 'S₃는 order 1에 도달합니다. S₅에서는 원소가 60개인 A₅까지 줄어들지만, 다음 derived subgroup도 A₅이므로 더는 줄어들지 않습니다.'], ['The remaining A₅ has order 60 and equals its own derived subgroup.', '남은 A₅는 order가 60이며, derived subgroup을 구해도 다시 A₅입니다.']] }
  ];
  const api = { identity, compose, inverse, commutator, permutations, parity, groupClosure, derivedSubgroup, derivedSeries, namedGroup, pair, addPair, mulPair, conjugatePair, inversePair, rootPermutationChecks, v4Closure, fixedBasis, QUINTICS, bisection, formatBisectionBracket, QUIZ };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const labs = window.ArticleLabs;
  function setup(ctx) { ctx.root.classList.add('galois-lab', 'al-workshop'); }
  function block(ctx, label, value, extra = '') { return ctx.el('div', { class: 'galois-card' + (extra ? ' ' + extra : '') }, [ctx.el('div', { class: 'galois-card-label' }, label), ctx.el('div', { class: 'galois-expression' }, value)]); }
  function message(ctx, en, ko, kind = '') { return ctx.el('p', { class: 'galois-explanation' + (kind ? ' ' + kind : '') }, ctx.text(en, ko)); }
  function permutationDiagram(ctx, p) {
    const label = p.map((image, i) => `${ROOT_LABELS[i]} maps to ${ROOT_LABELS[image]}`).join('; ');
    const connectors = ctx.svg('svg', { viewBox: '0 0 120 176', preserveAspectRatio: 'none', class: 'galois-connectors', 'aria-hidden': 'true' });
    p.forEach((image, i) => {
      const y1 = 22 + 44 * i, y2 = 22 + 44 * image;
      connectors.append(ctx.svg('line', { x1: 0, y1, x2: 113, y2, stroke: 'currentColor', 'stroke-width': 1.6 }), ctx.svg('polygon', { points: `120,${y2} 111,${y2 - 4} 111,${y2 + 4}`, fill: 'currentColor' }));
    });
    const column = () => ctx.el('div', { class: 'galois-root-column' }, ROOT_LABELS.map(value => ctx.el('strong', { class: 'galois-root-node' }, value)));
    return ctx.el('div', { class: 'galois-permutation-diagram', role: 'img', 'aria-label': label }, [column(), connectors, column()]);
  }
  function quizDiagram(ctx, index) {
    if (index === 2) return permutationDiagram(ctx, [2, 3, 0, 1]);
    if (index === 1) return ctx.el('div', { class: 'galois-quiz-diagram' }, [
      ctx.el('div', { class: 'galois-linear-map' }, [
        ctx.el('div', { class: 'galois-matrix', 'aria-label': 'matrix rows 1,0 and 0,2' }, ['1', '0', '0', '2'].map(s => ctx.el('span', {}, s))),
        ctx.el('div', { class: 'galois-vector', 'aria-label': 'column vector a,b' }, ['a', 'b'].map(s => ctx.el('span', {}, s))),
        ctx.el('span', {}, '='),
        ctx.el('div', { class: 'galois-vector', 'aria-label': 'column vector a,2b' }, ['a', '2b'].map(s => ctx.el('span', {}, s)))
      ]), ctx.el('div', { class: 'galois-quiz-diagram-line' }, 'T(√2)²  ?  T((√2)²)')
    ]);
    if (index === 3) return ctx.el('div', { class: 'galois-quiz-diagram' }, [
      block(ctx, 'L', 'Q(√2, √3)'), ctx.el('div', { class: 'galois-down', 'aria-hidden': 'true' }, '│'), block(ctx, 'E', 'Q(√3)'), ctx.el('div', { class: 'galois-down', 'aria-hidden': 'true' }, '│'), block(ctx, 'F', 'Q'), message(ctx, 'σ flips √2 but fixes √3.', 'σ는 √2의 부호를 바꾸지만 √3은 그대로 둡니다.')
    ]);
    if (index === 4) return ctx.el('div', { class: 'galois-quiz-diagram' }, ['S₃: 6 → 3 → 1', 'S₅: 120 → 60 → 60'].map(row => {
      const [name, numbers] = row.split(': ');
      return ctx.el('div', { class: 'galois-series-row' }, [ctx.el('strong', {}, name), ctx.el('div', { class: 'galois-series' }, numbers.split(' → ').flatMap((value, i) => [i ? ctx.el('span', { 'aria-hidden': 'true' }, '→') : null, ctx.el('span', { class: 'galois-order-node' }, value)]))]);
    }));
    return ctx.el('div', { class: 'galois-quiz-diagram' }, [
      ctx.el('div', { class: 'galois-formula-inputs' }, ['a', 'b', 'c'].map(s => ctx.el('span', { class: 'galois-order-node' }, s))),
      ctx.el('div', { class: 'galois-down', 'aria-hidden': 'true' }, '↓'), block(ctx, ctx.text('Allowed operations', '허용된 연산'), '+  −  ×  ÷  m-th root'), ctx.el('div', { class: 'galois-down', 'aria-hidden': 'true' }, '↓'), block(ctx, ctx.text('Finite expression', '유한한 표현'), '(−b ± √(b²−4ac)) / (2a)')
    ]);
  }
  function shareFields(ctx, id, fields, render, extra = {}) {
    ctx.stateLink(id, {
      read: () => ({ fields: fields.map(f => f.type === 'checkbox' ? f.checked : f.value), ...extra.read?.() }),
      write: state => {
        if (!state || !Array.isArray(state.fields) || state.fields.length !== fields.length) return false;
        const valid = state.fields.every((value, i) => {
          const f = fields[i]; if (f.type === 'checkbox') return typeof value === 'boolean';
          if (typeof value !== 'string') return false;
          if (f.tagName === 'SELECT') return [...f.options].some(o => o.value === value);
          // Number('0x1') is 1, but HTML number inputs reject hexadecimal.
          // Validate native decimal syntax before changing ANY saved field.
          return /^-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value) && integer(Number(value), Number(f.min), Number(f.max));
        });
        if (!valid || extra.valid && !extra.valid(state)) return false;
        fields.forEach((f, i) => { if (f.type === 'checkbox') f.checked = state.fields[i]; else f.value = state.fields[i]; }); extra.write?.(state); render(); return true;
      }
    });
  }
  labs.register('galois-quadratic', ctx => {
    setup(ctx);
    const fields = [['a', 1], ['b', 1], ['c', 1], ['d', -1]].map(([name, value]) => ctx.control(name, name, { type: 'number', min: -9, max: 9, step: 1, value }));
    const norm = ctx.metric('Field norm uσ(u)', 'Field norm uσ(u)'), relation = ctx.metric('Conjugation matrix', 'Conjugation matrix');
    function render() {
      const [a, b, c, d] = fields.map(f => Number(f.value)), u = pair(a, b), v = pair(c, d), n = a * a - 2 * b * b;
      norm.textContent = String(n); relation.textContent = 'diag(1, −1)';
      const inverse = inversePair(u), inverseText = inverse ? inverseLabel(inverse) : ctx.text('not defined (u = 0)', '정의되지 않음 (u = 0)');
      ctx.view.replaceChildren(ctx.el('div', { class: 'galois-card-grid' }, [block(ctx, 'u', pairLabel(u)), block(ctx, 'v', pairLabel(v)), block(ctx, 'u + v', pairLabel(addPair(u, v))), block(ctx, 'u × v', pairLabel(mulPair(u, v))), block(ctx, 'σ(u)', pairLabel(conjugatePair(u))), block(ctx, '1/u', inverseText)]), message(ctx, 'Coordinates (a,b) represent a+b√2. Multiplication uses (√2)²=2; conjugation keeps a and changes b to −b. The product uσ(u)=a²−2b² is called the field norm. For u=1+√2 it is −1, so it is not a vector length.', '좌표 (a,b)는 a+b√2를 나타냅니다. 곱셈에는 (√2)²=2를 쓰고, conjugation은 a를 그대로 두고 b의 부호만 바꿉니다. uσ(u)=a²−2b²라는 곱을 field norm이라고 합니다. u=1+√2이면 이 값은 −1입니다. Vector의 길이와는 다른 개념입니다.'), message(ctx, n ? 'The inverse stays in the same two-dimensional Q-vector space.' : 'u=0 has no multiplicative inverse. For rational a,b, a²−2b² vanishes only when both are zero.', n ? 'Inverse도 같은 2차원 Q-vector space 안에 있습니다.' : 'u=0에는 multiplicative inverse가 없습니다. 유리수 a,b에 대해 a²−2b²=0이 성립하려면 a=b=0이어야 합니다.'));
    }
    ctx.onChange(render); shareFields(ctx, 'galois-quadratic', fields, render);
  });
  labs.register('galois-permutations', ctx => {
    setup(ctx); const all = permutations(4);
    const selected = ctx.control('Candidate permutation', '확인할 permutation', { type: 'select', value: '0', options: all.map((p, i) => ({ value: String(i), en: `${i + 1}. [${p.map(x => ROOT_LABELS[x]).join(', ')}]`, ko: `${i + 1}. [${p.map(x => ROOT_LABELS[x]).join(', ')}]` })) });
    const valid = ctx.metric('Valid Q-automorphisms', '가능한 Q-automorphism'), position = ctx.metric('Candidate', '현재 후보');
    ctx.button('Previous', '이전 후보', () => { selected.value = String((Number(selected.value) + 23) % 24); render(); });
    ctx.button('Next', '다음 후보', () => { selected.value = String((Number(selected.value) + 1) % 24); render(); });
    ctx.button('Exchange the two pairs', '두 쌍 서로 바꾸기', () => { selected.value = String(all.findIndex(p => p.join(',') === '2,3,0,1')); render(); });
    function render() {
      const index = Number(selected.value), p = all[index], check = rootPermutationChecks(p);
      valid.textContent = '4 / 24'; position.textContent = `${index + 1} / 24`;
      const rows = permutationDiagram(ctx, p);
      ctx.view.replaceChildren(rows, ctx.el('div', { class: 'galois-tests' }, [block(ctx, ctx.text('Same quartic root set', '같은 quartic의 근'), 'PASS'), block(ctx, ctx.text('Squares 2 and 3 preserved', '제곱값 2와 3 보존'), check.squares ? 'PASS' : 'FAIL'), block(ctx, ctx.text('Opposite pairs preserved', '서로 음수인 쌍 보존'), check.negatives ? 'PASS' : 'FAIL')]), message(ctx, check.valid ? 'Allowed: independently choose the sign of √2 and √3. These choices extend to exactly four Q-automorphisms.' : 'Forbidden: remaining among the four roots is not enough. A Q-automorphism must preserve the individual relations α²=2 and β²=3, as well as addition.', check.valid ? '가능합니다. √2와 √3의 부호를 각각 고르는 네 가지 선택이 모두 Q-automorphism으로 확장됩니다.' : '불가능합니다. 같은 네 근 중 하나로 보낸다는 것만으로는 부족합니다. Q-automorphism은 덧셈뿐 아니라 α²=2, β²=3이라는 각각의 관계도 보존해야 합니다.', check.valid ? 'galois-pass' : 'galois-fail'));
    }
    ctx.onChange(render); shareFields(ctx, 'galois-permutations', [selected], render);
  });
  labs.register('galois-correspondence', ctx => {
    setup(ctx);
    const sigma = ctx.control('Generator σ: flip √2', 'Generator σ: √2의 부호 바꾸기', { type: 'checkbox', value: true });
    const tau = ctx.control('Generator τ: flip √3', 'Generator τ: √3의 부호 바꾸기', { type: 'checkbox', value: false });
    const both = ctx.control('Generator στ: flip both', 'Generator στ: 두 부호 바꾸기', { type: 'checkbox', value: false });
    const order = ctx.metric('Subgroup order |H|', 'Subgroup order |H|'), dimension = ctx.metric('[Lᴴ : Q]', '[Lᴴ : Q]'), product = ctx.metric('|H| × [Lᴴ : Q]', '|H| × [Lᴴ : Q]');
    const names = ['1', '√2', '√3', '√6'];
    function render() {
      const h = v4Closure([sigma.checked ? 1 : 0, tau.checked ? 2 : 0, both.checked ? 3 : 0]), basis = fixedBasis(h);
      const field = basis.length === 4 ? 'L = Q(√2, √3)' : basis.length === 1 ? 'Q' : `Q(${names[basis[1]]})`;
      order.textContent = String(h.length); dimension.textContent = String(basis.length); product.textContent = String(h.length * basis.length);
      ctx.view.replaceChildren(block(ctx, ctx.text('Generated subgroup (closed under composition)', 'Composition으로 생성한 subgroup'), `{${h.map(g => V4_LABELS[g]).join(', ')}}`), ctx.el('div', { class: 'galois-basis' }, names.map((name, i) => ctx.el('div', { class: 'galois-basis-term ' + (basis.includes(i) ? 'galois-kept' : 'galois-removed') }, [ctx.el('strong', {}, name), ctx.el('span', {}, ctx.text(basis.includes(i) ? 'fixed' : 'not fixed', basis.includes(i) ? '고정됨' : '고정되지 않음'))]))), block(ctx, 'Lᴴ', field), message(ctx, 'A coefficient may remain only if every element of H fixes its basis vector. Adding generators can enlarge H and shrink the fixed field.', 'H의 모든 원소가 고정하는 basis vector만 남습니다. Generator를 추가하면 H가 커지고 fixed field가 작아질 수 있습니다.'));
    }
    ctx.onChange(render); shareFields(ctx, 'galois-correspondence', [sigma, tau, both], render);
  });
  labs.register('galois-derived', ctx => {
    setup(ctx);
    const selected = ctx.control('Permutation group', 'Permutation group', { type: 'select', value: 'S3', options: ['C5', 'S3', 'S4', 'A5', 'S5'].map(name => ({ value: name, en: name, ko: name })) });
    let step = 0, current = selected.value;
    const order = ctx.metric('Current order', '현재 order'), end = ctx.metric('Solvable?', 'Solvable한가?');
    ctx.button('Next derived subgroup', '다음 derived subgroup 구하기', () => { step = Math.min(step + 1, namedGroup(selected.value).length - 1); render(); });
    ctx.button('Show complete series', '전체 series 보기', () => { step = namedGroup(selected.value).length - 1; render(); });
    ctx.onReset(() => { step = 0; current = 'S3'; });
    function render() {
      if (current !== selected.value) { current = selected.value; step = 0; }
      const series = namedGroup(current); step = Math.min(step, series.length - 1); const solvable = series.at(-1).length === 1;
      order.textContent = String(series[step].length); end.textContent = ctx.text(solvable ? 'Yes' : 'No', solvable ? '그렇다' : '아니다');
      const sequence = ctx.el('div', { class: 'galois-series', 'aria-label': ctx.text('Orders of derived subgroups', 'Derived subgroup의 order') }, series.slice(0, step + 1).flatMap((g, i) => [i ? ctx.el('span', { class: 'galois-series-arrow', 'aria-hidden': 'true' }, '→') : null, block(ctx, `G${i ? '⁽' + i + '⁾' : ''}`, String(g.length))]));
      const atEnd = step === series.length - 1;
      ctx.view.replaceChildren(sequence, block(ctx, ctx.text('Operation at each arrow', '화살표마다 하는 연산'), 'G′ = ⟨p q p⁻¹ q⁻¹ : p,q ∈ G⟩'), message(ctx, atEnd ? solvable ? 'The series reaches {e}. Earlier groups may be nonabelian; solvable does not mean abelian.' : 'The series has stabilized at A₅, not {e}. Repeating the commutator operation no longer shrinks the group.' : 'Each step generates a subgroup from all commutators. Use Next to compute the next order.', atEnd ? solvable ? 'Series가 {e}에 도달했습니다. 앞 단계의 group은 nonabelian일 수 있습니다. Solvable과 abelian은 같은 뜻이 아닙니다.' : 'Series가 {e}가 아닌 A₅에서 멈췄습니다. Commutator로 subgroup을 다시 구해도 크기는 그대로입니다.' : '각 단계에서는 모든 commutator로 생성한 subgroup을 구합니다. 다음 버튼을 눌러 order의 변화를 확인해 보세요.'));
    }
    ctx.onChange(render); shareFields(ctx, 'galois-derived', [selected], render, { read: () => ({ step }), valid: s => integer(s.step, 0, namedGroup(s.fields[0]).length - 1), write: s => { step = s.step; current = s.fields[0]; } });
  });
  labs.register('galois-quintic', ctx => {
    setup(ctx);
    const selected = ctx.control('Polynomial', 'Polynomial', { type: 'select', value: 'obstruction', options: [{ value: 'obstruction', en: 'x⁵ − 5x + 1: nonsolvable', ko: 'x⁵ − 5x + 1: nonsolvable' }, { value: 'radical', en: 'x⁵ − 2: solvable', ko: 'x⁵ − 2: solvable' }] });
    const branch = ctx.control('Real-root interval', '실근을 포함하는 구간', { type: 'select', value: '0', options: [{ value: '0', en: 'Left / unique real root', ko: '왼쪽 실근 / 유일한 실근' }, { value: '1', en: 'Middle real root', ko: '가운데 실근' }, { value: '2', en: 'Right real root', ko: '오른쪽 실근' }] });
    let steps = 0, preset = selected.value, root = branch.value;
    const width = ctx.metric('Bracket width', '구간의 폭'), residual = ctx.metric('f(midpoint)', 'f(midpoint)'), count = ctx.metric('Bisections', 'Bisection 횟수');
    ctx.button('Bisect once', '한 번 이등분', () => { steps = Math.min(40, steps + 1); render(); });
    ctx.button('Bisect 20 times', '20번 이등분', () => { steps = Math.min(40, steps + 20); render(); });
    ctx.onReset(() => { steps = 0; preset = 'obstruction'; root = '0'; });
    function render() {
      if (selected.value !== preset) { preset = selected.value; branch.value = '0'; root = '0'; steps = 0; }
      if (preset === 'radical' && branch.value !== '0') branch.value = '0';
      if (branch.value !== root) { root = branch.value; steps = 0; }
      branch.disabled = preset === 'radical';
      const result = bisection(preset, Number(root), steps), model = QUINTICS[preset];
      width.textContent = result.width.toExponential(3); residual.textContent = result.residual.toExponential(3); count.textContent = String(steps);
      const [start, stop] = result.initial, points = Array.from({ length: 81 }, (_, i) => { const x = start + (stop - start) * i / 80; return [x, model.f(x)]; });
      const ys = points.map(p => p[1]), margin = (Math.max(...ys) - Math.min(...ys)) * .07 || 1;
      ctx.view.replaceChildren(ctx.plot({ x: [start, stop], y: [Math.min(...ys) - margin, Math.max(...ys) + margin], title: model.label, xLabel: 'x', yLabel: 'f(x)', series: [{ points, color: 'blue' }, { points: [[start, 0], [stop, 0]], color: 'muted' }], bands: [{ x0: result.lo, x1: result.hi, y0: Math.min(...ys) - margin, y1: Math.max(...ys) + margin, color: 'gold' }], markers: [{ x: result.middle, y: result.residual, label: '', color: 'red' }], xTicks: [start, (start + stop) / 2, stop], yTicks: [0] }), block(ctx, ctx.text('Current bracket', '현재 구간'), formatBisectionBracket(result.lo, result.hi)), block(ctx, ctx.text('Midpoint approximation', 'Midpoint로 구한 근삿값'), result.middle.toPrecision(12)), message(ctx, preset === 'obstruction' ? 'Bisection converges even though the Galois group is S₅ and no radical solution exists. The graph and decimal residual visualize approximation; irreducibility and conjugation prove the algebraic claim in the text.' : 'The positive real root is exactly 2^(1/5). The full splitting field over Q has a solvable group of order 20, not C₅. The other four roots are complex.', preset === 'obstruction' ? 'Galois group이 S₅여서 radical로 풀리지 않아도 bisection은 수렴합니다. 그래프와 소수로 표시한 residual은 근사 계산 결과입니다. 대수적 결론은 본문의 irreducibility와 conjugation 논증으로 증명합니다.' : '양의 실근은 정확히 2^(1/5)입니다. Q 위 전체 splitting field의 group은 order가 20인 solvable group이며, C₅가 아닙니다. 나머지 네 근은 복소수입니다.'));
    }
    ctx.onChange(render); shareFields(ctx, 'galois-quintic', [selected, branch], render, { read: () => ({ steps }), valid: s => integer(s.steps, 0, 40) && !(s.fields[0] === 'radical' && s.fields[1] !== '0'), write: s => { steps = s.steps; preset = s.fields[0]; root = s.fields[1]; } });
  });
  labs.register('galois-quiz', ctx => {
    setup(ctx);
    const selected = ctx.control('Question', '문제', { type: 'select', value: '0', options: QUIZ.map((q, i) => ({ value: String(i), en: `${i + 1}. ${q.en}`, ko: `${i + 1}. ${q.ko}` })) });
    let answers = Array(5).fill(null), hints = Array(5).fill(false);
    const progress = ctx.metric('Correct / answered', '정답 수 / 푼 문제 수');
    ctx.button('Previous question', '이전 문제', () => { selected.value = String((Number(selected.value) + 4) % 5); render(); });
    ctx.button('Next question', '다음 문제', () => { selected.value = String((Number(selected.value) + 1) % 5); render(); });
    ctx.button('Hint', '힌트', () => { hints[Number(selected.value)] = true; render(); });
    ctx.onReset(() => { answers = Array(5).fill(null); hints = Array(5).fill(false); });
    function render() {
      const index = Number(selected.value), q = QUIZ[index], answer = answers[index];
      progress.textContent = `${answers.filter((a, i) => a === QUIZ[i].answer).length} / ${answers.filter(a => a !== null).length}`;
      const diagram = quizDiagram(ctx, index);
      const choices = ctx.el('div', { class: 'galois-choices', role: 'group', 'aria-label': ctx.text('Choose an answer', '답 선택') }, q.options.map((option, i) => {
        const button = ctx.el('button', { type: 'button', class: 'galois-choice' + (answer === i ? ' galois-selected' : ''), 'aria-pressed': answer === i ? 'true' : 'false' }, ctx.text(...option));
        button.addEventListener('click', () => { answers[index] = i; render(); }); return button;
      }));
      const feedback = answer === null ? message(ctx, 'Choose an option to see why it is right or wrong.', '보기를 선택하면 그 답이 맞거나 틀린 이유를 확인할 수 있습니다.') : message(ctx, `${answer === q.answer ? 'Correct. ' : 'Not quite. '}${q.optionReasons[answer][0]}`, `${answer === q.answer ? '맞았습니다. ' : '다시 생각해 보세요. '}${q.optionReasons[answer][1]}`, answer === q.answer ? 'galois-pass' : 'galois-fail');
      ctx.view.replaceChildren(ctx.el('h4', {}, ctx.text(q.en, q.ko)), diagram, message(ctx, q.promptEn, q.promptKo), choices, ...(hints[index] ? [message(ctx, q.hintEn, q.hintKo, 'galois-hint')] : []), feedback);
    }
    ctx.onChange(render); shareFields(ctx, 'galois-quiz', [selected], render, { read: () => ({ answers: answers.slice(), hints: hints.slice() }), valid: s => Array.isArray(s.answers) && s.answers.length === 5 && s.answers.every(a => a === null || integer(a, 0, 2)) && Array.isArray(s.hints) && s.hints.length === 5 && s.hints.every(x => typeof x === 'boolean'), write: s => { answers = s.answers.slice(); hints = s.hints.slice(); } });
  });
})();
