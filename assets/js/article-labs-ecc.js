/* Small exact binary codes. All indices exposed to readers are one-based. */
(function () {
  'use strict';
  const H = [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]];
  const POSITIONS = ['p1', 'p2', 'd1', 'p4', 'd2', 'd3', 'd4', 'q'];
  const integer = (n, lo, hi) => Number.isInteger(n) && n >= lo && n <= hi;
  function bits(value, length) {
    if (typeof value === 'string') {
      if (value.length !== length || !/^[01]+$/.test(value)) throw new Error('Expected a binary word of the stated length.');
      value = value.split('').map(Number);
    }
    if (!Array.isArray(value) || value.length !== length || !Array.from(value).every(x => x === 0 || x === 1)) throw new Error('Expected binary bits.');
    return value.slice();
  }
  function word(value, length) { return bits(value, length).join(''); }
  function fromInteger(value, length) {
    if (!integer(value, 0, 2 ** length - 1) || !integer(length, 1, 8)) throw new Error('Invalid binary integer.');
    return value.toString(2).padStart(length, '0').split('').map(Number);
  }
  function parity(value) { return bits(value, value.length).reduce((sum, bit) => sum ^ bit, 0); }
  function distance(a, b) { const x = bits(a, a.length), y = bits(b, a.length); return x.reduce((sum, bit, i) => sum + (bit ^ y[i]), 0); }
  function encode(message) {
    const [a, b, c, d] = bits(message, 4);
    return [a ^ b ^ d, a ^ c ^ d, a, b ^ c ^ d, b, c, d];
  }
  function syndrome(received) {
    const r = bits(received, 7);
    return H.map(row => row.reduce((sum, entry, i) => sum ^ (entry & r[i]), 0));
  }
  function indexFromSyndrome(s) { const a = bits(s, 3); return a[0] + 2 * a[1] + 4 * a[2]; }
  function dataBits(codeword) { const c = bits(codeword, 7); return [c[2], c[4], c[5], c[6]]; }
  function decode(received) {
    const corrected = bits(received, 7), s = syndrome(corrected), index = indexFromSyndrome(s);
    if (index) corrected[index - 1] ^= 1;
    return { syndrome: s, index, corrected, data: dataBits(corrected) };
  }
  function encodeExtended(message) { const c = encode(message); return [...c, parity(c)]; }
  function decodeExtended(received) {
    const corrected = bits(received, 8), s = syndrome(corrected.slice(0, 7)), index = indexFromSyndrome(s), overall = parity(corrected);
    let status = 'clean', flipped = 0;
    if (index && !overall) status = 'double-detected';
    else if (overall) { flipped = index || 8; corrected[flipped - 1] ^= 1; status = index ? 'single-corrected' : 'parity-corrected'; }
    return { status, syndrome: s, index, overall, flipped, corrected: status === 'double-detected' ? null : corrected, data: status === 'double-detected' ? null : dataBits(corrected.slice(0, 7)) };
  }
  function applyFlips(codeword, positions) {
    const c = bits(codeword, codeword.length);
    if (!Array.isArray(positions) || new Set(positions).size !== positions.length || !positions.every(i => integer(i, 1, c.length))) throw new Error('Invalid error positions.');
    positions.forEach(i => { c[i - 1] ^= 1; }); return c;
  }
  function majority(received) {
    const r = bits(received, received.length);
    if (!r.length || r.length % 2 !== 1) throw new Error('Majority requires a positive odd length.');
    return r.reduce((a, b) => a + b, 0) > r.length / 2 ? 1 : 0;
  }
  function repetitionError(n, p) {
    if (![1, 3, 5, 7, 9].includes(n) || !Number.isFinite(p) || p < 0 || p > 1) throw new Error('Invalid repetition channel.');
    let sum = 0, choose = 1;
    for (let k = 0; k <= n; k++) {
      if (k >= (n + 1) / 2) sum += choose * p ** k * (1 - p) ** (n - k);
      choose = choose * (n - k) / (k + 1);
    }
    return sum;
  }
  function distanceCandidates(received, t) {
    const r = bits(received, 3);
    if (!integer(t, 0, 3)) throw new Error('Invalid error bound.');
    return ['000', '111'].filter(c => distance(c, r) <= t);
  }
  function quizProblem(number) {
    if (!integer(number, 0, 111)) throw new Error('Invalid challenge.');
    const message = fromInteger((number * 7 + 10) % 16, 4), position = (number * 3 + 4) % 7 + 1;
    return { message, position, codeword: encode(message), received: applyFlips(encode(message), [position]) };
  }
  const api = { H, POSITIONS, bits, word, fromInteger, parity, distance, encode, syndrome, indexFromSyndrome, dataBits, decode, encodeExtended, decodeExtended, applyFlips, majority, repetitionError, distanceCandidates, quizProblem };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  const labs = window.ArticleLabs;
  function setup(ctx) { ctx.root.classList.add('al-workshop', 'ecc-lab'); }
  function text(ctx, en, ko, kind = '') { return ctx.el('p', { class: 'ecc-text' + (kind ? ' ecc-' + kind : '') }, ctx.text(en, ko)); }
  function heading(ctx, en, ko) { return ctx.el('div', { class: 'ecc-label' }, ctx.text(en, ko)); }
  function bitRow(ctx, values, options = {}) {
    if (options.click && !/^[a-z]+(?:-[a-z]+)*$/.test(options.key || '')) throw new Error('Editable bit rows need a stable key.');
    const row = ctx.el('div', { class: 'ecc-bits', role: 'group', 'data-ecc-row': options.key || null, 'data-ecc-count': values.length, 'aria-label': options.name || ctx.text('Bits', 'Bit 배열') });
    values.forEach((value, i) => {
      const label = options.labels ? options.labels[i] : String(i + 1), changed = options.changed && options.changed.includes(i + 1), selected = options.selected === i + 1;
      const cell = ctx.el(options.click ? 'button' : 'div', { type: options.click ? 'button' : null, class: 'ecc-bit' + (changed ? ' ecc-flipped' : '') + (selected ? ' ecc-selected' : ''), 'aria-label': `${label}: ${value}` + (changed ? ctx.text(', flipped', ', 뒤집힘') : ''), 'aria-pressed': options.click ? String(options.selectable ? selected : options.changed ? Boolean(changed) : Boolean(value)) : null, 'data-ecc-position': i + 1 }, [ctx.el('span', { class: 'ecc-position' }, label), ctx.el('strong', {}, value), ctx.el('span', { class: 'ecc-bit-mark' }, changed ? ctx.text('flip', '변경') : options.selectable && selected ? ctx.text('chosen', '선택') : '·')]);
      if (options.click) cell.addEventListener('click', () => {
        options.click(i);
        // render() replaces the view, so restore keyboard focus to the same
        // position in the same row rather than leaving it on a removed button.
        const replacementRow = ctx.root.querySelector(`[data-ecc-row="${options.key}"]`);
        const replacement = replacementRow && replacementRow.querySelector(`[data-ecc-position="${i + 1}"]`);
        if (replacement) replacement.focus({ preventScroll: true });
      });
      row.append(cell);
    });
    return row;
  }
  function fact(ctx, en, ko, value) { return ctx.el('div', { class: 'ecc-fact' }, [heading(ctx, en, ko), ctx.el('strong', {}, value)]); }
  function transmittedReadout(ctx, codeword) {
    return ctx.el('p', { class: 'ecc-transmitted' }, [
      ctx.el('span', {}, ctx.text('Sent codeword c (simulation truth): ', '전송한 codeword c (시뮬레이션의 원본): ')),
      ctx.el('code', {}, codeword.join(''))
    ]);
  }
  function modelLink(ctx, id, read, validate, write, render) {
    ctx.stateLink(id, { read, write: state => { if (!state || typeof state !== 'object' || Array.isArray(state) || !validate(state)) return false; write(state); render(); return true; } });
  }
  const validFlags = (values, n) => Array.isArray(values) && values.length === n && Array.from(values).every(x => typeof x === 'boolean');
  const indices = flags => flags.flatMap((flag, i) => flag ? [i + 1] : []);
  const select = (ctx, en, ko, values, initial) => ctx.control(en, ko, { type: 'select', value: initial, options: values.map(value => typeof value === 'object' ? value : { value: String(value), en: String(value), ko: String(value) }) });

  labs.register('ecc-repetition', ctx => {
    setup(ctx);
    const sent = select(ctx, 'Message bit', '보낼 bit', [0, 1], 1), n = select(ctx, 'Repetitions n', '반복 횟수 n', [1, 3, 5, 7, 9], 3);
    const p = ctx.control('Independent flip probability p', 'Bit 하나가 뒤집힐 확률 p (서로 독립)', { type: 'range', min: 0, max: .5, step: .01, value: .1 });
    let flips = Array(9).fill(false);
    const probability = ctx.metric('Decoded bit error probability', 'Decoder가 틀릴 확률'), rate = ctx.metric('Code rate', 'Code rate');
    ctx.button('Clear flips', '뒤집은 bit 되돌리기', () => { flips.fill(false); render(); });
    ctx.button('Flip a majority', '과반수를 뒤집기', () => { flips = flips.map((_, i) => i < (Number(n.value) + 1) / 2); render(); });
    function render() {
      const length = Number(n.value), bit = Number(sent.value), active = indices(flips).filter(i => i <= length), received = applyFlips(Array(length).fill(bit), active), decoded = majority(received), error = repetitionError(length, Number(p.value));
      probability.textContent = (100 * error).toFixed(4) + '%'; rate.textContent = `1/${length}`;
      const points = Array.from({ length: 51 }, (_, i) => [i / 100, repetitionError(length, i / 100)]);
      const markerProbability = ctx.format ? ctx.format(error, 4) : error.toPrecision(4);
      ctx.view.replaceChildren(heading(ctx, 'Click received bits to flip them', '받은 bit를 눌러 뒤집어 보세요'), bitRow(ctx, received, { key: 'received', name: ctx.text('Received repetition bits', '받은 repetition bit'), changed: active, click: i => { flips[i] = !flips[i]; render(); } }), ctx.el('div', { class: 'ecc-facts' }, [fact(ctx, 'Sent → majority vote', '보낸 bit → majority vote', `${bit} → ${decoded}`), fact(ctx, 'Manual flips', '직접 뒤집은 bit 수', `${active.length}/${length}`)]), text(ctx, decoded === bit ? 'This pattern decodes to the original bit.' : 'This pattern gives the wrong message bit.', decoded === bit ? '이 error pattern에서는 원래 bit를 복구합니다.' : '이 error pattern에서는 원래와 다른 bit를 내놓습니다.', decoded === bit ? 'success' : 'warning'), ctx.plot({ x: [0, .5], y: [0, .5], title: ctx.text('Majority-decoding error on an independent binary symmetric channel', 'BSC에서 majority decoding이 틀릴 확률'), xLabel: ctx.text('Flip probability p per transmitted bit', 'Bit 하나가 뒤집힐 확률 p'), yLabel: ctx.text('Probability of a wrong decoded message bit', 'Decoder가 내놓은 message bit가 틀릴 확률'), series: [{ points, color: 'blue', label: `n = ${length}` }, { points: [[0, 0], [.5, .5]], color: 'muted', dash: true, label: ctx.text('Uncoded: error probability p', 'Code 없이 보낼 때: 오류 확률 p') }], markers: [{ x: Number(p.value), y: error, label: `p=${p.value}, P(error)=${markerProbability}`, color: 'gold' }] }), text(ctx, 'The curve assumes independent random flips. The buttons above select a specific error pattern, not a random sample from that model.', '그래프는 각 bit가 독립적으로 뒤집히는 BSC에서 majority vote가 틀릴 확률을 보여 줍니다. 위의 버튼은 뒤집을 위치를 직접 고르는 실험으로, BSC에서 오류를 무작위로 발생시키는 것은 아닙니다.'));
      const correctable = (length - 1) / 2;
      ctx.status.textContent = ctx.text(`${length} transmitted ${length === 1 ? 'bit carries' : 'bits carry'} one message bit. Majority corrects at most ${correctable} ${correctable === 1 ? 'flip' : 'flips'}.`, `Message bit 하나를 ${length}개 bit로 보냅니다. 최대 ${correctable}개가 뒤집혀도 majority vote로 복구할 수 있습니다.`);
    }
    ctx.onChange(render); ctx.onReset(() => { flips = Array(9).fill(false); });
    modelLink(ctx, 'ecc-repetition', () => ({ bit: Number(sent.value), n: Number(n.value), p: Number(p.value), flips: flips.slice() }), s => integer(s.bit, 0, 1) && [1, 3, 5, 7, 9].includes(s.n) && Number.isFinite(s.p) && s.p >= 0 && s.p <= .5 && Math.abs(s.p * 100 - Math.round(s.p * 100)) < 1e-9 && validFlags(s.flips, 9), s => { sent.value = String(s.bit); n.value = String(s.n); p.value = String(s.p); flips = s.flips.slice(); }, render);
  });

  labs.register('ecc-parity', ctx => {
    setup(ctx); let message = [1, 0, 1, 1], flips = Array(5).fill(false);
    const result = ctx.metric('Receiver parity check', '받은 word의 parity check');
    ctx.button('One flip: position 2', '2번 bit 하나 뒤집기', () => { flips = [false, true, false, false, false]; render(); });
    ctx.button('Two flips: positions 2, 4', '2번·4번 bit 뒤집기', () => { flips = [false, true, false, true, false]; render(); });
    ctx.button('Clear flips', '뒤집은 bit 되돌리기', () => { flips.fill(false); render(); });
    function render() {
      const encoded = [...message, parity(message)], active = indices(flips), received = applyFlips(encoded, active), check = parity(received);
      result.textContent = check ? ctx.text('FAIL: odd parity', 'FAIL: odd parity') : ctx.text('PASS: even parity', 'PASS: even parity');
      ctx.view.replaceChildren(heading(ctx, 'Message: click to edit', 'Message: 눌러서 바꾸기'), bitRow(ctx, message, { key: 'message', name: ctx.text('Message bits', 'Message bit'), labels: ['d1', 'd2', 'd3', 'd4'], click: i => { message[i] ^= 1; render(); } }), heading(ctx, 'Received word: click to flip a bit', '받은 word: 눌러서 bit 뒤집기'), bitRow(ctx, received, { key: 'received', name: ctx.text('Received bits with chosen channel errors', '오류를 넣은 수신 bit'), labels: ['d1', 'd2', 'd3', 'd4', 'p'], changed: active, click: i => { flips[i] = !flips[i]; render(); } }), text(ctx, check ? 'The receiver detects an error, but this one check does not identify its position.' : 'The receiver sees a valid even-parity word. Zero flips and any even number of flips give the same check result.', check ? '오류가 있다는 사실은 알 수 있지만, 이 check 하나만으로는 어느 bit가 뒤집혔는지 알 수 없습니다.' : '받은 word가 even parity를 만족합니다. 오류가 없을 때도, 짝수 개 bit가 뒤집혔을 때도 같은 결과가 나옵니다.', check ? 'warning' : ''), fact(ctx, 'Known only in this simulation: actual flips', '실제로 뒤집힌 bit 수 (시뮬레이션에서만 확인 가능)', active.length), text(ctx, active.length ? `The simulation knows that ${active.length === 1 ? 'position' : 'positions'} ${active.join(', ')} changed. The receiver does not have the original word for comparison.` : 'No bits have changed in the simulation.', active.length ? `이 실험에서는 ${active.join(', ')}번 bit를 뒤집었습니다. 실제 수신자는 비교할 원본을 갖고 있지 않습니다.` : '아직 어떤 bit도 뒤집지 않았습니다.'));
      ctx.status.textContent = ctx.text('Even parity detects every odd number of bit flips; it misses every nonzero even number.', 'Even parity는 홀수 개 bit 오류를 모두 검출하지만, 짝수 개 bit 오류는 놓칩니다.');
    }
    ctx.onChange(render); ctx.onReset(() => { message = [1, 0, 1, 1]; flips = Array(5).fill(false); });
    modelLink(ctx, 'ecc-parity', () => ({ message: message.slice(), flips: flips.slice() }), s => { try { bits(s.message, 4); return Array.isArray(s.message) && validFlags(s.flips, 5); } catch (_) { return false; } }, s => { message = s.message.slice(); flips = s.flips.slice(); }, render);
  });

  function checksDiagram(ctx, received) {
    const r = received.slice(0, 7), s = syndrome(r);
    const description = H.map((row, j) => `s${2 ** j}: ${row.flatMap((entry, i) => entry ? [r[i]] : []).join(' XOR ')} = ${s[j]}`).join('; ');
    const grid = ctx.el('div', { class: 'ecc-check-grid', role: 'img', 'aria-label': ctx.text('Received bits used by each check. ', '각 check에서 사용하는 수신 bit. ') + description });
    grid.append(ctx.el('span', { class: 'ecc-check-corner' }, 'r_j'), ...Array.from({ length: 7 }, (_, i) => ctx.el('span', { class: 'ecc-check-head' }, String(i + 1))), ctx.el('span', { class: 'ecc-check-head' }, 's'));
    H.forEach((row, j) => { grid.append(ctx.el('strong', {}, `s${2 ** j}`), ...row.map((entry, i) => ctx.el('span', { class: entry ? 'ecc-check-member' : 'ecc-check-empty' }, entry ? r[i] : '·')), ctx.el('strong', { class: s[j] ? 'ecc-check-fail' : 'ecc-check-pass' }, s[j])); });
    return ctx.el('div', { class: 'ecc-checks' }, [grid, ctx.el('p', { class: 'ecc-check-key' }, ctx.text(
      'Cells show the received bits r_j used in each check, not the coefficients of H. · means not included; the rightmost s is the XOR of the included bits.',
      '칸의 숫자는 H의 계수가 아니라 각 check에서 사용하는 수신 bit r_j입니다. ·는 그 check에서 제외한 위치입니다. 오른쪽 s는 숫자가 표시된 bit들을 XOR한 결과입니다.'
    ))]);
  }
  labs.register('ecc-hamming', ctx => {
    setup(ctx);
    const mode = select(ctx, 'Decoder', 'Decoder', [{ value: 'hamming', en: 'Hamming (7,4)', ko: 'Hamming (7,4)' }, { value: 'extended', en: 'Extended (8,4): SECDED', ko: 'Extended (8,4): SECDED' }], 'hamming');
    let message = [1, 0, 1, 1], flips = Array(8).fill(false);
    const summary = ctx.metric('Syndrome (s1, s2, s4)', 'Syndrome (s1, s2, s4)'), location = ctx.metric('s1 + 2s2 + 4s4', 's1 + 2s2 + 4s4');
    for (const [en, ko, positions] of [['Clean', '오류 없음', []], ['Single: 5', '한 bit: 5번', [5]], ['Double: 3, 5', '두 bit: 3·5번', [3, 5]], ['Triple: 3, 5, 6', '세 bit: 3·5·6번', [3, 5, 6]]]) ctx.button(en, ko, () => { flips = flips.map((_, i) => positions.includes(i + 1)); render(); });
    function render() {
      const extended = mode.value === 'extended', code = extended ? encodeExtended(message) : encode(message), active = indices(flips).filter(i => i <= code.length), received = applyFlips(code, active), decoded = extended ? decodeExtended(received) : decode(received), s = decoded.syndrome;
      summary.textContent = `(${s.join(', ')})`; location.textContent = String(decoded.index);
      const matched = decoded.data && word(decoded.data, 4) === word(message, 4);
      const nodes = [heading(ctx, 'Message: click to edit', 'Message: 눌러서 바꾸기'), bitRow(ctx, message, { key: 'message', name: ctx.text('Message bits', 'Message bit'), labels: ['d1', 'd2', 'd3', 'd4'], click: i => { message[i] ^= 1; render(); } }), transmittedReadout(ctx, code), heading(ctx, 'Received: click positions to toggle errors', '받은 word: 눌러서 오류 만들기'), bitRow(ctx, received, { key: 'received', name: ctx.text('Received bits with chosen channel errors', '오류를 넣은 수신 bit'), labels: POSITIONS.slice(0, code.length).map((label, i) => `${i + 1} ${label}`), changed: active, click: i => { flips[i] = !flips[i]; render(); } }), heading(ctx, 'Three checks on positions 1–7', '1–7번 bit에 대한 세 가지 check'), checksDiagram(ctx, received)];
      if (extended) {
        nodes.push(fact(ctx, 'Overall parity of all eight received bits', '받은 8개 bit 전체의 parity', decoded.overall));
        const explanations = {
          clean: ['s = 0 and overall parity = 0: accept unchanged under the ≤2-error model.', 's = 0이고 전체 parity도 0입니다. 오류가 최대 두 개라면 오류가 없는 경우이므로 받은 word를 그대로 둡니다.'],
          'single-corrected': [`Odd overall parity and nonzero syndrome: flip position ${decoded.flipped} under the ≤2-error model.`, `전체 parity가 홀수이고 syndrome이 0이 아닙니다. 오류가 최대 두 개라면 한 bit 오류이므로 ${decoded.flipped}번 bit를 뒤집습니다.`],
          'parity-corrected': ['s = 0 and overall parity = 1: flip the eighth parity bit under the ≤2-error model.', 's = 0이고 전체 parity가 1입니다. 오류가 최대 두 개라면 8번 parity bit만 뒤집힌 경우이므로 이 bit를 고칩니다.'],
          'double-detected': ['Nonzero syndrome with even overall parity: detect two errors and refuse correction under the ≤2-error model.', 'Syndrome은 0이 아니지만 전체 parity는 짝수입니다. 오류가 최대 두 개라면 두 bit가 뒤집힌 경우입니다. 위치를 정할 수 없으므로 수정하지 않습니다.']
        };
        nodes.push(text(ctx, ...explanations[decoded.status], decoded.status === 'double-detected' ? 'warning' : ''));
      } else {
        nodes.push(text(ctx, decoded.index ? `The one-error decoder flips position ${decoded.index}. It does not know how many bits actually changed.` : 'Zero syndrome: the decoder changes nothing. This does not prove that no error occurred.', decoded.index ? `한 bit 오류라고 가정하고 ${decoded.index}번 bit를 뒤집습니다. Decoder는 실제로 bit가 몇 개 뒤집혔는지 모릅니다.` : 'Syndrome이 0이므로 받은 word를 그대로 둡니다. 다만 syndrome이 0이라는 사실만으로 오류가 없었다고 단정할 수는 없습니다.'));
      }
      if (decoded.corrected) nodes.push(heading(ctx, 'Decoder output', 'Decoder 출력'), bitRow(ctx, decoded.corrected, { key: 'decoder-output', name: ctx.text('Decoder output bits', 'Decoder가 출력한 bit'), labels: POSITIONS.slice(0, code.length) }), fact(ctx, 'Decoder output message', 'Decoder가 내놓은 message', decoded.data.join('')));
      nodes.push(text(ctx, `Simulation truth: ${active.length} ${active.length === 1 ? 'flip' : 'flips'}. ${decoded.data ? matched ? 'The output message matches the original.' : 'The output message is WRONG.' : 'The decoder returned no message.'}`, `실제로 뒤집힌 bit는 ${active.length}개입니다. ${decoded.data ? matched ? 'Decoder가 내놓은 message가 원본과 같습니다.' : 'Decoder가 내놓은 message가 원본과 다릅니다.' : 'Decoder는 message를 내놓지 않았습니다.'}`, decoded.data && !matched ? 'warning' : decoded.data ? 'success' : ''));
      if (active.length > (extended ? 2 : 1)) nodes.push(text(ctx, 'This pattern exceeds the stated decoder guarantee. A plausible output is not a correctness certificate.', '이 decoder는 이 오류 수에 대해서는 올바른 결과를 보장하지 않습니다. 결과가 codeword라고 해도 원본을 복구했다고 볼 수는 없습니다.', 'warning'));
      ctx.view.replaceChildren(...nodes);
      ctx.status.textContent = ctx.text(extended ? 'SECDED: correct any one-bit error; detect any two-bit error. Three or more errors may be misclassified.' : 'Positions are one-based. Syndrome components are written (s1,s2,s4), not a high-bit-first binary numeral.', extended ? 'SECDED는 한 bit 오류를 고치고 두 bit 오류를 검출합니다. 세 bit 이상이 뒤집히면 잘못 판단할 수 있습니다.' : 'Bit 위치는 1부터 셉니다. Syndrome은 (s1,s2,s4) 순서로 씁니다. 높은 자리부터 적는 보통의 binary 표기와는 순서가 다릅니다.');
    }
    ctx.onChange(render); ctx.onReset(() => { message = [1, 0, 1, 1]; flips = Array(8).fill(false); });
    modelLink(ctx, 'ecc-hamming', () => ({ mode: mode.value, message: message.slice(), flips: flips.slice() }), s => { try { bits(s.message, 4); return Array.isArray(s.message) && ['hamming', 'extended'].includes(s.mode) && validFlags(s.flips, 8); } catch (_) { return false; } }, s => { mode.value = s.mode; message = s.message.slice(); flips = s.flips.slice(); }, render);
  });

  function cube(ctx, received, t) {
    const node = ctx.svg('svg', { viewBox: '0 0 300 270', class: 'ecc-cube', role: 'img', 'aria-label': ctx.text('All eight three-bit words, connected when one bit differs', '세 bit로 만든 여덟 word 중 한 bit만 다른 쌍을 선으로 연결한 cube') });
    node.append(ctx.svg('title', {}, ctx.text('Exact Hamming neighborhoods of codewords 000 and 111', 'Codeword 000과 111에서 Hamming distance가 t 이하인 word들')));
    const xy = n => { const a = (n >> 2) & 1, b = (n >> 1) & 1, c = n & 1; return [50 + 120 * a + 70 * b, 170 - 100 * c + 40 * b]; };
    for (let n = 0; n < 8; n++) for (const mask of [1, 2, 4]) if (n < (n ^ mask)) { const [x1, y1] = xy(n), [x2, y2] = xy(n ^ mask); node.append(ctx.svg('line', { x1, y1, x2, y2, class: 'ecc-cube-edge' })); }
    for (let n = 0; n < 8; n++) {
      const w = n.toString(2).padStart(3, '0'), inA = distance(w, '000') <= t, inB = distance(w, '111') <= t, selected = w === received, [cx, cy] = xy(n);
      const color = inA && inB ? 'both' : inA ? 'a' : inB ? 'b' : 'none';
      node.append(ctx.svg('circle', { cx, cy, r: selected ? 23 : 20, class: `ecc-cube-node ecc-cube-${color}` + (selected ? ' ecc-cube-current' : '') }), ctx.svg('text', { x: cx, y: cy + 5, 'text-anchor': 'middle', class: 'ecc-cube-word' }, w));
      if (n === 0 || n === 7) node.append(ctx.svg('text', { x: cx, y: cy + 37, 'text-anchor': 'middle', class: 'ecc-cube-caption' }, 'codeword'));
    }
    return node;
  }
  labs.register('ecc-distance', ctx => {
    setup(ctx); const bound = select(ctx, 'Allowed flips t', '허용할 오류 수 t', [0, 1, 2], 1); let received = [0, 0, 1];
    const count = ctx.metric('Possible transmitted codewords', '원본으로 가능한 codeword 수');
    function render() {
      const t = Number(bound.value), r = received.join(''), candidates = distanceCandidates(r, t); count.textContent = String(candidates.length);
      ctx.view.replaceChildren(heading(ctx, 'Received word: click to edit', '받은 word: 눌러서 바꾸기'), bitRow(ctx, received, { key: 'received', name: ctx.text('Received three-bit word', '수신한 세 bit'), click: i => { received[i] ^= 1; render(); } }), cube(ctx, r, t), ctx.el('div', { class: 'ecc-legend' }, [ctx.el('span', { class: 'ecc-legend-a' }, ctx.text('Within t of 000', '000에서 거리 ≤ t')), ctx.el('span', { class: 'ecc-legend-b' }, ctx.text('Within t of 111', '111에서 거리 ≤ t')), ctx.el('span', { class: 'ecc-legend-both' }, ctx.text('Both neighborhoods', '두 영역에 모두 포함'))]), ctx.table([ctx.text('Codeword c', 'Codeword c'), ctx.text('Distance to received r', '받은 r과의 거리'), ctx.text('Distance ≤ t?', '거리 ≤ t?')], ['000', '111'].map(c => [c, distance(c, r), candidates.includes(c) ? ctx.text('Yes', '예') : ctx.text('No', '아니요')])), text(ctx, candidates.length === 1 ? `Only ${candidates[0]} could have been sent if at most ${t} ${t === 1 ? 'bit' : 'bits'} changed.` : candidates.length === 2 ? 'Both codewords could have produced this received word within the allowed error bound. Unique correction is impossible for this observation.' : 'Neither codeword is within the allowed error bound. The observation contradicts the assumed bound.', candidates.length === 1 ? `오류가 최대 ${t}개라면 원래 codeword는 ${candidates[0]}뿐입니다.` : candidates.length === 2 ? '두 codeword 모두 최대 t번의 flip으로 이 수신값이 될 수 있습니다. 받은 값만으로는 어느 쪽을 보냈는지 알 수 없습니다.' : '어느 codeword도 최대 t번의 flip으로 이 수신값이 될 수 없습니다. 오류가 최대 t개라는 가정과 맞지 않습니다.', candidates.length !== 1 ? 'warning' : ''));
      ctx.status.textContent = ctx.text('Each edge changes one bit. Graph distance is Hamming distance; the flat drawing is not a Euclidean-distance plot.', 'Edge 하나를 따라가면 bit 하나가 바뀝니다. 두 word를 잇는 가장 짧은 경로의 edge 수가 Hamming distance입니다. 화면에서 잰 직선거리가 아닙니다.');
    }
    ctx.onChange(render); ctx.onReset(() => { received = [0, 0, 1]; });
    modelLink(ctx, 'ecc-distance', () => ({ t: Number(bound.value), received: received.slice() }), s => { try { bits(s.received, 3); return Array.isArray(s.received) && integer(s.t, 0, 2); } catch (_) { return false; } }, s => { bound.value = String(s.t); received = s.received.slice(); }, render);
  });

  labs.register('ecc-quiz', ctx => {
    setup(ctx); let number = 0, checked = false, hint = false;
    const answer = select(ctx, 'Position to correct', '수정할 bit 위치', [1, 2, 3, 4, 5, 6, 7], 1);
    ctx.button('Check answer', '답 확인', () => { checked = true; render(); });
    ctx.button('Show parity checks', 'Parity check 보기', () => { hint = true; render(); });
    ctx.button('Next challenge', '다음 문제', () => { number = (number + 1) % 112; checked = hint = false; answer.value = '1'; render(); });
    function render() {
      const problem = quizProblem(number), chosen = Number(answer.value), correct = chosen === problem.position;
      const candidate = applyFlips(problem.received, [chosen]);
      const nodes = [heading(ctx, `Challenge ${number + 1}: exactly ONE received bit is wrong`, `문제 ${number + 1}: 정확히 한 bit가 뒤집혔습니다`), bitRow(ctx, problem.received, { key: 'quiz-answer', name: ctx.text('Select the bit to correct', '수정할 bit 선택'), labels: POSITIONS.slice(0, 7).map((label, i) => `${i + 1} ${label}`), selected: chosen, selectable: true, click: i => { answer.value = String(i + 1); checked = false; render(); } }), text(ctx, 'Use the parity checks to select the bit to correct. You may click a position or use the selector.', 'Parity check를 계산해 뒤집힌 bit를 찾으세요. 해당 bit를 누르거나 메뉴에서 위치를 고르면 됩니다.')];
      if (hint || checked) nodes.push(checksDiagram(ctx, problem.received));
      if (checked) nodes.push(text(ctx, correct ? `Correct. Flip position ${chosen}; the recovered message is ${problem.message.join('')}.` : `Not yet. Flipping position ${chosen} gives ${candidate.join('')}, which still fails the Hamming checks. Recompute s1 + 2s2 + 4s4.`, correct ? `맞았습니다. ${chosen}번 bit를 뒤집으면 됩니다. 원래 message는 ${problem.message.join('')}입니다.` : `아직 정답이 아닙니다. ${chosen}번 bit를 뒤집어 얻은 값 ${candidate.join('')}은 Hamming check를 통과하지 못합니다. s1 + 2s2 + 4s4를 다시 계산해 보세요.`, correct ? 'success' : 'warning'));
      ctx.view.replaceChildren(...nodes);
      ctx.status.textContent = ctx.text('Every challenge guarantees exactly one error in a Hamming (7,4) codeword. Reset returns to challenge 1 and clears the hint and answer.', '모든 문제에서는 Hamming (7,4) codeword의 정확히 한 bit만 뒤집었습니다. 초기값으로 돌아가면 힌트와 답을 지우고 첫 문제부터 다시 풀 수 있습니다.');
    }
    answer.addEventListener('change', () => { checked = false; });
    ctx.onChange(render); ctx.onReset(() => { number = 0; checked = hint = false; });
    modelLink(ctx, 'ecc-quiz', () => ({ number, chosen: Number(answer.value), checked, hint }), s => integer(s.number, 0, 111) && integer(s.chosen, 1, 7) && typeof s.checked === 'boolean' && typeof s.hint === 'boolean', s => { number = s.number; answer.value = String(s.chosen); checked = s.checked; hint = s.hint; }, render);
  });
}());
