const assert = require('node:assert/strict');
const { parseCSV, format, encodeSharedState, decodeSharedState } = require('../assets/js/article-labs.js');
assert.deepEqual(parseCSV('\uFEFFname,value\r\n"A, B","a""b"\r\nC,"line1\nline2"\r\n'), [{ name: 'A, B', value: 'a"b' }, { name: 'C', value: 'line1\nline2' }]);
assert.deepEqual(parseCSV('a,b\n1,\n'), [{ a: '1', b: '' }]);
assert.deepEqual(parseCSV(''), []);
assert.throws(() => parseCSV('a,b\n1,2,3'), /column/);
assert.throws(() => parseCSV('a,a\n1,2'), /Duplicate/);
assert.throws(() => parseCSV('a,b\n"bad,2'), /Unclosed/);
assert.equal(format(NaN), '—'); assert.equal(format(0.0000001), '1.00e-7');
const settings = { input: '위상 → phase', gates: ['H', 'Z'], cursor: 0 };
const encoded = encodeSharedState('quantum-circuit-puzzle', settings);
assert.match(encoded, /^[A-Za-z0-9_-]+$/);
assert.deepEqual(decodeSharedState(encoded), { v: 1, key: 'quantum-circuit-puzzle', state: settings });
assert.throws(() => encodeSharedState('../other', {}), /key/);
assert.throws(() => encodeSharedState(undefined, {}), /key/);
assert.throws(() => encodeSharedState('test', { text: 'x'.repeat(6001) }), /long/);
assert.throws(() => decodeSharedState('!bad'), /settings/);
assert.throws(() => decodeSharedState('x'.repeat(8001)), /settings/);
assert.throws(() => decodeSharedState(btoa(JSON.stringify({ v: 2, key: 'test', state: {} }))), /settings/);
assert.throws(() => decodeSharedState(btoa(JSON.stringify({ v: 1, key: 'test' }))), /settings/);
assert.throws(() => decodeSharedState(btoa(JSON.stringify({ v: 1, state: {} }))), /settings/);
console.log('PASS runtime CSV, formatting and bounded UTF-8 experiment-link round trips/rejection');


// Reuse the same deliberately small DOM contract as the full model harness.
// Native rendering/touch geometry is checked separately in the browser.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const harness = fs.readFileSync(path.join(__dirname, 'check_article_labs_models.js'), 'utf8');
const classes = harness.slice(harness.indexOf('class StubEvent'), harness.indexOf('function walk(dir)'));
assert.ok(classes.startsWith('class StubEvent') && classes.includes('class StubNode'), 'DOM contract classes are available');
async function checkpointTests() {
  const sandbox = { console, URL, TextEncoder, TextDecoder, btoa, atob, setTimeout, clearTimeout };
  const context = vm.createContext(sandbox);
  vm.runInContext(classes + '\nthis.Node = StubNode; this.Event = StubEvent; this.StubNode = StubNode;', context);
  const StubNode = context.StubNode;
  const originalQuery = StubNode.prototype.querySelectorAll;
  StubNode.prototype.querySelectorAll = function(selector) {
    if (selector.includes(',')) { const matches = new Set(selector.split(',').flatMap(item => originalQuery.call(this, item.trim()))); return originalQuery.call(this, '*').filter(node => matches.has(node)); }
    return originalQuery.call(this, selector);
  };
  StubNode.prototype.hasAttribute = function(name) { return this.getAttribute(name) !== null; };
  StubNode.prototype.open = false;
  const ids = ['quantum-circuit-puzzle', 'kaggriculture-farm-game', 'security-trust-puzzle', 'gemma-test-challenge', 'rsna-linked-viewer', 'generic-controls', 'checkpoint-history'];
  const roots = ids.map(id => {
    const root = new StubNode('section', { id: 'lab-' + id, 'data-article-lab': id, 'data-language': 'en', 'data-preset': '', 'data-asset-version': 'bundle-abc' });
    for (const [tag, attr] of [['fieldset', 'controls'], ['div', 'view'], ['dl', 'metrics'], ['p', 'status'], ['button', 'reset'], ['p', 'note']]) root.append(new StubNode(tag, { ['data-lab-' + attr]: '' }));
    return root;
  });
  const document = {
    readyState: 'loading',
    querySelectorAll: selector => selector === '[data-article-lab]' ? roots : [],
    createElement: tag => new StubNode(tag),
    createElementNS: (_ns, tag) => new StubNode(tag),
    createTextNode: text => String(text)
  };
  Object.assign(sandbox, { document, location: { href: 'http://127.0.0.1/article/', hash: '' }, navigator: { clipboard: { writeText: async () => {} } }, addEventListener() {} });
  sandbox.window = sandbox;
  const runtime = fs.readFileSync(path.join(__dirname, '../assets/js/article-labs.js'), 'utf8');
  vm.runInContext(runtime, context);
  for (const file of ['quantum-workbench', 'farm-game', 'agent-challenges', 'rsna-viewer']) vm.runInContext(fs.readFileSync(path.join(__dirname, '../assets/js/article-labs-' + file + '.js'), 'utf8'), context);
  sandbox.ArticleLabs.register('generic-controls', ctx => {
    const option = ctx.control('Preset', 'Preset', { type: 'select', value: 'a', options: [{ value: 'a' }, { value: 'b' }] });
    const value = ctx.control('Value', 'Value', { type: 'number', min: 0, max: 20, step: 1, value: 2 });
    const enabled = ctx.control('Enabled', 'Enabled', { type: 'checkbox', value: false });
    option.addEventListener('change', () => { value.value = '3'; });
    ctx.onChange(() => { ctx.root.dataset.result = option.value + ':' + value.value + ':' + enabled.checked; });
  });
  sandbox.ArticleLabs.register('checkpoint-history', ctx => {
    let actions = [];
    const paint = () => { ctx.root.dataset.history = actions.join(','); };
    ctx.button('Take a step', 'Take a step', () => { actions.push(actions.length + 1); paint(); });
    ctx.onChange(paint);
    ctx.checkpointState({
      read: () => ({ actions: actions.slice() }),
      write: value => {
        if (!value || !Array.isArray(value.actions) || value.actions.length > 20 || value.actions.some((action, index) => action !== index + 1)) return false;
        actions = value.actions.slice(); paint();
      }
    });
  });
  await sandbox.ArticleLabs.restoreCheckpoint({ v: 1, labs: [] });
  const api = sandbox.ArticleLabs;
  const plain = value => JSON.parse(JSON.stringify(value));
  const initial = plain(api.checkpoint());
  assert.equal(initial.labs.length, ids.length, 'Every ready root is captured');
  const change = (node, value) => { node.value = value; node.dispatchEvent(new context.Event('input', { bubbles: true })); };
  const press = (root, text) => {
    const button = root.querySelectorAll('button').find(node => node.textContent === text);
    assert.ok(button, 'Button exists: ' + text); button.dispatchEvent(new context.Event('click'));
  };
  const [quantum, farm, security, gemma, rsna, generic, history] = roots;
  press(history, 'Take a step'); press(history, 'Take a step');
  assert.equal(history.querySelectorAll('.al-share').length, 0, 'Checkpoint-only provider adds no share UI');
  press(quantum, 'Insert before slot');
  quantum.querySelector('details').open = true;
  farm.querySelectorAll('.fg-tile')[1].dispatchEvent(new context.Event('click'));
  press(farm, 'Plant'); press(farm, 'End day →'); press(farm, 'Undo action'); press(farm, 'End day →');
  farm.querySelector('.fg-ledger').open = true;
  const choices = security.querySelectorAll('select');
  change(choices[1], 'external'); change(choices[2], choices[2].options[1].value); press(security, 'Check decision');
  const input = gemma.querySelector('textarea');
  change(input, '{"value":0}'); change(gemma.querySelector('input.ac-input'), '0'); press(gemma, 'Run your test');
  change(input, '{unfinished json'); change(gemma.querySelector('input.ac-input'), '-');
  rsna.querySelector('.rv-slice').dispatchEvent(new context.Event('keydown', { key: 'ArrowRight' }));
  rsna.querySelector('details').open = true;
  const regular = generic.querySelectorAll('input, select, textarea');
  change(regular[0], 'b'); change(regular[1], '9'); regular[2].checked = true; regular[2].dispatchEvent(new context.Event('input', { bubbles: true }));
  // An unfinished number keeps the last successfully rendered value as well as
  // the empty draft; restoration must never coerce that draft to numeric zero.
  change(quantum.querySelector('input[type="number"]'), '');
  change(regular[1], '');
  const saved = plain(api.checkpoint());
  assert(saved.labs.every(item => !item.controls.some(field => field.value.includes('labstate='))), 'Readonly share links are omitted');
  assert.notDeepEqual(saved.labs[0].model.circuit, initial.labs[0].model.circuit, 'Edited circuit is captured');
  assert.ok(saved.labs[1].model.actions.length && saved.labs[1].model.reference.length, 'Farm history and comparison branch are captured');
  assert.equal(saved.labs[2].model.checked, true, 'Security feedback state is captured');
  assert.equal(saved.labs[3].model.suite.length, 1, 'Saved Gemma tests are captured');
  assert.equal(saved.labs[3].model.input, '{unfinished json', 'Unparsed JSON draft remains plain text');
  assert.equal(saved.labs[4].model.position[0], 49, 'Linked crosshair state is captured');
  assert.equal(saved.labs[5].applied[1].value, '9', 'Last complete numeric render is recorded');
  // Simulate a new page opening the older link, then restore the newer local
  // checkpoint. The old URL must not win over current experiment progress.
  sandbox.location.href = 'http://127.0.0.1/article/?labstate=' + encodeSharedState('quantum-circuit-puzzle', initial.labs[0].model);
  await api.restoreCheckpoint(initial);
  const restored = plain(await api.restoreCheckpoint(saved));
  assert.equal(restored.failures.length, 0, JSON.stringify(restored.failures));
  assert.equal(restored.restored.length, ids.length);
  assert.deepEqual(plain(api.checkpoint()), saved, 'Full workshop state, raw drafts, ordinary controls and open details round-trip');
  assert.equal(history.dataset.history, '1,2', 'Checkpoint-only provider restores hidden history');
  assert.equal(generic.dataset.result, 'b:9:true', 'Invalid numeric draft does not replace the last valid model result');
  assert.equal(generic.querySelector('input[type="number"]').value, '', 'Empty number draft is restored after rendering');
  for (const mutate of [
    value => { value.v = 2; },
    value => { value.labs[0].assetVersion = 'another-build'; },
    value => { value.labs[0].key = 'another-experiment'; },
    value => { value.labs[0].controls[0].value = 'unknown-option'; },
    value => { value.labs[0].applied[2].value = '-1'; },
    value => { value.labs[0].model.circuit = [{ op: 'UNSAFE', target: 0, controls: [] }]; },
    value => { value.labs[0].details[0].open = 'yes'; },
    value => { value.labs[6].model.actions = [99]; }
  ]) {
    const invalid = plain(saved); mutate(invalid);
    const before = plain(api.checkpoint());
    const rejected = await api.restoreCheckpoint(invalid);
    assert(rejected.failures.length > 0, 'Malformed settings must be rejected');
    assert.deepEqual(plain(api.checkpoint()), before, 'Rejected settings leave the original experiment intact');
  }
  const tooLarge = plain(saved); tooLarge.labs[0].controls[0].value = 'x'.repeat(262145);
  assert.equal((await api.restoreCheckpoint(tooLarge)).restored.length, 0, 'Oversized checkpoint does not mutate experiments');
  // The factory can still be loading when the restore request arrives.
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const slow = new StubNode('section', { id: 'lab-slow', 'data-article-lab': 'slow', 'data-language': 'en', 'data-preset': '', 'data-asset-version': 'bundle-abc' });
  for (const [tag, attr] of [['fieldset', 'controls'], ['div', 'view'], ['dl', 'metrics'], ['p', 'status'], ['button', 'reset'], ['p', 'note']]) slow.append(new StubNode(tag, { ['data-lab-' + attr]: '' }));
  roots.push(slow);
  api.register('slow', async ctx => { await gate; const value = ctx.control('Value', 'Value', { value: 1, min: 0, max: 9 }); ctx.onChange(() => { ctx.root.dataset.result = value.value; }); });
  let complete = false;
  const slowSaved = { v: 1, labs: [{ id: 'lab-slow', key: 'slow', preset: '', language: 'en', assetVersion: 'bundle-abc', controls: [{ tag: 'input', type: 'range', value: '7', checked: false }], applied: [{ tag: 'input', type: 'range', value: '7', checked: false }], details: [] }] };
  const loading = api.restoreCheckpoint(slowSaved).then(value => { complete = true; return value; });
  await Promise.resolve(); assert.equal(complete, false, 'Restoration waits for pending factory initialization');
  release(); assert.equal((await loading).failures.length, 0); assert.equal(slow.dataset.result, '7');
  const hung = new StubNode('section', { id: 'lab-hung', 'data-article-lab': 'hung', 'data-language': 'en', 'data-preset': '', 'data-asset-version': 'bundle-abc' });
  for (const [tag, attr] of [['fieldset', 'controls'], ['div', 'view'], ['dl', 'metrics'], ['p', 'status'], ['button', 'reset'], ['p', 'note']]) hung.append(new StubNode(tag, { ['data-lab-' + attr]: '' }));
  roots.push(hung); api.register('hung', async () => { await new Promise(() => {}); });
  sandbox.setTimeout = (callback, delay) => { assert(delay <= 15000, 'Pending restoration has a 15-second upper bound'); queueMicrotask(callback); return 0; };
  sandbox.clearTimeout = () => {};
  const hungSaved = plain(slowSaved); hungSaved.labs[0].id = 'lab-hung'; hungSaved.labs[0].key = 'hung';
  const timedOut = await api.restoreCheckpoint(hungSaved);
  assert.equal(timedOut.failures.length, 1, 'A hung factory reports a failure instead of blocking indefinitely');
  assert.equal(timedOut.restored.length, 0);
  assert.equal(api.checkpoint().labs.some(item => item.id === 'lab-hung'), false, 'Loading roots are not captured as ready');
  for (const order of ['checkpoint-first', 'share-first']) {
    const duplicate = new StubNode('section', { id: 'lab-' + order, 'data-article-lab': order, 'data-language': 'en', 'data-preset': '', 'data-asset-version': 'bundle-abc' });
    for (const [tag, attr] of [['fieldset', 'controls'], ['div', 'view'], ['dl', 'metrics'], ['p', 'status'], ['button', 'reset'], ['p', 'note']]) duplicate.append(new StubNode(tag, { ['data-lab-' + attr]: '' }));
    roots.push(duplicate);
    const reported = [];
    sandbox.console = { ...console, error: (...args) => reported.push(args) };
    api.register(order, ctx => {
      const provider = { read: () => ({}), write: () => {} };
      if (order === 'checkpoint-first') { ctx.checkpointState(provider); ctx.stateLink(order, provider); }
      else { ctx.stateLink(order, provider); ctx.checkpointState(provider); }
    });
    await Promise.resolve(); await Promise.resolve();
    assert.equal(duplicate.dataset.ready, 'error', 'A second state provider must be rejected');
    assert.equal(reported.length, 1, 'A duplicate state provider reports one initialization error');
  }
  console.log('PASS checkpoints: five real workshops, generic controls, invalid drafts, stale link override, atomic rejection, details, pending factories bounded loading timeout and checkpoint-only hidden history');
}
checkpointTests().catch(error => { console.error(error); process.exitCode = 1; });
