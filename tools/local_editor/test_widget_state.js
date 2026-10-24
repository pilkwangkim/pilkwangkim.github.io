#!/usr/bin/env node
'use strict';

// Exercise the actual preview reload bridge without writing an article.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const client = fs.readFileSync(path.join(__dirname, 'client.js'), 'utf8');
const functions = ['saveWidgetCheckpoint', 'restoreWidgetCheckpoint'].map(name => {
  const match = client.match(new RegExp(`  (?:async )?function ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n  \\}`));
  assert.ok(match, `Production ${name} exists`);
  return match[0];
});
const storage = new Map();
let restoredLabs, restoredNEGF, restoreFailures = [];
const experiment = { v: 1, labs: [{ key: 'farm', model: { cursor: 2, history: [{ type: 'plant' }] } }] };
const exactNEGF = { v: 1, height: 0.4, width: 16, energy: 0.1413123456789 };
const root = { dispatchEvent(event) {
  if (event.type === 'article-preview:capture') event.detail.state = exactNEGF;
  else { restoredNEGF = event.detail.state; event.detail.restored = true; }
} };
const context = vm.createContext({
  JSON, Date, Number,
  snapshot: { revision: 'current-source' },
  location: { pathname: '/posts/Fixture/', search: '?labstate=old-url-state' },
  widgetCheckpointKey: 'checkpoint',
  CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
  sessionStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) },
  document: { querySelectorAll: () => [root] },
  window: { ArticleLabs: { checkpoint: () => experiment, restoreCheckpoint: async data => { restoredLabs = data; return { failures: restoreFailures }; } } }
});
vm.runInContext(functions.join('\n'), context);
let checks = 0;
async function restore(expected, label) {
  assert.equal(await context.restoreWidgetCheckpoint(), expected, label);
  assert.equal(storage.size, 0, `${label}: checkpoint is consumed once`);
  checks += 1;
}

(async () => {
  context.saveWidgetCheckpoint();
  await restore(false, 'Save/undo/refresh restore full experiment state despite an older share URL');
  assert.deepEqual(restoredLabs, experiment);
  assert.deepEqual(restoredNEGF, exactNEGF, 'Exact resonance energy must not round to a slider step');
  restoredLabs = undefined;
  await restore(false, 'An ordinary reload cannot replay the consumed checkpoint');
  assert.equal(restoredLabs, undefined);
  context.saveWidgetCheckpoint('externally-updated-source');
  context.snapshot.revision = 'externally-updated-source';
  await restore(false, 'External-source refresh checkpoints the rebuilt revision');
  assert.deepEqual(restoredLabs, experiment);
  context.snapshot.revision = 'current-source'; restoredLabs = undefined;
  for (const [field, value] of [['v', 2], ['revision', 'different-source'], ['route', '/posts/Other/'], ['created', Date.now() - 600001], ['created', Date.now() + 60000], ['negf', {}]]) {
    context.saveWidgetCheckpoint();
    const data = JSON.parse(storage.get('checkpoint')); data[field] = value;
    storage.set('checkpoint', JSON.stringify(data));
    await restore(false, `${field}: stale or mismatched checkpoint is ignored`);
    assert.equal(restoredLabs, undefined);
  }
  storage.set('checkpoint', '{');
  await restore(true, 'Malformed local state is discarded and reported');
  storage.set('checkpoint', 'x'.repeat(256001));
  await restore(true, 'Oversized local state is bounded');
  context.saveWidgetCheckpoint(); restoreFailures = [{ id: 'farm', reason: 'changed model version' }];
  await restore(true, 'A changed component reports incomplete restoration');
  context.saveWidgetCheckpoint();
  context.window.ArticleLabs = undefined;
  await restore(true, 'Removed components cannot crash editor initialization');

  const negf = fs.readFileSync(path.join(__dirname, '../../assets/js/negf-interactive.js'), 'utf8');
  const handlers = [...negf.matchAll(/root\.addEventListener\('(article-preview:(?:capture|restore))', event => \{([\s\S]*?)\n    \}\);/g)];
  assert.equal(handlers.length, 2, 'Exercise both production NEGF handlers');
  const callbacks = {};
  const bridge = vm.createContext({
    root: { dataset: { ready: 'true' }, addEventListener: (name, callback) => { callbacks[name] = callback; } },
    controls: { height: { value: '0.4', min: '0', max: '0.65' }, width: { value: '16', min: '6', max: '30' }, energy: { value: '0.1413' } },
    energy: exactNEGF.energy, limits: { energyMinEV: 0.001, energyMaxEV: 0.6 }, frame: 0,
    window: { cancelAnimationFrame() {} }, update() {}
  });
  handlers.forEach(match => vm.runInContext(`root.addEventListener('${match[1]}', event => {${match[2]}\n});`, bridge));
  const capture = { detail: {} }; callbacks['article-preview:capture'](capture);
  assert.equal(capture.detail.state.energy, exactNEGF.energy);
  bridge.energy = 0.2;
  const valid = { detail: { state: capture.detail.state, restored: false } };
  callbacks['article-preview:restore'](valid);
  assert.equal(valid.detail.restored, true);
  assert.equal(bridge.energy, exactNEGF.energy);
  checks += 2;
  for (const change of [{ v: 2 }, { height: -1 }, { width: 6.5 }, { width: 31 }, { energy: NaN }, { energy: 0.9 }]) {
    const invalid = { detail: { state: { ...exactNEGF, ...change }, restored: false } };
    callbacks['article-preview:restore'](invalid);
    assert.equal(invalid.detail.restored, false);
    assert.equal(bridge.energy, exactNEGF.energy, 'Rejected NEGF state must not mutate a valid probe');
    checks += 1;
  }
  console.log(`Editor/widget checkpoint: ${checks} behavior checks passed.`);
})().catch(error => { console.error(error); process.exitCode = 1; });
