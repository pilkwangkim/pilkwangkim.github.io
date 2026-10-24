#!/usr/bin/env node
'use strict';

// Exercise the production draft/unload logic without a browser or changing a post.
// Run: node tools/local_editor/test_client_state.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, 'client.js'), 'utf8');
const structureValue = source.match(/  const structureValue = [^\n]+/)?.[0];
const dirtyCount = source.match(/  const dirtyCount = [^\n]+/)?.[0];
const pendingFunction = source.match(/  function hasUnappliedStructureChanges\(\) \{[\s\S]*?\n  \}/)?.[0];
const unloadBody = source.match(/window\.addEventListener\('beforeunload', event => \{([\s\S]*?)\n  \}\);/)?.[1];
assert.ok(structureValue && dirtyCount && pendingFunction && unloadBody, 'The production pending-draft/unload handlers must be exercised.');

const context = vm.createContext({
  currentStructure: undefined,
  structureDialog: { open: false },
  structuredChanges: new Map(),
  changes: new Map(),
  metadataChanges: new Map()
});
vm.runInContext(`
  ${structureValue}
  ${dirtyCount}
  ${pendingFunction}
  function unload(event) { ${unloadBody} }
`, context);

let checks = 0;
function expectsWarning(expected, label) {
  const event = { prevented: false, preventDefault() { this.prevented = true; } };
  context.unload(event);
  assert.equal(event.prevented, expected, label);
  assert.equal(Object.hasOwn(event, 'returnValue'), expected, `${label}: browser warning flag`);
  checks += 1;
}
function equation(value = 'x+1') {
  context.currentStructure = { type: 'math', item: { id: 'math-1', latex: 'x+1' }, input: { value } };
  context.structureDialog.open = true;
}
function table(value = '**2**') {
  context.currentStructure = { type: 'table', inputs: new Map([['cell-1', { value }], ['cell-2', { value: '$x$' }]]), originals: new Map([['cell-1', '**2**'], ['cell-2', '$x$']]) };
  context.structureDialog.open = true;
}

expectsWarning(false, 'A clean article can navigate away');
equation();
expectsWarning(false, 'Opening an unchanged equation is clean');
equation('x+2');
expectsWarning(true, 'Unapplied inline/display LaTeX is protected');
equation('\\frac{');
expectsWarning(true, 'An invalid preview must still protect the user’s typed draft');
equation();
expectsWarning(false, 'Returning LaTeX to its current draft clears the pending warning');
table();
expectsWarning(false, 'Opening an unchanged table is clean');
table('**3**');
expectsWarning(true, 'Unapplied cell Markdown is protected');
table();
context.currentStructure.inputs.get('cell-2').value = '$x^2$';
expectsWarning(true, 'Unapplied math inside a table cell is protected');
context.structureDialog.open = false;
expectsWarning(false, 'Closing/canceling the editor intentionally discards the unapplied draft');
context.structureDialog.open = true;
context.currentStructure = undefined;
expectsWarning(false, 'The item chooser has no pending draft');
context.changes.set('text-1', { text: 'Updated text' });
expectsWarning(true, 'Text drafts remain protected');
context.changes.clear();
context.metadataChanges.set('tags', ['new-tag']);
expectsWarning(true, 'Metadata drafts remain protected');
context.metadataChanges.clear();
context.structuredChanges.set('math-1', { text: 'x+2' });
equation('x+2');
assert.equal(context.hasUnappliedStructureChanges(), false, 'A reopened saved-to-draft equation compares against its staged value');
expectsWarning(true, 'Applied structure drafts remain protected before Save changes');
context.structuredChanges.clear();
context.structuredChanges.set('cell-1', { text: '**3**' });
table('**3**');
assert.equal(context.hasUnappliedStructureChanges(), false, 'A reopened table compares against its staged cell values');
context.currentStructure.inputs.get('cell-1').value = '**4**';
assert.equal(context.hasUnappliedStructureChanges(), true, 'Changes to an already staged cell remain pending');
expectsWarning(true, 'Changes to a reopened structure also retain the warning');

console.log(`Client draft state: ${checks} behavioral checks passed.`);
