const assert = require('node:assert/strict');
const m = require('../assets/js/article-labs-rsna-viewer.js');
let checks = 0;
for (const plane of Object.keys(m.PLANES)) {
  for (const point of [[0, 0, 0], [48, 48, 61], [96, 96, 128], [68, 55, 61]]) {
    const projected = m.planePosition(plane, point);
    assert.deepEqual(m.planePoint(plane, ...projected, point), point); checks++;
    const r = m.cropRect(plane, point, 60), size = m.PLANES[plane].size;
    assert(r.x >= 0 && r.y >= 0 && r.x + r.width <= size[0] && r.y + r.width <= size[1]); checks++;
  }
  const position = [48, 48, 61], start = [20, 30], end = [60, 70];
  const change = m.cropFromDrag(plane, start, end, position);
  assert.equal(change.width, 40); checks++;
  assert.deepEqual(m.planePosition(plane, change.position), [40, 50]); checks++;
  assert.equal(change.position[m.PLANES[plane].fixed], position[m.PLANES[plane].fixed]); checks++;
  const data = m.sample(plane, position, { x: 0, y: 0, width: 96, height: 96 }, 32);
  assert.equal(data.values.length, 1024); checks++;
  assert(data.values.every(v => Number.isFinite(v) && v >= 0 && v <= 1)); checks++;
}
assert(m.phantomAt([68, 55, 61]) > .95); checks++;
assert.equal(m.phantomAt([-1, 30, 30]), 0); checks++;
const valid = { plane: 'coronal', mode: 'crop', position: [48, 48, 61], width: 60, pixels: 64 };
assert.deepEqual(m.restore(valid), valid); checks++;
assert.notEqual(m.restore(valid).position, valid.position); checks++;
for (const bad of [null, {}, { ...valid, width: 0 }, { ...valid, pixels: 100000 }, { ...valid, position: [100, 0, 0] }, { ...valid, plane: 'patient' }, { ...valid, mode: 'code' }, ...['__proto__', 'constructor', 'toString'].map(plane => ({ ...valid, plane }))]) { assert.throws(() => m.restore(bad)); checks++; }
for (const plane of ['__proto__', 'constructor', 'toString']) { assert.throws(() => m.planePoint(plane, 0, 0, valid.position)); assert.throws(() => m.planePosition(plane, valid.position)); checks += 2; }
console.log(`PASS RSNA linked phantom: ${checks} spatial projection, crop bounds, shared-volume sampling and restore checks`);
