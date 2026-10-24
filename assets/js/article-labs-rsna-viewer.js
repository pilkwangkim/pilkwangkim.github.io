/* Analytic synthetic volume: linked spatial views, not patient imagery. */
(function () {
  'use strict';
  const SIZE = [96, 96, 128];
  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
  const PLANES = {
    axial: { axes: [0, 1], fixed: 2, size: [96, 96], en: 'Axial · x–y', ko: 'Axial · x–y' },
    coronal: { axes: [0, 2], fixed: 1, size: [96, 128], en: 'Coronal · x–z', ko: 'Coronal · x–z' },
    sagittal: { axes: [1, 2], fixed: 0, size: [96, 128], en: 'Sagittal · y–z', ko: 'Sagittal · y–z' }
  };
  const STRUCTURES = [
    { c: [48, 48, 107], r: [14, 17, 40], grey: .68 },
    { c: [31, 49, 76], r: [16, 23, 17], grey: .72 },
    { c: [65, 49, 76], r: [16, 23, 17], grey: .72 },
    { c: [48, 48, 43], r: [32, 23, 12], grey: .62 },
    { c: [48, 48, 17], r: [17, 17, 31], grey: .63 },
    { c: [48, 20, 74], r: [11, 7, 15], grey: .83 },
    { c: [31, 49, 59], r: [16, 21, 2.2], grey: .91 },
    { c: [65, 49, 59], r: [16, 21, 2.2], grey: .91 }
  ];
  function phantomAt(point) {
    if (!Array.isArray(point) || point.length !== 3 || point.some((v, i) => !Number.isFinite(v) || v < 0 || v > SIZE[i])) return 0;
    const q = point.reduce((s, v, i) => s + ((v - [48, 48, 65][i]) / [39, 36, 77][i]) ** 2, 0);
    if (q > 1) return .025;
    let value = .16 + .06 * (1 - q) + .018 * Math.sin(point[0] * .6 + point[1] * .3 + point[2] * .2);
    for (const shape of STRUCTURES) {
      const d = point.reduce((s, v, i) => s + ((v - shape.c[i]) / shape.r[i]) ** 2, 0);
      if (d <= 1) value = Math.max(value, shape.grey * (.74 + .26 * Math.sqrt(1 - d)));
    }
    // A small bright geometric target makes loss of context visible.
    if (point.reduce((s, v, i) => s + (v - [68, 55, 61][i]) ** 2, 0) <= 16) value = .98;
    return clamp(value, 0, 1);
  }
  function planePoint(plane, u, v, position) {
    if (!Object.hasOwn(PLANES, plane)) throw new Error('Unknown plane');
    const p = PLANES[plane];
    const out = position.slice(); out[p.axes[0]] = clamp(u, 0, p.size[0]); out[p.axes[1]] = clamp(p.size[1] - v, 0, p.size[1]);
    return out;
  }
  function planePosition(plane, position) {
    if (!Object.hasOwn(PLANES, plane)) throw new Error('Unknown plane');
    const p = PLANES[plane]; return [position[p.axes[0]], p.size[1] - position[p.axes[1]]];
  }
  function cropRect(plane, position, width) {
    const p = PLANES[plane], center = planePosition(plane, position);
    width = clamp(width, 24, Math.min(...p.size));
    return { x: clamp(center[0] - width / 2, 0, p.size[0] - width), y: clamp(center[1] - width / 2, 0, p.size[1] - width), width };
  }
  function cropFromDrag(plane, start, end, position) {
    const p = PLANES[plane];
    const width = Math.round(clamp(Math.max(Math.abs(end[0] - start[0]), Math.abs(end[1] - start[1])), 24, Math.min(...p.size)));
    return { position: planePoint(plane, (start[0] + end[0]) / 2, (start[1] + end[1]) / 2, position).map(Math.round), width };
  }
  function sample(plane, position, region, pixels) {
    const height = Math.max(1, Math.round(pixels * region.height / region.width));
    const values = [];
    for (let y = 0; y < height; y++) for (let x = 0; x < pixels; x++) values.push(phantomAt(planePoint(plane, region.x + (x + .5) * region.width / pixels, region.y + (y + .5) * region.height / height, position)));
    return { values, width: pixels, height };
  }
  function restore(value) {
    if (!value || !Object.hasOwn(PLANES, value.plane) || !['crosshair', 'crop'].includes(value.mode) || !Array.isArray(value.position) || value.position.length !== 3 || value.position.some((v, i) => !Number.isInteger(v) || v < 0 || v > SIZE[i]) || !Number.isInteger(value.width) || value.width < 24 || value.width > 96 || ![32, 64, 96].includes(value.pixels)) throw new Error('Invalid viewer settings');
    return { plane: value.plane, mode: value.mode, position: value.position.slice(), width: value.width, pixels: value.pixels };
  }
  const api = { SIZE, PLANES, phantomAt, planePoint, planePosition, cropRect, cropFromDrag, sample, restore };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined' || !window.ArticleLabs) return;
  window.ArticleLabs.register('rsna-linked-viewer', ctx => {
    ctx.root.classList.add('al-rsna-viewer');
    const initial = () => ({ plane: 'coronal', mode: 'crosshair', position: [48, 48, 61], width: 60, pixels: 64 });
    let state = initial(), gesture = null;
    const plane = ctx.control('Active view', '선택한 단면', { type: 'select', value: state.plane, options: Object.entries(PLANES).map(([value, p]) => ({ value, en: p.en, ko: p.ko })) });
    const mode = ctx.control('Pointer action', '화면 조작', { type: 'select', value: 'crosshair', options: [{ value: 'crosshair', en: 'Move crosshair', ko: '위치 선택' }, { value: 'crop', en: 'Draw square crop', ko: '정사각 crop 그리기' }] });
    const coords = ['x', 'y', 'z'].map((axis, i) => ctx.control(`${axis} position (mm)`, `${axis} 위치 (mm)`, { type: 'range', min: 0, max: SIZE[i], step: 1, value: state.position[i] }));
    const width = ctx.control('Crop width (mm)', 'Crop 폭 (mm)', { type: 'range', min: 24, max: 96, step: 1, value: 60 });
    const pixels = ctx.control('Crop output', 'Crop 출력', { type: 'select', value: '64', options: [32, 64, 96].map(n => ({ value: String(n), en: `${n} × ${n}`, ko: `${n} × ${n}` })) });
    const coordinates = ctx.el('details', { class: 'rv-coordinates' }, ctx.el('summary', {}, ctx.text('Position coordinates (mm)', '위치 좌표 (mm)')));
    coords.forEach(input => coordinates.append(input.closest('.al-control'))); ctx.ui.append(coordinates);
    const pitch = ctx.metric('Crop sampling pitch', 'Crop sampling 간격'), intensity = ctx.metric('Shared point intensity', '공통 위치의 밝기');
    const provenance = ctx.el('p', { class: 'rv-provenance' }, ctx.text('Synthetic knee phantom · the three planes sample one volume.', '합성 무릎 모형 · 세 단면은 같은 3차원 모형을 자릅니다.'));
    const panels = ctx.el('div', { class: 'rv-planes' });
    const tiles = {};
    for (const [key, p] of Object.entries(PLANES)) {
      const svg = ctx.svg('svg', { viewBox: `0 0 ${p.size[0]} ${p.size[1]}`, tabindex: '0', role: 'img', 'aria-label': ctx.text(`${p.en}. Click to move; arrow keys move the crosshair.`, `${p.ko}. 눌러 위치를 고르고 방향키로 이동하세요.`), class: 'rv-slice' });
      const tile = ctx.el('div', { class: 'rv-tile', 'data-plane': key }, [ctx.el('div', { class: 'rv-plane-label' }, ctx.text(p.en, p.ko)), svg]);
      panels.append(tile); tiles[key] = { tile, svg };
      const point = event => { const box = svg.getBoundingClientRect(); return [clamp((event.clientX - box.left) / box.width * p.size[0], 0, p.size[0]), clamp((event.clientY - box.top) / box.height * p.size[1], 0, p.size[1])]; };
      svg.addEventListener('pointerdown', event => {
        if (event.button !== 0 || gesture) return;
        event.preventDefault(); svg.focus(); svg.setPointerCapture(event.pointerId);
        state.plane = key; plane.value = key; gesture = { plane: key, pointerId: event.pointerId, start: point(event), last: point(event) };
        if (mode.value === 'crosshair') state.position = planePoint(key, ...gesture.start, state.position).map(Math.round);
        sync(); render();
      });
      svg.addEventListener('pointermove', event => {
        if (!gesture || gesture.plane !== key || gesture.pointerId !== event.pointerId) return;
        gesture.last = point(event);
        if (mode.value === 'crosshair') state.position = planePoint(key, ...gesture.last, state.position).map(Math.round);
        else { const next = cropFromDrag(key, gesture.start, gesture.last, state.position); state.position = next.position; state.width = next.width; }
        sync(); render();
      });
      const finish = event => { if (gesture && gesture.pointerId === event.pointerId) gesture = null; };
      svg.addEventListener('pointerup', finish); svg.addEventListener('pointercancel', finish); svg.addEventListener('lostpointercapture', finish);
      svg.addEventListener('keydown', event => {
        const delta = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
        if (!delta) return;
        event.preventDefault(); const at = planePosition(key, state.position);
        state.plane = key; plane.value = key; state.position = planePoint(key, at[0] + delta[0], at[1] + delta[1], state.position).map(Math.round); sync(); render();
      });
    }
    const cropped = ctx.svg('svg', { viewBox: '0 0 96 96', class: 'rv-cropped', role: 'img', 'aria-label': ctx.text('Resampled square crop from the active plane', '선택한 단면에서 다시 sampling한 정사각 crop') });
    const details = ctx.el('p');
    const cropPanel = ctx.el('div', { class: 'rv-crop-panel' }, [cropped, ctx.el('div', {}, [ctx.el('strong', {}, ctx.text('Crop → resize', 'Crop → resize')), details])]);
    ctx.view.append(provenance, panels, cropPanel);
    function sync() {
      plane.value = state.plane; mode.value = state.mode; width.value = String(state.width); pixels.value = String(state.pixels);
      coords.forEach((input, i) => { input.value = String(state.position[i]); });
      [plane, mode, width, pixels, ...coords].forEach(input => input.dispatchEvent(new Event('input', { bubbles: false })));
    }
    function paint(svg, values, region) {
      const paths = Array.from({ length: 16 }, () => []);
      const dx = region.width / values.width, dy = region.height / values.height;
      values.values.forEach((v, i) => {
        const x = region.x + (i % values.width) * dx, y = region.y + Math.floor(i / values.width) * dy;
        paths[clamp(Math.round(v * 15), 0, 15)].push(`M${x.toFixed(2)} ${y.toFixed(2)}h${(dx + .02).toFixed(2)}v${(dy + .02).toFixed(2)}h-${(dx + .02).toFixed(2)}z`);
      });
      svg.replaceChildren(...paths.map((parts, i) => ctx.svg('path', { d: parts.join(''), fill: `rgb(${i * 17},${i * 17},${i * 17})` })));
    }
    function render() {
      state = restore({ plane: plane.value, mode: mode.value, position: coords.map(input => Number(input.value)), width: Number(width.value), pixels: Number(pixels.value) });
      for (const [key, p] of Object.entries(PLANES)) {
        const { tile, svg } = tiles[key]; tile.dataset.active = String(key === state.plane);
        const region = { x: 0, y: 0, width: p.size[0], height: p.size[1] };
        paint(svg, sample(key, state.position, region, 40), region);
        const at = planePosition(key, state.position);
        svg.append(ctx.svg('path', { d: `M0 ${at[1]}H${p.size[0]}M${at[0]} 0V${p.size[1]}`, class: 'rv-crosshair' }));
        if (key === state.plane) { const r = cropRect(key, state.position, state.width); svg.append(ctx.svg('rect', { x: r.x, y: r.y, width: r.width, height: r.width, class: 'rv-crop-box' })); }
      }
      const rect = cropRect(state.plane, state.position, state.width), region = { ...rect, height: rect.width };
      cropped.setAttribute('viewBox', `${rect.x} ${rect.y} ${rect.width} ${rect.width}`);
      paint(cropped, sample(state.plane, state.position, region, state.pixels), region);
      const at = planePosition(state.plane, state.position);
      cropped.append(ctx.svg('path', { d: `M${at[0] - 2} ${at[1]}h4M${at[0]} ${at[1] - 2}v4`, class: 'rv-crosshair' }));
      pitch.textContent = `${ctx.format(state.width / state.pixels, 3)} mm/pixel`;
      intensity.textContent = ctx.format(phantomAt(state.position), 3);
      details.textContent = ctx.text(`${state.width} mm → ${state.pixels} pixels. Drag a square in crop mode; change the active plane to compare the same point.`, `${state.width} mm → ${state.pixels} pixel. Crop 모드에서 정사각 영역을 그리고, 단면을 바꿔 같은 위치를 비교하세요.`);
      ctx.status.textContent = ctx.text('Blue crosshairs mark the same (x, y, z). Gold marks the crop. Arrow keys and position controls also move the point.', '파란 십자선은 같은 (x, y, z), 금색 테두리는 crop입니다. 방향키와 위치 조절로도 이동할 수 있습니다.');
      ctx.root.dataset.viewerPosition = state.position.join(','); ctx.root.dataset.viewerPlane = state.plane;
      ctx.root.dataset.viewerCrop = String(state.width);
    }
    ctx.onReset(() => { state = initial(); gesture = null; sync(); });
    ctx.onChange(render);
    if (ctx.stateLink) ctx.stateLink('rsna-linked-viewer', { read: () => ({ ...state, position: state.position.slice() }), write: value => { const next = restore(value); state = next; sync(); render(); } });
  });
})();
