/* Small shared UI for article-local, reproducible teaching models. */
(function (global) {
  'use strict';
  function parseCSV(source) {
    const rows = []; let row = [], cell = '', quoted = false;
    for (let i = 0; i < source.length; i += 1) {
      const c = source[i];
      if (quoted) {
        if (c === '"' && source[i + 1] === '"') { cell += '"'; i += 1; }
        else if (c === '"') quoted = false;
        else cell += c;
      } else if (c === '"' && cell === '') quoted = true;
      else if (c === ',') { row.push(cell); cell = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && source[i + 1] === '\n') i += 1;
        row.push(cell); if (row.some(value => value !== '')) rows.push(row);
        row = []; cell = '';
      } else cell += c;
    }
    if (quoted) throw new Error('Unclosed quoted CSV field');
    if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
    if (!rows.length) return [];
    const headers = rows.shift().map((value, i) => i === 0 ? value.replace(/^\uFEFF/, '') : value);
    if (new Set(headers).size !== headers.length) throw new Error('Duplicate CSV headers');
    return rows.map(values => {
      if (values.length !== headers.length) throw new Error('CSV column count differs');
      return Object.fromEntries(headers.map((key, i) => [key, values[i]]));
    });
  }
  function format(value, digits) {
    if (!Number.isFinite(Number(value))) return '—';
    value = Number(value); digits = digits === undefined ? 4 : digits;
    if (value === 0) value = 0;
    if (value !== 0 && Math.abs(value) < 1e-4) return value.toExponential(2);
    return value.toLocaleString('en-US', { maximumFractionDigits: digits, useGrouping: false });
  }
  function encodeSharedState(key, state) {
    if (typeof key !== 'string' || !/^[a-z][a-z0-9-]{0,79}$/.test(key)) throw new Error('Invalid experiment key');
    const json = JSON.stringify({ v: 1, key, state });
    const bytes = new TextEncoder().encode(json);
    if (bytes.length > 6000) throw new Error('Experiment settings are too long');
    return global.btoa(Array.from(bytes, b => String.fromCharCode(b)).join('')).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }
  function decodeSharedState(encoded) {
    if (typeof encoded !== 'string' || encoded.length > 8000 || !/^[A-Za-z0-9_-]+$/.test(encoded)) throw new Error('Invalid experiment settings');
    const bytes = Uint8Array.from(global.atob(encoded.replace(/-/g, '+').replace(/_/g, '/')), c => c.charCodeAt(0));
    if (bytes.length > 6000) throw new Error('Experiment settings are too long');
    const value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    if (!value || value.v !== 1 || typeof value.key !== 'string' || !/^[a-z][a-z0-9-]{0,79}$/.test(value.key) || !Object.prototype.hasOwnProperty.call(value, 'state')) throw new Error('Invalid experiment settings');
    return value;
  }
  if (typeof module === 'object' && module.exports) module.exports = { parseCSV, format, encodeSharedState, decodeSharedState };
  if (typeof document === 'undefined') return;
  const NS = 'http://www.w3.org/2000/svg';
  const factories = new Map();
  const states = new WeakMap();
  let fieldNumber = 0;
  function childrenInto(parent, children) {
    for (const child of [children].flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      parent.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return parent;
  }
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      if (value === false || value === null || value === undefined) return;
      if (key === 'checked') node.checked = Boolean(value);
      else if (key === 'disabled') node.disabled = Boolean(value);
      else node.setAttribute(key, value === true ? '' : String(value));
    });
    return childrenInto(node, children === undefined || children === null ? [] : children);
  }
  function svg(tag, attrs, text) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = String(text);
    return node;
  }
  const paint = color => ({ blue: 'var(--al-blue)', gold: 'var(--al-gold)', muted: 'var(--al-muted)', red: 'var(--al-red)' }[color] || color || 'var(--al-blue)');
  function plot(spec) {
    const [xmin, xmax] = spec.x, [ymin, ymax] = spec.y;
    if (![xmin, xmax, ymin, ymax].every(Number.isFinite) || xmax <= xmin || ymax <= ymin) throw new Error('Invalid plot domain');
    const frame = { left: 64, right: 620, top: 25, bottom: 164 };
    const x = value => frame.left + (value - xmin) / (xmax - xmin) * (frame.right - frame.left);
    const y = value => frame.bottom - (value - ymin) / (ymax - ymin) * (frame.bottom - frame.top);
    const node = svg('svg', { viewBox: '0 0 640 200', role: 'img', class: 'al-chart', 'aria-label': spec.title || [spec.yLabel, spec.xLabel].filter(Boolean).join(' / ') });
    node.append(svg('title', {}, spec.title || spec.yLabel || 'Chart'));
    const ticks = (a, b) => Array.from({ length: 5 }, (_, i) => a + (b - a) * i / 4);
    for (const value of spec.yTicks || ticks(ymin, ymax)) {
      node.append(svg('line', { x1: frame.left, x2: frame.right, y1: y(value), y2: y(value), class: 'al-grid' }));
      node.append(svg('text', { x: frame.left - 10, y: y(value) + 5, 'text-anchor': 'end' }, format(value, 3)));
    }
    for (const value of spec.xTicks || ticks(xmin, xmax)) {
      node.append(svg('line', { x1: x(value), x2: x(value), y1: frame.bottom, y2: frame.bottom + 4, class: 'al-axis' }));
      node.append(svg('text', { x: x(value), y: 187, 'text-anchor': 'middle' }, format(value, 3)));
    }
    for (const band of spec.bands || []) {
      const top = y(band.y1 === undefined ? ymax : band.y1), bottom = y(band.y0 === undefined ? ymin : band.y0);
      node.append(svg('rect', { x: x(band.x0), y: Math.min(top, bottom), width: x(band.x1) - x(band.x0), height: Math.abs(bottom - top), fill: paint(band.color), opacity: 0.12 }));
    }
    node.append(svg('path', { d: `M${frame.left},${frame.top}V${frame.bottom}H${frame.right}`, class: 'al-axis' }));
    for (const series of spec.series || []) {
      const path = series.points.map((point, i) => {
        if (!point.every(Number.isFinite)) throw new Error('Nonfinite plot point');
        return `${i ? 'L' : 'M'}${x(point[0]).toFixed(3)},${y(point[1]).toFixed(3)}`;
      }).join('');
      const line = svg('path', { d: path, fill: 'none', stroke: paint(series.color), 'stroke-width': 2.3, ...(series.dash ? { 'stroke-dasharray': '6 5' } : {}) });
      if (series.label) line.append(svg('title', {}, series.label));
      node.append(line);
    }
    const baseline = y(Math.max(ymin, Math.min(ymax, 0)));
    for (const bar of spec.bars || []) {
      const barWidth = (bar.width || (xmax - xmin) / 20) / (xmax - xmin) * (frame.right - frame.left);
      const rect = svg('rect', { x: x(bar.x) - barWidth / 2, y: Math.min(y(bar.y), baseline), width: barWidth, height: Math.abs(y(bar.y) - baseline), fill: paint(bar.color), opacity: 0.85 });
      rect.append(svg('title', {}, bar.label || `${format(bar.x)}: ${format(bar.y)}`)); node.append(rect);
    }
    for (const marker of spec.markers || []) {
      const circle = svg('circle', { cx: x(marker.x), cy: y(marker.y), r: 4, fill: paint(marker.color), tabindex: '0', 'aria-label': marker.label || `${format(marker.x)}: ${format(marker.y)}` });
      circle.append(svg('title', {}, marker.label || `${format(marker.x)}: ${format(marker.y)}`)); node.append(circle);
    }
    const legends = (spec.series || []).filter(item => item.label);
    return el('div', { class: 'al-chart-wrap' }, [
      spec.yLabel ? el('div', { class: 'al-axis-heading' }, spec.yLabel) : null,
      node,
      spec.xLabel ? el('div', { class: 'al-axis-caption' }, spec.xLabel) : null,
      legends.length ? el('div', { class: 'al-chart-legend' }, legends.map(item => el('span', {}, [
        el('i', { style: 'border-color:' + paint(item.color) + (item.dash ? ';border-top-style:dashed' : ''), 'aria-hidden': 'true' }), item.label
      ]))) : null
    ]);
  }
  function table(headers, rows) {
    return el('div', { class: 'al-table-wrap' }, el('table', {}, [
      el('thead', {}, el('tr', {}, headers.map(header => el('th', { scope: 'col' }, header)))),
      el('tbody', {}, rows.map(row => el('tr', {}, row.map(value => el('td', {}, value)))))
    ]));
  }
  async function initialize(root, factory) {
    const ui = root.querySelector('[data-lab-controls]'), view = root.querySelector('[data-lab-view]');
    const metrics = root.querySelector('[data-lab-metrics]'), status = root.querySelector('[data-lab-status]');
    const reset = root.querySelector('[data-lab-reset]');
    const ko = root.dataset.language === 'ko';
    const text = (en, korean) => ko ? korean : en;
    const defaults = [];
    const updateLabels = [];
    let render = () => {}, resetState = () => {}, sharedState = null;
    const state = { phase: 'loading', invalid: false }; states.set(root, state);
    root.dataset.ready = 'loading'; root.setAttribute('aria-busy', 'true'); ui.disabled = true; reset.disabled = true;
    ui.replaceChildren(el('legend', { class: 'al-sr-only' }, text('Controls', '조작'))); metrics.replaceChildren(); view.replaceChildren(); status.textContent = '';
    const ctx = {
      root, ko, text, ui, view, metrics, status, el, svg, plot, table, format,
      asset: path => (root.dataset.baseurl || '').replace(/\/$/, '') + '/' + path.replace(/^\//, ''),
      note: (en, korean) => { root.querySelector('[data-lab-note]').textContent = text(en, korean); },
      metric: (en, korean) => {
        const output = el('output'); metrics.append(el('div', {}, [el('dt', {}, text(en, korean)), el('dd', {}, output)])); return output;
      },
      control: (en, korean, options) => {
        const id = 'article-lab-field-' + (++fieldNumber);
        const config = options || {}, type = config.type || 'range';
        const input = el(type === 'select' ? 'select' : 'input', { id, ...(type !== 'select' ? { type } : {}), ...Object.fromEntries(['min', 'max', 'step'].filter(key => config[key] !== undefined).map(key => [key, config[key]])) });
        if (type === 'select') (config.options || []).forEach(option => input.append(el('option', { value: option.value }, text(option.en === undefined ? option.value : option.en, option.ko === undefined ? (option.en === undefined ? option.value : option.en) : option.ko))));
        if (type === 'checkbox') input.checked = Boolean(config.value); else input.value = String(config.value === undefined ? '' : config.value);
        defaults.push({ input, value: input.value, checked: input.checked });
        const label = el('label', { for: id }, text(en, korean));
        const output = el('output', { for: id, class: 'al-control-value' }, input.value);
        const updateLabel = () => { output.textContent = type === 'number' || type === 'range' ? format(Number(input.value), 5) : ''; };
        input.addEventListener('input', updateLabel); input.addEventListener('change', updateLabel); updateLabel();
        updateLabels.push(updateLabel);
        ui.append(el('div', { class: 'al-control' + (type === 'checkbox' ? ' al-checkbox' : '') }, type === 'checkbox' ? [input, label] : [el('div', { class: 'al-control-label' }, [label, output]), input]));
        return input;
      },
      button: (en, korean, handler) => {
        const button = el('button', { type: 'button', class: 'al-button' }, text(en, korean)); button.addEventListener('click', () => safely(handler, false)); ui.append(button); return button;
      },
      onChange: handler => { render = handler; render(); },
      onReset: handler => { resetState = handler; },
      stateLink: (key, config) => {
        if (key !== root.dataset.articleLab || !config || typeof config.read !== 'function' || typeof config.write !== 'function') throw new Error('Invalid experiment link configuration');
        root.classList.add('al-workshop');
        sharedState = { key, ...config };
        const fallback = el('input', { type: 'text', readonly: true, class: 'al-share-url', 'aria-label': text('Experiment link', '실험 설정 링크'), hidden: true });
        const copy = el('button', { type: 'button', class: 'al-button al-share-button' }, text('Copy experiment link', '실험 설정 링크 복사'));
        copy.addEventListener('click', async () => {
          try {
            const url = new URL(global.location.href);
            url.searchParams.set('labstate', encodeSharedState(key, config.read()));
            url.hash = root.id;
            fallback.value = url.href;
            try {
              await global.navigator.clipboard.writeText(url.href);
              fallback.hidden = true;
              status.textContent = text('Experiment link copied.', '실험 설정 링크를 복사했습니다.');
            } catch (_) {
              fallback.hidden = false; fallback.focus(); fallback.select();
              status.textContent = text('Copy the selected experiment link.', '선택한 실험 설정 링크를 복사해 주세요.');
            }
          } catch (_) { status.textContent = text('These settings cannot fit in a share link.', '설정이 길어 공유 링크를 만들 수 없습니다.'); }
        });
        ui.append(el('div', { class: 'al-share' }, [copy, fallback]));
      }
    };
    const safely = (callback, validateFirst = true) => {
      const numbers = [...ui.querySelectorAll('input[type="number"]')];
      const invalid = numbers.find(input => input.value.trim() === '' || !input.validity.valid);
      const warning = text('Enter a number within the field’s limits to update the figure.', '입력란의 범위에 맞는 숫자를 넣으면 그림이 갱신됩니다.');
      if (invalid && validateFirst) {
        numbers.forEach(input => input.setAttribute('aria-invalid', String(input === invalid)));
        state.invalid = true;
        status.textContent = warning;
        return;
      }
      if (state.invalid && status.textContent === warning) status.textContent = '';
      try {
        callback(); updateLabels.forEach(update => update()); root.dataset.ready = 'true';
        const remaining = numbers.find(input => input.value.trim() === '' || !input.validity.valid);
        numbers.forEach(input => input.setAttribute('aria-invalid', String(input === remaining)));
        state.invalid = Boolean(remaining);
        if (remaining) status.textContent = warning;
      }
      catch (error) { root.dataset.ready = 'error'; status.textContent = text('Calculation failed. Reset to retry.', '계산하지 못했습니다. 초기값으로 다시 시도해 주세요.'); console.error('Article lab ' + root.dataset.articleLab, error); }
    };
    ui.oninput = () => safely(render); ui.onchange = () => safely(render);
    const dataURL = path => ctx.asset(path) + (path.includes('?') ? '&' : '?') + 'lab=' + encodeURIComponent(root.dataset.assetVersion || '');
    ctx.fetchJSON = async path => { const response = await fetch(dataURL(path)); if (!response.ok) throw new Error('Data unavailable: ' + response.status); return response.json(); };
    ctx.fetchCSV = async path => { const response = await fetch(dataURL(path)); if (!response.ok) throw new Error('Data unavailable: ' + response.status); return parseCSV(await response.text()); };
    reset.onclick = () => {
      if (state.phase === 'error') { initialize(root, factory); return; }
      defaults.forEach(item => { item.input.value = item.value; item.input.checked = item.checked; item.input.dispatchEvent(new Event('change', { bubbles: false })); });
      safely(() => { resetState(); render(); }, false);
    };
    try {
      await factory(ctx);
      if (sharedState) {
        const encoded = new URL(global.location.href).searchParams.get('labstate');
        if (encoded) {
          try {
            const saved = decodeSharedState(encoded);
            if (saved.key === sharedState.key) {
              if (sharedState.write(saved.state) === false) throw new Error('Rejected experiment settings');
              updateLabels.forEach(update => update());
              status.textContent = text('Shared experiment settings restored.', '공유된 실험 설정을 불러왔습니다.');
            }
          } catch (_) {
            defaults.forEach(item => { item.input.value = item.value; item.input.checked = item.checked; });
            resetState(); render(); updateLabels.forEach(update => update());
            status.textContent = text('Invalid shared settings. Initial settings are shown.', '공유 설정이 올바르지 않아 초기값을 표시합니다.');
          }
        }
      }
      state.phase = 'ready'; root.dataset.ready = 'true'; ui.disabled = false;
    } catch (error) {
      state.phase = 'error'; root.dataset.ready = 'error';
      status.textContent = text('Interactive data could not load. Reset to retry.', '자료를 불러오지 못했습니다. 초기값으로 다시 시도해 주세요.');
      console.error('Article lab ' + root.dataset.articleLab, error);
    } finally { reset.disabled = false; root.setAttribute('aria-busy', 'false'); }
  }
  global.ArticleLabs = Object.freeze({
    register(id, factory) {
      if (factories.has(id)) throw new Error('Duplicate article lab ' + id);
      factories.set(id, factory);
      document.querySelectorAll('[data-article-lab]').forEach(root => {
        if (root.dataset.articleLab === id && !states.has(root)) initialize(root, factory);
      });
    }
  });
  // Long equations change the article height during MathJax startup.
  const anchor = () => {
    if (!location.hash.startsWith('#lab-')) return;
    Promise.all([global.MathJax?.startup?.promise || Promise.resolve(), document.fonts.ready]).then(() => {
      const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      if (target && target.matches('[data-article-lab]')) target.scrollIntoView({ block: 'start', behavior: 'instant' });
    });
  };
  if (document.readyState === 'complete') anchor(); else global.addEventListener('load', anchor, { once: true });
})(typeof globalThis !== 'undefined' ? globalThis : this);
