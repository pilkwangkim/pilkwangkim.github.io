/* Article-only controls and SVG rendering. Physics lives in negf-core.js. */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const BOX = { left: 68, right: 620, top: 24, bottom: 164 };
  const WIDTH = BOX.right - BOX.left;
  const HEIGHT = BOX.bottom - BOX.top;

  function element(tag, attributes, text) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attributes || {}).forEach(([name, value]) => node.setAttribute(name, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function axes(group, xTicks, yTicks, xScale, yScale, xLabel, yLabel) {
    yTicks.forEach(([value, label]) => {
      const y = yScale(value);
      group.append(element('line', { x1: BOX.left, x2: BOX.right, y1: y, y2: y, class: 'negf-grid' }));
      group.append(element('text', { x: BOX.left - 12, y: y + 5, 'text-anchor': 'end', class: 'negf-tick' }, label));
    });
    xTicks.forEach(([value, label]) => {
      const x = xScale(value);
      group.append(element('line', { x1: x, x2: x, y1: BOX.bottom, y2: BOX.bottom + 5, class: 'negf-axis' }));
      group.append(element('text', { x, y: BOX.bottom + 25, 'text-anchor': 'middle', class: 'negf-tick' }, label));
    });
    group.append(element('path', { d: `M${BOX.left},${BOX.top}V${BOX.bottom}H${BOX.right}`, class: 'negf-axis' }));
    group.append(element('text', { x: (BOX.left + BOX.right) / 2, y: 214, 'text-anchor': 'middle', class: 'negf-axis-label' }, xLabel));
    group.append(element('text', { x: BOX.left, y: 15, class: 'negf-axis-label' }, yLabel));
  }

  function initialize(root) {
    const core = window.NEGFCore;
    const ko = root.dataset.lang === 'ko';
    const find = (name) => root.querySelector(`[data-negf-${name}]`);
    const status = find('status');
    if (!core) {
      status.textContent = ko ? '계산기를 불러오지 못했습니다. 페이지를 새로고침해 주세요.' : 'The calculator could not load. Please reload the page.';
      root.querySelectorAll('input, button').forEach((control) => { control.disabled = true; });
      return;
    }
    const controls = { height: find('height'), width: find('width'), energy: find('energy') };
    Object.values(controls).forEach((control) => { control.disabled = false; });
    find('reset').disabled = false;
    const potentialSVG = find('potential');
    const transmissionSVG = find('transmission');
    const potentialGroup = potentialSVG.querySelector('[data-negf-chart]');
    const transmissionGroup = transmissionSVG.querySelector('[data-negf-chart]');
    const peakButtons = find('resonances');
    const limits = core.constants;
    let device;
    let spectrum;
    let energy = Number(controls.energy.value);
    let frame = 0;
    let signature = '';
    let potentialProbe;
    let transmissionProbe;
    let transmissionDot;
    const cache = new Map();
    const xPotential = (x) => BOX.left + WIDTH * x / (limits.deviceSites * limits.spacingNM);
    const yPotential = (u) => BOX.bottom - HEIGHT * u / 0.7;
    const xTransmission = (e) => BOX.left + WIDTH * (e - limits.energyMinEV) / (limits.energyMaxEV - limits.energyMinEV);
    const yTransmission = (t) => BOX.bottom - HEIGHT * t;
    const energyText = (e) => `${e.toFixed(7).replace(/0+$/, '').replace(/\.$/, '')} eV`;
    const transmissionText = (t) => t < 0.0001 && t > 0 ? t.toExponential(3) : t.toFixed(5);

    function drawPotential() {
      potentialGroup.replaceChildren();
      axes(potentialGroup, [0, 20, 40, 60, 80, 100].map((x) => [x, String(x)]),
        [0, 0.2, 0.4, 0.6].map((u) => [u, u.toFixed(1)]), xPotential, yPotential,
        ko ? '위치 x (nm)' : 'Position x (nm)', 'U (eV)');
      let path = `M${xPotential(0)},${yPotential(0)}`;
      device.potential.forEach((u, site) => {
        path += `H${xPotential(site)}V${yPotential(u)}H${xPotential(site + 1)}`;
      });
      path += `V${yPotential(0)}`;
      potentialGroup.append(element('path', { d: path + 'Z', class: 'negf-potential-fill' }));
      potentialGroup.append(element('path', { d: path, class: 'negf-potential-line' }));
      if (device.heightEV > 0) {
        const start = xPotential(device.wellStart);
        const end = xPotential(device.wellEnd);
        const y = BOX.bottom - 20;
        potentialGroup.append(element('path', { d: `M${start},${y - 4}V${y + 4}M${start},${y}H${end}M${end},${y - 4}V${y + 4}`, class: 'negf-axis' }));
        potentialGroup.append(element('text', { x: (start + end) / 2, y: y - 9, 'text-anchor': 'middle', class: 'negf-barrier-label' }, `${device.wellSites} nm`));
      }
      potentialProbe = element('line', { x1: BOX.left, x2: BOX.right, class: 'negf-probe-line' });
      potentialGroup.append(potentialProbe);
    }

    function drawTransmission() {
      transmissionGroup.replaceChildren();
      axes(transmissionGroup, [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map((e, i) => [e, i === 0 ? '0.001' : e.toFixed(1)]),
        [0, 0.25, 0.5, 0.75, 1].map((t) => [t, String(t)]), xTransmission, yTransmission,
        ko ? '입사 에너지 E (eV)' : 'Incident energy E (eV)', 'T');
      const path = Array.from(spectrum.energies, (e, i) =>
        `${i === 0 ? 'M' : 'L'}${xTransmission(e).toFixed(3)},${yTransmission(spectrum.transmissions[i]).toFixed(3)}`).join('');
      transmissionGroup.append(element('path', { d: path, class: 'negf-transmission-line' }));
      spectrum.peaks.forEach((peak) => {
        transmissionGroup.append(element('circle', { cx: xTransmission(peak.energyEV), cy: yTransmission(peak.transmission), r: 2.5, class: 'negf-resonance-dot' }));
      });
      transmissionProbe = element('line', { y1: BOX.top, y2: BOX.bottom, class: 'negf-probe-line' });
      transmissionDot = element('circle', { r: 5, class: 'negf-probe-dot' });
      transmissionGroup.append(transmissionProbe, transmissionDot);
      peakButtons.replaceChildren();
      spectrum.peaks.forEach((peak, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'negf-peak-button';
        button.textContent = `${peak.energyEV.toFixed(4)}`;
        button.setAttribute('aria-label', `${ko ? '공명' : 'Resonance'} ${index + 1}: ${energyText(peak.energyEV)}`);
        button.title = energyText(peak.energyEV);
        button.addEventListener('click', () => selectEnergy(peak.energyEV));
        peakButtons.append(button);
      });
      peakButtons.hidden = spectrum.peaks.length === 0;
      find('first-resonance').textContent = spectrum.peaks.length ? energyText(spectrum.peaks[0].energyEV) : (ko ? '없음' : 'None');
      const firstWidth = spectrum.peaks[0]?.widthEV;
      find('first-width').textContent = firstWidth ? `${ko ? '반치전폭' : 'FWHM'} ${(firstWidth * 1e6).toPrecision(3)} µeV` : '';
    }

    function updateProbe() {
      const t = core.transmission(energy, device);
      find('energy-value').textContent = energyText(energy);
      controls.energy.setAttribute('aria-valuetext', energyText(energy));
      find('probe-energy').textContent = energyText(energy);
      find('probe-transmission').textContent = transmissionText(t);
      const potentialY = yPotential(energy);
      potentialProbe.setAttribute('y1', potentialY);
      potentialProbe.setAttribute('y2', potentialY);
      const x = xTransmission(energy);
      transmissionProbe.setAttribute('x1', x);
      transmissionProbe.setAttribute('x2', x);
      transmissionDot.setAttribute('cx', x);
      transmissionDot.setAttribute('cy', yTransmission(t));
    }

    function selectEnergy(selected) {
      energy = Math.max(limits.energyMinEV, Math.min(limits.energyMaxEV, selected));
      // The thumb is approximate; the readout and NEGF calculation retain the
      // full resonance energy, including peaks narrower than one slider step.
      controls.energy.value = String(energy);
      if (device) updateProbe();
    }

    function update() {
      frame = 0;
      try {
        const height = Number(controls.height.value);
        const width = Number(controls.width.value);
        const nextSignature = `${height}:${width}`;
        find('height-value').textContent = `${height.toFixed(2)} eV`;
        find('width-value').textContent = `${width} nm`;
        controls.height.setAttribute('aria-valuetext', `${height.toFixed(2)} eV`);
        controls.width.setAttribute('aria-valuetext', `${width} nm`);
        if (nextSignature !== signature) {
          device = core.profile(height, width);
          spectrum = cache.get(nextSignature);
          if (!spectrum) {
            spectrum = core.spectrum(device);
            if (cache.size >= 12) cache.delete(cache.keys().next().value);
            cache.set(nextSignature, spectrum);
          }
          drawPotential();
          drawTransmission();
          signature = nextSignature;
        }
        updateProbe();
        status.textContent = height === 0
          ? (ko ? '장벽이 없으면 이 에너지 구간에서 투과율은 1입니다.' : 'With no barriers, transmission is 1 throughout this energy range.')
          : (ko ? `0.001–0.600 eV 구간의 공명 ${spectrum.peaks.length}개` : `${spectrum.peaks.length} resonances · 0.001–0.600 eV`);
        root.dataset.ready = 'true';
      } catch (error) {
        status.textContent = ko ? '계산하지 못했습니다. 초기값으로 다시 시도해 주세요.' : 'Calculation failed. Reset to try again.';
        root.dataset.ready = 'error';
        console.error('NEGF lab:', error);
      } finally {
        root.setAttribute('aria-busy', 'false');
      }
    }

    function schedule() {
      if (frame) return;
      root.setAttribute('aria-busy', 'true');
      frame = window.requestAnimationFrame(update);
    }
    controls.height.addEventListener('input', schedule);
    controls.width.addEventListener('input', schedule);
    controls.energy.addEventListener('input', () => { energy = Number(controls.energy.value); if (device) updateProbe(); });
    find('reset').addEventListener('click', () => {
      controls.height.value = String(limits.defaultHeightEV);
      controls.width.value = String(limits.defaultWellSites);
      energy = 0.1413;
      controls.energy.value = String(energy);
      schedule();
    });
    transmissionSVG.addEventListener('click', (event) => {
      const point = transmissionSVG.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const matrix = transmissionSVG.getScreenCTM();
      if (!matrix || !spectrum) return;
      const location = point.matrixTransform(matrix.inverse());
      if (location.x < BOX.left || location.x > BOX.right || location.y < BOX.top || location.y > BOX.bottom) return;
      const nearby = spectrum.peaks.reduce((best, peak) => Math.abs(xTransmission(peak.energyEV) - location.x) < Math.abs(xTransmission(best.energyEV) - location.x) ? peak : best,
        { energyEV: Infinity });
      const selected = Math.abs(xTransmission(nearby.energyEV) - location.x) < 8
        ? nearby.energyEV : limits.energyMinEV + (location.x - BOX.left) / WIDTH * (limits.energyMaxEV - limits.energyMinEV);
      selectEnergy(selected);
    });
    schedule();
    if (window.location.hash === '#negf-lab-title') {
      const anchorAfterTypeset = () => {
        const mathReady = window.MathJax?.startup?.promise || Promise.resolve();
        Promise.all([mathReady, document.fonts.ready]).then(() => {
          if (window.location.hash === '#negf-lab-title') root.scrollIntoView({ block: 'start', behavior: 'instant' });
        });
      };
      if (document.readyState === 'complete') anchorAfterTypeset();
      else window.addEventListener('load', anchorAfterTypeset, { once: true });
    }
  }

  document.querySelectorAll('[data-negf-lab]').forEach(initialize);
})();
