(() => {
  'use strict';
  const context = document.getElementById('local-editor-context');
  if (!context || !['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname)) return;
  const { url } = JSON.parse(context.textContent);
  const api = '/__local_editor/';
  const stateDefiningCommands = /\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand|newenvironment|renewenvironment|provideenvironment|def|gdef|edef|xdef|let|futurelet|global|DeclareMathOperator|Newextarrow|newextarrow|definecolor|colorlet|begingroup|endgroup|begingroupReset|begingroupSandbox|csname|endcsname|expandafter|catcode|mathcode|setOptions)\b/;
  const changes = new Map();
  const nodes = new Map();
  const metadataChanges = new Map();
  const metadataInputs = new Map();
  const structuredChanges = new Map();
  const structuredTargets = new Map();
  const availableMath = new Map();
  const availableTables = new Map();
  let metadataOpen = false;
  let snapshot;
  let token;
  let editing = false;
  let busy = false;
  let composing = false;
  let receipt;
  let undoRevision;

  const bar = document.createElement('section');
  bar.id = 'local-editor-bar';
  bar.setAttribute('aria-label', 'Local page editor');
  bar.lang = 'en';
  bar.innerHTML = '<div class="local-editor-label"><strong>Local editor</strong><span id="local-editor-status" role="status" aria-live="polite">Loading…</span></div><div class="local-editor-actions"><button type="button" id="local-editor-toggle" disabled>Edit page</button><button type="button" id="local-editor-metadata-toggle" aria-expanded="false" aria-controls="local-editor-metadata" disabled>Metadata</button><button type="button" id="local-editor-structures-toggle" aria-haspopup="dialog" disabled>Tables &amp; equations</button><button type="button" id="local-editor-save" disabled>Save changes</button><button type="button" id="local-editor-discard" disabled>Discard</button><button type="button" id="local-editor-undo" disabled>Undo last save</button><button type="button" id="local-editor-refresh" disabled>Refresh preview</button></div>';
  document.body.append(bar);
  const status = bar.querySelector('#local-editor-status');
  const toggle = bar.querySelector('#local-editor-toggle');
  const save = bar.querySelector('#local-editor-save');
  const discard = bar.querySelector('#local-editor-discard');
  const undo = bar.querySelector('#local-editor-undo');
  const refresh = bar.querySelector('#local-editor-refresh');
  const metadataToggle = bar.querySelector('#local-editor-metadata-toggle');
  const structuresToggle = bar.querySelector('#local-editor-structures-toggle');
  const metadataPanel = document.createElement('section');
  metadataPanel.id = 'local-editor-metadata';
  metadataPanel.hidden = true;
  metadataPanel.setAttribute('aria-labelledby', 'local-editor-metadata-title');
  metadataPanel.innerHTML = '<div class="local-editor-metadata-heading"><h2 id="local-editor-metadata-title">Article metadata</h2><button type="button" class="local-editor-metadata-close" aria-label="Close metadata">Close</button></div><form id="local-editor-metadata-form" novalidate><fieldset class="local-editor-metadata-fields"></fieldset></form><p class="local-editor-metadata-note">Save changes applies text and metadata together. File names and article addresses stay the same.</p>';
  bar.prepend(metadataPanel);
  const metadataForm = metadataPanel.querySelector('form');
  const metadataFields = metadataPanel.querySelector('fieldset');
  const dirtyCount = () => changes.size + metadataChanges.size + structuredChanges.size;
  const sameValue = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const listValue = value => [...new Set(value.split(',').map(item => item.trim()).filter(Boolean))];
  const baselineMetadata = () => snapshot?.metadata?.values || {};
  const widgetCheckpointKey = `local-editor:widgets:${url}`;

  function saveWidgetCheckpoint(revision = snapshot.revision) {
    try {
      const negf = [...document.querySelectorAll('main article [data-negf-lab]')].map((root, index) => {
        const detail = {};
        root.dispatchEvent(new CustomEvent('article-preview:capture', { detail }));
        return { index, state: detail.state };
      }).filter(item => item.state);
      const data = {
        v: 1, route: location.pathname + location.search, revision,
        created: Date.now(), labs: window.ArticleLabs?.checkpoint(), negf
      };
      const encoded = JSON.stringify(data);
      if (encoded.length > 256000) throw new Error('Experiment checkpoint is too large');
      sessionStorage.setItem(widgetCheckpointKey, encoded);
    } catch (_) { sessionStorage.removeItem(widgetCheckpointKey); }
  }

  async function restoreWidgetCheckpoint() {
    const encoded = sessionStorage.getItem(widgetCheckpointKey);
    sessionStorage.removeItem(widgetCheckpointKey);
    if (!encoded) return false;
    try {
      if (encoded.length > 256000) throw new Error('Experiment checkpoint is too large');
      const data = JSON.parse(encoded);
      if (data.v !== 1 || data.route !== location.pathname + location.search || data.revision !== snapshot.revision ||
          !Number.isFinite(data.created) || Date.now() - data.created < 0 || Date.now() - data.created > 600000 ||
          !Array.isArray(data.negf) || data.negf.length > 20) return false;
      let incomplete = false;
      if (data.labs) {
        if (!window.ArticleLabs?.restoreCheckpoint) incomplete = true;
        else incomplete = (await window.ArticleLabs.restoreCheckpoint(data.labs)).failures.length > 0;
      }
      const roots = [...document.querySelectorAll('main article [data-negf-lab]')];
      for (const item of data.negf) {
        if (!Number.isInteger(item.index) || !roots[item.index]) { incomplete = true; continue; }
        const detail = { state: item.state, restored: false };
        roots[item.index].dispatchEvent(new CustomEvent('article-preview:restore', { detail }));
        if (!detail.restored) incomplete = true;
      }
      return incomplete;
    } catch (_) { return true; }
  }

  function changedMessage() {
    const parts = [];
    if (changes.size) parts.push(`${changes.size} text ${changes.size === 1 ? 'change' : 'changes'}`);
    if (metadataChanges.size) parts.push(`${metadataChanges.size} metadata ${metadataChanges.size === 1 ? 'field' : 'fields'}`);
    if (structuredChanges.size) parts.push(`${structuredChanges.size} table/equation ${structuredChanges.size === 1 ? 'change' : 'changes'}`);
    message(parts.length ? `${parts.join(' · ')} · not saved` : 'Ready · changes stay in local files');
    controls();
  }

  function setMetadataOpen(value, restoreFocus = false) {
    metadataOpen = Boolean(value && snapshot?.metadata);
    metadataPanel.hidden = !metadataOpen;
    bar.classList.toggle('metadata-open', metadataOpen);
    metadataToggle.setAttribute('aria-expanded', String(metadataOpen));
    sessionStorage.setItem(`local-editor:metadata:${url}`, metadataOpen ? '1' : '0');
    if (restoreFocus) metadataToggle.focus();
  }

  function metadataValue(key) {
    const input = metadataInputs.get(key);
    if (!input) return baselineMetadata()[key];
    if (key === 'tags' || key === 'categories') return listValue(input.value);
    if (key === 'series') return input.value || null;
    if (key === 'series_order') return metadataInputs.get('series').value ? (input.value === '' ? null : Number(input.value)) : null;
    if (key === 'date') {
      const wallTime = input.value;
      const zone = metadataInputs.get('time_zone').value.trim();
      return wallTime ? `${wallTime.replace('T', ' ')}${wallTime.length === 16 ? ':00' : ''} ${zone}` : '';
    }
    return input.value;
  }

  function noteMetadataChange(key) {
    if (!snapshot?.metadata?.writable?.[key]) return;
    const value = metadataValue(key);
    if (sameValue(value, baselineMetadata()[key])) metadataChanges.delete(key);
    else metadataChanges.set(key, value);
    metadataInputs.get(key)?.closest('.local-editor-metadata-field').classList.toggle('is-changed', metadataChanges.has(key));
    if (key === 'date') metadataInputs.get('time_zone')?.closest('.local-editor-metadata-field').classList.toggle('is-changed', metadataChanges.has(key));
    changedMessage();
  }

  function fillSeriesOptions() {
    const input = metadataInputs.get('series');
    if (!input) return;
    const current = input.value;
    const topic = metadataInputs.get('topic').value;
    input.replaceChildren(new Option('None', ''));
    (snapshot.metadata.options.series || []).filter(item => item.topic === topic).forEach(item => input.add(new Option(item.title, item.id)));
    input.value = [...input.options].some(option => option.value === current) ? current : '';
    const order = metadataInputs.get('series_order');
    if (order) {
      order.disabled = busy || !snapshot.metadata.writable.series_order || !input.value;
      order.required = Boolean(input.value);
    }
  }

  function renderMetadata() {
    metadataInputs.clear();
    metadataFields.replaceChildren();
    if (!snapshot?.metadata) return;
    const { values, writable, options } = snapshot.metadata;
    const addField = (key, label, input, hint) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'local-editor-metadata-field';
      const labelNode = document.createElement('label');
      input.id = `local-editor-field-${key}`;
      labelNode.htmlFor = input.id;
      labelNode.textContent = label;
      input.disabled = !writable[key === 'time_zone' ? 'date' : key];
      metadataInputs.set(key, input);
      wrapper.append(labelNode, input);
      if (hint || input.disabled) {
        const help = document.createElement('small');
        help.id = `${input.id}-help`;
        help.textContent = input.disabled ? 'This field is protected. Edit it in the Markdown source.' : hint;
        input.setAttribute('aria-describedby', help.id);
        wrapper.append(help);
      }
      metadataFields.append(wrapper);
      input.addEventListener('input', () => noteMetadataChange(key === 'time_zone' ? 'date' : key));
      return wrapper;
    };
    [['tags', 'Tags', 'Separate tags with commas. Use lowercase words joined by hyphens; existing tags appear as suggestions.'], ['categories', 'Categories', 'Separate categories with commas. Existing categories appear as suggestions.']].forEach(([key, label, hint]) => {
      const input = document.createElement('input');
      input.type = 'text';
      input.autocomplete = 'off';
      input.value = (values[key] || []).join(', ');
      const wrapper = addField(key, label, input, hint);
      const suggestions = document.createElement('datalist');
      suggestions.id = `${input.id}-suggestions`;
      const updateSuggestions = () => {
        const parts = input.value.split(',');
        const prefix = parts.slice(0, -1).map(part => part.trim()).filter(Boolean);
        const partial = parts.at(-1).trim().toLowerCase();
        suggestions.replaceChildren();
        (options[key] || []).filter(value => !prefix.includes(value) && value.toLowerCase().includes(partial)).forEach(value => suggestions.append(new Option([...prefix, value].join(', '))));
      };
      input.setAttribute('list', suggestions.id);
      input.addEventListener('input', updateSuggestions);
      updateSuggestions();
      wrapper.append(suggestions);
    });
    const topic = document.createElement('select');
    (options.topics || []).forEach(item => topic.add(new Option(item.title, item.id)));
    topic.value = values.topic || '';
    addField('topic', 'Topic', topic, 'The selected topic’s required tag is kept automatically.');
    const series = document.createElement('select');
    series.add(new Option('None', ''));
    (options.series || []).forEach(item => series.add(new Option(item.title, item.id)));
    series.value = values.series || '';
    addField('series', 'Series', series, 'Only series belonging to the selected topic are shown.');
    const order = document.createElement('input');
    order.type = 'number';
    order.min = '0';
    order.step = '1';
    order.value = values.series_order ?? '';
    addField('series_order', 'Series order', order, 'A whole number starting at 0. Required when a series is selected.');
    fillSeriesOptions();
    topic.addEventListener('change', () => {
      fillSeriesOptions();
      if (!series.value) order.value = '';
      noteMetadataChange('series');
      noteMetadataChange('series_order');
    });
    series.addEventListener('change', () => {
      if (!series.value) order.value = '';
      fillSeriesOptions();
      noteMetadataChange('series_order');
    });
    const date = document.createElement('input');
    date.type = 'datetime-local';
    date.step = '1';
    date.required = true;
    const parsed = String(values.date || '').match(/^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(?::(\d{2}))?(?:\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?$/);
    date.value = parsed ? `${parsed[1]}T${parsed[2]}:${parsed[3] || '00'}` : '';
    addField('date', 'Publication date', date, 'Future dates remain visible in the local preview.');
    const zone = document.createElement('input');
    zone.type = 'text';
    zone.value = parsed?.[4] === 'Z' ? '+0000' : (parsed?.[4]?.replace(':', '') || '+0900');
    zone.pattern = '(?:\\+|-)(?:(?:0[0-9]|1[0-3])[0-5][0-9]|1400)';
    zone.maxLength = 5;
    zone.required = true;
    zone.placeholder = '+0900';
    zone.title = 'UTC offset, for example +0900 or -0500';
    addField('time_zone', 'Time zone', zone, 'UTC offset in ±HHMM format, for example +0900.');
    const language = document.createElement('select');
    (options.languages?.length ? options.languages : [{ id: 'en', title: 'English' }, { id: 'ko', title: '한국어' }]).forEach(item => {
      const option = new Option(item.id === 'ko' ? '한국어' : item.title, item.id);
      if (item.id === 'ko') option.lang = 'ko';
      language.add(option);
    });
    language.value = values.lang || 'en';
    addField('lang', 'Language', language);
    const related = snapshot.metadata.related || snapshot.related || [];
    if (related.length) {
      const note = document.createElement('p');
      note.className = 'local-editor-metadata-related';
      note.textContent = `Tags, categories, topic, series and series order also update ${related.length} related ${related.length === 1 ? 'translation or edition' : 'translations or editions'}. Publication date and language apply only to this article.`;
      metadataFields.prepend(note);
    }
    metadataFields.disabled = busy;
  }

  function prepareMetadata() {
    if (!snapshot?.metadata) return true;
    const tag = snapshot.metadata.options.topics?.find(item => item.id === metadataValue('topic'))?.tag;
    const tags = metadataValue('tags');
    if (tag && snapshot.metadata.writable.tags && !tags.includes(tag)) {
      metadataInputs.get('tags').value = [...tags, tag].join(', ');
      noteMetadataChange('tags');
    }
    if (metadataChanges.size && !metadataForm.checkValidity()) {
      setMetadataOpen(true);
      metadataForm.reportValidity();
      message('Check the highlighted metadata fields before saving.', true);
      return false;
    }
    return true;
  }
  metadataForm.addEventListener('submit', event => { event.preventDefault(); submit('save'); });
  metadataForm.addEventListener('compositionstart', () => { composing = true; });
  metadataForm.addEventListener('compositionend', () => { composing = false; });
  metadataToggle.addEventListener('click', () => setMetadataOpen(!metadataOpen));
  metadataPanel.querySelector('.local-editor-metadata-close').addEventListener('click', () => setMetadataOpen(false, true));
  if ('ResizeObserver' in window) new ResizeObserver(() => {
    document.documentElement.style.setProperty('--local-editor-height', `${bar.getBoundingClientRect().height}px`);
  }).observe(bar);

  const structureDialog = document.createElement('dialog');
  structureDialog.id = 'local-editor-structure-dialog';
  structureDialog.lang = 'en';
  structureDialog.setAttribute('aria-labelledby', 'local-editor-structure-title');
  structureDialog.setAttribute('aria-describedby', 'local-editor-structure-help');
  structureDialog.innerHTML = '<div class="local-editor-structure-heading"><button type="button" id="local-editor-structure-back">All items</button><h2 id="local-editor-structure-title">Tables &amp; equations</h2><button type="button" id="local-editor-structure-close" aria-label="Close tables and equations">Close</button></div><p id="local-editor-structure-help"></p><div id="local-editor-structure-content"></div><p id="local-editor-structure-error" role="status" aria-live="polite"></p><div class="local-editor-structure-footer"><span>Apply to draft, then Save changes or ⌘/Ctrl+S.</span><button type="button" id="local-editor-structure-cancel">Cancel</button><button type="button" id="local-editor-structure-apply" disabled>Apply to draft</button></div>';
  document.body.append(structureDialog);
  const structureTitle = structureDialog.querySelector('h2');
  const structureHelp = structureDialog.querySelector('#local-editor-structure-help');
  const structureContent = structureDialog.querySelector('#local-editor-structure-content');
  const structureError = structureDialog.querySelector('#local-editor-structure-error');
  const structureApply = structureDialog.querySelector('#local-editor-structure-apply');
  const structureBack = structureDialog.querySelector('#local-editor-structure-back');
  const structureFooter = structureDialog.querySelector('.local-editor-structure-footer');
  let currentStructure;
  let structureReturnFocus;
  let previewVersion = 0;
  let previewTimer;
  let mathPreviewQueue = Promise.resolve();

  const structureValue = (id, original) => structuredChanges.get(id)?.text ?? original;
  const tableCells = table => table.rows.flat();
  const mathError = node => node.querySelector('mjx-merror, merror, [data-mjx-error]') || [...node.querySelectorAll('[mathcolor="red"], [style*="color: red"], [style*="color: rgb(255, 0, 0)"]')].find(item => /^\\[A-Za-z]+$/.test(item.textContent.trim()));

  function hasUnappliedStructureChanges() {
    const editor = currentStructure;
    if (!structureDialog.open || !editor) return false;
    if (editor.type === 'math') return editor.input.value !== structureValue(editor.item.id, editor.item.latex);
    return [...editor.inputs].some(([id, input]) => input.value !== structureValue(id, editor.originals.get(id)));
  }

  function clearStructurePreview() {
    const preview = structureContent.querySelector('.local-editor-structure-preview');
    if (preview && window.MathJax?.typesetClear) window.MathJax.typesetClear([preview]);
  }

  function closeStructureEditor() {
    clearTimeout(previewTimer);
    previewVersion += 1;
    clearStructurePreview();
    currentStructure = undefined;
    if (structureDialog.open) structureDialog.close();
  }

  function prepareStructureView(title, help) {
    clearTimeout(previewTimer);
    previewVersion += 1;
    clearStructurePreview();
    currentStructure = undefined;
    structureTitle.textContent = title;
    structureHelp.textContent = help;
    structureContent.replaceChildren();
    structureError.textContent = '';
    structureApply.disabled = true;
    structureBack.hidden = false;
    structureFooter.hidden = false;
    if (!structureDialog.open) {
      structureReturnFocus = document.activeElement;
      structureDialog.showModal();
    }
  }

  // Conversion shares the page's TeX input jax. Keep preview numbering and
  // labels local to each serialized job; never reset the article's MathJax.
  function withPreviewMath(action) {
    const job = mathPreviewQueue.catch(() => {}).then(async () => {
      const math = window.MathJax;
      if (!math?.startup?.promise) throw new Error('MathJax is still loading. Reopen the editor after the page finishes loading.');
      await math.startup.promise;
      if (!math.typesetPromise) throw new Error('The page’s equation renderer is not ready. Refresh preview and try again.');
      const jaxes = Array.isArray(math.startup.document.inputJax) ? math.startup.document.inputJax : [math.startup.document.inputJax];
      const userDefinitions = jaxes.filter(jax => jax?.parseOptions?.handlers).map(jax => ({
        jax,
        maps: ['new-Command', 'new-Delimiter', 'new-Environment'].map(name => ({ name, entries: new Map(jax.parseOptions.handlers.retrieve(name)?.map || []) })),
        colors: new Map(jax.parseOptions.packageData?.get('color')?.model?.userColors || [])
      }));
      const tagStates = jaxes.filter(jax => jax?.parseOptions?.tags).map(jax => {
        const tags = jax.parseOptions.tags;
        const descriptors = Object.getOwnPropertyDescriptors(tags);
        Object.values(descriptors).forEach(descriptor => {
          const value = descriptor.value;
          if (Array.isArray(value)) descriptor.value = value.slice();
          else if (value instanceof Map) descriptor.value = new Map(value);
          else if (value instanceof Set) descriptor.value = new Set(value);
          else if (value && Object.getPrototypeOf(value) === Object.prototype) descriptor.value = { ...value };
        });
        return { tags, descriptors };
      });
      try { return await action(math); }
      finally {
        // Keep canceled/failed previews from teaching later previews a macro
        // or named color that does not exist in the article source.
        userDefinitions.forEach(({ jax, maps, colors }) => {
          maps.forEach(({ name, entries }) => {
            const definitions = jax.parseOptions.handlers.retrieve(name)?.map;
            if (!(definitions instanceof Map)) return;
            definitions.clear();
            entries.forEach((value, key) => definitions.set(key, value));
          });
          const userColors = jax.parseOptions.packageData?.get('color')?.model?.userColors;
          if (userColors instanceof Map) { userColors.clear(); colors.forEach((value, key) => userColors.set(key, value)); }
        });
        tagStates.forEach(({ tags, descriptors }) => {
          Object.keys(tags).filter(key => !Object.hasOwn(descriptors, key)).forEach(key => { delete tags[key]; });
          Object.defineProperties(tags, descriptors);
        });
      }
    });
    mathPreviewQueue = job.catch(() => {});
    return job;
  }

  function setStructurePreviewStatus(editor, text, valid = false) {
    if (currentStructure !== editor || !structureDialog.open) return;
    editor.valid = valid;
    structureError.textContent = text;
    structureError.classList.toggle('is-error', !valid && Boolean(text) && text !== 'Updating preview…');
    structureApply.disabled = !valid;
  }

  function scheduleStructurePreview(editor) {
    clearTimeout(previewTimer);
    const version = ++previewVersion;
    setStructurePreviewStatus(editor, 'Updating preview…');
    previewTimer = setTimeout(() => updateStructurePreview(editor, version), 250);
  }

  function validateEquationInput(latex, item) {
    if (typeof latex !== 'string' || !latex.trim() || /[\uD800-\uDFFF]/u.test(latex) || new TextEncoder().encode(latex).length > 100000) {
      throw new Error('An equation must be nonempty valid UTF-8 and shorter than 100 KB.');
    }
    if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\r]/.test(latex)) throw new Error('An equation cannot contain control characters.');
    if (!item.display && latex.includes('\n')) throw new Error('Inline equations must stay on one line.');
    if (stateDefiningCommands.test(latex)) throw new Error('Keep equations self-contained. Macro, environment and color definitions cannot be edited here.');
    if (latex.includes('{{') || latex.includes('{%') || /<\/?[A-Za-z][A-Za-z0-9:-]*(?:\s[^>]*|\/?)>|<!(?:--|DOCTYPE)|<\/?(?:script|style|iframe|img|svg)\b/i.test(latex) || /\\(?:href|url|style|class|cssId|htmlClass|htmlId|htmlData|require)\b/.test(latex)) {
      throw new Error('Equations cannot contain HTML, template syntax or HTML/link commands.');
    }
    for (let index = 0; index < latex.length; index += 1) {
      if (latex[index] !== '$' && !(latex[index] === '\\' && /[()[\]]/.test(latex[index + 1] || ''))) continue;
      let slashes = latex[index] === '\\' ? 1 : 0;
      for (let cursor = index - 1; cursor >= 0 && latex[cursor] === '\\'; cursor -= 1) slashes += 1;
      if ((latex[index] === '$' && slashes % 2 === 0) || (latex[index] === '\\' && slashes % 2 === 1)) {
        throw new Error('Enter only LaTeX without surrounding math delimiters.');
      }
    }
  }

  async function updateStructurePreview(editor, version) {
    const current = () => currentStructure === editor && structureDialog.open && version === previewVersion;
    try {
      if (editor.type === 'math') {
        const latex = editor.input.value;
        validateEquationInput(latex, editor.item);
        let renderHost;
        await withPreviewMath(async math => {
          if (!current()) return;
          math.typesetClear?.([editor.preview]);
          renderHost = document.createElement('div');
          renderHost.textContent = editor.item.display ? `\\[${latex}\\]` : `\\(${latex}\\)`;
          editor.preview.replaceChildren(renderHost);
          // Dynamic typesetting also installs the new CHTML glyph CSS. Direct
          // string conversion alone can leave new glyphs invisible in v4.
          await math.typesetPromise([renderHost]);
        });
        if (!current()) { if (renderHost) window.MathJax?.typesetClear?.([renderHost]); return; }
        // Labels in the preview must not duplicate article anchor IDs.
        editor.preview.querySelectorAll('[id]').forEach(node => node.removeAttribute('id'));
        if (!editor.preview.querySelector('mjx-container')) throw new Error('The equation could not be rendered. Refresh preview and try again.');
        const error = mathError(editor.preview);
        if (error) throw new Error(error.getAttribute('data-mjx-error') || error.textContent || 'Check the LaTeX syntax.');
      } else {
        if ([...editor.inputs.values()].some(input => stateDefiningCommands.test(input.value))) throw new Error('Keep equations self-contained. Macro, environment and color definitions cannot be edited here.');
        const edits = [...editor.inputs].filter(([id, input]) => input.value !== editor.originals.get(id)).map(([id, input]) => ({ id, text: input.value, original: editor.originals.get(id) }));
        const result = await request('preview', { url, revision: snapshot.revision, table_id: editor.item.id, edits });
        if (!current()) return;
        let renderHost;
        await withPreviewMath(async math => {
          if (!current()) return;
          math.typesetClear?.([editor.preview]);
          renderHost = document.createElement('div');
          renderHost.innerHTML = result.html;
          editor.preview.replaceChildren(renderHost);
          await math.typesetPromise([renderHost]);
        });
        if (!current()) { if (renderHost) window.MathJax?.typesetClear?.([renderHost]); return; }
        editor.preview.querySelectorAll('[id]').forEach(node => node.removeAttribute('id'));
        const error = mathError(editor.preview);
        if (error) throw new Error(error.getAttribute('data-mjx-error') || error.textContent || 'Check the LaTeX in the table cells.');
      }
      if (current()) setStructurePreviewStatus(editor, 'Preview ready', true);
    } catch (error) {
      if (current()) setStructurePreviewStatus(editor, error.message || 'Preview failed. Check the input and try again.');
    }
  }

  function appendPreview() {
    const heading = document.createElement('h3');
    heading.textContent = 'Preview';
    const preview = document.createElement('div');
    preview.className = 'local-editor-structure-preview';
    preview.setAttribute('aria-label', 'Rendered preview');
    structureContent.append(heading, preview);
    return preview;
  }

  function openMathEditor(item) {
    const index = [...availableMath.keys()].indexOf(item.id) + 1;
    prepareStructureView(`${item.display ? 'Display' : 'Inline'} equation ${index}`, 'Edit the LaTeX inside the equation. Its inline/display placement and surrounding delimiters stay the same.');
    const label = document.createElement('label');
    label.htmlFor = 'local-editor-equation-input';
    label.textContent = 'LaTeX (without surrounding $ or $$)';
    const input = document.createElement('textarea');
    input.id = label.htmlFor;
    input.rows = item.display ? 6 : 3;
    input.spellcheck = false;
    input.autocapitalize = 'off';
    input.value = structureValue(item.id, item.latex);
    const shortcuts = document.createElement('div');
    shortcuts.className = 'local-editor-equation-shortcuts';
    [['Fraction', '\\frac{a}{b}'], ['Square root', '\\sqrt{x}'], ['Superscript', '^{2}'], ['Subscript', '_{i}']].forEach(([text, latex]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = text;
      button.addEventListener('click', () => {
        input.setRangeText(latex, input.selectionStart, input.selectionEnd, 'select');
        input.focus();
        scheduleStructurePreview(currentStructure);
      });
      shortcuts.append(button);
    });
    structureContent.append(label, input, shortcuts);
    const editor = { type: 'math', item, input, preview: appendPreview(), valid: false };
    currentStructure = editor;
    input.addEventListener('input', () => scheduleStructurePreview(editor));
    input.addEventListener('compositionstart', () => { composing = true; });
    input.addEventListener('compositionend', () => { composing = false; scheduleStructurePreview(editor); });
    scheduleStructurePreview(editor);
    input.focus();
  }

  function openTableEditor(item, cellId) {
    const index = [...availableTables.keys()].indexOf(item.id) + 1;
    prepareStructureView(`Table ${index}`, 'Edit each cell using Markdown and LaTeX, for example **bold** or $x^2$. Rows, columns and alignment stay the same.');
    const scroller = document.createElement('div');
    scroller.className = 'local-editor-table-grid-scroll';
    scroller.setAttribute('role', 'region');
    scroller.setAttribute('aria-label', 'Table cells; scroll horizontally for more columns');
    scroller.tabIndex = 0;
    const grid = document.createElement('table');
    grid.className = 'local-editor-table-grid';
    const body = document.createElement('tbody');
    const inputs = new Map();
    const originals = new Map();
    item.rows.forEach((row, rowIndex) => {
      const tr = document.createElement('tr');
      row.forEach((cell, columnIndex) => {
        const td = document.createElement(cell.header ? 'th' : 'td');
        if (cell.header) td.scope = 'col';
        const input = document.createElement('textarea');
        input.rows = 3;
        input.spellcheck = false;
        input.setAttribute('aria-label', `${cell.header ? 'Header' : 'Row'} ${rowIndex + 1}, column ${columnIndex + 1}`);
        input.value = structureValue(cell.id, cell.source);
        inputs.set(cell.id, input);
        originals.set(cell.id, cell.source);
        td.append(input);
        tr.append(td);
      });
      body.append(tr);
    });
    grid.append(body);
    scroller.append(grid);
    structureContent.append(scroller);
    const editor = { type: 'table', item, inputs, originals, preview: appendPreview(), valid: false };
    currentStructure = editor;
    inputs.forEach(input => {
      input.addEventListener('input', () => scheduleStructurePreview(editor));
      input.addEventListener('compositionstart', () => { composing = true; });
      input.addEventListener('compositionend', () => { composing = false; scheduleStructurePreview(editor); });
    });
    scheduleStructurePreview(editor);
    (inputs.get(cellId) || inputs.values().next().value)?.focus();
  }

  function markStructuredDrafts() {
    structuredTargets.forEach(({ node, item, type, button }) => {
      const dirty = type === 'math' ? structuredChanges.has(item.id) : tableCells(item).some(cell => structuredChanges.has(cell.id));
      node.classList.toggle('local-editor-structured-changed', dirty);
      button?.classList.toggle('local-editor-structured-changed', dirty);
    });
  }

  function applyStructureDraft() {
    const editor = currentStructure;
    if (!editor || !editor.valid || composing) {
      if (editor) structureError.textContent = 'Wait for a valid preview before applying changes.';
      return false;
    }
    const stage = (id, text, original) => {
      if (text === original) structuredChanges.delete(id);
      else structuredChanges.set(id, { id, text, original });
    };
    if (editor.type === 'math') stage(editor.item.id, editor.input.value, editor.item.latex);
    else editor.inputs.forEach((input, id) => stage(id, input.value, editor.originals.get(id)));
    markStructuredDrafts();
    closeStructureEditor();
    changedMessage();
    return true;
  }

  function openStructureChooser() {
    if (busy || !snapshot) return;
    if (!editing) { setEditing(true); sessionStorage.setItem(`local-editor:editing:${url}`, '1'); }
    prepareStructureView('Tables & equations', 'Choose an item, or click it directly in the article while Edit page is on. Applied drafts appear after Save changes rebuilds the page.');
    structureBack.hidden = true;
    structureFooter.hidden = true;
    [['Tables', availableTables, 'table'], ['Equations', availableMath, 'math']].forEach(([title, items, type]) => {
      if (!items.size) return;
      const heading = document.createElement('h3');
      heading.textContent = `${title} (${items.size})`;
      const list = document.createElement('ol');
      list.className = 'local-editor-structure-list';
      items.forEach((item, id) => {
        const li = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        const index = [...items.keys()].indexOf(id) + 1;
        button.textContent = type === 'math' ? `${item.display ? 'Display' : 'Inline'} equation ${index}: ${structureValue(id, item.latex).replace(/\s+/g, ' ').slice(0, 95)}` : `Table ${index}: ${item.rows.length} rows × ${item.rows[0]?.length || 0} columns · ${item.rows[0]?.map(cell => structureValue(cell.id, cell.source)).join(' / ').slice(0, 80) || 'Empty header'}`;
        button.addEventListener('click', () => type === 'math' ? openMathEditor(item) : openTableEditor(item));
        li.append(button);
        list.append(li);
      });
      structureContent.append(heading, list);
    });
  }

  function bindStructures() {
    const register = (type, item, node, index) => {
      const label = type === 'math' ? `Edit ${item.display ? 'display' : 'inline'} equation ${index + 1}` : `Edit table ${index + 1}`;
      node.classList.add('local-editor-structured-target');
      node.title = label;
      let button;
      if (type === 'table') {
        button = document.createElement('button');
        button.type = 'button';
        button.className = 'local-editor-table-edit-button';
        button.textContent = label;
        button.hidden = true;
        node.before(button);
        button.addEventListener('click', () => { if (!busy) openTableEditor(item); });
      }
      const open = event => {
        if (!editing || busy) return;
        event.preventDefault();
        event.stopPropagation();
        if (type === 'math') openMathEditor(item);
        else openTableEditor(item, event.target.closest('[data-local-cell-id]')?.dataset.localCellId);
      };
      node.addEventListener('click', open, true);
      node.addEventListener('keydown', event => {
        if ((event.key === 'Enter' || event.key === ' ') && event.target === node) open(event);
      });
      structuredTargets.set(item.id, { type, item, node, button, label });
    };
    (snapshot.structures?.math || []).forEach(item => {
      const node = [...document.querySelectorAll('main article [data-local-math-id]')].find(candidate => candidate.dataset.localMathId === item.id && !candidate.closest('table, [data-local-editor-skip]'));
      if (!node) return;
      availableMath.set(item.id, item);
      register('math', item, node, availableMath.size - 1);
    });
    (snapshot.structures?.tables || []).forEach(item => {
      const node = [...document.querySelectorAll('main article table[data-local-table-id]')].find(candidate => candidate.dataset.localTableId === item.id && !candidate.closest('[data-local-editor-skip]'));
      if (!node || !item.rows?.length || tableCells(item).some(cell => ![...node.querySelectorAll('[data-local-cell-id]')].some(candidate => candidate.dataset.localCellId === cell.id))) return;
      availableTables.set(item.id, item);
      register('table', item, node, availableTables.size - 1);
    });
  }

  structuresToggle.addEventListener('click', openStructureChooser);
  structureApply.addEventListener('click', applyStructureDraft);
  structureBack.addEventListener('click', openStructureChooser);
  structureDialog.querySelector('#local-editor-structure-close').addEventListener('click', closeStructureEditor);
  structureDialog.querySelector('#local-editor-structure-cancel').addEventListener('click', closeStructureEditor);
  structureContent.addEventListener('click', event => { if (event.target.closest('a')) event.preventDefault(); });
  structureDialog.addEventListener('cancel', event => { event.preventDefault(); closeStructureEditor(); });
  structureDialog.addEventListener('close', () => { composing = false; structureReturnFocus?.focus(); });

  function message(text, error = false) {
    status.textContent = text;
    bar.classList.toggle('has-error', error);
  }

  function controls() {
    toggle.disabled = busy || !snapshot;
    metadataToggle.disabled = busy || !snapshot?.metadata;
    structuresToggle.disabled = busy || !snapshot || !(availableMath.size + availableTables.size);
    metadataFields.disabled = busy;
    if (metadataInputs.has('series_order')) metadataInputs.get('series_order').disabled = busy || !snapshot.metadata.writable.series_order || !metadataInputs.get('series').value;
    save.disabled = busy || dirtyCount() === 0;
    discard.disabled = busy || dirtyCount() === 0;
    undo.disabled = busy || dirtyCount() > 0 || !receipt;
    refresh.disabled = busy || dirtyCount() > 0 || !token;
    toggle.textContent = editing ? 'Finish editing' : 'Edit page';
    toggle.setAttribute('aria-pressed', String(editing));
  }

  function setEditing(value) {
    editing = value;
    document.body.classList.toggle('local-editor-active', value);
    nodes.forEach(({ node, segment }) => {
      if (value && !busy) {
        node.setAttribute('contenteditable', 'plaintext-only');
        node.setAttribute('role', 'textbox');
        node.setAttribute('aria-label', `Edit ${segment.field || segment.kind || 'text'}`);
        node.setAttribute('aria-multiline', 'false');
        node.tabIndex = 0;
        node.spellcheck = true;
      } else {
        ['contenteditable', 'role', 'aria-label', 'aria-multiline', 'tabindex', 'spellcheck'].forEach(key => node.removeAttribute(key));
      }
    });
    structuredTargets.forEach(({ node, button, label }) => {
      if (button) { button.hidden = !value; button.disabled = busy; }
      if (value && !busy) {
        node.setAttribute('tabindex', '0');
        node.setAttribute('aria-label', label);
        if (node.tagName !== 'TABLE') node.setAttribute('role', 'button');
      } else {
        ['tabindex', 'aria-label'].forEach(key => node.removeAttribute(key));
        if (node.tagName !== 'TABLE') node.removeAttribute('role');
      }
    });
    controls();
  }

  async function request(path, body) {
    const response = await fetch(api + path, {
      method: body ? 'POST' : 'GET',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify({ ...body, token }) : undefined,
      cache: 'no-store',
      credentials: 'same-origin'
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.message || `Request failed (${response.status})`);
    return data;
  }

  function noteChange(node, segment) {
    const text = node.textContent.replace(/[\r\n]+/g, ' ');
    if (text === segment.text) changes.delete(segment.id);
    else changes.set(segment.id, { id: segment.id, text, original: segment.text });
    node.classList.toggle('local-editor-changed', changes.has(segment.id));
    changedMessage();
  }

  function discardChanges() {
    nodes.forEach(({ node, segment }) => {
      node.textContent = segment.text;
      node.classList.remove('local-editor-changed');
    });
    changes.clear();
    metadataChanges.clear();
    structuredChanges.clear();
    markStructuredDrafts();
    renderMetadata();
    message('Changes discarded');
    controls();
  }

  function bind(snapshotData) {
    snapshot = snapshotData;
    const byId = new Map(snapshot.segments.map(segment => [segment.id, segment]));
    document.querySelectorAll('main article [data-local-edit-id]').forEach(node => {
      const segment = byId.get(node.dataset.localEditId);
      // A rendered token can be uneditable (e.g. smart punctuation). Never infer
      // the corresponding source from a similar-looking string.
      if (!segment || node.textContent !== segment.text || node.closest('table, pre, code, mjx-container, .highlight, [data-local-editor-skip]')) return;
      nodes.set(segment.id, { node, segment });
      node.classList.add('local-editor-text');
      node.addEventListener('input', () => { if (!composing) noteChange(node, segment); });
      node.addEventListener('compositionstart', () => { composing = true; });
      node.addEventListener('compositionend', () => { composing = false; noteChange(node, segment); });
      node.addEventListener('beforeinput', event => {
        const ranges = event.getTargetRanges?.() || [];
        const selection = window.getSelection();
        const crossesBoundary = ranges.length
          ? [...ranges].some(range => !node.contains(range.startContainer) || !node.contains(range.endContainer))
          : selection?.rangeCount && !node.contains(selection.getRangeAt(0).commonAncestorContainer);
        if (crossesBoundary) {
          event.preventDefault();
          message('Edit one text section at a time · equations and tables are protected');
          return;
        }
        if (event.inputType === 'insertParagraph' || event.inputType === 'insertLineBreak' || event.inputType.startsWith('format')) event.preventDefault();
      });
      node.addEventListener('paste', event => {
        event.preventDefault();
        const text = event.clipboardData.getData('text/plain').replace(/[\r\n]+/g, ' ');
        const selection = window.getSelection();
        if (!selection.rangeCount) return;
        const range = selection.getRangeAt(0);
        if (!node.contains(range.commonAncestorContainer)) return;
        range.deleteContents();
        const inserted = document.createTextNode(text);
        range.insertNode(inserted);
        range.setStartAfter(inserted);
        range.collapse(true);
        selection.removeAllRanges();
        selection.addRange(range);
        noteChange(node, segment);
      });
      node.addEventListener('drop', event => event.preventDefault());
      node.addEventListener('keydown', event => {
        if (event.isComposing) return;
        if (event.key === 'Enter') event.preventDefault();
        if (event.key === 'Escape') {
          event.preventDefault();
          node.textContent = segment.text;
          noteChange(node, segment);
          node.blur();
        }
      });
    });
    renderMetadata();
    bindStructures();
    message(nodes.size || snapshot.metadata || availableMath.size || availableTables.size ? 'Ready · changes stay in local files' : 'No editable text on this page');
    controls();
  }

  async function finishBuild(result, operation) {
    if (result.document) {
      snapshot.revision = result.document.revision;
      changes.forEach((edit, id) => { if (nodes.has(id)) nodes.get(id).segment.text = edit.text; });
      changes.clear();
      metadataChanges.clear();
      structuredChanges.clear();
      markStructuredDrafts();
      if (result.document.metadata) snapshot.metadata = result.document.metadata;
      renderMetadata();
    }
    receipt = operation === 'Last save undone' ? undefined : result.receipt;
    undoRevision = result.document?.revision || undoRevision;
    if (receipt) sessionStorage.setItem(`local-editor:undo:${url}`, JSON.stringify({ receipt, revision: undoRevision }));
    else sessionStorage.removeItem(`local-editor:undo:${url}`);
    sessionStorage.setItem(`local-editor:position:${url}`, String(window.scrollY));
    sessionStorage.setItem(`local-editor:editing:${url}`, editing ? '1' : '0');
    const limit = Date.now() + 180000;
    while (Date.now() < limit) {
      const build = await request('status');
      if (build.status === 'error') {
        const saved = operation === 'Preview refreshed' ? 'Preview rebuild failed.' : `${operation}, but preview rebuild failed.`;
        throw new Error(`${saved} Use Undo last save or Refresh preview; see the terminal for details.`);
      }
      if (build.status === 'ready') {
        changes.clear();
        sessionStorage.setItem(`local-editor:notice:${url}`, operation);
        // An external source edit changes the revision during Refresh preview.
        // Bind the checkpoint to the completed build, not the old open page.
        const rebuilt = operation === 'Preview refreshed' ? await request(`document?url=${encodeURIComponent(url)}`) : undefined;
        saveWidgetCheckpoint(rebuilt ? (rebuilt.document || rebuilt).revision : snapshot.revision);
        location.reload();
        return;
      }
      await new Promise(resolve => setTimeout(resolve, 600));
    }
    throw new Error('Saved to disk. Preview rebuild is taking longer than expected; check the terminal before continuing.');
  }

  async function submit(operation) {
    if (busy || composing || (operation === 'save' && !dirtyCount())) return;
    if (operation === 'save' && !prepareMetadata()) return;
    const edits = [...changes.values(), ...structuredChanges.values()];
    const metadata = operation === 'save' ? Object.fromEntries(metadataChanges) : {};
    const related_revisions = Object.fromEntries((snapshot.metadata?.related || snapshot.related || []).map(item => [item.path, item.revision]));
    busy = true;
    setEditing(editing);
    message(operation === 'save' ? 'Saving and rebuilding preview…' : 'Restoring the previous save…');
    try {
      const result = await request(operation, { url, revision: snapshot.revision, edits, metadata, related_revisions, receipt });
      if (result.changed === false || result.changed === 0) {
        changes.clear();
        metadataChanges.clear();
        structuredChanges.clear();
        markStructuredDrafts();
        if (result.document?.metadata) { snapshot.metadata = result.document.metadata; renderMetadata(); }
        nodes.forEach(({ node }) => node.classList.remove('local-editor-changed'));
        message('No changes to save');
        return;
      }
      await finishBuild(result, operation === 'save' ? 'Saved to local file' : 'Last save undone');
    } catch (error) {
      message(error.message, true);
    } finally {
      busy = false;
      setEditing(editing);
    }
  }

  toggle.addEventListener('click', () => {
    setEditing(!editing);
    sessionStorage.setItem(`local-editor:editing:${url}`, editing ? '1' : '0');
    if (!dirtyCount()) message(editing ? 'Click text, a table or an equation to edit' : 'Preview mode · changes stay in local files');
  });
  save.addEventListener('click', () => submit('save'));
  undo.addEventListener('click', () => submit('undo'));
  discard.addEventListener('click', discardChanges);
  refresh.addEventListener('click', async () => {
    if (busy || dirtyCount()) return;
    busy = true;
    setEditing(editing);
    message('Rebuilding preview from local files…');
    try {
      const result = await request('rebuild', {});
      await finishBuild({ ...result, receipt, document: undefined }, 'Preview refreshed');
    } catch (error) { message(error.message, true); }
    finally { busy = false; setEditing(editing); }
  });
  document.addEventListener('click', event => {
    if (editing && event.target.closest('a') && event.target.closest('[contenteditable]')) event.preventDefault();
  }, true);
  document.addEventListener('keydown', event => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
      event.preventDefault();
      if (structureDialog.open && currentStructure && !applyStructureDraft()) return;
      if (structureDialog.open && !currentStructure) closeStructureEditor();
      submit('save');
    }
  });
  window.addEventListener('beforeunload', event => {
    if (dirtyCount() || hasUnappliedStructureChanges()) { event.preventDefault(); event.returnValue = ''; }
  });

  // This origin is dedicated to the local editor; stale preview caches must not
  // substitute another generation of the source map or rendered document.
  async function initialize() {
    try {
      if ('serviceWorker' in navigator) {
        const registrations = await navigator.serviceWorker.getRegistrations();
        await Promise.all(registrations.map(registration => registration.unregister()));
      }
      const state = await request('status');
      token = state.token;
      controls();
      const result = await request(`document?url=${encodeURIComponent(url)}`);
      token = result.token;
      bind(result.document || result);
      const previous = sessionStorage.getItem(`local-editor:undo:${url}`);
      if (previous) {
        const state = JSON.parse(previous);
        if (state.revision === snapshot.revision && state.receipt === result.receipt) { receipt = state.receipt; undoRevision = state.revision; }
        else sessionStorage.removeItem(`local-editor:undo:${url}`);
      }
      setEditing(sessionStorage.getItem(`local-editor:editing:${url}`) === '1');
      setMetadataOpen(sessionStorage.getItem(`local-editor:metadata:${url}`) === '1');
      const incompleteWidgets = await restoreWidgetCheckpoint();
      const position = sessionStorage.getItem(`local-editor:position:${url}`);
      if (position) { window.scrollTo(0, Number(position)); sessionStorage.removeItem(`local-editor:position:${url}`); }
      const notice = sessionStorage.getItem(`local-editor:notice:${url}`);
      if (notice) { message(notice + (incompleteWidgets ? ' · some experiment settings could not be restored' : '')); sessionStorage.removeItem(`local-editor:notice:${url}`); }
    } catch (error) { message(error.message, true); }
  }
  initialize();
})();
