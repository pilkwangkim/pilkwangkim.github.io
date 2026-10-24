# Article labs integration contract

Each bundle owns `assets/js/article-labs-NAME.js`, `_data/article_labs_NAME.yml`, its focused `tools/test_article_labs_NAME.js`, and assigned post inserts. Do not edit the shared runtime, shared CSS, layout or generator from a bundle task.

Post front matter: `interactive_labs: [NAME]`. Include at the relevant section: `{% include interactive/article-lab.html id="NAME-CONCEPT" %}`. Keep all existing article bytes other than these additions. The title, source data and formula conditions belong to that section. EN and KO share identifiers and use their own labels.

Registry entry:

```yaml
NAME-CONCEPT:
  bundle: NAME
  title: {en: "Short title", ko: "짧은 제목"}
  note: {en: "Precise model/data conditions.", ko: "정확한 모형·자료 조건."}
  data: [/assets/path/to/optional-public-data.json]
```

Browser bundle registers factories with `window.ArticleLabs.register('NAME-CONCEPT', ctx => {...})`. An async factory is supported. Pure functions should also be exposed through `module.exports` for Node verification; no DOM initialization in Node.

The context has `root`, `ui` (controls), `view` (charts/content), `metrics` and `status` (a status paragraph). Keep the main view compact, ideally 1–2 plots or a bounded stepper. Fields and readouts live beside the view on wide screens. The common header supplies Reset.

- `ctx.text(en, ko)` returns the current language string; `ctx.ko` is boolean.
- `ctx.el(tag, attrs = {}, children = [])` creates HTML; children may be a string, node or array. Set event listeners normally. `class` is an ordinary attribute. No inline event attributes/HTML strings.
- `ctx.svg(tag, attrs = {}, text)` creates an SVG node.
- `ctx.control(enLabel, koLabel, options)` appends and returns a native input/select. Options: `{type:'range'|'number'|'select'|'checkbox', value, min, max, step, options:[{value,en,ko}]}`. Read `.value` / `.checked`. Checkbox `value` is its initial boolean.
- `ctx.metric(enLabel, koLabel)` appends a labelled output and returns that output; update `.textContent`.
- `ctx.button(enLabel, koLabel, handler)` appends a button to controls and returns it.
- `ctx.onChange(render)` delegates native input/change events and invokes `render()` immediately. Custom buttons may call the same render function. Reset restores all native fields and runs render.
- `ctx.onReset(handler)` optionally resets custom JS state before the common render.
- `ctx.note(en, ko)` replaces the collapsible model/data note.
- `ctx.plot(spec)` returns a chart wrapper with an accessible SVG (viewBox `0 0 640 200`), wrapping HTML axis labels and an optional legend; append it to `ctx.view`. Spec: `{x:[min,max],y:[min,max],xLabel,yLabel,title,series:[{points:[[x,y],...],color:'blue'|'gold'|'muted'|'red',dash:false,label}],markers:[{x,y,label,color}],bars:[{x,y,width,color,label}],bands:[{x0,x1,y0,y1,color}],xTicks:[number...],yTicks:[number...]}`. Optional ticks; defaults are generated. No chart clipping to hide invalid model values. Keep units explicit.
- `ctx.table(headers, rows)` returns a horizontally contained semantic table. Arrays contain display strings/numbers.
- `ctx.format(number, digits = 4)` formats numeric values.
- `ctx.asset('/assets/...')` adds site baseurl.
- `await ctx.fetchJSON(path)` and `await ctx.fetchCSV(path)` load bundled public data. CSV returns an array of objects and supports quoted cells.

The root has `data-local-editor-skip` and `tex2jax_ignore`; no application state is saved to post source. Initial controls are unavailable until the factory succeeds. Errors show a concise status with retry via Reset. Do not add model/API calls, invented measured results or new dependencies. Historical records use discrete presets; synthetic examples state their exact assumptions. Module assets are fingerprinted and loaded only by posts naming that bundle.

Declare every fetched JSON/CSV path in the registry's `data` array. Those source bytes participate in the bundle fingerprint and the versioned data URL.

Registry fingerprints use sorted-key UTF-8 JSON rather than Ruby object inspection, so local and CI Ruby versions produce the same asset URLs. Run `bundle exec ruby tools/test_article_lab_assets.rb` to check this contract.
