# Interactive article components

## Article-local models and record explorers

The shared article-lab system adds 74 distinct examples to 52 articles (100 English/Korean source pages). Existing prose, equations, figures, routes and metadata stay intact; posts add an `interactive_labs` bundle declaration and includes beside the relevant explanations. Eight prose-led essays retain their reading layout. Two essays with explicit numerical models have optional calculations; the Assignment essay uses its Compact edition. Original prose editions stay unchanged.

| Bundle | Lab types | Articles | Examples |
| --- | ---: | ---: | --- |
| Quantum Circuit | 7 | 1 | Relative phase, clean scratch, normalization/Newton, GF(4), Givens order/precision, stage counts, resource cost |
| Kaggriculture | 8 | 4 | Delayed return, BC gradients, PPO clipping, four investment plans, replay alignment, world bootstrap, harvest timing, Fibonacci wages |
| AI Agent Security | 14 | 12 | Score cells, per-model filling, cost, versioned deadlines, ordered prefixes, paired portfolios, stage accounting |
| Gemma 4 | 6 | 2 | Bug repair, patch boundary, reconstructed parser, context admission, deadline model, paired outcomes |
| RSNA | 9 | 6 | Ranking/AUC, resampling geometry, target/window losses, reducers, budgets, teacher folds, 154 recorded submissions |
| BioHub | 9 | 9 | Matching and scored edges, folds, graph pruning, reach, tail mixtures, divisions, alignment, registration, comparison arms |
| CASMI | 4 | 1 | Public spectra, mass differences, cosine matches, ranking, adduct/ppm filtering |
| ARC-AGI-3 | 3 | 3 | RHAE, token allocation, mismatch gate |
| ROGII | 3 | 3 | Authored GR alignment, mean/mode risk, five recorded Public/Private pairs |
| Predictive modeling | 5 | 5 | Thresholds, target encoding, measured color presets, class multipliers, as-of cutoff |
| Game agents | 3 | 3 | Orbit intercept, Maze transfer/jump, legal Pokémon action blending |
| BirdCLEF | 1 | 1 | Column ranks, row masks, taxonomy smoothing, gated movement budget |
| Essays | 2 | 2 | Assignment totals, two-variable covariance |

All calculations run in the browser; no models, competition services or external APIs are called. Measured presets use the article's bundled data. Adjustable illustrations identify their exact assumptions in **Model & data**. Reset restores every input and custom stepper. Empty or out-of-range numeric fields preserve the last figure until a valid value or preset is supplied.

Wide layouts place controls beside figures. Narrow layouts use two columns of controls and wrapping labels/legends, with bounded scrolling inside result tables. Widget roots are excluded from MathJax and the local source editor: changing a simulation does not change the post or create editable table/math targets. Source tables and equations remain editable through the existing panel.

Assets load only on posts declaring the bundle. The shared runtime/CSS/include and each module/registry have content fingerprints; declared JSON/CSV bytes also change the bundle's fingerprint and data URL. To add a model, follow [the integration contract](article-labs-api.md).

Verification:

```sh
for test in tools/test_article_labs_*.js; do node "$test" || exit; done
node tools/check_article_labs_models.js
bundle exec ruby tools/check_article_labs_output.rb \
  --site /path/to/built/site --metadata /path/to/metadata-report.json
```

The model-contract check exercises both languages, actual article presets, select options, range endpoints, checkboxes, buttons, resets, local data reads and finite in-domain chart coordinates. It also checks native input step grids. It is a DOM contract stub; real browser interaction and responsive review remain separate checks.

## NEGF resonant tunneling

The English and Korean NEGF articles include a resonant-tunneling calculator immediately after Figure 8. Preview locally with `./tools/edit.sh`; open the article and go to **Resonant tunneling lab / 공명 터널링 실험실** in section 8.

- Barrier height: 0–0.65 eV. Well width: 6–30 nm, in 1 nm steps.
- Probe energy: 0.001–0.600 eV. Click the transmission plot or an energy button below it to inspect a resonance. The buttons preserve the exact peak energy even when its width is narrower than the slider increment.
- Reset restores the source notebook's Example A geometry: 0.40 eV barriers, a 16 nm well, 3 nm barrier thickness and a 100-site device.

Controls, readouts and Reset sit beside the plots when the article has enough room. Narrow views use compact rows or stacked controls; Model details expands the fixed model assumptions.

The calculator uses the article's zero-bias coherent finite-difference NEGF model, with effective mass 0.067 m₀ and 1 nm grid spacing. It calculates potential energy U(x) and single-channel transmission T(E). The first resonance's FWHM is displayed in µeV. It does not calculate occupation or current.

The component is `_includes/interactive/negf-barrier.html`; its styles and scripts are `assets/css/negf-interactive.css`, `assets/js/negf-core.js` and `assets/js/negf-interactive.js`. Only posts with `interactive: negf-barrier` load these assets. Their shared fingerprint changes when any component asset changes. Interactive descendants are excluded from the local visual editor; moving a slider does not modify the post source.

Numerical checks:

```sh
node tools/test_negf_core.js
python3 tools/check_negf_reference.py --node node --output /tmp/negf-reference.json
```

The second command needs NumPy and independently solves the dense 100 × 100 complex device system. The Node suite includes all 1,650 allowed barrier-height/well-width combinations and adaptive sampling of narrow resonances.

The article's Example A occupation figures use a separate resonance-resolved integration grid. The original 3,000-point linear mesh did not resolve the narrow first occupied resonance. Reproduce the grid, cutoff and regulator checks with:

```sh
python3 tools/check_negf_occupation.py --output /tmp/negf-occupation.json
```

This check needs NumPy and Node.js. It independently solves the two endpoint Green-function columns, compares them with dense solves, and checks site/spectral occupation, grid refinement, the Fermi-tail cutoff and the numerical regulator. At `eta=1e-9 eV`, refinement 16 and cutoff `mu_L + 18*kT`, the well contains about 4.87203 electrons and the Landauer current is about 263.0911 nA. The browser calculator itself continues to compute only potential and transmission.

Regenerate the shared workflow diagram and Figures 9–10 with a Python environment containing NumPy and Matplotlib:

```sh
python3 tools/negf_flow_figure.py
python3 tools/negf_example_a_figures.py
```

The workflow has desktop and mobile SVG/PNG versions. Figures 9–10 also have SVG companions, and both use the same quadrature settings as the occupation check. Plotting samples are independent of quadrature weights, so changing the image resolution does not change the reported integrals.
