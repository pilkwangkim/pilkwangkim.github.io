# MS-BTE tutorial package

Paired posts: `Multi-Subband-Boltzmann-Transport-From-Scratch{,-KR}`.
Date: `2026-11-06 09:00:00 +0900`. Topic: `quantum-transport`.
Audience: undergraduate quantum mechanics; no assumed statistical mechanics or
kinetic-equation course. English is the mathematical source; Korean preserves
equations/code and uses natural prose with English technical terms.

## Physical and numerical contract

- Rectangular hard-wall wire, x transport and y,z confinement. Example widths
  10nm and 8nm, illustrative isotropic effective mass 0.19m0, one valley, spin2.
- Total energy E_n=epsilon_n+hbar^2 k^2/(2m). Epsilon includes electrostatic
  potential energy where present; do not add the same electric force twice.
- Occupation per spin is dimensionless, not a density. State measure g_s dk/2pi.
- Both spin populations are identical and scattering is spin-conserving. The
  effective wavefunction is an envelope and k is relative to the retained band
  minimum. Local x,k label a kinetic coarse-graining, not sharp joint eigenstates.
- Positive electron flux points +x; conventional charge current is negative
  for that flux. The positive electron-flow current is I_e=-I_charge.
- One-electron bath/scatterer gain/loss and Pauli blocking precede the reduced
  solver; the formula is not a general electron-electron collision kernel. Only the
  chosen reciprocal elastic same-energy model has exact Pauli cancellation.
- Uniform educational solver: symmetric nonnegative inverse-length kappa,
  C_a=|v_a|sum_b kappa_ab(f_b-f_a). DOS weights are proportional to1/|v_a|;
  weighted collision counts cancel. This is not calibrated material scattering.
- States ordered n+,n-. Closed subbands never enter the propagating-state graph.
  Intermode pairs have kappa=gamma/2, intramode reversal has kappa=beta_n.
- Only inflows are prescribed. A deliberately nonthermal mode-selective
  injector is an optional lab preset, not a thermal-reservoir boundary or a
  passive-filter simulation. Its second-mode occupation can exceed the first.
- First-order cell-centred upwind FV. Exactly conserved quantity is FACE flux,
  not the signed sum of occupations evaluated in one cell.
- One mode continuum T=1/(1+beta L), discrete
  T_N=(1+beta dx)/(1+beta L+beta dx). Exact conservation does not remove grid error.
- For identical coefficients and common inflow across M modes, symmetry gives
  alpha=beta+(M-1)gamma/2 and M times the one-pair flux. The scattering lab
  reports the transmission-factor grid error only when this benchmark applies;
  unequal mode injection is explicitly marked inapplicable.
- Fixed prescribed edges, elastic independent energy shells. No implemented
  self-consistent Poisson, phonon energy exchange, longitudinal coherence,
  tunneling, or quantitative material/device calibration.
- Separate spatial, quadrature, energy-cutoff, and retained-mode convergence.
  First three explicit modes are not an automatically converged full wire.
- For the reference's constant-coupling intervals, default integration is
  analytic using finite differences of Fermi integrals. Optional
  `integration="gauss"` resolves both chemical potentials on the kBT scale.
  The return value names the integration method. This prevents the old
  edge-only 24-point rule's approximately 4.98% error in the 10K regression.

## Assets and reproducibility

- `assets/code/multi-subband-bte.py`: NumPy reference, block-tridiagonal solve,
  arbitrary symmetric collision graphs, complete hard-wall mode enumeration,
  energy integration and diagnostic arrays. Python3.9+ and NumPy required.
- `tools/test_msbte_physics.py`: confinement, collisions, equilibrium,
  positivity, face flux, exact and discrete one-mode limits, independent
  two-mode continuum profiles, low-temperature integration, worked-example
  arithmetic, and convergence; elementary BTE limits and density-matrix reduction
  add independent checks (45 tests total).
- `assets/js/article-labs-msbte.js`, scoped CSS and
  `_data/article_labs_msbte.yml`: five bounded bilingual labs. Pure functions
  exported for Node tests, validated state links, Reset, no user code execution.
- `tools/msbte_figures.py` and `tools/msbte_reduction_figures.py`: native scientific
  SVGs, desktop/mobile layouts, optional PNG via librsvg. Seven body figures and
  a cover; force example folded. New figures distinguish streaming/relaxation and
  exact basis change/coherence deletion/state truncation with actual data.
  The cover is an equation-led plate: subband energy, scalar MS-BTE occupation
  balance, and the local electron-flow current identity. It has no separate
  free-flight plot or BTE-to-MS-BTE transformation arrow. Equations are LaTeX
  paths embedded in SVG; rebuilding covers requires `latex` with amsmath/xcolor
  and `dvisvgm`. The finished SVG/PNG has no runtime font or MathJax dependency.
  Use `--cover-only --png` to rebuild it without rewriting body figures.
  `--body-only --png` rebuilds figures 01–05 without touching the cover or
  streaming/reduction assets.
- `tools/build_msbte_notebooks.py`: executes actual post snippets in order,
  embeds body figures, removes Liquid/Kramdown syntax, resolves browser links.
  Exercise starter is not replaced by the reference answer during Run All.
  The checker is also tested against mutations of f, v, and K.
  Twelve code cells include exact elementary BTE limits, storage arithmetic,
  a density-matrix reduction example, and scattering-to-ampere integration.
- `tools/check_msbte_article.py`: EN/KR equation/code/asset/reference parity,
  source hygiene, Python3.9 syntax and notebook integrity.

## Validation commands

Use an interpreter with NumPy; a task-local virtual environment is sufficient.

```sh
python3 tools/msbte_figures.py --png
python3 tools/msbte_reduction_figures.py --png
python3 tools/test_msbte_physics.py
node tools/test_article_labs_msbte.js
python3 tools/build_msbte_notebooks.py
python3 tools/check_msbte_article.py
node tools/check_article_labs_models.js
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/test_article_lab_assets.rb
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/test_responsive_pictures.rb
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/check_blog_metadata.rb --output /tmp/msbte-metadata.json
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/ecc_render_metadata.rb /tmp/msbte-metadata.json /tmp/msbte-routes.json
JEKYLL_ENV=production PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec jekyll build
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/check_article_labs_output.rb --site _site --metadata /tmp/msbte-routes.json
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec htmlproofer _site --disable-external
```

Inspect both languages on desktop and at 390px/320px mobile widths. Open all
folds, confirm MathJax has no error nodes, inspect responsive sources and table
column widths, and operate every lab. Unit tests do not establish visual QA.
Keep build destinations and QA receipts outside tracked source. No commit,
push, or deployment follows automatically from authoring or validation.

## Undergraduate-QM reading review, 2026-11-06

- Added worked bridges for envelope states, population versus coherence,
  reservoir statistics, DOS-weighted counts, and scattering-to-ampere integration.
  Longer examples remain folded; both languages retain identical math and code.
- Both notebooks executed nine cells each. All 26 reference physics tests and
  2,008 lab-model plus 216 UI/state checks passed. Production build, metadata,
  generated lab output, and internal HTML-Proofer checks passed.
- Both rendered languages were checked at 1280px, 390px, and 320px, with all
  folds open: 40 display-math blocks, no MathJax errors or raw math delimiters,
  two correctly proportioned tables, and no page-width overflow. Long mobile
  equations scroll within their own containers. All four labs were operated,
  including all four quiz answers in both languages.
- Confirmed the updated asset after the production preview's PWA Update action;
  the mode-2 control describes an independent incoming occupation, not a filter.
- Publication date, routes, and all 20 body-figure files were preserved. No
  commit, push, or deployment was performed for this review.

## BTE-first restructuring, 2026-11-06

- Both posts now have 19 sections: occupation balance and BTE, two solvable
  limits, full phase-space cost, transverse confinement, and the distinct steps
  leading to population MS-BTE, followed by contacts, collisions, conservative
  transport, current, electrostatics, alternatives, and exercises.
- A complete basis expansion preserves information. Longitudinal semiclassical
  kinetics, population closure, and Markov/rate assumptions are physical
  approximations; retaining a finite mode set is a separate truncation. The
  density-matrix example is a normalized one-electron toy, not the normalization
  of a device occupation. The partial-Wigner bridge remains folded.
- New streaming/relaxation figure and lab use two independent initial-value
  problems, not a combined spatial scattering simulation. New matrix figure
  displays exact rotation, diagonal approximation, and unnormalized projection.
  Existing cover and figures 01–05 are unchanged.
- The storage comparison is a direct tensor-grid count versus a chosen
  ten-mode population count, not an equal-accuracy speedup benchmark. Mode-pair
  counts, eigenproblems, scattering and Poisson overhead, Monte Carlo, angular
  expansions, moments, and real/mode-space quantum transport are distinguished.
  Confinement subbands are not identified with Brillouin-zone folding.
- Final checks: 34 Python physics tests; 6,996 lab-model and 304 UI/state tests;
  121 shared-runtime factories; two executed notebooks with 12 cells and seven
  embedded figures each. Source parity covers 42 numbered equations, 49 display
  blocks, 12 snippets, five labs, seven responsive body figures, and 15 references.
  All nine original teaching code cells are preserved.
- Production build, metadata/routing, rendered lab output, and HTML-Proofer
  passed (434 HTML files, 4,276 internal links). All 28 old heading/lab anchors in
  each rendered language remain available, including the relocated BTE anchor.
- Both languages were inspected at actual viewport widths 1280, 390, and 320px.
  All 27 folds were opened in each language: 49 display blocks, zero MathJax
  errors or raw delimiters, seven loaded responsive figures, two tables, and no
  page-width overflow. Long equations/code scroll within their own containers.
  The new lab was operated in both languages; the narrow table labels were
  shortened and the final production content verified after the PWA update.
- Date, routes, classification, solver physics and unrelated assets were
  preserved. No commit, push, or deployment was performed.

## Independent physics audit and body-figure polish, 2026-11-06

- Checked all 42 numbered equations, numerical examples, units, exercise
  answers, collision weights, boundary conditions, and current signs in both
  languages. Eq. (2) now includes the first-order trajectory's O(dt²) remainder.
  Population closure now explicitly addresses initial/contact coherence, not
  weak coupling alone; a verified SISPAD 2005 primary reference was added.
- The coding checker verifies that f, v and K are all unchanged, using
  untouched snapshots for its oracle, and rejects outputs sharing input memory.
  Notebook generation rejects deliberately input-mutating candidates as well.
- The default three-mode continuum current is 2.029909021 µA, versus
  2.187730239 µA for the 24-cell FV solution with analytic energy integration
  and 2.187451852 µA for the teaching midpoint integral. The text separates
  +7.7748% spatial error from -0.012725% midpoint error on that same grid.
- The scattering lab exposes the exact common-inflow transmission benchmark.
  At beta=gamma=.04/nm and N=12 its 40% error is visible despite conserved
  face flux; N=36 reduces it to 13.95%. Unequal mode inflows show inapplicability.
  JS now rejects sparse state/energy arrays and nonzero kappa diagonals.
- All seven body figures were polished in desktop/mobile layouts. Figure 2
  has energy-ordered mode labels, a common physical density scale, and a
  separate dispersion legend. Contact arrows distinguish inflow/outflow;
  elastic energy stacks sum to the same total energy. Mode colors agree with
  the labs. Four cover files and paired front matter remain byte-identical.
- Final checks passed: 42 Python tests, 7,272 JS model checks, 328 UI/state
  checks, and two executed 12-cell notebooks with seven embedded figures each.
  Independent Python/JS comparisons cover 59 geometry/contact/transport cases.
  The source has 49 display blocks and 16 reference definitions in each language.
- Production build, metadata/routing, generated lab output, and HTML-Proofer
  passed (434 files, 4,276 links). Source/build parity covers 36 package assets.
  The pre-existing duplicate site.webmanifest build warning is unrelated.
- Actual browser CSS widths 1280, 390 and 320px were checked in both languages.
  All 27 folds per language were opened: zero MathJax errors and no page-width
  overflow; both tables retain proportional columns. Long mobile equations
  scroll locally. All five labs and eight paired quiz answers were operated;
  the generated experiment link restored its state (OS clipboard not verified).
  Screenshots and DOM receipts are outside the repository. Viewport overrides
  were reset. No commit, push, or deployment was performed.

## Equation-led cover and Korean copyedit, 2026-11-06

- Replaced the cover's three plot panels and BTE-to-MS-BTE arrow with an
  equation-led plate: parabolic subband energy, scalar occupation balance,
  and local electron-flow current. Desktop/mobile retain their existing sizes.
  LaTeX paths prevent missing mathematical glyphs; all embedded glyph references
  are local and unique. Signs, units and mobile line continuation were reviewed.
- Reworded all 19 Korean sections, folds and captions. Literal open/closed
  subband translations now describe whether propagating states exist at the
  specified energy; the nonzero-scattering and virtual-coupling qualifications
  remain. The Korean summary, 17 lab strings and three registry strings were
  also revised. English body and all non-Korean JS source remain unchanged.
- Protected content matches the preceding version: 49 display blocks, 484
  inline math spans, 12 Python examples, 1,208 body numeric tokens, IDs, links
  and reference definitions. All 28 body-figure files, reference solver and CSS
  are unchanged. Both 12-cell notebooks were regenerated and executed.
- Source/model checks, metadata, final production build, generated lab output
  and HTML-Proofer passed. Both languages were checked at actual CSS widths
  1280, 390 and 320px; correct cover variants, zero MathJax errors and no page
  overflow. All 27 Korean folds and revised lab states/quiz feedback were checked.
  The final Korean summary was verified after the PWA Update action.
- Date, routes, classification and article titles remain unchanged. No commit,
  push or deployment was performed; screenshots and receipts are outside source.

## Korean idiom and semantic-preservation pass, 2026-11-07

- Reread all 19 Korean sections, folds, captions, exercises and references.
  Reworked 41 source lines, principally paragraphs with abstract subjects,
  noun stacks, ambiguous predicates and English-shaped syntax. Technical
  terms remain English; observable and time complexity are named consistently.
  Self-consistency now states a distribution-to-potential fixed-point condition.
- Updated 15 Korean lab strings and four registry notes. Left/right contact
  equality is distinguished from equal mode inflows at each contact. All
  non-Korean JS source, identifiers, computations and control ranges are unchanged.
- Preserved front matter, English article/notebook, all 49 display blocks,
  484 inline math spans, 12 Python blocks, 25 inline code spans, 28 HTML IDs,
  reference definitions and numerical tokens. Figures, covers, solver and CSS
  remain unchanged. Rebuilt notebooks execute 12 cells per language.
- An independent changed-paragraph review caught a weakened equal-accuracy
  comparison qualifier; the original storage-versus-simulation distinction
  was restored before the final notebook/build.
- Source checks, 7,272 JS model and 328 UI/state checks, shared-runtime tests,
  metadata/routing, production build, generated lab output and HTML-Proofer pass.
  Korean was inspected at actual 1280/390/320px CSS widths with all 27 folds
  opened, zero MathJax errors and no page overflow. All five revised labs were
  operated; an English mobile smoke check also passed. Receipts/screenshots
  are outside source. No commit, push or deployment was performed.

## Pre-commit verification, 2026-11-07

- Independently rederived the signs, units, state-counting weights, boundary
  conditions and conservation statements in all 42 numbered equations. Checked
  worked examples, exercise answers and the Korean meaning against English.
  No additional physics correction was needed within the stated educational
  model. This is not material/device calibration or a validation of omitted
  coherence, inelastic scattering or self-consistent electrostatics.
- Added retained regressions for 1 K injection exactly at a subband edge,
  degenerate thresholds, extreme bias, mode enumeration, unit conversion,
  spatial rescaling, stiff solver limits and hidden-mode state restoration.
  All 45 Python tests, 7,382 JS model checks and 346 UI/state checks pass.
  Shared-runtime checks pass: 121 factories, 246 language/preset cases and
  3,538 renders. Both notebooks were rebuilt and all 12 cells executed in each;
  source/attachment parity was also checked. The article checker now rejects
  stale notebook code and out-of-order execution counts.
- Fixed the mobile Figure 1 time-axis/legend overlap. Replaced one slash-joined
  English table expression with ordinary words to avoid mid-word wrapping.
  On this article's mobile layout only, hide the shared floating back-to-top
  button that obscured figure labels and table cells. Desktop and other articles
  retain that button. No solver, lab computation, Korean prose, cover or
  publication metadata changed in this pass.
- Built the final production site, checked metadata and routes, validated the
  generated lab bundles and passed HTML-Proofer: 434 HTML files, 4,276 internal
  links. The pre-existing duplicate site.webmanifest warning is unrelated.
- Inspected both rendered languages at actual CSS widths 1280, 390 and 320px
  on a fresh localhost origin. Opened all 27 folds per language: 49 display-math
  blocks, no MathJax errors or raw delimiters, seven loaded responsive figures,
  two proportionally sized tables and no page-width overflow. Mobile formulas
  scroll inside their own containers; a 320px horizontal-scroll interaction
  confirmed this. All 16 native figure layouts and article-width raster views
  were checked, then actual browser screenshots were reviewed independently.
- Operated all five labs in both languages, including equilibrium, reverse
  current, closed-mode exclusion, unequal injection and the coarse-grid error
  benchmark. Checked all four correct quiz answers per language, wrong-answer
  feedback, Reset and generated-link restoration. OS clipboard contents were
  not verified. Screenshots and DOM receipts are outside source; viewport
  overrides were reset. Nothing was staged, committed, pushed or deployed.
