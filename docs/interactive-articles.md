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

Regenerate the shared workflow and the article's figures with NumPy and Matplotlib:

```sh
python3 tools/negf_flow_figure.py
python3 tools/negf_foundation_figures.py
python3 tools/negf_transport_figures.py --notebook /path/to/negf_from_scratch.ipynb
python3 tools/negf_example_a_figures.py
python3 tools/negf_ndr_figures.py
```

`negf_foundation_figures.py` draws Figures 1–4 and the separate horizontal cover. `negf_transport_figures.py` draws Figures 5–8 from selected calculation cells in the [audited notebook revision](https://github.com/pilkwangkim/Physics/blob/7fd9f32c390e117ab6b04bb7f787fc5b43c959e2/negf_from_scratch.ipynb); it verifies the notebook SHA-256 before executing those cells. The source notebook is never modified. The remaining two generators draw Figures 9–10 and 11–12 respectively.

Every figure has desktop and mobile SVG/PNG versions. The article embeds the SVGs through `.negf-figure-desktop` / `.negf-figure-mobile` wrappers, switching to stacked panels at 640 px. SVG glyphs are outlined to preserve math fonts across browsers. Primary labels target roughly 16–18 px at the article's 691 px desktop and 358 px mobile content widths; ticks and legends remain readable without opening the image zoom. Panel (e) in Figure 12 is the original full-sweep inset, enlarged without changing its data.

Figures 9–10 use the same quadrature settings as the occupation check. Plotting samples are independent of quadrature weights, so changing the image resolution does not change the reported integrals. The interactive calculator uses larger axis text and stacked plots at intermediate widths; its numerical solver is unchanged.

## A* graph-search tutorial

The paired `A-Star-From-Scratch` articles use `interactive_labs: [pathfinding, search-unweighted, search-dijkstra, search-astar]` and thirteen local activities. The original activities and the final coding exercise are:

- `astar-traversal`: DFS/BFS stepper with a stack/queue frontier, map editing, Run/Pause, and Reset.
- `astar-weighted`: BFS, Dijkstra, greedy, and A* on one editable weighted map; four/eight moves, compatible heuristic choices, no corner cutting, and a heuristic multiplier from 0 to 3.
- `astar-priority-lab`: a bounded arithmetic-expression editor in `g` and `h`, not an arbitrary code interpreter. Three test maps compare its returned cost with Dijkstra; passing them is not a proof of a universal guarantee.
- `astar-quiz`: one five-problem workbench with an input diagram, wrapping answer choices, hint and immediate reasoning for each problem. The former separate section-17 exercises are consolidated here: goal acceptance, reopening, movement rules, boundary cases and a road-time/teleport lower bound. Previous/Next preserves answers; Reset clears them. Previous exercise heading URLs select the matching question.
- `astar-code-test`: guided A* completion rather than arbitrary Python execution. The reader supplies a bounded `g`/`h` priority expression, a comparison of `candidate` with `old_g`, and goal-pop versus discovery-time acceptance. Eight nonnegative-cost graph cases test costs, reopening, zero-cost cycles and stale entries; two also impose explicit expansion budgets. On the 15×15 open grid, `g+h` expands 28 states and `g` expands 224 under the stated lower-h tie rule. Dijkstra's costs remain correct; its map-specific effort budget fails. Every run stops after at most 5,000 raw heap removals, with a limit result distinct from no route. A reference button fills the three decisions and runs the tests. The full Python scaffold and fixtures are downloadable at `/assets/code/astar-code-test.py` and embedded in both notebooks.

Eight additional activities give DFS, BFS, Dijkstra and A* different problems to solve:

- `search-dfs-maze` (`search-unweighted`): seeded randomized DFS carves a maze one passage at a time; after generation, the reader walks through it. The construction stack and the player's route are separate. A completed N-cell maze is a tree with N−1 passages.
- `search-dfs-queens` (`search-unweighted`): manual N-Queens placement and a DFS backtracking trace for N=4, 5 or 6. Candidate checks, conflicts, pruning and returns are visible; the solver enumerates all solutions independently of the manual board. A saved-solution gallery shows the completed arrangements without changing the search counts and distinguishes them from the current DFS stack.
- `search-bfs-words` (`search-unweighted`): build a word ladder with one-letter changes, then compare it with BFS layers on the complete bundled dictionary. Each allowed change costs one move.
- `search-bfs-jugs` (`search-unweighted`): fill, empty or pour between two water jugs, then inspect the shortest BFS action sequence. The 2L/4L target-3L preset is unreachable; a pour stops only when the source empties or the destination fills.
- `search-dijkstra-delivery` (`search-dijkstra`): plan an A→H delivery on an eight-stop weighted network, undo moves, check the result, and change traffic or road closures. The shortest normal route takes 10 minutes rather than the direct road's 18. Queue removals, candidate relaxations and obsolete entries appear separately; goal acceptance occurs on a valid minimum-g pop.
- `search-dijkstra-dispatch` (`search-dijkstra`): move two stations and select an emergency stop. Multi-source Dijkstra initializes both stations at g=0 and runs to queue exhaustion, assigning each reachable stop its fastest station and arrival time. Equal times go to the alphabetically earlier station. All road costs are fixed positive minutes on an undirected schematic, not geographic distances.
- `search-astar-tiles` (`search-astar`): manually solve an 8-puzzle, request a hint, compare Manhattan A* with h=0 Dijkstra, and play the solver's solution separately from the manual board. Presets have exact optimal distances 4, 8 and 12, independently verified by reverse BFS. Manhattan excludes the blank; long searches run in scheduled bounded batches.
- `search-astar-detective` (`search-astar`): inspect a four-state directed counterexample, select heuristic presets, and toggle reopening. The admissible-but-inconsistent preset returns cost 6 without reopening and the optimal cost 5 with it; an overestimated h can return 6 even with reopening. The h* values and consistency checks expose the conditions behind each result.

In the original weighted-map activity, costs are `step_length × destination_terrain`, with terrain 1 or 5 and lengths 1 or √2. The start has no entry charge. Every map edit restarts the search. Heap ties prefer smaller `h`, then insertion order; strict improvements reopen states, and stale entries are discarded before goal testing. Expansions count successor examinations, including repeats after reopening, but exclude the goal. Frontier/peak readouts count distinct active states rather than raw heap records. Each added activity declares its own state, action costs, tie rules and stopping condition in **Model & data**; the grid-cost convention does not apply to the word, jug, network or tile problems.

The **Figure 7 comparison maze** preset exactly matches the static comparison figure's map, paths, costs, and expansion traces. The simpler **Expensive direct road** preset is intentionally a different map with a different optimal cost. `figure-data.json` records the static inputs and results; Weighted A* figures also include Python-only weights above the browser's limit of 3.

The original bundle owns `_data/article_labs_pathfinding.yml`, `assets/js/article-labs-pathfinding.js`, and `assets/css/pathfinding-lab.css`. The three added bundles each own the matching `article_labs_search_*.yml` registry, `article-labs-search-*.js` script and `search-*.css` stylesheet. All four scripts export pure functions for Node tests and register browser activities without external dependencies, `eval`, or `Function` construction. The grid and maze boards support arrow-key navigation; network stops and legal moves also support keyboard activation. Reset restores native fields and custom board/search state and cancels running animation timers.

Reproduce figures, self-contained notebooks, source parity, and algorithm checks:

```sh
python3 tools/astar_figures.py
python3 tools/build_astar_notebooks.py
python3 tools/check_astar_article.py
node tools/test_pathfinding_core.js
node tools/test_search_unweighted.js
node tools/test_search_dijkstra.js
node tools/test_search_astar.js
node tools/test_astar_exercises.js
node tools/test_search_checkpoints.js
node tools/check_article_labs_models.js
bundle exec ruby tools/test_article_lab_assets.rb
```

The focused added-bundle tests use independent references: maze connectivity and cycle checks, queen row-permutation enumeration, Floyd–Warshall word/network distances, integer-transfer jug relaxation, and reverse BFS over the reachable 8-puzzle state space. They also exercise bilingual controls, manual moves, invalid actions, deterministic ties, road closures, stale records, presets, animation/Reset behavior and the deliberately incorrect no-reopening variant. These checks verify the declared finite examples and implementations; they do not turn a tested heuristic into a general optimality proof.

The figure generators need NumPy, Matplotlib, and Pillow. In addition to `tools/astar_figures.py`, `tools/astar_exercise_figures.py` creates six answer-free problem diagrams with desktop/mobile SVG and PNG variants. The new figures have renderer checks for text bounds/collisions, arrow intersections and node-label fit; their graph inputs and the three grid cost models are independently checked.

The notebooks use only Python's standard library, embed fourteen figures (eight article figures and six exercise inputs), and execute seventeen Python cells each during generation. The original models include word-ladder, multi-source dispatch and tile-puzzle code. The five illustrated problems and the coding scaffold/fixtures come from `/assets/data/astar-exercises.json`, the same source as the browser activities. The intentionally unfinished TODO starter reports failed tests; a separately named reference solution must pass all eight. Notebook cells are tagged `exercise-starter`, `exercise-tests` and `exercise-solution`; the reference cell starts source-hidden in compatible Jupyter interfaces. The generator also writes the standalone Python challenge and judge from these exact inputs. No runtime download is needed for the notebook exercises.

Optional `<details>` become ordinary notebook subheadings because HTML folds cannot span multiple Jupyter cells. The other eleven browser-only activities link to the corresponding article section. English and Korean code, equations, figure sources/dimensions, and lab IDs are checked together. `check_astar_article.py` additionally guards Markdown-sensitive absolute-value bars, superscript stars, blog-only attribute leakage, all fourteen embedded images and the generated challenge cells. The problem JSON and all twelve browser SVGs participate in the pathfinding bundle fingerprint; image URLs carry that version.

Figures have desktop/mobile SVG and PNG variants with outlined SVG text. At 640 px the article switches to the vertically arranged mobile figures. Long display equations are broken into readable steps, with article-local horizontal containment as a fallback. Production builds and real browser checks at desktop/mobile widths remain required in addition to the Node and source checks.

The prose distinguishes problem states from search records, expansion order from returned paths, and current prefix costs from theoretical optimal values. Worked traces use the same five-state graph for DFS, BFS, Dijkstra and A*, including the obsolete cost-11 goal entry and its cost-5 replacement. Geometry, terrain costs, puzzle encoding, complexity and weighted priority are connected to numerical examples. Advanced heap and search variants remain in optional folds. The three short trace/weight tables use `.astar-worked-table` to wrap at mobile widths instead of inheriting the site's wide-table minimum; the notebook generator removes that blog-only Kramdown attribute, and the source checker rejects leaked attributes.

## Sorting tutorial

The paired `Sorting-From-Scratch` articles use the existing Data Structure & Algorithm topic and the `sorting` bundle. Eight bilingual activities cover sorting traces, random shuffles, inversions, stable merging, three-way partitioning, counting/radix passes, a five-question quiz, and a bounded coding exercise. Their settings, progress and answers participate in local-editor checkpoints. See [Sorting tutorial assets and checks](sorting-article.md) for source ownership, operation-count conventions, downloadable Python implementations and notebooks, and reproduction commands.

## Galois theory tutorial

The paired `Galois-Theory-From-Scratch` articles use the shared Algebra category and topic, and the `galois` bundle. The `galois-theory` tag identifies their specific subject. Six bilingual activities cover arithmetic in Q(√2), root permutations, fixed fields, derived series, numerical quintic roots, and a five-question quiz. Settings and answers participate in local-editor checkpoints. The ten figures have separate desktop and mobile compositions. See [Galois tutorial assets and verification](galois-article.md) for exact calculation conventions, source ownership, notebook requirements, and reproduction commands.
