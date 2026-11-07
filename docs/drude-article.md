# Drude tutorial package

Paired posts: `2026-11-07-Drude-Model-From-Scratch{,-KR}.md`.
Stable routes: `/posts/Drude-Model-From-Scratch{,-KR}/`.
Date: `2026-11-07 18:00:00 +0900`. Topic: `quantum-transport`.
Audience: undergraduate quantum mechanics and elementary calculus/linear algebra;
no assumed semiconductor transport or statistical mechanics course.

## Reading structure

1. Charge, electric potential and field, electron drift and conventional current.
2. Poisson plus boundary conditions; continuity as local particle conservation.
3. Collisionless acceleration first; scattering-event averaging then supplies
   the homogeneous mean-velocity equation, DC mobility and channel resistance.
4. Field-step response, drift energy versus electronic heating, AC phase response.
5. Bloch states, tight binding, group velocity and inverse-curvature mass.
6. Nonparabolic and anisotropic examples; distribution-based conductivity.
7. Finite-time amplitude, continuum golden rule and normalization conventions.
8. Fourier matrix elements, Born cross sections and angular transport relaxation.
9. Short-range, screened Coulomb and acoustic-phonon examples; thermal averages.
10. A periodic self-consistent charge mode; conceptual and coding exercises.

English is drafted first. Korean retains the equations, numerical conditions,
code, links and technical English terms while using idiomatic Korean sentences.
Longer derivations and answers are folded, without hiding essential assumptions.

## Physical contract

- Positive e; electron charge -e. n counts the mobile conduction population;
  stationary bound charges are represented in the fixed net positive background.
  That background gives rho=e(ND-n).
  E=-grad(phi), -div(epsilon grad(phi))=rho, J=-en u. No particle sources.
- Real positive constant background permittivity; conduction electrons are not
  counted twice inside it. Neutral bulk permits a boundary-imposed constant E.
- Uniform low-field scalar Drude closure with positive constant response mass
  and momentum-relaxation time. It is not a general nonuniform fluid model.
  The channel I-V excludes contact resistance and fixed-temperature nonlinearities.
- The fixed-density current perturbation uses even dispersion and an energy-only
  positive transport time. Its scalar relaxation law is derived on a single
  fixed-speed isotropic elastic shell, or adopted as an additional approximation.
- Original free-electron Drude uses m0. The positive curvature mass is a band-edge
  extension. Inverse-Hessian response, mass tensor, local negative curvature and
  distribution-averaged conductivity are distinct; no negative DC-conductivity
  claim is made by substituting a local negative mass into ne²tau/m.
- AC uses exp(-i omega t): sigma=sigma0/(1-i omega tau), positive imaginary
  part and current lag. Zero frequency is static, not a plotted oscillation.
- Momentum relaxation is not energy cooling, recombination, collision duration
  or universally the event waiting time. Elastic scattering alone does not fix
  the electron temperature in sustained driving.
- Finite-time perturbation probability concerns a distinct final state b!=a.
  The energy delta is an integrated continuum kernel, not delta(0). Weak
  coupling, a dense continuum and kinetic coarse-graining are explicit.
- Box-normalized matrix element M=U_tilde/V has energy units. The outgoing 3D
  Born amplitude f=-m U_tilde/(2 pi hbar²) has length units. The 1D Gaussian
  Fourier example instead has energy-length U_tilde and uses M=U_tilde/L.
- Uncorrelated impurity positions; spatially uniform average potential is
  absorbed into the band. Per-spin final DOS: m k/(2 pi² hbar²), no extra spin
  factor for a spin-conserving final state. Total density/current count both spins.
- Angular 1-cos(theta) weighting applies to the isotropic elastic current mode.
  More general anisotropic/inelastic kernels require their transport equation.
- Short-range Born amplitude is prescribed, not an exact 3D delta scatterer.
  k*l>>1 or dilute impurities alone do not prove weak individual-potential Born
  validity. A shell mobility is not an automatically thermally averaged mobility.
- Screened Coulomb q_s is prescribed and positive. The lab normalizes total
  event rate to isolate angular shape; physical fixed-strength rates also change.
- Acoustic deformation-potential example: one longitudinal branch, high phonon
  occupation and quasielastic isotropic intraband limit; not a material fit.
  Thermal mu~T^(-3/2) requires the stated 3D nondegenerate model and fixed inputs.
- Independent rates add at a fixed energy/common angular mode; inverse averaged
  mobilities do not generally add when energy dependences differ.
- Charge dynamics: q>0, qLcell=2pi periodic small perturbation, pressureless
  quasistatic linear electron fluid, no contacts. a''+a'+Omega²a=0. Critical
  Omega=1/2; Maxwell epsilon/sigma is the slow overdamped limit, not tau_tr.

## Owned implementation

- `assets/code/drude-model.py`: NumPy SI reference and numerical diagnostics.
- `assets/js/article-labs-drude*.js`, scoped CSS and `_data/article_labs_drude*.yml`:
  sixteen bilingual bounded labs. The original six (`step`, `ac`, `band`,
  `scattering`, `charge`, `quiz`) retain their physical models; axis ticks are
  spaced for mobile readability. Separate foundations,
  quantum and microscopic bundles add continuity, Bloch phase, occupations,
  packet evolution, finite-time transitions, Fourier matrix elements, deflection,
  short-range impurities, phonon factors and energy averaging. Models have pure
  Node exports, validated state links and Reset. No arbitrary code execution.
  Additional bundles reuse the stylesheet loaded once by the mandatory `drude`
  base bundle; they do not declare duplicate stylesheet URLs.
- `tools/drude_figures.py` and `tools/drude_section_*.py`: twenty scientific
  figure pairs plus responsive cover, SVG and PNG (84 files). Each numbered
  section has exactly one visible figure; its number matches the section.
  The original seven asset URLs and cover are preserved. Desktop/mobile layouts
  are independently arranged.
  Equations use local LaTeX/dvisvgm paths; PNG uses rsvg-convert. Scientific
  labels use a fixed TeX font size so derivative/fraction glyphs are not shrunk
  relative to single symbols. Hero cover equations retain their own scaling.
  Figure 2 is a continuum boundary-value schematic followed by aligned potential
  and electric-field profiles; no premature conductivity or I-V normalization is
  assumed. Figure markers identify switch times, waveform maxima, curvature zeros
  or a selected density-mode time, rather than individual carriers.
  Both source variants declare their intrinsic SVG dimensions so lazy images
  reserve the correct desktop/mobile height before loading; the source checker
  verifies those dimensions against the assets.
  The cover pairs the DC-conductivity identity with one analytic current
  turn-on curve, normalized by sigma0 E0 and the transport time. Its gold point
  is 1-exp(-1) at t=tau; the dashed asymptote is the steady current, not another
  dataset. Desktop uses two columns; mobile stacks the equation and the plot.
- `tools/build_drude_notebooks.py`: actual article snippets, executed in order,
  twenty embedded figures, browser-lab anchors, preserved coding starter and a
  separate reference answer. Notebook execution needs Python3.9+ and NumPy.
- `tools/check_drude_article.py`: paired math/code/IDs/assets/reference parity,
  sequential numbering, source hygiene, Python3.9 syntax and notebook integrity.
- Focused Python and JS suites include independent quadrature, finite-difference
  and RK4 oracles plus article-example and exercise-checker comparisons.

## Reproduction and final gates

Use an interpreter with NumPy; the task reused a local virtual environment.

```sh
python3 tools/drude_figures.py --png
python3 tools/test_drude_physics.py
node tools/test_article_labs_drude.js
node tools/test_article_labs_drude_foundations.js
node tools/test_article_labs_drude_quantum.js
node tools/test_article_labs_drude_microscopic.js
python3 tools/build_drude_notebooks.py
python3 tools/check_drude_article.py
node tools/check_article_labs_models.js
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/test_article_lab_assets.rb
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/test_responsive_pictures.rb
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/check_blog_metadata.rb --output /tmp/drude-metadata.json
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/ecc_render_metadata.rb /tmp/drude-metadata.json /tmp/drude-routes.json
JEKYLL_ENV=production PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec jekyll build
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/check_article_labs_output.rb --site _site --metadata /tmp/drude-routes.json
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec htmlproofer _site --disable-external
```

Actual browser QA is separate from model tests: both languages, desktop and
390/320px CSS widths, opened folds, MathJax, figure sources, table columns,
interactive endpoints, quiz feedback, state restoration and no page overflow.
Before the front-matter time, a local preview needs `jekyll build --future` and
a matching Jekyll routing report generated with `future: true`. The ordinary
production build intentionally omits future-dated posts. Use a fresh localhost
origin after rebuilding to avoid an older service-worker response masking an
image or a corrected equation.
Keep screenshots, build directories and receipts outside tracked source.
No commit, push or deployment is implied by article authoring or verification.

Focused checks: 7,547 Python assertions; 13,897 JS physics assertions and 2,354
bilingual UI/state/reset assertions, with 783,380 finite in-domain plot points.
The three additional bundles contribute 30,782 / 177,171 / 7,175 model/data
assertions and 1,361 / 2,280 / 1,427 bilingual UI/state/reset assertions,
respectively. Their independent plot checks cover 17,796 / 1,064,456 / 476,384
finite in-domain points. These are executable-model checks, not browser geometry.
The article checker independently replays 32 notebook code cells, compares their
stored output, checks 34 Markdown cells and 40 byte-exact embedded PNGs, and
preserves the exercise starter. All 65 numbered equations and 730 inline-math
occurrences match between languages. Multi-relation displays are stacked rather
than shrunk to fit a phone; intrinsically long expressions can scroll locally.
At widths up to 540px, numbered Drude displays put the number on a separate
line below the formula. This removes the label-width margins that otherwise
clip short equations on narrow screens, without reducing the glyph size.
The rendered check compares all 91 displays and 730 inline formulas with their
source and tests the 65 labels for overlap, clipping and inaccessible left
overflow; page width, table columns and responsive figure sources are checked
separately in both languages.

## Section-by-section visual and interactive support

All twenty numbered sections retain one essential static figure outside folds.
Thirteen new pairs visualize signed boundary-flow budgets, collisionless versus
relaxed drift, separate mobility/density comparisons, complex Bloch coefficients, Gaussian packet moments, occupation
imbalance, finite-time transition peaks, 3D phase fronts/cross sections, angular
projection loss, short-range DOS scaling, lattice strain and Bose factors,
energy weights, and the coding-test input. Figure numbers now match sections;
the seven pre-existing figure URLs and all four cover files remain byte-exact.

The packet uses a real, zero-chirp initial Gaussian and names its position width
Delta x0 to avoid sigma0 conductivity ambiguity. Golden-rule peak integrals state
their normalization and units; individual small probabilities do not establish a
small summed continuum probability. Wavefronts are phase surfaces, Bloch arrows
are complex coefficients, and wavevectors are not microscopic particle paths.
The exact-forward coding fixture is a null-deflection test channel, not a quantum
transition to the identical state. Original equation/code bytes remain intact.

New activities separate fixed integrated Gaussian strength from fixed peak
height, single-event count from angular current loss, Bose population from
kinematically allowed transitions, and particle averaging from transport
averaging. Each states its inputs, held assumptions and units, provides bounded
presets and Reset, and validates shared settings before any mutation.
The continuity activity uses a dynamic direction schematic and numerical budget,
not a straight-line accumulated-charge graph. Static sections 1 and 4 also use
schematics rather than plotting direct proportionalities as filler.

## Undergraduate-QM first-read refinement

The paired text now includes charge-balance arithmetic, a quantitative low-field
and path-length check, AC steady/transient matching, a four-site matrix/Bloch
example, wave-packet and state-counting bridges, and a small occupation imbalance
that carries current without changing density. The Boltzmann equation follows
from transport of occupations before its stationary linearization is used.

The continuum golden-rule derivation explicitly distinguishes the smooth energy
scale, unresolved level spacing and small total transition probability. A finite
energy-window quadrature checks the resulting linear-in-time total probability.
Oscillator ladder factors explain phonon absorption/emission; numerical energy
ratios check the acoustic approximation. The mobility average displays its
normalized transport weight and compares it with the different particle average.
The charge mode is one-dimensional spatial variation of a 3D bulk fluid, not a
strictly 1D wire plasmon; linearization and omission of pressure are separate.

The published finite-time probability snippet now uses the absolute square of a
complex coupling, with a phase-invariance regression that fails the old formula.
Tiny-wavevector band energy and near-critical density-mode evaluation use stable
equivalent expressions. The fixed-order angular quadrature explicitly rejects
shapes beyond its verified numerical range; the physical formula has no such cap.

Rendered inline formulas are also compared with their source, rather than relying
only on MathJax's error counter: valid-looking prose can be misread as mathematics
after Markdown emphasis crosses a formula. The phonon displacement hat groups its
subscript within the accent, prime symbols use explicit TeX commands rather than
smart-quote-sensitive apostrophes, and the rotated mass matrix is display math so
its row separator is not consumed by Markdown.

## Causal-flow refinement

The opening starts with the familiar fixed-voltage observation, not a contrast
with unexplained scattering. The visible derivation follows charge crossing a
section, the boundary-imposed field, collisionless Newton acceleration, a stated
complete-randomization event model, the averaged recurrence, the Drude equation,
steady-state balance and the approach to that balance. Equations (5) and (6) now
number the recurrence and its differential equation respectively; all references
were updated. The two extra unnumbered displays show collisionless acceleration
and the steady force balance. Original free electrons use m0; the constant
positive response mass is generalized only after the band derivation.

Later bridges make each calculation's purpose explicit: one packet versus all
occupied states; band velocities versus the missing relaxation time; coherent
perturbation amplitude versus coarse-grained rate; Bloch states versus scalar
envelope states; total event count versus angular current loss; fixed-energy
times versus transport-weighted thermal averages; imposed field versus a field
that evolves with the density. Essential definitions no longer depend on opening
the normalization, oscillator or pressure-limit folds. The periodic mode's
wavenumber and plasma frequency are defined before the pressure scale test.

Existing code, numerical examples, routes, metadata, figure assets and lab
implementations remain unchanged. Both notebooks are regenerated from the paired
text and their code cells are re-executed before source-parity validation.

## Quantum packet versus carrier drift

Section 7 is titled for its actual content: electron states and energy bands in
a crystal. Section 8 defines a normalized localized packet's center as its
position expectation value, then defines center velocity and acceleration as
successive time derivatives. The many-carrier drift mean is a distinct average.
A stationary fixed-wavevector Bloch eigenstate can have nonzero velocity/current
expectation but does not define a localized transport trajectory.

Equations (18)–(21) explicitly concern a narrow-wavevector, isolated-band packet
under a smooth, adiabatic external drive, between scattering events and without
Berry or magnetic corrections. The center variables and the approximate meaning
of the band velocity are stated before Hamilton's equations. The effective mass
relates external force to packet-center velocity change; the microscopic
Ehrenfest equation instead retains bare mass and both lattice and external
forces. A Fitzpatrick reference supports that distinction. Figure 4's caption
and exercise 2 use the same packet-center interpretation. No numerical code,
interactive implementation, image asset, metadata or numbered formula changes.

## Whole-article definitions and approximation audit

All twenty section titles were checked against their actual derivations and
examples. Descriptive headings now identify charge conservation, the mean-velocity
equation, DC response, switching transients, occupations and conductivity, the
golden rule, fixed-energy impurity scattering, longitudinal acoustic scattering,
thermal averaging and a periodic bulk density mode. Stable H2 IDs are retained.

Definitions distinguish a same-time carrier average in the wire frame, stochastic
event histories, a quantum packet center, a rate-weighted angular mean, and an
energy-weighted transport mean. The mobile population, positive background,
orthonormal site basis, spin counting, integration domain, conductivity matrix,
Kronecker delta, vibration modes and signed density amplitudes are explicit.

Linear response, spatial homogeneity, scalar collision closure, first-order
amplitude versus quadratic probability, coarse-graining, envelope normalization,
unit-incident-amplitude normalization, short range versus weak Born scattering,
and the equipartition/quasielastic limits are stated near their uses. Density-mode
exponents are distinguished by physical versus dimensionless time. Exercises
state the inherited memoryless-event and zero-spatial-mean-field assumptions.
No existing display formula, code cell, numerical example, image asset or lab
implementation was changed during this audit.

## Source audit

Reviewed author/institutional sources for the relevant derivations: Tong's band,
electron-dynamics, electromagnetism and scattering chapters; MIT Zwiebach's
finite-time perturbation/Born notes; Haus-Melcher charge relaxation; Kim-Woo-Min
isotropic versus anisotropic transport; Bardeen-Shockley deformation potential;
Giustino's review together with its 2019 correction; NIST constants. EPW links
resolve through its official documentation to `gitlab.com/epw/q-e`.
The toy-model numerical derivations were also checked independently rather than
treating agreement with a copied formula as sufficient validation.
