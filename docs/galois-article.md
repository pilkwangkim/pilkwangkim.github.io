# Galois tutorial assets and verification

The paired articles are
`_posts/2026-4Q/2026-10-25-Galois-Theory-From-Scratch{,-KR}.md`.
They belong to Mathematics and the Galois Theory topic. English was authored
first; the Korean version preserves all displayed equations and Python fences.

The main concrete obstruction is `x^5-5x+1`: translating by `x-1` exposes
Eisenstein at 5, its derivative proves three real roots, and complex conjugation
exchanges the remaining pair. Transitivity in prime degree and the conjugate-edge
graph argument establish S5. The independent proof A5'=A5 uses an explicit
commutator and generation by 3-cycles, without assuming simplicity of A5.

Source ownership:

- Articles: prose, 62 displayed equations, six Python fences, seven coding fixtures.
- `assets/js/article-labs-galois.js` and `_data/article_labs_galois.yml`:
  six bilingual browser activities and the five-question illustrated quiz.
- `assets/css/galois-lab.css`: scoped lab styles, responsive figures, and equation containment.
- `assets/code/galois-reference.py`: standalone standard-library permutation calculations.
- `tools/galois_figures.py`: reproducible outlined SVG/PNG desktop/mobile figures and cover.
- `tools/galois_visual_{style,early,late,core}.py`: shared design, coordinate/root
  geometry, composed label actions, subgroup-order plots, and the A5 cover graph.
- `tools/build_galois_notebooks.py`: full executed notebooks and
  `assets/code/galois-article-examples.py` generated directly from article fences.

The browser performs exact bounded integer/permutation calculations, except for
the explicitly numerical double-precision bisection activity. The article's
bisection code instead uses Fraction endpoints; both start with width-two
brackets. Numerical residuals are not used to decide radical solvability.

Permutations are zero-based image arrays. Composition applies the right argument
first: `compose(p,q)[i]=p[q[i]]`. The Python/SymPy comparison checks invariant
group properties, not library-specific permutation multiplication conventions.

Each notebook embeds all ten figures, expands the actual five browser quiz
questions with every option's reasoning, and executes six Python cells. The
optional symbolic cell needs SymPy (verified with 1.14.0). The editable coding
starter is defined but not called by Run All; a separately named reference
solution passes seven fixtures and never overwrites the starter.

Reproduce the artifacts and checks:

```sh
/opt/anaconda3/bin/python tools/galois_figures.py
/opt/anaconda3/bin/python tools/test_galois_figures.py
node tools/test_galois.js
/opt/anaconda3/bin/python tools/build_galois_notebooks.py
/opt/anaconda3/bin/python tools/check_galois_article.py
/opt/anaconda3/bin/python tools/test_galois_math.py --sympy required --require-assets
node tools/check_article_labs_models.js
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/test_article_lab_assets.rb
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/check_blog_metadata.rb
JEKYLL_ENV=production PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec jekyll build --future
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec htmlproofer _site --disable-external
```

Figure generation needs Matplotlib, NumPy, Pillow, SymPy, and NetworkX. The core article examples
and standalone group reference use the standard library; SymPy is additional
only for the clearly labelled symbolic check and its notebook execution.

The publication date is 2026-09-25 09:00 +0900 in both languages. The existing
source filenames and asset directory retain their 2026-10-25 prefix so that
article URLs, downloads, and figure references remain unchanged. Local QA builds
do not commit, push, or publish the articles.

Source and model tests are distinct from browser review. Check both languages
at desktop and mobile widths, all six controls and reset, MathJax errors and
equation overflow, intrinsic image dimensions, reciprocal links, topic filtering,
and optional proofs/quiz input folds.

## Learning layers

The prerequisite is a basic second-year undergraduate linear-algebra course:
systems of equations, matrices, vector spaces, bases, dimension, and eigenvalues.
No prior group theory, field theory, quotient construction, or Galois theory is
assumed. An unfamiliar term must have a local explanation and a small example
before it is used in an argument. General existence proofs and longer algebraic
calculations belong in optional sibling details blocks, not in the visible
concept-to-example sequence.

The visible explanation motivates each new concept with a short calculation.
The recurring examples are Q(sqrt2), its sign automorphism, and the four sign
maps of Q(sqrt2,sqrt3). They connect vector-space coordinates to subgroups,
fixed fields, cosets, and restriction homomorphisms. General proofs, longer
worked calculations, and full implementations live in sibling details blocks.
Main section headings and all six browser activities remain outside those
blocks; `check_galois_article.py` checks this and paired fold placement.

The learning-layer revision preserves all 62 numbered displayed equations,
six Python snippets, ten responsive figure pairs, lab identities, and post
metadata. Added inline examples have independent exact checks in
`test_galois_math.py`: quadratic inverses, S3 products and cosets, the V4
restriction example, Cardano's chosen cubes, the degree-four product basis,
and the concrete quartic resolvent. These checks verify the examples, not
whether a particular reader will understand the text.

Inline sets use the TeX commands `lbrace` and `rbrace` with leading backslashes.
Markdown consumes backslash-escaped literal braces before MathJax runs, so
ordinary escaped braces in inline math can disappear even with no MathJax
error. The source checker rejects that spelling, and browser review checks
that the actual set delimiters are visible. Numbered display equations retain
their existing source unchanged.

## Scientific figure design

The cover is an adjacency-preserving geometric embedding of the actual
60-vertex Cayley graph of A5 generated by (12345) and (23)(45). Vertices denote
group elements, not roots. The graph's shape is not a proof of nonsolvability.
The body uses plotted functions, coefficient transformations, root positions,
a paired Hasse diagram, cyclic actions, and full four-stage commutator tracks.
The derived-series chart plots discrete orders on an explicitly logarithmic
axis; the identity has order 1.

Label bounds are measured from the rendered Matplotlib Text objects, including
axis ticks and legends. All labels must remain inside the canvas and meet a
16px minimum at the documented body widths. Separate mobile compositions
retain readable labels. These geometry checks supplement, not replace, visual
inspection at actual desktop and mobile article widths.

The mobile composition is used through a 680px viewport and capped at its
480px source width. The former 640px switch made a 641px viewport render
desktop labels at about 15.4px. The revised boundary keeps those labels at
least 16px when the desktop layout first appears.

Caption emphasis uses asterisks rather than underscores. An outer underscore
can close at a TeX subscript such as M-alpha, splitting the math delimiter
across HTML emphasis nodes before MathJax sees it. Browser review checks for
raw dollar delimiters in every rendered caption as well as MathJax errors.
