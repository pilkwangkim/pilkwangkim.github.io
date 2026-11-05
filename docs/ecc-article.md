# Error-correcting codes article

The bilingual `Error-Correcting-Codes-From-Scratch` posts introduce classical
binary block codes to readers with basic undergraduate linear algebra.

## Conventions and guarantees

- Column vectors over F2: `c = Gm`, `Hc = 0`, `s = Hr`.
- Hamming positions: `p1,p2,d1,p4,d2,d3,d4`.
- Syndrome tuple: `(s1,s2,s4)`; position index: `s1 + 2*s2 + 4*s4`.
- Message `1011` encodes to `0110011`; position-5 flip gives `0110111`.
- Ordinary Hamming correction guarantees recovery under at most one flip.
  A distance-three membership detector can detect two errors, but the ordinary
  single-error corrector miscorrects all exactly-two-flip patterns.
- Extended parity is appended at position eight. SECDED's four-way
  classification assumes at most two flips. Triple errors can miscorrect.
- The repetition curve is exact for independent BSC flips. Manual bit toggles
  select a pattern and are not stochastic draws from that model.

## Owned assets

- Paired articles: `_posts/2026-4Q/2026-11-05-Error-Correcting-Codes-From-Scratch{,-KR}.md`.
- Native SVG figure generator: `tools/ecc_figures.py`; outputs four responsive
  body figure pairs and two SVG/PNG covers under
  `assets/img/posts/2026-11-05-error-correcting-codes/`.
- Labs: `assets/js/article-labs-ecc.js`, `assets/css/article-labs-ecc.css`,
  `_data/article_labs_ecc.yml`.
- Standard-library reference: `assets/code/error-correcting-codes.py`.
- Executed notebooks: `assets/notebooks/error-correcting-codes-{en,ko}.ipynb`.

The article has compact teaching functions; the independent download returns
diagnostic result objects. The notebook executes the actual article snippets,
embeds all four desktop figures as PNG attachments, and replaces browser labs
with links. The exercise starter remains separate from its reference solution,
so Run All does not silently complete or overwrite the learner's answer.
The downloadable module and article examples support Python 3.9+ with only the
standard library. Jekyll block attributes are removed from notebook Markdown.

## Checks

```sh
python3 tools/ecc_figures.py --png
python3 tools/test_error_correcting_codes.py --articles
node tools/test_article_labs_ecc.js
python3 tools/build_ecc_notebooks.py
python3 tools/check_ecc_article.py
node tools/check_article_labs_models.js
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/test_article_lab_assets.rb
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/check_blog_metadata.rb --output _draft/_checks/ecc-metadata.json
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/ecc_render_metadata.rb _draft/_checks/ecc-metadata.json _draft/_checks/ecc-routes.json
JEKYLL_ENV=production PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec jekyll build
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/check_article_labs_output.rb --site _site --metadata _draft/_checks/ecc-routes.json
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec htmlproofer _site --disable-external
```

SVG-to-PNG rendering uses `rsvg-convert`. Article code needs no third-party
Python packages. Source/model tests are not browser evidence: inspect both
languages, responsive images, MathJax, semantic tables, folds, and all five labs
at desktop and narrow mobile widths before claiming visual validation.
