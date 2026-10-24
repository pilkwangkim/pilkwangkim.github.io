# Sorting tutorial assets and checks

The English and Korean `Sorting-From-Scratch` posts belong to the existing Data
Structure & Algorithm topic. They use the `sorting` article-lab bundle, eight
desktop/mobile figure pairs, and self-contained executed Python notebooks.

Source ownership:

- Article prose and Python implementations: `_posts/2026-4Q/2026-10-24-Sorting-From-Scratch{,-KR}.md`.
- Bilingual local activities: `assets/js/article-labs-sorting.js`,
  `assets/css/sorting-lab.css`, `_data/article_labs_sorting.yml`.
- Reproducible vector/PNG figures: `tools/sorting_figures.py`.
- Full Python coding scaffold/reference: `assets/code/sorting-code-test.py`.
- Notebook conversion and execution: `tools/build_sorting_notebooks.py`.

The playground counts key comparisons and array-slot writes, including auxiliary
arrays, rather than elapsed time. It uses a deterministic first-key quicksort
pivot to make bad inputs visible. The article's randomized Python implementation
has a different pivot policy; its expected bound is over random pivot choices.
Animation traces allocate snapshots and are not implementations with the
workspace bounds of the underlying sorting algorithms.

The five questions share inputs with Figure 8. The notebook generator reads the
browser's question definitions and verifies that the eight Python coding
fixtures match the browser fixtures. Browser coding controls select bounded
algorithm decisions; they do not execute arbitrary Python. The downloadable
notebook and `.py` file support editing the full Python implementation.

All coding versions mutate and return the same list object. The executable
starter has three editable decisions, working counters and finite safety stops.
Its initial settings genuinely pass 3/8 fixtures; the reference must pass all
eight. Browser result cards separate remaining unknown records, unsorted output,
pointer-invariant violations and work-budget stops. Quiz feedback explains each
selected option, not just the correct option; notebook conversion carries all
option rationales and case purposes.
The notebook's later compact article solution is named `article_sort_colors`
so Run All does not overwrite the editable `sort_colors` starter. Generation
retries the starter tests after all cells to verify that editing workflow.

Visible section titles may change, but empty inline span anchors preserve the
earlier fragment URLs, including numeric-leading IDs that Kramdown's explicit-ID
syntax does not accept. The notebook generator removes these blog-only anchors.
The figure generator writes reproducible outlined SVG/PNG pairs with a shared
publication-style palette, bounds/collision checks and a dimension manifest.
Figure 8 uses exactly the five quiz inputs, including [1,2,0] and its inclusive
low/mid/high pointers; it does not reveal the update or result.

The worked traces connect array changes to pointer/index updates before the
correctness and cost arguments. Optional advanced material includes an executable
four-wire bitonic network and halving-gap Shell sort. The network has exactly
four inputs; its six comparators and three parallel stages measure different
resources. Shell's stated bounds apply to the shown gap policy, not every Shell
variant. `tools/test_sorting_python.py` tests these examples as well as the core
algorithms, including the deliberately unstable Shell record example.

Reproduce the artifacts and checks:

```sh
/opt/anaconda3/bin/python tools/sorting_figures.py
node tools/test_sorting.js
python3 tools/test_sorting_python.py
python3 tools/build_sorting_notebooks.py
python3 tools/check_sorting_article.py
node tools/check_article_labs_models.js
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/test_article_lab_assets.rb
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby tools/check_blog_metadata.rb
JEKYLL_ENV=production PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec jekyll build
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec htmlproofer _site --disable-external
```

The figure generator needs Matplotlib and Pillow. Article/notebook sorting code
uses only Python's standard library. Node/source tests check numerical and
integration contracts; desktop/mobile browser review separately checks the
rendered equations, figures, tables, controls, and interactions. No publishing is
performed by these checks.
