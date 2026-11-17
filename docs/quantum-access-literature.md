# Quantum access literature map

Six user-selected seed papers anchor a frozen literature collection dated **2026-11-13**. The expanded dataset contains 36 papers and covers all 15 seed-paper pairs for bibliographic coupling. References, subsequent citing papers, shared foundations, and selected device-platform papers provide context. This is a curated collection, not a field-wide publication-volume survey.

The forward search used seven dated OpenAlex queries, each capped at 25 returned records. The saved responses contain 88 unique records, and three queries have more matches than were retrieved. The 36 displayed papers are a selected subset plus backward-reference foundations; they are not all citing papers in the index.

## Files

- `_data/quantum_access_literature.json`: bibliography, bilingual paper descriptions and detailed analysis, source links, editorial branches, retrieval receipts, known reference DOIs, citation evidence, coupling pairs, and qualitative historical stages.
- `_includes/interactive/quantum-access-literature.html`: native Jekyll include used by both articles.
- `assets/js/quantum-access-literature.js` and `assets/css/quantum-access-literature.css`: interaction and responsive presentation.
- `assets/data/quantum-access-literature.json`: downloadable dataset generated from the native data source during the build.
- Paired posts with the slug `Quantum-Access-Resistance-Literature-Map` and `Quantum-Access-Resistance-Literature-Map-KR`, translation key `quantum-access-resistance-literature-map`, and `published: false`. Both articles are retained locally and excluded from production output.
- `tools/check_quantum_access_literature.py`: focused source and optional rendered-output checks.

## Data choices

Citation arrows point **from the citing paper to the cited paper**. DOI-backed edges have a matching DOI in the publisher-deposited Crossref reference list. Kim2025 → Trivedi2005 instead uses the publisher's full-text reference 30: its title, authors, journal volume, and pages match, but the deposited reference omits its DOI and prints a different year. Its separate provenance is retained. The checker reports actual edge counts and OpenAlex confirmations; an unconfirmed secondary-index flag does not invalidate a publisher reference.

Bibliographic coupling is calculated independently for the six seeds from the exact intersection of their normalized `reference_dois`. If the two observed DOI sets are \(A\) and \(B\), the score is \(|A\cap B|/\sqrt{|A||B|}\). The fifteen original pair results remain fixed: twelve positive and three zero. DOI-less or unmatched references are omitted. Zero therefore means no overlap in this observed subset, not proof that the papers share no references. The shared-foundations view shows **A → R ← B** and links each shared reference to its paper details.

Every paper has English/Korean `rich` fields for its question, method, findings, constriction implications, limits, and reading scope, plus source URLs. Constriction implications are interpretations rather than new measurements. The four `branches` values—`foundations`, `mode-coupling`, `scattering`, and `device-design`—are editorial navigation labels. Fabrication-platform papers supply geometric and process context; they do not independently measure quantum access resistance.

OpenAlex citation counts and reference counts retain their retrieval timestamps. Journal issue years are used for the visible timeline, including Venugopal2004, Trivedi2005, and the critical-backscattering paper whose issue year is 2022. IDs are stable keys and need not reproduce the display year or author. When Crossref lacks a publication date, the indexed year is normally the fallback. QDAME's IEDM paper instead uses its documented 2002 conference year: `year_evidence` identifies Kim2025's publisher reference 19, while the OpenAlex year remains 2003. The optional raw check verifies that exact DOI/reference receipt. Historical stages are a qualitative synthesis. Pourghaderi2026's technical interpretation remains limited to its public abstract and nominal-overlap condition; its citation edges come from publisher-deposited references.

## Checks

Run the scoped checker after editing the dataset, include, or paired articles:

```sh
python3 tools/check_quantum_access_literature.py
```

Optional local evidence checks accept the expanded `expanded-raw/` directory and the original eleven-paper `study.json`. They make no network requests. The original papers, factual fields, citation receipts, and seed-pair results must survive as a subset; added papers and richer descriptions are permitted. Files such as `*-citers.json` are query receipts and are not treated as paper records:

```sh
python3 tools/check_quantum_access_literature.py \
  --evidence-dir /absolute/path/to/expanded-raw \
  --snapshot /absolute/path/to/quantum-access-study/study.json
```

The source check validates the six exact seed DOI/title identities, all 36 bilingual analyses and source links, citation provenance, shared-reference coverage, all seed-pair intersections and cosine scores, and matching article metadata and publication state. The optional raw-evidence check also checks exact publisher reference keys, issue-year/fallback choices, and OpenAlex confirmation flags against the saved API responses.

Build the production site before checking rendered output:

```sh
PATH=/opt/homebrew/opt/ruby/bin:$PATH JEKYLL_ENV=production bundle exec jekyll build
python3 tools/check_quantum_access_literature.py --site-dir _site
```

For unpublished articles, the production check verifies that both routes are absent from the generated site and its public indexes. For published articles, it verifies both routes, embedded/exported JSON parity, three view controls, all 36 paper options, search/branch/network-scope controls, translation links, and versioned local asset references. Metadata checks, HTML-Proofer, and actual desktop/mobile interaction checks are run separately using the repository's existing workflow. The checker does not claim browser or hosted validation.

To inspect an unpublished preview, build with Jekyll's `--unpublished` option and pass `--include-unpublished` alongside the checker's `--site-dir` option.

## Verification record

On 2026-11-13, the expanded source checker passed against the original eleven-paper snapshot, all 36 paper retrievals, and seven forward-query receipts. It verified 102 directed citations (101 publisher DOI matches and one documented bibliography match), 102 OpenAlex confirmation flags, all 36 bilingual detailed analyses, and the unchanged 15 seed-pair results. The twelve distinct shared-reference DOIs all have paper details in the expanded dataset.

The production preview build with `--unpublished`, global metadata check, scoped rendered-output check, and embedded/exported JSON parity passed. HTML-Proofer passed on 446 HTML files and 4,393 internal links. The existing duplicate `site.webmanifest` destination warning remains outside this change.

Browser checks covered EN and KO at 1280, 390, and 320 pixels. Direct-neighbor and full-network views, editorial branch filters, DOI search, zoom/reset, four-reference and zero-reference comparisons, timeline navigation, and dark styling passed. The full overview contains 36 papers and 102 citation links. All 72 language/paper combinations rendered six populated detail sections. The timeline and shared-foundation views hide the common paper selector; their paper buttons move focus and scroll to the detailed explanation. Touch panning, URL restoration, and Back/Forward were also exercised. The checked states had no page overflow, graph-label overlaps, clipped graph nodes, JavaScript exceptions, or MathJax errors; each article rendered two display equations.

The final in-app preview was opened on a fresh localhost origin and confirmed the 36-paper collection and the actual four references shared by Venugopal2004 and Laux2004. Local receipts and screenshots are under `_draft/_checks/quantum-access-literature-20261113/`. That review covered the explorer, dataset, paired drafts, validator/documentation, the asset fingerprint hook, and its conditional layout loading; it did not include deployment.

The publication update enabled both articles with their original 13 November 2026 dates. On 17 November 2026, both were returned to `published: false` at the author's request. Their content, dates and data remain in the repository. The explorer is excluded from local-editor source mapping, automatic equation conversion and heading collection, while the surrounding article remains editable. The Pages workflow runs the source checker before the build and the rendered-output checker afterwards.

The static bibliography uses a derived list ordered by year, author label and stable paper ID. This gives local and hosted builds the same order within each year without changing the original evidence dataset. Author labels already include the year, so each entry displays it once.

## Hover preview

The original graph layout and controls are retained. Hovering a graph node, or focusing it with the keyboard, shows a small floating preview with its full title, author/year, concise summary and comparison/evidence scope. The preview does not change the selected paper, URL or graph positions. Clicking a node still selects it and updates the detailed analysis below the graph. Escape dismisses the preview. The preview is bounded by the viewport, stays away from the pointer, and lets all pointer events pass through to the graph. It does not reserve page space.


## Network defaults and manual arrangement

The default citation scope is the full network. An explicit `scope=neighbors` share link still opens the selected paper and its direct neighbors. The HTML selector, URL parser and share-link fallback use the same overview default.

Dragging a paper circle or diamond changes its position and updates its citation paths without selecting it. A short click still selects the paper. Overview arrangements survive paper selection, filtering and resizing within the current page; neighbor arrangements are kept separately for each selected paper. Pan mode moves the complete graph. Fit changes the view while retaining the arrangement; Reset restores the automatic layout for the current scope. Manual coordinates stay in page memory and are not written to the dataset or share URL.
