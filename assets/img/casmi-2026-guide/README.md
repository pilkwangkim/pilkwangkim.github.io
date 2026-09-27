# CASMI guide figures

Prepared for the English and Korean introductory posts on 2026-09-27.

Both posts use this single set of English-language figures and the same cover; there are no language-specific image copies. Captions, alternative text, tables, and surrounding explanations are localized in each manuscript. The 23 displayed equations, seven code examples, and source URLs are kept in agreement. Image URLs include content hashes so local previews can distinguish revised assets from cached images.

## Cover

- Final asset: `hero.png`.
- Source: the user-supplied `/Users/pilkwang/Downloads/header.png`, identified by the user as the official Kaggle CASMI header. Competition: <https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra>.
- Editing method: built-in image generation tool, with the local header as the edit target. The source file was not changed.
- Changes requested: restrained saturation, softer background highlights and edge decoration, preservation of molecular composition and geometry. The cover is an editorial illustration; it is not the eugenol structure analyzed in the article.
- No claim of ownership or independent licensing of the source artwork is made here.

Final editing prompt:

> Use case: precise-object-edit. Edit target: the provided Kaggle CASMI official header image. Asset type: professional scientific blog cover, wide 2:1 landscape. Make a restrained, subtle editorial adjustment of this exact image. Preserve the central ball-and-stick molecule EXACTLY: all atom positions, bonds, colors and geometry, and preserve the existing composition and subject scale. Preserve the recognizable warm amber/orange character of the supplied artwork. Slightly reduce orange saturation and luminous bokeh intensity, soften the busy out-of-focus edge decorations, and make the background more understated with balanced contrast so the sharp molecule is clearly legible. The final result should look like a lightly refined version of the same official artwork, not a redesign. No new objects, no extra molecules, no text, no logos, no frame, no glitter, no futuristic additions. Opaque background. Keep molecule centered slightly right as in source; natural polished scientific editorial appearance.

## Scientific figures

`tools/casmi_2026_guide_figures.py` generates the eleven PNG/SVG pairs used in the article. It does not overwrite the cover. The figures use a white background, dark gray labels, muted blue-green and brown accents, and explicit captions distinguishing computed quantities, source aggregates, and schematic examples.

- `isomers`: molecular drawings and exact masses computed with RDKit 2026.03.3.
- `spectrum-anatomy`, `collision-energy-series`: synthetic peaks for explanation.
- `fragmentation-pathways`: arbitrary schematic energy curves.
- `ion-mass-balance`: calculated mass bookkeeping; not an observed reaction assignment.
- `molecule-to-ranking`, `evidence-routes`, `validation-design`: conceptual diagrams of the article's pipeline and evaluation design.
- `reciprocal-rank`, `candidate-diagnosis`: exact metric arithmetic on illustrative ranks, not measured model results.
- `training-sources`: previously audited local library-level aggregate counts. No raw competition spectra are included.

To regenerate the body figures in the author's environment:

```sh
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/casmi26/bin/python tools/casmi_2026_guide_figures.py
```
