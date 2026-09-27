# CASMI guide figures and public example data

Prepared for the English and Korean posts on 2026-09-27. Both posts share the same twelve English-language figures and cover. Captions, alternative text, tables, and body prose are localized. Body image URLs include SHA256 prefixes.

## Public experimental spectra

The three Caffeine records are experimental HCD measurements on LTQ Orbitrap XL, with nominal collision-energy settings of 30%, 60%, and 75%. Percentages are not converted to eV. These are public MassBank examples, not CASMI train/test rows.

Authors: **Stravs M, Schymanski E, Singer H, Department of Environmental Chemistry, Eawag**. Copyright (C) 2012 Eawag, Duebendorf, Switzerland. Each original record states **CC BY**, without specifying a license version; no particular version is inferred here.

- [EA030309, 30% nominal](https://github.com/MassBank/MassBank-data/blob/befc8a1e2f2aef899747797c081a5d80fab12fe7/Eawag/MSBNK-Eawag-EA030309.txt)
- [EA030311, 60% nominal](https://github.com/MassBank/MassBank-data/blob/befc8a1e2f2aef899747797c081a5d80fab12fe7/Eawag/MSBNK-Eawag-EA030311.txt)
- [EA030312, 75% nominal](https://github.com/MassBank/MassBank-data/blob/befc8a1e2f2aef899747797c081a5d80fab12fe7/Eawag/MSBNK-Eawag-EA030312.txt)
- [MassBank LICENSE field documentation](https://github.com/MassBank/MassBank-web/blob/dev/Documentation/MassBankRecordFormat.md#215-license)

The `data/` directory contains the original three text records and `caffeine-massbank.json`, with all 22 listed peaks, tentative formula annotations, exact source URLs, original intensities, source-file SHA256 values, and metadata. Derived intensity = original intensity / maximum original intensity within that record. Original rounded 0–999 relative intensities are retained separately. The JSON's `collision_energy_ev` is null; the source percent setting has its own field.

EA030312's selected precursor metadata are 195.0877, while its observed residual precursor peak is 195.0878. The measured peak differences shown in Figure 7 use 195.0878. Formula annotations are tentative, and matching neutral masses do not prove sequential reaction pathways.

## Figure evidence

`tools/casmi_2026_guide_figures.py` regenerates twelve PNG/SVG pairs without overwriting the cover.

| Asset | Evidence and interpretation |
|---|---|
| molecule-to-ranking | Real three-record Caffeine spectra and metadata; illustrative local query ID and output formatting. |
| reciprocal-rank | Exact metric arithmetic. |
| candidate-diagnosis | Constructed candidate lists containing real Eugenol/Isoeugenol structures. |
| isomers | Molecular graphs and masses calculated with RDKit 2026.03.3. |
| spectrum-anatomy | All 11 measured EA030312 peaks. |
| fragmentation-pathways | Analytic fixed-rate first-order kinetics; not fitted to an experiment. |
| ion-mass-balance | Measured EA030312 peak differences and calculated neutral formula masses. |
| collision-energy-series | All peaks in the three measured Caffeine records, independently normalized. |
| training-sources | Previously audited competition source-level counts only; no raw competition spectra. |
| evidence-routes | Measured cosine comparison plus clearly labeled hypothetical fingerprint/forward predictions. |
| validation-design | Proposed Class 2-like exclusion using the three known records. |
| runtime-breakdown | Existing project v1/v2 logs on visible 1,213-spectrum/400-molecule input, not hidden-runtime estimates. |

Runtime intervals combine operations: the MetFrag-to-our-ranker interval includes 12 GBM fits, and the next includes 16 GBM fits. Totals are rounded to 0.1 minutes. Initial setup in the chart is the residual of those rounded totals and displayed intervals. Reported totals end at submission/CSV construction; subsequent manifest hashing and platform finalization are not included. Original project evidence: `artifacts/20260927-public-baseline/kernel_v1_log.json` and `artifacts/20260927-public-baseline-cpu/kernel_v2_log.json` in the CASMI workspace. They are not part of the public example dataset.

To regenerate:

```sh
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/casmi26/bin/python tools/casmi_2026_guide_figures.py
```

## Cover provenance

`hero.png` is an editorially adjusted version of the user-supplied official Kaggle CASMI header. Its molecular composition was preserved while saturation and background highlights were restrained. It is not an image of the Eugenol example. No independent license or ownership of the original artwork is claimed. [Competition source](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).
