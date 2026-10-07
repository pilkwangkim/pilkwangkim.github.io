# Restored essay Originals: source and copy-edit record

English source selection follows verified Git history. The recovered Benevolence, Narrative and Mirage prose predates the expanded versions introduced by d31f77db465b69ce657db99f1504405fcbf9b9cd. Their source is its parent, 92fc75a2502183260a97ad6469379b76f2056e35. Current titles, publication dates, category and tag metadata, topic, language and translation keys were retained so that Original and Compact refer to the same logical essay.

The English edits are local grammar and clarity changes. Paragraph order, examples, claims and qualifiers are retained, and no sections, tables or equations were added. Original spelling conventions are retained. In Benevolence the inverted sentence about ethics was clarified to “True ethics, therefore, does not end with good will; it begins there”, consistent with its following argument and unchanged Korean version. Mirage’s unidiomatic “while appearing decisiveness” and “white knight demotion” were corrected. The author’s historical claims, including the autonomous-driving and MacGuffin discussion, were not expanded or replaced.

Corporate Hallucinator has no earlier unsectioned prose version in the available Git history. Its first recorded body at 5c20f8292488a940d1e8bd807a6fab2a9137aea9 matches the current body byte for byte before the local English copy edit. Its existing title, subtitle, eight section headings and quotations remain. Original identifies the earliest available source here; it does not imply a recovered, previously unsectioned version.

The three Korean Original bodies are exact byte-for-byte copies of their current Korean counterparts, also verified equal to their first added versions at ed12d9407721ce6c0206bc3b18dc804ffee286f1. No Korean text was edited and no new translations were created. Existing current posts were not modified by this editorial task.

| New file | Source commit | Words before / after | Local edits |
| --- | --- | --- | --- |
| `_posts/2026-1Q/2026-01-30-The-Paradox-of-Benevolence-Original.md` | `92fc75a2502183260a97ad6469379b76f2056e35` | 364 / 354 | 6 |
| `_posts/2026-2Q/2026-04-12-The-Narrative-Trap-Original.md` | `92fc75a2502183260a97ad6469379b76f2056e35` | 512 / 496 | 7 |
| `_posts/2026-2Q/2026-04-19-The-Mirage-of-Merit-Original.md` | `92fc75a2502183260a97ad6469379b76f2056e35` | 477 / 471 | 9 |
| `_posts/2026-2Q/2026-05-02-The-Corporate-Hallucinator-Original.md` | `5c20f8292488a940d1e8bd807a6fab2a9137aea9` | 725 / 710 | 13 |
| `_posts/2026-1Q/2026-01-30-The-Paradox-of-Benevolence-Original-KR.md` | `ed12d9407721ce6c0206bc3b18dc804ffee286f1` | 297 / 297 | 0 |
| `_posts/2026-2Q/2026-04-12-The-Narrative-Trap-Original-KR.md` | `ed12d9407721ce6c0206bc3b18dc804ffee286f1` | 490 / 490 | 0 |
| `_posts/2026-2Q/2026-04-19-The-Mirage-of-Merit-Original-KR.md` | `ed12d9407721ce6c0206bc3b18dc804ffee286f1` | 417 / 417 | 0 |

All seven new frontmatter blocks passed YAML parsing and duplicate-key checks. Each has exactly one `article_version: original`, a full 40-character source commit, the `essays` topic, its expected language and the existing logical translation key. English body block/headings/table/display-math counts remain equal to their selected sources; Korean body byte equality was checked after writing.

## Exact source receipts

Body SHA-256 receipts below use the same extraction as the preservation check: Ruby `raw.split(/^---\s*$\n?/, 3)[2]`, where `raw` is the source Git file or new file bytes. The delimiter expression consumes frontmatter delimiters and adjacent whitespace; no further body normalization is applied. Structure and word metrics are unchanged.

### _posts/2026-1Q/2026-01-30-The-Paradox-of-Benevolence-Original.md

- Source: `92fc75a2502183260a97ad6469379b76f2056e35:_posts/2026-1Q/2026-01-30-The Paradox of Benevolence.md`
- Source body SHA-256: `43003bc419caeab3e8938a696bfc2f1f2dd897d37bc6ad5a882522ef4ba9abc1`
- New body SHA-256: `e040b0be8a6c6666fe3773bcf137cadb7b03a222195560b78d79d20839316101`
- Body structure before / after: `{"words": 364, "blocks": 4, "headings": 0, "tables": 0, "display_math": 0}` / `{"words": 354, "blocks": 4, "headings": 0, "tables": 0, "display_math": 0}`

### _posts/2026-2Q/2026-04-12-The-Narrative-Trap-Original.md

- Source: `92fc75a2502183260a97ad6469379b76f2056e35:_posts/2026-2Q/2026-04-12-The Narrative Trap.md`
- Source body SHA-256: `77036997218caa7cb3aa3d590516f241c9969eac0df691ccc0aa77e64039f489`
- New body SHA-256: `5bf14b35f15c46cf1221d32024f9909a32bd8da8cb7eaa548fe9a5862dc412c2`
- Body structure before / after: `{"words": 512, "blocks": 7, "headings": 0, "tables": 0, "display_math": 0}` / `{"words": 496, "blocks": 7, "headings": 0, "tables": 0, "display_math": 0}`

### _posts/2026-2Q/2026-04-19-The-Mirage-of-Merit-Original.md

- Source: `92fc75a2502183260a97ad6469379b76f2056e35:_posts/2026-1Q/2026-02-12-The Mirage of Merit and the Geopolitics of Positioning.md`
- Source body SHA-256: `a1e9ef98a2e201ceccb329d4ac3c8a42add936c24379a69b77d02074fea15ae1`
- New body SHA-256: `fad523d81c0194b4c51d657db119a6cbbe2412daaeccdf82596c657f15c141eb`
- Body structure before / after: `{"words": 477, "blocks": 4, "headings": 0, "tables": 0, "display_math": 0}` / `{"words": 471, "blocks": 4, "headings": 0, "tables": 0, "display_math": 0}`

### _posts/2026-2Q/2026-05-02-The-Corporate-Hallucinator-Original.md

- Source: `5c20f8292488a940d1e8bd807a6fab2a9137aea9:_posts/2026-2Q/2026-05-03-The-Corporate-Hallucinator.md`
- Source body SHA-256: `0a59521dfdf70d07b47504a226b141b3105e1ad0de653d9b7b2206b151bf8a43`
- New body SHA-256: `dd820c8b7093e525cc0d257c8b8b5d9a23fd611b5befa6b032be2430b6f48fa6`
- Body structure before / after: `{"words": 725, "blocks": 31, "headings": 9, "tables": 0, "display_math": 0}` / `{"words": 710, "blocks": 31, "headings": 9, "tables": 0, "display_math": 0}`

### _posts/2026-1Q/2026-01-30-The-Paradox-of-Benevolence-Original-KR.md

- Source: `ed12d9407721ce6c0206bc3b18dc804ffee286f1:_posts/2026-1Q/2026-01-30-The-Paradox-of-Benevolence-KR.md`
- Source body SHA-256: `bd536f5ca2ea4b3feec1f6c6957bf6e6396593cba76fd03d926a03acb9ad6806`
- New body SHA-256: `bd536f5ca2ea4b3feec1f6c6957bf6e6396593cba76fd03d926a03acb9ad6806`
- Body structure before / after: `{"words": 297, "blocks": 14, "headings": 4, "tables": 0, "display_math": 0}` / `{"words": 297, "blocks": 14, "headings": 4, "tables": 0, "display_math": 0}`

### _posts/2026-2Q/2026-04-12-The-Narrative-Trap-Original-KR.md

- Source: `ed12d9407721ce6c0206bc3b18dc804ffee286f1:_posts/2026-2Q/2026-04-12-The-Narrative-Trap-KR.md`
- Source body SHA-256: `224a4c993bec0da0768851b9c68b1fc00ee642ee3b324b5e6cad5fb8d4de6bcf`
- New body SHA-256: `224a4c993bec0da0768851b9c68b1fc00ee642ee3b324b5e6cad5fb8d4de6bcf`
- Body structure before / after: `{"words": 490, "blocks": 21, "headings": 5, "tables": 0, "display_math": 0}` / `{"words": 490, "blocks": 21, "headings": 5, "tables": 0, "display_math": 0}`

### _posts/2026-2Q/2026-04-19-The-Mirage-of-Merit-Original-KR.md

- Source: `ed12d9407721ce6c0206bc3b18dc804ffee286f1:_posts/2026-2Q/2026-04-19-The-Mirage-of-Merit-KR.md`
- Source body SHA-256: `4106bb48cf698c8413dc9880f5349e96dabbc004d6bbd61c0a4e4e37b0958ae1`
- New body SHA-256: `4106bb48cf698c8413dc9880f5349e96dabbc004d6bbd61c0a4e4e37b0958ae1`
- Body structure before / after: `{"words": 417, "blocks": 17, "headings": 4, "tables": 0, "display_math": 0}` / `{"words": 417, "blocks": 17, "headings": 4, "tables": 0, "display_math": 0}`
