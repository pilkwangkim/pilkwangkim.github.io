---
title: "RSNA Knee, Part 4: Conflicting Evidence and Two Final Candidates"
date: 2026-10-22 09:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, rsna-knee, medical-imaging, ensemble, cross-validation, inference]
math: true
pin: false
published: true
image:
  path: /assets/img/posts/rsna-knee-series/cover-4.webp
  alt: "Methodology cover: selecting two separate packages from a candidate set."
  width: 1659
  height: 948
  fit: contain
  hide_caption: true
topic: rsna-knee
series: rsna-knee
series_order: 4
translation_key: rsna-knee-04
lang: en
---

My final pair did not contain the submission with my highest Public score. The public Helios recipe, H, displayed 0.958. I selected **v229 at 0.957** and **v236 at 0.953**, passing over a closely related clean composition at 0.954. This was a decision about which two predictions to keep under conflicting evidence, rather than a claim that the leaderboard had ranked them incorrectly.

The distinction matters because the two selected submissions were alternatives, not ingredients of a new average. A second candidate was valuable if it could outperform the first on Private. A different architecture or a lower correlation could motivate that possibility, but neither established it. I needed to explain why the second prediction deserved a slot and what unfavorable evidence I was accepting.

This account is written **as of October 22, 2026, at 09:00 KST (00:00 UTC)**, using only results known by that time. The deadline is still ahead: **October 22 at 23:59 UTC, or October 23 at 08:59 in Korea**. Private scores and the final rank remain unknown. The question is how I made the choice, not how an unseen result might eventually justify it.

## Public predictions and the criteria for further improvement

By the final week, a strong publicly reproducible prediction was a realistic alternative to another training investment. Successive public systems combined different useful elements: a CoAtNet recipe with mixed supervision and auxiliary data; compact ConvNeXt inputs with finding-specific pooling; and rank or probability blends whose weights varied by finding. Their published gains were not additive. The same checkpoint could appear in several notebooks, and two apparently different stacks could share most of their predictive content.

I separated three possible uses of that material. A ready checkpoint could strengthen a final test predictor immediately, without supplying compatible OOF. A complete recipe could be retrained on my folds to create a clean family. An input or pooling idea could motivate a new controlled model comparison. The first route bought deployment readiness; the second bought held-out evidence and a reusable prediction family; the third tested a mechanism. Requiring all three to satisfy the same evidence rule would either discard useful test predictors or misclassify full-data predictions as clean.

The competitive setting made that distinction urgent. In the saved October 21 snapshot, 1,471 of 6,307 teams displayed 0.958; another 228 teams displayed a higher score. The 0.958 block occupied ranks 229 through 1,699. That was a very different situation from treating 0.958 as an isolated frontier model.

![Numbers of teams at each displayed Public score in the saved October 21 snapshot]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part4-public-density.png){: width="1920" height="1408" }
_Figure 1. Team-best Public scores at or above 0.953, from the October 21 09:00:41 UTC snapshot. The large 0.958 group describes score density. It does not prove identical prediction vectors, identify every team's model ancestry, or assign a hypothetical final rank to an individual submission._

Copying an exact prediction can provide a strong benchmark, but cannot distinguish its copy from the original on the same cases. Merely observing the same three-decimal score is weaker: different unrounded AUCs and different prediction vectors can share that display. I therefore could not infer that all 1,471 teams were exact copies, or predict that the whole group would move together on Private.

The decision was consequently more specific than choosing the highest visible number. I could keep H's observed Public advantage; use H in a fixed blend with my own predictions; or devote a slot to a clean composition supported by another kind of evidence. A distinct composition was useful only if its differences could improve the ranking on unseen studies. Being different, being self-trained, or having clean OOF was not itself a score gain. Equally, H's crowded Public score did not prove it could not succeed on Private.

This framed the remaining work. A public recipe deserved reproduction when its mechanism and full delivery cost made it a credible new family. Another public blend deserved a test when the exact change could alter the final choice. Repeatedly changing small weights around a popular stack would consume comparisons without resolving whether my models supplied useful information beyond it. The subsequent experiments were attempts to answer those concrete questions.

## The pending supervision experiments changed what could be combined

The previous entry ended on October 15 with N's multimodal-label comparison unresolved. Later that day, N-mm12 scored **0.945** on Public against **0.942** for its definition-guided version. The positive local result now had a corresponding external result in the full-field learner. The comparison supported that particular combination of visual inputs and labels; it also made N-mm12 a useful contributor to subsequent compositions.

Two complete reproductions of public recipes expanded the clean comparison further. **R949F**, a CoAtNet recipe with the donor's preprocessing, continuous mixed supervision, and auxiliary OAI data, scored **0.944** on October 16. The related earlier submission used for comparison had scored **0.937**. **S6F**, a ConvNeXt-S recipe with finding-specific spatial and window attention, scored **0.952** on October 17. Both produced complete image-held-out OOF, retaining the source-label exposure to expert cases described in Part 3. S6F also strengthened the evaluated whole-bank selection procedure under the declared runtime budgets.

These results supported reproducing a coherent learning system when its components plausibly worked together. Input geometry determines what the encoder sees; the attention head determines which features contribute to each finding; supervision determines the distinctions those features learn. Changing all three means that the observed improvement cannot be assigned to the backbone alone. It can nevertheless establish a stronger model family to combine with independently trained alternatives.

Teacher supervision produced a different kind of candidate. **N-T\*** scored **0.951** on October 16. Its targets included predictions whose ancestry did not preserve the exclusions needed for clean student OOF, so I treated it as final-only. Combining the full-field and 2.5D-window T\* students scored **0.950**, below N-T\* alone. Two successful students did not imply an improved average: their common teacher content could reinforce similar rankings while their remaining differences hurt useful positive–negative pairs.

I therefore retained two ways to improve a final prediction. Clean OOF let me compare combinations on aligned held-out studies and evaluate how a selection procedure behaved across folds. Strong public and teacher-derived predictions could still be useful on test, but their complete mixtures required direct transfer comparisons. The distinction determined which evidence could select a combination locally, rather than prohibiting useful test predictors.

## The operators define two substantially different candidates

The first selected candidate, **v229**, was dominated by the public Helios prediction H. My contribution was a block containing the five fold models of N-mm12 and five of V3-mm12, the 80-slice model. Write $R$ for a per-finding percentile-rank transformation. The ten-member block averaged individual member ranks, and the outer blend was

$$
\begin{aligned}
A_{10}&=\frac{1}{10}\sum_{m=1}^{10}R(p_m),\\
C2_{0.25}&=0.75\,R(H)+0.25\,R(A_{10}).
\end{aligned}
$$

Ranking before averaging prevents one member's probability scale from dominating simply because its predictions are more spread out. It does not make all aggregation orders equivalent. Ranking a family's probability mean and averaging the ranks of its members can produce different study orders. The extra outer rank also converts the own block's aggregated values back to a common scale beside H.

The complete v229 mixture was **final-only** because H lacked compatible clean OOF. Its N/V3 constituents retained their clean status. I could study their behavior on held-out training studies without pretending those measurements evaluated the H-dominated final mixture.

The second selected candidate, **v236**, was **K-W2**, a probability combination of three clean families:

$$
\begin{aligned}
K\text{-}W2&=0.50\,B_{10}+0.25\,N+0.25\,V3,\\
B_q&=\sigma\!\left(\frac{1}{q}\sum_{m=1}^{q}z_m\right).
\end{aligned}
$$

Here $z_m$ is an S6F checkpoint's logit. $B_{10}$ combines two seeds across five folds each. N and V3 are their respective five-fold probability means. The earlier **K-W, v231**, used $B_5$ with the same outer weights. Adding an S6F seed changed the estimate inside a family while leaving its final weight at one half. These expressions describe hidden inference. For a local OOF study, I use only the held-out checkpoint from each seed, comparing $q=1$ with $q=2$ inside the S6F block.

This nesting is important. Sigmoid of mean logits is not mean probability; averaging twenty checkpoint predictions with equal weights would also give a different answer. A checkpoint count describes computational work, not its final vote. These exact operators define what the comparison between K-W and K-W2 can tell me.

| Comparison | v229 | v236, K-W2 |
|---|---|---|
| Public | **0.957** | **0.953** |
| Composition | H 0.75 + N/V3 rank block 0.25 | S6F two-seed block 0.50 + N 0.25 + V3 0.25 |
| Local evaluation | Whole mixture final-only | Complete clean OOF for the constituent families |
| Close alternative | v227, own share 0.50, Public 0.956 | v231, one S6F seed, Public 0.954 |

The final pair contrasted an H-dominated rank blend with an S6F-heavy clean probability blend. Both retained N and V3, so the difference in construction did not make them independent.

## Longer training versus additional seeds

A promising ensemble did not settle what to train next. S6F was already the strongest clean family on Public, but its 16-epoch recipe left a plausible question: would a larger training budget improve it further? Another seed asked a different question. It repeated a working learning recipe to change the estimate obtained from its finite training process, rather than trying to learn a stronger representation with a longer schedule.

I tested the longer-budget hypothesis with a fresh 32-epoch S6F pilot. This stretched the cosine and warmup schedule from initialization; it did not continue the 16-epoch checkpoint for sixteen more epochs. The planned complete five-fold cost was about 18.5 GPU-hours, of which about 3.7 belonged to the first fold and 14.8 remained afterward. These were reserved training hours on the cached single-GPU machine, not the nine-hour hidden-inference allowance.

The pilot provided a particularly useful distinction between standalone quality and ensemble contribution. On 870 matched non-gold studies, the result was:

| Fold-0 replacement, e32 minus e16 | Primary difference | Saved report-group interval |
|---|---:|---:|
| S6F alone | −0.00622 | [−0.01117, −0.00140] |
| S6F replaced inside fixed K-W | +0.00062 | [−0.00165, +0.00267] |

![Standalone and fixed-ensemble effects of the S6F 32-epoch pilot]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part4-e32-decision.png){: width="1920" height="1072" }
_Figure 2. The longer S6F recipe deteriorated alone while its fixed K-W replacement remained uncertain. Both contrasts use the same 870 matched fold-0 studies. The intervals describe conditional report-group variation; the remaining four folds were not trained._

Nine of twelve standalone finding scores fell. Yet replacing S6F inside K-W produced a small positive point estimate with an interval spanning zero. A weaker standalone prediction can change the difficult pairwise orderings of an existing ensemble differently from those it gets wrong alone. The pilot therefore did not justify the easy conclusion that a lower standalone score meant no possible ensemble value.

The difficult decision concerned what information the remaining 14.8 planned hours would buy. Completing the family could reveal whether fold 0 misrepresented the longer recipe and whether the full ensemble benefited. It would also delay the multimodal-label S6F variant and another seed of the established recipe by roughly fifteen hours. Their checkpoints still needed export, evaluation, and a complete test prediction before they could affect the final choice. A training result arriving just before the deadline would not have the same practical value as one I could actually compare.

I stopped the e32 pilot after weighing those alternatives. The uncertain K-W benefit, the standalone and target deterioration, and the alternative uses of the same machine together favored that choice. The partial predictions and checkpoint remained available, but there was no completed e32 family or Public result. Stopping gave up the full-family answer; it did not establish that S6F had converged or that every longer schedule would fail.

The existing two-seed S6F evidence helped make the alternative credible, but was not a calibrated return-per-hour comparison. Its approximately +0.000945 inside K-W came from complete OOF, whereas the e32 estimate came from one fold. Dividing those values by training hours would hide the different experiments. The choice was to prioritize a working recipe with positive complete evidence while preserving the unresolved longer-training idea, rather than to turn a pilot into a universal early-stopping rule.

## OOF composition selection and inference readiness

A separate question was whether the accumulated families could already produce substantially more than the compact K-W prediction. Restricting every experiment to adding one model to the current reference could miss a better replacement or a different combination. I therefore used whole-bank searches that could add, remove, and replace families. The reference remained a comparison, not compulsory membership.

The October 19 search compared two conditions on 4,349 non-gold studies. **P540** used one S6F seed and a 540-minute planning ceiling. **PUNC** used the two-seed S6F prediction and effectively removed the ceiling. Both optimized the Opus/GPT primary over 146 families with probability aggregation, a beam width of 16, at most ten rounds, and at most 50,000 sets per outer fold. They were substantial bounded searches, not exhaustive optimization over every ensemble operator and weight.

The distinction between a selected procedure and one final fixed composition matters here. For fold $k$, the selector chose a composition using the other folds and evaluated it on $k$:

$$
S_{-k}=\operatorname{Select}(D_{-k}),
\qquad
\widehat A_{\mathrm{select}}=\frac1K\sum_k A_r(D_k,p_{S_{-k}}).
$$

Fitting one composition on all the data gives another object, $S_{\mathrm{all}}$. Its score on the data used to choose it cannot simply inherit the procedure's held-out estimate. In this study, a smaller procedure that picked the best fourth family chose different GPT-label models across folds. I then fixed one additional family, $U$, and combined it as $C1U=0.8\,K\text{-}W+0.2\,U$. C1U was this four-family prediction, not the added family alone. Its post-selection score was not a new untouched validation of that winner.

| Comparison on the same studies and primary | Pooled AUC | Mean within-fold AUC |
|---|---:|---:|
| Fixed K-W | 0.901375 | 0.901405 |
| Best-fourth-family selection procedure | 0.902831 | 0.902951 |
| P540 c1 procedure | 0.902772 | 0.903094 |
| PUNC c1 procedure | 0.902750 | 0.903067 |
| Fixed C1U, selected after the scan | 0.903310 | 0.903372 |

The large searches did not show a compelling extra primary payoff over the small fourth-family procedure. Their fitted delivery proposals were much heavier: seven families at 535.3 planning minutes for P540, and eleven at 1009.7 for PUNC. The near-cap P540 alternative also introduced decoding groups and little nominal margin. Similar conditional scores did not make these proposals equally ready to execute.

![Whole-bank selection estimates and the separate cost of their fitted proposals]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part4-search-frontier.png){: width="1920" height="1296" }
_Figure 3. Selection-procedure scores and fitted-proposal planning times answer different questions. P540 and PUNC also use different S6F inputs, so their score difference does not isolate the effect of removing the time ceiling. The fitted rosters do not inherit the procedure estimates as their own validated AUCs._

The tempting summary was that the cap did not bind. That was too strong. Capacity clearly changed the selected roster and its cost, and the uncapped condition changed S6F content as well. What I had observed was no useful additional primary point gain from this larger best-found reference under this particular search. A different objective, operator, or search could behave differently. The result justified declining one expensive proposal, not declaring the bank exhausted.

There was another reason to keep the fixed transfer comparison. C1U's local gain depended on the evaluation labels: Opus gave about +0.001910 and DeepSeek +0.001773, while Gemini3f gave −0.001235 and dread −0.001779. Dread's losses covered all twelve findings. These were substantive opposing descriptions of the same predictions, not independent votes. Primary also included the GPT source used to train the added family. I retained the supportive evidence without allowing its larger maximum to erase the disagreement.

The estimates remained conditional on an already exposed OOF bank. Selection cross-fitting did not retrain every base model inside each outer split, and the saved bootstrap intervals did not include all recipe, seed, and winner-selection uncertainty. Those limits did not make the clean OOF useless. They explained why the next useful comparison was one fixed C1U transfer test, with K-W still available, rather than implementation of a near-cap roster solely because a search had produced it.

The later 155-family refresh reinforced that distinction. New seeds changed the proposed composition, but its fitted 533.8-minute roster still needed an assembled, tested implementation. Three equal-probability S6F groups gave S6F three fifths of the total; collapsing them into one logit-averaged bundle would change the predictor. Its mixed fold and target evidence did not justify displacing the ready comparisons. Producing another complete family expanded what I could analyze, without requiring me to append it to every final prediction.

## Public comparisons were useful when they answered a concrete question

The equal-family combination of S6F, N-mm12, and V3-mm12 scored **0.953**. Increasing S6F's probability weight to one half produced **K-W at 0.954**. This identified a useful clean reference and suggested that treating every family equally was not necessarily the best use of the available prediction strength.

On the mixed side, the declared comparison concerned how much own prediction to combine with H. An own share of **0.50 gave v227 at 0.956**; a share of **0.25 gave v229 at 0.957**; H alone scored **0.958**. In these comparisons, more own contribution lowered the observed Public score. The data did not establish that the own block improved H on Public, and I kept that unfavorable result in the decision.

![Completed candidate comparisons and the selected pair]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part4-candidates.png){: width="1920" height="1120" }
_Figure 4. Completed Public comparisons, including H, the selected v229 and v236, and their scored alternatives. A timeout supplies no AUC and is not represented as a low-quality prediction._

A small number of coarse weights was useful because it tested transfer of the complete mixed prediction, which clean OOF could not evaluate. Repeatedly filling in nearby weights would increasingly tune to the same exposed Public studies. I kept a weak preference for 0.25 over 0.50 instead of reading the displayed difference as a precise optimum. H alone remained a serious alternative, particularly for a reader who gave the best observed Public result the greatest weight.

The clean variants tested content as well as allocation:

| Candidate | Change relative to K-W | Public |
|---|---|---:|
| C1U, v233 | Additional GPT-label family and changed allocation | 0.951 |
| N2, v234 | Two N seeds, with changed content and allocation | 0.954 |
| S6FX, v235 | Full-data model inside the S6F block; final-only | 0.954 |
| K-W2, v236 | Two S6F seeds, keeping the family weights | 0.953 |

C1U's unfavorable Public result resolved the fixed transfer question raised by the whole-bank analysis, while leaving the added family's clean predictions available for other combinations. N2's equal Public score did not isolate a seed effect because its N/V3 allocation also changed. S6FX could be useful as a final predictor, but the added full-data model changed what its local evaluation could establish. K-W2 supplied the clearest matched question by retaining N, V3, and the outer weights.

Each comparison changed a specific part of the prediction. The results did not support automatically retaining every additional family in a larger composition.

## K-W2 exposed a real conflict between local and Public evidence

K-W2 improved the matched clean primary score over K-W by approximately **0.00095**. The saved conditional interval was about **+0.00055 to +0.00136**. All five fold differences were positive, as were the other evaluated valid report-label readers, although four findings had small losses. Public moved in the opposite direction: **0.953 instead of 0.954**.

The local result was persuasive in a specific way. The change replaced one S6F seed with two while holding the rest of the prediction fixed. The improvement was distributed across folds rather than concentrated in one obvious partition. Agreement across evaluated readers made it less dependent on one label convention, although those readers shared reports and related assumptions. The result described a stable conditional improvement on the observed cohort, not independent expert confirmation.

![Local improvement and the opposite Public difference for K-W2]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part4-evidence.png){: width="1920" height="1040" }
_Figure 5. K-W2 minus K-W: the local primary difference is about +0.00095, with its saved conditional interval, while displayed Public changes by −0.001. The interval concerns the local comparison; it does not measure Public noise or the probability of a Private reversal._

Why could another seed help locally and hurt elsewhere? For a probability blend, with $m$ indexing its probability components, the margin on a positive–negative pair is

$$
D_{ab}=\sum_m w_m\bigl[p_m(x_a^+)-p_m(x_b^-)\bigr].
$$

AUC counts whether these margins are positive, with half-credit for ties. An added prediction can reinforce a correct ordering, repair a wrong one, or reverse a narrow correct margin. Reducing prediction variability is not the same objective as improving every relevant ordering. S6F's inner logit average changes the calculation, but the same pairwise issue remains: the studies near a ranking boundary matter more than agreement on easy pairs.

This is also why overall prediction correlation is insufficient. Two models can correlate closely across thousands of easy cases and disagree on the relatively few pairs that determine the AUC difference. A new reference changes which studies count as positive; a new test distribution changes which difficult examples occur. The relative usefulness of a seed can therefore change without either observed result being fictitious.

## Execution limits of larger ensembles

The strongest available weights also supported larger compositions. A second N seed could accompany the two S6F seeds; a third S6F seed could replace the two-seed estimate; a full-data model X could enter S6F's logit average. These changes were worth distinguishing because they altered different parts of the prediction.

![Model counts and execution outcomes for related compositions]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part4-budget.png){: width="1920" height="1200" }
_Figure 6. Fold-checkpoint counts for related S6F, N, and V3 combinations, with X separately identified as a full-data model. Counts are not final probability weights. S6F retains an outer weight of 0.5, and X makes its descendants final-only. C2-N3 and R7+O use different constructions._

N2W2 combined two S6F seeds with two N seeds, changing N/V3 allocation as well as model content. K-W3 added a third S6F seed at the same outer weight. In S6FX, X was one of six S6 models; in K-W2+X, it was one of eleven. A favorable result for the former would not predict the latter's contribution, because X's weight inside the block changed. C2-N3 instead enlarged v229's own rank block from ten to fifteen members at a fixed outer share.

All six larger complete-or-fail trials ended without a score:

| Candidate | Question | Result |
|---|---|---|
| v237, N2W2 | Two S6F seeds together with two N seeds | Timeout; no AUC |
| v238, C2-N3 | Fifteen-model own rank block beside H | Timeout; no AUC |
| v240, R7+O | Public R7 with a larger final-only own block | Timeout; no AUC |
| v242, K-W2+X | Full-data X inside the two-seed S6F block | Timeout; no AUC |
| v243, N2W2+X | X inside the larger S6F/N combination | Timeout; no AUC |
| v244 retry 1, K-W3 | Third S6F seed at the same family weight | Timeout; no AUC |

The original v244 attempt had a separate platform system error; its retry also timed out. These outcomes establish that the exact submitted implementations did not complete within the allowed execution. They do not assign a low AUC to the underlying ideas. Hidden-stage timings were insufficient to tell me how much extra time each needed or which stage failed.

Planning estimates could not settle this in advance. Shared image decoding and two-GPU scheduling make combined runtime differ from the sum of standalone runs. Conversely, increasing model count can change memory pressure and scheduling. A successful composition and a slightly larger failed one do not define a universal safe boundary in minutes. The useful result was narrower: the successful candidates remained selectable, while these six did not become alternatives.

The nine-hour constraint also made time for evaluating the result part of the experiment's value. Preparation, a full run, platform waiting, and result inspection all had to finish before a new prediction could inform the final choice. With no changed implementation or new execution evidence, repeating a timeout would spend the remaining opportunity on the same unanswered question. I retained the scored candidates instead.

## Why keep v229 and v236?

V229 offered a strong H-dominated predictor with a modest own contribution. V236 emphasized the strongest clean family and preserved a positive matched local comparison despite its unfavorable Public result. The value of keeping both depended on whether their different construction could make the second outperform the first on Private. Shared N/V3 content and related training data meant I could not treat that as independent insurance.

I initially planned to keep K-W if K-W2 displayed 0.953. Reexamining the broad matched clean improvement changed that preference to **v236**, despite its lower Public score. I had no calibrated exchange rate between a local AUC increment and a Public decrement; averaging them into a composite score would conceal the assumptions rather than resolve them.

Several alternatives remained defensible. H alone had the highest observed Public score at 0.958. V227 retained a larger own share at 0.956. **V231 remained a reasonable alternative**, with the stronger clean Public result at 0.954. Choosing v236 meant accepting that the local improvement I valued might not help on Private. Choosing v231 meant forgoing the change that had improved the matched local comparisons consistently. The evidence did not establish which would be better on Private.

I kept **v229 plus v236**. The clean predictions from the other complete families remained available for later analysis; their exclusion from the final pair did not erase the questions they could answer. The Private result will show how these two predictions performed. It will not retrospectively make the conflicting evidence disappear: the choice accepted a lower observed Public score in exchange for a different composition and the matched clean result.

**Previous:** [Part 3: Supervision and Public Performance]({{ site.baseurl }}/posts/RSNA-Knee-Part-3-When-Labels-and-the-Leaderboard-Disagreed/)
