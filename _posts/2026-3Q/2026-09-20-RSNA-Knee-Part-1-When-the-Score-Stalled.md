---
title: "RSNA Knee, Part 1: Training Targets and Validation References"
date: 2026-09-20 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, rsna-knee, weak-supervision, out-of-fold, model-selection, ensemble]
math: true
pin: false
published: true
image:
  path: /assets/img/posts/rsna-knee-series/cover-1.webp
  alt: "Methodology cover: distinct training-target and evaluation-reference paths."
  width: 1659
  height: 948
  fit: contain
  hide_caption: true
topic: rsna-knee
lang: en
translation_key: rsna-knee-01
series: rsna-knee
series_order: 1
---

By August 19, my RSNA Knee system had reached a Public AUC of 0.921. The next decision was where another experiment could help: a different report reader, a more detailed view of the knee, or another pretrained encoder. Each proposal sounded reasonable. The difficulty was telling whether a result evaluated that proposal at all. A label comparison, a trained image model, and a selected ensemble could all produce numbers called AUC while answering different questions.

One result made that distinction difficult to ignore. Revising the report labels increased their agreement with an expert panel from 0.8700 to 0.8809. An image model trained with the revised setup, however, scored 0.8322 on its annotated holdout, compared with 0.8400 under the earlier setup. These were different panels and procedures, so the changes cannot be converted into a causal relationship. They did pose a precise question: **what could improve in a label table without improving the model trained on it?**

The experiments through September 20 gave that question substance. A finer label schema assigned substantial loss to findings the report had never discussed. A smaller crop and a larger image partly bought the same improvement. An encoder rejected after thirty epochs recovered some strength when given more opportunity to adapt. To interpret those results, I had to distinguish the training target, the held-out prediction, and the reference used to score it—and then ask whether the training comparison was adequate.

## A training target is not a validation reference

In [Part 0]({{ site.baseurl }}/posts/RSNA-Knee-Twelve-Findings-From-One-MRI/), I described the basic task: infer twelve findings from an MRI study, using report-derived supervision during training. A report reader here is a language model that converts a radiology report into finding labels. It does not inspect the MRI or supply an independent clinical diagnosis.

For a study, I distinguish the soft training target $q_i^{(\ell)}$, the image model's prediction $p_i=f_\theta(X_i)$, and a binary validation reference $y_i^{(r)}$. The superscript $\ell$ identifies the training-label construction; $r$ identifies the reference interpretation. The training target may contain probabilities or uncertainty values, whereas the validation reference must define positive and negative cases for AUC. Changing either changes the comparison, even when the MRI inputs stay fixed.

The first label improvements were substantial. On the 58 supplied expert-annotated studies, rule-derived labels achieved about 0.812 macro AUC, compared with about 0.873 for a Sonnet report reader. Richer report interpretation was worth pursuing. The later rank-averaged Sonnet–Opus labels exposed the limit of using that agreement alone:

| Comparison | Earlier setup | Revised setup |
|---|---:|---:|
| Label-table AUC on a 57-study expert panel | 0.8700 | 0.8809 |
| Image-model AUC on its annotated holdout | 0.8400 | 0.8322 |

*The label-table comparison uses a 57-study panel, rather than the 58 studies used for the earlier rule-versus-reader comparison.*

The reader could use explicit wording, diagnostic conclusions, and descriptions of uncertainty. The image model had to learn from the selected views and slices. A target table could agree more closely with a small reference panel while changing the prevalence, confidence, or distinctions presented throughout the training set. The experiment did not identify a single cause of the model's decrease; it showed why a good label-table result could not stand in for a downstream training comparison.

## Rank transformations of predictions and training targets

AUC depends on ordering. Applying the same strictly increasing transformation to one predictor's scores preserves its positive–negative ordering, apart from numerical ties. This familiar property made rank averaging attractive when combining outputs whose scales differed.

A percentile rank is nevertheless a position in a distribution, not a probability. An example at the 60th percentile need not have a 60% probability of containing a finding. If most reports are negative for that finding, even many negative examples can receive middle-ranking values. Replacing probability targets with ranks therefore changes what the image model is asked to predict.

Binary cross-entropy uses the target value directly:

$$
\mathcal L(q,p)=-q\log p-(1-q)\log(1-p).
$$

For a model logit $z$, with $p=\sigma(z)$, its gradient is

$$
\frac{\partial\mathcal L}{\partial z}=p-q.
$$

Consider an illustrative example. The model currently predicts 0.20. With a target of 0.02, the logit gradient is +0.18, so gradient descent lowers the logit and predicted probability. If a rank transformation changes the target to 0.60, the gradient becomes −0.40: the same image now pushes the prediction upward. A transformation that preserves the label table's AUC can reverse its training signal.

![Binary cross-entropy logit gradient as a function of the target, at a fixed predicted probability]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part1-target-scale.png){: width="1920" height="1040" }
_Figure 1. Illustrative values at a fixed prediction of 0.20. The two target encodings produce opposite update directions. This explains the loss, rather than estimating the cause or size of an observed model-score change._

Rank-based targets can still define a useful learning task. They simply need to be tested as that task, rather than treated as probabilities with an unchanged meaning. Confidence weighting introduces a different change: it alters how strongly a cell contributes to the loss. The distinction became important in an actual label experiment.

## What should the model learn from an unmentioned finding?

The finer, or `wide`, label schema was intended to express more of the report's uncertainty. Its first difficulty concerned the order of two meanings: an explicit denial and a finding the report never raised. In the small annotated panel, 4.1% of 196 explicitly denied finding cells were positive in the image annotation, compared with 13.6% of 177 unmentioned cells. Those pooled cells were not independent patients, but the contrast gave a concrete reason not to assign silence a stronger negative meaning than an explicit denial.

The initial table had reversed that order. Its low correlation with the earlier labels, 0.557, first looked like a different supervision signal; correcting the ordering raised the correlation to 0.878. The interesting scientific question remained after that correction: did a finer schema make better image supervision?

The corrected table's model still lost roughly 0.013–0.027 under the report-derived references. Its target-level results were much less uniform:

| Findings grouped by how often reports left them unmentioned | Mean model AUC change |
|---|---:|
| Five findings with an omission rate below 11% | +0.0068 |
| Six findings with an omission rate above 30% | −0.0323 |

The remaining finding lay between those omission groups. Across findings, the reported rank correlation between omission rate and score change was −0.657. Extra label detail appeared useful where reports usually spoke, while the losses concentrated where reports often said nothing.

The loss gave a testable explanation. The example weight was $w_i=0.25+0.75c_i$, where $c_i$ was the reader's confidence. Even a very uncertain cell retained positive weight. Its logit gradient was $w_i(p_i-q_i)$: the model was still pushed toward the assigned value, just less strongly. When a large fraction of the table was unmentioned, reproducing that convention remained a substantial part of training.

A matched run that masked low-confidence wide-label cells improved by a reported +0.0083, with a paired interval of [+0.0055, +0.0112].[^mask-reference] Fracture, unmentioned in about 66% of its report cells, gained +0.0475; Contusion gained +0.0138. This made masking a serious candidate, rather than merely another encoding preference.

![Report omission, downstream model changes, and the matched masking comparison]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part1-omission-mask.png){: width="1920" height="1200" }
_Figure 2. The omission groups and the masking comparison describe different comparisons: the first compares the corrected wide-label model with its baseline, and the second changes masking within the wide-label recipe. The masking interval is the reported study-resampling interval for that matched comparison. It is not an uncertainty estimate for all label schemes._

The evaluation complication was equally concrete. A reference that marks unmentioned findings negative can reward a model for reproducing that same reporting habit. Removing their training loss may then reduce agreement with the reference without reducing agreement with image findings. Masking also removed studies with no supervised cells left—40 under the wide table—so the comparison had to use shared eligible studies. I could investigate the affected findings and compare references; neither the label score nor a single masked-run score resolved the clinical question by itself.

## Crop and resolution partly bought the same improvement

Pixel-side experiments supplied a useful contrast. At a fixed 336-pixel image size, changing the physical crop from 130 mm to 100 mm gained about 0.005 in paired comparisons of held-out predictions. Going wider to 160 mm lost 0.0015; cropping further to 80 mm gained only 0.0019. The useful crop was in the middle of the tested range. Increasing apparent detail helped until removing context or anatomy became a competing cost.

Resolution could change that balance in a different way. Increasing the grid from 336 to 448 pixels at a 130 mm crop gained about 0.003. Combining 448 pixels with the 100 mm crop gave about +0.0053, almost the same as the 100 mm crop at 336 pixels. Treating crop and resolution as independent gains would have exaggerated the value of their combination.

The representation explains why overlap was plausible. The nominal output pitch is the physical width divided by the grid width: 448 pixels over 130 mm gives about 0.290 mm per output pixel, close to 336 pixels over 100 mm at 0.298 mm. Both changes place a small structure across more input pixels. This is a property of the resized input, not a claim that interpolation improves the scanner's acquired resolution.

![Paired OOF effects of crop and output grid size]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part1-crop-scale.png){: width="1920" height="1184" }
_Figure 3. Rounded effects from the dated paired comparisons, relative to 336 pixels and a 130 mm crop. The combined change gains little beyond the smaller crop alone. Similar output pitch explains a shared effect, while the different field of view means the two representations are not equivalent._

Pitch was not the entire explanation: the 336-pixel, 100 mm representation beat the nearly pitch-matched 448-pixel, 130 mm representation by about 0.002. Cropping also changes which anatomy and background enter the model. The practical choice was therefore to favor the useful smaller representation over paying for a larger grid whose additional gain had not appeared in this combination.

These comparisons were also less sensitive to the reference labels than the reader swaps. In the tested set, pixel-side deltas varied across references by about 0.0002–0.0009, while label-side deltas varied by 0.0068–0.0073 and sometimes changed sign. That did not validate any reader as truth. It showed which experiments retained their interpretation when the reference changed, and which needed closer investigation.

## Image holdout and the role of reference labels

Out-of-fold predictions address a different source of optimism. With five folds, each study is predicted by a model trained without its fold:

$$
p_i^{\mathrm{OOF}}=f_{\theta_{-k(i)}}(X_i),
$$

where $k(i)$ is the study's fold. I also grouped repeated report text to reduce one route for duplicated information across folds. This supplies held-out image predictions across the cohort; it does not turn report-derived references into independent clinical truth.

For a specified predictor or ensemble $P$, a reference-label score is

$$
S_r(P)=\frac{1}{12}\sum_{t=1}^{12}
\operatorname{AUC}\!\left(y_t^{(r)},p_{P,t}^{\mathrm{OOF}}\right).
$$

The reference determines which studies count as positive and negative for each target. Changing how a reader interprets an uncertain phrase or an omission changes the positive–negative pairs used by AUC. The image predictions can stay identical while the measured score changes.

![Training supervision, held-out MRI predictions, and the different references used to evaluate them]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part1-evaluation.png){: width="1920" height="1040" }
_Figure 4. Image holdout and reference independence are separate properties. OOF addresses whether the image learner fitted a study; it does not establish how accurately or independently a report reader defines that study's findings._

This matters most when the training target and reference share a reader or a related interpretation. Agreement with that convention is an observable result, but its meaning remains conditional on the convention. The small expert panel supports direct examination of annotated cases; broader report-derived references support cohort-wide comparisons; Public AUC measures a submitted system on the competition's scored cases. Together they can expose disagreements. They do not provide interchangeable measurements.

## Limits of optimizing the minimum AUC

I initially tried to avoid improving under only one reference by maximizing the minimum score:

$$
S_{\min}(P)=\min_r S_r(P).
$$

This looks conservative: improve the weakest result before receiving credit. The difficulty was that the scores used different reference labels. Their absolute levels differed for reasons beyond the candidate change. If one reference stayed lower by much more than candidates moved, it would determine the minimum throughout the comparison.

That was the observed behavior. Among 200 sampled candidate ensembles, the rule-based v1 score was the minimum in all 200. Within that tested set,

$$
S_{\min}(P)=S_{\mathrm{v1}}(P).
$$

One recorded candidate shows the consequence:

| Reference | Starting ensemble | Change | Candidate ensemble |
|---|---:|---:|---:|
| Rules, v1 | 0.8210 | +0.0058 | 0.8268 |
| Opus reader | 0.8791 | −0.0048 | 0.8743 |
| GPT reader | 0.8835 | −0.0046 | 0.8788 |

*Values and changes are rounded independently; adding displayed entries can differ in the final digit.*

The candidate improved the minimum while worsening both other scores. Their declines were too small to close their large level gap with v1, so neither decline affected the objective. A rule intended to balance references instead rewarded one interpretation exclusively in this region.

![The minimum raw AUC rises while both other reference scores decline]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part1-minimum.png){: width="1920" height="1040" }
_Figure 5. The recorded example and the 200-candidate check expose the same mechanism: v1 controls the minimum because its score level is lower. This conclusion is limited to the evaluated candidates._

I replaced that use of the minimum with a working primary score based on the mean of Opus and GPT, retaining v1 as a separate veto check. The change let gains and losses under those references affect selection. It did not make Opus and GPT independent or uniquely correct; their agreement still had to be interpreted alongside individual findings, annotated cases, and competition results.

## Initial performance and further training

Even with a declared reference, another question remained: had the new model received enough adaptation for the result to be informative? The rad-dino experiments made this a concrete choice. The encoder had been pretrained on chest X-rays and could replace DINOv2-base without changing the input architecture. Holding the rest of the recipe fixed seemed a clean way to test a different initialization.

Under thirty epochs with six unfrozen blocks and a backbone learning rate of $8\times10^{-6}$, rad-dino was weak. Its five-fold OOF was not selected in any of the five folds under any of three references. Yet its recorded best epochs were 29, 29, 30, 30, and 29. The comparison had fixed the adaptation budget at a recipe developed for DINOv2. A different initialization might need a different amount or form of fine-tuning.

Two first-fold retests addressed that concern:

| rad-dino recipe | Best first-fold AUC against the GPT-derived reference |
|---|---:|
| Original 30 epochs | 0.8356 |
| Continue the same run to 60 epochs | 0.8417 |
| Open all 12 blocks, learning rate $2\times10^{-5}$, 30 epochs | 0.8429 |
| DINOv2 control | 0.8795 |

The longer run peaked at epoch 48 and ended at 0.8393. The retests recovered about 0.006–0.007, but left a substantial gap. I could stop prioritizing this line with a more informative comparison than the original thirty-epoch result. It remained a completed OOF family available for future combinations.

A terminal best epoch by itself would have been weaker evidence. The cyclic learning rate naturally becomes small at the end of each cycle, so peaks can accumulate there even without a large gain from another cycle. What mattered was that the retest actually measured additional training and passed its earlier peak. Nor did the two retests establish that no other recipe could work: opening more blocks and changing the rate tested a different adaptation strategy, not every possible one.

The ensemble comparison also prevented standalone weakness from deciding the question alone. In the original local comparison, adding rad-dino to the existing seven-family package reduced the worst-reference score by only 0.0007, far less than its standalone deficit. A small ensemble loss alone did not establish useful complementarity: mixing in a weaker member can also reduce the size of its effect. The unresolved adaptation question justified a bounded retest; its limited recovery then favored spending the next substantial training effort elsewhere.

## What the next experiment has to distinguish

By September 20, I had more useful questions than “did AUC go up?” For supervision, I could separate the reader's ordering from target values and loss weights. For the image representation, I could test whether two changes bought the same detail or altered coverage in different ways. For an encoder, I could distinguish an unfavorable result under the incumbent recipe from a result after a reasonable adaptation test.

None of these comparisons identified the final model. They made the competing uses of another experiment clearer. The next problem was to produce more complete, genuinely different predictions and decide which combinations were worth evaluating under the nine-hour inference limit.

[^mask-reference]: The original summary does not name the reference labels for this comparison. The reported local gain is not directly comparable with later primary gains or with clinical/Public performance.
