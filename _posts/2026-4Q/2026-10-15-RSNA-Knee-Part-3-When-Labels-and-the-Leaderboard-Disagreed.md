---
title: "RSNA Knee, Part 3: Supervision and Public Performance"
date: 2026-10-15 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, rsna-knee, medical-imaging, weak-supervision, cross-validation, knowledge-distillation]
math: true
pin: false
published: true
image:
  path: /assets/img/posts/rsna-knee-series/cover-3.webp
  alt: "Methodology cover: supervision sources, a learner, and local and Public evaluation."
  width: 1660
  height: 948
  fit: contain
  hide_caption: true
topic: rsna-knee
series: rsna-knee
series_order: 3
translation_key: rsna-knee-03
lang: en
---

In early October, I could make an image model score worse locally and better on the leaderboard by changing its training labels. This happened across five matched imaging setups. A different experiment, changing only the training objective, improved both. A third held the labels fixed, replaced the image recipe, and again reversed the local ordering on Public. These were competing explanations of what my models had learned, not a single noisy ranking to be averaged away.

Most of the 4,407 training studies had radiology reports rather than the expert annotations used for competition evaluation. A report reader converted the text into twelve target values. That left several possible bottlenecks: how the finding was defined, what the report omitted, which parts of the MRI the model could see, and how its aggregation head combined those images. The strongest public recipes changed several of these together. I needed to decide which complete systems deserved reproduction and which narrower hypotheses could be tested with the models already available.

This article follows those choices using results available by **October 15, 2026, at 21:00 KST (12:00 UTC)**. The native-field model's multimodal-label comparison still has no Public result. Its pending outcome, and the incomplete reproductions of two strong public recipes, are the unresolved questions at the end of this period.

## Training targets and the optimal prediction function

For one finding, let $X$ denote the MRI, $R$ its report, $Y$ the desired expert label, and $q(R)\in[0,1]$ the target extracted from the report. With image prediction $p_\theta(X)=\sigma(z_\theta(X))$, binary cross-entropy is

$$
\mathcal L(q,p)=-q\log p-(1-q)\log(1-p),
\qquad \frac{\partial\mathcal L}{\partial z}=p-q.
$$

Part 1 used this gradient to explain why a percentile and a probability are different training instructions. The consequence here goes beyond the update on one example. An ideal learner minimizing expected BCE, with sufficient capacity and no other restriction, approaches

$$
p^*(x)=\mathbb E[q(R)\mid X=x].
$$

That is the expected report-derived target conditional on the image. It need not equal $\Pr(Y=1\mid X=x)$, the expert-defined quantity whose ranking matters on test. A mild finding may be visible but absent from the report. A reader may interpret “possible” as uncertainty about presence, another as a severity boundary. More examples can make the model better at reproducing either convention without resolving the difference between them. A calibration difference alone need not change AUC; what matters is whether the two conditional quantities order studies differently.

Target transformations therefore change the learning problem. Hardening $q$ at a threshold $\tau$ asks the learner to estimate $\Pr(q>\tau\mid X=x)$ instead of $\mathbb E[q\mid X=x]$. Replacing $q$ by its percentile makes the desired output depend on the score distribution across the dataset. AUC's invariance to a monotonic transformation of a *finished prediction* does not make these training operations equivalent.

Confidence weighting changes another part of the objective. For a nonnegative weight $w$ with positive conditional mean, the ideal weighted-BCE solution is

$$
p_w^*(x)=\frac{\mathbb E[wq\mid X=x]}{\mathbb E[w\mid X=x]}.
$$

This can reduce the influence of unreliable examples, but it can also favor conspicuous findings and conventional report language. The target says what answer to learn; the weight says how much that answer matters. I needed separate comparisons for label content, target hardening, and confidence weights rather than treating all three as interchangeable forms of label cleanup.

The local **primary** metric averaged AUC against the Opus- and GPT-derived report-reference tables; **v1** was an older reference. Their binary decoding rules belonged to their definitions and were distinct from the soft training targets. Complete five-fold OOF supplied image predictions from models that excluded the evaluated fold. The 58 expert-annotated studies, **gold58**, supplied a smaller expert comparison, but repeated inspection had already exposed them to many choices. None of these channels was an untouched universal authority on the test ranking.

## Agreement and reversals between local and Public comparisons

The first useful evidence came from holding the imaging setup fixed. By October 6, five earlier GPT-label students had matched counterparts trained on the alternative Raptor table:

| Imaging setup | Public, earlier → alternative labels | Public difference | Local primary difference |
|---|---:|---:|---:|
| 64-slice fusion | 0.920 → 0.932 | +0.012 | −0.0054 |
| 80-slice gated model | 0.935 → 0.939 | +0.004 | −0.0157 |
| 2D window model | 0.899 → 0.903 | +0.004 | −0.0204 |
| ConvNeXt-B gated model | 0.923 → 0.925 | +0.002 | −0.0222 |
| 80-slice fusion | 0.934 → 0.936 | +0.002 | −0.0181 |

![Local and Public differences after matched training-label changes]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part3-labels.png){: width="1920" height="1200" }
_Figure 1. Changing training labels reduced primary and increased displayed Public AUC in five related setups. Shared data and label sources make these related experiments, not five independent estimates of a reversal probability._

The result was a reason to retain this supervision hypothesis despite its negative primary differences. Shared interpretive choices between the control's training labels and the local references could explain part of the discrepancy. Prevalence, severity encoding, and optimization could also contribute. The experiments established the reversal; they did not identify a unique cause or a correction factor that would translate every future local difference into Public AUC.

Giving the report reader explicit finding definitions produced a different pattern. Definition-guided students improved Public by 0.012, 0.015, and 0.005 in three setups, while primary changed by +0.0031, +0.0109, and −0.0012. Removing confidence weights gave a much smaller local change, about +0.0009 with an interval crossing zero. Hardening the definition-guided targets was substantially worse: primary fell by about 0.0147, with negative differences in every fold and all twelve findings. The continuous target encoding was doing useful work in this recipe, even though that result did not tell me whether its useful content was uncertainty, severity, or regularization.

I also needed to distinguish a reference mismatch from a defect in every local comparison. A paired-window DINOv2-B experiment provided a useful counterexample. The two training views had probabilities $p_1,p_2$ and the same study-level target $q$. The control averaged the two losses; the alternative applied BCE after averaging their probabilities:

$$
\begin{aligned}
\mathcal L_{\mathrm{mean}}&=\frac{\mathcal L(q,p_1)+\mathcal L(q,p_2)}2,\\
\mathcal L_{\mathrm{prob}}&=\mathcal L\!\left(q,\frac{p_1+p_2}{2}\right).
\end{aligned}
$$

At fixed predictions, convexity of BCE in its probability argument gives $\mathcal L_{\mathrm{prob}}\leq\mathcal L_{\mathrm{mean}}$. The distinction matters when a positive study contains windows without the lesion. Mean BCE asks both windows to match the study target individually; probability BCE judges their combined prediction. It better matches the declared probability-averaging inference operation, although the inequality alone cannot establish which trained model will have better AUC.

Here the empirical comparison agreed across channels. With the backbone, labels, input, five-member count, and inference held fixed, probability BCE improved primary by **0.00693**, with a recorded paired interval of **[0.00534, 0.00859]**. Public rose from **0.896 to 0.905**. By contrast, the **G25 2.5D window model** beat the **V3 80-slice gated recipe** by **0.00942** on primary, with the same definition-label table, but scored **0.929 versus 0.940** on Public.

![Two different local-to-Public transfer comparisons]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part3-transfer-contrasts.png){: width="1920" height="1088" }
_Figure 2. A controlled objective change transferred in the local direction; a whole-image-recipe change with the same labels did not. Local intervals are the saved paired intervals for fixed OOF predictions and references. Public points are displayed outcomes, without an assumed noise model. The two contrasts do not isolate a general architecture effect._

Keeping the labels fixed was therefore insufficient to make every local ranking portable. Keeping the architecture fixed was useful for interpreting a label intervention, but did not make the references independent of its content. My working decision was to use local evidence for the particular mechanism it measured, preserve the complete OOF predictions for ensemble comparisons, and spend a limited number of Public comparisons on explicit transfer questions. Discarding all cross-architecture OOF would throw away useful complementary predictions; treating its absolute ordering as the test ranking would repeat the G25 mistake.

## Public recipes: components and reproduction hypotheses

Reading high-scoring public code changed what reproduction meant. A public model could be useful immediately as a final-only predictor. Retraining a source-derived system on my folds could instead create a new set of image-held-out predictions. Borrowing its crop, attention head, or labels was a third option: a mechanism experiment whose result did not reproduce the original system. These uses had different costs and answered different questions.

The [r949 recipe](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) made this distinction concrete. Its **0.949** result combined a CoAtNet encoder, a particular physical crop, per-finding attention, mixed report labels, partially labeled external knees, and mirrored inference. The accompanying performance ladder changed several choices together. It was evidence that the complete system was strong, rather than a measured gain attributable to any single component.

Even the meaning of its input required reading the implementation. Its processed field was a **140 mm centered square resized to 384 pixels**; accuracy inference took a **320-pixel center crop**, approximately 116.7 mm when the physical crop was not clamped by the source image. This was broader training context relative to its inference crop, not an uncropped whole-DICOM image. Reusing a different 336-pixel cache or the existing head from the 96-slice G96 model would change the hypothesis before training began.

| Recipe | Input and aggregation mechanism | What a new reproduction would test |
|---|---|---|
| r949-derived R949F | Fixed sequence slots; three-slice windows; plain per-finding window attention; mirrored read | Whether the coupled source geometry, training and supervision transfer to declared folds |
| S6-derived S6F | External knee locations; per-finding spatial attention followed by window attention; view and position information | Whether localization and two-stage selection preserve useful evidence more effectively |
| Native-field N | Broad context plus 100 mm local views at field center and physical offsets | Whether retaining context without the same anchor helps alongside a local view |

The aggregation mechanisms were substantively different. Window attention chooses which image summaries matter to each finding. S6 also chooses spatial locations *inside* a window before combining windows, allowing the representation of, for example, meniscus and effusion to emphasize different regions. Localization can spend resolution near the anatomy, but it also creates a dependency on the location estimate and its coverage. N addressed that dependency with a broad context view plus local views. Its declared native reader sampled at 0.4 mm up to a 204.8 mm field, retaining aspect ratio and padding the context; “native field” did not mean unlimited coverage at arbitrary resolution.

Training details could alter these mechanisms even when model names matched. The r949 source used one physical study per batch with gradient accumulation over sixteen studies. That does not reproduce the batch-normalization behavior of a physical batch of sixteen. Its external OAI cohort contained **2,399 knees**, with only three supervised findings. Masked BCE divided by all twelve target cells, so a partially observed external row did not receive the same normalization as one with twelve observed targets. The external OA and synovitis definitions also differed from my earlier mapping. “Add OAI” was not a sufficiently precise intervention to explain or reproduce its result.

Its supervision had similarly coupled ancestry. Writing $G$ for one Gemini report table, $V$ for a public blended table, and $H$ for the JEV hybrid, the ordinary branch mixed $B=(G+V)/2$ and then $Q=(B+H)/2$. JEV itself used $H=0.95S+0.05E$, where $S$ selected existing public columns and $E$ combined related report reads. Thus, when those sources were present,

$$
Q=0.25G+0.25V+0.475S+0.025E.
$$

An “unstated” branch in $V$ was handled separately. The formula mattered because a newly named table was largely a rearrangement of existing label sources, not a new independent annotator. Source selection and some blending had used exposed expert cases. Retraining image models on held-out folds could protect image-level evaluation, but could not make those reused cases an independent test of the label design.

I chose to pursue coupled reproductions while keeping their uncertainty explicit. By the cutoff, R949F had completed its first fold and a small source-input replay; the remaining folds and its Public comparison were unfinished. S6F used an ImageNet-pretrained ConvNeXt-S and SKM-TEA V-Net-derived locations, and its port and data preparation had started. The public S6 route reported 0.950 for an all-data model and 0.948 for a folds-1–4 model, but included an r949 fallback for unreadable studies. Those scores did not establish the behavior of my standalone five-fold reproduction.

This was also a material allocation choice. The S6F plan estimated **250–310 GB** of cache, **10–14 hours** of porting work, and roughly **9 GPU-hours** for five folds, with a pessimistic training estimate of sixteen hours. A stride-one T4×2 read was estimated at **2.9–3.2 hours**. These were preparation estimates, not completed measurements. R949's one-checkpoint 94-window mirrored inference would become **940 backbone windows per study** for five folds. A promising public score could justify that investment, but it could not make the cache, complete OOF, or inference capacity free.

## Image-assisted labels had to earn their value through students

Reports can omit visible evidence. A multimodal reader can instead produce $q(R,I)$ from the report $R$ and a selected image summary $I$. Its apparent confidence was not the result I needed: the test was whether those target changes helped an image student rank studies better.

The pilot used **100 studies outside gold58**, each with twelve images: four slices from each of three planes in fluid-sensitive sequences. I compared report-plus-image answers with report-only answers. On **25 studies**, I repeated the same input and substituted another study's images. For effusion, mean absolute probability changes were **0.072** after adding images, **0.014** on repetition, and **0.094** with wrong-study images. Synovitis gave 0.038, 0.016, and 0.040; contusion gave 0.016, 0.003, and 0.043.

![Image-addition, repetition, and wrong-study controls for three findings]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part3-multimodal.png){: width="1920" height="1120" }
_Figure 3. The pilot measured response to image content and repetition, not clinical accuracy. Image addition used 100 studies; repeat and wrong-study controls used 25. Sparse image summaries affected different findings differently._

The stronger response for effusion made an effusion-only update plausible. The other findings created a competing possibility: smaller changes spread across the remaining targets might be more useful than the pilot's most conspicuous effect. I specified both student comparisons before expanding labeling. **mmE** changed only effusion; **mm12** used the image-assisted targets for all twelve findings. Confidence weights, supervision masks, and the gold rows remained unchanged, making this a target-content comparison rather than a change in which examples counted. The saved usage implied a pilot-plus-corpus estimate near **\$94.9**, not a reconciled invoice.

| Same small 2D student | Primary difference from definition labels | Recorded paired interval | Public |
|---|---:|---:|---:|
| Definition control | — | — | 0.911 |
| mmE, effusion only | +0.0003 | [−0.0029, +0.0036] | 0.912 |
| mm12, all twelve targets | +0.0063 | [+0.0031, +0.0093] | 0.914 |

The all-target version was the more useful result in this student. Its largest primary gains were in lateral meniscus, synovitis, and fracture, rather than effusion alone, while several other targets fell. Sensitivity of the reader's effusion answer had not identified the best scope of the training intervention. Nor had the wrong-image control established correctness: a fluent reader could be influenced by irrelevant images, and twelve slices could miss a focal lesion. The students tested the value of the resulting supervision, not the clinical accuracy of every changed label.

Stronger learners then tested whether that benefit depended on the visual model. mm12 improved primary by about **0.0064** in the 64-slice fusion student and **0.0037** in the 2.5D window student. More than half the fusion model's gain came from one fold, making the distribution of the improvement important alongside its pooled value. A separate 80-slice comparison had only a first-fold result at its earlier checkpoint and did not yet supply another complete five-fold comparison.

The **N** model was the more consequential pending test. Its definition-guided version had completed five folds and scored **0.942** on Public on October 14, against **0.929** for G25 with the same definition labels. That difference concerned the complete image recipes, not field of view alone. N-mm12 now had primary **0.8913 versus 0.8885**, with a saved difference of **+0.0027**, interval **[+0.0010, +0.0044]**, and positive differences in all five folds. The displayed scores and their difference are rounded separately. Its Public result remained unknown. Holding N's image recipe fixed would ask whether the new supervision transferred in a stronger learner, rather than assuming that the small student's result generalized.

## A teacher needed a control for what its targets changed

A visual teacher offers another way to connect studies whose reports use different wording. But replacing a report target with a report–teacher mixture changes both its content and its numerical distribution. Improvement after distillation does not identify which part helped.

The first teacher experiment kept the small 2D student fixed. Let $q$ be its definition-guided target, $t(X)$ a teacher prediction, and $\pi$ the target prevalence computed on the training partition. I compared

$$
q^{S_b}=0.5q+0.5t(X),
\qquad
q^R=0.5q+0.5\pi.
$$

**S-b** used teacher mixtures for all twelve targets. **R** pulled targets toward an image-independent prior with the same mixing coefficient. **S-a** applied the teacher mixture only to effusion, contusion, fracture, and Baker's cyst. That four-target choice was informed by gold58, so its subsequent expert comparison was exposed.

R was a useful control because it separated a particular kind of prior shrinkage from image-dependent teacher content. It was not an exact match for target entropy: equal coefficients do not make the two target distributions equally soft. The result was nevertheless informative. Relative to the definition control, S-b increased primary by **0.0078**, while R decreased it by **0.0092**. Directly comparing S-b with R gave **+0.0170**, with a recorded interval **[+0.0141, +0.0200]**, positive in all five folds and all twelve findings.

![Teacher mixtures, prior shrinkage, and their Public outcomes]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part3-teacher-controls.png){: width="1920" height="1216" }
_Figure 4. The same student under definition targets, all-target teacher mixtures, a prior mixture, and a four-target teacher mixture. Local intervals describe paired differences on the stored OOF predictions. The four-target mask used exposed gold58; the Public outcomes are transfer evidence rather than an independent validation of that mask's selection procedure._

Public added an important qualification. S-b scored **0.910**, R **0.905**, and S-a **0.914**, against the control's **0.911**. Teacher content beat this prior control on Public as well as primary, but the all-target teacher still failed to beat the original control's displayed score. The local winner was S-b; the Public winner was S-a. The reused expert panel had favored the prior mixture, which finished lowest on Public. Choosing any one channel as the general ranking authority would have missed another part of this experiment.

The practical consequence was to retain teacher content as a useful hypothesis without declaring all-target distillation universally better. An image-dependent teacher might recover information beyond the report, but its strength, target coverage, and interaction with the student mattered. This motivated a stronger available teacher mixture rather than a larger collection of unexamined variants of the same weak teacher.

## A stronger teacher changed the evidence I could claim

For nonnegative weights summing to one, BCE is linear in its target:

$$
\mathcal L\!\left(\sum_s\lambda_s q_s,p\right)
=\sum_s\lambda_s\mathcal L(q_s,p).
$$

A supervision mixture is therefore an explicit weighted compromise among the sources' learning instructions. This identity does not make separately trained students equivalent to one student trained on mixed targets. It does make clear what the stronger **T\*** recipe asked a student to learn:

$$
q^{T^*}=0.50\,q^{\mathrm{definition}}
+0.25\,p^{\mathrm{ensemble}}
+0.25\,p^{\mathrm{public\ S6}}.
$$

The public visual teacher could add distinctions missing from report interpretation. Its provenance created a different evaluation problem. For a student holding out fold $j$, consider a training example in fold $k$. A teacher excluding only $k$ may have learned from $j$. The student's training targets can then contain information from its validation labels even though it never trains directly on the validation images.

Protecting the student evaluation requires excluding $j$ from teacher training. The **B2** construction used in the earlier controlled experiment additionally excludes $k$, keeping the training example's teacher target itself out of sample. The available T\* ensemble and public predictions could not establish the required exclusion of the student's evaluation fold. T\* students and their descendants were therefore **final-only**: usable for test prediction, but excluded from clean-OOF ensemble selection. Training the student locally did not remove the information inherited through its teacher.

![Double-exclusion teachers and the final-only T-star construction]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part3-teachers.png){: width="1920" height="1120" }
_Figure 5. B2 excludes both the student's validation fold and the source example's fold. T\* uses stronger available predictions without that guarantee. This teacher-training issue is separate from exposure of expert cases during report-label design._

By October 14, the **G25 T\*** student had scored **0.945**, against **0.939** for its earlier ensemble-teacher version and **0.929** for definition-guided supervision. By the morning of October 15, a **G96 T\*** student also scored **0.945**, compared with **0.934** for its earlier no-OAI Raptor-label version. The G96 change also used definition-derived ranking pairs and label-aware batches, so it was a supervision-recipe comparison rather than an isolated target-mixture coefficient. Both students inherited overlapping teacher sources; two scores did not establish two independent discoveries of new visual information.

These results made stronger visual supervision worth pursuing, while its optimistic local values remained unsuitable for selecting clean ensembles. A comparison against a B2 teacher would change teacher content and training construction together, so it could not identify a gain caused by leakage alone. The useful distinction was operational and scientific: preserve clean image-held-out families for conditional ensemble analysis, and evaluate final-only students through declared external comparisons with their ancestry visible.

At this cutoff, I still had three unresolved decisions. Would N's favorable multimodal-label comparison transfer to Public? Would complete R949F and S6F reproductions improve the visual models enough to justify their preparation and inference cost? And would stronger teacher-derived students complement those models rather than simply reproduce shared rankings? The next allocations depended on those particular results. No report reader, reused expert panel, or Public point had resolved all three.

**Next:** [Part 4: Conflicting Evidence and Two Final Candidates]({{ site.baseurl }}/posts/RSNA-Knee-Part-4-The-Final-Two-Before-the-Deadline/)
