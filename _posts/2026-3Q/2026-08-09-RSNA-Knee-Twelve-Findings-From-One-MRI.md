---
title: "RSNA Knee, Part 0: One MRI Study, Twelve Findings"
date: 2026-08-09 09:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, rsna-knee, dicom, dinov2, knee-mri, medical-imaging, weak-supervision, roc-auc]
math: true
pin: false
published: true
image:
  path: /assets/img/posts/rsna-knee-series/cover-0.webp
  alt: "Methodology cover: study-level inputs aggregated into twelve finding scores."
  width: 1659
  height: 948
  fit: contain
  hide_caption: true
topic: rsna-knee
lang: en
translation_key: rsna-knee-competition-structure
series: rsna-knee
series_order: 0
interactive_labs: [rsna]
---

The [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) asks for twelve finding probabilities from one MRI study. The input is an entire examination, assembled from several image series; the output is a fixed-length vector. A useful model has to identify relevant evidence within that collection and combine it at the level of the examination.

The training data make supervision unusually important. There are 4,407 studies, but only 58 have the supplied expert annotations for the twelve findings. The remaining 4,349 rely on radiology reports written across nine languages. I need to convert those reports into training targets, although reports are unavailable at test time. The image model must ultimately learn the connection between visual evidence and the findings, rather than depend on the text used to supervise it.

## Prediction unit and output structure

A study contains several series acquired with different orientations and sequences. Each series contains a stack of slices. I can write the input and output schematically as

$$
\begin{aligned}
\mathcal X&=\{x_{v,k}:v=1,\ldots,V;\ k=1,\ldots,n_v\},\\
\hat{\mathbf p}(\mathcal X)&\in[0,1]^{12}.
\end{aligned}
$$

Here $v$ indexes a series and $k$ a slice within it. Both the series present and their slice counts can vary. The model must turn that variable collection into the same twelve outputs for every study.

The targets are abnormalities of the anterior cruciate ligament (ACL) and medial collateral ligament (MCL); medial and lateral meniscal abnormalities; osteoarthritis in the medial, lateral, and patellofemoral compartments; effusion; synovitis; a Baker's cyst; contusion; and fracture. Several can occur together. Twelve separate output positions are therefore needed, rather than one categorical prediction whose probabilities must sum to one.

The study-level label also determines what supervision is missing. A report may describe a meniscal tear without marking the slice that shows it. Assigning the study's positive label to every slice would incorrectly treat all of those images as positive examples. The architecture needs to combine image evidence before making the study-level prediction.

![From an MRI examination and its training report to twelve predictions]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part0-task.png){: width="1920" height="1040" }
_Figure 1. Images provide the test-time input. Reports supply training targets, attached to the examination rather than to individual slices._

## AUC and study ordering

The competition score is the unweighted mean of twelve ROC AUCs:

$$
S=\frac{1}{12}\sum_{t=1}^{12}\operatorname{AUC}(y_t,p_t).
$$

For a single finding, AUC compares positive studies with negative studies. If $\mathcal P$ and $\mathcal N$ denote those two sets, its pairwise form is

$$
\operatorname{AUC}
=\frac{1}{|\mathcal P|\,|\mathcal N|}
\sum_{i\in\mathcal P}\sum_{j\in\mathcal N}
\left[\mathbf 1(p_i>p_j)+\tfrac12\mathbf 1(p_i=p_j)\right].
$$

Each correctly ordered pair contributes one, a reversed pair contributes zero, and a tie contributes half. One badly placed study can affect many pairs. A decision threshold such as 0.5 is absent from the calculation: the task is to place studies in a useful order, not to count correct binary decisions at one threshold.

This has two practical consequences. First, a strictly increasing transformation preserves one predictor's AUC when ordering and ties remain unchanged. A high AUC therefore does not establish that an output of 0.8 is a clinically calibrated 80% probability. Second, the twelve findings receive equal weight regardless of prevalence. Common findings provide more positive training examples, but do not receive a larger coefficient in the final average. I need to inspect changes across all twelve targets, especially when a representation favors some anatomical structures over others.

Score scale becomes relevant again in an ensemble. Averaging model probabilities retains the distances between their scores; averaging ranks replaces those distances with relative positions. Transforming separate predictors before combining them can change the final order even when each predictor's own AUC stays fixed. The aggregation operator is consequently part of the prediction model, not a neutral formatting choice.

{% include interactive/article-lab.html id="rsna-ranking" %}

## Report omissions and image findings

A radiology report records selected conclusions about an examination. It can assert a finding, explicitly deny it, qualify it, or say nothing about it. Converting the report into twelve values requires preserving those distinctions.

Consider this invented fragment, used only to illustrate the extraction problem:

> The anterior cruciate ligament is intact. A tear involves the medial meniscus. A small joint effusion is present.

The first sentence supports absence of one finding, the second supports presence of another, and the third requires interpretation of severity against the target definition. The remaining findings are unmentioned. Their absence from the text does not determine their state in the images. Both a present finding and an absent finding can go unreported.

A rule extractor can normalize text, recognize anatomy, and handle negation. Its output is inspectable, but its coverage depends on vocabulary and sentence structure in each language. An LLM reader can interpret a wider range of wording. Neither directly observes an image finding omitted from the report, although other wording may support an inference. This separates extraction error—misreading information that is present—from information that was never written down.

![Why an unmentioned finding has more than one possible image label]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part0-evidence.png){: width="1920" height="1040" }
_Figure 2. Either image state can lead to no mention. Encoding that case as negative, unknown, or a soft target is a supervision decision; the schematic does not estimate reporting frequencies._

The resulting labels are **weak supervision**. An explicit negative can be encoded differently from an unknown entry; an uncertain statement can supply a soft target or receive a different training weight. These choices determine how the image model is encouraged to respond. A reader's confidence that it understood a sentence is also a different quantity from the probability that a finding exists in the image.

The 58 expert-annotated studies let me examine disagreements between extracted labels and the supplied findings. That comparison addresses the label table. To evaluate the image learner, I also need held-out predictions. OOF predictions exclude each study from its corresponding model's training, but scoring them against extracted labels still measures agreement with those labels. Better report interpretation and better image prediction are connected questions that require separate comparisons.

## Image representation and anatomical coverage

A model cannot directly inspect anatomy removed during preparation; any inference about it must rely on the remaining evidence. Series selection, physical scale, and slice sampling therefore deserve as much attention as the encoder.

My initial representation uses six sequence slots: sagittal, coronal, and axial fluid-sensitive views with fat suppression; a sagittal fluid-sensitive view without fat suppression; and coronal and sagittal T1 views. Sequence descriptions and header values help choose a series for each slot. A missing slot must be represented explicitly. The slots give comparable kinds of images comparable roles across examinations, rather than making the model infer every series' role from an unordered collection.

Within a series, physical position determines slice order. Filenames are identifiers and can give an anatomically meaningless order. DICOM orientation and position provide a geometric coordinate, provided the images belong to a consistent stack.

<details markdown="1">
<summary>How DICOM geometry orders a stack</summary>

The in-plane direction vectors $\mathbf r$ and $\mathbf c$ define a slice normal:

$$
\mathbf n=\mathbf r\times\mathbf c.
$$

For image position $\mathbf x_i$, the sorting coordinate is

$$
k_i=\mathbf x_i\cdot\mathbf n.
$$

Repeated positions, inconsistent orientations, localizers, or incomplete headers need separate treatment. An instance number can be a fallback, but cannot establish geometric consistency by itself.

</details>

Physical cropping introduces a separate choice. If a crop spans $L$ millimetres and is resized to $P$ pixels, its nominal output pitch is

$$
s_{\mathrm{eff}}=\frac{L}{P}\quad\text{millimetres per pixel}.
$$

A 336-pixel image covering a wide field of view depicts a small structure at a different scale from a 336-pixel image tightly cropped around the joint. A tighter crop gives the local structure more pixels but removes context. Increasing the output resolution retains more sampled detail at greater memory and computational cost. The effective pitch describes that representation; the acquisition's own resolution and contrast still limit the information available.

Slice sampling creates the same tension through the volume. With a fixed number of slices, broad sampling may include anatomy outside a narrow band, while its wider gaps may miss a small lesion. Dense sampling preserves neighboring relationships within a more restricted range. A larger encoder cannot compensate directly for an informative slice that was never selected.

![Physical ordering, crop size, and a fixed image grid]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part0-sampling.png){: width="1920" height="1040" }
_Figure 3. Geometry establishes spatial order and scale. Crop size and sampling density then determine which anatomical evidence reaches the model._

{% include interactive/article-lab.html id="rsna-geometry" %}

## Aggregation across slices and views

A pretrained encoder maps each selected image to a feature representation. The remaining model combines those representations across slices and views. This is where the difference between an image label and a study label becomes important: the model must discover which parts of the examination support each finding.

A simple average treats all selected images equally. Evidence confined to a few slices can be diluted by many uninformative images. A learned attention mechanism can assign different weights to the features, potentially using different weights for different findings. It still has to learn that relevance from the study-level supervision. Attention cannot guarantee localization merely because its weights are available to inspect.

The supervision and representation choices meet here. If an unmentioned finding is repeatedly treated as negative, the model may be penalized for recognizing visual evidence of it. If the relevant images are omitted, even an accurate positive target cannot teach their appearance. Changing a label table, crop, or aggregation method can therefore change the value of the other components. I need comparisons that make those changes explicit instead of attributing the whole result to the encoder's name.

## Inference paths and cost constraints

The nine-hour allowance on two T4 GPUs applies to the complete hidden-test prediction. It must cover image decoding, preparation, model computation, aggregation, and writing the output. Training cost and test-time cost are consequently different constraints.

A reusable cache can accelerate repeated training by storing a prepared representation. For one byte per pixel, its leading storage term is approximately

$$
N_{\mathrm{studies}}\,N_{\mathrm{slots}}\,N_{\mathrm{slices}}\,P^2.
$$

Doubling image width and height quadruples that term before masks and storage overhead. Increasing resolution affects preparation and storage as well as the model's GPU use.

I have also compared how many windows to read from each study. A window is one sampled group of images; several windows let the same learned model see different parts of the examination. For a probability-averaged read with $M$ checkpoints and $W$ windows, the output for one finding can be written as

$$
\bar p_t=\frac{1}{MW}\sum_{m=1}^{M}\sum_{w=1}^{W}p_{m,w,t}.
$$

An additional window can improve anatomical coverage or average away an unstable prediction. An additional independently trained model can change the representation or reduce some training variation. They spend the same inference allowance on different sources of error. Neither is equivalent to merely increasing the number of votes.

In the August 8 local sweep, one, four, and ten windows produced AUCs of 0.8218, 0.8397, and 0.8428. Four windows recovered most of the observed one-to-ten-window gain. Reading six more windows added 0.0031, while nominal model work increased from four to ten window evaluations. These are local results, not competition scores or measured wall times.

![The early local window sweep beside its nominal computational work]({{ site.baseurl }}/assets/img/posts/rsna-knee-series/part0-windows.png){: width="1920" height="1040" }
_Figure 4. Local AUC increased with additional windows, with diminishing gains in this recorded sweep. The second panel shows nominal work at a fixed model and input configuration; it is not measured runtime. This comparison did not separate better coverage from averaging effects._

Ten windows remain a useful option if the rest of the predictor fits comfortably. A tighter budget requires comparing the last windows with another checkpoint or another input representation. This curve alone cannot settle that choice. It does show why inference design belongs inside model selection: changing which images a trained model sees can change its predictions.

An ensemble's inference cost depends on these shared computations. Models reading the same prepared images can reuse decoding and preprocessing; models requiring another crop or slice schedule may need additional preparation. Counting checkpoints therefore cannot determine what fits. The comparison has to concern the whole study predictor: the information it reads, the twelve scores it produces, and the time required to compute them on the permitted hardware.
