---
title: "BioHub Cell Tracking Working Note 7: Where a Local Gain Has to Be Measured"
date: 2026-09-12 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, hand-labels, label-convention, pseudo-labels, logit-alignment, deployment-regime, leakage, oof, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-09-12-biohub-working-note-7/cover.png
  alt: "Title card for BioHub Working Note 7: where a local gain has to be measured"
---

<style>
.content .table-wrapper > table {
  table-layout: fixed;
  width: 100%;
  min-width: 36rem;
}
.content .table-wrapper > table th,
.content .table-wrapper > table td {
  white-space: normal;
  overflow-wrap: break-word;
  vertical-align: top;
}
</style>

# BioHub Cell Tracking Working Note 7: Where a Local Gain Has to Be Measured

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  - [Working Note 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
  - [Working Note 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
  - [Working Note 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)
  - [Working Note 6: The Universe We Were Selecting In]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-The-Universe-We-Were-Selecting-In/)
- Korean version: [BioHub Cell Tracking 작업 기록 7: 로컬 이득은 어디서 재야 하는가]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Where-a-Local-Gain-Has-to-Be-Measured-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 8: Choosing the Final Two Without the Leaderboard]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-Choosing-the-Final-Two-Without-the-Leaderboard/)

Related public notebooks:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

Note 6 ended on 2026-09-04 with what I expected was the one source of ranking signal left: new labels.
I had begun labeling division candidates on training movies by hand, and the first $69$ judgments moved the deployed-stack replay from $0.7535$ to $0.7547$.
That note promised the labels would be priced like the week's other verdicts: by the shipped runtime module, on the deployed-stack replay, with both embryos non-negative.

This note covers 2026-09-05 to 2026-09-12, and six submissions.
The labels were priced that way and passed, and the board rejected the first labeled kernel the same day.
A detector trained on our own tracks then cleared the embryo-out gates I wrote for it, two of them only after I changed them, and collapsed inside the kernel.
When the leak in that detector's evaluation was removed, its gain did not reproduce.

Over 2026-09-05 and 09-06 I put the selection rule back on local evidence.
A Public difference within $0.002$ became a tie, and the board kept one job: falsifying a large move.

Note 6 moved the measurement into the deployed code.
This week showed that the deployed code running different weights is still a different machine.
A local gain counts only where the model ships: in the composition, weights and logit regime the kernel actually runs.

The short version is:

```text
A local gain is a statement about the machine it was measured on.
Hand labels priced positive on both embryos and lost 0.005 on the board;
by my arithmetic, the local proxy gave false divisions almost no price.
A detector cleared its embryo-out gates, two only after I relaxed them,
and collapsed in the kernel, which aligns two fields whose whole-frame
mean logits differ by about ten. With the leak removed, its gain did not
reproduce. The board kept one job, falsifying large moves. From 09-10 a
detector change must also pass a kernel-regime check that can only veto.
```

The note follows that sequence:

| Sections | Question |
|---|---|
| 0 | What state did the week open in? |
| 1--2 | Why did hand labels that passed locally fail on the board? |
| 3 | What rule replaced reading the board? |
| 4 | Which lanes closed without a submission? |
| 5--6 | How did a detector clear its local gates and collapse on deployment? |
| 7 | What did a control pair on the board say, and what could it not say? |
| 8 | What happened when the leak was removed? |
| 9--10 | Which rule selected this period, and what did the period establish? |

---

## 0. Where the Week Opened

The pipeline is the one from Note 6.
A primary detector and a secondary, whose logit field is aligned to the primary's and blended in before peak extraction, feed a transformer edge scorer.
An integer linear program (ILP) selects the graph, deterministic post-stages follow, and a gradient-boosted division verifier is retrained inside the notebook from a shipped candidate table.
The kernel is the submitted Kaggle notebook; the hidden set is the test movies behind the board.

The $199$ training movies come from two embryos, one of $128$ movies and one of $71$, which I call the larger and the smaller embryo.
Embryo-out means a model trained on one embryo and scored on the other.
I call the main local instrument the deployed-stack replay: the shipped runtime code run over all 199 training movies with embryo-out models, scored by the official scorer, each embryo reported separately.
The project called it kernel-faithful.
The four example movies with public labels are copies of training movies, so scoring the deployed models on them is a hold-in check: it can catch damage but cannot select.

| item, entering 2026-09-05 | value |
|---|---:|
| deployed kernel | v83 |
| its deployed-stack replay score, 199 movies | $0.7535$ |
| its local division TP / FP | $8 / 19$ |
| its Public score | $0.944$ |

Two beliefs were in force.
One was that the division ranker was starved of labels, and that hand labels on training movies were the one lever left.
The other was Note 6's working expectation that division changes are not damped on the hidden set, while edge changes were believed to damp.
The written rule was also out of date.
Since 2026-08-28 it had said the board was the objective, while in practice every selection of the previous week had been made locally.
This note withdraws the second belief, finds that the first stopped paying, and rewrites the rule.

---

## 1. Hand Labels and a Legitimate Falsification

### 1.1 What the Labels Bought Locally

Each labeling item showed one candidate division from a training movie, sampled across the verifier's score bands, and I answered yes, no or skip.
Fifty blind gold rows with known ground truth measured my agreement.
The first $250$ items gave $66$ yes, $86$ no and $98$ skip, with $14$ of $16$ gold rows correct, and added $58$ positives to the table, for $232$ in total.

| table (deployed-stack replay, 199 movies) | score | division TP / FP |
|---|---:|---:|
| deployed table, threshold $0.75$ | $0.7535$ | $8 / 19$ |
| $+58$ hand positives, threshold $0.75$ | $0.7573$ | $19 / 71$ |
| $+58$ hand positives, threshold $0.65$ | $0.7583$ | $24 / 96$ |

Both embryos rose in both arms: $+0.0038$ at the deployed operating point and $+0.0048$ at $0.65$.
Kernel v85 shipped the $0.65$ arm.

One column in that table should have stopped me.
False positives went from $19$ to $96$, fivefold, and the local score did not care.

### 1.2 The Result

The reading of v85 was written down before its score existed: $0.947$ or more would confirm the label lever, $0.944$ to $0.946$ would be neutral, and anything below $0.944$ would mean the hand rows or the lower threshold hurt on the hidden set.
It returned $0.939$, a move of $-0.005$.
That is large enough for the board to resolve, and it landed in the harmful band: the board doing its proper job, falsifying a large move the local evidence had approved.

The four example movies could not have warned me.
They contain only three annotated divisions, and v85, v86 and v87 each scored no true division among them.
On those four movies v85's edge counts were exactly v83's, which points the loss at division precision on the hidden set.

### 1.3 Why the Local Proxy Mispriced It

The official score is a node-count-adjusted edge Jaccard plus the division Jaccard at weight $0.1$, and the division term is micro-averaged over events:

$$
J_{\mathrm{div}}=\frac{TP}{TP+FP+FN}.
$$

A batch of new picks raises $J_{\mathrm{div}}$ only if its precision exceeds $J_{\mathrm{div}}/(1+J_{\mathrm{div}})$.
Locally the division term is dominated by misses: $J_{\mathrm{div}}$ sat between $0.06$ and $0.10$, with $127$ of $151$ events missed.
Near zero almost any pick pays, so $8 \to 24$ true positives outweighed $19 \to 96$ false ones.
The Note 6 probe put the hidden division Jaccard near $0.32$, where the same trade loses.
My arithmetic that day took the hidden set from $32$ true, $20$ false and $48$ missed divisions to $40$, $100$ and $40$: about $0.32 \to 0.22$ in $J_{\mathrm{div}}$, about $-0.01$ on the score.
A second channel sat beside it, since every added division also rewires edges.
That is arithmetic, not a measurement of the hidden set.

![Break-even precision J/(1+J) against the division Jaccard, with v85's added picks at precision 0.172]({{ site.baseurl }}/assets/img/posts/2026-09-12-biohub-working-note-7/fig-01-division-breakeven.png)
_Figure 1. The precision new division picks need rises with the current division Jaccard. v85's added picks, at $0.172$ by arithmetic, cleared the local break-even and fell short of the one implied by a hidden level near $0.32$._

Against my own labels, picks scored $0.90$ to $0.95$ were right $76\%$ of the time; the bands from $0.65$ to $0.85$, where most of v85's additions sat, only $13$ to $29\%$.

Two consequences followed.
I would now choose division operating points on the precision frontier, with false positives near v83's level and the most true positives at that level, not at the local score maximum.
And I withdrew Note 6's expectation that division changes are not damped on the hidden set: v85 was the first division change whose local gain came back negative.

---

## 2. The Labeler Was Not Scoring What the Scorer Scores

### 2.1 A Tie, and a Model That Predicted More

v85's reading rule had named its own follow-up, the same table at $0.75$, and that is not what was submitted next.
A second batch of $500$ items ($97$ yes, $207$ no, $196$ skip; gold precision $1.00$, recall $0.82$) gave a table with $253$ positives.
v86 shipped it at the precision frontier, threshold $0.90$, where it scored $0.7591$ locally with $18$ true and $25$ false divisions.
Because v86 changed both the table and the threshold, its result cannot separate them.

A pick-count model, which weighted each added pick by my labels' precision in its score band, said v86 should beat v83: v83 added $2{,}242$ picks at an expected precision of $0.41$, and v86 added $2{,}324$ at $0.825$.
The board returned $0.943$, the same score as v83 at the board's resolution.
Most of v85's loss was gone, and nothing was gained over the kernel without hand labels.

### 2.2 Measuring the Convention

The gold rows held a hint: I had answered "no" to roughly $15\%$ of the annotated divisions.
The next day I measured how the ground truth places all $151$ annotated divisions.
At the annotated parent frame, $75\%$ show exactly one detection within $7\,\mu\mathrm{m}$ of the parent, $15\%$ show two and $8\%$ none.
The annotation puts the division edge where the parent is still one nucleus, and the daughters appear about $10\,\mu\mathrm{m}$ apart one frame later.

Then I looked at the seven gold divisions I had rejected.
In every one the annotated daughters sat on our candidate pair: the candidates were right and my judgment was wrong.
The daughters were $6.5$ to $12.4\,\mu\mathrm{m}$ apart one frame later, and gold divisions I had accepted had the same geometry.
The annotations' median daughter separation was $10\,\mu\mathrm{m}$; the median of my own positives was $8.7\,\mu\mathrm{m}$.
The analysis read this as a far daughter mistaken for a neighbor that had moved in.
That was its hypothesis, not my account: when I went back to the items, my "no" answers had not been about distance.

I re-judged the rejected items whose daughters were at least $9\,\mu\mathrm{m}$ apart, $99$ from the first batch and $109$ from the second, under the annotation's convention.
Sixty changed from no to skip, and none became yes.
Those sixty were cases where a cell appears from behind, a division into depth, and as "no" labels they had been teaching the verifier to reject candidates the annotation may count.

This is a correction of convention, not a case for more labels.
Labels of the same kind had already stopped paying: the union of both batches priced below the second batch alone.
Whether the mismatch explains v85 and v86 on the board is a hypothesis I did not test.
What transfers is simpler: a human label has to be checked against the definition the scorer uses before it is priced.

### 2.3 The One Clean Contrast

v87 kept v86's table and threshold and changed only the verifier's runtime features.
Three features were added, each encoding a cue I used when labeling: the distance to the nearest detection at the candidate daughter's position one frame early, and the brightness there at that frame and the frame before.
Locally it scored $0.7609$ with $23$ true and $40$ false divisions.
On the board it returned $0.946$, $+0.003$ over v86, at the edge of what the board resolves.
Against v83, the kernel without hand labels, it is the same score at the board's resolution.

The attribution has to stay narrow.
The one attributable board step, v86 to v87, belongs to the runtime features; the kernels that changed only the labels were v85 at $-0.005$ and v86 at a tie.
Locally, the features did nothing without the labels: on the old table at $0.90$ they scored $0.7489$ with no true divisions.

---

## 3. The Rule, Rewritten

Before v86's score, I had set a condition for the night: submit v87 if v86 reached $0.945$ or more, and not if it was $0.944$ or less.
v86 read $0.943$, and v87 was submitted anyway, reframed as a controlled probe of v86.
The design argument was sound, since v86 to v87 was the only single-axis contrast in the program.
The procedure was not: a condition written down in advance was set aside after its input was known.

The next day I looked at the condition itself and saw the larger problem.
The Public split is $29\%$ of the test set.
Reading v86's $-0.001$ as a verdict, and hanging a submission on "$0.945$ or more", is p-hacking on a small sample.
The gate I had set was the mistake.
Both things are true, and the last line of the rule I wrote next exists because of that night.
Over 2026-09-05 and 2026-09-06 I rewrote the project's selection rule:

```text
select on leakage-safe local evidence: embryo-out, kernel-faithful,
  both embryos non-negative, division operating points on the precision frontier
a Public difference within 0.002 is a tie
the board is cited only to falsify a large move (about 0.003 or more)
submit only hypotheses written down before the result exists
never rewrite a rule after a score
```

This replaced the rule of 2026-08-28, which had made the board the objective after a stall.
That rule reacted to too little contact with the board; this one reacted to reading it too closely.

There were no submissions between v87 and 2026-09-10; every verdict in those four days was reached locally.

---

## 4. Three Lanes Closed Without a Submission

On 2026-09-06 I wrote a plan with four lanes, each with a gate written before its first number: graph-stage constants, new rival-parent features for the division verifier, a division-aware fine-tune of the edge head, and a detector trained on pseudo-labels.
Three lanes closed within two days.

**The edge-head fine-tune died on its pilot the day it was planned.**
The pilot asked whether the deployed association head already separates a daughter captured by a neighboring track from a neighbor that moved in.
The AUC of the head's score margin between the two was $0.610$ and $0.683$ for the two deployed heads, against a gate of $0.70$.
A head trained with more than ten thousand real zebrafish divisions reached $0.643$, so the fine-tune's own supervision carried no more signal.

**Rival-parent features did not move the frontier.**
A three-feature logistic separated captured daughters with an embryo-out AUC of $0.80$ and passed the rule written for it.
Inside the runtime, at $0.90$, the verifier scored $0.7599$ with $21$ true and $39$ false divisions, against $0.7605$ with $23$ and $47$ for the shipped features on the same labels.
It traded two true positives for eight fewer false ones, and the rule for shipping it (eight more true positives at no more than $45$ false) missed widely.
The conclusion was that ranking among the captured-daughter candidates the graph already produces was saturated with respect to geometry and appearance; where the missed divisions sit had not been measured.

**Graph constants split the embryos.**
The lane began with a bug: environment overrides in the replay had been silently ignored, so earlier sweeps had measured nothing.
After the fix, none of fourteen arms on twelve diagnostic movies passed the both-embryo rule.
One was large: lowering the edge threshold from $0.50$ to $0.35$ gave $+0.0646$ on the smaller embryo ($8$ of $9$ movies up) and $-0.0117$ on three movies of the larger one, with the ratio of predicted nodes to the supplied cell-count estimate rising from $1.056$ to $1.146$.
The threshold drops candidate edges before the ILP, which then deletes disconnected nodes: cells on the smaller embryo, over-detections on the larger.
A bracket narrowed the value to $0.45$, which on all 199 movies gave $+0.0081$ on the smaller embryo and $+0.0008$ on the larger.
The rule asked for $+0.003$ on both embryos, and I closed the lane.

---

## 5. A Detector Cleared Locally

### 5.1 The Mechanism

The detector's trainer marks only annotated nuclei as positive, and every other voxel as negative.
Annotated nuclei are about $2.8\%$ of the real ones, so roughly $97\%$ of real nuclei were being trained as background.
Our own embryo-out tracks, added to the ground truth, offered $36\times$ more supervision from the same domain: $4.76$ million nodes against $133$ thousand.
I call the model whose tracks became labels the teacher, and the detector trained on them the student.
Note 6's augmentation had changed the input to reach dim cells.
This changed the definition of what counts as a cell.
The student would over-detect, and I expected the ILP to prune the extra peaks.

### 5.2 The Gates

| student, embryo-out, official scorer | tail movies | typical movies |
|---|---:|---:|
| epoch 10, as primary, $15$ larger-embryo movies | $+0.1280$ | $+0.0307$ |
| epoch 50, as primary, same movies | $+0.1358$ | $+0.0774$ |
| reciprocal student, as primary, $9$ smaller-embryo movies | $+0.1124$ | $+0.1033$ |

Tail movies are those where the deployed pipeline scored worst; typical movies sit near the median.
The reciprocal student was trained the other way round.
The pass rule for the first row, written in advance, asked for at least $-0.003$ on typical movies and $+0.03$ on the tail.
Its $+0.0307$ is the $+0.031$ headline that Section 8 returns to.
It was my first detector gain on typical movies, where Note 6's augmented detector had lost.

Two clauses written in advance did not hold, and I changed both rather than obeying them.
The reciprocal rule capped the pooled node-to-estimate ratio at the base plus $0.05$, and it rose from $1.027$ to $1.201$.
Before the 199-movie numbers existed, I replaced the cap with $+0.003$ on both embryos, a score that already includes the count penalty, plus a cap of $1.5$ on each embryo's median per-movie node ratio.
Later a composition rule allowed the six smaller-embryo movies that had lost most as primary to lose at most $0.02$; the chosen composition lost $0.0257$, and I accepted it after that number was known.
Both changes had reasons, both are recorded, and both moved toward shipping.

### 5.3 From 199 Movies to a Kernel

As primary on all 199 movies, the student gained $+0.0634$ on the larger embryo ($7$ of $128$ movies negative) and $+0.0357$ on the smaller ($23$ of $71$ negative).
One slice was written down before any submission.
Smaller-embryo movies with a base score of at least $0.85$, which I call high-base, lost $0.0043$, with $14$ of $27$ negative; the larger embryo's high-base movies gained $0.031$.
My estimate of the hidden set's edge level put the relevant expectation between those two numbers, not at the means.

The composition carried forward used the student as the *secondary* detector, beside the replay's embryo-out primary.
On all 199 movies it moved the official score by $+0.042302511$, with both embryos positive.
Forty-three movies fell, including $17$ of $28$ high-base movies of the smaller embryo.

Kernel v88 swapped the secondary weights for an all-train student, selected at about epoch $52$ by a trainer proxy on $40$ training movies.
It also carried two small validity repairs: a guard against an added division giving a cell a third child, and integer output coordinates kept inside the volume.
The all-train student that v88 loaded was never scored anywhere; every 199-movie number came from the embryo-out pair.
The reading written before the score called $0.943$ or less materially adverse and $0.949$ or more a positive transfer.

---

## 6. Why the Kernel Collapsed

v88 returned $0.924$, $-0.022$ against v87, far below the adverse line.
The board did what it is for: it caught a collapse that the kernel's own output already showed but no pre-submission check had compared.

### 6.1 One Alignment Formula

The kernel mixes the two detectors' logit fields before peak extraction:

$$
A=\left(S-\mu_S\right)\operatorname{clip}\!\left(\frac{\sigma_P}{\sigma_S},\,0.5,\,2\right)+\mu_P,
\qquad
B=0.525\,P+0.475\,A,
$$

where $P$ is the primary's field, $S$ the secondary's, and $\mu$, $\sigma$ are taken over the whole frame.
The formula assumes both detectors were trained the same way.
On a bright example movie, the kernel's primary, an older all-train model, has a whole-frame mean logit of $-14.7$.
The student, trained on dense pseudo-labels, fires on $10$ to $16\%$ of voxels against $3\%$ for the primary, and its mean on the same frame is $-5.0$.
Alignment shifts every student logit down by about ten.

It was then measured on one frame, through the deployed test-time augmentation and temporal window:

| detector field (one frame, deployed path) | peaks |
|---|---:|
| primary alone | $281$ |
| primary with the old secondary (v87) | $253$ |
| student alone | $644$ |
| student after alignment, alone | $0$ |
| primary with the student (v88) | $25$ |

![Peaks on one frame: primary 281, with the old secondary 253, student 644, aligned student 0, primary with student 25]({{ site.baseurl }}/assets/img/posts/2026-09-12-biohub-working-note-7/fig-02-alignment-peaks.png)
_Figure 2. Peaks on one frame of a bright example movie, through the deployed path. Aligned by whole-frame statistics, the new detector left $25$ peaks where the primary alone had $281$. This is a mechanism on one frame, not a recall measurement._

At the primary's $281$ peak locations, $92.88\%$ fell below the detection threshold in the blend, against $17.08\%$ with the old secondary.
These are primary-peak locations, not ground-truth cells, so this is a mechanism and not a recall measurement.

### 6.2 Why No Local Gate Saw It

The deployed-stack replay runs the deployed code with embryo-out weights.
Its primaries sit at mean logits of about $-6$ to $-9$.
In that regime the alignment is harmless, and the student adds recall.
The composition that shipped, the old all-train primary with the student, was never measured locally.

On one example movie with a supplied estimate of $32{,}795$ cells, the kernel produced $18{,}423$ nodes, a ratio of $0.56$, while the replay produced $47{,}740$, a ratio of $1.46$.
Every count-sensitive lever priced on the replay had been priced on a field that over-detects a movie the kernel under-detects.

Rerunning the kernel's own snapshot, command and environment reproduced it to within a few nodes: $25{,}786$ against $25{,}786$ for v87, $12{,}981$ against $12{,}975$ for v88.
The student alone over-detects, with $61{,}553$ nodes on the same movie.
Only the blend collapses.
The reproduction ran without the validity repairs, so they, the solver and the hardware are ruled out as causes.

Note 6 found that the selection universe differed from the deployed one at the level of code.
This was the same failure one level down: the replay ran the shipped code with weights the kernel never loads.
The name kernel-faithful had described the code, not the weights.

### 6.3 What the Kernel's Own Output Showed

Scored against the four example movies, the two submitted files differ plainly:

| four example movies, submitted outputs | v87 | v88 |
|---|---:|---:|
| official score | $0.889473$ | $0.857810$ |
| final predicted nodes | $120{,}450$ | $77{,}002$ |
| labeled-node recall | $0.994528$ | $0.957592$ |
| edge TP / FP / FN | $2027 / 156 / 100$ | $1939 / 158 / 188$ |

The pre-submission validation checked identifiers, degrees and in-volume coordinates, not score.
A $77{,}002$-node file, missing more than a third of its parent's nodes, passed every structural check.
A score check against the parent had been listed as a release step and was not run.

On one movie the collapse even read as a win: v88 halved the nodes and the adjusted edge score rose from $0.905$ to $0.992$.
The score sees only labeled cells, and its count term scales a movie's edge Jaccard by $1-0.1r$, with $r$ the relative excess of predicted nodes over the supplied estimate of all cells.
That movie kept every labeled node, so its dropped cells appeared only as a negative $r$, which raises the score.

### 6.4 What Changed

I adopted three rules on 2026-09-10.
First, every release is scored on the four example movies against the kernel it modifies before a submission is requested.
There is no fixed veto line; an unexpected loss of recall or edges has to be explained.
Second, a change to detection or to detector composition is measured on two paths.
The embryo-out replay asks whether it helps on an unseen embryo.
A kernel-regime panel runs the deployed all-train models through the kernel's own code on training movies, starting with the four example movies, and asks whether the shipped composition behaves.
Those models were trained on every panel movie, so the panel is hold-in: it cannot select, but it can veto, and nothing ships while the two paths disagree.
Third, run records name the weights measured beside the weights that ship.

---

## 7. A Control Pair on the Board

A second package, chosen before v88's score was known, averaged the primary's feature maps across the test-time views before the edge scorer reads them; I call it E.
On all 199 movies E was $+0.004523282$ over its control, with both embryos positive.
It also had harms: $21$ of the $28$ high-base movies of the smaller embryo fell, and division false positives rose from $37$ to $41$.

v89 combined E with the same two validity repairs v88 carried.
It returned $0.943$, $-0.003$ against v87, while its official score on the four example movies moved $+0.0028158874$, the opposite sign.

Because v89 changed three things at once, a control followed.
It was specified after v89's score, so it is a matched control, not an independent replication.
v90 was v89 with E switched off.
Its official scores on the four example movies were identical to v87's, and it returned $0.946$, the same score as v87.

The pair supports a modest reading: the two repairs cost nothing the board could see, and the E package most likely transferred negatively.
It cannot show that the repairs are free, because a tie at three decimals is censored.
Nor can it show that E alone, rather than its combination with the rest, is harmful, because $-0.003$ sits on the board's resolution floor.

---

## 8. The Leak, Removed

### 8.1 The Control

The pseudo-labels had carried a caveat from the first day.
The tracks on the student's training embryo came from a model trained on the other embryo, the one the student was then evaluated on.
Part of the $+0.031$ could be that embryo's annotations distilled back through the teacher, not denser supervision.

A first written version would have taken the teacher from the other fold, recreating exactly that path.
The control that ran trained its teacher only on the embryo the student itself was trained on.
A pseudo student and a ground-truth-only model were trained with the same seed, windows and fixed $60$-epoch endpoint.
Each ran standalone, doing its own detection and association with the secondary and verifier off, on the four example movies in both directions.

| clean control, standalone, four movies | ground truth only | pseudo student | difference |
|---|---:|---:|---:|
| all four | $0.7459$ | $0.7253$ | $-0.0206$ |
| larger-embryo movies | $0.7413$ | $0.7215$ | $-0.0198$ |
| smaller-embryo movies | $0.8474$ | $0.8112$ | $-0.0362$ |

The mechanism is the one that closed Note 6's augmentation program.
Labeled-node recall rose from $0.936$ to $0.971$.
Final nodes rose by $24{,}712$, to $171{,}083$.
Edges gained $83$ true positives and $122$ false ones.
The count adjustment alone contributed $-0.0175$, and true, false and missed divisions were unchanged at $0$, $8$ and $3$.
Recall was bought with nodes that crossed the count boundary.

This forces a retraction: on 2026-09-06 I wrote that the ILP prunes the student's inflated peaks, and in the clean control it did not.

This is a standalone composition on four movies, not a rerun of the fifteen-movie panel behind $+0.031$, and not a verdict on pseudo-supervision as a family.
The rule written on 2026-09-06 had named a larger panel; I report the smaller control at its size.
It is still the only measurement with the leak removed, and it is negative in both embryos.

### 8.2 The Repair Path

The obvious repair for v88 was a regime-independent alignment.
It was measured first in the kernel regime, on the four example movies with all-train models.

| composition (kernel regime, four movies) | official score | difference |
|---|---:|---:|
| C: deployed detection and association | $0.8898631853$ | — |
| A: deployed detection, student association | $0.8905639836$ | A−C $+0.0007007983$ |
| Q: A plus the student in detection, quantile alignment | $0.8818274412$ | Q−A $-0.0087365423$ |
| P: A plus the student in detection, probability blend | $0.8736703501$ | P−A $-0.0168936334$ |

Q aligns on the median and a high quantile instead of the mean and standard deviation.
Q and P were both positive on the smaller embryo, negative on the larger, and both lost recall.
The same association-only change with embryo-out models on the same movies gave $-0.003265956$, negative on both embryos.
The two paths disagreed in sign, so under the rule adopted on 2026-09-10 the association-only change did not ship.

The claim that only the alignment formula needed to change was negative on its first measurement.

---

## 9. The Rule We Selected By

The rule in force from 2026-09-06 was local-first: leakage-safe, embryo-out evidence on the deployed-stack replay, both embryos non-negative, division operating points on the precision frontier.
The board was a $29\%$ sample: within $0.002$ a tie, and only moves of about $0.003$ or more could falsify a hypothesis written down in advance.

The board kept to those roles: it falsified v85 at $-0.005$, read v86 and v90 as ties, caught v88's collapse at $-0.022$, and chose nothing.

What broke the rule this time was not the board.
The detector's local numbers came from the wrong machine: shipped code running weights the kernel never loads, and never the composition that shipped.
The rule also bent where I bent it.
v87 went in after the condition written for it had failed.
Two clauses of the detector's gates were changed on the way to v88, the second after its number was known, which is what the rule's last line forbids.
And v88 shipped on evidence that was not yet leakage-safe: the control on its headline returned two days later.
The repair moved part of the measurement inside the kernel: a kernel-regime panel that can veto, a score check against the parent before any release, and nothing ships while the two paths disagree.

| | this period |
|---|---|
| rule in force | local-first; Public ties within $0.002$; board cited only for large falsifying moves |
| where it was measured | deployed-stack replay (shipped code, embryo-out weights); from 2026-09-10 also the kernel regime |
| what the board was used for | falsifying v85, detecting v88's collapse, one matched control pair (v89, v90) |
| what broke, what held | broke: a gain measured on weights the kernel never loads; v87 past its failed condition; two detector clauses relaxed; v88 shipped before its leakage control. held: v85 and v88 read as falsifications, v86 and v90 as ties; no selection made on the board |

---

## 10. What the Period Established

### Established

1. A hand-label verifier priced at $+0.0048$ on the deployed-stack replay, both embryos up, returned $-0.005$ on the board, with edge counts on the four example movies unchanged.
2. Locally the division term is dominated by misses ($J_{\mathrm{div}}$ $0.06$ to $0.10$), so the local proxy gives false divisions almost no price.
3. $75\%$ of annotated divisions show one nucleus at the parent frame; all seven gold divisions I rejected sat on our candidate pair; a re-review under the annotation's convention moved $60$ labels from no to skip.
4. The single-axis step v86 to v87 is credited to the runtime features; the label-only kernels were v85 ($-0.005$) and v86 (a tie).
5. Three of four planned lanes closed locally within two days, without a submission.
6. With the pseudo-label student as secondary, the deployed-stack replay gained $+0.042302511$ on all 199 movies. The kernel carrying the all-train student scored $0.924$ on the board and $0.889473 \to 0.857810$ on the four example movies, and passed structural validation with more than a third of its parent's nodes gone. On one measured frame, whole-frame logit alignment left the blend $25$ peaks against $281$ for the primary alone.
7. With E off, v90 tied v87; with E on, v89 was $0.003$ lower, although E was $+0.004523282$ locally with both embryos positive.
8. With the teacher confined to the embryo the student was trained on, the pseudo student lost $0.0206$ standalone on four movies, both embryos negative, through the node count; quantile and probability-space alignment were both negative in the kernel regime on the same four movies.

### Supported but Unconfirmed

1. That v85's loss sits in division precision on the hidden set, with rewired edges as a second channel; this is arithmetic, not a measurement.
2. That the label-convention mismatch explains v85's and v86's board results.
3. That E itself, rather than its combination with the repairs, caused v89's lower score.
4. That most of the original $+0.031$ was cross-embryo distillation; the clean control is smaller and standalone.

### Open Questions

1. Where do the $151$ annotated divisions fall in the current path: no correct candidate, a correct candidate outranked, or a correct candidate scored below $0.90$?
2. Can any composition deliver the pseudo detector's recall in the kernel regime without paying for it in node count?
3. For changes that never touch the detection field, how much of an embryo-out gain survives in the kernel regime?

---

## Closing

This period had two large board moves, and I read both as falsifications, not as noise.
The first showed that the local proxy was blind to the cost of false divisions.
The second showed that, for any change that mixes detector fields, the local replay was not the kernel.
The ties, v86 and v90, stayed ties.

The question I now ask of a local gain has three parts.
Is it free of leakage?
Is it measured in the code that ships?
Is it measured on the weights and composition the kernel actually loads?
Note 4 taught the first, Note 6 the second, and this week the third.
Each time, the criterion that failed was clean by the standards I held when I built it.
The detector failed the first question as well, on a leak I had flagged before it shipped.

The local evidence at the end of the week is narrow.
The pseudo detector has no path into the kernel with a positive measurement behind it: the clean control lost on both embryos, and both repaired alignments lost in the kernel regime.
The hand labels bought local signal and a corrected convention, but no board gain of their own.
The division channel ends the week with its ranking lanes closed locally and no measured account yet of where the missed divisions sit.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
- [Part 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)
- [Part 6: The Universe We Were Selecting In]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-The-Universe-We-Were-Selecting-In/)
- **Part 7: Where a Local Gain Has to Be Measured**
