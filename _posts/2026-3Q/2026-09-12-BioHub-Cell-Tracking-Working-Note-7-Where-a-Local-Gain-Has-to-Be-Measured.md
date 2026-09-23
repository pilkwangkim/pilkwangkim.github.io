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

> **About this series.** These notes follow one Kaggle competition, BioHub Cell Tracking During Development:
> rebuilding cell lineages from 3D time-lapse microscopy of zebrafish embryos. The training data is 199 movies
> from two embryos; the leaderboard shows 29% of a hidden test of unseen embryos, and the final ranking uses the
> rest. The series asks one question — how to choose models by an internal criterion when the leaderboard cannot
> be trusted to choose — and each note adds what one period taught about it.
{: .prompt-info }

Note 6 ended with a repair and a question.
The repair moved local measurement into the deployed universe: a candidate is scored by the code that ships, over all $199$ training movies, with models that never saw the embryo being scored.
The question was whether a gain measured there survives on the hidden test's unseen embryos.
The first candidate to answer it was a division verifier retrained with labels I had begun making by hand; the first $69$ judgments had moved the deployed-stack replay from $0.7535$ to $0.7547$.

This note covers 2026-09-05 to 2026-09-12 and six submissions, and asks whether "measured in the deployed universe" is enough.
Twice the local evidence and the board disagreed by far more than the board's rounding, and each time the disagreement located a gap in the local instrument: once in what the local score can see, once in what the local replay actually ran.

Every submission had the same job: the board is the only view of unseen embryos, so a change that advanced locally was shipped once as a transfer check, its reading written before the score existed.
A reading within $0.002$ is a tie and leaves the local verdict standing; a large move against the expectation means the local instrument is missing something worth finding.
Over 09-05 and 09-06 that became the written selection rule (Section 3).

The short version is:

```text
A local gain is a statement about the machine it was measured on.
Hand labels priced +0.0048 locally and read -0.005 on the board. Locally,
misses dominate the division term, so false divisions cost almost nothing;
separately, my labels did not follow the annotation's convention.
A detector trained on our own tracks passed its local gates, two of them as
rewritten, and read 0.924: the replay had never run the shipped weights, and
with its leak removed its gain did not reproduce. From 09-10, changes are also
measured where they ship.
```

The note follows that sequence:

| Sections | Question |
|---|---|
| 0 | Where did the week start? |
| 1--2 | What did the board say about hand labels, and why? |
| 3 | What did the one single-axis contrast show? |
| 4 | What was answered without a submission? |
| 5--6 | How did a detector pass locally and collapse in the kernel? |
| 7--8 | What did the controls and the repair say? |
| 9--10 | Decisions, the criterion, and findings |

---

## 0. Where the Week Started

The pipeline is Note 6's.
A detector scores every voxel of each 3D frame, and the peaks become candidate nuclei; a secondary detector's field is aligned to the primary's and blended in first.
A transformer scores links between frames, an integer linear program (ILP) selects a consistent graph, and a gradient-boosted division verifier, retrained inside the notebook from a shipped table of labeled candidates, decides which candidate divisions to add.
The kernel is the submitted Kaggle notebook; the hidden set is the test movies behind the board.

The $199$ training movies come from two embryos, of $128$ and $71$ movies, which I call the larger and the smaller embryo; embryo-out means trained on one and scored on the other.
The main local instrument is the deployed-stack replay: the shipped runtime code run over all 199 movies with embryo-out models, scored by the official scorer, each embryo reported separately.
The project called it kernel-faithful.
The four example movies with public labels are copies of training movies, so scoring the deployed models on them is hold-in: it can catch damage but cannot select.

The deployed kernel entering 2026-09-05 was v83: $0.7535$ on the deployed-stack replay, with $8$ true and $19$ false divisions locally, and $0.944$ on the board.

Two beliefs were in force: that the division ranker was starved of labels, so hand labels were the lever left, and Note 6's working expectation that division changes are not damped on the hidden set.
The written rule of 2026-08-28, a response to a long stall, still named the board as the objective, although every selection of the previous week had been made locally.

---

## 1. The First Labeled Verifier Meets the Board

### 1.1 What the Labels Bought Locally

Each labeling item showed one candidate division from a training movie, and I answered yes, no or skip; blind gold rows (items whose answer the annotation already fixes, mixed in unmarked) measured my agreement.
The first $250$ items ($14$ of $16$ gold rows correct) added $58$ positives, $232$ in total.

| table (deployed-stack replay, 199 movies) | score | division TP / FP |
|---|---:|---:|
| deployed table, threshold $0.75$ | $0.7535$ | $8 / 19$ |
| $+58$ hand positives, threshold $0.75$ | $0.7573$ | $19 / 71$ |
| $+58$ hand positives, threshold $0.65$ | $0.7583$ | $24 / 96$ |

Both embryos rose in both arms (overall $+0.0038$ and $+0.0048$), and kernel v85 shipped the higher, $0.65$ arm.
False positives went from $19$ to $96$, and the local score barely registered it; Section 1.3 is why.

### 1.2 The Transfer Check

v85's reading was written before its score: $0.947$ or more would count as support for the label lever, $0.944$ to $0.946$ would be neutral, and below $0.944$ would mean the hand rows or the lower threshold hurt.
It returned $0.939$, a move of $-0.005$, well outside the board's rounding.
The check did what it was for: a change approved locally on both embryos was rejected on unseen embryos, by a margin that made the disagreement real, and the question became what the local instrument was missing.
On the four example movies, which hold only three annotated divisions, v85's edge counts were exactly v83's, which pointed the loss at division precision on the hidden set.

### 1.3 Why the Local Score Could Not See the Cost

The official score is a node-count-adjusted edge Jaccard plus the division Jaccard at weight $0.1$, the division term counted over all events together (TP matched divisions, FP unmatched predictions, FN missed annotations):

$$
J_{\mathrm{div}}=\frac{TP}{TP+FP+FN}.
$$

New picks raise $J_{\mathrm{div}}$ only if their precision exceeds $J_{\mathrm{div}}/(1+J_{\mathrm{div}})$, so a false division's price depends on where $J_{\mathrm{div}}$ already sits.
Locally the term is dominated by misses: $J_{\mathrm{div}}$ sat between $0.06$ and $0.10$, with $127$ of $151$ events missed, and near zero almost any pick pays.
The Note 6 probe put the hidden division Jaccard near $0.32$, where the same trade loses.
My arithmetic that day assumed a hidden set of $32$ true, $20$ false and $48$ missed divisions and took it to $40$, $100$ and $40$: about $0.32 \to 0.22$ in $J_{\mathrm{div}}$, about $-0.01$ on the score, with rewired edges as a second channel.
That is arithmetic, not a measurement of the hidden set.

![Break-even precision J/(1+J) against the division Jaccard, with v85's added picks at precision 0.172]({{ site.baseurl }}/assets/img/posts/2026-09-12-biohub-working-note-7/fig-01-division-breakeven.png)
_Figure 1. The precision new division picks need rises with the current division Jaccard. v85's added picks, at $0.172$ by arithmetic, cleared the local break-even and fell short of the one implied by a hidden level near $0.32$._

My own labels agreed: the bands from $0.65$ to $0.85$, where most of v85's additions sat, were right only $13$ to $29\%$ of the time.

Two things changed.
Division operating points would now sit on the precision frontier: false positives near v83's level, and the most true positives at that level.
And I withdrew Note 6's expectation that division changes are not damped: v85 was the first division change whose local gain came back negative, so it had been a pattern in a few observations, not a rate.

---

## 2. What the Labels Were Teaching

### 2.1 A Second Batch, at the Precision Frontier

v85's written reading had also named a follow-up that would separate its two changes; the diagnosis pointed more precisely at the cost of false divisions, so the next kernel tested that prescription.
A second batch of $500$ items gave $253$ positives, and v86 shipped them at threshold $0.90$, scoring $0.7591$ locally with $18$ true and $25$ false divisions.
A pick-count model, weighting each added pick by my labels' precision in its score band, expected v86 to beat v83 ($2{,}324$ added picks at an expected precision of $0.825$, against $2{,}242$ at $0.41$).
The board returned $0.943$, a tie with v83: most of v85's loss was gone, and the board showed no gain over v83, the kernel without hand labels.
v85 and v86 had each changed the label table and the threshold together, so neither reading can say which change did what; that shaped the next kernel and became clause C16 (Section 9).

### 2.2 Measuring the Labeler Against the Annotation

The gold rows held a hint: I had answered "no" to roughly $15\%$ of the annotated divisions.
Perhaps my labels were precise about something other than what the scorer counts, so I measured how the ground truth places all $151$ annotated divisions.
At the annotated parent frame, $75\%$ show exactly one detection within $7\,\mu\mathrm{m}$ of the parent, $15\%$ show two and $8\%$ none: the annotation puts the division edge where the parent is still one nucleus, and the daughters appear about $10\,\mu\mathrm{m}$ apart one frame later.

In all seven gold divisions I had rejected, the annotated daughters sat on our candidate pair: the candidates were right and my judgment was not.
The annotations' median daughter separation was $10\,\mu\mathrm{m}$, the median of my own positives $8.7\,\mu\mathrm{m}$.
I re-judged the rejected items whose daughters were at least $9\,\mu\mathrm{m}$ apart ($99$ and $109$ from the two batches) under the annotation's convention.
Sixty changed from no to skip, and none became yes: cells appearing from behind, divisions into depth, which as "no" labels had taught the verifier to reject candidates the annotation may count.

This is a correction of convention, not a case for more labels: the union of both batches already priced below the second batch alone.
Whether the mismatch explains v85 and v86 on the board is a hypothesis I did not test.
What carries forward became clause C17: a human label is checked against the scorer's definition before it is priced.

---

## 3. One Axis at a Time

v87 kept v86's table and threshold and changed only the verifier's runtime features.
The three new features encode cues I used when labeling: the distance to the nearest detection at the daughter's position one frame early, and the brightness there over two frames.
Locally it scored $0.7609$ with $23$ true and $40$ false divisions, and its reading was written before it ran.

Before v86's score existed, I had also set a condition for that night: submit v87 if v86 reached $0.945$ or more, and not if it read $0.944$ or less.
v86 read $0.943$.
Taken literally, the condition now let a difference of $0.001$ against v83, on $29\%$ of the test, decide whether an experiment would run.
That difference is a tie, and reading it as a verdict on the label lane is what the board's resolution cannot support.
Nor did v87's question depend on v86's level: v86 was its control, and the pair was the only comparison in the program that changed one thing.
So I submitted v87 as the controlled probe it had been built to be; the record keeps the condition, set aside after its input was known, with that reason.

v87 returned $0.946$: $+0.003$ over v86, at the edge of what the board resolves, and a tie with v83.
The one single-axis step, v86 to v87, read $+0.003$, at the board's resolution floor; if it is real it belongs to the runtime features, which locally did nothing without the labels ($0.7489$ on the old table, no true divisions); the label-only kernels were v85 at $-0.005$ and v86 at a tie.

Over 2026-09-05 and 2026-09-06 I rewrote the project's selection rule:

```text
select on leakage-safe local evidence: embryo-out, kernel-faithful,
  both embryos non-negative, division operating points on the precision frontier
a Public difference within 0.002 is a tie
the board is cited only to falsify a large move (about 0.003 or more)
submit only hypotheses written down before the result exists
never rewrite a rule after a score
```

The v87 night explains why two of those lines belong together: a gate whose input is a board difference within $0.002$ should not be written at all, and a gate that is written is not rewritten after a score.
The 08-28 rule had answered a stall; this one answers the opposite risk, choosing by readings the board cannot resolve.
There were no submissions between v87 and 2026-09-10.

---

## 4. Three Questions Answered Without a Submission

On 2026-09-06 I wrote a plan with four lanes, each with a gate written before its first number: graph-stage constants, rival-parent features for the division verifier, a division-aware fine-tune of the edge head, and a detector trained on our own tracks (Section 5).
Three lanes closed within two days on their own gates, each answering a question.

**Can the edge head tell a captured daughter from a neighbor that moved in? Not well enough.**
The AUC of the deployed heads' score margin between the two cases was $0.610$ and $0.683$, against a gate of $0.70$; a head trained with more than ten thousand real zebrafish divisions reached $0.643$, no better.
The lane closed within hours of its pilot.

**Do rival-parent features move the division frontier? No.**
A three-feature logistic separated captured daughters with an embryo-out AUC of $0.80$, but inside the runtime at $0.90$ the verifier scored $0.7599$ with $21$ true and $39$ false divisions, against $0.7605$ with $23$ and $47$ for the shipped features, far from its shipping rule.
Ranking among the candidates the graph already produces was saturated with respect to geometry and appearance.

**Can a graph constant help both embryos? Not by the margin the rule asks.**
After fixing a bug that had silently ignored environment overrides in the replay, none of fourteen arms on twelve diagnostic movies passed the both-embryo rule.
Lowering the edge threshold from $0.50$ to $0.35$ gave $+0.0646$ on the smaller embryo and $-0.0117$ on three movies of the larger, the node-to-estimate ratio rising from $1.056$ to $1.146$: the threshold drops edges before the ILP, which then deletes disconnected nodes, and lowering it kept real cells on the smaller embryo and over-detections on the larger.
At $0.45$, all 199 movies gave $+0.0081$ and $+0.0008$ against a rule of $+0.003$ on both, and I closed the lane.

---

## 5. A Detector Trained on Our Own Tracks

### 5.1 The Idea

The detector's trainer marks only annotated nuclei as positive and every other voxel as negative.
Annotated nuclei are about $2.8\%$ of the real ones, so roughly $97\%$ of real nuclei were being trained as background.
Pseudo-labels are labels produced by a model instead of a person; here, our own embryo-out tracks, which added to the ground truth offered $36\times$ more supervision from the same domain ($4.76$ million nodes against $133$ thousand).
I call the model whose tracks became labels the teacher, and the detector trained on them the student.
I expected the student to over-detect and the ILP to prune the extra peaks, an expectation Section 8 tests.

The leak argument was written before the first number: the student never trains on the embryo it is scored on.
One caveat was written beside it, with a control planned for it: the tracks on the student's training embryo came from a teacher trained on the other embryo, the one the student was then evaluated on, so part of any gain could be that embryo's annotations distilled back through the teacher.

### 5.2 The Gates, and Two Clauses I Rewrote

| student, embryo-out, official scorer | tail movies | typical movies |
|---|---:|---:|
| epoch 10, as primary, $15$ larger-embryo movies | $+0.1280$ | $+0.0307$ |
| epoch 50, as primary, same movies | $+0.1358$ | $+0.0774$ |
| reciprocal student, as primary, $9$ smaller-embryo movies | $+0.1124$ | $+0.1033$ |

Tail movies are where the deployed pipeline scored worst, typical movies sit near the median, and the reciprocal student was trained the other way round.
The first row's pass rule, written in advance, asked for at least $-0.003$ on typical movies and $+0.03$ on the tail; its $+0.0307$ is the $+0.031$ headline, my first detector gain on typical movies, where Note 6's augmented detector had lost.

Two clauses written in advance did not hold, and I changed both; the gate labels stay in the record.
The reciprocal rule capped the pooled node-to-estimate ratio at the base plus $0.05$, and it rose from $1.027$ to $1.201$; before the 199-movie numbers existed, I replaced the cap with $+0.003$ on both embryos, a score that already includes the count penalty, plus a cap of $1.5$ on each embryo's median per-movie node ratio.
A later composition rule allowed the six smaller-embryo movies that had lost most as primary to lose at most $0.02$; the chosen composition lost $0.0257$, and I accepted it after that number was known and left the decision to the 199-movie run, whose rule was already written.
Both changes loosened a gate toward shipping, the direction C2 exists to guard, so each stays in the record beside the clause it replaced.

### 5.3 From 199 Movies to a Kernel

The composition carried forward used the student as the *secondary* detector, beside the replay's embryo-out primary.
On all 199 movies it moved the official score by $+0.042302511$, with both embryos positive; forty-three movies fell, including $17$ of $28$ high-base movies of the smaller embryo (base score at least $0.85$), a slice written down before any submission.

Kernel v88 swapped the secondary weights for an all-train student, selected at about epoch $52$ by a trainer proxy on $40$ training movies, and carried two small validity repairs: a guard against an added division giving a cell a third child, and integer output coordinates kept inside the volume.
The all-train student that v88 loaded was never scored anywhere; every 199-movie number came from the embryo-out pair.
The leak control had not yet run; its written reading would change how v88 was read, not whether it could be.
v88's own reading, written before its score, called $0.943$ or less materially adverse and $0.949$ or more a positive transfer.

---

## 6. What the Kernel Showed That the Replay Could Not

v88 returned $0.924$, $-0.022$ against v87, far below its adverse line.
Every local gate, as rewritten in Section 5.2, had passed, and the board saw a failure none of them could see: the transfer check doing the job it was assigned.

### 6.1 One Alignment Formula

A detector's raw output for each voxel is a logit, strongly negative where it sees background and positive where it sees a nucleus.
The kernel aligns the secondary's logit field to the primary's, then blends them before peak extraction:

$$
A=\left(S-\mu_S\right)\operatorname{clip}\!\left(\frac{\sigma_P}{\sigma_S},\,0.5,\,2\right)+\mu_P,
\qquad
B=0.525\,P+0.475\,A,
$$

where $P$ is the primary's field, $S$ the secondary's, and $\mu$, $\sigma$ are taken over the whole frame.

In plain terms, the formula assumes the two detectors were trained the same way and differ only by an offset, so it slides the secondary's whole field until its average matches the primary's.
On a bright example movie, the kernel's primary, an older all-train model, averages $-14.7$ over the frame.
The student, trained on dense pseudo-labels, fires on $10$ to $16\%$ of voxels against $3\%$ for the primary, and averages $-5.0$: its background level is about ten higher because it calls more of the frame cell.
Sliding it down by about ten drags its real peaks below the threshold too; Figure 2 shows one frame, through the deployed test-time augmentation and temporal window.

![Peaks on one frame: primary 281, with the old secondary 253, student 644, aligned student 0, primary with student 25]({{ site.baseurl }}/assets/img/posts/2026-09-12-biohub-working-note-7/fig-02-alignment-peaks.png)
_Figure 2. Peaks on one frame of a bright example movie, through the deployed path. Aligned by whole-frame statistics, the new detector left $25$ peaks where the primary alone had $281$. This is a mechanism on one frame, not a recall measurement._

### 6.2 Why the Local Instrument Could Not See It

The deployed-stack replay runs the deployed code with embryo-out weights, whose primaries sit at mean logits of about $-6$ to $-9$; there the alignment is harmless and the student adds recall.
The local pricing chain never used the deployed primary, so the composition that shipped, the old all-train primary with the student, was never measured locally.
The regimes did not even agree on a movie's cell count: on one example movie with a supplied estimate of $32{,}795$ cells, the kernel produced $18{,}423$ nodes, a ratio of $0.56$, and the replay $47{,}740$, a ratio of $1.46$.
Every count-sensitive lever priced on the replay had been priced on a field that over-detects a movie the kernel under-detects.

Rerunning the kernel's own snapshot, command and environment without the repairs reproduced it within a few nodes, ruling out the repairs, the solver and the hardware; the student alone over-detects ($61{,}553$ nodes on that movie), and only the blend collapses.
Note 6 found the selection universe differed from the deployed one at the level of code.
This was the same gap one level down: shipped code, running weights the kernel never loads.
The name kernel-faithful had described the code, not the weights.

### 6.3 The Kernel's Own Output

| four example movies, submitted outputs | v87 | v88 |
|---|---:|---:|
| official score | $0.889473$ | $0.857810$ |
| final predicted nodes | $120{,}450$ | $77{,}002$ |
| labeled-node recall | $0.994528$ | $0.957592$ |
| edge TP / FP / FN | $2027 / 156 / 100$ | $1939 / 158 / 188$ |

The pre-submission validation checked identifiers, degrees and in-volume coordinates, not score, so a file missing more than a third of its parent's nodes passed every structural check.
A score check against the parent was listed as a release step, sat outside the automated validation, and was not run.

### 6.4 What Changed: Measuring in the Kernel Regime

I adopted three rules on 2026-09-10; together they are clause C14.
First, every release is scored on the four example movies against the kernel it modifies before a submission is requested; there is no fixed veto line, but an unexpected loss of recall or edges has to be explained.
Second, a change to detection or detector composition is measured on two paths: the embryo-out replay asks whether it helps on an unseen embryo, and a kernel-regime panel, running the deployed all-train models through the kernel's own code on training movies, asks whether the shipped composition behaves.
The panel is hold-in, so it can veto but not select, and nothing ships while the two paths disagree.
Third, run records name the weights measured beside the weights that ship.

---

## 7. A Matched Control Pair on the Board

A second package, chosen before v88's score was known, averaged the primary's feature maps across the test-time views before the edge scorer reads them; I call it E.
On all 199 movies E was $+0.004523282$ over its control, with both embryos positive, and it had harms: $21$ of the $28$ high-base movies of the smaller embryo fell, and division false positives rose from $37$ to $41$.

v89 combined E with the two validity repairs v88 carried, and its plan said in advance that one board reading would not separate them.
It returned $0.943$, $-0.003$ against v87, while its official score on the four example movies moved $+0.0028158874$, the opposite sign.
Rather than infer a split from one reading, I ran a control: v90 was v89 with E switched off, specified after v89's score, so a matched control rather than an independent replication.
Its scores on the four example movies were identical to v87's, and it returned $0.946$, the same score as v87.

The pair supports a modest reading: the repairs cost nothing the board could see, and the E package points negative, at the resolution floor.
It cannot show the repairs are free, because a tie at three decimals is censored, nor that E alone is harmful, because $-0.003$ sits on the board's resolution floor.

---

## 8. The Leak Control, and the Repair

### 8.1 The Teacher, Moved Inside the Training Embryo

The control trained its teacher only on the student's own training embryo; a first written version, taking the teacher from the other fold, would have recreated the very path it was meant to remove.
A pseudo student and a ground-truth-only model were trained with the same seed, windows and fixed $60$-epoch endpoint, and each ran standalone (its own detection and association, secondary and verifier off) on the four example movies in both directions.

| clean control, standalone, four movies | ground truth only | pseudo student | difference |
|---|---:|---:|---:|
| all four | $0.7459$ | $0.7253$ | $-0.0206$ |
| larger-embryo movies | $0.7413$ | $0.7215$ | $-0.0198$ |
| smaller-embryo movies | $0.8474$ | $0.8112$ | $-0.0362$ |

The mechanism is the one that closed Note 6's augmentation program.
Labeled-node recall rose from $0.936$ to $0.971$, but final nodes rose by $24{,}712$ and edges gained $83$ true positives and $122$ false ones; the count adjustment alone contributed $-0.0175$, and divisions were unchanged.
Recall was bought with nodes that crossed the count boundary.
That forces a retraction: on 2026-09-06 I wrote that the ILP prunes the student's inflated peaks, and in the clean control it did not.

This is a standalone composition on four movies, smaller than the panel the 09-06 rule had named, and not a verdict on pseudo-supervision as a family.
It is still the only measurement with the leak removed, and it is negative in both embryos.
A headline gain waits for its leak-free control before it is trusted: clause C15.

### 8.2 Repairing the Alignment

The obvious repair for v88 was a regime-independent alignment; under the 09-10 rule it was measured first in the kernel regime, on the four example movies with all-train models.

| composition (kernel regime, four movies) | official score | difference |
|---|---:|---:|
| base: deployed detection and association | $0.8898631853$ | — |
| A: deployed detection, student association | $0.8905639836$ | A−base $+0.0007007983$ |
| Q: A plus the student in detection, quantile alignment | $0.8818274412$ | Q−A $-0.0087365423$ |
| P: A plus the student in detection, probability blend | $0.8736703501$ | P−A $-0.0168936334$ |

Q and P were both positive on the smaller embryo, negative on the larger, and both lost recall.
The association-only change A with embryo-out models on the same movies gave $-0.003265956$, negative on both embryos; the two paths disagreed in sign, so it did not ship.

---

## 9. Decision Log

The rule in force from 2026-09-06 was the local-first rule of Section 3.
The board's one job was a transfer check against an expectation written before the score: a tie within $0.002$ changes nothing, and a move of about $0.003$ or more says the local instrument is missing something.
A local criterion can be wrong in ways it cannot see from inside; this week the check found two such ways, v85 and v88, and chose nothing.

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| Ship v85, first label-trained verifier | Local $+0.0048$, both embryos up | $0.939$, $-0.005$ | False divisions priced near zero locally; precision frontier |
| Ship v86 at the precision frontier | Test the diagnosed mechanism | $0.943$, a tie with v83 | Two axes at once cannot be read (C16); labeler measured (C17) |
| Submit v87 after v86 read $0.943$ | Single-axis contrast; a gate on a tie reads noise as a verdict | $0.946$, $+0.003$ over v86 | The one single-axis reading, at the resolution floor; rule of 09-06 |
| Close three lanes on their gates | Stops written before the first number | Each below its gate | Three questions answered, no submission |
| Ship v88, pseudo-label student as secondary | $+0.042302511$ locally, both embryos up; adverse line written | $0.924$, $-0.022$ | Kernel regime and score check before release (C14) |
| Submit v89, then control v90 | E $+0.004523282$ locally; control, not inference | v90 $0.946$, a tie with v87 | Repairs cost nothing visible; E points negative, at the resolution floor |
| Leak control, teacher inside the training embryo | The $+0.031$ carried a cross-embryo teacher | $-0.0206$, both embryos negative | Leak-free control before a headline (C15) |

### The criterion at the end of this period

| clause | wording | since |
|---|---|---|
| C1 | Measure every graph edit out of fold: fit, calibrate and evaluate on disjoint movies, scored by the official metric on the whole graph | Note 2 |
| C2 | Write each gate down before the result exists | Note 3 (07-15) |
| C3 | Calibrate a rule on the population it will act on | Note 3 |
| C4 | Judge a component by the graph it produces, in an exact replay, not by its own accuracy | Note 3 |
| C5 | Compare levels only inside one reference universe; compare deltas across | Note 3 |
| C6 | A candidate must finish on the hidden set within the time limit | Note 3 |
| C7 | Folds hold out a whole embryo (embryo-out) | Note 4 |
| C8 | Numbers from movies the deployed model trained on (hold-in) are not evidence of generalization | Note 4 |
| C9 | Use the board for matched transfer checks with a written expectation, not to choose adjacent settings | Note 4 (08-10) |
| C10 | Measure the ceiling of an action space before optimizing inside it | Note 5 |
| C11 | A gate must be able to end in a decision | Note 5 |
| C12 | Measure in the deployed universe: replay the pipeline that ships | Note 6 |
| C13 | Price a hidden-only term with a designed decomposition probe | Note 6 |
| C14 **(new)** | Measure in the kernel regime (shipped weights and code); score the notebook's own output against its parent before submitting | Note 7 |
| C15 **(new)** | Run the leak-free control before trusting a headline | Note 7 |
| C16 **(new)** | One axis per submission, or a matched control arm | Note 7 |
| C17 **(new)** | Labels follow the scorer's convention; measure the labeler against ground truth first | Note 7 |

---

## 10. What the Period Established

### Established

1. A hand-label verifier priced at $+0.0048$ locally, both embryos up, returned $-0.005$ on the board; locally, misses dominate the division term, so false divisions cost almost nothing there.
2. $75\%$ of annotated divisions show one nucleus at the parent frame, and all seven gold divisions I rejected sat on our candidate pair.
3. The pseudo-label student gained $+0.042302511$ on the deployed-stack replay; the kernel carrying its all-train version scored $0.924$ and passed structural validation with more than a third of its parent's nodes gone.
4. With E off, v90 tied v87; with E on, v89 was $0.003$ lower.
5. With the teacher confined to the student's training embryo, the pseudo student lost $0.0206$ standalone on four movies, both embryos negative; both repaired alignments were negative in the kernel regime.

### Supported but Unconfirmed

1. That v85's loss sits in division precision on the hidden set; this is arithmetic.
2. That the label-convention mismatch explains v85's and v86's board results.
3. That the v86-to-v87 step ($+0.003$, at the resolution floor) belongs to the runtime features; the label-only kernels were v85 ($-0.005$) and v86 (a tie).
4. That E itself, rather than its combination with the repairs, caused v89's lower score.
5. That most of the original $+0.031$ was cross-embryo distillation.

### Open Questions

1. Can any composition deliver the pseudo detector's recall in the kernel regime without paying for it in node count?
2. For changes that never touch the detection field, how much of an embryo-out gain survives in the kernel regime?

---

## Closing

Both large board moves of the period were transfer checks that found something: v85 a local proxy blind to the cost of false divisions, v88 a local replay that was not the kernel.
The ties stayed ties, and v87 was the week's one reading that changed one thing.

The question I now ask of a local gain has three parts: is it free of leakage, is it measured in the code that ships, and is it measured on the weights the kernel actually loads?
Note 4 added the first, Note 6 the second, and this week the third, each from a gap the board exposed in an instrument that had looked clean.

The local evidence at the end of the week is narrow.
The pseudo detector has no path into the kernel with a positive measurement behind it; the hand labels bought local signal and a corrected convention but no board gain of their own; and in the division channel, ranking among the candidates the graph already produces is saturated with respect to geometry and appearance.
The instruments now exist: an embryo-out replay that asks whether a change helps on an unseen embryo, a kernel-regime panel that asks whether the shipped model behaves, a score check before release, and a leak control before a headline.
The next note's question is what the rule looks like when it has to choose the final two submissions.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
- [Part 5: Optimizing an Objective That Could Not Reach Gold]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold/)
- [Part 6: The Universe We Were Selecting In]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-The-Universe-We-Were-Selecting-In/)
- **Part 7: Where a Local Gain Has to Be Measured**
