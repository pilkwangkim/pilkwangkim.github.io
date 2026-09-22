---
title: "BioHub Cell Tracking Working Note 5: Optimizing an Objective That Could Not Reach Gold"
date: 2026-08-28 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, objective-design, oracle-bounds, process-debt, division-recovery, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-08-28-biohub-working-note-5/cover.png
  alt: "Title card for BioHub Working Note 5: optimizing an objective that could not reach gold"
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

# BioHub Cell Tracking Working Note 5: Optimizing an Objective That Could Not Reach Gold

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
  - [Working Note 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
  - [Working Note 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
- Korean version: [BioHub Cell Tracking 작업 기록 5: 도달할 수 없는 목적함수를 최적화하기]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 6: The Universe We Were Selecting In]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-The-Universe-We-Were-Selecting-In/)

Related public notebooks:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

Note 4 ended on a cap.
On the replay, 99.25% of the missed annotated cells had no predicted node inside the gate, and the deployed pipeline's own cap was one run away.
It also left a repaired, embryo-disjoint evaluation contract, and a board that had refused the largest clean association gain I had recorded, rightly.
The rule going into this note had two layers.
The 08-10 program still named a Public score of $0.970$ as the outcome target.
Selection ran on a frozen local comparator, and the charter said the board was not the selection objective.

This note covers 2026-08-17 to 2026-08-28.
It is the period in which the project kept that rule most faithfully and got the least from it.
Almost every result was negative or exactly zero.
The one positive went through the full release chain and came back from the board with the same score as the model it replaced.
The public score, $0.921$, had last moved on 2026-08-05.

On 2026-08-24 I measured what the comparator could pay.
A ground-truth filter over every action it selected was worth $+0.000365$.
On the last day of the window the project reset its rules and swung back to the board as the objective, as a rule against standing still.
The number itself supports something narrower.
Being independent of the board does not make a criterion worth optimizing.
An internal criterion needs a ceiling measurement first: the best score reachable inside it with perfect knowledge, set against the gap it is supposed to close.

The same day, division recovery, a channel whose ceiling was measured before anything was optimized in it, produced a local candidate worth $+0.0053$.
It went to the board that evening. The board had not answered when this window closed.

The short version is:

```text
The rule held: a frozen local comparator, not the board, was the selection objective.
Every experiment against it was correct, and almost none of them moved it.
A ground-truth filter over every action it selected was worth +0.000365.
An internal criterion needs a ceiling measurement before it is worth optimizing.
The 08-28 reset swung back to the board, as an anti-stall rule.
In the division channel the ceiling came first (+0.0557), and one day gave +0.0053.
```

The note is arranged so that the ceiling arrives before the experiments:

| Sections | Question |
|---|---|
| 0 | What rule was in force, and why was it a good rule? |
| 1 | What was being maximized, and what could it pay? |
| 2--3 | What did correct experiments produce under that ceiling, and why could the process not see it? |
| 4 | What did looking outside the comparator show? |
| 5 | What did the 08-28 reset change, and what did it get wrong on its first day? |
| 6 | What happened in a channel whose ceiling was measured first? |
| 7--8 | Which rule did this period select by, and which claims survive? |

---

## 0. The Rule, Stated Fairly

At the start of this window the project's charter contained two sentences that decided almost everything that follows.
The first took the leaderboard out of the selection loop:

> Public leaderboard values are transfer observations, not the model-selection objective. Fold-pure local OOF and exact graph replay remain the promotion authority.

The second rationed the channel:

> Public transfer probes are optional and sparse. A planning cycle permits at most one Kaggle submission whose primary purpose is transfer measurement; many cycles should use none.

The case for that rule is strong.
The public board shows a 29% slice of a hidden test set drawn from unseen embryos, which may amount to roughly one embryo.
Its values are rounded to three decimals, so a displayed tie does not order two models.
By this point the project had made about 160 submissions, many of them one-axis sweeps around a single operating point.
Taking the argmax of a noisy, rounded, partial statistic that many times fits the statistic, not the task.
Note 2 was written because the board could not separate three explanations of a plateau.
Against that history, "choose out of fold, not on the board" was the correct response to a real failure mode.

The training data holds two embryos, called prefixes below after the first characters of their movie names.
Embryo-out means scored only by models that never saw that embryo; hold-in means scored on movies the models trained on.
Local instruments are named where they are used, are never compared with each other, and reach the board only through the arithmetic of section 1.4:

- the comparator: a 199-movie embryo-out replay, frozen at $0.6013708666$ once the fragment matcher of section 2.2 shipped, and its subset on the four distributed test movies (copies of training movies), where the frozen graph scores about $0.64$;
- replays of the deployed notebook's own pipeline on the same four movies: hold-in ($0.8899$) and, in section 5.4, with embryo-disjoint detectors (about $0.81$); a different pipeline, never set against the $0.64$;
- a 177-movie replay in a different node universe, base near $0.69$, and one single held-out movie.

The board returned three scores in the window: for a deployment migration that carried no new model (2026-08-17), for a runtime probe with byte-identical output (2026-08-20), and for the release of the period's one local positive (2026-08-22).
All three returned $0.921$, the same score at the board's resolution as the 2026-08-05 candidate.
Nothing was tuned to those numbers.
The board was used exactly as the rule said: rarely, and as an observation.

---

## 1. What I Was Maximizing, and What It Could Pay

### 1.1 The Objective

Write the deployed system as in Note 2: a graph produced by a model composed with a deterministic post-processing stack, $\hat G=P_\theta(M_W(X))$.
The rule froze $M_W$ and most of $P_\theta$.
It exposed a restricted set of legal graph edits over a fixed node universe and a fixed base topology, and made the objective

$$
\theta^\star
=
\arg\max_{\theta\,\in\,\Theta_{\mathrm{frozen}}}
S_{\mathrm{OOF}}\!\left(\hat G_{\mathrm{frozen}} \oplus \mathcal A(\theta)\right),
\qquad
S_{\mathrm{OOF}}\!\left(\hat G_{\mathrm{frozen}}\right)=0.6013708666 .
$$

Two properties of that objective matter, and neither was checked when it was installed.
It has an action space, and an action space has a ceiling: the best score reachable if every action were chosen with perfect knowledge.
And it is a different quantity from the one that decides the competition, related to it only through a transfer ratio.

### 1.2 The Oracle

On 2026-08-24 I asked the question that should have come first.
Not "does this candidate improve the comparator?", but "what is the most that any candidate could improve it?"

Take the 3,178 actions the comparator selected across the 199 movies, replayed from the out-of-fold graph they were applied to.
Keep an action only if the labels say it strictly helps: no true edge lost, no false edge added, and at least one of the two strictly better.
Drop it otherwise, and rescore.
It is a filter over the comparator's own selections, not a policy anyone could fit.

The answer, in the 199-movie embryo-out replay:

$$
S_{\mathrm{OOF}}:\;0.6013708666 \;\longrightarrow\; 0.6017363254,
\qquad
\Delta^{\mathrm{oracle}}_{\max}=+0.0003654588 .
$$

All of it came from one embryo prefix; the other could not be moved at all.

### 1.3 The Same Ceiling From the Other Side

A second measurement looked at the damage in the graph instead of the policy's actions, so it is independent of the first.
A replay of the comparator's candidate generation recovered 58,427 candidate edges, the complete inventory of edge swaps it could make without adding a node.
It then searched that inventory for each of the 38,060 ground-truth edges the graph was missing.
Only 183 had any replacement at all, and at most 115, 0.30% of the missing edges, were topology-legal swaps.

![Bars: 38,060 missing ground-truth edges, 183 with any replacement, at most 115 topology-legal swaps]({{ site.baseurl }}/assets/img/posts/2026-08-28-biohub-working-note-5/fig-01-edge-swap-inventory.png)
_Figure 1. Of $38{,}060$ ground-truth edges missing from the comparator graph, $183$ had any replacement in the complete swap inventory and at most $115$ were topology-legal swaps. Division edits are outside this count._

The two measurements agree from opposite directions.
The binding constraint was not the ranker, the representation or the calibration.
For edge repair by swapping, the graph did not contain the pieces it needed.
Division edits were outside both counts; section 6 returns to them.

### 1.4 Pricing the Ceiling Against the Competition's Target

The competition is decided on the hidden test set, and its only visible view was the board, where the distance to the gold band was measured in hundredths.
For the edge channel the project had written down one transfer band.
On 2026-08-06, local edge deltas had been recorded as reaching the board at between $0.14\times$ and $0.9\times$ their local size.
That is a historical range across heterogeneous replays, not a controlled coefficient.
Note 4's standalone association gain ($+0.0073$ locally) reached the board as a tie with its forward control, so the band is at best an optimistic bound.
Taken loosely, it sets a visibility condition: the board prints three decimals, and a change must reach half a display unit to have any chance of showing:

$$
\rho\,\Delta_{\mathrm{local}} \;\ge\; \tfrac{1}{2}\times 10^{-3},
\qquad
\rho \in [0.14,\,0.9].
$$

Solving gives a local threshold of $5.6\times10^{-4}$ at the favorable end and $3.6\times10^{-3}$ at the other.
That is arithmetic, not a measurement.

```text
Perfect knowledge over the comparator's actions: +0.000365 locally.
At the top of that band: at most +0.00033 on the board.
Half of one display unit: +0.0005. The gap: hundredths.
```

The ceiling took less than a day to compute, and the rule that installed the comparator never asked for it.
Some experiments in the next section ran before I measured it and some after; those measured against the comparator ran underneath it.

---

## 2. Correct Experiments Under the Ceiling

These are the measured results of the window, each in the instrument it was measured in.
Several concern the track-fragment matcher, a learned model that relinks a broken track end to a successor one or two frames later.

| experiment | instrument | delta | observation |
|---|---|---:|---|
| additional detector seed, two-fold exact replay (about 36 GPU-hours) | 177-movie replay, base $\approx 0.69$ | $-0.0000325$ | folds $+0.0000560$ and $-0.0003967$ |
| matcher, one training-group weight $2.0\to4.0$ | 199-movie embryo-out | $-0.0000427$ | both prefixes negative |
| matcher with a track-history input | 199-movie embryo-out | $-0.0000451$ | passed every pre-registered check; closed |
| auxiliary-center-detector and gap-filling composition pilots | 199-movie embryo-out | $+0.0000032$, $-0.0003537$ | neither passed its gates |
| association rank with a candidate filter applied first | one held-out movie | $0.0$ exactly | $0.8277207263$ in both arms |
| three local-representation families | comparator, four-movie subset | $0.0$ exactly, three times | zero legal actions (section 2.3) |
| **matcher, retrained independently at a fixed checkpoint** | 199-movie embryo-out, pre-matcher graph | $\mathbf{+0.0006686}$ | 34 movies improved, 9 worsened, 156 tied |

Not every row sits inside the comparator.
The last row was measured against the out-of-fold graph before any fragment matcher ran ($0.6006730$), and the weight row used the last row's arm as its control.
That matcher line is what set the comparator's level, $0.6013708666$; the oracle of section 1 prices what was left above it.
The detector-seed row lives in a universe of its own.

### 2.1 Thirty-Six GPU-Hours at the Fifth Decimal

The largest compute item of the campaign was a two-fold exact-graph replay for an additional detector seed.
One fold was measured at 12.78 hours of training and the other projected at roughly 23.
The aggregate moved the incumbent from $0.6925340486$ to $0.6925015848$, a delta of $-0.0000325$.

When I reviewed the period on 2026-08-28, my first summary quoted this line as $+5.6\times10^{-5}$.
That is the positive fold only; the aggregate is the number.
Quoting the friendly half of a two-fold result is exactly the habit an internal criterion exists to prevent.

### 2.2 The One Positive, Released, and a Tie

The period's one positive of any size was the matcher itself, retrained independently at a fixed checkpoint as the control arm of the weight experiment.
Against the pre-matcher out-of-fold graph it measured $+0.0006685607$ overall: $+0.0014526$ on one prefix and $+0.0005501$ on the other, across 3,207 edits that kept the graph a valid lineage, with the division false-positive count unchanged.
Two successors in the same line added correlation inputs and about $0.00003$ more; the last of them is what was released, and its score became the comparator.

The release chain then ran end to end without a skipped step: an all-train refit, new immutable registry versions, and a private notebook that had to reproduce the local replay's prediction bytes before it could be submitted.
It did, went to the board on 2026-08-22, and returned $0.921$.

That tie is what the arithmetic of section 1.4 said was most likely.
At the top of the transfer band, $+0.00067$ locally maps to about $+0.0006$ on the board, just past half a display unit; at the other end, to $+0.00009$.
So the tie is evidence neither against the local result nor for it.
What it establishes is narrower: increments of this size could not be relied on to add up to something the board could see, in a space that section 1 shows was nearly empty.

### 2.3 Three Representations, Zero Actions

Three local-representation families were built to give the same fixed-node action policy a better matching signal: a three-dimensional phase correlation, a learned dense-descriptor cost volume, and a soft-ternary census cost volume.
All three ran to completion through the paired official scorer and passed every pre-registered check.
All three selected zero legal actions in both prefixes.
So all three returned the frozen graph's four-movie score, $0.6391791890184004$, to sixteen digits.

The queue was then declared blocked, because no other representation was ready under the same fixed-node action policy.
That diagnosis was scoped to a family of ideas the evidence had just shown to be irrelevant.
The policy was the frozen half of every comparison, so nothing in the process could produce evidence against it.

### 2.4 The Level Was Real

One negative is worth keeping for the opposite reason.
A classical baseline, full-$Z$ Otsu thresholding with adjacent-frame Hungarian linking, went through the same four movies and scorer.
It scored $0.4626$ against the frozen learned graph's $0.6392$, a gap of $-0.1766$.
The learned stack's level was real.
Only its increments inside this objective were not.

---

## 3. Why the Process Could Not See the Ceiling

The rule of section 0 was enforced by machinery built during Note 4's period: predeclared experiment cards, gates a run had to pass before it could start or count, immutable receipts, and a separate module tree for each experiment.
Each piece was defensible on its own.
Together they could refute any candidate, and had no step that asked what the objective itself was worth.

### 3.1 The Cost of One Question

Each experiment required about sixteen new files, and the last one of the window changed a single axis: the number of sampler draws per epoch.
One planning file written on 2026-08-16 described 331 tasks.
Its own status field declared it non-authorizing, and its task counter never left 0/331.
On 2026-08-20 and 08-21 that counter was held at zero by two receipt schemas that disagreed about the name of one field: one called the file length `bytes`, the other `size`.
Verifying three non-executing compatibility adapters re-read about 696 GB.
Nothing was trained, predicted or scored.

### 3.2 A Fixed Bug Needed a New Experiment

On 2026-08-21 an association experiment completed training, prediction and the graph reload, then died at the scorer handoff on a one-line field-name mismatch.
Under the rules the fix had to launch as a new experiment with its own name and declaration, so the full pipeline ran twice to learn that the candidate changed nothing: $0.8277207263$ in both arms.

### 3.3 Gates That Could Not Close Anything

After the oracle, the plan turned to the node universe, where the ceiling pointed: a native spot detector, gated behind predeclared readiness floors such as held-out detector recall of at least $0.50$ at $7\,\mu\mathrm{m}$.

It ran twice, at 64 and then 256 sampler draws per epoch.
Detector recall was $0.179$ in one held-out fold and $0.036$ in the other.
Zero of 15 decoder settings were jointly feasible in either fold.
The held-out inputs were never opened, and no movie was scored.
Under the rules, a readiness failure was explicitly neither a score result nor a closure of the model family, so the outcome was unfalsifiable in both directions.

A gate has to be able to close something.
If failing it produces an outcome that is evidence neither for nor against, the gate protects the process from information.

### 3.4 The Idle Accelerator

At 2026-08-28 00:02 UTC, with 32 days to the deadline, the RTX 5090 was at 0% utilization with 2 MiB of 32,607 MiB in use.
The planning record beside it stated that an idle GPU is not a reason to start a run.
The review estimated that the twenty-day GPU program had used roughly 60 to 70 productive hours of a planned 480, about 15%; no per-job ledger exists to confirm it.

Every run that produced a number was correct.
Correctness inside an unpriced objective looks, from inside, exactly like progress.

---

## 4. Looking Outside the Comparator

The imports were negative, and the comparison produced the one new structural fact of the fortnight.

### 4.1 External Models on Our Frozen Nodes

On 2026-08-20 four externally trained models were adapted onto our frozen nodes and topology, so that only their association or repair decisions could differ.
On the comparator's four-movie subset, a forward-acceleration lookahead found zero eligible switches ($0.0$), an exact-coordinate detector adapter scored $-0.0044123$, and a learned movement-field model $-0.0017270$.
A 4-D convolutional adapter failed closed in $0.324$ seconds, before scoring anything.
Every fixed-node import was negative or inert: further evidence that the substrate was binding, not the association rule.

### 4.2 A Node Residual, Found and Not Used

The same day, a cheap scratch detector trained for 40 epochs on an inner fold of one embryo produced 11,594 proposals on one held-out movie; 1,190 of them lay farther than $7\,\mu\mathrm{m}$ from any node we had.
The gates stopped the line there, because no adapter for the detector's output format had been declared in advance, and whether those proposals were real cells was never measured.
The lever Note 4's error anatomy called binding gave a signal on its first try and was stopped for a reason unrelated to the signal.

### 4.3 Where the Difference Was

On 2026-08-26 I compared our graphs with an independently built reference pipeline on the four locally scorable movies, twice.
Against the reference's native build, the reference carried 53,858 nodes ours lacked and ours carried 59,382 it lacked; among those extra nodes the reference recovered 103 annotated cells to our 19.
Against a second build of the reference, with about 14,000 exclusive nodes on each side, the edge Jaccard over the 106,543 shared nodes was $0.9978$, and the reference emitted 384 parents with two children to our 37.

On shared nodes the two association results nearly agree; the difference is which nodes exist, and how many divisions are emitted.
This is an inference from our own diagnostics on four movies, not a measurement of anyone's score.
It also explains section 2.3: new association representations could not help, because association over shared nodes was already the part that was not broken.

---

## 5. The Reset of 2026-08-28

### 5.1 The Pressure

By the last day of the window the public score had not moved for twenty-three days.
The board had been asked three times in the window, against an allowance of five submissions a day, and the accelerator was idle.

### 5.2 The Review and the New Rules

On 2026-08-28 I audited the whole project.
The review named four root causes:

1. the objective had been inverted, because a frozen local comparator had replaced the board;
2. process had replaced modeling;
3. the GPU sat idle while work existed;
4. proven gains had been left on the shelf.

The apparatus of section 3 was retired.
Six rules replaced it: the public board is the objective and local out-of-fold is a selection tool; no governance code; the GPU never idles while work exists; two hours from a new idea to the first prediction-facing artifact; a failed run is retried under the same name after the bug is fixed; closed results are never re-derived.
The plan written with them set out to reach the gold band on the board by the deadline, and it counted up to five submissions a day as nearly free transfer evidence.

### 5.3 What Kind of Rule the First One Is

The reset's first rule is an anti-stall rule.
After twenty-three days in which the board did not move, the last twelve spent on a comparator that section 1.4 said could never move it, the board was the only instrument on the competition's target.
The rule asks every experiment a question the old rule never asked: could the board ever see this?
It does not supply a ceiling, and it does not say what stops the adaptation of section 0 from coming back once submissions are counted as nearly free.
Section 1 says an internal criterion has to be priced before it is optimized; the reset replaced the criterion instead of pricing a better one.
The ceiling measurement applies to any objective, and section 6 shows what it looked like on the same day.

### 5.4 The Fourth Root Cause, Tested the Same Day

The new plan's top item was to redeploy a shelved association composition whose replay evidence read $+0.0144$ and then $+0.0313$, all folds and prefixes positive, projected at $+0.005$ to $+0.025$ on the board.
Those figures came from the four-fold research replay (base near $0.74$), whose folds, as Note 4 showed, mixed both embryos: movie-out, not embryo-out.
The 2026-08-10 audit had demoted them.
The plan put them first anyway, the failure of Note 4 repeated on the first day of the new rules.
Three A/B harnesses tested it that day on the four locally scorable movies, each against its own control:

| rung | regime | control | treatment | delta |
|---|---|---:|---:|---:|
| 1 | hold-in, control not faithful to the deployed kernel | $0.8831$ | $0.8842$ | $+0.0011$ |
| 2 | hold-in, control reproduces the deployed kernel exactly | $0.8866$ | $0.8858$ | $-0.0008$ |
| 3 | embryo-disjoint detectors, four-fold association heads | $0.8064$ | $0.8129$ | $+0.0065$ |

Rung 1's control sat below the deployed kernel's own $0.8899$, weaker than the pipeline it stood in for, which disqualifies it.
Rung 2, the honest hold-in measurement, is slightly negative.
Rung 3 swapped in embryo-disjoint detectors and is positive, with all four movies non-negative.
But it still scored the lever with the four-fold association heads that Note 4 had identified as prefix-mixed, so it is not an embryo-out test of the lever.

The fourth root cause did not survive the day: against the faithful deployed base, the shelved lever measured $-0.0008$.

One reading concerns what a shelved number means: a lever's value, $\Delta_L(B)=S(L\circ B)-S(B)$, is a property of the pair $(L,B)$, not of the lever.
Since the shelved measurement, the deployed base had gained test-time augmentation, Note 4's reverse-time harmonic fusion, consensus and a tuned edge threshold.
These rungs do not separate that base absorption from fold leakage.
The "about 75% absorbed" I wrote down that day compares deltas from three universes; it is not a decomposition.

---

## 6. A Channel With Its Ceiling Measured First

### 6.1 The Structure, Not the Last Edit

The plan's second item was a division recoverability audit.
Division carries a tenth of the score, $S = J^{\mathrm{adjusted}}_{\mathrm{edge}} + 0.1\,J_{\mathrm{division}}$.
For weeks the channel had been treated as a guard not to disturb, because touching it had cost about $-0.00018$; that was true of the one edit that had been tried.
The audit measured the channel's structure instead, on the 199-movie embryo-out graphs (a fork is a node with two outgoing edges, a predicted division):

| quantity | value |
|---|---:|
| ground-truth divisions in all 199 movies | 151 |
| fully recoverable (parent and both daughters matched within $7\,\mu\mathrm{m}$) | 107 |
| with at least one of the two parent-to-daughter edges already in the graph | 101 |
| with both parent-to-daughter links present (13 through a single predicted child, not a fork) | 15 |
| predicted forks at annotated parents | 849 |
| of those, sitting on a true division parent | 2 |

The graph already held most of what these events needed.
It was emitting 849 forks at annotated parents and landing 2.

### 6.2 A Gate Pointed the Wrong Way

Recoverable true daughter pairs separate at a median of $10.7\,\mu\mathrm{m}$; the forks the scorer counted as false separate at a median of $5.4\,\mu\mathrm{m}$.
The deployed acceptance rule admitted a fork only when the two putative sisters were close together and each was close to the parent.
Its sister-distance limit admitted 96.3% of the counted false forks and 29.1% of the true divisions.

The deployed division gate was not badly tuned.
It was selecting the geometric complement of the truth: tight pairs, mostly false forks, in; wide pairs, mostly real divisions, out.

### 6.3 The Ceiling

With the geometry understood, the division stage was rebuilt in oracle mode across all 199 graphs: a daughter could be moved to a new parent (reparenting), and forks were chosen by the labels.

| arm | score | delta vs $0.6014$ | division detail |
|---|---:|---:|---|
| fork-free base | $0.6005$ | $-0.0009$ | prices the whole deployed division stage at $+0.0009$ |
| deployed strict stage | $0.6014$ | — | the incumbent |
| ground-truth oracle, reparenting allowed | $\mathbf{0.6571}$ | $\mathbf{+0.0557}$ | $J_{\mathrm{div}}=0.5577$; 87 TP, 5 FP, 64 FN; 107 forks added, 37 by reparenting |

Compare that with section 1, on the same 199 movies with the same scorer.
Perfect knowledge over the frozen comparator's selected actions was worth $+0.000365$.
In the division channel it was worth $+0.0557$, two orders of magnitude more.
This time the ceiling was known before anything was optimized.

![Log-scale comparison of the comparator ceiling (+0.000365) and the division ceiling (+0.0557) with measured gains]({{ site.baseurl }}/assets/img/posts/2026-08-28-biohub-working-note-5/fig-02-two-ceilings.png)
_Figure 2. The same scorer on the same 199 movies. A ground-truth filter over every action the comparator selected was worth less than the local size the board could show; the division stage's ceiling was two orders of magnitude larger. The shaded band is arithmetic from a historical transfer range, not a measurement._

### 6.4 A Verifier Inside the Ceiling

A learned verifier was then fitted to rank the reparent-enabled candidates, out of fold and prefix-pure: the ranker applied to each embryo was trained only on the other one.
It went through three versions in one day.
A linear model on geometry alone reached $+0.0015$ with 6 of the oracle's 87 true positives: better than the deployed stage, and almost none of the ceiling.
A gradient-boosted ranker with track features reached $+0.0035$.
Adding appearance features reached $+0.0053$, a score of $0.6067$, with 11 true positives and 26 false positives.

The same day measured two more limits.
On the embryo it was trained on, each ranker's simulated division Jaccard was $0.43$ and $0.67$; applied to the other embryo, the pooled figure was $0.062$.
The loss was in ranking transfer across embryos, not in the threshold.
And the candidate generator covered only 75 of the 151 events, a second ceiling underneath the first.

### 6.5 The Local Gate, and a Submission the Board Had Not Answered

The order of operations that day was ceiling first, optimization inside it second, and a local gate third.
A predeclared hold-in gate checked that division edits would not damage the edge channel.
The candidate kernel reproduced the previous kernel's four-movie hold-in score of $0.8899$ with identical per-movie edge counts; all 45 of its division edits landed away from the sparse annotated tracks the scorer checks, which is a property of this check, not a general license.

The submission went out on 2026-08-28 at 12:51 UTC, and the board had not returned a score when this window closed.
The only expectation written that day for this channel was that its gain would reach the board at a similar order, not damped like edge deltas.
The record behind it points two ways.
The edge band of section 1.4 would put a local $+0.0053$ anywhere from about $+0.0007$ to $+0.005$ on the board; that is arithmetic.
The two rounded division pairs of Notes 3–4, $+0.000949$ and $+0.0010142$ locally with a $+0.004$ step on the board each, were larger than any edge lever of comparable local size had produced; that is a direction on two rounded pairs, not a rate.
Whatever the board returns is one observation of how the division channel transfers.
A clear drop would falsify the candidate, the job Note 4 showed the board can do.

---

## 7. The Rule We Selected By

The rule in force was independent of the board, it was obeyed, and it still could not select anything worth having.
Independence from the board is necessary.
This period showed that it is not sufficient.
The frozen comparator was clean, embryo-out and leakage-safe, and every action it selected, filtered by ground truth, was worth $+0.000365$, which section 1.4 prices below half a display unit on the board even at the top of its band.
An internal criterion earns the right to be optimized when its label-oracle ceiling exceeds the gap it is supposed to close, and that has to be measured before the first experiment, not after the twentieth.

The reset of 2026-08-28 swung back to the board as the objective.
It was adopted as a rule against standing still.
Whether it can also serve as a rule for selecting is a question this period raised and did not answer.
The candidate that left on the same day was chosen by a local ceiling, a local out-of-fold verifier and a local gate, before the board had said anything.

| | this period |
|---|---|
| rule in force | a frozen 199-movie embryo-out comparator is the selection objective; the board is a sparse transfer observation (a Public $0.970$ outcome target on paper) |
| where it was measured | embryo-disjoint twofold replay, official scorer, both prefixes reported |
| what the board was used for | three submissions, each the same score at the board's resolution; none could have resolved the deltas being measured; a fourth, the 08-28 division candidate, unanswered at the close |
| what broke the rule | a label-oracle ceiling of $+0.000365$ over every action the comparator selected |
| what replaced it on 08-28 | "the public board is the objective", adopted as an anti-stall rule |

---

## 8. What the Period Established

### Established

1. A ground-truth keep-or-drop filter over the frozen comparator's 3,178 selected actions was worth $+0.0003654588$ ($0.6013708666 \to 0.6017363254$), all of it in one embryo prefix.
2. Only 115 of 38,060 missing ground-truth edges (0.30%) had a topology-legal corrective swap in the frozen node universe.
3. The fragment-matcher line, whose fixed-checkpoint retrain measured $+0.0006686$ over the pre-matcher graph with both prefixes positive, returned the same public score at the board's resolution after a complete release chain.
4. Three unrelated local-representation families each selected zero legal actions and scored exactly the frozen graph's $0.6391791890184004$.
5. Every fixed-node import of an externally trained model was negative or inert; a classical Otsu-plus-Hungarian control scored $0.4626$ against the learned graph's $0.6392$.
6. Of 151 ground-truth divisions, 107 were fully recoverable from the embryo-out graphs and 101 already carried at least one of the two required edges; of 849 forks predicted at annotated parents, 2 sat on a true division parent.
7. The deployed division gate's sister-distance limit admitted 96.3% of counted false forks and 29.1% of true divisions.
8. A reparent-enabled ground-truth oracle over the division channel was worth $+0.0557$ on the 199-movie embryo-out replay, and a prefix-pure learned verifier reached $+0.0053$ inside it in one day.

### Supported but Unconfirmed

1. That the remaining headroom lies in the node universe and the division channel rather than the association rule. It rests on two four-movie comparisons against a reference pipeline (section 4.3).
2. That the deployed machinery had absorbed most of the shelved association gain. The rungs were measured in different regimes, and base absorption and fold leakage are not separated.
3. That the $0.14\times$ to $0.9\times$ edge transfer band is a usable coefficient. It is a historical range, not a controlled measurement.

### Open Questions

1. What is the label-oracle ceiling of the objective I am optimizing now? This question is cheap, and I intend to make it a precondition for any new optimization loop.
2. If the board is the objective again, what keeps it from being fitted the way section 0 describes?
3. Does the division gain reach the board undamped, as the record of 08-28 expected and the two rounded pairs of Notes 3–4 hint, or does it damp like the edge channel? The submission out on 2026-08-28 is one observation, not a rate.
4. Is the graph universe the verifier was fitted in the one the shipped notebook produces? It was fitted on the 199 comparator graphs, which the project called the deployed chain; I have not checked that the notebook's own pipeline produces the same graphs.
5. Does the verifier generalize across embryos? With two embryo domains and about 80 labeled positives, "prefix-pure" is two folds and carries no statistical guarantee.
6. How much of the division ceiling is reachable, given a candidate generator that covers 75 of 151 events?

---

## Closing

The uncomfortable part of this window is not that the experiments failed.
It is that they were correct.
An audit of any single experiment from that fortnight would have found nothing to object to.
That is why nothing stopped: each day produced a defensible result and a closed question, and no gate asked the question one level up.

I do not think the rule at the start of this window was wrong about the board.
The board really is a rounded, partial statistic that will absorb any number of submissions and give back adaptation.
What was wrong was treating one frozen comparator as if it inherited the authority of that principle without measuring what it could pay.
A criterion is not worth optimizing because the reasoning that produced it was sound.
It is worth optimizing when its ceiling exceeds the gap it is being used to close.

Two things changed on 2026-08-28.
A rule put the board back in charge, as a reaction to twenty-three days of standing still.
And a division verifier was chosen by a measured ceiling, before the board had said anything.
The local evidence behind the second is what this window leaves.
Over the comparator's own actions, perfect knowledge was worth $+0.000365$.
In the division channel it was worth $+0.0557$, and a verifier fitted out of fold took $+0.0053$ of it in one day, with its limits already measured: ranking transfer across embryos, and a candidate generator that reaches half the events.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
- **Part 5: Optimizing an Objective That Could Not Reach Gold**
