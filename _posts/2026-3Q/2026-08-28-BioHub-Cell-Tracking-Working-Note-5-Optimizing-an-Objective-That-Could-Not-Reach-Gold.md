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
- Korean version: [BioHub Cell Tracking 작업 기록 5: 금메달권에 닿을 수 없던 목적함수]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-Optimizing-an-Objective-That-Could-Not-Reach-Gold-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 6: The Universe We Were Selecting In]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-The-Universe-We-Were-Selecting-In/)

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

Note 4 ended with an evaluation contract that could no longer leak: every fold now held out a whole embryo.
Measured that way, the gains that remained were small, and the question left over was whether anything inside the objective could still move it.

This note covers 2026-08-17 to 2026-08-28.
A fold leak had just made August's largest local numbers unreliable, so the first priority was an objective that could neither leak nor drift.
I froze one: a local comparator, one graph over all 199 training movies, each movie predicted by models that never saw its embryo, scored by the official metric at $0.6013708666$.
A candidate could change that graph only through a fixed set of allowed edits, its action space, and the leaderboard stayed out of the selection loop.

The comparator never leaked and never drifted, and it barely moved: nearly every experiment came back at exactly zero or at the fifth decimal, and the one clear positive, $+0.0006686$, read on the board the same score as the model it replaced.

On 2026-08-24 I asked how much any candidate could pay: if the ground-truth labels chose every allowed edit, what would the comparator read?
The answer, $+0.000365$, is the center of this note: less than the board can display, and far from the gap to the gold band.
An objective can be clean and still have no room in it, and the two properties have to be measured separately.

On the last day of the window I reviewed the project and put the board back as the objective.
The same day, in the division channel, the tenth of the score that rewards finding cell divisions, the ceiling was measured before anything was optimized: $+0.0557$.
A verifier fitted out of fold took $+0.0053$ of it within the day and went to the board, which had not answered when this window closed.

The short version is:

```text
After Note 4's fold leak, a frozen embryo-out comparator became the objective.
Experiments against it were exact nulls or tiny; the one positive (+0.0006686) tied on the board.
On 08-24 its whole action space, with labels choosing every edit, was worth +0.000365.
New clauses: measure an action space's ceiling first; a gate must be able to end in a decision.
The 08-28 reset made the board the objective; a shelved +0.0313 lever measured -0.0008 that day.
In the division channel the ceiling came first (+0.0557); a verifier reached +0.0053 and was submitted.
```

The note follows the window in order:

| Sections | Question |
|---|---|
| 0 | Why a frozen local comparator, after Note 4? |
| 1 | What did experiments against it return? |
| 2 | How much could its action space pay at all? |
| 3 | What could a readiness gate decide? |
| 4 | What did measurements outside the comparator show? |
| 5 | Why did the board come back on 08-28, and what was tested that day? |
| 6 | What happened where the ceiling was measured first? |
| 7 | Decision Log |
| 8 | What is established, and what remains open? |

---

## 0. Why the Objective Was a Frozen Local Comparator

### 0.1 The charter

At the start of this window the project's charter took the leaderboard out of the selection loop:

> Public leaderboard values are transfer observations, not the model-selection objective. Fold-pure local OOF and exact graph replay remain the promotion authority.

A second sentence rationed the board to at most one transfer submission per planning cycle, adding that many cycles should use none.
The reasons were concrete.
The board shows a 29% slice of a hidden test drawn from unseen embryos and rounds it to three decimals, so a displayed tie does not order two models.
The project had made about 160 submissions by then, many of them one-axis sweeps around a single operating point, and the best of many looks at a noisy, rounded, partial statistic fits the statistic, not the task.
The 08-10 program still named a Public score of $0.970$ as the outcome target, but selection ran on local evidence.

### 0.2 What a frozen comparator is

A comparator is a fixed local benchmark that every candidate is scored against, and this one was built around the evaluation clauses of Notes 2 to 4.
Each of the 199 training movies is predicted by models that never saw its embryo (embryo-out, the clause Note 4 added); the whole graph is built and scored by the official metric; and there is one reference number, $0.6013708666$, so no level is compared across baselines, the problem Note 3 had left with seven of them.
The graph is frozen: the models and most of the post-processing are fixed, and a candidate may only apply a restricted set of legal edits, the comparator's action space.
In Note 2's notation, with the deployed system written $\hat G=P_\theta(M_W(X))$, the objective was

$$
\theta^\star
=
\arg\max_{\theta\,\in\,\Theta_{\mathrm{frozen}}}
S_{\mathrm{OOF}}\!\left(\hat G_{\mathrm{frozen}} \oplus \mathcal A(\theta)\right),
\qquad
S_{\mathrm{OOF}}\!\left(\hat G_{\mathrm{frozen}}\right)=0.6013708666 ,
$$

where $\mathcal A(\theta)$ is the set of edits a policy with parameters $\theta$ selects.
The two training embryos are called prefixes below, after the first characters of their movie names; hold-in means scored on movies the models trained on.

After Note 4 this was the right thing to build first: it could not leak, drift, or be fitted to the board.
One property had not been measured: the ceiling of its action space, the best score reachable if every action were chosen with perfect knowledge.
Section 2 measures it.

### 0.3 Where the numbers come from

Besides the comparator, the note uses these instruments:

| instrument | movies | level |
|---|---|---:|
| the comparator's subset | the four distributed test movies (copies of training movies) | about $0.64$ |
| the deployed notebook's pipeline, hold-in | the same four | $0.8899$ |
| the deployed notebook's pipeline, embryo-disjoint detectors | the same four | about $0.81$ |
| a replay in a different node universe | 177 | near $0.69$ |
| one held-out movie | 1 | — |

Deltas are read inside each instrument, and levels are never compared across them.

The board returned three scores in the window: a deployment migration with no new model (08-17), a runtime probe with byte-identical output (08-20), and the release of the period's one local positive (08-22).
All three read $0.921$, the same score at the board's resolution as the 08-05 candidate.

---

## 1. What the Comparator Measured

Several results concern the track-fragment matcher, a learned model that relinks a broken track end to a successor one or two frames later.

| experiment | instrument | delta | observation |
|---|---|---:|---|
| additional detector seed, two-fold exact replay (about 36 GPU-hours) | 177-movie replay, base $\approx 0.69$ | $-0.0000325$ | folds $+0.0000560$ and $-0.0003967$ |
| matcher, one training-group weight $2.0\to4.0$ | 199-movie embryo-out | $-0.0000427$ | both prefixes negative |
| matcher with a track-history input | 199-movie embryo-out | $-0.0000451$ | passed every pre-registered check; closed |
| auxiliary-center-detector and gap-filling composition pilots | 199-movie embryo-out | $+0.0000032$, $-0.0003537$ | neither passed its gates |
| association rank with a candidate filter applied first | one held-out movie | $0.0$ exactly | $0.8277207263$ in both arms |
| three local-representation families | comparator, four-movie subset | $0.0$ exactly, three times | zero legal actions (section 1.3) |
| **matcher, retrained independently at a fixed checkpoint** | 199-movie embryo-out, pre-matcher graph | $\mathbf{+0.0006686}$ | 34 movies improved, 9 worsened, 156 tied |

The last row was measured against the out-of-fold graph before any fragment matcher ran ($0.6006730$), and the weight row used its arm as control; that line set the comparator's level.
The detector-seed row lives in a universe of its own.

### 1.1 A second detector seed, at the fifth decimal

The window's largest compute item asked whether one more detector seed would change the graph enough to matter.
A two-fold exact-graph replay answered it, with one fold measured at 12.78 hours of training and the other projected at roughly 23.
The 177-movie incumbent moved from $0.6925340486$ to $0.6925015848$, a delta of $-0.0000325$, with the folds split in sign.
My first summary on 08-28 quoted $+5.6\times10^{-5}$, which is the positive fold alone; the aggregate is the number.

### 1.2 The one positive, and its transfer check

The period's one positive of any size was the matcher itself, retrained independently at a fixed checkpoint.
Against the pre-matcher graph it measured $+0.0006685607$: $+0.0014526$ on one prefix and $+0.0005501$ on the other, across 3,207 edits that kept the graph a valid lineage, with the division false-positive count unchanged.
Two successors added correlation inputs and about $0.00003$ more; the last was released through a private notebook that had to reproduce the local replay's prediction bytes, and its score became the comparator.

It went to the board as v78 (submitted notebook versions are named vN from here on) on 08-22, for the reason the board keeps any role at all.
A local criterion can be wrong in ways it cannot see from inside, and the board is the only window onto unseen embryos, so a candidate that advances locally is shipped once as a check that its gain survives there.
A difference of $0.002$ or less reads as a tie: no failure detected, never a confirmation.

v78 read $0.921$, a tie, and by the arithmetic of section 2.3 that was the likeliest outcome: at the top of the recorded transfer band, $+0.00067$ locally maps to about $+0.0006$ on the board; at the bottom, to $+0.00009$.
The tie is evidence neither for the local result nor against it.
What it established is narrower: a local gain below $10^{-3}$ is not reliably visible on the board, which made the size of the space itself the next question.

### 1.3 Three representations, zero actions

Three local-representation families were built to give the fixed-node action policy a better matching signal: a three-dimensional phase correlation, a learned dense-descriptor cost volume, and a soft-ternary census cost volume.
The question was whether the matching signal limited the policy.
All three passed every pre-registered check, selected zero legal actions in both prefixes, and returned the frozen graph's four-movie score, $0.6391791890184004$, to sixteen digits.

Three unrelated representations landing on the same exact zero is an answer: the signal was not what bound the policy.
The queue was declared blocked, since no other representation was ready under the same policy, and the scope of that verdict is the finding.
The policy was the frozen half of every comparison, so no comparison of this kind could produce evidence about it; that question had to go to the action space directly.

### 1.4 The level was real

One negative is worth keeping for the opposite reason.
A classical baseline, full-$Z$ Otsu thresholding with adjacent-frame Hungarian linking, went through the same four movies and scorer.
It scored $0.4626$ against the frozen learned graph's $0.6392$, a gap of $-0.1766$.
The learned stack's level was real; what was small was the room for increments inside this objective.

---

## 2. How Much Could the Comparator Pay?

### 2.1 The oracle ceiling

By 08-24 the pattern of section 1 was clear, so I asked the question one level up: not "does this candidate improve the comparator?", but "what is the most any candidate could improve it?"
The tool is an oracle ceiling: let the ground-truth labels make every choice the policy is allowed to make, and read the score.
No real policy can beat it inside the same action space.

Take the 3,178 actions the comparator selected across the 199 movies.
Keep an action only if the labels say it strictly helps (no true edge lost, no false edge added, and at least one of the two strictly better), drop it otherwise, and rescore.
This is a filter over the comparator's own selections, not a policy anyone could fit:

$$
S_{\mathrm{OOF}}:\;0.6013708666 \;\longrightarrow\; 0.6017363254,
\qquad
\Delta^{\mathrm{oracle}}_{\max}=+0.0003654588 .
$$

All of it came from one embryo prefix; the other could not be moved at all.

### 2.2 The same ceiling from the other side

A second measurement looked at the damage in the graph instead of the policy's actions, so it is independent of the first.
A replay of the comparator's candidate generation recovered 58,427 candidate edges, the complete inventory of edge swaps it could make without adding a node.
It then searched that inventory for each of the 38,060 ground-truth edges the graph was missing.
Only 183 had any replacement at all, and at most 115, 0.30% of the missing edges, were topology-legal swaps.

![Bars: 38,060 missing ground-truth edges, 183 with any replacement, at most 115 topology-legal swaps]({{ site.baseurl }}/assets/img/posts/2026-08-28-biohub-working-note-5/fig-01-edge-swap-inventory.png)
_Figure 1. Of $38{,}060$ ground-truth edges missing from the comparator graph, $183$ had any replacement in the complete swap inventory and at most $115$ were topology-legal swaps. Division edits are outside this count._

The two measurements agree from opposite directions: for edge repair by swapping, the graph did not contain the pieces it needed, whatever the ranker, representation or calibration.
Division edits were outside both counts; section 6 returns to them.

### 2.3 Pricing the ceiling against the board

The competition's only visible view was the board, where the distance to the gold band was measured in hundredths.
For the edge channel the project had one recorded transfer band: on 08-06, local edge deltas had been recorded as reaching the board at between $0.14\times$ and $0.9\times$ their local size.
That is a historical range across heterogeneous replays, not a controlled coefficient; Note 4's standalone association gain ($+0.0073$ locally) tied its forward control on the board, so the band is at best an optimistic bound.
Taken loosely, it sets a visibility condition, since a change must reach half a display unit to have any chance of showing:

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
Half of one display unit: +0.0005. The gap to gold: hundredths.
```

### 2.4 What changed: measure the ceiling first

The comparator had been chosen for being clean, and it was.
Cleanliness and headroom are separate properties, though, and the charter had asked only about the first.
The ceiling took less than a day to compute, and it bounds every experiment in section 1, including those that ran before it.
So the criterion gained a clause:

```text
C10. Measure the ceiling of an action space before optimizing inside it.
```

It is cheap, and it applies to any objective, including one the board supplies.
Both measurements pointed away from edge swaps over fixed nodes: toward the node universe, and, four days later, toward the division channel.

---

## 3. What a Readiness Gate Could Decide

### 3.1 The machinery, and what a question cost

The rule of section 0 was enforced by machinery built during Note 4's period: predeclared experiment cards, gates a run had to pass before it could start or count, immutable receipts, and a separate module tree for each experiment.
Its founding reason was sound: Note 4 had seen a notebook run write every output while recording its own failure in a field nothing read, and a process exit code is not a scientific decision.

In this window its cost per question became measurable.
An experiment took about sixteen new files, even one that changed a single axis.
On 08-20 and 08-21 a field-name mismatch between two record formats held a 331-task plan at zero, and checking three adapters that ran nothing re-read about 696 GB.
A one-line field-name fix forced an association experiment to relaunch under a new name, so the full pipeline ran twice to return $0.8277207263$ in both arms.
Each rule was defensible on its own; together they made a question cost more than its answer was likely to be worth.
The sharper case was a gate that could not end in a decision.

### 3.2 A native detector behind readiness floors

After the oracle the plan turned to the node universe, where section 2 pointed.
The candidate was a native spot detector, gated behind readiness floors written in advance, such as held-out detector recall of at least $0.50$ at $7\,\mu\mathrm{m}$, so that a detector unable to find cells would not spend a scoring run.
It ran twice, at 64 and then 256 sampler draws per epoch.
Recall was $0.179$ in one held-out fold and $0.036$ in the other, zero of 15 decoder settings were jointly feasible in either fold, and no movie was scored.

Under the rules, a readiness failure was explicitly neither a score result nor a closure of the model family, so the outcome could neither launch the family nor close it.

The hardware shows the same state.
At 2026-08-28 00:02 UTC the RTX 5090 was at 0% utilization while runs waited for admission.
The 08-28 review estimated that the twenty-day GPU program had used roughly 60 to 70 productive hours of a planned 480, about 15%; no per-job ledger exists to confirm it.
Runs needed admission, and the gates that granted it could hold a family without deciding anything about it.

### 3.3 What changed: a gate has to be able to decide

A gate earns its cost when each of its outcomes changes what happens next: a pass launches the next step, and a fail closes the line or names what to fix.
A gate whose failure is evidence neither for nor against a family protects the process from information rather than from error.

```text
C11. A gate must be able to end in a decision.
```

On 08-28 the readiness machinery was retired with the rest of the apparatus (section 5).

---

## 4. Looking Outside the Comparator

If the comparator's action space was nearly empty, where was the room?
Three measurements looked outside it.

### 4.1 External models on our frozen nodes

On 08-20 four externally trained models were adapted onto our frozen nodes and topology, so that only their association or repair decisions could differ.
On the four-movie subset, a forward-acceleration lookahead found zero eligible switches ($0.0$), an exact-coordinate detector adapter scored $-0.0044123$, a learned movement-field model $-0.0017270$, and a 4-D convolutional adapter failed closed in $0.324$ seconds.
Every fixed-node import was negative or inert, consistent with section 2: the substrate was binding, not the association rule.

### 4.2 A node residual, found and not measured

The same day, a scratch detector trained for 40 epochs on an inner fold of one embryo produced 11,594 proposals on one held-out movie; 1,190 lay farther than $7\,\mu\mathrm{m}$ from any node we had.
The line stopped there because no adapter for the detector's output format had been declared in advance, so whether those proposals were real cells was never measured: the pattern of section 3, a stop that decided nothing about the signal.

### 4.3 Where the difference was

On 08-26 I compared our graphs with an independently built reference pipeline on the four locally scorable movies, twice.
Against the reference's native build, the reference carried 53,858 nodes ours lacked and ours carried 59,382 it lacked; among those extra nodes the reference recovered 103 annotated cells to our 19.
Against a second build of the reference, with about 14,000 exclusive nodes on each side, the edge Jaccard over the 106,543 shared nodes was $0.9978$, and the reference emitted 384 parents with two children to our 37.

On shared nodes the two association results nearly agree; the difference is which nodes exist, and how many divisions are emitted.
This is an inference from our own diagnostics on four movies, not a measurement of anyone's score.
It also explains section 1.3: new association representations could not help, because association over shared nodes was already the part that worked.

---

## 5. The Reset of 2026-08-28

### 5.1 The review and the six rules

By 08-28 twenty-three days had passed without a transfer check the board could read.
The board had been asked three times in this window, against an allowance of five submissions a day, and none of the three carried a local gain it could resolve; section 2 had shown why none was coming.
The objective being optimized could not produce a candidate worth checking.
That, with the idle hardware of section 3.2, is why I reviewed the whole project on 08-28.
The review read the period in four findings: the frozen comparator, not the hidden-set score, had become the quantity being maximized, and section 2 had shown its ceiling; the checking machinery cost more per question than the questions were worth (section 3.1); GPU time went unused while runs waited for admission (section 3.2); and the largest local numbers on record had not been re-measured against the deployed pipeline (section 5.3).

The apparatus of section 3 was retired.
Six rules replaced it: the public board is the objective and local out-of-fold is a selection tool; no governance code; the GPU never idles while work exists; two hours from a new idea to the first prediction-facing artifact; a failed run is retried under the same name after the bug is fixed; closed results are never re-derived.
The plan written with them set out to reach the gold band on the board by the deadline, and it counted up to five submissions a day as nearly free transfer evidence.

### 5.2 What the first rule answers, and what it leaves open

The first rule is an anti-stall rule.
After twenty-three days the board was the only instrument on the competition's target, and the rule asks every experiment a question the comparator never asked: could the board ever see this?
It leaves two things open: it supplies no ceiling, though C10 applies to the board as much as to the comparator, and it does not say what keeps the board from being fitted the way section 0 describes once submissions are counted as nearly free.
The candidate that left for the board that same day was chosen on local evidence (section 6).

### 5.3 The shelved composition, measured the same day

The fourth finding put a shelved composition at the top of the new plan: Note 4's joint lineage-action mix, whose replay evidence read $+0.0144$ and then $+0.0313$, projected at $+0.005$ to $+0.025$ on the board.
It ranked first because these were the largest local numbers the project had recorded, and it needed a test first because they came from the four-fold research replay (base near $0.74$), whose folds Note 4 had found mixing both embryos.
So the first measurement under the new rules put that lever against the deployed pipeline (the kernel: the submitted Kaggle notebook as it runs), in three A/B harnesses on the four locally scorable movies:

| rung | regime | control | treatment | delta |
|---|---|---:|---:|---:|
| 1 | hold-in, control not faithful to the deployed kernel | $0.8831$ | $0.8842$ | $+0.0011$ |
| 2 | hold-in, control reproduces the deployed kernel exactly | $0.8866$ | $0.8858$ | $-0.0008$ |
| 3 | embryo-disjoint detectors, four-fold association heads | $0.8064$ | $0.8129$ | $+0.0065$ |

Rung 1's control sat below the deployed kernel's own $0.8899$, weaker than the pipeline it stood in for, so it does not count.
Rung 2, the faithful hold-in measurement, is slightly negative.
Rung 3 used embryo-disjoint detectors and is positive, with all four movies non-negative, but it still scored the lever with the prefix-mixed four-fold association heads, so it is not an embryo-out test of the lever.
Against the faithful deployed base the lever measured $-0.0008$: the plan's first item was falsified on the day it was written.

The general reading is that a lever's value, $\Delta_L(B)=S(L\circ B)-S(B)$, belongs to the pair of lever and base, not to the lever alone.
Since the shelved measurement the deployed base had gained test-time augmentation, Note 4's reverse-time harmonic fusion, consensus and a tuned edge threshold.
These rungs do not separate that base absorption from fold leakage; the "about 75% absorbed" I wrote down that day compares deltas from three universes and is an estimate, not a decomposition.

---

## 6. The Division Channel, Ceiling First

### 6.1 Why division, and what its structure looked like

The plan's second item was a division recoverability audit.
Division carries a tenth of the score, $S = J^{\mathrm{adjusted}}_{\mathrm{edge}} + 0.1\,J_{\mathrm{division}}$, and for weeks the channel had been treated as a guard not to disturb, because the one edit tried in it had cost about $-0.00018$.
Two readings pointed at it: in Notes 3 and 4 the board had read the division stage as a step of four thousandths where the replay credited about one, and section 4.3 had found the reference emitting 384 two-child parents to our 37.
So, as C10 asks, the channel's structure and ceiling were measured first, on the 199-movie embryo-out graphs (a fork is a node with two outgoing edges, a predicted division):

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

### 6.2 A gate that selected the complement of the truth

Recoverable true daughter pairs separate at a median of $10.7\,\mu\mathrm{m}$; the forks the scorer counted as false separate at a median of $5.4\,\mu\mathrm{m}$.
The deployed acceptance rule admitted a fork only when the two putative sisters were within $8.5\,\mu\mathrm{m}$ of each other and each was within $4.66\,\mu\mathrm{m}$ of the parent.
Its sister-distance limit admitted 96.3% of the counted false forks and 29.1% of the true divisions.

The gate was not badly tuned.
It was selecting the geometric complement of the truth: tight pairs, mostly false forks, passed; wide pairs, mostly real divisions, did not.

### 6.3 The ceiling

With the geometry understood, the division stage was rebuilt in oracle mode across all 199 graphs: a daughter could be moved to a new parent (reparenting), and forks were chosen by the labels.

| arm | score | delta vs $0.6014$ | division detail |
|---|---:|---:|---|
| fork-free base | $0.6005$ | $-0.0009$ | prices the whole deployed division stage at $+0.0009$ |
| deployed strict stage | $0.6014$ | — | the incumbent |
| ground-truth oracle, reparenting allowed | $\mathbf{0.6571}$ | $\mathbf{+0.0557}$ | $J_{\mathrm{div}}=0.5577$; 87 TP, 5 FP, 64 FN; 107 forks added, 37 by reparenting |

Compare that with section 2, on the same 199 movies with the same scorer.
Perfect knowledge over the frozen comparator's selected actions was worth $+0.000365$.
In the division channel it was worth $+0.0557$, two orders of magnitude more, and this time the ceiling was known before anything was optimized.

![Log-scale comparison of the comparator ceiling (+0.000365) and the division ceiling (+0.0557) with measured gains]({{ site.baseurl }}/assets/img/posts/2026-08-28-biohub-working-note-5/fig-02-two-ceilings.png)
_Figure 2. The same scorer on the same 199 movies. A ground-truth filter over every action the comparator selected was worth less than the local size the board could show; the division stage's ceiling was two orders of magnitude larger. The shaded band is arithmetic from a historical transfer range, not a measurement._

### 6.4 A verifier inside the ceiling

A learned verifier was then fitted to rank the reparent-enabled candidates, out of fold and prefix-pure: the ranker applied to each embryo was trained only on the other one.
In one day it went from a linear model on geometry ($+0.0015$, 6 of the oracle's 87 true positives) to a gradient-boosted ranker with track features ($+0.0035$) and then with appearance features ($+0.0053$, a score of $0.6067$, 11 true positives and 26 false positives).

The same day measured two limits.
On the embryo it was trained on, each ranker's simulated division Jaccard was $0.43$ and $0.67$; applied to the other embryo, the pooled figure was $0.062$, so the loss was in ranking transfer across embryos, not in the threshold.
And the candidate generator covered only 75 of the 151 events, a second ceiling underneath the first.

### 6.5 The local gate, and the transfer check

The order that day was ceiling first, optimization inside it second, and a local gate third.
A predeclared hold-in gate checked that division edits would not damage the edge channel: the candidate kernel reproduced the previous kernel's four-movie hold-in score of $0.8899$ with identical per-movie edge counts, and all 45 of its division edits landed away from the sparse annotated tracks the scorer checks, a property of this check rather than a general license.

The candidate, v79, went to the board as a transfer check late on 08-28, for the reason of section 1.2 with more riding on it: the largest local gain of the window, in a channel the replay had underpriced before.
The only expectation written that day was that the gain would reach the board at a similar order, not damped like edge deltas.
The record behind it points two ways.
The edge band of section 2.3 would put a local $+0.0053$ anywhere from about $+0.0007$ to $+0.005$ on the board, which is arithmetic; the two rounded division pairs of Notes 3–4, $+0.000949$ and $+0.0010142$ locally with a $+0.004$ step on the board each, are a direction, not a rate.
Whatever the board returns is one observation of how the division channel transfers, and a clear drop would falsify the candidate.
When this window closed, it had not answered.

---

## 7. Decision Log

For most of this period the rule in force was the charter of section 0: a frozen embryo-out comparator was the selection objective, and the board was a sparse transfer observation, with a Public $0.970$ target on paper.
The board was asked three times; one was a transfer check of a locally advanced candidate, and it tied, as the arithmetic of section 2.3 said it most likely would.
On 08-24 the comparator's own ceiling was measured, and on 08-28 I made the board the objective as an anti-stall rule, then sent a division candidate chosen on local evidence as the period's second transfer check.

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| Make a frozen embryo-out comparator ($0.6013708666$) the objective | after Note 4's fold leak, an objective that could not leak, drift or be fitted to the board | clean measurements; nearly every delta a null or below $10^{-4}$ | the question of what it could pay |
| Build three new representations for the fixed-node policy | did the matching signal limit the policy? | zero legal actions, three times | the policy needed its own test |
| Ship the matcher gain ($+0.0006686$) as v78 (08-22) | transfer check: a local gain goes to the board once, so a blind spot of the local criterion can show | $0.921$, a tie | a local gain below $10^{-3}$ is not reliably visible on the board |
| Measure the action space's oracle ceiling (08-24) | after a run of nulls: was there room at all? | $+0.000365$; 115 of 38,060 missing edges swappable | C10 |
| Native detector behind readiness floors | the ceiling pointed at nodes; a detector that cannot find cells should not spend a scoring run | recall $0.179$ and $0.036$; neither a score nor a closure | C11 |
| Reset; the board becomes the objective (08-28) | no transfer check the board could read for twenty-three days, and a comparator that could not produce one | six rules; the process apparatus retired | open: what keeps the board from being fitted |
| Test the shelved $+0.0313$ composition (08-28) | the review's fourth finding; its folds flagged in Note 4 | $-0.0008$ against a faithful base | a lever's value belongs to lever and base together |
| Division: ceiling first, then a verifier; submit v79 (08-28) | C10 applied; the board had read the division stage in Notes 3–4 | ceiling $+0.0557$; verifier $+0.0053$; not yet answered | the next transfer check is pending |

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
| C10 **(new)** | Measure the ceiling of an action space before optimizing inside it | Note 5 |
| C11 **(new)** | A gate must be able to end in a decision | Note 5 |

The 08-28 rule that makes the board the objective sits in the log above, not in this table; how it fits with C9 is an open question.

---

## 8. What the Period Established

### Established

1. A ground-truth keep-or-drop filter over the frozen comparator's 3,178 selected actions was worth $+0.0003654588$ ($0.6013708666 \to 0.6017363254$), all of it in one embryo prefix.
2. Only 115 of 38,060 missing ground-truth edges (0.30%) had a topology-legal corrective swap in the frozen node universe.
3. The fragment-matcher line ($+0.0006686$ over the pre-matcher graph, both prefixes positive) returned the same public score at the board's resolution after a complete release chain.
4. Three unrelated representation families each selected zero legal actions and scored exactly the frozen graph's $0.6391791890184004$; every fixed-node import of an externally trained model was negative or inert.
5. Two runs of a native detector stopped at their readiness floors (held-out recall $0.179$ and $0.036$) with no movie scored, an outcome the rules counted as neither a result nor a closure.
6. In a harness that reproduces the deployed kernel exactly, the shelved association composition measured $-0.0008$ on the four hold-in movies.
7. Of 151 ground-truth divisions, 107 were fully recoverable from the embryo-out graphs and 101 already carried at least one required edge; 2 of 849 forks at annotated parents sat on a true division parent; the deployed sister-distance limit admitted 96.3% of counted false forks and 29.1% of true divisions.
8. A reparent-enabled oracle over the division channel was worth $+0.0557$ on the 199-movie embryo-out replay, and a prefix-pure verifier reached $+0.0053$ inside it in one day.

### Supported but Unconfirmed

1. That the remaining headroom lies in the node universe and the division channel rather than the association rule. It rests on two four-movie comparisons against a reference pipeline (section 4.3).
2. That the deployed machinery had absorbed most of the shelved association gain. The rungs were measured in different regimes, and base absorption and fold leakage are not separated.
3. That the $0.14\times$ to $0.9\times$ edge transfer band is a usable coefficient. It is a historical range, not a controlled measurement.

### Open Questions

1. What is the label-oracle ceiling of the objective being optimized now? C10 makes that measurement a precondition for any new optimization loop.
2. With the board as the objective, what keeps it from being fitted the way section 0 describes, and how does that rule fit with C9?
3. Does the division gain reach the board undamped, as the 08-28 expectation says, or damp like the edge channel? The submission out on 08-28 is one observation, not a rate.
4. Is the verifier's graph universe the one the shipped notebook produces? It was fitted on the 199 comparator graphs, which the project called the deployed chain; I have not checked that the notebook's own pipeline produces the same graphs.
5. Does the verifier generalize across embryos? With two embryo domains and about 80 labeled positives, "prefix-pure" is two folds and carries no statistical guarantee.
6. How much of the division ceiling is reachable with a candidate generator that covers 75 of 151 events?

---

## Closing

The comparator did what it was built to do after Note 4: it never leaked, never drifted, and never borrowed from the board.
What it could not do was pay: perfect knowledge over every action it selected was worth $+0.000365$, below what the board can display.

That is a finding about objectives, not about the board.
The charter was right that the board gives back adaptation for every look, but independence from the board does not give an objective headroom.
A criterion is worth optimizing when its ceiling exceeds the gap it is meant to close; that became C10, and gates that could hold a family without deciding anything became C11.

On 08-28 a rule put the board back as the objective, and the same day a division candidate was chosen by a measured ceiling, an out-of-fold verifier and a local gate.
On the same 199 movies and scorer, perfect knowledge in that channel was worth $+0.0557$, and the verifier took $+0.0053$ of it in one day, with its limits already measured: weak ranking transfer across embryos, and a candidate generator that reaches 75 of 151 events.

The division channel now has a measured ceiling; the hidden set has not yet answered.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- [Part 3: What the OOF Machine Refused]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused/)
- [Part 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)
- **Part 5: Optimizing an Objective That Could Not Reach Gold**
