---
title: "BioHub Cell Tracking Working Note 3: What the OOF Machine Refused"
date: 2026-07-31 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, oof, cross-fitting, division-recovery, deployment-constraints, working-note]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-07-31-biohub-working-note-3/cover.png
  alt: "Title card for BioHub Working Note 3: what the OOF machine refused"
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

# BioHub Cell Tracking Working Note 3: What the OOF Machine Refused

- Competition: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- Official metric notes: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Previous notes:
  - [Working Note 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
  - [Working Note 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- Korean version: [BioHub Cell Tracking 작업 기록 3: OOF 기계가 거절한 것들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 4: When the Largest Local Gain Hurt the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-When-the-Largest-Local-Gain-Hurt-the-Board/)

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

Note 2 ended with a decision rather than a result.
The public score had flattened near $0.902$, and one more submission could not separate three explanations of that plateau: correlated model errors, a calibration inherited from the wrong distribution, and adaptation to the visible leaderboard.
So I stopped moving parameters on the leaderboard and set out to build a different instrument.

This note calls it the OOF machine.
OOF stands for out-of-fold: every training movie is scored by a model that never trained on it.
The machine replays the notebook's graph stages exactly on those predictions and scores the result with the official metric, so it can ask what a graph edit (any change to which nodes and edges the lineage graph contains) does to the score on movies the models have not seen.

This note covers its first seventeen days, 07-15 to 07-31.
Its most useful property was that it could say no, and say why: it refused nine of its own candidates on grounds written down before each one ran, and each refusal named a mechanism.
A selector calibrated on one population fired on another; a better parent chooser built a worse graph; a better classifier inserted worse nodes.

What it approved was small: four policies at $+0.00027$, $+0.00063$, $+0.00095$ and $+0.00067$, each measured against a different local reference graph.
The public leaderboard (the board) rounds to three decimals, so every approval sat below what it can display.
That gap set the board's July role: answering questions the machine could not yet pose, and checking whether an approval survived on unseen embryos (a transfer check).

The question this note takes up is what a local criterion can do once it is strict enough to refuse, and what it still lacks before it can choose.

The short version is:

```text
On 07-15 the gates were written down before any result existed.
The machine then refused nine candidates, and each refusal named a mechanism.
Its four approvals were 10^-4 in size; the board shows 10^-3.
Which points exist was a board question in July; edge reweighting was the machine's.
A division approval, checked on the board with its control, moved the board by +0.004.
The strongest structural result returned no score in seven submissions.
The approvals sat on seven unreconciled reference graphs and cannot be ranked.
```

| Sections | Question |
|---|---|
| 0 | What was written down before the first result? |
| 1--2 | What are the hand-built graph stages worth on unseen movies, and what was the first promotion? |
| 3 | Why did questions about which points exist go to the board? |
| 4--5 | What did the machine refuse, and which mechanism did each refusal name? |
| 6--7 | What did the division channel show, and does deciding before the solver change an edit's value? |
| 8--9 | Why did the strongest result return no score, and what was everything measured against? |
| 10--11 | Which decisions did the period make, and what is established? |

---

## 0. Writing the Gates Down Before the First Result

On 2026-07-15, before running any of it, I wrote the program down: capture fold-correct out-of-fold predictions, compute exact error anatomy, build one minimal operator for the dominant recoverable failure, promote only embryo-stable positive operators, and only then train full-data models.
The same document fixed the promotion gates and a submission budget of three configurations plus two in reserve.

The official score is an edge agreement plus a small division term: an adjusted edge Jaccard, which also penalizes predicting more nodes than the movie's estimated cell count, plus one tenth of a division Jaccard, both counted only where annotators labeled.

The gates came first for a reason: a gate chosen after the result is seen can always be passed.
An operator $R$ was promoted only if

$$
\Delta S(R)>\delta_{0}
\;\wedge\;
\min_{p\in\{A,B\}}\Delta S_{p}(R)\ge 0
\;\wedge\;
\min_{k}\Delta S^{(k)}(R)\ge 0
\;\wedge\;
|R|\le n_{\max},
$$

where $\Delta S$ is the exact change in the official combined score after replaying the full deterministic pipeline, $p$ ranges over the two embryo prefixes (a movie's ID prefix names its embryo), $k$ over the outer cross-fit folds, and $|R|$ is the edit count.
Each check was recorded separately, so each refusal named its own reason; the last three did most of the refusing.

Underneath the program, the instrument changed.
The host patched an exploit in the division metric (scorer change dated 2026-07-18, full leaderboard rescore announced 2026-07-23), which makes Note 2's description of the division condition obsolete.
The patched condition is stricter and local: a parent-side node matched to a ground-truth node, a genuine fork there, two daughter branches that do not immediately merge again, and one-to-one assignment between predicted forks and ground-truth divisions.
Roughly $0.028$ came off the top of the board, so part of the gap Note 2 reasoned about was the old metric rather than a different method.
None of my submissions was scored both before and after the change, so whether my pipeline used the exploit cannot be measured; I infer from its shape that it did not, and read my July scores as one scale.

---

## 1. What the Hand-Built Graph Stages Are Worth on Unseen Movies

Note 2's first question for strict OOF was what the existing graph stages are worth on held-out data, and I wanted that number before replacing any of them.
On 07-17 I replayed the deterministic graph stages of my submitted notebook exactly (ILP selection, motion reassignment, short-component pruning, one-frame gap recovery, safe division repair) on the epoch-$100$ twofold out-of-fold predictions, and scored the result with the official scorer.
The ILP is the solver: an integer linear program that chooses which candidate nodes and edges form the graph.

| quantity | raw model graph | after exact replay | delta |
|---|---:|---:|---:|
| official score | $0.647453$ | $0.699640$ | $+0.052187$ |
| node recall | $0.916820$ | $0.911433$ | $-0.005387$ |
| movies improved | — | $183$ of $199$ | — |

The hand-built post-processing is worth $+0.052$ on held-out movies: not over-fitted to the leaderboard, and not the bottleneck.
The node-recall row shows how it earns that: the stages trade nodes away and buy edges with them, an asymmetry that recurs through this note.

Of roughly $25{,}150$ remaining edge false negatives (FNs), $11{,}020$ had **both endpoint nodes unmatched**.

```text
An edge FN between two matched nodes is an association error.
An edge FN between two unmatched nodes is a detection error
wearing an association error's clothes.
No threshold, no ranker and no gap policy can reach the second kind.
```

That one number pointed the program at a question it could not yet measure: which points exist.

A side experiment in the same report, a short-track rescue, tested an idea carried since Note 1: that restoring nodes helps, because node recall is part of the score.
It restored $14{,}917$ nodes and raised node recall by $+0.00186$; the official score fell by $0.000735$, and $156$ of $199$ movies got worse.
Under the node-count adjustment, a restored node that attracts no true edge is a cost in the edge term and in the count ratio at once.

---

## 2. The First Promotion With Every Stage Separated

On 07-19 the out-of-fold edge-replacement ranker finished a nested five-fold cross-fit at the fixed $200$-epoch capture.
Nested means that fitting the policy, calibrating its threshold and the final evaluation each sat on disjoint movie sets; the thresholds were fixed to fold medians, so no test movie needs a fold identity to be scored.
The nested figure was $+0.0002706$ from $1{,}447$ replacements, with both embryo prefixes positive.

It was the project's first decision with every stage separated from every other, and at $2.7\times10^{-4}$ it was below the resolution of the only external instrument I had.

---

## 3. Which Points Exist: Why July's Questions Went to the Board

In the third week of July the machine still ranked edits to a fixed point set: its captures froze the detections.
A change to the detection field changes the vertices of the graph, so until a replay that re-extracts the points arrived later in the window (section 5), only the board could answer a question about which points exist.

### 3.1 The Association Question

The first slots of this window went to a question Note 2 had left open: whether a second, independently seeded model adds information that a jointly calibrated mixture can use.
From 07-20 through 07-23 seventeen submissions tested it through the association channel, with the mixing rule as the variable.
Logit mixtures, margin-adaptive mixing, consensus gates and basin searches all read between $0.905$ and $0.908$ in the submission record; my log reads one of them $0.909$.

The right denominator is the strongest single-seed graph, which my log puts at about $0.909$ around 07-19/20.
Against it, no association variant was better at the board's resolution, and the weakest read $0.003$ to $0.004$ lower.
These candidates reweighted edges on a fixed point set, which the machine can measure at $10^{-4}$; the conclusion was a division of labor, with edge reweighting belonging to the machine.
The step from the $0.902$–$0.903$ band where Note 2 ended to that $0.909$, about $+0.006$, is attributable to no experiment in this window.

### 3.2 The Field Average and Its Ablation

On 07-24 one candidate averaged the two aligned **detection logit fields before peak extraction**, so both models contributed to which points exist rather than to which points connect.
Since only the board could answer that, the candidate went in on the same day as its ablation and its brackets, which let the answer be attributed.

| candidate | change | public |
|---|---|---:|
| field average before peak extraction | balanced detection field, dual-seed association retained | $0.911$ |
| **balanced field, primary-only association** | **the ablation** | $0.910$ |
| primary-weighted field | off-balance toward the primary seed | $0.909$ |
| secondary-weighted field | off-balance toward the second seed | $0.907$ |

The ablation is the informative row: with the dual-seed association removed, the balanced field read the same score at the board's resolution.
At the board's resolution, removing the dual-seed association changed nothing, so whatever the second seed contributed, the board could see it only through the shared node field; a direction, carried as unconfirmed (section 11).

A slightly primary-leaning field (secondary detection weight $0.475$ instead of $0.5$, same threshold) read $0.912$ on 07-25, the same score as the balanced field at the board's resolution.
It became the frozen reference for most of the out-of-fold replays that followed; the board could not separate it from the balanced field, so nothing below depends on it being the better of the two.
Against the approximate single-seed figure it reads about $+0.003$, at the edge of the board's resolution: a direction, not a measured gain.
A two-sided probe of the detection threshold read the same score on both sides, closing that axis as flat rather than bracketed.

### 3.3 The Submission Budget

The 07-15 budget of three configurations plus two in reserve fits a program in which the board sees what the machine has promoted.
July's node-field questions had no local instrument, so the board served as one.
A budget sized for promoted operators did not fit that use; the gates were checked on every operator, the budget was not, and it lapsed without being amended.
There were $109$ scored submissions that month, and what the board is for, once the machine can measure more, became a question for August.

---

## 4. Four Refusals and the Mechanisms They Named

Four of the nine refusals share a pattern: each candidate had a plausible mechanism and an improving component metric, and a gate written before it ran showed why that metric and the graph disagreed.

### 4.1 A Selector Calibrated Where Labels Exist Fired Where They Do Not

**The case for it.** A dual-seed association selector was strictly model-out-of-fold, cross-fitted on top, calibrated on $177$ beneficial against $43$ harmful labeled groups, and clean under a leakage audit.
A group is one cell's candidate parents; firing means swapping in the second seed's parent.

**What the replay measured.** It was calibrated where labels exist and deployed where they mostly do not:

$$
\pi_{\mathrm{cal}}
=\frac{249}{122{,}007}
=0.204\%,
\qquad
\pi_{\mathrm{dep}}
=\frac{15{,}062}{276{,}077}
=5.456\%,
\qquad
\frac{\pi_{\mathrm{dep}}}{\pi_{\mathrm{cal}}}
\approx 26.7 .
$$

The exact replay lost $0.000068190$ and worsened $112$ of $199$ movies.
No post-hoc fold subset cleared the floor, and none could be deployed, since a test movie carries no fold identity.
To move the official edge counts by $+16$ true positives (TP), $+26$ false positives (FP) and $-16$ FN, the selector changed $3{,}001$ raw nodes and $3{,}450$ raw edges: the annotated skeleton it was scored on is a thin slice of the graph it rewrites.

**What it established.** The selector had no leakage; it had the wrong population, which a leakage audit cannot see.
Comparing its firing rate on the calibration and deployment populations can: a rule whose rate changes by more than a small factor between them is not calibrated for where it will act.
That became clause C3: calibrate a rule on the population it will act on.

### 4.2 Picking the Right Parent More Often Did Not Build a Better Graph

**The case for it.** A model that picks the correct parent more often on held-out movies should reconstruct a better graph.
Two models tested that within four days.

| model | component metric | exact graph replay |
|---|---|---|
| temporal-flow gate (switches to a motion model's parent) | parent top-1 $0.867186 \to 0.879198$ ($+0.012012$); five of five outer folds nonnegative | $-0.0007833$; $63$ improved / $132$ worsened |
| higher-order parent matcher (compares candidate parents jointly) | parent top-1 $0.867589 \to 0.883202$ ($+0.015613$) | $+0.000482$; one outer fold at $-0.000788$; one prefix at $-0.000280$ |

**What the replay measured.** The flow gate reversed sign: a component metric that improved on every fold produced a graph that was worse on two movies out of three.
The higher-order matcher stayed positive, but a top-1 gain of $+0.0156$ arrived as $+0.00048$ of graph score, an attenuation of about $32\times$.

**The mechanism.** The flow gate was leak-proof; its loss happens downstream.
Changing which parent a node attaches to changes which connected components exist, and so what short-component pruning removes and what gap recovery may bridge: a locally correct switch can delete a component that carried several correct edges, and the parent-choice metric never sees the component.
The gate also raised node recall by $+0.001145$ while lowering the score, the same trade as the short-track rescue.

**Checking the refusal.** The higher-order matcher failed its fold and prefix gates and was not promoted.
A refusal is also a prediction, that the policy does not help on unseen embryos, and this one rested on a mixed picture (positive in aggregate, negative in one fold and one prefix), so I submitted three variants as probes of it.
The board read the two broad arms $0.004$ lower than the $0.916$ division configuration they were built on, a large move in the direction the local verdict predicted and consistent with the refusal.

Together the two models became clause C4: judge a component by the graph it produces, in an exact replay, not by its own accuracy.

### 4.3 The Better Classifier Inserted the Worse Nodes

**The case for it.** Of two versions of the candidate-proposal model, which proposes cells the detector missed, the one that classifies better should insert better nodes.
The predeclared ensemble beat the single model on every offline metric I had: known-label AP $0.9999984$ against $0.9999973$, and recovery of the frozen graph's node misses $0.53075$ against $0.52633$.

**What the replay measured.** After insertion into the finished graph, the single model was the better one: $+0.0000506$ from $842$ repairs against the ensemble's $+0.0000435$ from $845$.
Neither was promoted on its own merit, and with one run and a gap of $7\times10^{-6}$ I hold the ordering as a warning rather than a law.
Classification quality and intervention utility are different objectives; this is C4 from a different direction.

Why the family was weak shows in one diagnostic: two-sided bridge repairs, which insert a node only where it links to existing fragments on both sides, produced zero repairs out of $1{,}354$ candidate triples.
The missing nodes are not one-frame holes but regions where the detector fails over several frames at once.

---

## 5. Four More Refusals, Four More Mechanisms

Four more refusals each named a different mechanism; the ninth, a third division seed, is in section 6.

**A per-embryo detector did not hold across folds.**
A full out-of-fold detection grid, which re-extracts the points for every weight, showed one embryo improving as the secondary detection weight fell while the other weakened, so a per-embryo router looked as if it should pay.
A nested cross-fit inside each embryo rejected it at $-0.0003550114$, both embryo-level deltas negative; the development folds chose $0.40$ twice and the baseline twice in one embryo, and the opposite extremes, $0.25$ and $0.70$, in the other.
The reciprocal cross-fit was $-0.0003884963$.
The two model folds coincide with the two embryo prefixes, so this cannot separate embryo identity from fold identity; the subgroup effect was real as a description, not stable enough to be a rule.

**Emitting forks is not how the division term gets paid.**
The frozen out-of-fold graph already held $12{,}794$ predicted binary forks, against $151$ annotated divisions in the whole training set, and scored $4$ division true positives against $720$ false positives: what was scarce was clean forks at a matched parent.
A cross-fitted validator that pruned the weaker daughter edge of low-ranked forks cut division false positives from $720$ to $317$ and reached $+0.000328$, but it cost $197$ edge true positives and two of four outer folds were negative.
Buying division precision with edge recall trades a term weighted at one tenth against a term weighted at one.

**The solver does not decide what a division is.**
To put divisions into the solver's objective, I added a hyperedge variable to the ILP, forced to equal the conjunction of two daughter edges and rewarded by an out-of-fold division rank score.
As the reward rose from $1.00$ to $2.00$, the solver selected $7$, then up to $43$ complete fork events.
The patched division counts stayed at exactly $5$ TP / $507$ FP / $146$ FN at every reward.

```text
The solver chose more forks.
The downstream graph filter removed every one of them.
In this pipeline, what counts as a division is decided after the solver.
```

That one fact about ordering closed a whole family of "put it in the objective" proposals.

**Keeping every hypothesis alive is not a plan.**
My own 07-28 survey (section 6) had ranked this idea first: keep several graph hypotheses alive and let a joint optimizer arbitrate.
Two days later a gate written before the run stopped it.
The union of the frozen graph, five other detection blends and the proposal model raised node recall from $0.943001$ to $0.965008$, but it recovered only $38.61\%$ of the frozen graph's node misses against a gate of $40\%$, and needed $25.98$ novel points per frame against a gate of $6$.

The residue explains the refusal: the union can recover $1{,}704$ missed ground-truth edges, but they sit inside $649{,}702$ candidates outside the frozen graph.
A correct added edge raises both numerator and denominator of the edge Jaccard $J$ and a wrong one only the denominator, so the break-even precision is $p^{*}=J/(1+J)=0.422315$ at $J=0.731046$.
Exactly one pattern of proposal sources clears it in both prefixes and all four folds, at $0.526814$, and one pattern is not a channel.
Three fixed rules for admitting new tracklets scored strongly positive on a sparse edge-utility proxy and raised node recall; their official deltas were $-0.003536$, $-0.002364$ and $-0.006235$.
That triple (proxy strongly positive, node recall up, official score down) appeared three times in this window, on unrelated operators, and is the most reliable failure signature I have.

---

## 6. The Division Channel: A Tenth of the Metric, Nearly Unused

On 07-28 a score-upside survey priced the error budget of the frozen graph across all $199$ movies:

| term | value |
|---|---:|
| official score | $0.725553$ |
| edge TP / FP / FN | $109{,}363$ / $20{,}715$ / $19{,}520$ |
| node recall | $0.9477$ |
| division TP / FP / FN | $4$ / $720$ / $147$ |

Of the edge false negatives, $5{,}455$ ($27.9\%$) still had both endpoints unmatched.
The division term contributes

$$
0.1\cdot J_{\mathrm{div}}
=0.1\cdot\frac{4}{4+720+147}
=0.00046
$$

out of the $0.1$ weight the metric reserves for it.
A tenth of the metric was delivering under five ten-thousandths while most of the compute went to the edge term, and that changed my priorities.

The same day, the strict-division rank ensemble passed every gate.
Two independently seeded division-event models were combined by a fixed equal-weight percentile rule, and only a small top fraction of ranked candidates, its action fraction, was added after the frozen graph.

| arm | outer cross-fit delta | folds |
|---|---:|---|
| seed A alone (the single-seed control) | $+0.000661$ | — |
| seed B alone | $+0.000834$ | — |
| **two-seed percentile ensemble** | $+0.000949$ | all four positive; the same action fraction ($0.016$) in every fold |
| three-seed ensemble | $+0.000697$ | action fractions $0.064$ / $0.032$ / $0.032$ / $0.024$ |

It recovered nine division true positives at a small edge-Jaccard cost.

**The transfer check.** A local $+0.000949$ is below the board's resolution, so why submit it?
It was the first approval in the division channel, and a local result never checked against unseen embryos can hide a blind spot of the local criterion.
The ensemble and its single-seed control went in together, so the board could attribute its answer.
Both read $0.916$, $+0.004$ over the $0.912$ configuration: the board registered the division channel and could not separate the ensemble from its control.

This is the window's one upward step that clears the board's resolution against a recorded reference, and it points the same way as the local result.
Its size is the finding: the replay priced the channel at $10^{-4}$, and no promoted edge lever of comparable local size moved the board upward (section 7).
I carried forward an observation, not a rate, since it rests on one rounded pair: the hidden set appears to weigh the division term far more than the local replay does.

**The third seed** was diverse (out-of-fold score correlations of about $0.69$ and $0.84$ against the two promoted models) and positive alone, yet the three-seed ensemble scored below the two-seed one.
Its folds chose three different action fractions, spanning a factor of nearly three, and a policy whose operating budget moves when its fitting data moves has not found an operating point.

---

## 7. Does Deciding Before the Solver Change an Edit's Value?

Note 2 recorded that graph edits are not additive, and this window priced it: the first pre-solver temporal-flow policy earned $+0.000365$ of adjusted edge, and one true division turned false ($-0.000153$ at one-tenth weight) consumed $42\%$ of that gain.
The value of an edit also depends on **where in the pipeline it is applied**.
Writing $\Pi$ for the solver,

$$
\Delta S\!\left(\Pi\circ R\right)
\ne
\Delta S\!\left(R\circ \Pi\right).
$$

The clean measurement is the candidate-proposal model: the same proposals, from the same weights, in two positions.
Inserted into the finished graph they were worth $+0.0000506$; appended to the point set before the solver, at $0.50$ nodes per frame, $+0.0006748$, about $13\times$ more.
The arms are not budget-matched, so this compares positions, not intervention counts.
What makes it causal is the construction: scores between the original points are restored exactly from the frozen capture, so a zero budget reproduces the frozen graph, verified to a maximum edge-probability delta of $5.96\times10^{-8}$.

| budget (nodes/frame) | delta | folds |
|---:|---:|---|
| $0.10$ | $+0.0000776$ | two negative |
| $0.25$ | $+0.0002702$ | one negative |
| $0.50$ | $+0.0006748$ | all four positive (promoted) |
| $0.75$ | $+0.0011047$ | all four nonnegative |
| $1.00$ | $+0.0012498$ | one fold at $-0.0006285$ |

The aggregate is monotone in the budget; the fold picture is not, as with the third division seed.

"Decide earlier" is a channel-specific result, not a principle.
The track-fragment matcher passed every gate after the solver, at $+0.0006325$ from $3{,}224$ edge replacements.
Its hidden-safe submission, a transfer check, read $0.912$, the score of the configuration it was added to: a tie, no failure detected.
Moved before the solver, its outer cross-fit was $+0.0000178$, indistinguishable from zero.
The two figures sit on reference graphs whose node recall differs by roughly six points, so I do not compare them as a ratio.
The division hyperedge of section 5 was moved earlier with no effect at all.

Earlier helps when the earlier stage can create options that did not exist, since a node never proposed cannot be linked later by any ranker; it does nothing when the later stage was already the better arbiter of options both can see.
Nodes belong to the first class; edge reweighting and division events, in this pipeline, to the second.

---

## 8. Seven Submissions That Returned No Score

Pre-solver activation (adding proposed points before the solver runs) was the strongest structural result of the month, so its transfer was the next question, asked at several activation budgets: on a curve monotone in aggregate but not across folds, which point survives on unseen embryos?
By the end of July the family had seven submissions, five candidates and two resubmissions, and none returned a score: the first two failed with an unhandled error on hidden data, the other three completed with no score.

Each blank was compatible with several causes, so the next steps tested them one at a time.
An audit found five deterministic failure paths that could fire only on hidden data, from a key error on an unseen embryo prefix to movie length inferred from the last detected node, each invisible on the public example movies, which are copies of training movies.
All five were patched, two budgets were resubmitted, and both again completed with no score.

With the patched outputs valid on every movie I could see, the remaining candidate was runtime: the public four-movie run alone took roughly $65$ minutes, about $30$ of them in four-fold proposal inference, and that cost grows with the number of hidden movies.
That attribution is arithmetic, not an isolated measurement.
The next deployment I have planned cuts proposal inference to one model per movie; whether that is enough is a question for August.

```text
The machine asks: is this policy better on held-out data?
The hidden set also asks: can it finish, inside twelve hours,
on an unknown number of movies from an unseen embryo?
The second question has veto power.
```

This is a deployment failure, not a model rejection: inference cost is a property of a policy, and it became clause C6, a candidate must finish on the hidden set within the time limit.

---

## 9. Seven Reference Graphs in Use at Once

Every delta in this note is a change to some $199$-movie local reference graph, and by the end of July seven were in simultaneous use, never reconciled: $0.6006730$, $0.6529429$, $0.6996400$, $0.7255533$, $0.7347169$, $0.7395609$ and $0.7405959$.
Node recall across them spans roughly $0.880$ to $0.948$.
On graphs that far apart the same edit has a different denominator, and a weaker graph has more broken structure to repair, which flatters any repair operator measured on it.
The four promotions of the window show the problem:

| promoted policy | measured delta | reference graph it was measured on |
|---|---:|---:|
| edge replacement, nested five-fold | $+0.0002706$ | $0.6529429$ |
| track-fragment matcher, post-solver | $+0.0006325$ | $0.6006730$ |
| strict-division rank ensemble | $+0.000949$ | $0.7255533$ |
| pre-solver candidate activation, $0.50$/frame | $+0.0006748$ | $0.7405959$ |

![Four promoted policies plotted above the seven 199-movie local reference graphs they were measured on]({{ site.baseurl }}/assets/img/posts/2026-07-31-biohub-working-note-3/fig-01-four-baselines.png)
_Figure 1. Each July promotion was measured against a different 199-movie local reference graph. Each delta is a change to its own graph, so the four cannot be ranked against each other._

The fragment matcher's figure is the only one measured on the weakest graph in the set, and the data cannot say how much of the ordering is policy quality rather than choice of reference.
The rule this forces became clause C5: compare levels only inside one reference universe (one reference graph and the replay built on it), and carry only deltas across; even a delta travels imperfectly, for the denominator reason above.

Two further gaps belong beside it.
The model-level twofold split is embryo-disjoint, but several policy cross-fits on top of it are not: the outer folds of the flow gate and the higher-order matcher each hold $14$–$15$ movies of one prefix and $25$–$26$ of the other, movie-out within seen embryos, while the hidden test is an unseen embryo.
And almost no experiment recorded its wall-clock cost, so the machine could not yet say whether the next refusal was worth buying.

---

## 10. Decision Log

The rule in force was the one written on 07-15: exact score change above a floor, both embryos and every outer fold nonnegative, bounded edits, and fitting, calibration and evaluation on disjoint movies.
The board had two uses.
It answered questions the machine could not yet pose, above all which points exist, because the machine held the point set fixed.
And it checked whether a local verdict survived on unseen embryos: the division approval with its control, the probes of a refusal, the fragment matcher, the pre-solver budgets.
There were $109$ scored submissions that month; the board was used most while the local instrument was youngest.

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| 07-15: write gates and budget before any result | a gate chosen after the result can always be passed | nine refusals on predeclared grounds; the budget went untracked | C2 |
| 07-17: exact replay of the graph stages on OOF | know their held-out value before replacing any | $+0.052187$; $11{,}020$ edge FNs with both endpoints unmatched | turned the program to which points exist |
| 07-20 to 07-25: two-seed and node-field questions to the board | the machine froze the point set; Note 2 left the two-seed question open | mixes $0.905$–$0.909$; field average $0.911$, ablation $0.910$ | node field to the board for now; edge reweighting to the machine |
| refuse selector, flow gate, parent matcher, proposal ensemble | gates written before each run | $26.7\times$ firing rate; sign reversal; $32\times$ attenuation; probes $0.004$ lower | C3, C4 |
| 07-28: division ensemble with its single-seed control | first division approval ($+0.000949$, 4/4 folds); check transfer | both $0.916$, $+0.004$ | observation: the hidden set weighs division far more than the replay |
| 07-30/31: pre-solver activation at several budgets | strongest structural result; check transfer | seven submissions, no score | C6 |
| record each delta with its reference graph | a delta depends on the graph it changes | seven references; approvals unrankable | C5 |

### The criterion at the end of this period

| clause | wording | since |
|---|---|---|
| C1 | Measure every graph edit out of fold: fit, calibrate and evaluate on disjoint movies, scored by the official metric on the whole graph | Note 2 |
| C2 **(new)** | Write each gate down before the result exists | Note 3 (07-15) |
| C3 **(new)** | Calibrate a rule on the population it will act on | Note 3 |
| C4 **(new)** | Judge a component by the graph it produces, in an exact replay, not by its own accuracy | Note 3 |
| C5 **(new)** | Compare levels only inside one reference universe; compare deltas across | Note 3 |
| C6 **(new)** | A candidate must finish on the hidden set within the time limit | Note 3 |

---

## 11. What the Period Established

### Established

1. Exact replay of the deterministic graph stages on held-out predictions is worth $+0.052187$ over the raw model graph.
2. Much of the edge-FN residual is unreachable by association: $11{,}020$ of about $25{,}150$ have both endpoints unmatched.
3. A strictly out-of-fold selector can still fail on population: firing rate $0.204\%$ calibrated, $5.456\%$ deployed.
4. Component metrics do not track graph score: parent top-1 gains became a sign reversal and a $32\times$ attenuation.
5. Restoring nodes is not a score strategy under the node-count adjustment; three operators raised node recall and lowered the score.
6. In this pipeline the graph filter, not the solver, decides what counts as a division.
7. The division term is nearly unused: $0.00046$ of its $0.1$ weight, on a graph carrying $12{,}794$ forks against $151$ annotated divisions.
8. The same proposals are worth about $13\times$ more when the solver arbitrates them than when inserted afterwards.
9. A policy can pass every scientific gate and return nothing from the board: seven submissions, no score.

### Supported but Unconfirmed

1. Whatever the second detector adds arrives through the shared node field, not association reweighting: a board ablation pair reading the same score, and seventeen association submissions that never exceeded the approximate single-seed figure. No local measurement yet.
2. The board registers the division channel where promoted edge levers of similar local size did not: the ensemble and its control both read $0.916$; the fragment matcher ($+0.0006325$) read its base configuration's score. A direction, not a rate.
3. Classification quality is not intervention utility; the evidence is one run and a gap of $7\times10^{-6}$.
4. Fold-level operating-point stability is a better ensemble-selection signal than mean delta.
5. Adding edges outside annotated context is viable only above $p^{*}=0.422315$, and one pattern of proposal sources clears it.

### Open Questions

1. Do policy gains measured under movie-out folds survive embryo-pure folds?
2. Can the seven local reference graphs be reconciled into one replay universe?
3. How do local deltas relate to board deltas, per channel, with more than one point?
4. Is the both-unmatched residual reachable by a detector change at all?
5. What is a policy's inference budget, as a number, before it is proposed?

---

## Closing

The machine Note 2 asked for mostly refused, each time on a gate written before the run, including the idea I had ranked first in my own survey two days earlier.
The refusals share one finding: **every metric that improved and then failed was measured on something other than what the policy would change**, whether sparse annotated context for the dense graph, parent choice for connected components, or classification quality for intervention utility.
Out-of-fold machinery guards against training on the evaluation data; the new clauses guard against evaluating on the wrong object.

Approving is the harder half.
The four approvals were $10^{-4}$ in size, on four different reference graphs, against a board that shows $10^{-3}$.
The division ensemble, checked with its control, moved the board by $+0.004$; the strongest approval could not finish on the hidden set; and several cross-fits behind the approvals held out movies, not embryos.

So the question I carry into August is this:

```text
Can local evidence decide,
with the board used only to check that a local decision survives on unseen embryos?
```

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- **Part 3: What the OOF Machine Refused**
