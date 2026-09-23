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
- Korean version: [BioHub Cell Tracking 작업 기록 3: OOF에 기반한 판단들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused-KR/)
- Follow-up: [BioHub Cell Tracking Working Note 4: Why the Largest Local Gain Did Not Show on the Board]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board/)

Related public notebooks:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

> **About this series.** These notes follow one Kaggle competition, BioHub Cell Tracking During Development:
> rebuilding cell lineages from 3D time-lapse microscopy of zebrafish embryos, trained on 199 movies from two
> embryos. The Public leaderboard shows 29% of a hidden test of unseen embryos, and a score chased there over many
> submissions overfits it. So each decision was made on its logic, a mechanism stated in advance, and the logic was
> tested by local validation. The series follows how that validation was made trustworthy, where the search itself
> fell short, and how the final submissions were chosen.
{: .prompt-info }

Note 2 ended with the Public score flat near $0.902$ and three explanations that one more submission could not separate: correlated model errors, a calibration inherited from the wrong distribution, and adaptation to the Public leaderboard (the board), which more tuning on the board would only deepen.
So I built the OOF machine: every training movie is scored out of fold (OOF), by a model that never trained on it, and the notebook's graph stages are replayed exactly on those predictions and scored with the official metric, so a graph edit (any change to which nodes and edges the lineage graph contains) is measured on unseen movies.

This note asks what the machine refused, approved and could not see from 07-15 to 07-31.
It refused nine candidates on predeclared gates, each refusal locating where a stated mechanism failed; its four approvals were below the board's three-decimal resolution.
Where it was blind, the board answered, and it served as a sanity check: a Public submission that checks whether a local decision holds on unseen embryos.

The short version is:

```text
Note 2 stopped tuning on the board; changes would be measured out of fold.
OOF refused nine candidates on predeclared gates, each for a named mechanism.
Its four approvals were 10^-4, on four reference graphs; the board shows 10^-3.
Where OOF was blind (which points exist), the board answered.
A division approval and its control both moved the board by +0.004.
The strongest structural approval returned no score in seven submissions.
OOF could refuse with reasons; whether it could decide was left for August.
```

| Sections | Question |
|---|---|
| 0 | What would count as a pass, fixed before the first result? |
| 1--2 | What is the existing pipeline worth out of fold, and what was the first decision made there? |
| 3 | What could OOF not yet see, and what did the board answer there? |
| 4--5 | Which mechanisms did the candidates state, and where did OOF show each one failing? |
| 6--7 | Where did the error budget point, and does an edit's value depend on where it acts? |
| 8--9 | Did the structural approval survive the hidden set, and what was each approval measured against? |
| 10--11 | Which decisions did the period make, and what is established? |

---

## 0. What Would Count, Written Down Before the First Result

A gate chosen after the result is seen can always be passed, so on 2026-07-15, before running anything, I wrote down the program (capture fold-correct out-of-fold predictions, compute exact error anatomy, build one minimal operator for the dominant recoverable failure, promote only embryo-stable positive operators, then train full-data models), the promotion gates, and a submission budget of three configurations plus two in reserve.

The official score is an adjusted edge Jaccard, which also penalizes predicting more nodes than the movie's estimated cell count, plus one tenth of a division Jaccard, both counted only where annotators labeled.
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
Each check was recorded separately, so each refusal named its reason; the last three did most of the refusing.

On 2026-07-18 the host patched an exploit in the division metric (full leaderboard rescore announced 2026-07-23), and the condition Note 2 described became stricter and local: a parent-side node matched to a ground-truth node, a genuine fork there, two daughter branches that do not immediately merge again, and one-to-one assignment between predicted forks and ground-truth divisions.
Roughly $0.028$ came off the top of the board, so part of the gap Note 2 reasoned about was the old metric.
No submission of mine was scored under both versions; from my pipeline's shape I infer that it did not use the exploit, and I read my July scores as one scale.

---

## 1. The First Test: What the Existing Graph Stages Are Worth on Unseen Movies

Note 2 had left open whether the hand-built graph stages were method or a calibration adapted to the board.
On 07-17 I replayed them exactly as submitted (ILP selection, motion reassignment, short-component pruning, one-frame gap recovery, safe division repair) on the epoch-$100$ twofold out-of-fold predictions.
The ILP is the solver: an integer linear program that chooses which candidate nodes and edges form the graph.

| quantity | raw model graph | after exact replay | delta |
|---|---:|---:|---:|
| official score | $0.647453$ | $0.699640$ | $+0.052187$ |
| node recall | $0.916820$ | $0.911433$ | $-0.005387$ |
| movies improved | — | $183$ of $199$ | — |

The stages are worth $+0.052$ on held-out movies: method, not board calibration, and not the bottleneck.
They trade nodes for edges, an asymmetry that recurs through this note.

Of roughly $25{,}150$ remaining edge false negatives (FNs), $11{,}020$ had **both endpoint nodes unmatched**.

```text
An edge FN between two matched nodes is an association error.
An edge FN between two unmatched nodes is a detection error
wearing an association error's clothes.
No threshold, no ranker and no gap policy can reach the second kind.
```

That one number pointed the program at a question the machine could not yet measure, which points exist (section 3); edits on a fixed point set stayed local (section 2).

A short-track rescue in the same report tested an idea carried since Note 1, that restoring nodes helps because node recall is part of the score.
It restored $14{,}917$ nodes and raised node recall by $+0.00186$; the official score fell by $0.000735$, and $156$ of $199$ movies got worse.
Under the node-count adjustment, a restored node that attracts no true edge costs in the edge term and in the count ratio at once.

---

## 2. The First Decision Made Entirely Out of Fold

The edge-replacement ranker tested the smallest claim: if a ranker trained out of fold can tell which of the pipeline's edges to replace, the replacements should help on unseen movies in both embryos.
On 07-19 it finished a nested five-fold cross-fit at the fixed $200$-epoch capture: fitting the policy, calibrating its threshold and the final evaluation each sat on disjoint movie sets, and thresholds fixed to fold medians meant no test movie needs a fold identity.
The result was $+0.0002706$ from $1{,}447$ replacements, both embryo prefixes positive.
It was the project's first fully separated decision, promoted on local evidence alone: at $2.7\times10^{-4}$ it sat below the board's resolution.

---

## 3. Where OOF Was Blind: Which Points Exist Went to the Board

The machine's captures froze the detections, so in the third week of July it could rank only edits to a fixed point set; until a replay that re-extracts the points arrived (section 5), only the board could price a change to the detection field.

### 3.1 Does a Second Seed Help Through Association?

Note 2 had left open whether a second, independently seeded model adds information that a jointly calibrated mixture can use.
From 07-20 through 07-23 seventeen submissions varied the mixing rule in the association channel: logit mixtures, margin-adaptive mixing, consensus gates and basin searches all read between $0.905$ and $0.908$ in the submission record (my log reads one of them $0.909$).
None was better than the strongest single-seed graph, about $0.909$ in my log around 07-19/20, at the board's resolution; the weakest read $0.003$ to $0.004$ lower.
Edge reweighting on a fixed point set went back to the machine, which measures it at $10^{-4}$.
The step of about $+0.006$ from the $0.902$–$0.903$ band where Note 2 ended to that $0.909$ is attributable to no experiment in this window.

### 3.2 Or Through the Node Field? An Ablation Pair

On 07-24 the question moved from edges to points: one candidate averaged the two aligned **detection logit fields before peak extraction**, and went in on the same day as its ablation and its brackets, so the board's answer could be attributed.

| candidate | change | public |
|---|---|---:|
| field average before peak extraction | balanced detection field, dual-seed association retained | $0.911$ |
| **balanced field, primary-only association** | **the ablation** | $0.910$ |
| primary-weighted field | off-balance toward the primary seed | $0.909$ |
| secondary-weighted field | off-balance toward the second seed | $0.907$ |

With the dual-seed association removed, the balanced field read the same score at the board's resolution: whatever the second seed contributed, the board saw it only through the shared node field, a direction carried as unconfirmed (section 11).

A slightly primary-leaning field (secondary detection weight $0.475$ instead of $0.5$, same threshold) read $0.912$ on 07-25, tied with the balanced field, and became the frozen reference for most of the out-of-fold replays that followed.
A two-sided probe of the detection threshold read the same score on both sides, closing that axis as flat.

### 3.3 What That Use Did to the Submission Budget

The 07-15 budget assumed the board would see only what the machine had promoted.
With no local instrument for the node field, the board served as one; the gates were checked on every operator, the budget was not, and it lapsed unamended at $109$ scored submissions for the month.

---

## 4. Testing Stated Mechanisms: Four Refusals and Where Each Failed

Four of the nine refusals share a pattern: the candidate's component metric improved, and the graph did not.

### 4.1 A Selector Calibrated Where Labels Exist Fired Where They Do Not

**The claim.** A leak-free selector that learns from labeled cases when the second seed's parent is the better one should help wherever it fires.
The dual-seed association selector was strictly model-out-of-fold, cross-fitted, calibrated on $177$ beneficial against $43$ harmful labeled groups (a group is one cell's candidate parents), and clean under a leakage audit.

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

The exact replay lost $0.000068190$ and worsened $112$ of $199$ movies; no post-hoc fold subset cleared the floor, and none could be deployed, since a test movie carries no fold identity.
To move the official edge counts by $+16$ true positives (TP), $+26$ false positives (FP) and $-16$ FN, the selector changed $3{,}001$ raw nodes and $3{,}450$ raw edges: the annotated skeleton it was scored on is a thin slice of the graph it rewrites.

**What it established.** The selector had no leakage but the wrong population, which a leakage audit cannot see and a firing-rate comparison can: a rule whose rate changes by more than a small factor between calibration and deployment is not calibrated for where it acts.
That became clause C3: calibrate a rule on the population it will act on.

### 4.2 Picking the Right Parent More Often Did Not Build a Better Graph

**The claim.** A model that picks the correct parent more often on held-out movies should reconstruct a better graph; two models tested that within four days.

| model | component metric | exact graph replay |
|---|---|---|
| temporal-flow gate (switches to a motion model's parent) | parent top-1 $0.867186 \to 0.879198$ ($+0.012012$); five of five outer folds nonnegative | $-0.0007833$; $63$ improved / $132$ worsened |
| higher-order parent matcher (compares candidate parents jointly) | parent top-1 $0.867589 \to 0.883202$ ($+0.015613$) | $+0.000482$; one outer fold at $-0.000788$; one prefix at $-0.000280$ |

**What the replay measured.** The flow gate reversed sign: a component metric that improved on every fold produced a graph worse on two movies in three.
The higher-order matcher stayed positive, attenuated about $32\times$ from top-1 gain to graph score.

**The mechanism.** The flow gate was leak-proof; the loss happens downstream.
Changing a node's parent changes which connected components exist, and so what short-component pruning removes and what gap recovery may bridge: a locally correct switch can delete a component that carried several correct edges, and the parent-choice metric never sees components.
The gate also raised node recall by $+0.001145$ while lowering the score, the same trade as the short-track rescue.

**Checking the refusal.** The higher-order matcher failed its fold and prefix gates and was not promoted.
Its picture was mixed (positive in aggregate, negative in one fold and one prefix), so I submitted three variants as sanity checks of the refusal; the board read the two broad arms $0.004$ lower than the $0.916$ division configuration they were built on, consistent with it.

Together the two models became clause C4: judge a component by the graph it produces, in an exact replay, not by its own accuracy.

### 4.3 The Better Classifier Inserted the Worse Nodes

**The claim.** Of two versions of the candidate-proposal model, which proposes cells the detector missed, the one that classifies better should insert better nodes.
The predeclared ensemble beat the single model on every offline metric I had: known-label AP $0.9999984$ against $0.9999973$, and recovery of the frozen graph's node misses $0.53075$ against $0.52633$.

**What the replay measured.** Inserted into the finished graph, the single model was the better one: $+0.0000506$ from $842$ repairs against the ensemble's $+0.0000435$ from $845$.
Neither was promoted, and with one run and a gap of $7\times10^{-6}$ the ordering is a warning, not a law: classification quality and intervention utility are different objectives, C4 from a different direction.

Two-sided bridge repairs, which insert a node only where it links to existing fragments on both sides, produced zero repairs out of $1{,}354$ candidate triples: the missing nodes are regions where the detector fails over several frames at once, not one-frame holes.

---

## 5. Four More Claims About Where Score Was Left, Tested and Refused

The next four candidates placed the missing score elsewhere: in a per-embryo operating point, in division precision, in the solver's objective, in keeping alternative graphs alive.
The ninth refusal, a third division seed, is in section 6.

**A per-embryo detector did not hold across folds.**
A full out-of-fold detection grid, which re-extracts the points for every weight, showed one embryo improving as the secondary detection weight fell while the other weakened, suggesting a per-embryo router.
A nested cross-fit inside each embryo rejected it at $-0.0003550114$ (reciprocal cross-fit $-0.0003884963$), both embryo-level deltas negative; the development folds chose $0.40$ twice and the baseline twice in one embryo, and the opposite extremes, $0.25$ and $0.70$, in the other.
The two model folds coincide with the two embryo prefixes, so embryo identity cannot be separated from fold identity; the subgroup effect was real as a description but not stable enough to be a rule.

**Emitting forks is not how the division term gets paid.**
The frozen out-of-fold graph already held $12{,}794$ predicted binary forks, against $151$ annotated divisions in the whole training set, and scored $4$ division true positives against $720$ false positives: what was scarce was clean forks at a matched parent.
A cross-fitted validator that pruned the weaker daughter edge of low-ranked forks cut division false positives from $720$ to $317$ and reached $+0.000328$, but it cost $197$ edge true positives and two of four outer folds were negative: it bought division precision, weighted at one tenth, with edge recall, weighted at one.

**The solver does not decide what a division is.**
I added a hyperedge variable to the ILP, forced to equal the conjunction of two daughter edges and rewarded by an out-of-fold division rank score.
As the reward rose from $1.00$ to $2.00$, the solver selected $7$, then up to $43$ complete fork events, and the patched division counts stayed at exactly $5$ TP / $507$ FP / $146$ FN at every reward.

```text
The solver chose more forks.
The downstream graph filter removed every one of them.
In this pipeline, what counts as a division is decided after the solver.
```

That closed the whole family of "put it in the objective" proposals.

**Keeping every hypothesis alive is not a plan.**
My 07-28 survey (section 6) had ranked this idea first: keep several graph hypotheses alive and let a joint optimizer arbitrate.
Two days later its predeclared gates stopped it: the union of the frozen graph, five other detection blends and the proposal model raised node recall from $0.943001$ to $0.965008$, but recovered only $38.61\%$ of the frozen graph's node misses against a gate of $40\%$, and needed $25.98$ novel points per frame against a gate of $6$.

The union can recover $1{,}704$ missed ground-truth edges, but they sit inside $649{,}702$ candidates outside the frozen graph.
A correct added edge raises both numerator and denominator of the edge Jaccard $J$ and a wrong one only the denominator, so the break-even precision is $p^{*}=J/(1+J)=0.422315$ at $J=0.731046$; exactly one pattern of proposal sources clears it in both prefixes and all four folds, at $0.526814$, and one pattern is not a channel.
Three fixed rules for admitting new tracklets scored strongly positive on a sparse edge-utility proxy and raised node recall; their official deltas were $-0.003536$, $-0.002364$ and $-0.006235$.
That triple (proxy strongly positive, node recall up, official score down) appeared three times in this window, on unrelated operators, and is the most reliable failure signature I have.

---

## 6. Following the Error Budget to the Division Channel, and Checking It on the Board

On 07-28, with most candidates refused, a score-upside survey priced the error budget of the frozen graph across all $199$ movies:

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

out of the $0.1$ weight the metric reserves for it; most of the compute had gone to the edge term, so I turned to the division channel.

The same day, the strict-division rank ensemble passed every gate.
Its claim: two independently seeded division-event models make partly different ranking errors, so combining their ranks should place true divisions higher than either alone.
A fixed equal-weight percentile rule combined them, and only a small top fraction of ranked candidates, the action fraction, was added after the frozen graph.

| arm | outer cross-fit delta | folds |
|---|---:|---|
| seed A alone (the single-seed control) | $+0.000661$ | — |
| seed B alone | $+0.000834$ | — |
| **two-seed percentile ensemble** | $+0.000949$ | all four positive; the same action fraction ($0.016$) in every fold |
| three-seed ensemble | $+0.000697$ | action fractions $0.064$ / $0.032$ / $0.032$ / $0.024$ |

It recovered nine division true positives at a small edge-Jaccard cost.

**The sanity check.** A first approval in a new channel is where a blind spot of the local criterion would show, so the ensemble went to the board with its single-seed control.
Both read $0.916$, $+0.004$ over the $0.912$ configuration: the board registered the division channel and could not separate the ensemble from its control.
It is the window's one upward step clearing the board's resolution against a recorded reference, and its size is the finding: the replay priced the channel at $10^{-4}$, and no promoted edge lever of comparable local size moved the board upward (section 7).
Resting on one rounded pair, it is an observation, not a rate: the hidden set appears to weigh the division term far more than the local replay does.

**The third seed** was diverse (out-of-fold score correlations of about $0.69$ and $0.84$ against the two promoted models) and positive alone, yet the three-seed ensemble scored below the two-seed one; its folds chose three different action fractions, spanning a factor of nearly three, and a policy whose operating budget moves when its fitting data moves has not found an operating point.

---

## 7. Testing Where an Edit Acts: Before or After the Solver

The refusals of sections 4 and 5 kept locating an edit's failure in what later stages did with it, and this window priced Note 2's finding that graph edits are not additive: the first pre-solver temporal-flow policy earned $+0.000365$ of adjusted edge, and one true division turned false ($-0.000153$ at one-tenth weight) consumed $42\%$ of that gain.
The claim to test: the value of an edit depends on **where in the pipeline it is applied**.
Writing $\Pi$ for the solver,

$$
\Delta S\!\left(\Pi\circ R\right)
\ne
\Delta S\!\left(R\circ \Pi\right).
$$

The candidate-proposal model gives the clean test: the same proposals, from the same weights, in two positions.
Inserted into the finished graph they were worth $+0.0000506$; appended to the point set before the solver, at $0.50$ nodes per frame, $+0.0006748$, about $13\times$ more.
The arms are not budget-matched, so this compares positions, not intervention counts.
The construction makes it causal: scores between the original points are restored exactly from the frozen capture, so a zero budget reproduces the frozen graph, verified to a maximum edge-probability delta of $5.96\times10^{-8}$.

| budget (nodes/frame) | delta | folds |
|---:|---:|---|
| $0.10$ | $+0.0000776$ | two negative |
| $0.25$ | $+0.0002702$ | one negative |
| $0.50$ | $+0.0006748$ | all four positive (promoted) |
| $0.75$ | $+0.0011047$ | all four nonnegative |
| $1.00$ | $+0.0012498$ | one fold at $-0.0006285$ |

The aggregate is monotone in the budget; the fold picture is not, as with the third division seed.

Deciding earlier is channel-specific.
The track-fragment matcher passed every gate after the solver, at $+0.0006325$ from $3{,}224$ edge replacements, and its hidden-safe sanity check read $0.912$, the score of the configuration it was added to: a tie, no failure detected.
Moved before the solver, its outer cross-fit was $+0.0000178$, indistinguishable from zero (the two figures sit on reference graphs whose node recall differs by roughly six points, so they are not a ratio).
Moving the division hyperedge of section 5 earlier had no effect.

Earlier helps when the earlier stage can create options that did not exist, since a node never proposed cannot be linked later by any ranker; it does nothing when the later stage was already the better arbiter of options both can see.
Nodes belong to the first class; edge reweighting and division events, in this pipeline, to the second.

---

## 8. Checking the Structural Approval on the Hidden Set: Seven Blank Submissions

Pre-solver activation (adding proposed points before the solver runs) held in section 7's clean test; the next question was which budget survives on unseen embryos.
Five candidates returned no score: the first two failed with an unhandled error on hidden data, the other three completed with no score.

An audit found five deterministic failure paths that could fire only on hidden data, from a key error on an unseen embryo prefix to movie length inferred from the last detected node, each invisible on the public example movies, which are copies of training movies.
All five were patched and two budgets resubmitted; both again completed with no score, seven blank submissions by the end of July.

With the patched outputs valid on every movie I could see, the remaining candidate is runtime: the public four-movie run alone took roughly $65$ minutes, about $30$ of them in four-fold proposal inference, and that cost grows with the number of hidden movies (an arithmetic attribution, not an isolated measurement).
The next deployment I have planned cuts proposal inference to one model per movie.

```text
The machine asks: is this policy better on held-out data?
The hidden set also asks: can it finish, inside twelve hours,
on an unknown number of movies from an unseen embryo?
The second question has veto power.
```

That became clause C6: a candidate must finish on the hidden set within the time limit.

---

## 9. Before Comparing the Approvals: What Each Was Measured Against

Ranking the four approvals needs a shared reference.
By the end of July seven $199$-movie local reference graphs were in simultaneous use, never reconciled: $0.6006730$, $0.6529429$, $0.6996400$, $0.7255533$, $0.7347169$, $0.7395609$ and $0.7405959$; node recall across them spans roughly $0.880$ to $0.948$.
On graphs that far apart the same edit has a different denominator, and a weaker graph has more broken structure to repair, which flatters any repair operator measured on it:

| promoted policy | measured delta | reference graph it was measured on |
|---|---:|---:|
| edge replacement, nested five-fold | $+0.0002706$ | $0.6529429$ |
| track-fragment matcher, post-solver | $+0.0006325$ | $0.6006730$ |
| strict-division rank ensemble | $+0.000949$ | $0.7255533$ |
| pre-solver candidate activation, $0.50$/frame | $+0.0006748$ | $0.7405959$ |

![Four promoted policies plotted above the seven 199-movie local reference graphs they were measured on]({{ site.baseurl }}/assets/img/posts/2026-07-31-biohub-working-note-3/fig-01-four-baselines.png)
_Figure 1. Each July promotion was measured against a different 199-movie local reference graph. Each delta is a change to its own graph, so the four cannot be ranked against each other._

The fragment matcher's figure is the only one measured on the weakest graph, and the data cannot separate policy quality from choice of reference.
That became clause C5: compare levels only inside one reference replay (one reference graph and the replay built on it), and carry only deltas across; even a delta travels imperfectly, for the denominator reason above.

Two further gaps remain.
The model-level twofold split is embryo-disjoint, but several policy cross-fits on top of it are not: the outer folds of the flow gate and the higher-order matcher each hold $14$–$15$ movies of one prefix and $25$–$26$ of the other, movie-out within seen embryos, while the hidden test is an unseen embryo.
And almost no experiment recorded its wall-clock cost, so the machine could not yet price the next refusal.

---

## 10. Decision Log

Candidates were decided by the rule written on 07-15: exact score change above a floor, both embryos and every outer fold nonnegative, bounded edits, and fitting, calibration and evaluation on disjoint movies.
Over $109$ scored submissions the board answered what the machine could not yet pose, above all which points exist, and served as a sanity check of local verdicts.

| decision | reason at the time | what came back | what it changed |
|---|---|---|---|
| 07-15: write gates and budget before any result | a gate chosen after the result can always be passed | nine refusals on predeclared grounds; the budget went untracked | C2 |
| 07-17: exact replay of the graph stages on OOF | know their held-out value before replacing any | $+0.052187$; $11{,}020$ edge FNs with both endpoints unmatched | turned the program to which points exist |
| 07-20 to 07-25: two-seed and node-field questions to the board | the machine froze the point set; Note 2 left the two-seed question open | mixes $0.905$–$0.909$; field average $0.911$, ablation $0.910$ | node field to the board for now; edge reweighting to the machine |
| refuse selector, flow gate, parent matcher, proposal ensemble | gates written before each run | $26.7\times$ firing rate; sign reversal; $32\times$ attenuation; probes $0.004$ lower | C3, C4 |
| 07-28: division ensemble with its single-seed control | first division approval ($+0.000949$, 4/4 folds); sanity check | both $0.916$, $+0.004$ | observation: the hidden set weighs division far more than the replay |
| 07-30/31: pre-solver activation at several budgets | strongest structural result; sanity check | seven submissions, no score | C6 |
| record each delta with its reference graph | a delta depends on the graph it changes | seven references; approvals unrankable | C5 |

### The criterion at the end of this period

| clause | wording | since |
|---|---|---|
| C1 | Measure every graph edit out of fold: fit, calibrate and evaluate on disjoint movies, scored by the official metric on the whole graph | Note 2 |
| C2 **(new)** | Write each gate down before the result exists | Note 3 (07-15) |
| C3 **(new)** | Calibrate a rule on the population it will act on | Note 3 |
| C4 **(new)** | Judge a component by the graph it produces, in an exact replay, not by its own accuracy | Note 3 |
| C5 **(new)** | Compare levels only inside one reference replay; compare deltas across | Note 3 |
| C6 **(new)** | A candidate must finish on the hidden set within the time limit | Note 3 |

---

## 11. What the Period Established

### Established

1. Exact replay of the deterministic graph stages on held-out predictions is worth $+0.052187$ over the raw model graph.
2. Much of the edge-FN residual is unreachable by association: $11{,}020$ of about $25{,}150$ have both endpoints unmatched.
3. A strictly out-of-fold selector can still fail on population: firing rate $0.204\%$ calibrated, $5.456\%$ deployed.
4. Component metrics do not track graph score: parent top-1 gains became a sign reversal and a $32\times$ attenuation.
5. Restoring nodes is not a score strategy under the node-count adjustment: three operators raised node recall and lowered the score.
6. In this pipeline the graph filter, not the solver, decides what counts as a division.
7. The division term is nearly unused: $0.00046$ of its $0.1$ weight, on a graph carrying $12{,}794$ forks against $151$ annotated divisions.
8. The same proposals are worth about $13\times$ more when the solver arbitrates them than when inserted afterwards.
9. A policy can pass every gate and return nothing from the board: seven submissions, no score.

### Supported but Unconfirmed

1. Whatever the second detector adds arrives through the shared node field, not association reweighting: one board ablation pair read the same score, seventeen association submissions never exceeded the approximate single-seed figure, and nothing was measured locally.
2. The board registers the division channel where a promoted edge lever of similar local size did not: ensemble and control both read $0.916$; the fragment matcher ($+0.0006325$) read its base configuration's score.
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

OOF refused nine candidates on predeclared gates.
The refusals share one finding: **every metric that improved and then failed was measured on something other than what the policy would change**, whether sparse annotated context for the dense graph, parent choice for connected components, or classification quality for intervention utility.
Out-of-fold machinery guards against training on the evaluation data; the new clauses guard against evaluating on the wrong object.

The four approvals were $10^{-4}$ in size, on four reference graphs, against a board that shows $10^{-3}$.
The division ensemble and its control both moved the board by $+0.004$; the strongest structural approval could not finish on the hidden set; several cross-fits behind the approvals held out movies, not embryos.
The question I carry into August is whether those approvals can choose what ships:

```text
Can OOF decide,
with the board used only as a sanity check that a local decision survives on unseen embryos?
```

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- **Part 3: What the OOF Machine Refused**
