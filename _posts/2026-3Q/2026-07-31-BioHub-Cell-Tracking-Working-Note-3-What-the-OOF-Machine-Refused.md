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

Note 2 ended with a plan rather than a result.
It argued that one more public submission could not separate three explanations of a flat score: correlated errors between the models, a calibration inherited from the wrong distribution, and adaptation to the visible part of the leaderboard.
It proposed a different unit of experiment: a held-out counterfactual score change per graph edit, accepted only through promotion gates written down before the result existed.

That machine got built in the second half of July.
Its most useful property was that it could say no, and say why.
Over seventeen days it refused nine of its own candidates on predeclared grounds, and four of those refusals removed three beliefs I held with some confidence.
A leakage-audited selector had been calibrated on one population and fired on another.
A model that picked the correct parent more often built a worse graph.
The better of two candidate classifiers made the worse graph.

What it approved was small.
Four policies cleared every gate, with local gains of $+0.00027$, $+0.00063$, $+0.00095$ and $+0.00067$, each measured against a different local reference graph and each $10^{-4}$ in size, against a public board that displays $10^{-3}$.
Only one of the four, a division policy, came with a board step large enough to count.
The other change I kept from this window, averaging two detectors' fields before peak extraction, was a question the machine could not yet pose, and I settled it on the board.

This was not yet a project that ignored the board.
July ended with 109 scored submissions, and I broke the submission budget I had written down on 07-15.
But it was the first time a rule other than the public score could overrule me, and a rule that can refuse is the first step off the board.

The short version is:

```text
A strict out-of-fold machine mostly says no, and it says why.
It overruled a leakage-audited selector, a better parent chooser and a better classifier.
Each had been scored on something other than the graph it would change.
What it approved was 10^-4 in size, against a board that shows 10^-3.
The one approval large enough for the board to see was a division policy.
Which points exist was still a question only the board could answer.
```

| Sections | Question |
|---|---|
| 0--2 | What was written down on 07-15, and what did exact replay and the first promotion establish? |
| 3 | Why was the node field a question for the board, and what did the ablation show? |
| 4--5 | Which candidates were refused, and what did each refusal remove? |
| 6--7 | What did the division channel show, and does moving a decision before the solver change its value? |
| 8--9 | Why did the pre-solver result return no score, and against what was it all measured? |
| 10--11 | What rule did I select by, and what is established, supported, and open? |

---

## 0. The Program Written Down on 07-15, and a Metric That Moved

On 2026-07-15, before running any of it, I wrote down a program: capture fold-correct out-of-fold predictions, compute exact error anatomy, build one minimal operator for the dominant recoverable failure, promote only embryo-stable positive operators, and only then train full-data models.
The same document fixed the promotion gates and a submission budget of three configurations plus two in reserve.
The gates matter more than the steps, because a gate written after seeing the result is not a gate.
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

where $\Delta S$ is the exact change in the official combined score after replaying the full deterministic pipeline, $p$ ranges over the two embryo prefixes (the training movies come from two embryos, and a movie's ID prefix names its embryo), $k$ over the outer cross-fit folds, and $|R|$ is the edit count.
Every check was recorded separately, so each refusal named its own reason.
The last three conjuncts did most of the refusing.

Underneath the program, the instrument changed.
The host patched an exploit in the division metric; the scorer change is dated 2026-07-18, and a full leaderboard rescore was announced on 2026-07-23.
That forces a correction to Note 2, whose description of the division condition (a weakly connected predicted component covering the pre-division stage and both daughter lineages) is obsolete from 07-18.
The patched condition is stricter and local: a parent-side node matched to a ground-truth node, a genuine fork there, two daughter branches that do not immediately merge again, and one-to-one assignment between predicted forks and ground-truth divisions.

After the rescore, roughly $0.028$ came off the top of the board, so part of the gap Note 2 reasoned about was the old metric rather than a different method.
No submission of mine straddles the scorer change, so the claim that my pipeline never used the exploit is an inference from its shape, not a measurement.

---

## 1. The Exact Replay, and the Residual It Exposed

On 07-17 I replayed the deterministic graph stages of my submitted notebook exactly (ILP selection, motion reassignment, short-component pruning, one-frame gap recovery, safe division repair) on the epoch-$100$ twofold out-of-fold predictions, and scored the result with the official scorer.

| quantity | raw model graph | after exact replay | delta |
|---|---:|---:|---:|
| official score | $0.647453$ | $0.699640$ | $+0.052187$ |
| node recall | $0.916820$ | $0.911433$ | $-0.005387$ |
| movies improved | — | $183$ of $199$ | — |

The intended reading is that the hand-built post-processing is worth $+0.052$ on held-out data.
It is not over-fitted to the leaderboard, and it is not the bottleneck.

The second reading is in the node-recall row.
The stages trade nodes away and buy edges with them, an asymmetry that recurs through this note.

The residual set the direction for the rest of the month.
Of roughly $25{,}150$ remaining edge false negatives, $11{,}020$ had **both endpoint nodes unmatched**.

```text
An edge FN between two matched nodes is an association error.
An edge FN between two unmatched nodes is a detection error
wearing an association error's clothes.
No threshold, no ranker and no gap policy can reach the second kind.
```

Three side experiments closed in the same report.
A learned residual gap policy could not be trained, because its $976$ candidate rows held exactly one positive.
A hierarchical division reconnection had negative event utility.
A short-track rescue was harmful when always on, and that one removed a belief I had carried since Note 1: that restoring nodes helps, because node recall is part of the score.
It restored $14{,}917$ nodes and raised node recall by $+0.00186$.
The official score fell by $0.000735$, and $156$ of $199$ movies got worse.
Under the node-count adjustment, a restored node that attracts no true edge is a cost in the edge term and in the count ratio at once.

---

## 2. The First Fully Cross-Fitted Promotion

On 07-19 the out-of-fold edge-replacement ranker finished a nested five-fold cross-fit at the fixed $200$-epoch capture.
Policy fitting, threshold calibration and final evaluation each sat on disjoint movie sets, and the thresholds were fixed to fold medians so that no test movie needs a fold identity to be scored.
The nested cross-fit, which is the number that counts, was $+0.0002706$ from $1{,}447$ replacements, with both embryo prefixes positive.

This was the first time in the project that every stage of a decision was separated from every other.
But a local gain of $2.7\times10^{-4}$ cannot be seen on a board rounded to $10^{-3}$.
The machine worked, and what it approved was below the resolution of the only external instrument I had.

---

## 3. Which Points Exist: A Question for the Board

In the third week of July the machine still ranked edits to a fixed point set: its captures froze the detections and scored association on them.
A change to the detection field changes the vertices of the graph, so it cannot be calibrated that way.
A replay that re-extracts the points for each detection setting came later in the window (section 5).

The association slots came first, and they were not that kind of question.
From 07-20 through 07-23 I spent seventeen submissions on one hypothesis: that a second, independently seeded detector would pay off through the association channel, and the only open question was the mixing rule.
Those candidates reweighted edges on a fixed point set, which the machine could measure, and I spent the board on them anyway.
Logit mixtures, margin-adaptive mixing, consensus gates and basin searches all read between $0.905$ and $0.908$ in the submission record.
My log reads one of them $0.909$.
The honest denominator was not the notebooks this work branched from but the strongest single-seed graph, which my log puts at about $0.909$ around 07-19/20.
Against it, no association variant was better at the board's resolution, and the weakest read $0.003$ to $0.004$ lower.
The step from the $0.902$–$0.903$ band where Note 2 ended to that $0.909$, about $+0.006$, I cannot attribute to any experiment in this window.

On 07-24 one candidate stopped mixing edge logits.
It averaged the two aligned **detection logit fields before peak extraction**, so both models contributed to which points exist rather than to which points connect.
That was a question the machine could not pose yet, and I submitted it with its ablation and its brackets on the same day.

| candidate | change | public |
|---|---|---:|
| field average before peak extraction | balanced detection field, dual-seed association retained | $0.911$ |
| **balanced field, primary-only association** | **the ablation** | $0.910$ |
| primary-weighted field | off-balance toward the primary seed | $0.909$ |
| secondary-weighted field | off-balance toward the second seed | $0.907$ |

The ablation is the informative row.
With the dual-seed association removed, the balanced field read the same score at the board's resolution.
Whatever the second seed contributed, it did not arrive through association.
Submitting the field average alone would have given $0.911$ and the wrong story.
The pair pointed at the mechanism.

A slightly primary-leaning field (secondary detection weight $0.475$ instead of $0.5$, same threshold) read $0.912$ on 07-25, the same score as the balanced field at the board's resolution.
It was frozen as the reference configuration for most of the out-of-fold replays that followed.
Against the approximate single-seed figure it reads about $+0.003$, at the edge of the board's resolution: a direction, not a measured gain.
A two-sided probe of the detection threshold then closed that axis.
Both sides read the same score at the board's resolution, so it closed as flat rather than as a bracketed optimum.

One more entry belongs in this record.
The budget of three plus two did not survive five available slots a day.
It was never amended and never appealed; it simply stopped being tracked.
The gates were enforced on operators, not on my own consumption of the one resource they existed to protect.

---

## 4. Three Beliefs the Refusals Removed

Four of the nine refusals removed three beliefs, and those beliefs are the core of this note.
In each, a component metric improved and the graph did not, and I had believed the component metric.

### 4.1 The Calibration Population Was Not the Deployment Population

**Belief.** A dual-seed association selector that is strictly model-out-of-fold, whose policy is cross-fitted on top, and which is calibrated on $177$ beneficial against $43$ harmful labeled groups, is ready to deploy.
A group is one cell's candidate parents, and firing means swapping in the second seed's parent.

**Measured.** It was calibrated where labels exist and deployed where they mostly do not:

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

That is a $26.7\times$ expansion in firing rate.
The exact replay lost $0.000068190$ and worsened $112$ of $199$ movies.
No post-hoc fold subset cleared the floor, and none could have been deployed anyway, since a test movie carries no fold identity.
To move the official edge counts by $+16$ TP, $+26$ FP and $-16$ FN, the selector changed $3{,}001$ raw nodes and $3{,}450$ raw edges: the annotated skeleton it was scored on is a thin slice of the graph it rewrites.

The check I should have run first: compare the firing rate on the calibration population with the rate on the deployment population, and distrust any threshold that changes by more than a small factor.
A leakage audit does not catch this.
The selector had no leakage. It had the wrong population.

### 4.2 Parent Top-1 Is Not a Graph Score

**Belief.** If a model picks the correct parent more often on held-out movies, the reconstructed graph is better.

**Measured.** On two models within four days:

| model | component metric | exact graph replay |
|---|---|---|
| temporal-flow gate (switches to a motion model's parent) | parent top-1 $0.867186 \to 0.879198$ ($+0.012012$); five of five outer folds nonnegative | $-0.0007833$; $63$ improved / $132$ worsened |
| higher-order parent matcher (compares candidate parents jointly) | parent top-1 $0.867589 \to 0.883202$ ($+0.015613$) | $+0.000482$; one outer fold at $-0.000788$; one prefix at $-0.000280$ |

The flow gate reversed sign: a component metric that improved on every fold produced a graph that was worse on two movies out of three.
The higher-order matcher stayed positive, but a top-1 gain of $+0.0156$ arrived as $+0.00048$ of graph score, an attenuation of about $32\times$.

The flow gate was built to be leak-proof and passed every component-level gate.
Its loss happens downstream.
Changing which parent a node attaches to changes which connected components exist, and so what short-component pruning removes and what gap recovery may bridge.
A locally correct switch can delete a component that carried several correct edges, and the proxy never sees the component.
The gate also raised node recall by $+0.001145$ while lowering the score, the same trade as the short-track rescue.

The higher-order matcher failed its fold and prefix gates and was not promoted.
I submitted three variants anyway, as probes outside the deployment rule, and the board read the two broad arms $0.004$ lower than the $0.916$ division configuration they were built on.
The board agreed with a refusal the machine had already made, and I had let submissions run ahead of the gates.

### 4.3 The Better Classifier Made the Worse Graph

**Belief.** Of two versions of the candidate-proposal model, which proposes cells the detector missed, the one that classifies better will insert better nodes.
The predeclared ensemble beat the single model on every offline metric I had: known-label AP $0.9999984$ against $0.9999973$, and recovery of the frozen graph's node misses $0.53075$ against $0.52633$.

**Measured.** After insertion into the finished graph, the single model was the better one: $+0.0000506$ from $842$ repairs against the ensemble's $+0.0000435$ from $845$.
Both numbers are tiny, and neither was promoted on its own merit.
It is one run with a gap of $7\times10^{-6}$, so I hold the ordering as a warning rather than a law.
Classification quality and intervention utility are different objectives, and nothing guarantees that a ranking under one survives the other.

The gate meant to decide which repairs to keep could not be trained at all: $15$ positives and $6$ reliable negatives among $2{,}024$ rows gave a known-label AUC of $0.411$.
It was refused, the same data-sufficiency failure as the one-positive gap policy of section 1.

One diagnostic explains why the family was weak.
Two-sided bridge repairs, which insert a node only where it links to existing fragments on both sides, produced zero repairs out of $1{,}354$ candidate triples.
The missing-node error here is not a one-frame hole but a region where the detector fails over several frames at once.

---

## 5. Four More Refusals, Four Mechanisms

Four other refusals each named a different mechanism.
The ninth, a third division seed, is in section 6.

**No stable per-embryo detector.**
A full out-of-fold detection grid, which re-extracts the points for every weight, showed one embryo improving as the secondary detection weight fell while the other weakened, so a per-embryo router looked as if it should pay.
A nested cross-fit inside each embryo rejected it at $-0.0003550114$, with both embryo-level deltas negative.
In one embryo the four development folds chose weight $0.40$ twice and the baseline twice; in the other they chose the two opposite extremes, $0.25$ and $0.70$, twice each.
Held-out fold deltas changed sign.
The grid's own reciprocal cross-fit, fitted on one embryo and tested on the other, was $-0.0003884963$, because the two development folds chose different settings.
The two model folds coincide with the two embryo prefixes, so none of this separates embryo identity from model-fold identity.
The subgroup effect was real as a description, and not stable enough to be a rule.

**Emitting forks is not how the division term gets paid.**
The frozen out-of-fold graph already held $12{,}794$ predicted binary forks, against $151$ annotated divisions in the whole training set, and scored $4$ division true positives against $720$ false positives.
Forks were not scarce.
Forks at a matched parent that fork cleanly and survive one-to-one assignment were scarce.
A cross-fitted validator that pruned the weaker daughter edge of low-ranked forks cut division false positives from $720$ to $317$ and reached $+0.000328$, but it cost $197$ edge true positives and two of four outer folds were negative.
A rule that buys division precision with edge recall trades a term weighted at one tenth against a term weighted at one, and it has to be very precise to break even.

**The solver does not decide what a division is.**
I added a hyperedge variable to the ILP, forced to equal the conjunction of two daughter edges and rewarded by an out-of-fold division rank score.
As the reward rose from $1.00$ to $2.00$, the solver selected $7$, then up to $43$ complete fork events.
The patched division counts stayed at exactly $5$ TP / $507$ FP / $146$ FN at every reward.

```text
The solver chose more forks.
The downstream graph filter removed every one of them.
In this pipeline, what counts as a division is decided after the solver.
```

That one fact about ordering closed a whole family of "put it in the objective" proposals.

**Keeping every hypothesis alive is not a plan.**
In my own 07-28 score-upside survey (section 6), I had ranked this idea first: keep several graph hypotheses alive and let a joint optimizer arbitrate.
Two days later a gate written before the run killed it.
The union of the frozen graph, five other detection blends and the proposal model raised node recall from $0.943001$ to $0.965008$, but it recovered only $38.61\%$ of the frozen graph's node misses against a gate of $40\%$, and needed $25.98$ novel points per frame against a gate of $6$.

The residue explains the refusal.
The union can recover $1{,}704$ missed ground-truth edges, but they sit inside $649{,}702$ candidates outside the frozen graph.
A correct added edge raises both numerator and denominator of the edge Jaccard $J$, and a wrong one raises only the denominator, so the break-even precision is

$$
p^{*}=\frac{J}{1+J},
\qquad
J=0.731046
\;\Longrightarrow\;
p^{*}=0.422315 .
$$

Exactly one pattern of proposal sources in the union clears it in both prefixes and all four folds, at $0.526814$, and one pattern is not a channel.
Three fixed rules for admitting new tracklets scored strongly positive on a sparse edge-utility proxy and raised node recall.
Their official deltas were $-0.003536$, $-0.002364$ and $-0.006235$.
That triple (proxy strongly positive, node recall up, official score down) appeared three times in this window, on unrelated operators.
It is the most reliable failure signature I have.

---

## 6. The Division Channel

On 07-28 a score-upside survey priced the error budget of the frozen graph across all $199$ movies:

| term | value |
|---|---:|
| official score | $0.725553$ |
| edge TP / FP / FN | $109{,}363$ / $20{,}715$ / $19{,}520$ |
| node recall | $0.9477$ |
| division TP / FP / FN | $4$ / $720$ / $147$ |

Of the edge false negatives, $5{,}455$ ($27.9\%$) still had both endpoints unmatched.

The division line changed my priorities.
The division term contributes

$$
0.1\cdot J_{\mathrm{div}}
=0.1\cdot\frac{4}{4+720+147}
=0.00046
$$

out of the $0.1$ weight the metric reserves for it.
A tenth of the metric was delivering under five ten-thousandths, while most of my compute had gone to the edge term.

The same day, the strict-division rank ensemble passed every gate.
Two independently seeded division-event models were combined by a fixed equal-weight percentile rule, and only a small top fraction of ranked candidates, its action fraction, was added after the frozen graph.

| arm | outer cross-fit delta | folds |
|---|---:|---|
| seed A alone (the single-seed control) | $+0.000661$ | — |
| seed B alone | $+0.000834$ | — |
| **two-seed percentile ensemble** | $+0.000949$ | all four positive; the same action fraction ($0.016$) in every fold |
| three-seed ensemble | $+0.000697$ | action fractions $0.064$ / $0.032$ / $0.032$ / $0.024$ |

It recovered nine division true positives at a small edge-Jaccard cost, and the four positive folds are what I promoted it on.
The ensemble's submission and its single-seed control both read $0.916$, $+0.004$ over the $0.912$ configuration.
The board could not tell the two-seed ensemble ($+0.000949$) from the single-seed control ($+0.000661$); it saw the division channel, not the ensemble.
That is the one upward step in this window that clears the board's resolution against a recorded reference, and it points the same way as the local result.
It is still one rounded pair, so I do not turn it into a rate.
What it supports is qualitative: a division lever of $10^{-4}$ local size was visible on the board, and no promoted edge lever of comparable local size moved it upward (section 7).

The third seed was refused for a reason I did not expect.
It was diverse, with out-of-fold score correlations of about $0.69$ and $0.84$ against the two promoted models, and positive alone, yet the three-seed ensemble scored below the two-seed one.
The symptom was not the mean: its folds chose three different action fractions, spanning a factor of nearly three.
A policy whose operating budget moves when its fitting data moves has not found an operating point.

---

## 7. Deciding Earlier: Where It Worked, and Where It Did Not

Note 2 recorded that graph edits are not additive.
In this window that acquired a price.
The first pre-solver temporal-flow policy earned $+0.000365$ of adjusted edge and lost $0.001529$ of division Jaccard, which is $-0.000153$ at one-tenth weight.
One true division became a false one ($5$ / $507$ / $146$ to $4$ / $508$ / $147$), and that single event consumed $42\%$ of an edge gain earned across $199$ movies.
A division-safe variant scored $+0.00010523$ with two of four folds negative: protecting the division removed about seven tenths of the edge gain as well.
The two terms were the same moves, read twice.

The window added a stronger statement: the value of an edit depends on **where in the pipeline it is applied**.
Writing $\Pi$ for the solver,

$$
\Delta S\!\left(\Pi\circ R\right)
\ne
\Delta S\!\left(R\circ \Pi\right).
$$

The clean measurement is the candidate-proposal model: the same proposals, from the same weights, in two positions.
Inserted into the finished graph, they were worth $+0.0000506$.
Appended to the point set before the solver, at $0.50$ nodes per frame, they were worth $+0.0006748$, about $13\times$ more.
The arms are not budget-matched, so this compares positions, not numbers of interventions.
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
Its hidden-safe submission read $0.912$, the score of the configuration it was added to.
Moved before the solver, its outer cross-fit was $+0.0000178$, indistinguishable from zero.
The two figures sit on reference graphs whose node recall differs by roughly six points, so I do not compare them as a ratio.
The division hyperedge of section 5 was moved earlier with no effect at all.

Earlier helps when the earlier stage can create options that did not exist.
A node that was never proposed cannot be linked later, at any threshold, by any ranker.
Earlier does nothing when the later stage was already the better arbiter of options both stages can see.
Nodes belong to the first class; edge reweighting and division events, in this pipeline, belong to the second.

---

## 8. Seven Submissions, No Score

Pre-solver activation was the strongest structural result of the month, so I submitted it at several activation budgets.
By the end of July I had made seven submissions for this family, five candidates and two resubmissions, and none returned a score.
The first two failed with an unhandled error on hidden data.
The other three completed with no score.

An audit found five deterministic failure paths that could fire only on hidden data, from an unseen-prefix key error to movie length inferred from the last detected node.
Each is invisible on the public example movies, which are copies of training movies.
All five paths were patched, and two budgets were resubmitted.
Both again completed with no score.
Since the patched outputs were valid on every movie I could see, I attributed the repeat to runtime: the public four-movie run alone took roughly $65$ minutes, about $30$ of them in four-fold proposal inference, and that cost grows with the number of hidden movies.
That attribution is arithmetic, not an isolated measurement.
The next deployment I have planned cuts proposal inference to one model per movie; whether that is enough is a question for August.
Until the resubmissions, each blank was compatible with several causes, which is exactly when one keeps spending slots.

```text
The question I had been asking for three weeks:
  is this policy better on held-out data?
The question the hidden set asks:
  can this policy finish, inside twelve hours,
  on an unknown number of movies from an unseen embryo?
The second question has veto power.
```

This is a deployment failure, not a model rejection.
Nothing here says pre-solver activation is worthless.
It says that inference cost is a property of a policy, to be measured before the policy is proposed.

---

## 9. Seven Baselines, None Reconciled

The deltas in this note were computed against seven different $199$-movie local reference graphs, in simultaneous use and never reconciled: $0.6006730$, $0.6529429$, $0.6996400$, $0.7255533$, $0.7347169$, $0.7395609$ and $0.7405959$.
Node recall across them spans roughly $0.880$ to $0.948$.
On graphs that far apart the same edit has a different denominator, and a weaker graph has more broken structure to repair, which flatters any repair operator measured on it.
That applies to the four promotions of the window:

| promoted policy | measured delta | reference graph it was measured on |
|---|---:|---:|
| edge replacement, nested five-fold | $+0.0002706$ | $0.6529429$ |
| track-fragment matcher, post-solver | $+0.0006325$ | $0.6006730$ |
| strict-division rank ensemble | $+0.000949$ | $0.7255533$ |
| pre-solver candidate activation, $0.50$/frame | $+0.0006748$ | $0.7405959$ |

![Four promoted policies plotted above the seven 199-movie local reference graphs they were measured on]({{ site.baseurl }}/assets/img/posts/2026-07-31-biohub-working-note-3/fig-01-four-baselines.png)
_Figure 1. Each July promotion was measured against a different 199-movie local reference graph. Each delta is a change to its own graph, so the four cannot be ranked against each other._

Read down the middle column and the four numbers rank cleanly.
Read the right-hand column and the ranking dissolves.
The fragment matcher's figure is the only one measured on the weakest graph in the set.
I do not know how much of the ordering is policy quality and how much is the choice of reference graph.

A second gap belongs beside it.
The model-level twofold split is embryo-disjoint, but several policy cross-fits on top of it are not: the outer folds of the flow gate and the higher-order matcher each hold $14$–$15$ movies of one prefix and $25$–$26$ of the other.
They are movie-out within seen embryos, and the hidden test is an unseen embryo.

Finally, the window recorded a wall-clock cost for almost none of its experiments.
A machine that decides at $10^{-5}$ resolution without that cannot tell me whether the next refusal is worth buying.

---

## 10. The Rule We Selected By

| aspect | this period |
|---|---|
| rule in force | Gates written down on 07-15: exact $\Delta S$ above a floor, both prefixes $\ge 0$, every outer fold $\ge 0$, bounded edits, disjoint fitting, calibration and evaluation |
| where it was measured | Exact pipeline replay on $199$ fold-held-out movies, against seven unreconciled reference graphs, partly under movie-out policy folds |
| what the board was used for | The question the machine could not pose (which points exist); one upward step large enough to count ($+0.004$, division); against the rule, seventeen association-mixing slots and a probe of a refused policy, which read $0.004$ lower; and seven slots on the pre-solver promotion, which returned no score |
| what held, what broke | Held: nine refusals on predeclared grounds. Broke: the submission budget; the assumption that a held-out metric measures what a policy acts on; and deployability, since no gate measured inference cost and the pre-solver promotion returned no score on hidden data |

Before this window, the only instrument that could tell me no was the public score.
It could say it only at $10^{-3}$, and only after I spent a slot.
In this window a second instrument existed.
It refused at $10^{-5}$, on reasons written down before the experiment ran, and each refusal named its reason.

It did not replace the board.
The node-field result was a board result, because at the time the machine could not move the point set.

What changed is narrower, and I think it matters more.
A local rule overruled me on things I was confident about.
It could not yet choose, for four reasons visible by the end of July.
Its four approvals sat on four different local reference graphs.
Several of its policy cross-fits were movie-out within seen embryos, and the hidden test is an unseen one.
Its approvals were $10^{-4}$ in size, so for most of them nothing outside the machine could check the sign.
And it had no gate on inference cost, so the approval I spent the most slots on could not return a score on hidden data.

A rule that can refuse is the first step off the board.
It is not yet a rule I can trust to choose.

---

## 11. What the Period Established

### Established

1. Exact replay of the deterministic graph stages on held-out predictions is worth $+0.052187$ over the raw model graph.
2. Much of the edge-FN residual is unreachable by association: $11{,}020$ of about $25{,}150$ on the 07-17 replay have both endpoints unmatched.
3. A strictly out-of-fold, cross-fitted selector can still fail on population: firing rate $0.204\%$ calibrated, $5.456\%$ deployed.
4. Component metrics do not track graph score: parent top-1 gains became a sign reversal and a $32\times$ attenuation.
5. Restoring nodes is not a score strategy under the node-count adjustment. Three operators raised node recall and lowered the score.
6. In this pipeline the graph filter, not the solver, decides what counts as a division.
7. The division term is nearly unused: $0.00046$ of its $0.1$ weight, on a graph carrying $12{,}794$ forks against $151$ annotated divisions.
8. The same proposals are worth about $13\times$ more when the solver arbitrates them than when inserted afterwards.
9. A policy can pass every scientific gate and return nothing from the board: seven submissions, no score.

### Supported but Unconfirmed

1. Whatever the second detector adds arrives through the shared node field, not association reweighting. The evidence is a board ablation pair that reads the same score and seventeen association submissions that never exceeded the approximate single-seed figure. The addition is at the edge of the board's resolution, with no local measurement yet.
2. The board registers the division channel where promoted edge levers of similar local size did not: the ensemble and its single-seed control both read $0.916$, while the fragment matcher ($+0.0006325$) read the score of its base configuration. That is a direction, not a rate.
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

The machine Note 2 asked for mostly refused, on grounds written before each experiment ran.
It refused a router with a real subgroup effect, a diverse third seed, and my own top-ranked next project two days after I ranked it first.
But it judged only what it was handed.
In the third week of July it could not yet pose which points exist, and the one approval the board could see came from the division term, which my graph had barely used.

The refusals that surprised me most share one sentence.
**Every metric that improved and then failed was measured on something other than what the policy would change.**
Sparse annotated context against the dense graph, for the selector.
Parent-choice groups against connected components, for the flow gate and the parent matcher.
Classification quality against intervention utility, for the proposal model.
Sparse edge-utility proxies against the official score, for the tracklet admission rules.
Out-of-fold machinery guards against training on the evaluation data.
It does not guard against evaluating on the wrong object.

So the question I am carrying into August is not "which policy passes?" It is:

```text
On which population will this rule actually fire,
and have I measured it there?
```

Underneath it is a measurement problem I can no longer defer.
Seven local reference graphs, spanning $0.60$ to $0.74$, are in use and none is reconciled to the others.
Until they are, a promotion decided at $10^{-4}$ is partly a decision about which reference graph I happened to use.

Series:

- [Part 1: Learned Lineage Graphs and Metric-Aware Repair]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)
- [Part 2: From a Leaderboard Plateau to OOF Structural Diagnostics]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/)
- **Part 3: What the OOF Machine Refused**
