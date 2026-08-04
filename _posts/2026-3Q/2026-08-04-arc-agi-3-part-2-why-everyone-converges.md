---
title: "ARC-AGI-3 Part 2: Why Everyone Converges on the Same Strategy"
date: 2026-08-04 21:00:00 +0900
categories: [AI, Kaggle]
tags: [arc-agi, arc-agi-3, benchmarks, reinforcement-learning, agents, world-models, reverse-engineering, agi]
math: true
pin: false
---

# ARC-AGI-3 Part 2: Why Everyone Converges on the Same Strategy — Reading the Rules from the Bottom Up

> **This is a working note.** Part 1 drew a map of the competition; this one is a record of six weeks of submitting, failing, and reading the scoring engine source that ships with the competition data. The short version: my four submissions scored 0.86 / 0.88 / 0.89 / 0.93, and that narrow band turned out to be the key to understanding how this competition actually works. All numbers are snapshots as of August 4, 2026.

Part 1: [ARC-AGI-3: The Benchmark Nobody Has Solved Yet]({{ site.baseurl }}{% post_url 2026-2Q/2026-06-24-arc-agi-3-the-benchmark-nobody-has-solved %})

## 0. What happened in six weeks

The board has moved quite a bit since I revised Part 1 in late July.

- The leaderboard swelled to 2,046 teams. While first place sits at 1.86, a new second place appeared at 1.69 — with only 7 submissions — and third at 1.64. The rank-100 bar rose from 1.24 to **1.29**, and rank-300 from 0.99 to **1.08**. Textbook inflation.
- The middle of the board is still packed with copies and shallow variants of the Milestone #1 winner (the Duck). The copies score anywhere from 0.79 to 1.21.
- Another full-code notebook was published at LB 1.17, together with a 300-game run-diagnostics dataset. Openness begets openness.
- On July 29, OpenAI reported that GPT-5.6 Sol's public-set score jumped **from 13.3% to 38.3% by flipping two API settings**, with zero weight changes. This result becomes important later.
- Meanwhile I submitted four times, survived two infrastructure accidents, and watched one "mathematically perfect" idea die to a single line in the rules.

This post starts from a simple question: **why are two thousand teams submitting what is essentially the same program?** At first it looked like laziness or free-riding. I now read it differently. The convergence is close to a necessity manufactured by the rules — and if that is true, the way out must also be found inside the rules.

## 1. The monoculture, observed

Numbers first. On June 30, Tufa Labs' Duck (a local Qwen3.6-27B plus a Python REPL harness) won Milestone #1 at 1.21%, and per the rules the full code went public. Here is what the following month looked like:

| When | Leaderboard landscape |
|---|---|
| Late June | Heuristic/BFS entries in the 0.1–0.5 range, top around ~1.2 |
| Mid July | The Duck-copy plateau forms: 295 teams at 1.0+, copies spread over 0.79–1.21 |
| Early August | Differentiation on top of the plateau: a handful above 1.5, leader at 1.86, cutoffs still rising |

The organizers did not fight the spread. They explicitly advised that the fastest path into Milestone #2 is to copy and modify a public submission. This is not an accident; it is the design. The milestone prizes require open-sourcing precisely to buy this diffusion.

The interesting question, as a participant, is the next one. If copying is this easy, why do so few teams get *past* the copy? Why did every alternative approach die out? Answering that required looking at the rules, not the leaderboard.

## 2. The mechanics of convergence, derived from the rules

### 2.1 The constraints fold the design space

The hard constraints of the Kaggle track:

1. No internet during evaluation. No hosted APIs; local open-weight models only.
2. One GPU (RTX Pro 6000), roughly a 9-hour total budget.
3. The hidden evaluation consists of **110 independent runs**. The execution framework that ships with the competition names this constant explicitly, and the run set contains multiple clones of the same underlying games.
4. Runs are scheduled with concurrency 28, and each run has a wall-clock cap of 7,920 seconds.

These numbers look unrelated until you do the arithmetic, at which point a single blueprint appears:

$$
\underbrace{\left\lceil \tfrac{110}{28} \right\rceil}_{4\ \text{waves}} \times 7{,}920\,\text{s} = 31{,}680\,\text{s} \approx 9\,\text{h}, \qquad 7{,}920 = \tfrac{32{,}400}{4} - 180.
$$

The per-run cap is not an arbitrary number — it is **nine hours divided into exactly four waves**. The budget is already fully spent. Raise the cap and you fail to finish the waves; unplayed games score zero.

That exposes the real bottleneck. Twenty-eight runs share one GPU, so the resource a run actually receives is not time but **tokens**. Using throughput I measured under submission-identical conditions (28 cloned runs, competition mode, locally simulated):

$$
\text{tokens/run} \approx \frac{204\ \text{tok/s} \times 7{,}920\ \text{s}}{28} \approx 5.8\times 10^{4}.
$$

Measured values agreed: 50–60k tokens per run, which converts to roughly **30–150 actions per run**. That is more than three times poorer than the environment I had been measuring in before (a 4-game commit run, ~190k tokens per run). In other words, many participants — myself included — **had been measuring their agents somewhere other than where they are scored.**

Under this constraint set, list the designs that survive and there is almost nothing to choose from:

| Design basin | Cause of death |
|---|---|
| Frontier-API harnesses | No internet. Dead on arrival. |
| Large (70B+) local models | Throughput collapses; split 28 ways, a run gets a few dozen actions. |
| Pure heuristics / BFS | Real-world scores 0.1–0.5. No goal inference, so no deep levels. |
| CNN + RL | Won the preview — and its own authors abandoned it for the main event. Section 4 explains why, with math. |
| **A ~27B local LLM plus a thin harness** | The only basin left. All three Milestone #1 winners live here. |

That is the first reason for convergence. **It is not that the design space was wide and everyone happened to crowd into one corner; the constraints folded the space down to a single basin.** The only difference between teams was who arrived first.

### 2.2 The open-source rule creates a copying equilibrium

The second reason is game-theoretic. Milestone prizes force disclosure, and the leaderboard is best-of-N: only your maximum survives, so resubmission has no downside.

It matters here that the total score is an average over 110 runs. Individual runs are wildly noisy — I have watched the same game score 0.583 / 0.444 / 0.015 across three tries — but averaging 110 of them shrinks the variance, per the central limit theorem. My four submissions landed within a 0.07-wide band (0.86–0.93). If the per-submission standard deviation $\sigma$ were as large as 0.176, the probability of three samples landing within 0.07 of each other is about 4%. The estimate consistent with observation is $\sigma \approx 0.05$–$0.10$.

With that $\sigma$, best-of-N expectations follow:

$$
\mathbb{E}[\max_{i\le N} S_i] \approx \mu + \sigma\, a_N, \qquad a_{10}\approx1.5,\; a_{24}\approx1.9,\; a_{60}\approx2.2.
$$

With $\mu = 0.89$ and $\sigma = 0.10$, submitting 24 times yields an expected best of about **1.08**. Meanwhile the current #2 sits at 1.69 after seven submissions. Two conclusions drop out simultaneously:

1. **Copying plus resubmission gets you to 1.0–1.2.** Entry cost is near zero, so copying is rational even for casual participants. That is why the plateau is thick.
2. **Nothing above ~1.4 is explained by luck.** The gap between the plateau and the top is a real difference in engine mean $\mu$, not draws. There is a region copying can never reach.

The second conclusion cost me something to accept. For a while I told myself my scores were low because I had few samples. The moment four samples clustered tightly, that hypothesis died. My engine's $\mu$ was simply lower.

### 2.3 What the "abuse-like" thing actually is

So what should we call this convergence? It looked like abuse or free-riding at first. Having now read the rules to the bottom, my conclusion is different.

- The milestone disclosure requirement is a **deliberate knowledge-diffusion device**, and it worked exactly as designed.
- Best-of-N scoring with small $\sigma$ makes copy-and-resubmit a **cheap, if not dominant, strategy**.
- The constraint geometry never left room for design diversity to grow in the first place.

The monoculture is neither collusion nor cheating. It is **the shadow the rules cast**. It does have one side effect, though: everyone runs the same code, and **nobody reads it**. That side effect became my opportunity — which is the next section.

## 3. Reverse engineering: I actually did it, and here is the honest accounting

"Can reverse engineering break this game open?" First, split the target:

| Target | Feasible? | Notes |
|---|---|---|
| The hidden games themselves | No | Invisible by design. They are the object of generalization, not of cracking. |
| Server-side scoring | Off-limits | Cannot be touched, should not be touched. |
| **The scoring engine & framework that ship with the competition** | **Entirely legal** | The source is included in the competition data. It is there to be read. |
| Your own run logs and artifacts | Obviously | Astonishingly, most people don't. |

I spent a few days on the bottom two rows. Four findings.

### 3.1 Finding ①: a mathematically perfect idea dies to one line

My first differentiation idea went like this. In the scoring engine, a card's score is the **maximum over its plays**:

```python
# scorecard: EnvironmentScoreList.score
return max(run.score for run in self.runs)
```

So: immediately after winning a game, strip the wasted actions from the winning trajectory and replay the **shortest path through the observed transition graph** as a second play. The shortest path can never be longer than the winning route (which it contains as a subgraph), every replayed step can be verified against recorded frames, and any divergence aborts at zero cost because the recorded win still owns the max. A monotone, risk-free score machine — and since RHAE is squared, halving the actions quadruples the level score. I designed it, built it, and it passed every unit test.

Then I found this in another file of the engine:

```python
# api: competition mode
if scorecard.competition_mode and scorecard.has_environment(game_id):
    return None, False
```

**Competition mode refuses a second run of the same game id.** The second play you would take a max over cannot exist. A device that works flawlessly offline is structurally unreachable on the scored path. As a footnote: the public fork I build on ships its own win-then-replay machinery of the same family — equally dormant in competition mode. Of the hundreds of teams copying that code, I wonder how many know.

### 3.2 Finding ②: the determinism premise fails, measurably

Transition caches and replay schemes share one premise: same visible board plus same action equals same result. I cross-checked every (board, action) pair that appeared more than once in my run logs — 760 repeated pairs — and **270 of them (35.5%) produced different next states**. The spread across games is wide: some games are perfectly clean at 0%, others fail at 57%. There are timers and animation phases the frame does not show. Any cache keyed on the visible state alone gets betrayed probabilistically.

### 3.3 Finding ③: I had been measuring in the wrong place (token starvation)

This is where the arithmetic of §2.1 came from. The framework ships a local simulator that reproduces the submission's exact shape — cloned runs, competition mode, hidden baselines. Nobody uses it. Only after wiring it into my commit pipeline did I first measure that the scored regime is token-starved rather than time-starved, at ~100 actions per run. That single measurement inverted my entire improvement queue: machinery that trims 900-action thrash never even gets to fire in a 100-action regime.

### 3.4 Finding ④: the bug nobody saw because nobody reads

The most satisfying find. The harness contains a designed eviction-survival mechanism: each turn it asks the model to write labeled note lines (`World model:`, `Plan:`, …), harvests them from the assistant message, and re-injects the carried summary into later prompts. But the harvest reads only the response's `content` field — and a reasoning-parsing serving stack puts the model's entire output into `reasoning`, leaving `content` empty. In my logs, **465 out of 465 responses had zero content characters**. The memory mechanism has never fired, not once, since launch. In code that hundreds of teams are running.

OpenAI's Sol announcement independently priced this bug for me. Their official evaluation harness had been discarding reasoning after every move and truncating history; enabling "retained reasoning" and "compaction" — two settings — took the score from 13.3% to 38.3%. **Three times the score, just from not erasing memory.** Our harness's dead channel is the same disease.

### 3.5 So, was reverse engineering worth it?

An honest ledger.

**Cost**: a few days of source reading and log forensics. No GPU hours, no submission slots.

**Return**:
- Two dead ideas (replay banking, naive transition caching) killed **before** submission. One submission costs ~11 GPU-hours and a fifth of the daily quota; three or four misdirected submissions would have exceeded the entire cost of the reading.
- The measurement point corrected — every A/B after this actually means something.
- Two legal levers discovered (cross-clone transfer and the dead-channel repair — see §5).

**Against the counterfactual** it is even clearer. Without reverse engineering, the best available move is resubmission, whose expected gain is $\sigma(a_N - a_4) \approx 0.05$–$0.15$ points. Reverse engineering does not target $\sigma$; it targets $\mu$ itself. Once you accept that the gap to the top *is* a gap in $\mu$, reading the rules to the bottom stops being optional.

**The limits are just as clear.** This is not "breaking" anything. The hidden games and the server-side scorer are uncrackable by design — correctly so. What reverse engineering does is **find the slack the rules left behind before others do**. The clone-family structure is the canonical example of that slack.

## 4. The RL angle: what this benchmark says to reinforcement learning

Written formally, this is a pure RL problem: a partially observed MDP, sparse rewards, unknown transition function. And then three rules kill the classical RL recipe with surgical precision.

**First, one episode per environment.** The second-run refusal above means no hidden game is ever played twice. No replay buffer, no policy improvement across episodes — by definition.

**Second, exploration carries a price tag.** Differentiate the level score $s = \min((h/a)^2, 1.15)$ with respect to action count:

$$
\frac{\partial s}{\partial a} = -\,\frac{2h^{2}}{a^{3}},
$$

so near $a \approx h$ the marginal cost of one action is about $2/h$. On a level with a 20-action baseline, one exploratory click costs ten percent of that level's score. Whether it is $\varepsilon$-greedy or a novelty bonus, "just try it and see" is itself a penalty. In this benchmark, regret is not something you log — **it is engraved into the score, squared.**

**Third, the budget pushes learning inside the forward pass.** ~100 actions and 50–60k tokens per run. There is no time to take gradient steps. If learning happens at all, it must happen in-context, as inference.

Put the three together and the fate of the preview-winning CNN+RL approach explains itself. Value-based RL needs thousands of steps per environment to learn anything even in toy settings; this benchmark offers **a single episode of ~100 steps**. The sample complexity is short by more than two orders of magnitude — which is presumably why its own authors retired it.

So I read the current winning paradigm this way: **it is not the abandonment of RL; it is meta-RL / in-context RL made concrete.** The correspondences are exact:

| RL concept | Its physical form in the current harness |
|---|---|
| Policy | The LLM's forward pass (the context *is* the policy state) |
| Replay buffer | The `transitions` variable injected into the sandbox |
| Cross-episode transfer | Experience sharing across clone families (§5's transfer) |
| Options / skills | Published level segments (verified action sequences) |
| Distributed actors sharing experience | The process-global family store — structurally isomorphic to A3C's shared server |
| Information-directed exploration | The prompt's instruction to pick actions on which competing hypotheses disagree |

Sol's threefold jump reads naturally in this frame too. A harness that discards reasoning every move is **an RNN whose hidden state is reset at every step**. Letting it persist tripled the score — strong evidence that the bottleneck here is not model size but **continuity of memory**.

There is a place for RL to come back, but offline rather than online: distilling action priors or dynamics models from public-game trajectories, and eventually fine-tuning the local weights on one's own traces — legal, since local open weights are the whole premise. Distribution shift is waiting there, as it always is; I have queued it for after Milestone #2.

## 5. What we actually built: v6

Where all of the above converges is what I shipped this week — two things.

### 5.1 Cross-clone transfer — the one "experience sharing" channel the rules allow

Replay banking (§3.1) is dead, but one structural fact survives: the scored run contains **multiple clones of the same underlying game**, and clones carry distinct game ids, so they do not trip the second-run refusal. Therefore:

- The first clone of a family (the scout) publishes a waste-stripped action segment to a process-global store each time it clears a level.
- Sibling clones replay that segment mechanically on their own (single) run, **verifying every step** against the recorded states; the first divergence aborts the replay and falls back to the normal loop, so a wrong guess costs a few actions.
- A scheduler reorders the game list so each family's scout dispatches before its siblings.

The expected payoff computes naturally in the token-starvation frame. A sibling that clears the early levels by replay spends its entire 50–60k-token budget on **levels nobody has reached**, and since level weights grow linearly with the level index, marginal score per token is highest exactly there. On heavily non-deterministic games (§3.2), verified replay aborts early on its own — a built-in off switch for the games where the premise fails.

An honest footnote: the skeleton of this machinery already existed in the public fork I build on. It simply was never armed. My contribution was recognizing it as the only rules-valid experience-sharing channel, verifying it, and turning it on. The monoculture's paradox again: hundreds of teams run this code; almost none has flipped the lever sleeping inside it.

### 5.2 Repairing the dead channel — and the trap the review caught

The bug of §3.4 has exactly one patch point. At the single gateway every response passes through, if `content` is empty and the reasoning parses to labeled note lines, surface **only those lines** as `content`. The existing harvest-and-reinject code does the rest untouched.

One lesson from this is worth recording. After implementing it, I ran an adversarial review — which discovered that the bundle's label parser is **greedy**: unlabeled follow-on lines keep gluing onto the previous label, so the entire chain-of-thought following a `Plan:` line (1,709 characters, measured) could be harvested as "the plan". I was one code review away from shipping a bug fix that injects two kilobytes of noise into every prompt. It now carries a 280-character per-key cap, the same value as the bundle's own summary cap. Trust code whose tests reproduce the real parser's semantics — not code whose tests merely pass.

### 5.3 Measuring where the scoring happens

The commit pipeline now runs the competition simulator (28 cloned runs, competition mode, hidden baselines) instead of a 4-game offline slice. Every future improvement can be A/B'd at the scored operating point. The simulator cannot compute RHAE client-side (baselines are hidden), so the metrics to read are level depth, actions, and token distributions.

## 6. The plan from here

Milestone #2 lands on September 30 — about eight weeks. Working backwards:

**Now (this week): measure.** From the v6 submission and simulator logs, check the transfer publish/adopt events and the actual contents of the revived memory channel. This commit already showed the first live segment publishes ever (ten of them, 5–59 actions each). The next question is sibling adoption rates, and whether adoption actually shifts the level-depth distribution.

**Short term (2–3 weeks): memory and transfer, strengthened.**
- *Text dossier transfer*: for families where action replay dies to non-determinism, transfer the learned rules as text ("press the green blocks to open the door") instead of actions.
- *Eviction → compaction*: summarize old history instead of truncating it. Sol's 3× shows the ceiling of this direction; what a local 27B recovers is an experiment.
- *Animation signatures*: much of the 35.5% non-determinism is animation phase. Exposing a signature of the intermediate frames after each action should dissolve a good share of the "fake" non-determinism.

**Medium term (before M2): throughput and priors.**
- In a token-starved regime, tokens per second is the amount of intelligence you get to run. I am watching for Blackwell-native 4-bit (NVFP4) weights of our model; when they appear, the throughput gain gets verified in the simulator before switching.
- Offline distillation of a small action prior from public-game trajectories (mine plus the published 300-game diagnostics).

**Two standing principles.** First, an improvement measured anywhere other than the scored operating point does not count as an improvement. Second, submissions are the final stage of hypothesis testing, not lottery tickets. That phase is over.

## 7. Closing

Six weeks ago I looked at the leaderboard and asked why everyone was submitting the same thing. The answer I have now: the constraints folded the design space into a single basin, the disclosure rule turned the basin's floor into a commons, and best-of-N scoring made copying cheap. The convergence is not the participants' failure; it is the shadow of the rules.

And the way out was also inside the rules. One line of the scoring engine killed one of my ideas; another structural fact (the clones) opened a new one; and inside code nobody reads, I found a memory mechanism that had never once fired. Reverse engineering here was not a trick — it was **the act of reading the problem statement precisely**. If this benchmark measures efficient learning in unfamiliar worlds, then perhaps the first unfamiliar world a participant must learn is not the games. It is the rules.

The next post will likely cover the measured results of transfer and the memory channel, and the eviction-to-compaction experiment. Whichever way the numbers fall, I will write them down.

---

### Resources

- [ARC Prize 2026 - ARC-AGI-3 on Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- [Milestone #1 official results](https://arcprize.org/blog/arc-prize-2026-milestone-1) · [Duck harness code](https://github.com/Tufalabs/duck-harness)
- [Scoring methodology](https://docs.arcprize.org/methodology)
- [Official GPT-5.6 Sol results](https://arcprize.org/results/openai-gpt-5-6-sol) and [how two settings tripled it (the-decoder)](https://the-decoder.com/openai-claims-gpt-5-6-sol-beats-opus-5-on-arc-agi-3-with-its-latest-api-and-two-additional-settings/)
- [Executable World Models for ARC-AGI-3](https://arxiv.org/abs/2605.05138) (accepted at AGI 2026, code released)
- [Tycho: Active Abstraction with Programmatic World Models](https://arxiv.org/abs/2607.28287)
- [Explore Before You Solve: The Speed–Depth Trade-off in Epistemic Agents](https://arxiv.org/abs/2605.25931)
- [The Schema harness report](https://schema-harness.github.io/) — note: self-reported, not verified by ARC Prize
