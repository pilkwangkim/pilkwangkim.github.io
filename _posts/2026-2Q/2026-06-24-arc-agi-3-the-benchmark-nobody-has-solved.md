---
title: "ARC-AGI-3: The Benchmark Nobody Has Solved Yet"
date: 2026-06-24 21:00:00 +0900
categories: [AI, Kaggle]
tags: [arc-agi, arc-agi-3, benchmarks, reinforcement-learning, agents, world-models, agi]
math: true
pin: false
---

# ARC-AGI-3: The Benchmark Nobody Has Solved Yet

> **Status note (written 2026-06-25).** ARC-AGI-3 is an active competition and the scaffolding around it is still moving. I have checked the durable pieces below against the official ARC Prize pages, docs, and technical report, but leaderboard positions, notebook runtimes, and milestone mechanics can change. Treat the exact leaderboard figures as timestamped, not permanent.

Competition link:  
[ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)

Official benchmark page:  
[ARC Prize 2026 - ARC-AGI-3 Competition](https://arcprize.org/competitions/2026/arc-agi-3)

Technical report:  
[ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence](https://arxiv.org/abs/2603.24621)

> **One-paragraph version.** ARC-AGI-3 reframes "reasoning" as something you have to *do*, not something you *output*. An agent is dropped into an unfamiliar 2D grid game with no instructions and is scored on how *efficiently* it learns to win, relative to a human who has never seen the game either. The striking fact — the one that should anchor everything else you read about it — is that as of mid-2026 **every system, frontier LLMs and purpose-built agents alike, scores below ~2%.** That is not a measurement artifact. It is the entire point of the benchmark, and it is why I think this is the most interesting open problem on Kaggle right now.

## Why I'm writing this

I'm working through ARC-AGI-3 seriously, and writing is how I think. This post is my attempt to lay out — as precisely as I can — what the competition actually is, how the scoring really works (there is a unit confusion that trips up almost everyone, including me), what the current state of the art looks like once you strip away the leaderboard theater, and how I'd approach it. The framing is opinionated on purpose. Where I'm stating fact I'll try to be careful; where I'm stating a bet, I'll say so.

If you take one thing away: **do not trust the leaderboard number you see attached to a notebook.** I'll explain why in detail, but the short version is that the public scoreboard is a mix of stale scores computed under an older scoring rule, scores inflated by overfitting to the public games, and — until recently — scores inflated by an exploit that has since been patched. The honest signal is much lower and much more uniform than it looks.

### A beginner's map

If ARC-AGI-3 is new to you, here is the entire thing in ordinary language:

| Term | Plain-English meaning |
|---|---|
| **ARC** | A family of grid-based reasoning benchmarks. Think small colored boards, not natural-language trivia. |
| **AGI** | Here, not a magic word. The benchmark uses it in a narrow operational sense: can a system acquire a new skill efficiently? |
| **Task / game / environment** | One small world with its own hidden rules. The agent sees a grid and can take actions. |
| **Level** | A stage inside one game. Later levels usually reuse and combine earlier mechanics. |
| **Agent** | Your program. It observes the current frame and chooses the next action. |
| **Action** | A move such as up/down/left/right, interact, click a coordinate, undo, or reset. |
| **World model** | The agent's internal guess about how the game works: "if I press this, that object moves", "blue blocks are walls", "the goal might be to align these shapes." |
| **Generalization** | Doing well on games the agent has never seen before, not memorizing the 25 public games. |

The easiest mental picture is this: imagine opening a strange puzzle game with no manual. You press keys, watch what changes, infer the rules, guess the goal, and then solve it. ARC-AGI-3 asks an AI system to do that, and then asks whether it did it with roughly human efficiency.

### What background do you actually need?

You do **not** need to begin as an RL researcher. In fact, if you start by reading a stack of deep-RL papers before you have logged a single game frame, you will probably make slower progress. ARC-AGI-3 is research-grade, but the first useful steps are much more concrete.

Here is the minimum background I would want before touching an agent:

| Background | Why it matters | Beginner version |
|---|---|---|
| Python | The starter kit is Python-first. | Be comfortable editing one file, running commands, and reading stack traces. |
| Arrays / grids | Observations are 2D integer grids. | Know how to compare two arrays and find changed cells. |
| Search | Agents need to try action sequences. | Breadth-first search, depth-limited search, and visited-state sets are enough to start. |
| Logging | You cannot debug an interactive agent by staring at the final score. | Save frames, actions, state hashes, and why an action was chosen. |
| Basic ML vocabulary | Many strong approaches use CNNs, value models, or world models. | You only need to know what these models are trying to predict. |
| Kaggle code competitions | Submissions are notebooks that Kaggle runs. | Understand that your code must be self-contained and internet-free at evaluation time. |

And here is what you **do not** need on day one:

- You do not need a giant language model. Hosted APIs are unavailable during final evaluation anyway.
- You do not need a polished neural architecture. A bad logger plus a fancy model is worse than a simple graph search you can inspect.
- You do not need to solve all 25 public games manually. Playing a few by hand is useful; memorizing them is a trap.
- You do not need to understand every Kaggle forum argument before writing code. Start with the official docs, the starter kit, and one local game.

A good beginner mindset is: **treat every game as an unknown machine.** Your agent's first job is not to be clever. It is to run experiments, record what changed, avoid repeating known-useless actions, and preserve enough memory to make the next experiment better.

That is also why this benchmark is unusually educational. It forces the same loop that good debugging forces:

```text
observe carefully
make a small intervention
record the effect
update the hypothesis
try the next intervention
```

If you can build that loop, even with hand-written heuristics, you have already built the skeleton of an ARC-AGI-3 agent.

## 1. The lineage: from passive puzzles to active games

ARC-AGI (the *Abstraction and Reasoning Corpus*) is François Chollet's attempt to operationalize a specific definition of intelligence: not *skill*, but **skill-acquisition efficiency** — how quickly a system can pick up a new skill from very little data, leaning only on a small set of innate "core knowledge priors" (objectness, basic geometry, counting, agentness).

- **ARC-AGI-1 (2019)** — static. You see a handful of input→output grid pairs, infer the transformation, and produce one output grid. It surfaced the emergence of large reasoning models; o3 was the first system to crack it convincingly.
- **ARC-AGI-2 (2025)** — still static, but harder and more compositional (multiple interacting rules, contextual rule application). It tracked how far test-time compute scaling could push reasoning. The static format had a vulnerability: you could brute-force it by sampling thousands of candidate solutions in parallel.
- **ARC-AGI-3 (2026)** — interactive. The static format is gone. Now the agent must *explore* an environment, *build a model* of how it works, *infer the goal* with no instructions, and *plan* a path to that goal — correcting as it goes. This closes the parallel-sampling loophole: you can't pre-generate a million guesses for a game whose rules you don't yet know.

The conceptual shift is from "intelligence as pattern-matching on a fixed dataset" to "intelligence as adaptive behavior in an open-ended environment." That shift is exactly why current systems fall off a cliff.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/lineage.svg" alt="ARC-AGI lineage from static puzzles to interactive games" width="92%">
</p>

Here is a toy contrast.

In a static ARC task, you might be shown:

```text
input grid:   one red square on the left
output grid:  the same red square shifted to the right
```

The job is to infer the transformation and produce the missing output. You never act inside the world. You just answer.

In ARC-AGI-3, the equivalent problem is closer to:

```text
frame 0:   a red square, a blue door, a green button
action 1:  press right
frame 1:   the red square moved right
action 2:  press the button
frame 2:   the blue door disappeared
```

Now the agent has to learn from interaction. It must discover that movement exists, that the button has an effect, that the door was blocking something, and perhaps that the final goal is behind the door. That is a much harder contract than "complete this output grid."

## 2. What it actually measures

ARC-AGI-3 decomposes agentic intelligence into four capabilities, and a good agent needs all four:

1. **Exploration** — information is not handed to you; you have to act to obtain it.
2. **Modeling** — turning raw observations into a generalizable world model that predicts future states. (This is the through-line from earlier ARC generations.)
3. **Goal-setting** — identifying a desirable target state *without being told what it is*. The agent must decide "what to aim for" from intrinsic drive and environmental cues.
4. **Planning & execution** — mapping an action path from the current state to the inferred goal, and course-correcting under feedback.

The metric is **skill-acquisition efficiency**: how few actions you need to win a level, relative to a human baseline. Crucially, scoring is about **generalization** — agents are ranked on *unseen* games, not the public ones they could practice on. An agent that memorizes or overfits to the 25 public games gets you nothing on the leaderboard.

The four capabilities are not abstract slogans. They correspond to concrete failure modes:

| Capability | What a beginner should picture | What goes wrong without it |
|---|---|---|
| Exploration | Try actions to find which objects respond. | The agent repeats useless moves or never discovers the key mechanic. |
| Modeling | Remember cause and effect: "button opens door", "object falls after push." | The agent cannot predict what a move will do. |
| Goal-setting | Infer what winning probably means from the board. | The agent wanders even after understanding some mechanics. |
| Planning & execution | Choose a short sequence of moves toward that inferred goal. | The agent knows facts but cannot turn them into a solution path. |

Most current AI benchmarks mostly test the last line: can the model output the correct answer? ARC-AGI-3 tests the whole loop: act, observe, update, choose again.

## 3. The games

The environment is deliberately minimal so that the difficulty is all in the *reasoning*, not the interface:

- **Observation**: a grid up to 64×64, each cell an integer color in 0–15, delivered as a JSON frame with metadata. Origin (0,0) at top-left.
- **Actions**: `RESET` plus the standardized `ACTION1`–`ACTION7` interface —
  - `RESET` — restart the level,
  - `ACTION1`–`ACTION4` — simple directional moves (up/down/left/right),
  - `ACTION5` — a general interaction (select / rotate / attach / execute / …, context-dependent),
  - `ACTION6` — a click at an (x, y) coordinate in 0–63,
  - `ACTION7` — undo (was unavailable during the preview competition).
- **Scale**: the benchmark composition described in the technical report is 25 public-demo environments plus 110 hidden environments, with levels inside each environment escalating from introductory to more compositional.

For a non-programmer, the grid can be imagined as a tiny board game:

```text
0 0 0 0 0
0 2 0 3 0
0 0 1 0 0
0 4 0 0 0
0 0 0 0 0
```

The numbers are not quantities. They are colors or object types. The agent is not told "1 means player" or "3 means door." It has to infer those meanings from what happens after actions.

A single turn looks like this:

| Step | What happens |
|---|---|
| 1 | The environment sends the agent the latest frame: grid, metadata, available actions, state. |
| 2 | The agent updates its memory: what changed since the previous frame? |
| 3 | The agent chooses one action, for example `ACTION1` or `ACTION6` with coordinates. |
| 4 | The environment applies the action and returns a new frame. |
| 5 | The loop repeats until the level is won, lost, reset, or the action budget is exhausted. |

The important limitation is that the action names are generic. `ACTION5` might mean "pick up" in one game and "rotate" in another. `ACTION6` means "click somewhere", but the frame does not hand you a list of useful click targets. So the first job is often not solving the puzzle; it is discovering the controls.

**The public/private split** is the part you have to internalize:

- **25 public games** — you get the source; you can train and debug on them.
- **110 private games** — never seen by you or your agent. Of these, **55 form the Public Leaderboard** and the other **55 the Private Leaderboard** (the one that decides final standings, in classic Kaggle public/private style).

So even the "public" leaderboard you watch is computed on **unseen** games. This is why solving a pile of public levels can still yield a competition score of **0.00**.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/split.svg" alt="ARC-AGI-3 public demo, semi-private, and fully private split" width="92%">
</p>

### What the starter kit gives you

The official path for beginners is the [ARC-AGI-3 Kaggle Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter). The docs describe it very deliberately: edit one Python file locally, watch it play real game environments, then push a Kaggle notebook submission. That matters because it keeps the first loop small.

The official quick-start shape is:

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git
cd ARC-AGI-3-Kaggle-Starter
make setup
make play-local
make submit
make status
```

The important file is usually:

```text
agent/my_agent.py
```

That is where you implement the agent's decision rule. The rest of the starter exists to load games, run the loop, package the notebook, and submit it through Kaggle. This is good design: it prevents you from confusing "competition plumbing" with "agent intelligence."

For a first attempt, I would split the workflow into three modes:

| Mode | What it is for | What to remember |
|---|---|---|
| **Local offline** | Fast iteration on public games. | Recommended for development; no online scorecards, but no rate limits. |
| **Online API / scorecard** | Shareable scorecards and replays. | Useful for inspection, slower and rate-limited. |
| **Kaggle submission** | Actual leaderboard scoring. | Scarce; do not use as your main debug loop. |

The official docs say local/offline execution is the recommended development path because it is fast and avoids online rate limits. Use Kaggle only after your local harness can answer basic questions:

- Which games did the agent attempt?
- Which levels did it complete?
- How many actions did each level take?
- How many actions changed the frame?
- How often did it repeat a state?
- When it clicked, where did it click and why?
- Did it behave differently on holdout public games than on development public games?

If your system cannot answer those questions, a leaderboard score is almost useless. It can tell you that the agent failed, but not *why*.

### What an agent should remember

The first serious upgrade over random is not a neural network. It is memory. ARC-AGI-3 punishes repeated, uninformative actions, so a beginner agent should record at least:

| Memory item | Example | Why it helps |
|---|---|---|
| State hash | hash of the grid and relevant metadata | Avoid revisiting the exact same frame without purpose. |
| Action outcome | `(state_hash, action) -> changed / no-op / error / win` | Stop repeating actions known to do nothing. |
| Changed cells | coordinates whose values changed after an action | Identify the controlled object and reactive objects. |
| Available actions | action set returned in the frame metadata | Narrow the action space before sampling. |
| Level identity | current game and level, if exposed by metadata | Reset assumptions when the level changes. |
| Short trajectory | last N states/actions | Detect loops even when the frame changes slightly. |

The simplest no-op detector is just a comparison between the previous grid and the next grid:

```python
def grid_changed(prev_grid, next_grid):
    return prev_grid != next_grid
```

In real code, you will want a proper array comparison, and you will want to include metadata carefully. But the idea is that simple: if `ACTION1` from this exact state produced the exact same state, do not keep choosing `ACTION1` from that state.

This sounds trivial, but it is a large part of why pure random agents waste their entire action budget. Random exploration does not know the difference between "I learned something" and "I pressed into a wall again."

## 4. Scoring: RHAE, and the unit that trips everyone up

Performance is **Relative Human Action Efficiency (RHAE)**. For a single level $\ell$:

$$
s_\ell \;=\; \left[\min\!\left(\frac{h_\ell}{a_\ell},\, 1.15\right)\right]^{2}
$$

where $h_\ell$ is the (upper-median) action count of first-time human players and $a_\ell$ is your agent's action count. Three design choices, each with strategic teeth:

- **It's squared.** A raw efficiency ratio of 0.5 becomes 0.25. Being twice as slow as a human doesn't cost you half — it costs you three-quarters. *Solving the level is not enough; you have to approach human efficiency.*
- **It's capped at 1.15** (this cap was raised from an earlier 1.0). You can score slightly above "human" by beating human efficiency, but a freak two-action win can't blow up the average.
- **There's an action budget**: you are cut off after $a_\ell = 5\,h_\ell$ actions (five times the human median). Run past it without winning and the level scores **0**.

A game's score is a **level-index-weighted** average — later levels count more — with a per-environment cap. If a game has $n$ levels and the agent completes the first $k$ sequential levels, then:

$$
S_g
\;=\;
\min\!\left(
\frac{\sum_{\ell=1}^{k} \ell}{\sum_{\ell=1}^{n} \ell},
\frac{\sum_{\ell=1}^{n} \ell\,s_{g,\ell}}{\sum_{\ell=1}^{n} \ell}
\right),
\qquad
S \;=\; \frac{1}{G}\sum_{g=1}^{G} S_g .
$$

The first term prevents a system from earning a high game score by being superhuman on early levels while failing later ones. The weighting rewards *transfer*: getting deep into a game, where mechanics compound, is worth disproportionately more than skimming level 1 of everything.

In code, the shape is roughly:

```python
def rhae_score(games):
    """Return ARC-AGI-3 score as a fraction; multiply by 100 for percent."""
    game_scores = []

    for levels in games:
        weighted = 0.0
        completed_weight = 0
        total_weight = sum(range(1, len(levels) + 1))

        for i, level in enumerate(levels, start=1):
            human_actions, agent_actions, solved = level
            if solved and 0 < agent_actions <= 5 * human_actions:
                level_score = min(human_actions / agent_actions, 1.15) ** 2
                weighted += i * level_score
                completed_weight += i

        env_cap = completed_weight / total_weight
        env_score = weighted / total_weight
        game_scores.append(min(env_cap, env_score))

    return sum(game_scores) / len(game_scores)
```

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/rhae.svg" alt="ARC-AGI-3 RHAE scoring pipeline" width="92%">
</p>

### A numerical example

Suppose a human baseline solves a level in 10 actions.

| Agent actions | Raw ratio $h/a$ | Squared score | Interpretation |
|---:|---:|---:|---|
| 10 | 1.00 | 1.00 | Human efficiency. |
| 20 | 0.50 | 0.25 | Twice as slow becomes only 25% credit. |
| 50 | 0.20 | 0.04 | Five times slower becomes 4% credit. |
| 51 | cutoff | 0.00 | Past $5h$, the level gets zero. |

This is the first counterintuitive point: **solving is not enough**. If a brute-force agent eventually wins after many random moves, it may still score almost nothing because the benchmark is measuring how quickly it acquired the skill.

Now suppose a game has five levels, and the agent only solves levels 1 and 2. Even if it solves those first two perfectly, its maximum game score is capped at:

$$
\frac{1 + 2}{1 + 2 + 3 + 4 + 5}
\;=\;
\frac{3}{15}
\;=\;
20\%.
$$

That cap is there because later levels are where the game usually tests whether the agent really understood the mechanic. Early levels can be tutorial-like; late levels combine rules.

### The unit confusion

Scores are reported as **percentages**, and this genuinely confuses people on the forums. A leaderboard entry of `0.46` means **0.46%** (a fraction of 0.0046), *not* 46%. The maximum is 100.00% (fraction 1.0, with the 1.15 per-level cap allowing slight local overshoot). I made this mistake myself while researching this post: a notebook "scoring 0.46" is scoring **0.46%**, which is a completely different — and far more sobering — claim. **Right now the entire field sits below ~2%.** Late-June public high-score reports are in roughly the **1.2%** band, not the 12% or 46% that a casual reading might suggest.

Internalize that and the whole landscape snaps into focus: nobody has solved this. The leaderboard isn't a ranking of strong generalizers; it's a ranking of who is least bad at an unsolved problem.

### What RHAE changes about development

RHAE makes ARC-AGI-3 very different from a normal "win/loss" game benchmark. A local run that says "level completed" is not automatically good. You need to ask how expensive the completion was.

Suppose two agents both solve a level with a human baseline of 20 actions:

| Agent | Actions used | Level score | What it tells you |
|---|---:|---:|---|
| A | 24 | $(20/24)^2 = 0.694$ | Slightly inefficient, but meaningful. |
| B | 90 | $(20/90)^2 = 0.049$ | It technically solved, but mostly by burning budget. |

If you only track completion count, A and B look equal. Under RHAE, they are not close. This is why your local reports should include:

- **completion count**
- **actions per completed level**
- **estimated RHAE per completed level**
- **no-op action rate**
- **repeated-state rate**
- **first state-changing action step**
- **first score-changing action step**
- **late-level reach rate**

The last metric matters because game score is level-weighted. Getting to level 3 or 4 is a different signal than repeatedly polishing level 1. In practical terms, I would rather have an agent that solves fewer early levels but reaches deeper levels on several games than an agent that over-optimizes a single public tutorial level.

### A small local validation protocol

The public games are not the leaderboard games, but they are still your only transparent development set. Treat them like a tiny dataset:

```text
public games
→ development split
→ holdout split
→ final sanity split
```

One simple split is:

| Split | Use | Rule |
|---|---|---|
| Development | Tune heuristics and models. | You may inspect logs freely. |
| Holdout | Compare candidates. | Do not tune after each result. |
| Final sanity | Check for accidental overfit. | Touch only before a real submission. |

Because there are only 25 public games, this split is statistically weak. But it is still better than using all public games for every decision. If an agent improves on the development games and collapses on the holdout games, that is exactly the warning you need before burning a Kaggle submission.

A beginner-friendly report might look like this:

```text
agent: graph_noop_v03
games: 18 development, 7 holdout

development:
  completed_levels: 14
  median_actions_per_completed_level: 37
  no_op_rate: 0.31
  repeated_state_rate: 0.18

holdout:
  completed_levels: 3
  median_actions_per_completed_level: 92
  no_op_rate: 0.62
  repeated_state_rate: 0.44
```

That is not a leaderboard-ready agent. But it is a *debuggable* agent. It tells you that the development games taught it some public-specific priors that did not transfer, and that the holdout failure is dominated by repeated no-ops and loops. That points to the next engineering step much more clearly than "score = 0.00" ever could.

## 5. The competition: rules, prizes, deadlines

ARC-AGI-3 is the headline track of **ARC Prize 2026** (total pool ~$2M across the ARC-AGI-2 and ARC-AGI-3 tracks). It runs as a **Kaggle code competition** with rules that set it apart:

- **No internet during evaluation.** This rules out calls to hosted models (GPT, Claude, Gemini) — a self-contained agent only. Frontier-model harnesses that need an API are simply ineligible.
- **Open-source to win.** Prize-eligible solutions must be open-sourced; the exact claim mechanics matter around milestone dates.
- **Notebook-only submission.** The Kaggle overview currently lists CPU and GPU notebooks at up to 9 hours of runtime, with internet disabled. The ARC Prize docs also provide a local starter path so you can test before burning submissions.
- **Timeline**: opened March 25, 2026; milestone checkpoints June 30 and September 30; final submissions due November 2; papers due November 8; results announced December 4.
- **Prize structure** (ARC-AGI-3 track): $850K total, consisting of a $700K Grand Prize for a 100% agent, a guaranteed $75K Top Score Award, and $75K in milestone prizes. The grand prize is, realistically, unclaimable this year — see the scores above.

The guaranteed ARC-AGI-3 pools break down as:

| Pool | Distribution |
|---|---|
| Top Score Award | $40K / $15K / $10K / $5K / $5K |
| Milestone #1 | $25K / $10K / $2.5K |
| Milestone #2 | $25K / $10K / $2.5K |

A practical consequence of "open-source to win": prize-eligible methods eventually have to become public. I would not over-specify the exact timing mechanics until the host clarifies milestone claim rules, but the broad incentive is clear: the leading approaches should become inspectable over the course of the competition rather than remaining permanently private.

If you have never done a Kaggle code competition, the key difference from a normal Kaggle tabular competition is this:

| Normal prediction competition | ARC-AGI-3 |
|---|---|
| You usually submit a CSV of predictions. | You submit code that Kaggle runs. |
| The test data is fixed rows. | The evaluation is an interactive game loop. |
| Your model can often be trained offline and only output numbers. | Your agent must act repeatedly, observe feedback, and adapt. |
| Internet is irrelevant after training. | Internet is disabled, so hosted API calls are not available. |

That last row matters. A notebook that calls GPT, Claude, Gemini, or any external server might be a useful research harness outside Kaggle, but it is not a valid final Kaggle submission if it depends on internet access. The submitted agent has to carry its machinery with it: rules, heuristics, small models, learned priors, search procedures, and local state.

## 6. The current landscape (with the theater removed)

Here is the honest state of the art:

- **Frontier LLMs: sub-1%.** At release, the technical report listed frontier models in the 0.10%-0.50% band on the semi-private evaluation: Anthropic Opus 4.6 (Max) at 0.50%, Google Gemini 3.1 Pro Preview at 0.40%, OpenAI GPT 5.4 (High) at 0.20%, and xAI Grok-4.20 at 0.10%. Reasoning models, which dominate most benchmarks, do *not* transfer cleanly to active skill-acquisition.
- **Purpose-built agents (preview): low, and they don't hold up.** The 30-day preview's winner, **StochasticGoose** (Tufa Labs, a CNN + sparse-RL action-prediction agent), scored **12.58%** on three hidden preview games — and then **dropped to 0.25%** on the full launch benchmark, right back into LLM territory. Second place, **Blind Squirrel**, scored 6.71% with a state-graph + value-model approach. The preview lead evaporated on the full set.
- **The whole Kaggle field: sub-2%**, with public high-score reports around the ~1.2% band as of late June 2026. The exact top score is volatile, but the qualitative fact is not: we are not looking at a mature leaderboard where the best systems are near human efficiency. A competitor's measurement of the best public notebook on ten public games came out to **0.66%**. Another reported solving ~15 public levels and still scoring **0.00** in the competition.

The single most important quote from the literature, paraphrased: *no approach has demonstrated clear generalization yet.* That is the opportunity.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/score-distribution.svg" alt="ARC-AGI-3 score distribution showing current systems below two percent" width="92%">
</p>

### Why current systems struggle

A frontier LLM is very good when the problem can be expressed as text and the needed knowledge already lives in its training distribution. ARC-AGI-3 removes both comforts.

| Usual LLM benchmark | ARC-AGI-3 |
|---|---|
| The task is stated in language. | The goal is not stated. |
| The model answers once. | The agent must act hundreds of times. |
| The input is mostly symbolic text. | The input is a changing grid world. |
| The model can lean on memorized patterns. | Private games are unseen and intentionally different from public demos. |
| Wrong answers are scored after the fact. | Bad actions consume the same limited action budget used for solving. |

The last row is brutal. In a normal benchmark, a wrong intermediate thought costs nothing if the final answer is right. In ARC-AGI-3, a wrong exploratory move is an action, and actions are the scoring currency. The agent has to spend actions to learn, but spending too many actions destroys the score. That is the central tension.

This is also why brute force looks much weaker than it sounds. You can try many moves, but each attempted move is visible to the scorer. If a human needs 12 actions and your agent needs 120, the squared efficiency term turns the credit into:

$$
\left(\frac{12}{120}\right)^2 = 0.01.
$$

That is 1% credit on that level, before level weighting and game caps. A brute-force win can be almost indistinguishable from a failure.

### Why the leaderboard "lies"

If everyone is sub-2%, why do you see notebooks with "0.4" attached? Three reasons, and you should hold all of them:

1. **Unit confusion** — "0.4" is 0.4%.
2. **Stale scores under an older rule** — Kaggle does not retroactively re-score historical submissions when the metric changes. The per-level score was changed to be *squared* (and the cap raised) partway through; old high entries kept their old-rule numbers, and the public-notebook ranking has been effectively frozen for months because new entries can't displace them under the harsher rule. The leaderboard is therefore an apples-and-oranges mix.
3. **A patched exploit** — for a while, some agents located the actual game *source code* on disk and ran exhaustive search against the real simulator (a white-box trick). That works on public games (whose source ships) but is dead on the private leaderboard games (API-only), and the dynamic-game-code-loading path has since been fixed. High historical scores that leaned on it don't reproduce.

So: anchoring on a published notebook's score is a mistake on at least three independent axes.

## 7. Approaches, and what's actually wrong with each

Before the approaches, translate the jargon:

| Term | Meaning here |
|---|---|
| **CNN** | A neural network that is good at local spatial patterns in grids or images. |
| **Sparse reward** | The agent gets useful feedback only rarely, usually when it wins or reaches a milestone. |
| **State graph** | A map of frames the agent has seen, with actions as edges between frames. |
| **Value model** | A model that estimates which state or action is closer to success. |
| **World model** | A learned simulator: "if I do action A in state S, what state will come next?" |
| **Intrinsic exploration** | Exploration driven by curiosity or uncertainty, not only by external score. |
| **Planning** | Searching over possible future action sequences before committing to one. |

The important distinction is between **reactive** systems and **model-based** systems. A reactive system mostly asks, "what action looks promising now?" A model-based system asks, "what do I think will happen if I take this action, then that action, then that one?" Humans lean heavily on the second style when learning a new game.

**(a) CNN + sparse-RL action prediction (the StochasticGoose lineage).** Learn a small CNN that predicts which actions change the frame, and bias exploration toward those. Hierarchical sampling (pick action *type*, then a click coordinate via a convolutional 64×64 head that keeps the 2D bias); a hash-deduplicated experience buffer for sample efficiency; reset the model per level; train with binary cross-entropy + light entropy regularization on sparse (level-completion) reward.
*Pros:* proven preview winner, sample-efficient, fully self-contained.
*Cons:* fundamentally *reactive*; it overfits to public-game priors and generalizes poorly (0.25% at launch). It explores the action space hoping to stumble into wins; it doesn't really *model* the world or *infer* goals.

**(b) State-graph + value model (Blind Squirrel / "just-explore").** Build a directed graph of observed states; prune actions that loop or change nothing; when the score improves, back-label the path with distances and train a small value model (e.g., a ResNet18) to rank (state, action) pairs toward the next milestone; repeat.
*Pros:* a training-free variant exists; exact on visited states; interpretable.
*Cons:* it only knows the frontier it has already explored — no *extrapolation* to goals it hasn't yet stumbled near.

**(c) Model-based RL with a learned world model + intrinsic exploration (the open direction).** Learn a transition model from observed $(s, a, s')$, plan with search over that model, and drive exploration with an intrinsic objective (prediction-disagreement / information gain) rather than random flailing.
*Pros:* it attacks the actual capability gap — modeling, goal inference, planning — instead of gaming the metric; planning over a learned model is sample-efficient; it's the research-valuable direction.
*Cons:* it's hard. Goal inference under sparse reward is unsolved; you have a compute budget; online test-time learning is finicky. Unproven — which is also why it's where the headroom is.

**(d) Frontier-LLM harness.** Wrap a strong model in scaffolding.
*Pros:* strong on *public* games with reasoning.
*Cons:* **ineligible on Kaggle** (no internet), expensive, and it overfits to specific public environments — performance is wildly bimodal across games and doesn't generalize.

For a beginner, the most useful lesson is not "copy one of these." It is to see the failure modes:

| Approach | First thing it teaches | First thing it fails at |
|---|---|---|
| Action-prediction RL | Learn which actions are nontrivial. | Understanding why an action matters. |
| State graph search | Avoid repeating useless states. | Guessing beyond states already visited. |
| World-model planning | Predict and plan before acting. | Building a reliable model quickly enough. |
| LLM harness | Use language reasoning and memory scaffolds. | Kaggle eligibility and generalization. |

My bias is that a strong solution will combine pieces: graph memory to avoid loops, an action prior to avoid wasting moves, a world model to predict transitions, and a planner that can exploit the model once the goal becomes plausible.

### Three implementation passes for a beginner

If I were mentoring someone starting this competition, I would not say "build a world model" on day one. I would give them three concrete passes.

#### Pass 1: make random less wasteful

Start with a legal random agent, then remove the most obvious waste:

- sample only from available actions;
- avoid actions already known to be no-ops in the exact same state;
- avoid immediate two-step loops (`A -> B -> A`);
- spread coordinate clicks over object-like cells instead of uniform random pixels;
- reset only when the state is stuck or the game is over.

The goal is not to be smart. The goal is to make random exploration produce useful data.

For click actions, "object-like cells" can be extremely crude at first:

```python
def candidate_clicks(grid):
    background = most_common_color(grid)
    cells = []
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value != background:
                cells.append((x, y))
    return cells
```

This will be wrong often. But it is usually less wrong than clicking uniformly over a 64×64 board, where most clicks may hit background.

#### Pass 2: build a state graph

Once no-op reduction exists, store every observed transition:

```text
state_hash --ACTION1--> next_state_hash
state_hash --ACTION5--> next_state_hash
state_hash --ACTION6(x=12,y=7)--> next_state_hash
```

Now you can ask questions like:

- Which actions leave this state unchanged?
- Which actions lead to unseen states?
- Which actions increase the score or change the game state?
- Which states are being visited repeatedly?
- Is the agent expanding the graph, or circling inside it?

The state graph is your first "world map." It is not a learned model yet, but it makes learning possible because it turns raw experience into structured data.

#### Pass 3: add priorities

After the graph exists, stop sampling actions uniformly. Score them:

| Signal | Meaning |
|---|---|
| `+ novelty` | Prefer actions predicted to reach unseen states. |
| `+ change_prior` | Prefer actions that have changed similar states before. |
| `+ object_focus` | Prefer actions involving non-background objects. |
| `+ score_delta` | Prefer transitions that improved score or level state. |
| `- no_op` | Penalize actions known to do nothing. |
| `- loop_risk` | Penalize actions that return to recent states. |

Even a hand-written priority function can be surprisingly useful:

```python
score = (
    2.0 * novelty
  + 1.0 * change_prior
  + 0.5 * object_focus
  + 3.0 * score_delta
  - 2.0 * no_op
  - 1.5 * loop_risk
)
```

That is not "AGI." But it is a coherent agent. It acts, observes, remembers, and changes its future actions based on what it learned. From there, a neural action prior or a learned transition model has somewhere sensible to attach.

### Failure modes to diagnose early

Most failed agents fail in boring ways. That is good news: boring failures are measurable.

| Symptom | Likely cause | First fix |
|---|---|---|
| Many actions, no frame changes | Sampling unavailable or irrelevant actions | Use available-action metadata and no-op cache. |
| Frame changes, but no score/level progress | Action prior finds motion but not goals | Add goal hypotheses and score-change tracking. |
| Same few frames repeat | Missing loop detector | Keep a recent-state window and penalize returns. |
| Good on one public game, awful elsewhere | Public-game overfit | Split public games; evaluate on holdout. |
| Click actions dominate but do nothing | Uniform coordinate sampling | Click non-background cells, changed cells, borders, or object centers. |
| Agent wins but score is tiny | Too many exploratory actions | Track actions per completed level and stop brute-force rollouts. |
| Runtime explodes | Search width too high | Cap candidate actions, planning depth, and per-turn model updates. |

This table is more useful than a giant architecture diagram. It tells you what to log and what to fix next.

## 8. The meta-game: scoring subtleties and open questions

This is the part you only learn by reading the forums, and it matters as much as the modeling:

- **Submission gotchas that silently zero you out.** Call `env.make()` exactly once, on the main thread — calling it per-process (e.g., when parallelizing) opens a scorecard per process and scores **0**. Don't set `MAX_ACTIONS = ∞`: recordings balloon and you hit Kaggle's storage quota and fail (a *storage* failure, not a logic one). Budget your rollout length to the notebook runtime limit rather than assuming long interactive search is free.
- **Is the human baseline visible to the agent?** There's an unresolved question (as of June 2026) about whether `environment_info.baseline_actions` — the second-best human's action count — is readable during submission for the test games. If it is, an agent could target human efficiency *exactly*. The host hasn't confirmed whether this is intentional or usable. Treat it as unsettled.
- **The open-source "copy" dilemma.** The moment a top team open-sources, thousands of copies get submitted, and with the inherent variance the original author can drop hundreds of places. So there's a perverse "small window" in which to open-source. Teams have asked for an AIMO-style fix: freeze the milestone ranking at a snapshot, then give a separate deadline to publish and claim — so you don't have to guess your rank before deciding to reveal your work. (As of this writing the exact milestone timing rules are still being clarified by the host.)
- **High variance.** Even the same agent code scores differently run-to-run. The leaderboard is noisy near the top.

## 9. How I'd approach it

My plan, and my advice:

1. **Build a local evaluation harness and stop iterating through submissions.** Kaggle allows ~1 submission/day, but *interactive* runs are free and the 25 public games run **offline**. Run your agent against the public games locally, and — critically — hold out a subset (say 18 train / 7 holdout) to estimate *generalization* before you ever submit. (Run through the official `main.py` path when you can; it's closer to the real runtime than driving the environment files yourself.)
2. **Benchmark against open methods, not the leaderboard number.** StochasticGoose is fully open (code + writeup); the graph-explore baseline is open. Reproduce them, measure them on your holdout, and beat *that*.
3. **Don't anchor on published-notebook scores** — stale, unit-confused, and partly exploit-inflated.
4. **Expect a shakeup and framework churn.** The metric and environment have already changed mid-competition; pin your environment and don't over-fit to the public games. The binding result is the final private re-run.
5. **Build incrementally**, each step locally measurable: a graph-exploration baseline with a learned action prior → a learned world model for frontier extrapolation → intrinsic-motivation exploration → goal inference and cross-level transfer. Any stopping point is submittable.
6. **Mine the milestone open-source releases** (June 30, Sept 30) for what the actual top teams do.

The starter kit makes the submission surface deliberately small. In practice, the interesting work lives behind one method:

```python
class MyAgent(Agent):
    def is_done(self, frames, latest_frame) -> bool:
        """Return True only when the agent wants to stop this playthrough."""
        return False

    def choose_action(self, frames, latest_frame) -> GameAction:
        """Observe the frame history and return one legal next action."""
        state = parse_grid(latest_frame)
        model = update_world_model(frames)
        goal = infer_goal(state, model)
        return plan_next_action(state, model, goal)
```

That stub is the whole research problem in miniature: perception, model update, goal inference, planning, and a fallback action when the model is wrong.

### A practical build ladder

If I were starting from zero, I would not begin with a neural network. I would build the ladder below, because each rung gives you a working artifact and a measurement.

| Rung | Build | Why it matters |
|---|---|---|
| 0 | Run the random starter locally. | Proves the environment, submission path, and logging work. |
| 1 | Save every frame/action/result to a replay log. | You cannot improve what you cannot inspect. |
| 2 | Detect "no-op" actions. | If an action does nothing in a state, do not keep repeating it. |
| 3 | Build a visited-state cache. | Avoid walking in circles. |
| 4 | Build a simple state graph. | Now the agent has a map of what it has already discovered. |
| 5 | Add action priors. | Prefer actions that changed the frame in related states. |
| 6 | Add local goal guesses. | Track candidate goals: reach a location, match colors, remove blockers, align objects. |
| 7 | Add short-horizon planning. | Search a few steps ahead using the state graph or a learned transition model. |
| 8 | Add learned transition prediction. | Try to predict next frames for actions not yet tested. |
| 9 | Add cross-level memory. | Later levels often reuse mechanics from earlier levels in the same game. |

The order matters. A learned world model without replay logging is a black box. A planner without a visited-state cache wastes time rediscovering loops. Cross-level memory without a level-change detector leaks stale assumptions into the next board. Boring infrastructure is not separate from intelligence here; it is how the agent gets enough traction to become intelligent.

The first useful local metrics are also simple:

| Metric | What it tells you |
|---|---|
| Number of unique states visited | Is exploration broadening, or looping? |
| No-op action rate | How much action budget is being wasted? |
| First useful action step | How long before the agent discovers a state-changing action? |
| Level completion count | The coarse success signal. |
| Actions per completed level | Whether wins are efficient enough to matter. |
| Cross-level reuse | Whether knowledge from level 1 helps level 2. |

Only after these are visible would I add heavier ML. Otherwise, you will not know whether the model improved reasoning or merely changed the failure mode.

### A concrete first-week plan

If the goal is to get from zero to a serious baseline, I would make the first week look like this:

| Day | Target | Output |
|---|---|---|
| 1 | Run the starter locally and submit the unchanged baseline once. | A working environment and a known submission path. |
| 2 | Add replay logging for frames, actions, state hashes, and outcomes. | `runs/YYYYMMDD/*.jsonl` or equivalent logs. |
| 3 | Implement no-op detection and repeated-state detection. | A random-but-less-wasteful agent. |
| 4 | Build a minimal state graph and visualize it for one game. | A graph dump showing states and transitions. |
| 5 | Add object-focused click candidates and available-action filtering. | Lower no-op rate on click-heavy games. |
| 6 | Split public games into development and holdout; run both. | A small comparison report. |
| 7 | Add a simple action-priority function and compare to baseline. | First real candidate plus failure notes. |

The goal of week one is not leaderboard score. It is instrumentation. By the end of the week you should be able to open a failed run and say:

```text
The agent got stuck because:
- ACTION5 changed nothing in 70% of states,
- coordinate clicks were mostly background,
- it revisited the same 11 states for 140 actions,
- and it never reached level 2.
```

That is progress. Once failures have names, they can be attacked.

### Things I would avoid early

I would avoid these traps:

- **Do not tune directly on leaderboard submissions.** The feedback is too sparse and too slow.
- **Do not hardcode public-game mechanics.** It will feel good locally and die on hidden games.
- **Do not start with an expensive neural model unless you can already produce clean logs.** Otherwise the model's failures are opaque.
- **Do not optimize only completion count.** A slow win can be nearly worthless under RHAE.
- **Do not let search expand without runtime caps.** A brilliant plan that times out is a zero.
- **Do not throw away failed trajectories.** Failed runs contain the data that tells you which actions are useless.

The discipline is to make every experiment leave behind an artifact: a replay, a metric table, a changed-state summary, or a failure label. If an experiment leaves only a leaderboard number, it probably taught you too little.

### What "world model + intrinsic exploration" means in practice

The phrase sounds grand, but the first version can be modest:

```python
memory = StateGraph()
model = TransitionModel()

while not done:
    state = observe()
    candidates = legal_actions(state)

    scored = []
    for action in candidates:
        predicted = model.predict(state, action)
        novelty = memory.novelty(predicted)
        uncertainty = model.uncertainty(state, action)
        goal_value = goal_heuristic(predicted)
        scored.append((novelty + uncertainty + goal_value, action))

    action = max(scored)[1]
    next_state = step(action)
    memory.add(state, action, next_state)
    model.update(state, action, next_state)
```

This is not a final solution. It is the shape of the solution I would trust: record what happened, learn what actions do, prefer uncertain or novel transitions while the goal is unknown, and shift toward goal-directed planning once the environment becomes legible.

## 10. What you actually get out of it

- **Prizes** — milestone pools, a top-5 pool, and the (currently unreachable) grand prize.
- **A separate Paper Prize track** rewards *methods*, not just scores — which fits a benchmark where the interesting contribution is an idea, not a leaderboard rank.
- **Kaggle medals/ranking**, if that's part of your portfolio.
- **Contribution to a genuinely open problem.** This is the rare benchmark where the field is near zero, so net-new general methods are both wanted and visible (the community explicitly disallows per-task hardcoding and rewards generalization).
- **Positioning** at the intersection of RL, world-models, and program-induction — and the plain intellectual value of working on something nobody has solved.

## 11. My take

I think the leaderboard is mostly a distraction, and the unit confusion makes it an *actively misleading* one. The real situation is clean and motivating: **the launch-era purpose-built preview winner fell back into the quarter-percent regime on the full benchmark, and the late-June Kaggle field is still under two percent.** That means the problem is open, and a genuinely different approach has room to matter in a way it rarely does on a mature benchmark.

My bet is that the winning ideas will come from **(c)** — a learned, test-time world model coupled with **intrinsic-motivation exploration** (drive toward states where your model is most uncertain, which is exactly where there's the most to learn) and **lightweight planning** over that model, with **explicit goal inference** layered on top. Reactive action-prediction (a) and pure frontier search (b) are the proven-but-capped baselines; they game the metric or exhaust the frontier without ever *understanding* the game. The four capabilities the benchmark names — exploration, modeling, goal-setting, planning — are a research agenda, and I'd rather build against that agenda than against the scoreboard.

I'll be writing up the build as I go.

---

### Resources

- [ARC Prize 2026 - ARC-AGI-3 on Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- [Official ARC-AGI-3 competition page](https://arcprize.org/competitions/2026/arc-agi-3)
- [ARC Prize 2026 overview and key dates](https://arcprize.org/competitions/2026)
- [ARC-AGI-3 technical report](https://arxiv.org/abs/2603.24621)
- [ARC-AGI-3 docs](https://docs.arcprize.org/arc-prize-2026), especially [Games](https://docs.arcprize.org/games), [Actions](https://docs.arcprize.org/actions), and [Scoring methodology](https://docs.arcprize.org/methodology)
- [ARC-AGI-3 Kaggle Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
- [ARC-AGI-3 Preview: 30-Day Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- [StochasticGoose preview solution](https://github.com/DriesSmit/ARC3-solution)
- [ARC-AGI Community Leaderboard](https://github.com/arcprize/ARC-AGI-Community-Leaderboard)
- Beginner-friendly explainers I found useful as secondary context: [DataCamp's ARC-AGI-3 overview](https://www.datacamp.com/blog/arc-agi-3), [Mark Barney's ARC-AGI-3 explainer](https://arc.markbarney.net/arc3), and [TokenCost's evaluation-cost writeup](https://tokencost.app/blog/arc-agi-3-benchmark-cost)
- Community tooling and game guides are moving quickly; I would look for the historical leaderboard tracker, simplified submission notebooks, offline recording viewers, and per-game mechanic wikis in the Kaggle discussion forum before relying on any one mirror.
