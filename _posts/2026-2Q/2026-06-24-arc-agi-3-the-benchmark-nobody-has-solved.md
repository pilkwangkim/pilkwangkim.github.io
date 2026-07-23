---
title: "ARC-AGI-3: The Benchmark Nobody Has Solved Yet"
date: 2026-06-24 21:00:00 +0900
categories: [AI, Kaggle]
tags: [arc-agi, arc-agi-3, benchmarks, reinforcement-learning, agents, world-models, agi]
math: true
pin: false
---

<link rel="stylesheet" href="{{ site.baseurl }}/assets/css/arc-agi-3.css?v=20260625-3">

# ARC-AGI-3: The Benchmark Nobody Has Solved Yet

> **Status note (updated 2026-07-23).** This article was originally written on June 24 and substantially revised after Milestone #1. I checked the update against the Kaggle CLI, the official ARC Prize result, the released submissions, and the latest papers. ARC-AGI-3 is still active, so every leaderboard number below is a dated snapshot rather than a permanent record.

Competition link:  
[ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)

Official benchmark page:  
[ARC Prize 2026 - ARC-AGI-3 Competition](https://arcprize.org/competitions/2026/arc-agi-3)

Technical report:  
[ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence](https://arxiv.org/abs/2603.24621)

<figure class="arc-agi-figure arc-agi-hero">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/arc-agi-3-banner.jpg" alt="ARC-AGI-3 title over a collage of small grid-game environments">
  <figcaption>ARC-AGI-3 moves the ARC benchmark family from static grid transformations into interactive, instruction-free game environments.</figcaption>
</figure>

> **One-paragraph version.** ARC-AGI-3 reframes "reasoning" as something you have to *do*, not something you *output*. An agent enters an unfamiliar 2D grid game with no instructions and is scored on how efficiently it learns to win relative to a human seeing the game for the first time. The post-Milestone #1 picture is more interesting than the launch picture: a local open-weight LLM can be a valid Kaggle agent, and Tufa Labs won with one. Yet the July 23 Kaggle leader was still only at **1.86%**, while far larger frontier systems reported much higher results on the visible public games. The benchmark is now exposing two problems at once: learning a world through action, and making that process generalize under an offline nine-hour sandbox.

## Why I'm writing this

I'm working through ARC-AGI-3 seriously, and writing is how I think. This post is my attempt to lay out — as precisely as I can — what the competition actually is, how the scoring works, what the current state of the art looks like once you strip away the leaderboard theater, and how I'd approach it. The framing is opinionated on purpose. Where I'm stating fact I'll try to be careful; where I'm stating a bet, I'll say so.

If you take one thing away: **never quote an ARC-AGI-3 score without its evaluation context.** A Kaggle hidden score, an ARC Prize public/semi-private evaluation, and a best-of public-game research run are not the same measurement. Older submissions also include stale scoring rules and a patched public-source exploit. The number is meaningful only when the game set, compute envelope, retry policy, and date travel with it.

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

- You do not need a giant language model on day one. Hosted APIs are unavailable during evaluation, but a local open-weight model packaged as a Kaggle input is legal — and Milestone #1 showed that this route is competitive.
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

Here is the same lineage as a compact comparison. The numbers are deliberately approximate because they refer to slightly different evaluation contexts, but the direction is what matters.

| Feature | ARC-AGI-1 (2019) | ARC-AGI-2 (2025) | ARC-AGI-3 (2026) |
|---|---|---|---|
| Format | Static grid puzzles | Static grid puzzles, harder and more compositional | Interactive game environments |
| Instructions | Input-output demo pairs | Input-output demo pairs | No natural-language instructions; discover rules through interaction |
| Best reported AI score | o3 reached 75.7% at the public compute limit and 87.5% with high compute on the semi-private set; public-eval reports went above 90% | 24.03% top Kaggle private score in ARC Prize 2025 | Not one comparable number: 12.58% in the three-game preview, 1.21% for the first full-competition milestone winner, and 13.33% public / 7.78% semi-private for the officially evaluated Sol Max harness |
| Human reference | Original private tasks were solved at 97-98% by individual testers, collectively 100%; many summaries use ~85% as the older rough human benchmark | Public eval human sample average was 66%; selected evaluation tasks were human-solvable | Humans can solve 100% of included environments; AI is scored by action efficiency relative to human baselines |
| Scoring | Task accuracy, typically binary solved/not solved | Accuracy plus cost-per-task reporting | Relative Human Action Efficiency: completion and action efficiency vs. humans |
| Dataset scale | 400 public train, 400 public eval, 100 semi-private, 100 private | 1,000 public train, 120 public eval, plus 120 semi-private and 120 private | 25 public demo environments plus 55 semi-private and 55 fully private environments, each with multiple levels |

The ARC-AGI-3 score row needs more care than the other two. **These figures come from different game sets, compute budgets, and submission rules.** The 12.58% result used three hidden preview games. The 1.21% milestone result came from Kaggle's offline hidden evaluation. The 13.33% / 7.78% Sol Max result used ARC Prize's public and semi-private benchmark protocol. They answer related questions, but they are not interchangeable leaderboard entries.

The conceptual shift is from "intelligence as pattern-matching on a fixed dataset" to "intelligence as adaptive behavior in an open-ended environment." That shift is exactly why current systems fall off a cliff.

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

<figure class="arc-agi-figure arc-agi-frame">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/arc-agi-3-game-frame.png" alt="Pixel-art ARC-AGI-3 style game frame with colored objects, a player block, and progress bars">
  <figcaption>A concrete game frame makes the abstraction less mysterious: the agent sees colored cells, but it is not told which object is the player, which tile is useful, or what the goal is.</figcaption>
</figure>

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

### What the programmer actually receives

This is where ARC-AGI-3 starts to feel very different from a normal ML problem. You do not call:

```python
answer = model(question)
```

You repeatedly call the environment, receive a bundle of state, interpret it, choose one legal action, and pay for that action immediately. In API form, the loop looks roughly like this:

```python
# Start a game or reset a running game.
bundle = post("/api/cmd/RESET", {
    "game_id": game_id,
    "card_id": card_id,
})

guid = bundle["guid"]

while bundle["state"] == "NOT_FINISHED":
    action = choose_action(bundle, memory)

    payload = {
        "game_id": game_id,
        "card_id": card_id,
        "guid": guid,
    }
    if action["id"] == 6:
        payload["x"] = action["x"]
        payload["y"] = action["y"]

    next_bundle = post(f"/api/cmd/{action['name']}", payload)
    update_memory(memory, bundle, action, next_bundle)
    bundle = next_bundle
```

A returned bundle is not a clean "observation tensor plus reward" in the textbook RL sense. It is closer to this:

```python
{
    "game_id": "ls20-016295f7601e",
    "guid": "2fa5332c-2e55-4825-b5c5-df960d504470",
    "frame": [[0, 0, 0, ...], [0, 3, 3, ...], ...],
    "state": "NOT_FINISHED",
    "levels_completed": 0,
    "win_levels": 254,
    "action_input": {"id": 6, "data": {"x": 12, "y": 34}},
    "available_actions": [1, 2, 3, 4, 6],
    "score": 0.0,  # present in many API/toolkit responses
}
```

The agent has to extract its own usable state from that bundle:

| Field | What it gives you | What it does not give you |
|---|---|---|
| `frame` | The visible grid. | Object identities, goals, physics, or which cell is worth clicking. |
| `available_actions` | Which action ids are legal now. | For `ACTION6`, it does **not** tell you useful `(x, y)` coordinates. |
| `state` | Whether the game is still active, won, or over. | How close you are to winning. |
| `levels_completed` | A coarse progress counter. | Which hidden mechanic was learned, or why progress happened. |
| `win_levels` | The total number of levels in that environment. | A plan for reaching the next level. |
| `action_input` | The action that produced the returned frame. | Whether that action was strategically good. |
| local action counter | How many actions you have spent. | This is not a magic "remaining turns" hint; you usually maintain it yourself or read it from your local framework wrapper. |

That last line is the kind of thing that makes the first implementation feel unexpectedly awkward. The environment may tell you the state and progress, but it does not hand you a friendly `remaining_turns_to_good_score` value. If your agent cares about action budget — and it must — it needs to track how many calls it has made, when a level changed, when it reset, and how many actions were spent per completed level.

Now imagine the first real debugging session. The agent calls `ACTION6` at `(12, 34)`. The next bundle comes back with the same `frame`, the same `levels_completed`, the same `state`, and the same `available_actions`. What did you learn?

```python
def update_memory(memory, prev, action, nxt):
    prev_grid = prev["frame"]
    next_grid = nxt["frame"]

    changed = prev_grid != next_grid
    progress = nxt.get("levels_completed", 0) - prev.get("levels_completed", 0)
    state_changed = nxt["state"] != prev["state"]

    key = (hash_grid(prev_grid), canonical_action(action))
    memory["outcome"][key] = {
        "changed": changed,
        "progress": progress,
        "state_changed": state_changed,
        "next_hash": hash_grid(next_grid),
    }

    if not changed and progress == 0 and not state_changed:
        memory["no_ops"].add(key)
```

This tiny function is already "learning." It does not learn a grand theory of the game, but it learns that a specific action in a specific state was probably useless. The next call should use that fact:

```python
def choose_action(bundle, memory):
    grid = bundle["frame"]
    legal = bundle["available_actions"]
    state_hash = hash_grid(grid)

    candidates = []
    for action_id in legal:
        if action_id == 6:
            for x, y in candidate_clicks(grid):
                candidates.append({"id": 6, "name": "ACTION6", "x": x, "y": y})
        else:
            candidates.append({"id": action_id, "name": f"ACTION{action_id}"})

    candidates = [
        a for a in candidates
        if (state_hash, canonical_action(a)) not in memory["no_ops"]
    ]

    return rank_candidates(candidates, bundle, memory)[0]
```

The "training data" for a small model can come from exactly these transitions. For example, StochasticGoose-style action learning does not begin by predicting the final goal. It begins with a smaller target:

```python
training_example = {
    "grid_before": prev["frame"],
    "action": encode_action(action),
    "label_changed": int(prev["frame"] != nxt["frame"]),
    "label_progress": int(nxt["levels_completed"] > prev["levels_completed"]),
}
```

A CNN can learn `P(frame changes | grid, action)` or `P(progress | grid, action)`. That is useful because it lets the agent spend fewer actions on obvious no-ops. But notice the narrowness: this does not yet solve the game. It only makes exploration less wasteful.

### Why "just use torch" or "just ask an LLM" fails

The tempting first idea is:

```python
policy = CNN()
action = policy(torch.tensor(frame))
```

or:

```python
prompt = f"What should I do in this grid?\n{frame}"
action = llm(prompt)
```

Both are understandable. Both run into concrete walls.

| Naive idea | What goes wrong |
|---|---|
| Train a CNN policy on the 25 public games. | The hidden games are different. A policy that memorizes public mechanics learns "what worked here", not "how to discover what works next." |
| Use RL from scratch. | Rewards are sparse and expensive. Each failed exploratory action lowers RHAE, and Kaggle runtime is finite. A blank-slate RL agent cannot afford millions of environment steps per hidden game. |
| Predict the next frame with torch. | Next-frame prediction is useful only after you have diverse transitions. At the beginning, the agent does not even know which actions produce transitions. |
| Call GPT, Claude, or another hosted API. | Kaggle disables internet access during evaluation, so the request never reaches the provider. |
| Run a local LLM and ask it for the next move. | This is legal and now proven viable, but a bare prompt is not enough. The model still needs compact observations, persistent memory, tools, legal-action checks, and a way to test its hypotheses against real transitions. |
| Use image classification. | There is no fixed class label like "cat" or "dog." The target is an action sequence under unknown rules. |
| Brute force action sequences. | The scorer sees every action. Finding a win after 500 random moves may score nearly zero compared with a human who needed 20. |

So the practical route is not "no neural nets," and it is no longer defensible to say "LLM harnesses cannot enter Kaggle." The practical route is to give any model a job that matches the data you actually have:

| Model job | Input | Target | Why it is plausible |
|---|---|---|---|
| Action-change predictor | `(frame, action)` | Did the frame change? | Labels are available after every step. |
| Progress predictor | `(frame, action)` | Did `levels_completed` or score improve? | Sparse, but directly measurable. |
| State embedding | frame | Similar states should have nearby embeddings. | Helps de-duplicate and detect loops. |
| Transition model | `(frame, action)` | Approximate next frame or changed cells. | Useful for short-horizon planning once enough transitions exist. |
| Goal hypothesis scorer | frame history | Which object/state looks like progress? | Can be trained or heuristic, but must be validated by action outcomes. |

The programmer's first wall is therefore not "which transformer should I use?" It is more basic:

```text
What exactly did I observe?
What action did I take?
Did anything change?
Did progress change?
Have I tried this transition before?
What did this teach me about the next call?
```

That is the little loop ARC-AGI-3 forces you to build. It feels mundane, but without it the expensive model has nothing reliable to learn from.

**The public/private split** is the part you have to internalize:

- **25 public games** — you get the source; you can train and debug on them.
- **110 private games** — never seen by you or your agent. Of these, **55 form the Public Leaderboard** and the other **55 the Private Leaderboard** (the one that decides final standings, in classic Kaggle public/private style).

So even the "public" leaderboard you watch is computed on **unseen** games. This is why solving a pile of public levels can still yield a competition score of **0.00**.

| Split | Count | Visible to competitors? | Role |
|---|---:|---|---|
| Public demo games | 25 | Yes | Source-visible environments for local development, debugging, and first experiments. |
| Semi-private hidden games | 55 | No | Used for public leaderboard and scorecard reporting during the competition. |
| Fully private hidden games | 55 | No | Final holdout used for the private leaderboard at the end. |

That split is the reason a public-game agent and a leaderboard-scoring agent can be very different systems. The public games are the practice field; the leaderboard asks whether the same ideas transfer to environments the agent has never seen.

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

## 4. Scoring: RHAE

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

In prose, the pipeline is:

| Stage | What is computed | Why it matters |
|---|---|---|
| Level score | Compare human action count `h` with agent action count `a`, using `min(h / a, 1.15)^2`; if `a > 5h`, the level gets 0. | Completion alone is not enough. The benchmark rewards human-like efficiency. |
| Game score | Average level scores with larger weights on later levels. | An agent that solves only the easy prefix should not look too strong. |
| Completion cap | Cap the game score by the sequential levels actually completed. | This prevents shortcuts where an agent misses the beginning but gets lucky later. |
| Benchmark score | Average hidden-game scores and report them on a 0-100% scale. | Broad generalization matters more than overfitting one environment. |

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

- **No internet during evaluation.** This rules out hosted API calls, not language models themselves. Publicly available pretrained weights may be attached as Kaggle inputs and served locally inside the notebook.
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

That last row matters. A notebook that calls GPT, Claude, Gemini, or any external server is not a valid final submission if it depends on internet access. But the official GPT-OSS-120B template demonstrates the legal alternative: load model weights from a Kaggle input, start a local vLLM server on `127.0.0.1`, and point an OpenAI-compatible client at that local process. "No internet" means **self-contained inference**, not "no LLM."

## 6. The current landscape (with the theater removed)

### Milestone #1 changed the reference point

The June 30 checkpoint produced the first prize-verified full-competition methods:

| Place | Team / system | Core approach |
|---|---|---|
| 1 | Tufa Labs, **The Duck** | Local Qwen 3.6 27B FP8, live Python REPL, multimodal observations, rolling context eviction |
| 2 | **Reki** | Local Gemma-4-31B, structured JSON planning and reflection, legal-action repair, NumPy click fallback |
| 3 | **forge** | Local Gemma-4-31B with structured plans, reflection, and a configurable generation/selection framework |

Duck's official milestone score was **1.21%**. It treats the model less like a policy network and more like a coding agent inside a live laboratory. Each observation becomes a Python variable; the model reasons, calls helper functions, writes code, executes an action, and inspects the result. It can switch among a rendered image, raw ASCII, and segmented regions. When context fills, the oldest interaction messages are evicted so play can continue.

One attribution is worth getting exactly right. The pure-NumPy "salient click" rule and the `dead-signature` that suppresses objects whose clicks repeatedly do nothing belong to **Reki**, not Duck. Reki and forge started from the official GPT-OSS template and replaced the base model with Gemma, whereas Duck is the only podium system in which the agent itself writes exploratory code.

The team history is suggestive but not a proof. Duck calls itself a successor to StochasticGoose, and Dries Smit joined the Duck team. That is strong practical evidence that a local LLM harness was more promising under the full Kaggle constraints than relying on a per-level CNN plus sparse RL alone. It does **not** prove that learned action priors or RL are useless; those components may still be valuable inside a stronger harness.

### Four scoreboards that should not be mixed

| Evaluation context | Result | What it establishes |
|---|---:|---|
| 30-day preview, three hidden games | StochasticGoose 12.58% | CNN + sparse-RL found real signal on a small preview set, but did not generalize to the full launch |
| Kaggle Milestone #1 hidden evaluation | Duck 1.21% | A self-contained local open-weight coding harness is legal and competitive |
| ARC Prize official model evaluation | Sol Max 13.33% public / 7.78% semi-private | A stronger frontier harness transfers partly beyond the visible games, outside the Kaggle submission envelope |
| Public-game research reports | Rodionov 58.12%; Schema 98.98% / 95.35% | Explicit working models and verification can solve many visible games, but the highest figures remain public-only and partly self-reported |

I also queried the competition through Kaggle CLI on **July 23, 2026**. The top three leaderboard scores were **1.86, 1.61, and 1.60**; Tufa Labs was at **1.45**. Ten teams were at or above 1.50, and 122 were at or above Duck's original 1.21. Duck therefore became the first widely inspectable baseline around the 1.2% band, not a permanent description of the frontier. The current 1.86 leader's method is not public, so it would be premature to say that every leading submission is a Duck derivative.

The copy dilemma did arrive. A CLI search now finds dozens of public Duck forks and variants. But similar scores do not prove common ancestry: the 0.86 milestone entry was forge, for example, and a roughly 0.79 Gemma dynamics notebook was a separate line. Method lineage has to be checked from code, not inferred from a score.

The public-game jump also needs restraint. Andrey Rodionov's executable-world-model agent reports 15 of 25 games fully solved and 58.12% mean RHAE with GPT-5.5 high. Schema reports 98.98% with an Opus/Fable fallback and 95.35% with GPT-5.6 Sol. Schema explicitly labels both results unverified, reruns weak games under a stronger fallback, and retains the better per-game score. Public visibility, repeated tuning, best-of selection, larger hosted models, and the absence of a hidden holdout all contribute to the gap between those numbers and Kaggle's 1.x% range.

The durable lesson is no longer "CNN beats LLM" or "LLMs cannot enter Kaggle." It is this: **the agent needs a process for turning interaction history into a working model, testing that model, and planning with what survives the test.** Which representation and base model do that most efficiently is still open.

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

### Why the leaderboard still needs context

If the Kaggle leaderboard is still below 2%, why do some public notebooks and research reports look much stronger? Three reasons matter most:

1. **Stale scores under an older rule** — Kaggle does not retroactively re-score historical submissions when the metric changes. The per-level score was changed to be *squared* (and the cap raised) partway through; old high entries kept their old-rule numbers, and the public-notebook ranking has been effectively frozen for months because new entries can't displace them under the harsher rule. The leaderboard is therefore an apples-and-oranges mix.
2. **A patched exploit** — for a while, some agents located the actual game *source code* on disk and ran exhaustive search against the real simulator (a white-box trick). That works on public games (whose source ships) but is dead on the private leaderboard games (API-only), and the dynamic-game-code-loading path has since been fixed. High historical scores that leaned on it don't reproduce.
3. **Different protocols** — hosted frontier-model reports can use much more compute, and some public studies rerun failures or select the best result per game. Kaggle runs one self-contained notebook against hidden games under a nine-hour limit.

So: anchoring on a published notebook's score is a mistake. The useful questions are whether the principle survives on unseen games, how many real actions it spends, and whether it fits the Kaggle runtime envelope.

## 7. Approaches, and what's actually wrong with each

Before the approaches, translate the jargon:

| Term | Meaning here |
|---|---|
| **CNN** | A neural network that is good at local spatial patterns in grids or images. |
| **Sparse reward** | The agent gets useful feedback only rarely, usually when it wins or reaches a milestone. |
| **State graph** | A map of frames the agent has seen, with actions as edges between frames. |
| **Value model** | A model that estimates which state or action is closer to success. |
| **World model** | Any explicit working account of the environment: a learned simulator, a state graph, or Python code that predicts "if I do action A in state S, what happens next?" |
| **Intrinsic exploration** | Exploration driven by curiosity or uncertainty, not only by external score. |
| **Planning** | Searching over possible future action sequences before committing to one. |

The important distinction is between **reactive** systems and **model-based** systems. A reactive system mostly asks, "what action looks promising now?" A model-based system asks, "what do I think will happen if I take this action, then that action, then that one?" Humans lean heavily on the second style when learning a new game.

**(a) CNN + sparse-RL action prediction (the StochasticGoose lineage).** Learn a small CNN that predicts which actions change the frame, and bias exploration toward those. Hierarchical sampling can pick an action type and then a click coordinate with a convolutional 64×64 head.
*Pros:* sample-efficient, fully self-contained, and still useful as an action prior or no-op filter.
*Cons:* weak as a complete agent. Predicting that an action changes the screen does not explain why the change matters, and the preview result did not transfer to the full benchmark.

**(b) State-graph + value model (Blind Squirrel / "just-explore").** Build a directed graph of observed states; prune actions that loop or change nothing; when the score improves, back-label the path with distances and train a small value model (e.g., a ResNet18) to rank (state, action) pairs toward the next milestone; repeat.
*Pros:* a training-free variant exists; exact on visited states; interpretable.
*Cons:* it only knows the frontier it has already explored — no *extrapolation* to goals it hasn't yet stumbled near.

**(c) Learned neural world model + intrinsic exploration.** Learn a transition model from observed $(s, a, s')$, plan over that model, and prefer actions with high disagreement or information gain.
*Pros:* it can predict beyond the exact states already visited and gives a compact action prior.
*Cons:* online learning is data-hungry and unstable under sparse reward. The full competition has not yet shown that this can build a reliable model quickly enough. It now looks more like a useful component than the obvious top-level controller.

**(d) Local LLM tool/coding harness (Duck, Reki, forge).** Run an open-weight model inside the offline notebook and surround it with observation tools, memory, legal-action validation, reflection, and a controlled action interface. Duck additionally gives the model a Python REPL so it can inspect and transform state programmatically.
*Pros:* Kaggle-legal, Milestone #1-validated, and able to change representations and reasoning procedures at test time instead of relying on one fixed policy.
*Cons:* a 27B- or 31B-class model consumes most of the runtime and memory envelope. Context management, malformed outputs, slow inference, and weak visual grounding remain concrete failure modes. Duck's 1.21% also shows that a fluent coding loop is not yet sufficient for hidden generalization.

**(e) Verified executable world model (Rodionov / Schema direction).** Let a coding agent maintain executable Python hypotheses about game dynamics, test predictions against observed transitions, simplify models that have accumulated exceptions, and plan in the surviving model before spending real actions.
*Pros:* this turns "reasoning" into falsifiable predictions. Rodionov reported 58.12% mean public RHAE, and Schema reported public scores in the mid-to-high 90s.
*Cons:* the strongest results use visible games and much larger hosted models. A later ablation found that model strength and reasoning effort were the largest effects; textual working models sometimes beat flexible executable ones. Verification was consistently useful, but Python code by itself is not a magic ingredient.

For a beginner, the most useful lesson is not "copy one of these." It is to see the failure modes:

| Approach | First thing it teaches | First thing it fails at |
|---|---|---|
| Action-prediction RL | Learn which actions are nontrivial. | Understanding why an action matters. |
| State graph search | Avoid repeating useless states. | Guessing beyond states already visited. |
| Learned world-model planning | Predict beyond visited states and plan before acting. | Building a reliable model quickly enough. |
| Local LLM harness | Change representation, write tools, and maintain semantic hypotheses. | Runtime, context, grounding, and hidden generalization. |
| Verified executable model | Make hypotheses testable and revise them after contradictions. | Choosing useful experiments and avoiding brittle over-modeling. |

My updated bias is a hybrid: a local LLM harness as the adaptive controller; exact transition memory for facts already observed; executable hypotheses for rules not yet known; disagreement-driven probes when hypotheses conflict; and shortest-path replay when a known route should be exploited efficiently. The model should propose and revise abstractions, while deterministic code protects it from forgetting what the environment has already proved.

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
- **The open-source "copy" dilemma is now observable, not hypothetical.** Duck was released, and public forks and light modifications quickly spread across the leaderboard. The host explicitly recommends copying a template or released submission as the fastest way to enter Milestone #2. That is good for baseline quality and awkward for attribution: a score alone no longer tells you whether a team discovered a method or changed a prompt in a public harness.
- **High variance.** Even the same agent code scores differently run-to-run. The leaderboard is noisy near the top.

## 9. How I'd approach it

My plan, and my advice:

1. **Build a local evaluation harness and stop iterating through submissions.** Run the 25 public games offline, save complete traces, and hold out a subset to catch the most obvious public-game overfitting. A public holdout is still not a hidden set, but it is better than tuning all 25 simultaneously.
2. **Reproduce the current open baselines.** Start with the official GPT-OSS notebook and Duck, then compare Reki-style structured control and the older graph-search/action-prior baselines under the same local protocol.
3. **Measure the Kaggle envelope, not only RHAE.** Record model load time, tokens per second, tool-call latency, peak memory, actions per game, and projected total runtime. A public solver that cannot process the hidden suite in nine hours is not a competition baseline.
4. **Separate facts from hypotheses.** Put exact observed transitions in a deterministic cache. Keep uncertain rules, goals, and object roles in a hypothesis ledger with supporting and contradicting evidence.
5. **Add planning in two layers.** Use shortest-path search over known transitions for cheap replay, and use model-generated programs or learned predictions only for states and actions the agent has not observed.
6. **Explore where hypotheses disagree.** A probe is valuable when plausible models predict different outcomes, not merely because the resulting frame looks novel.
7. **Treat Milestone #2 as a held-out engineering deadline.** Freeze a local suite before September 30, compare each component by ablation, and submit only after the local model, cache, and recovery path survive the full runtime budget.

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

### After week one: from Duck to a verified hybrid

Once the basic logger works, the next architecture I would test is not "let the LLM do everything." It is a division of labor:

```text
observe frame
→ normalize state and retrieve exact history
→ give the LLM compact evidence and available tools
→ let it propose hypotheses, code, and a legal action
→ execute one real action
→ record the exact transition
→ check every prediction against the result
→ preserve, revise, or reject the hypotheses
```

The central action boundary can stay small:

```python
def execute_and_learn(state, action, expected=None):
    next_state = env.step(action)
    edge = transition_cache.record(state, action, next_state)

    if expected is not None:
        hypotheses.update_from_prediction(expected, next_state)

    if edge.conflicts_with_previous_outcome:
        transition_cache.mark_latent_or_stochastic(state, action)

    return next_state
```

A cache key should include at least the game, level, normalized grid, and any stable metadata the API exposes. Hashing only the visible grid is dangerous: two identical images may hide a timer, animation phase, inventory flag, or other latent state. When the same key and action produce different outcomes, the agent should record a conflict instead of silently overwriting the old edge.

Known deterministic edges form a graph. Dijkstra or BFS can then find the cheapest route back to a discovered target after reset, failure, or a later level that reuses the same mechanics. This directly attacks repeated action waste. It does **not** refund the actions spent discovering the route, and it cannot plan through unobserved states. The cache is a memory and replay system, not a complete world model.

The LLM should therefore work where exact memory ends. It names objects, writes candidate transition functions, proposes goals, and chooses probes that distinguish competing explanations. If hypothesis $h_1$ predicts "the blue object moves" and $h_2$ predicts "the selected row rotates," an action on which they disagree is more informative than another generic novelty click.

The first ablation table should be equally concrete:

| Variant | Question |
|---|---|
| Duck baseline | What does the released harness achieve unchanged? |
| + exact transition cache | Does repeated-state and repeated-action waste fall? |
| + shortest-path replay | Are solved or partially solved trajectories reproduced with fewer actions? |
| + prediction verification | Do contradicted rules disappear instead of poisoning later plans? |
| + disagreement probes | Does the agent identify useful mechanics with fewer exploratory actions? |

Track `edge_reuse_rate`, `transition_conflict_rate`, `replay_actions_saved`, `invalid_action_rate`, `tokens_per_action`, wall-clock time, completed levels, and RHAE. Without these measurements, a more elaborate harness can look more intelligent while merely spending more compute.

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
- **Contribution to a genuinely open problem.** Public games are increasingly solvable, but hidden-game generalization under the Kaggle envelope remains near the floor. New general methods are still both wanted and visible.
- **Positioning** at the intersection of RL, world-models, and program-induction — and the plain intellectual value of working on something nobody has solved.

## 11. My take

Milestone #1 forced me to correct one of my original conclusions. I was right that a submission cannot depend on a hosted API. I was wrong to turn that into "an LLM harness is ineligible." A local open-weight model is legal, the official template demonstrates how to run one, and Duck won the milestone with exactly that class of system.

The newer evidence also changes my technical bet. I no longer think the answer is simply **a learned neural world model**. The stronger common pattern is a process: build a working account of the world, make predictions from it, test those predictions against interaction history, revise contradictions, and plan with what remains. An executable Python model is one useful representation, but the ablations warn against treating it as the cause of success by itself. Base-model capability and reasoning effort still matter enormously.

My current bet is therefore **local LLM harness + exact transition memory + verified executable hypotheses + disagreement-driven exploration + shortest-path replay**. A learned action prior or neural transition model can sit inside that system, but it should not be trusted with facts the environment has already demonstrated exactly.

The Kaggle leaderboard is still below 2% as of July 23, while visible public games are being solved at dramatically higher rates with larger systems. That gap is not an embarrassment to explain away. It is the research problem: how do we compress adaptive scientific reasoning into a self-contained agent that can face genuinely unseen worlds under a fixed action and compute budget?

I'll be writing up the build as I go.

---

### Resources

- [ARC Prize 2026 - ARC-AGI-3 on Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- [ARC-AGI-1 benchmark page](https://arcprize.org/arc-agi/1)
- [ARC-AGI-2 dataset repository](https://github.com/arcprize/ARC-AGI-2)
- [OpenAI o3 ARC-AGI-1 report](https://arcprize.org/blog/oai-o3-pub-breakthrough)
- [ARC Prize 2025 technical report](https://arxiv.org/abs/2601.10904)
- [Official ARC-AGI-3 competition page](https://arcprize.org/competitions/2026/arc-agi-3)
- [ARC Prize 2026 overview and key dates](https://arcprize.org/competitions/2026)
- [ARC-AGI-3 technical report](https://arxiv.org/abs/2603.24621)
- [ARC-AGI-3 docs](https://docs.arcprize.org/arc-prize-2026), especially [Games](https://docs.arcprize.org/games), [Actions](https://docs.arcprize.org/actions), [Scoring methodology](https://docs.arcprize.org/methodology), [Local vs Online](https://docs.arcprize.org/local-vs-online), and [Scorecards](https://docs.arcprize.org/scorecards)
- [ARC-AGI-3 Kaggle Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
- [ARC Prize 2026 Milestone #1 results](https://arcprize.org/blog/arc-prize-2026-milestone-1)
- [Tufa Labs: The Duck harness](https://tufalabs.ai/research/duck-harness/) and [released code](https://github.com/Tufalabs/duck-harness)
- [Official Kaggle GPT-OSS-120B template](https://www.kaggle.com/code/gregkamradt/arc-agi-3-gpt-oss-120b)
- [Executable World Models for ARC-AGI-3](https://arxiv.org/abs/2605.05138) and the [controlled ablation follow-up](https://arxiv.org/abs/2607.15439)
- [Schema harness report](https://schema-harness.github.io/) and [released trace dataset](https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces)
- [Official GPT-5.6 ARC-AGI-3 results](https://arcprize.org/results/openai-gpt-5-6)
- [ARC-AGI-3 Preview: 30-Day Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- [StochasticGoose preview solution](https://github.com/DriesSmit/ARC3-solution)
- [ARC-AGI Community Leaderboard](https://github.com/arcprize/ARC-AGI-Community-Leaderboard)
- Beginner-friendly explainers I found useful as secondary context: [DataCamp's ARC-AGI-3 overview](https://www.datacamp.com/blog/arc-agi-3), [Mark Barney's ARC-AGI-3 explainer](https://arc.markbarney.net/arc3), and [TokenCost's evaluation-cost writeup](https://tokencost.app/blog/arc-agi-3-benchmark-cost)
- Community tooling and game guides are moving quickly; I would look for the historical leaderboard tracker, simplified submission notebooks, offline recording viewers, and per-game mechanic wikis in the Kaggle discussion forum before relying on any one mirror.
