---
title: "Pokémon TCG AI Battle Working Note (Part 1): From Rules to an RL Pilot"
date: 2026-07-02 21:30:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, pokemon-tcg, game-ai, reinforcement-learning, behavior-cloning, offline-rl, self-play, benchmark, working-note]
math: true
pin: false
---

# Pokémon TCG AI Battle Working Note (Part 1): From Rules to an RL Pilot

> **First written on July 2, 2026, and rebuilt around the implementation and failures observed
> through July 15.** This is not a success story presenting a finished RL solution. It is a
> practical guide and error log: where I started without much RL experience, what I misunderstood,
> what the experiments actually established, and how I would build the next version.

Competition:
[The Pokémon Company - PTCG AI Battle Challenge Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)


## Part I. Reframing the problem before choosing an RL algorithm

Before choosing an RL method, I needed to define what the agent was actually learning. This part
turns the game into legal-option ranking, separates deck quality from pilot quality, and introduces
only the vocabulary needed by the rest of the note.

### 1. The short answer: what should RL do here?

In Pokémon TCG AI Battle, the simulator presents a list of actions that can currently be chosen.
The agent does not need to generate every possible game action from scratch. Its central job is to
**rank the options in the current menu**.

$$
a_t \in A_t = \text{Options}(o_t)
$$

Here (o_t) is the information visible to the agent and (A_t) is the variable-length option set.
A model can assign one score to every candidate (c_i\in A_t):

$$
s_i=f_\theta(o_t,c_i), \qquad
\pi_\theta(i\mid o_t,A_t)=
\frac{\exp(s_i)}{\sum_{j\in A_t}\exp(s_j)}.
$$

This framing gives a much more realistic learning path than “start PPO and hope”:

```text
instrument rule pilots and replay decisions
-> imitate observed choices with behavior cloning
-> repair errors on critical decisions and difficult matchups
-> combine learned scores with a safe rule fallback
-> use outcomes and value estimates to weight better actions
-> attempt offline RL and population self-play only after evaluation is trustworthy
```

I initially reasoned in the opposite direction: it is a game, therefore use RL; RL, therefore use
self-play. In practice, the first requirements were a reliable data contract, a representation for
dynamic action menus, faithful opponents, temporal holdouts, and a package that could survive an
actual validation episode. A more sophisticated optimizer only magnifies errors when these pieces are
missing.


### 2. Separate the deck, pilot, field, and evaluator

My first major mistake was treating an agent score as a direct measurement of deck quality.

| Component | Question it must answer |
|---|---|
| **Deck** | What win condition and potential does this exact 60-card list have? |
| **Pilot** | How much of that potential does the decision policy realize? |
| **Field** | Which decks and play styles are currently present on the ladder? |
| **Evaluator** | Are the local opponents and sample sizes good enough to judge the policy? |

Observed performance is closer to:

$$
\text{Observed Performance}
=f(\text{deck},\text{pilot},\text{opponents},\text{seat},\text{variance}).
$$

Two agents can submit the same deck hash and use very different policies. A fixed policy can also
behave very differently after changing the list. The useful evidence unit is therefore closer to:

```text
(date, team, exact_deck_hash, policy_or_epoch, opponent, seat)
```

Meta analysis still matters, but it is not the main subject of this revision. It serves three RL
functions:

1. deciding which deck behaviors need more data;
2. identifying missing benchmark pilots;
3. weighting evaluation matchups according to the current field.

Daily share and matchup charts belong in the Meta Snapshot notebooks. This note focuses on turning
that evidence into an RL dataset and an honest evaluation system.


### 3. Minimal RL vocabulary for this project

#### 3.1 The agent does not observe the full state

The opponent's hand, prizes, and full remaining deck are hidden. This is more naturally modeled as a
partially observable Markov decision process (POMDP) than a fully observed MDP.

| Term | Meaning in this project |
|---|---|
| Observation (o_t) | Board, revealed cards, counters, logs, and in-turn flags visible to the agent |
| Action set (A_t) | The variable-length option list presented by the engine |
| Policy (pi(a\mid o)) | A rule or model that chooses an action from the observation |
| Trajectory | A game sequence ((o_0,a_0,o_1,a_1,\ldots)) |
| Reward (r_t) | A learning signal; the true objective is the final game result |
| State value (V(o)) | Expected future return from the current information state |
| Action value (Q(o,a)) | Expected future return after choosing a particular action |
| Advantage (A(o,a)) | How much better an action appears than the state's baseline value |

#### 3.2 Behavior cloning is not RL, but it is the right starting point

Behavior cloning (BC) treats replay actions as supervised labels:

$$
\mathcal L_{BC}
=-\log \pi_\theta(a_t^*\mid o_t,A_t).
$$

BC does not directly optimize environment reward, so it is not reinforcement learning by itself.
It is still an unusually useful first step here:

- the policy does not have to rediscover basic card sequencing from random play;
- hundreds of thousands of real decisions are available as labels;
- it quickly tests whether action alignment and feature extraction are correct;
- it provides an initialization for value learning, offline RL, and self-play.

For this project, “starting RL” meant first building a learnable policy representation and an
evaluation harness that could be trusted. It did not mean immediately starting PPO.


---

## Part II. Turning replays into behavior-cloning data

Once the problem is defined, the next concern is data. Correct action alignment and shared
train/serve features matter before model sophistication. This part ends with a behavior-cloning pilot
that can actually be packaged and executed.

### 4. The most important data contract: align observation and action correctly

In the CABT replay schema I inspected, the action responding to the menu shown at step (i) is stored
in the next step's action field:

```text
features = steps[i][player].observation
label    = steps[i + 1][player].action
```

Getting this wrong teaches the model unrelated labels. The resulting accuracy can still look
plausible, which makes the bug especially dangerous. The parser must enforce index validity:

```python
select = steps[i][player]["observation"].get("select")
action = steps[i + 1][player].get("action")

if not select or not select.get("option") or not action:
    return None

n_options = len(select["option"])
assert all(0 <= idx < n_options for idx in action)
```

The initial 60-card deck response is not an in-game option label and must be routed separately. A
minimal decision record should retain enough identity to prevent incompatible policies and versions
from being mixed:

```python
DecisionRecord = {
    "game_id": str,
    "date": str,
    "engine_version": str,
    "player": str,
    "team": str,
    "deck_hash": str,
    "opponent_deck_hash": str,
    "outcome": int,
    "turn": int,
    "context": str,
    "observation": dict,
    "options": list[dict],
    "chosen_indices": list[int],
}
```

Raw replays should be converted once into compact Parquet decision and option tables. Date, engine
version, game, team, and exact deck identity must survive the conversion.


### 5. Code architecture: establish boundaries before choosing algorithms

The clearest current layout is:

```text
Archive/MMDD/
  raw replay JSON

DecisionStore/YYYYMMDD/
  decisions.parquet
  option_rows.parquet
  extraction_manifest.json

PilotModelStore/YYYYMMDD/MODEL_ID/
  model.json or model.pt
  feature_schema.json
  train_manifest.json
  offline_metrics.csv

PilotStore/YYYYMMDD/PILOT_ID/
  main.py
  deck.csv
  model artifacts
  package_validation.json
  fidelity_report.json

pokemon_benchmark_runs/RUN_ID/
  benchmark_manifest.json
  matchup_matrix.csv
  runtime_report.json
  holdout_decision.json
```

The implementation follows the same boundaries:

```text
replay parser
-> deterministic featurizer
-> dataset builder
-> trainer
-> runtime scorer
-> safe agent wrapper
-> package validator
-> matchup and fidelity evaluator
```

Training and inference must import the same feature implementation. Separate featurizers almost
inevitably drift in normalization, card resolution, or availability. A parity test should feed a
stored observation through both paths and require identical vectors.

True opponent archetype, exact opponent list, future reveals, and final outcome may be used for
training stratification or labels. They must never enter runtime features because the submitted agent
cannot observe them.


### 6. Stage 0: instrument the rule policy instead of discarding it

A rule-based pilot is not obsolete scaffolding. It provides:

1. a competent data-generating policy;
2. an interpretable baseline for policy differences;
3. a safe fallback when the model is uncertain or fails.

A useful Stage 0 agent records not only its action but its reasoning surface:

```python
RuleDecision = {
    "chosen": [2],
    "rule_scores": [0.1, -0.5, 1.3, 0.2],
    "decision_family": "attack",
    "phase": "pressure",
    "fallback_used": False,
}
```

Decision families should be functional rather than tied to current card names:

```text
setup / search / discard / attach / evolve / switch
attack / damage_target / effect_target / resource_management
```

These labels later support critical-decision weighting and reveal exactly where a learned pilot
fails.


### 7. Stage 1: use a legal-option ranker, not a fixed output classifier

The number and meaning of options change at every decision. A fixed class head for “option 0, 1, 2”
is therefore the wrong representation. Apply a shared encoder to each current option:

```python
class OptionRanker(nn.Module):
    def __init__(self, state_dim, option_dim, hidden=192):
        super().__init__()
        self.state_encoder = MLP(state_dim, hidden)
        self.option_encoder = MLP(option_dim, hidden)
        self.head = MLP(hidden * 2, 1)

    def forward(self, state, options, option_mask):
        h_state = self.state_encoder(state)[:, None, :]
        h_state = h_state.expand(-1, options.size(1), -1)
        h_option = self.option_encoder(options)
        scores = self.head(torch.cat([h_state, h_option], dim=-1)).squeeze(-1)
        return scores.masked_fill(~option_mask, float("-inf"))
```

This design handles variable action counts and is equivariant to menu ordering.

For contexts that select multiple cards, do not introduce a fixed-length output head. Select one
item, mask it, and rescore until `minCount` and `maxCount` are satisfied. Expanding a multi-select
label into the same sequence of masked single picks keeps training and inference behavior aligned.

#### State features

- turn and first-player status;
- prizes, hand/deck/bench counts;
- active and benched Pokémon HP, energy, tools, and status;
- whether supporter, energy, or retreat resources have been spent this turn;
- public opponent active, bench, discard, tools, and energy;
- a compact summary of recent public actions.

#### Option features

- option type and selection context;
- referenced card, Pokémon, or attack ID;
- target area, target player, and board index;
- attack cost, estimated damage, and immediate KO signal;
- evolution, attachment, movement, or switching relationship.

Card embeddings alone encourage memorization of one list. Combining IDs with structural properties
such as stage, HP, type, costs, and public attack features offers a better chance of transferring to a
nearby list.


### 8. Dataset splitting can matter more than model size

#### 8.1 Do not randomly split decisions

Dozens of decisions from one game are highly correlated. A row-level random split places nearly
identical states from one trajectory into both train and validation, inflating accuracy.

At minimum, split whole games. Prefer date- and policy-separated evaluation:

```text
train      : earlier team-deck-policy epochs
validation : a different date or later epoch
holdout    : the first future date arriving after candidate freeze
```

A team may change code during a day, even while keeping the same deck. I therefore also divide a
team-deck stream into sequential blocks of roughly 100 games. This does not identify the exact code
change; it reduces uncontrolled policy mixing.

#### 8.2 Winner-only data creates its own bias

I first assumed that retaining winner actions would produce clean expert data. It can raise average
label quality, but it also removes important information:

- mistakes from favorable games are labeled as expert actions;
- good recovery decisions from difficult losses disappear;
- dominant teams and matchups become overrepresented;
- losing and off-distribution states are scarcely observed.

The current extractor retains both players and stores outcome, opponent strength, team/deck epoch,
decision family, and impact weight separately. The observed action remains the BC label; outcome and
opponent quality become sample weights or inputs to later AWR experiments.

#### 8.3 Easy choices must not hide critical choices

Forced decisions are useful for conformance tests but provide little policy signal. Attack order,
retreat, evolution, search, discard, and target selection may be rare yet decisive. Metrics are
therefore separated into:

```text
overall top-1 / top-3
non-forced top-1 / top-3
critical-choice top-1 / top-3
decision-family accuracy
opponent and phase accuracy
mean chosen rank
```

Optimizing one overall accuracy number was not enough.


---

## Part III. What the first behavior-cloning experiments got wrong

The first models ran and produced plausible offline accuracy. That number did not represent full-game
strength. The four failures below directly changed the split, opponent, fidelity, and holdout design.

### 9. Error note 1: imitation accuracy is not policy strength

The most expensive lesson was treating offline top-1 accuracy as evidence of a strong pilot.

One exact Dragapult list had strong ladder evidence. Its learned pilot exceeded 60% top-1 accuracy on
an offline holdout. Yet focused full-game evaluation produced only about 17.5% and 18.8% score rates
for two policy variants.

Several mechanisms explain the gap:

1. frequent easy decisions dominate aggregate accuracy;
2. one early sequencing error changes every later state;
3. BC enters states absent from expert trajectories after its own mistakes;
4. logs show the chosen action but not the counterfactual outcome of rejected actions;
5. average behavior can look similar while a key matchup direction is reversed.

Offline accuracy answers:

> On states visited by the demonstrator, how often does the model reproduce its action?

It does not answer:

> Can the model recover from the states caused by its own mistakes and finish the game well?

This is the classic distribution-shift and compounding-error problem in imitation learning.


### 10. Error note 2: an archetype-average policy is not an exact deck pilot

The first ranker relied heavily on archetype, context, option type, and card-frequency backoff. It
was easy to implement and produced plausible top-3 accuracy, but it did not understand why the same
card changed value with board state.

```text
The same attack faces different HP and prize situations.
The same search is different with another hand, discard, or spent resource.
Two lists in one archetype can have different win conditions.
```

Version 2 moved to exact-deck, state-conditioned pairwise ranking. The selected action is trained to
score above every rejected action:

$$
\mathcal L_{pair}
=\log\left(1+\exp\left[-(s_{chosen}-s_{other})\right]\right).
$$

This improved representation quality, but it was not a submission certificate. Exact-deck offline
top-1 reached roughly 58–70% in several July 10 experiments, while independent game evaluation still
failed to establish a new submit-ready policy.


### 11. There are two pilot products: match the field or beat it

For a while I evaluated every pilot with the same objective. That was another category error.

| Role | Objective | Evaluation |
|---|---|---|
| **Field-proxy pilot** | Reproduce how the ladder plays a deck | Behavior and matchup fidelity |
| **Adoption pilot** | Play the deck better than the field/reference | Supremacy, fresh holdout, runtime safety |

A field proxy is a benchmark opponent. It should reproduce the direction and approximate magnitude of
real matchups. Faithful imitation is desirable.

An adoption policy is different. If it makes better decisions than the field, imitation accuracy may
decrease. Requiring field fidelity can therefore reject the very policy we hope to submit.

$$
\text{Field proxy}:\quad
\min_\pi |w_\pi(m)-w_{field}(m)|
$$

$$
\text{Adoption pilot}:\quad
\max_\pi w_\pi(m),
\qquad w_\pi-w_{reference}>0.
$$

Learned pilots now enter the system as proxy candidates first. Submit candidates follow a separate
search and holdout path.


### 12. Error note 3: weak opponent pilots make candidates look strong

Copying an exact deck list does not create a faithful opponent. If the local pilot misplays it, the
benchmark becomes easier and inflates candidate scores.

Field proxies require two audits.

#### Behavior fidelity

- Are action distributions similar in important contexts?
- Are attack, search, retreat, and evolution sequences plausible?
- Do opening, setup, pressure, and closing phases have similar behavior?
- Does the policy overuse fallback on rare but critical decisions?

#### Matchup fidelity

Compare the real field rate (r_m) with the local proxy rate (hat r_m):

$$
E_{fidelity}
=\sum_m q_m|\hat r_m-r_m|,
$$

where (q_m) represents current matchup relevance. A low average error is not enough if an important
matchup reverses sign.

Two Festival proxies built from July 14 data achieved same-day weighted matchup errors of roughly
4.6–5.7 percentage points. Because training and field targets came from the same date, they remain
`needs_fidelity`, not `fidelity_pass`. A later date must confirm transfer.


### 13. Error note 4: selection data cannot also be the final holdout

When many models and rule mixtures are tested, selecting the maximum also selects positive noise:

$$
\hat p_{selected}=\max_i(p_i+\epsilon_i).
$$

The more variants we inspect, the larger this winner's-curse bias can become. The evidence must have
separate jobs:

| Split | Purpose |
|---|---|
| Search/train | Choose features, models, and mixture parameters |
| Temporal validation | Compare model families on another completed date |
| Untouched holdout | Evaluate frozen payloads and panel once on a future date |

In the July 14 Spidops holdout, the challenger scored 55.77% and the incumbent 54.81%. The +0.96
percentage-point edge did not meet the precommitted +3-point gate. Some weighted summaries favored
the challenger strongly, but that edge came from small matchup cells and did not override the frozen
raw gate. The holdout was marked consumed and cannot be reused for tuning.

Candidate payloads, opponent panel, metrics, and thresholds should be hash-locked before the future
data arrives. Otherwise “holdout” becomes only a label.


---

## Part IV. Connecting learned pilots to live play safely

After observing BC's limits, I stopped treating a larger model as the default answer. The runtime
architecture instead assigns the learned component only the decisions it covers and keeps validated
rules underneath it.

### 14. A safer runtime policy: blend learning with rules

A hybrid policy is currently more realistic than a standalone network:

$$
s_{final}
=w_e s_{expert}
+w_g s_{global}
+w_r s_{rule}.
$$

- `expert`: a model specialized by decision family, phase, or inferred opponent;
- `global`: a state-conditioned model trained across the full dataset;
- `rule`: a validated heuristic score.

The learned component receives more weight only when it covers the context and has a meaningful
margin. Thin data, low opponent confidence, or low score separation route back to rules.

```python
if route_coverage < coverage_floor:
    return rule_choice(options)

scores = w_model * model_scores + w_rule * rule_scores

if confidence(scores) < confidence_floor:
    return rule_choice(options)

return legal_top_k(scores, min_count, max_count)
```

The wrapper must preserve several invariants:

```text
return the exact fixed 60-card deck on the initial call
return only current option indices during play
respect minCount and maxCount
never return duplicate indices
fall back to a structurally valid action after exceptions
run without importing the local repository
```

Package import, deck initialization, first action, mirror play, and runtime profiling belong to the
learning experiment. A good loss curve is irrelevant if the Validation Episode fails.


### 15. Opponent routing must respect the information boundary

Matchup-specific experts require an estimate of the opponent. Training data contains true archetype
and deck labels, but the live policy does not. Runtime inference may use only revealed cards:

$$
P(z\mid x_{visible})
\propto P(z)\prod_{c\in x_{visible}}P(c\mid z).
$$

Here (z) is an opponent archetype and (x_{visible}) contains public active, bench, discard,
tools, and attached energy. Before informative cards are revealed, the router falls back to current
field priors and generic experts.

Required diagnostics include:

```text
top-1 and top-3 archetype accuracy by turn
confidence and entropy by turn
unknown / low-confidence routing frequency
confusion matrix for important current archetypes
```

Low confidence must skip the specialized route rather than guess.

If one snapshot of revealed cards is insufficient, a later version can summarize public action
history with a GRU or small Transformer. The hidden state must reset at game start, and training
batches must preserve within-game order. I would add recurrence only after it improves matchup
fidelity and calibration over a feed-forward baseline. It cannot reveal hidden cards; it only builds
a better belief state from information that has already become public.


### 16. DAgger-style repair: learn from states caused by the current policy

BC sees expert states. After one mistake, the learned policy may enter unfamiliar states and compound
the error. [DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses this by executing the
current policy, asking an expert for labels on visited states, and aggregating those rows.

Full DAgger is unavailable here because the original ladder agent cannot be queried on arbitrary new
states. A practical approximation is:

```text
1. run full games with the current learned pilot;
2. collect states where model, rules, and teachers disagree;
3. relabel high-impact decisions with validated rules or a teacher zoo;
4. train only on synthetic stress rows with sufficient teacher agreement;
5. keep official and synthetic labels explicitly separated.
```

Teacher IDs are not automatically diverse. If several variants return identical actions on stored
official snapshots, they count as one effective behavior.

Synthetic trajectories can expand state coverage, but they cannot replace field fidelity, deck-power
evidence, temporal validation, or the untouched holdout.


---

## Part V. Moving beyond behavior cloning

Only here do reward and value estimation enter the pipeline. AWR, offline RL, reward shaping, and
self-play are added one at a time, while retaining the policy representation and evaluation system
already tested in earlier stages.

### 17. Outcome-aware imitation: AWR is a practical next step

Plain BC copies every observed action with equal weight. The next step can give more weight to actions
associated with better estimated outcomes:

At this point the BC record must be expanded into an RL transition:

```python
Transition = {
    "observation": o_t,
    "options": A_t,
    "action": a_t,
    "reward": r_t,
    "next_observation": o_next,
    "next_options": A_next,
    "done": done,
}
```

`next` must mean the next observation on which the same player makes a decision, not merely the next
JSON row. It must include intervening opponent actions and engine resolution. When no intermediate
reward is available, the clean baseline is zero reward for non-terminal transitions and the game
result on the terminal transition.

$$
\mathcal L_{AWR}
=-w_t\log\pi_\theta(a_t\mid o_t,A_t),
\qquad
w_t=\min\left(w_{max},\exp\frac{\hat A_t}{\beta}\right).
$$

[Advantage-Weighted Regression](https://arxiv.org/abs/1910.00177) fits a value function and then
uses estimated advantage to weight policy regression. It is attractive here because it extends the
BC pipeline rather than replacing it entirely.

The advantage target cannot simply be “+1 for every action in a win.” It should account for:

```text
terminal result
opponent strength
baseline matchup difficulty
estimated state value
source and teacher reliability
```

A value head can begin as an auxiliary terminal-outcome predictor. Its temporal calibration must be
checked before it is trusted to reweight the policy; an incorrect value model can make BC worse.


### 18. When is offline RL justified?

Replays contain outcomes only for chosen actions. They do not reveal what would have happened after
rejected options. Naive Q-learning can assign unrealistically high value to actions rarely represented
in the dataset.

[Conservative Q-Learning](https://papers.nips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
explicitly addresses overestimation under offline distribution shift.
[Implicit Q-Learning](https://arxiv.org/abs/2110.06169) attempts policy improvement without directly
evaluating out-of-dataset actions.

Choosing the algorithm name is not the first decision. The prerequisites are:

| Prerequisite | Why it matters |
|---|---|
| BC and hybrid pilots finish games reliably | State and action representations must work first. |
| Date- and policy-separated datasets exist | Offline evaluation leakage must be controlled. |
| Critical decisions have sufficient support | Otherwise the model learns mostly routine actions. |
| Opponent proxies reproduce matchup directions | Policy improvement needs a trustworthy evaluator. |
| Reward and value calibration are checked | A wrong objective can be optimized very effectively. |
| A future holdout remains untouched | Model-selection bias must be measured. |

Some target decks are only now reaching these prerequisites. CQL and IQL are next experiments, not
established solutions for this competition.

Their reference implementations are not drop-in solutions either. Because the legal action set
changes at every decision, the critic must score (Q(o,c_i)) for each current option, and the
Bellman target must be restricted to `next_options`. Incorrect masking or multi-select handling can
corrupt value learning even when the offline algorithm itself is conservative.


### 19. Reward design: do not replace winning with a convenient proxy

The safest primary reward is the terminal result:

$$
r_T\in\{-1,0,+1\}.
$$

The difficulty is delay and sparse feedback. Intermediate shaping can help, but a proxy can become a
new and incorrect objective.

#### Dangerous examples

- rewarding draws can encourage unnecessary draw loops;
- rewarding damage can prefer immediate damage over prize planning;
- rewarding bench development can expose the board to spread attacks;
- rewarding deck preservation can discourage necessary search.

When possible, shape with a potential difference:

$$
r'_t=r_t+\gamma\Phi(o_{t+1})-\Phi(o_t).
$$

Potential candidates can combine prize differential, immediate KO risk, next-turn attack readiness,
and resource exhaustion risk. Every shaped policy still needs ablations:

```text
Does terminal win performance improve without the shaping metric present at evaluation?
Does the policy exploit one proxy repeatedly?
Is the effect direction consistent across important matchups?
Do catastrophic behaviors increase relative to the rule baseline?
```

BC, AWR, and terminal-return prediction should work before elaborate shaping is introduced.


### 20. Self-play should be a league, not one mirror matchup

Repeated self-play on one deck can produce a mirror specialist. It may never learn current counters,
and two policies can cycle through mutually exploitable strategies.

A useful league would contain:

```text
current rule incumbents
validated field proxies
recent strong public references
frozen historical checkpoints
aggressive, defensive, and denial stress policies
multiple generations of the learned policy
```

Opponent sampling should use current field weight, weakness priority, recent losses, policy diversity,
and exploration for under-sampled opponents.

Beating the latest copy of itself is not a promotion criterion. A self-play policy must retain its edge
against frozen references and a new holdout, and it should add complementary wins to the final agent
portfolio.


---

## Part VI. Evaluation before algorithm choice

The evaluation contract remains the same when the learning algorithm changes. Offline metrics,
full-game execution, proxy fidelity, temporal transfer, and a future holdout must be checked in order.
When local and live results disagree, the cause is diagnosed before another policy is tuned.

### 21. The evaluation ladder

| Stage | Evidence | If it fails |
|---|---|---|
| 1. Data contract | Alignment, deck-action separation, engine version | Fix the parser. |
| 2. Offline | Non-forced and critical top-k on separated dates | Fix features and balance. |
| 3. Runtime | Isolated import, exact deck, valid actions, mirror smoke | Fix packaging and fallback. |
| 4. Behavior fidelity | Decision-family, phase, and opponent behavior | Do not use it as a field proxy. |
| 5. Matchup fidelity | Error and sign against the real field matrix | Repair the opponent panel. |
| 6. Reference supremacy | Same panel and seats versus the incumbent | Do not promote it. |
| 7. Temporal validation | Effect survives on another date | Record overfitting. |
| 8. Untouched holdout | Frozen payload clears a future-date gate | Do not reuse the holdout. |
| 9. Live calibration | Local-live difference can be diagnosed | Repair the evaluator first. |

Raw win rate is not enough. Rating-based matchmaking can keep strong agents near 50% while they face
stronger opponents. Local evaluation therefore uses a common panel, seat balance, opponent-strength
adjustment, matchup slices, and uncertainty.

The public engine interface did not expose a reliable seed control for identical random streams, so
true common-random-number pairing was unavailable. I instead use common opponent composition, balanced
seats, larger samples, Wilson intervals, and independent holdouts.


### 22. Diagnosing a local-good, live-poor result

“The pilot is weak” is only one possible cause.

| Cause | Diagnostic signal | Repair |
|---|---|---|
| Weak opponent pilots | Local matchups are much easier than field results | Rebuild and validate proxies. |
| Missing coverage | Important current decks have few or no local games | Expand the panel. |
| Stale weights | Local panel weights differ from the current field | Reweight with current evidence. |
| Adoption pilot only matches average play | Fidelity is high but reference supremacy is absent | Search for a beat-field policy. |
| Winner's curse | Performance drops after candidate selection | Use shrinkage and a fresh holdout. |
| Meta drift | Opponent distribution changed after submission | Separate dates and drift. |
| Runtime or package issue | Errors, fallback, or latency increase live | Inspect self-game and package diagnostics. |

The governing principle is to measure the dominant cause before repairing anything. Tuning a pilot
cannot fix an incomplete panel or a reused holdout.


---

## Part VII. Current status and the next implementation order

The final part separates what is already running in code from what remains a research plan. It also
provides the order I would follow if I restarted the project and a checklist of mistakes not to repeat.

### 23. What is implemented as of July 15

The following path now exists in code:

```text
official replay parsing
-> both-player decision extraction
-> exact-deck option rows
-> state-conditioned pairwise ranker
-> small isolated runtime package
-> initial-deck / first-action / mirror validation
-> field-proxy matchup fidelity
-> seat-balanced benchmark
-> one-use future holdout and artifact index
```

One recent daily corpus produced about 588,000 decisions from both players. Example exact-deck
training sets were:

| Target | Decisions | Epoch/temporal holdout top-1 | Interpretation |
|---|---:|---:|---|
| New Spidops exact deck | 12,228 | 54.9% | Needs a new multi-date pilot cycle |
| Festival exact deck A | 19,372 | 71.8% | Good same-day fit; future transfer untested |
| Festival exact deck B | 10,751 | 59.6% | Some matchup error remains too large |

These numbers show that the training path works. They do not prove submission strength. The Spidops
challenger, for example, cleared the incumbent by only 0.96 points on its untouched holdout and was
not promoted.

This is not a 100% complete RL system. The remaining work includes:

- merging exact-deck behavior across multiple dates;
- expanding critical-decision and difficult-matchup state coverage;
- testing teacher reliability across dates;
- calibrating the value model required by AWR;
- evaluating offline RL and self-play on independent holdouts;
- accumulating long-run local-live calibration.


### 24. The implementation order I would use now

#### Milestone 1: data contract

```text
[ ] test step-i observation against step-(i+1) action
[ ] separate the 60-card deck response from in-game decisions
[ ] assert every action index is in range
[ ] store engine version, date, game, team, and exact deck hash
[ ] prevent one game from crossing train/holdout boundaries
```

#### Milestone 2: instrumented rule baseline

```text
[ ] record rule scores by decision family
[ ] record fallback use
[ ] record runtime and errors
[ ] create an exact deck + policy manifest
```

#### Milestone 3: BC v1

```text
[ ] shared state/option featurizer
[ ] variable-length option ranker
[ ] non-forced, critical, and family metrics
[ ] date- or epoch-separated holdout
[ ] runtime wrapper with rule fallback
```

#### Milestone 4: field-proxy validation

```text
[ ] compare behavior profiles
[ ] measure matchup error and wrong-sign cells
[ ] compute current field coverage
[ ] pass package, mirror, and runtime tests
```

#### Milestone 5: adoption-pilot search

```text
[ ] search a small rule/model mixture surface
[ ] compare on the same opponents and seats as the incumbent
[ ] separate selection games from final validation
[ ] hash-lock candidate and panel
[ ] consume one future untouched holdout
```

#### Milestone 6: outcome-aware learning

```text
[ ] calibrated value head
[ ] opponent-strength and outcome weighting
[ ] AWR or related advantage-weighted BC
[ ] pass the same evaluation ladder as BC
```

#### Milestone 7: limited offline RL and population self-play

```text
[ ] small CQL/IQL experiment
[ ] audit out-of-distribution action values
[ ] build a diverse opponent league
[ ] evaluate against frozen checkpoints
[ ] remove shaping in ablations
[ ] use a new date holdout and live calibration
```

Do not advance because the next algorithm sounds more powerful. A new optimizer does not repair a
broken parser, unfaithful pilot, or incomplete evaluator.


### 25. Mistakes I do not want to repeat

| Earlier approach | Why it failed | Current rule |
|---|---|---|
| Copy a high-scoring deck and assume it is strong locally | Deck and pilot were conflated. | Version exact deck and runtime policy separately. |
| Treat aggregate top-1 as policy quality | Easy decisions dominated. | Report critical, family, matchup, and full-game metrics. |
| Train only on winner actions | Survivor and matchup bias increased. | Retain both players; use outcome as metadata or weight. |
| Randomly split decision rows | One trajectory leaked across splits. | Split by game, date, and policy epoch. |
| Declare fidelity on the training date | Temporal transfer was unknown. | Require a later-date validation. |
| Benchmark against weak proxies | Local performance was inflated. | Validate proxy behavior and matchup direction first. |
| Trust the best of many variants | Winner's curse was ignored. | Use shrinkage and a fresh holdout. |
| Rank deck power by raw win rate | Opponent difficulty was ignored. | Model opponent strength and exact policy epochs. |
| Plan self-play before evaluator quality | The policy could overfit a weak league. | Start with BC and validated hybrid pilots. |
| Postpone package validation | Good models failed validation episodes. | Treat isolated import and mirror smoke as model tests. |

This table may be the most useful RL result from the project so far. A larger model repeats the same
mistakes with greater confidence if the source of failure remains mixed together.


### 26. A reading order for the RL methods

There is no need to understand every derivation at once. Read in the order that matches the current
implementation bottleneck.

1. **Behavior cloning and distribution shift**
   Understand why supervised accuracy degrades after the policy changes its own state distribution.

2. **[DAgger](https://proceedings.mlr.press/v15/ross11a.html)**
   Learn why states visited by the current policy must return to the dataset.

3. **[Advantage-Weighted Regression](https://arxiv.org/abs/1910.00177)**
   See how to retain a supervised policy update while emphasizing better actions.

4. **[Conservative Q-Learning](https://papers.nips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)**
   Study overestimation of actions unsupported by an offline dataset.

5. **[Implicit Q-Learning](https://arxiv.org/abs/2110.06169)**
   Study policy improvement without directly evaluating out-of-dataset actions.

After reading a paper, connect it to one measured failure and run the smallest controlled experiment.
Do not rewrite the full system around a method name.


### 27. Conclusion

RL in this project has not meant replacing every rule with a neural network. It has meant building a
sequence of increasingly demanding contracts:

```text
define the observable information boundary
-> extract decisions with correct alignment
-> rank variable legal-option menus
-> evaluate critical decisions and temporal transfer
-> distinguish field-matching opponents from field-beating candidates
-> retain runtime safety through rule fallback
-> incorporate outcome information gradually
-> advance only when an untouched holdout preserves the edge
```

Replay extraction, state-conditioned ranking, isolated pilot packages, fidelity checks, and one-use
future holdouts are now connected. Offline RL and population self-play are still preparation-stage
work. Saying so matters.

The largest lesson came before the choice of RL algorithm: **give data and evaluation artifacts one
clear job each**. If deck quality, pilot quality, opponent weakness, and repeated exposure to the same
evidence are mixed together, almost any score can be misread.

The next Working Note will go deeper into multi-date exact-deck datasets, critical-decision coverage,
teacher zoos, and temporal pilot fidelity as concrete experiments rather than abstract roadmap items.
