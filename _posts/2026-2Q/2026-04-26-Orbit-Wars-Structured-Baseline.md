---
title: "Orbit Wars: Structured Baseline Methodology"
date: 2026-04-26 19:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, orbit-wars, game-ai, simulation, reinforcement-learning]
math: true
pin: false
---

# 🛰️ Orbit Wars: Structured Baseline Methodology

Competition link:
[Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)

Kaggle notebook link:
[Orbit Wars: Structured Baseline](https://www.kaggle.com/code/pilkwang/orbit-wars-structured-baseline)

Related benchmark notebook:
[Benchmark: How Strong Is Your Orbit Wars Agent?](https://www.kaggle.com/code/pilkwang/benchmark-how-strong-is-your-orbit-wars-agent)

<p style="font-size: 1.14rem; line-height: 1.7;">
Kaggle opened <strong>Orbit Wars</strong> on <strong>April 16, 2026 UTC</strong> as a Featured competition around a continuous-space strategy simulator.
In simulation terms, the problem can be read as <em>online multi-agent control with delayed action resolution</em>: launches are chosen now, but their consequences appear only after movement, production, existing fleets, and same-turn combat have changed the board.
</p>

This framing also matches the standard reinforcement-learning notation.
The simulator defines a transition kernel, a reward process, and a policy over repeated episodes:

$$
\pi(a_t \mid s_t), \quad
s_{t+1} \sim P(s_{t+1}\mid s_t,a_t), \quad
G_t = \sum_{k \ge 0}\gamma^k r_{t+k}
$$

The baseline structure below makes those simulator contracts explicit before introducing any learned policy.
The operational condition is:

> Find a legal direct shot, forecast the target at the true arrival time, and spend ships only when the mission still makes sense after earlier launches are committed.

This condition defines the code structure.
**Physics** determines whether an action is legal, the **world model** determines what the board will look like at arrival time, and **strategy** chooses between mission families under those forecasts.

| Core idea                        | Meaning                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| <u>Legal route first</u>         | Do not score a move before checking direct-shot feasibility                          |
| <u>Arrival-time state</u>        | Evaluate targets at ETA, not from the current snapshot                               |
| <u>Mission semantics</u>         | Treat capture, rescue, reinforce, recapture, snipe, and swarm as different contracts |
| <u>Commitment-aware planning</u> | Accepted launches become future facts for later decisions                            |

## 🧭 Agent Loop As A Contract

The one-turn loop exposes the constraints that later layers must satisfy.
The observation fields needed by this baseline are:

$$
O_t =
\left(
player,\ planets_t,\ fleets_t,\ \omega,\ initial\_planets,\ comets_t
\right)
$$

The output is a set of launches:

$$
A_t =
\left[
(source_j,\ \theta_j,\ n_j)
\right]_{j=1}^{k_t}
$$

where:

| Symbol     | Meaning                   |
| ---------- | ------------------------- |
| `source_j` | source planet id          |
| `theta_j`  | launch angle in radians   |
| `n_j`      | number of ships to launch |

The notebook implements a deterministic policy:

$$
\pi : O_t \mapsto A_t
$$

The important part is how each launch is validated before it becomes an action:

| Object / Signal    | Model Use                                           | Main Failure If Ignored             |
| ------------------ | --------------------------------------------------- | ----------------------------------- |
| 🪐 Planet state     | source inventory, target value, production forecast | over-expansion or weak defense      |
| 🚀 Fleet state      | future arrival ledger                               | stale ownership estimate            |
| ☀️ Sun              | direct-route legality                               | fleet destroyed on invalid path     |
| 🔄 Angular velocity | moving-target ETA prediction                        | aiming at the wrong future position |
| ☄️ Comet paths      | short-horizon moving target                         | chasing expired opportunities       |

The policy objective can be viewed as online planning under delayed action effects:

$$
\max_{\pi}\ \mathbb{E}\left[ R_{\text{final}} \mid \pi,\ O_0 \right]
$$

The hard part is the delay:

> **A launch selected at turn `t` resolves at `t + T`, after production, movement, visible fleets, and other players' arrivals have changed the board.**

## ⚡ TL;DR

| Theme          | Short Version                                                              |
| -------------- | -------------------------------------------------------------------------- |
| Main risk      | Invalid routes, stale target state, and bad tactical timing                |
| Physics layer  | Rejects sun-crossing routes and predicts moving-target intercepts          |
| World model    | Evaluates **ownership at arrival time**                                    |
| Strategy layer | Separates defense, capture, snipe, swarm, crash, and cleanup missions      |
| Commit loop    | Converts accepted launches into future arrivals before judging later moves |

## 🌍 Why Snapshot Logic Is Not Enough

Snapshot heuristics are brittle because the action selected now resolves later.
The baseline therefore treats every meaningful decision as an ETA-indexed query:

| Constraint                                  | Why Snapshot Heuristics Break                                   |
| ------------------------------------------- | --------------------------------------------------------------- |
| Continuous `100 x 100` board                | Distance and angle are real-valued, not grid steps              |
| `500` turns                                 | Long-horizon production and late arrivals matter                |
| `2` or `4` players                          | A move that is safe in a duel can be punished in a 4-player map |
| Very short action budget                    | Heavy search must be bounded                                    |
| `[from_planet_id, angle, num_ships]` action | The agent controls angle and send size directly                 |
| Rotating planets                            | Target coordinates can change before arrival                    |
| Ship-count-dependent speed                  | Send size changes ETA                                           |
| Sun collision                               | Some direct lines are illegal                                   |
| Same-turn combat                            | Multiple arrivals interact nonlinearly                          |

The central question is not:

> Which planet looks valuable right now?

It is:

> If a fleet with this ship count leaves this source now, what will the target state be at its arrival turn?

That time shift is the reason the baseline is organized around a **forward world model** instead of *snapshot-only* heuristics.

## 📌 The Rules That Shape The Agent

The baseline is easier to understand by first naming the physical constraints that the agent must preserve:

| Rule                                         | Practical consequence                                                         |
| -------------------------------------------- | ----------------------------------------------------------------------------- |
| Fleets travel in straight lines after launch | No waypoint path or curved detour can be assumed                              |
| A fleet touching the sun is destroyed        | Every candidate launch needs sun-safety filtering                             |
| Inner planets may rotate                     | Target position at arrival can differ from target position now                |
| Outer planets are effectively static         | A blocked static shot usually stays blocked unless the source or send changes |
| Comets behave like temporary moving planets  | Comet value must be short-horizon and timing-aware                            |
| Fleet speed depends on ship count            | A 10-ship launch and a 20-ship launch can have different ETAs                 |
| Same-turn arrivals are grouped by owner      | Ownership need must resolve owner-aware combat, not just add raw ships        |
| The largest two opposing forces cancel first | Simple `garrison + 1` estimates can be wrong                                  |
| Launches should not be merged at execution   | `(10) + (10)` is not the same action as `(20)` because speed and ETA change   |

This is why the notebook spends substantial code on feasibility checks before value scoring.
In Orbit Wars, physics is not a background detail.
Physics determines whether a tactical idea is even a legal action.

## 🧱 Why The Notebook Is Split Into Three Models

The notebook separates the agent into physics, world model, and strategy because those layers answer different mathematical questions.
Combining them into one large heuristic would make the policy shorter, but it would also mix legal feasibility, future-state prediction, and preference scoring into one opaque calculation.

The **modeling order** is:

$$
\text{physics} \rightarrow \text{world model} \rightarrow \text{strategy}
$$

The meaning is:

| Layer         | Owns This Question                                   | Does Not Own                 |
| ------------- | ---------------------------------------------------- | ---------------------------- |
| 🧲 Physics     | Is one direct launch legal, and when does it arrive? | Target value                 |
| 🌍 World model | Who owns the target at turn `T`?                     | Strategic preference weights |
| 🧭 Strategy    | Is this mission worth ships?                         | Raw route geometry           |

Lower layers should stay independent of higher-layer preferences.
Physics should not know whether a target is strategically attractive.
The world model should not change combat rules because a mission looks valuable.
Strategy should not assume ownership without asking the world model.

Runtime calls can point the other way, because strategy asks the world model for facts and the world model uses physics for route timing.
But the conceptual construction is still **physics first, world model second, strategy third**.

The decision pipeline can be summarized as:

$$
\text{legal route}
\Rightarrow
\text{arrival-time state}
\Rightarrow
\text{mission score}
\Rightarrow
\text{committed launch}
$$

This is the main structural idea of the notebook.
Each launch is accepted only when all four statements are true:

$$
\operatorname{Legal}(s,i,n)
\land
\operatorname{ArrivesUseful}(i,T)
\land
\operatorname{OwnsOrHolds}(i,T,n)
\land
\operatorname{BudgetSafe}(s,n)
$$

## 🧲 The Constraint Model

The baseline uses a small physics model before it allows strategy to think about value.
The board is treated as a square plane:

$$
p = (x, y), \quad x, y \in [0, 100]
$$

The sun sits at the center:

$$
c = (50, 50), \quad R_{\odot} = 10
$$

The agent adds an implementation safety margin around the sun:

$$
R_{\text{blocked}} = R_{\odot} + \epsilon
$$

where the notebook uses `epsilon = 1.5`.
This is intentionally conservative: **barely legal routes are treated as risky routes**.
The strategy layer only sees candidates that have already passed this feasibility filter.

For a source planet `s` and target planet `t`, the route is not timed from center to center.
It is timed from the source boundary to the first hit on the target circle:

$$
D(s,t) \approx \max\left(0,\lVert p_t - p_s \rVert - r_s - r_t - \delta\right)
$$

where `delta` is the launch clearance used to avoid starting exactly on the source boundary.
The direction is still the direct angle:

$$
\theta = \operatorname{atan2}(y_t-y_s, x_t-x_s)
$$

but the legal path is the segment from the launch point to the target-circle hit point.
That segment must not intersect the blocked sun disk:

$$
\operatorname{dist}\left(c,\overline{ab}\right) \ge R_{\text{blocked}}
$$

where `a` is the launch point and `b` is the predicted hit point.

Ship count changes fleet speed, so the baseline cannot reuse one ETA for all send sizes.
The notebook mirrors the game speed curve with:

$$
v(n) =
\begin{cases}
1, & n \le 1 \\
1 + 5 \cdot \left(\operatorname{clip}\left(\frac{\log n}{\log 1000},0,1\right)\right)^{1.5}, & n > 1
\end{cases}
$$

Then the estimated arrival turn is:

$$
T(s,t,n) = \left\lceil \frac{D(s,t)}{v(n)} \right\rceil
$$

This formula explains a large part of the code structure.
Changing `n` can change the speed, which changes the arrival turn, which changes the target position, which can make the same route legal or illegal.
That is why final sends are re-aimed.

For rotating planets, the target position is predicted at the arrival turn.
With center `c`, orbit radius `rho`, current angle `theta_0`, and angular velocity `omega`, the simplified position model is:

$$
p_t(T) =
c + \rho
\begin{bmatrix}
\cos(\theta_0 + \omega T) \\
\sin(\theta_0 + \omega T)
\end{bmatrix}
$$

The notebook treats planets outside the rotation boundary as static.
In the code this is the condition:

$$
\rho + r \ge 50
$$

Comets are handled as temporary moving planets whose future positions come from their path arrays.
That creates a practical constraint: even a high-value comet is only valuable if a legal shot can arrive before the comet's usable path disappears.

## 🗺️ The Baseline As A Decision Pipeline

The notebook is organized around one pipeline:

| Layer       | Question                                                          | Output                                   |
| ----------- | ----------------------------------------------------------------- | ---------------------------------------- |
| Physics     | Can this source legally reach this target with a direct shot?     | angle, ETA, or rejection                 |
| World model | Who owns the target at the ETA after visible arrivals and combat? | projected owner and garrison             |
| Strategy    | Which mission type is worth ships under that forecast?            | scored candidate mission                 |
| Commit loop | Which future facts must later decisions observe?                  | remaining inventory and planned arrivals |

Each layer has a separate responsibility.
If an action is invalid or strategically weak, the failure can usually be localized:

- bad angle or illegal route: inspect physics,
- bad target ownership estimate: inspect world model,
- bad priority: inspect strategy scoring,
- duplicate or overcommitted send: inspect the commit loop.

This separation keeps the agent auditable.
A route error belongs to physics, an ownership error belongs to the world model, and a priority error belongs to strategy scoring.

## 🧲 Layer 1: Physics Without Fake Routes

The physics layer has a deliberately narrow responsibility:

> Decide whether one legal direct launch exists.

That sounds modest, but it removes a large class of mistakes.
The agent does not invent waypoint paths.
It does not pretend that a source-to-target center line is enough.
It computes launch and hit geometry from the source boundary to the target circle, estimates travel time from fleet speed, and rejects direct segments that pass through the sun safety zone.

Moving targets add another complication.
A rotating planet or comet may not be where it is now by the time a fleet arrives.
The notebook handles this by searching future direct intercept windows instead of treating motion as a small visual detail.
Static targets are handled differently: if the direct segment is blocked by the sun, the strategy should not imagine that time will magically solve it.

This design principle is:

> Do not let strategy spend ships until physics has certified that the action exists.

In game environments, a lower-value legal move is preferable to a high-value action that the simulator will invalidate or punish.

The ship-count dependency is especially important.
If the final send amount changes, the route can change too, because the fleet speed and arrival turn may change.
That is why the notebook re-aims the final send instead of assuming a route found for one probe size remains valid for every other size.

| Physics Check   | Formula / Rule                  | Strategic Effect                            |
| --------------- | ------------------------------- | ------------------------------------------- |
| 🧭 Direct angle  | `atan2(y_t-y_s, x_t-x_s)`       | no fake waypoint route                      |
| ☀️ Sun safety    | `dist(c, segment) >= R_blocked` | unsafe launches are rejected                |
| 🚀 Fleet speed   | `v(n)`                          | send size changes ETA                       |
| 🔄 Moving target | `p_t(T)`                        | rotating targets require intercept search   |
| 🪨 Static target | `rho + r >= 50`                 | waiting does not open the same blocked shot |

<details markdown="1">
<summary>Show the routing and moving-target helpers</summary>

```python
def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)


def segment_hits_sun(x1, y1, x2, y2, safety=SUN_SAFETY):
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety


def launch_point(sx, sy, sr, angle):
    clearance = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * clearance, sy + math.sin(angle) * clearance


def actual_path_geometry(sx, sy, sr, tx, ty, tr):
    angle = math.atan2(ty - sy, tx - sx)
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_distance = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def safe_angle_and_distance(sx, sy, sr, tx, ty, tr):
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry(
        sx, sy, sr, tx, ty, tr
    )
    if segment_hits_sun(start_x, start_y, end_x, end_y):
        return None
    return angle, hit_distance


def estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns
```

</details>

<details markdown="1">
<summary>Show self-consistent moving-target intercept search</summary>

```python
def predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids):
    if target.id in comet_ids:
        return predict_comet_position(target.id, comets, turns)
    return predict_planet_position(target, initial_by_id, ang_vel, turns)


def search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    best = None
    best_score = None
    max_turns = min(HORIZON, ROUTE_SEARCH_HORIZON)
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))

    for candidate_turns in range(1, max_turns + 1):
        pos = predict_target_position(
            target, candidate_turns, initial_by_id, ang_vel, comets, comet_ids
        )
        if pos is None:
            continue

        est = estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
        if est is None:
            continue

        _, turns = est
        if abs(turns - candidate_turns) > INTERCEPT_TOLERANCE:
            continue

        actual_turns = max(turns, candidate_turns)
        actual_pos = predict_target_position(
            target, actual_turns, initial_by_id, ang_vel, comets, comet_ids
        )
        if actual_pos is None:
            continue

        confirm = estimate_arrival(
            src.x, src.y, src.radius, actual_pos[0], actual_pos[1], target.radius, ships
        )
        if confirm is None:
            continue

        delta = abs(confirm[1] - actual_turns)
        if delta > INTERCEPT_TOLERANCE:
            continue

        score = (delta, confirm[1], candidate_turns)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], actual_pos[0], actual_pos[1])

    return best
```

</details>

<details markdown="1">
<summary>Show route probing over ship-count-dependent ETAs</summary>

```python
def plan_shot(self, src_id, target_id, ships):
    ships = int(ships)
    key = (src_id, target_id, ships)
    cached = self.shot_cache.get(key)
    if key in self.shot_cache:
        return cached

    src = self.planet_by_id[src_id]
    target = self.planet_by_id[target_id]
    result = aim_with_prediction(
        src,
        target,
        ships,
        self.initial_by_id,
        self.ang_vel,
        self.comets,
        self.comet_ids,
    )
    self.shot_cache[key] = result
    return result


def probe_ship_candidates(self, src_id, target_id, source_cap, hints=()):
    source_cap = max(1, int(source_cap))
    target = self.planet_by_id[target_id]
    target_ships = max(1, int(math.ceil(target.ships)))

    values = set(range(1, min(6, source_cap) + 1))
    values.update(
        {
            source_cap,
            max(1, source_cap // 2),
            max(1, source_cap // 3),
            min(source_cap, PARTIAL_SOURCE_MIN_SHIPS),
            min(source_cap, target_ships + 1),
            min(source_cap, target_ships + 2),
            min(source_cap, target_ships + 4),
            min(source_cap, target_ships + 8),
        }
    )

    for hint in hints:
        base = max(1, min(source_cap, int(math.ceil(hint))))
        for delta in (-2, -1, 0, 1, 2):
            candidate = base + delta
            if 1 <= candidate <= source_cap:
                values.add(candidate)

    return sorted(values)


def best_probe_aim(self, src_id, target_id, source_cap, hints=(), anchor_turn=None):
    best = None
    best_key = None

    for ships in self.probe_ship_candidates(src_id, target_id, source_cap, hints=hints):
        aim = self.plan_shot(src_id, target_id, ships)
        if aim is None:
            continue

        angle, turns, dist_to_target, path_target = aim
        key = (turns, ships) if anchor_turn is None else (abs(turns - anchor_turn), turns, ships)
        if best_key is None or key < best_key:
            best_key = key
            best = (ships, (angle, turns, dist_to_target, path_target))

    return best
```

</details>

## 🌍 Layer 2: Ownership At Arrival Time

The world model asks the question that snapshot heuristics usually avoid:

> What will this planet look like when the controlled fleet arrives?

To answer that, it builds an arrival ledger from visible fleets, projects production, and resolves same-turn combat.
The same-turn part is not cosmetic.
Arrivals are grouped by owner, opposing forces cancel in order, and only then does the survivor interact with the planet garrison.
That means a target can be much cheaper or much more expensive than a snapshot estimate suggests.

For each planet, the world model maintains a projected state:

$$
S_i(t) = \left(o_i(t), g_i(t)\right)
$$

where `o_i(t)` is the owner and `g_i(t)` is the garrison at turn `t`.
Before arrivals resolve, a non-neutral owner receives production:

$$
g_i^-(t) =
\begin{cases}
g_i(t-1) + q_i, & o_i(t-1) \ne -1 \\
g_i(t-1), & o_i(t-1) = -1
\end{cases}
$$

Here `q_i` is planet production and `-1` denotes neutral ownership.

Next, all arrivals for that turn are aggregated by owner:

$$
A_k(t) = \sum_{f \in \mathcal{F}_{i,t},\ \operatorname{owner}(f)=k} \operatorname{ships}(f)
$$

The two largest opposing arrival groups are compared first.
If the largest and second-largest groups are equal, neither side survives the arrival fight.
Otherwise the largest group survives with the difference:

$$
B(t) = A_{k_1}(t) - A_{k_2}(t)
$$

Only that survivor then fights the current garrison.
If the survivor has the same owner as the planet, it reinforces:

$$
g_i(t) = g_i^-(t) + B(t)
$$

If the survivor is hostile, it subtracts from the garrison:

$$
g_i(t) = g_i^-(t) - B(t)
$$

and ownership flips only when this value becomes negative.
That is the detail that makes simple estimates such as `current ships + 1` unreliable.

This matters because several superficially different missions share the same underlying contract:

| Mission     | World-Model Question                                               |
| ----------- | ------------------------------------------------------------------ |
| 🏴 Capture   | Can the controlled player own the target after arrival?            |
| 🛟 Rescue    | Can reinforcement arrive before the controlled planet falls?       |
| 🛡️ Reinforce | Can ownership be preserved through a future horizon?               |
| 🔁 Recapture | If rescue is too late, can ownership be restored soon after loss?  |
| 🎯 Snipe     | Can another player's arrival reduce the required ship count?       |
| 🤝 Swarm     | Can multiple partial sources combine at one useful arrival window? |

All of those missions depend on the same ownership-need primitive.
The question is not "How many ships are on the planet now?"
The question is "How many ships are needed to own it at the relevant time?"

Mathematically, the ownership need can be written as:

$$
N_i(T, a) =
\min \left\{
n \ge 0 :
o_i\left(T;\ \mathcal{A}_i \cup \{(T,a,n)\}\right) = a
\right\}
$$

where:

- `i` is the target planet,
- `T` is the arrival turn,
- `a` is the attacking owner,
- \(\mathcal{A}_i\) is the already-known arrival ledger for that planet.

The notebook computes this by simulation and binary search rather than by a closed-form shortcut.
That choice is important because same-turn combat, production, planned commitments, and multi-owner arrivals interact in ways that are easy to get wrong by hand.

Reinforcement uses a related but stricter question.
It is not enough to own the planet at one instant.
The defense mission may need the planet to remain owned through a horizon:

$$
H_i(\tau, L) =
\min \left\{
n \ge 0 :
\forall t \in [\tau, L],\
o_i\left(t;\ \mathcal{A}_i \cup \{(\tau,p,n)\}\right) = p
\right\}
$$

Here `tau` is the reinforcement arrival turn, `L` is the hold-until turn, and `p` is the controlled player id.
This is the formal difference between a one-turn rescue and a true reinforce-to-hold mission.

<details markdown="1">
<summary>Show same-turn combat and timeline simulation</summary>

```python
def resolve_arrival_event(owner, garrison, arrivals):
    by_owner = {}
    for _, attacker_owner, ships in arrivals:
        by_owner[attacker_owner] = by_owner.get(attacker_owner, 0) + ships

    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_players = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)

    if owner == survivor_owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def simulate_planet_timeline(planet, arrivals, player, horizon):
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production

        prev_owner = owner
        group = by_turn.get(turn, [])
        if group:
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn

        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "fall_turn": fall_turn,
        "horizon": horizon,
    }
```

</details>

<details markdown="1">
<summary>Show arrival-time ownership and hold queries</summary>

```python
def projected_state(self, target_id, arrival_turn, planned_commitments=None, extra_arrivals=()):
    planned_commitments = planned_commitments or {}
    cutoff = max(1, int(math.ceil(arrival_turn)))
    if not planned_commitments.get(target_id) and not extra_arrivals:
        return state_at_timeline(self.base_timeline[target_id], cutoff)

    arrivals = [
        item for item in self.arrivals_by_planet.get(target_id, []) if item[0] <= cutoff
    ]
    arrivals.extend(
        item for item in planned_commitments.get(target_id, []) if item[0] <= cutoff
    )
    arrivals.extend(item for item in extra_arrivals if item[0] <= cutoff)

    target = self.planet_by_id[target_id]
    dyn = simulate_planet_timeline(target, arrivals, self.player, cutoff)
    return state_at_timeline(dyn, cutoff)


def min_ships_to_own_by(
    self,
    target_id,
    eval_turn,
    attacker_owner,
    arrival_turn=None,
    planned_commitments=None,
    extra_arrivals=(),
    upper_bound=None,
):
    planned_commitments = planned_commitments or {}
    eval_turn = max(1, int(math.ceil(eval_turn)))
    arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
    if arrival_turn > eval_turn:
        if upper_bound is not None:
            return max(1, int(upper_bound)) + 1
        return self._ownership_search_cap(eval_turn) + 1

    normalized_extra = tuple(
        (
            max(1, int(math.ceil(turns))),
            owner,
            int(ships),
        )
        for turns, owner, ships in extra_arrivals
        if ships > 0 and max(1, int(math.ceil(turns))) <= eval_turn
    )

    owner_before, ships_before = self.projected_state(
        target_id,
        eval_turn,
        planned_commitments=planned_commitments,
        extra_arrivals=normalized_extra,
    )
    if owner_before == attacker_owner:
        return 0

    def owns_at(ships):
        owner_after, _ = self.projected_state(
            target_id,
            eval_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=normalized_extra + ((arrival_turn, attacker_owner, int(ships)),),
        )
        return owner_after == attacker_owner

    if upper_bound is not None:
        hi = max(1, int(upper_bound))
        if not owns_at(hi):
            return hi + 1
    else:
        hi = max(1, int(math.ceil(ships_before)) + 1)
        search_cap = self._ownership_search_cap(eval_turn)
        while hi <= search_cap and not owns_at(hi):
            hi *= 2
        if hi > search_cap:
            hi = search_cap
            if not owns_at(hi):
                return hi + 1

    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if owns_at(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def reinforcement_needed_to_hold_until(
    self,
    planet_id,
    arrival_turn,
    hold_until,
    planned_commitments=None,
    upper_bound=None,
):
    planned_commitments = planned_commitments or {}
    arrival_turn = max(1, int(math.ceil(arrival_turn)))
    hold_until = max(arrival_turn, int(math.ceil(hold_until)))

    def holds_with_reinforcement(ships):
        timeline = self.projected_timeline(
            planet_id,
            hold_until,
            planned_commitments=planned_commitments,
            extra_arrivals=((arrival_turn, self.player, int(ships)),),
        )
        for turn in range(arrival_turn, hold_until + 1):
            if timeline["owner_at"].get(turn) != self.player:
                return False
        return True

    hi = max(1, int(upper_bound)) if upper_bound is not None else 1
    search_cap = self._ownership_search_cap(hold_until)
    while hi <= search_cap and not holds_with_reinforcement(hi):
        hi *= 2

    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if holds_with_reinforcement(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

</details>

## 🧭 Layer 3: Strategy As Mission Families

The strategy layer converts legal, forecasted facts into mission choices.
It could have been one large score formula.
The notebook instead keeps mission families separate because each mission has a different validity condition.

At a high level, each candidate mission can be represented as:

$$
m = (kind,\ target,\ sources,\ ships,\ T)
$$

The mission is considered only if its route and ownership constraints are satisfied:

$$
m \in \mathcal{M}_{valid}
\quad \Longleftrightarrow \quad
\operatorname{Legal}(m)
\land
\operatorname{TemporalCondition}(m)
\land
\operatorname{OwnershipCondition}(m)
$$

After validity filtering, missions are ranked by a value-minus-cost view:

$$
Score(m)
=
V(target, kind, T)
- C(ships, T)
- Risk(m)
$$

where:

| Term   | Meaning                                                                                  |
| ------ | ---------------------------------------------------------------------------------------- |
| `V`    | rewards production, strategic location, hostile pressure, and late-game swing            |
| `C`    | penalizes ship cost and travel time                                                      |
| `Risk` | accounts for contested timing, enemy reaction windows, and overcommitting source planets |

The important point is that scoring happens after feasibility and ownership checks.
A high-value mission that cannot legally arrive or cannot own at arrival time is not a candidate mission.

<details markdown="1">
<summary>Show mission data structures and final-send settling</summary>

```python
@dataclass(frozen=True)
class ShotOption:
    score: float
    src_id: int
    target_id: int
    angle: float
    turns: int
    needed: int
    send_cap: int
    mission: str = "capture"
    anchor_turn: int | None = None


@dataclass
class Mission:
    kind: str
    score: float
    target_id: int
    turns: int
    options: list[ShotOption] = field(default_factory=list)


def settle_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    modes,
    policy,
    mission="capture",
    eval_turn_fn=None,
    anchor_turn=None,
    anchor_tolerance=None,
    max_iter=4,
):
    seed_hint = max(1, min(src_cap, int(send_guess)))
    eval_turn_fn = eval_turn_fn or (lambda turns: turns)
    tested = {}

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        if send in tested:
            return tested[send]

        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None

        angle, turns, _, _ = aim
        eval_turn = int(math.ceil(eval_turn_fn(turns)))
        if eval_turn < turns:
            tested[send] = None
            return None

        need = world.min_ships_to_own_by(
            target.id,
            eval_turn,
            world.player,
            arrival_turn=turns,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None

        desired = min(
            src_cap,
            max(need, preferred_send(target, need, turns, src_cap, world, modes, policy)),
        )
        tested[send] = (angle, turns, eval_turn, need, send, desired)
        return tested[send]

    current_send = seed_hint
    for _ in range(max_iter):
        result = evaluate(current_send)
        if result is None:
            return None

        angle, turns, eval_turn, need, actual_send, desired = result
        if desired == actual_send:
            return angle, turns, eval_turn, need, actual_send

        current_send = max(1, min(src_cap, int(desired)))

    result = evaluate(current_send)
    if result is None:
        return None
    angle, turns, eval_turn, need, actual_send, _ = result
    if actual_send < need:
        return None
    return angle, turns, eval_turn, need, actual_send
```

</details>

That separation matters most on defense.
"Send ships to a controlled planet" is not one idea:

| Defense Mission     | Timing Meaning                                                          |
| ------------------- | ----------------------------------------------------------------------- |
| 🛡️ Reinforce-to-hold | the planet is controlled and should remain controlled through a horizon |
| 🛟 Rescue            | the planet can still be saved before the fall turn                      |
| 🔁 Recapture         | rescue is too late, but fast recovery is still valuable                 |
| 📦 Salvage           | the source is doomed, so useful ships should move elsewhere             |

Those are different missions with different timing semantics.
Collapsing them into one generic defense rule would make the agent look simpler while making it less correct.

Offense has the same pattern:

| Offensive Mission       | Core Idea                                              |
| ----------------------- | ------------------------------------------------------ |
| 🏴 Single-source capture | one source can own the target at ETA                   |
| ⚔️ Hostile pressure      | attack enemy production or ship mass                   |
| 🎯 Neutral snipe         | arrive around another player's timing window           |
| 🤝 Swarm                 | combine partial sources into a synchronized conversion |
| 💥 Crash exploit         | exploit enemy-vs-enemy collision timing                |
| ⏱️ Late follow-up        | prefer immediate ship swing when time is short         |

The result is not that every rule is perfect.
The result is that each tactical idea has a name and a contract.
The strategy layer therefore ranks only missions that have already passed physics and world-model feasibility checks.

The reserve rule gives the strategy layer a simple budget constraint.
For each source planet `s`, offensive spend is bounded by:

$$
B_s = \max(0,\ ships_s - R_s)
$$

where the reserve is the maximum of forecasted hold need and proactive danger:

$$
R_s = \max\left(
R^{forecast}_s,\ R^{reaction}_s
\right)
$$

This budget equation is why the strategy layer queries the world model before attacking.
Expansion and pressure should not consume ships that are already needed to keep existing production alive.

## 🔁 The Commit Loop Is The Real Glue

The most important implementation detail is the commitment loop.

Candidate moves are not independent.
If source A sends 40 ships, then source A no longer has those 40 ships.
If that fleet arrives on turn 17, then every later forecast for that target must include the planned arrival.

The notebook treats accepted launches as future facts:

| Step | Operation                      | Why It Matters                        |
| ---- | ------------------------------ | ------------------------------------- |
| 1    | 🎯 choose a mission             | starts from a valid candidate         |
| 2    | 🧭 re-aim the final send amount | ship count changes speed and ETA      |
| 3    | 🧮 reduce source inventory      | prevents overcommitting a planet      |
| 4    | 🧾 append the planned arrival   | future forecasts include the launch   |
| 5    | 🔍 re-evaluate later missions   | later decisions see the updated world |

In notation, the arrival ledger for planet `i` starts as:

$$
\mathcal{A}_i = \{(T, \operatorname{owner}, \operatorname{ships})\}
$$

After the agent accepts a launch, it creates a planned commitment:

$$
C_i \leftarrow C_i \cup \{(T_{\text{new}}, player, n_{\text{new}})\}
$$

Every later world query reads:

$$
\mathcal{A}_i \cup C_i
$$

instead of the original visible arrivals alone.
Without this update, mission candidates are evaluated against different future states.
With this update, all later ownership queries condition on the same arrival ledger:

$$
S_i(t \mid \mathcal{A}_i \cup C_i)
$$

Commitment-aware planning gives the policy temporal consistency.
It also preserves split launches when they are intentional.
Since ship count changes speed, two smaller synchronized launches can have a different tactical meaning from one larger launch.

<details markdown="1">
<summary>Show commitment-aware mission acceptance</summary>

```python
missions.sort(key=lambda item: -item.score)

for mission in missions:
    target = world.planet_by_id[mission.target_id]

    if mission.kind in ("single", "snipe", "rescue", "recapture", "reinforce", "crash_exploit"):
        option = mission.options[0]
        src = world.planet_by_id[option.src_id]
        left = source_attack_left(option.src_id)
        if left <= 0:
            continue

        plan = settle_plan(
            src,
            target,
            left,
            min(left, option.send_cap),
            world,
            planned_commitments,
            modes,
            policy,
            mission=option.mission,
        )
        if plan is None:
            continue

        angle, turns, _, need, send = plan
        if send < need or need > left:
            continue

        sent = append_move(option.src_id, angle, send)
        if sent < need:
            continue

        planned_commitments[target.id].append((turns, world.player, int(sent)))
        continue

    limits = []
    for option in mission.options:
        left = source_attack_left(option.src_id)
        limits.append(min(left, option.send_cap))
    if min(limits) <= 0:
        continue

    missing = world.min_ships_to_own_at(
        target.id,
        mission.turns,
        world.player,
        planned_commitments=planned_commitments,
        upper_bound=sum(limits),
    )
    if missing <= 0 or sum(limits) < missing:
        continue

    ordered = sorted(
        zip(mission.options, limits),
        key=lambda item: (item[0].turns, -item[1], item[0].src_id),
    )
    remaining = missing
    sends = {}
    for idx, (option, limit) in enumerate(ordered):
        remaining_other = sum(other_limit for _, other_limit in ordered[idx + 1 :])
        send = min(limit, max(0, remaining - remaining_other))
        sends[option.src_id] = send
        remaining -= send
    if remaining > 0:
        continue

    reaimed = []
    for option, _ in ordered:
        send = sends.get(option.src_id, 0)
        if send <= 0:
            continue
        src = world.planet_by_id[option.src_id]
        fixed_aim = world.plan_shot(src.id, target.id, send)
        if fixed_aim is None:
            reaimed = []
            break
        angle, turns, _, _ = fixed_aim
        reaimed.append((option.src_id, angle, turns, send))
    if not reaimed:
        continue

    actual_joint_turn = max(item[2] for item in reaimed)
    owner_after, _ = world.projected_state(
        target.id,
        actual_joint_turn,
        planned_commitments=planned_commitments,
        extra_arrivals=[(turns, world.player, send) for _, _, turns, send in reaimed],
    )
    if owner_after != world.player:
        continue

    committed = []
    for src_id, angle, turns, send in reaimed:
        actual = append_move(src_id, angle, send)
        if actual > 0:
            committed.append((turns, world.player, int(actual)))
    if sum(item[2] for item in committed) >= missing:
        planned_commitments[target.id].extend(committed)
```

</details>

## ♻️ Consistency Checks

The policy relies on the following consistency contracts:

| Contract     | Definition                                                           |
| ------------ | -------------------------------------------------------------------- |
| 🧭 Route      | direct launch, sun-safe segment, predicted ETA                       |
| 🌍 State      | projected owner and garrison at a specified turn                     |
| 🏴 Ownership  | minimum ships needed to own at or by a specified turn                |
| 🛡️ Hold       | minimum reinforcement needed to preserve ownership through a horizon |
| 🔁 Commitment | accepted launches become future arrivals for later decisions         |

These contracts define which facts the strategy layer is allowed to assume.

The verification checks follow the same decomposition:

| Check                            | Contract Tested                          |
| -------------------------------- | ---------------------------------------- |
| ☀️ sun-safe routing               | 🧭 route legality                         |
| 🔄 moving-target prediction       | ⏱️ ETA-aware target position              |
| ⚔️ same-turn combat               | 🏴 owner-aware arrival resolution         |
| 🛡️ hold semantics                 | 🛟 reinforce/rescue/recapture distinction |
| 🤝 swarm timing                   | 🧩 synchronized multi-source ownership    |
| 💥 crash-window timing            | 🎯 post-collision opportunity             |
| 📦 live salvage after commitments | 🔁 future-state consistency               |

The checks test whether the mathematical contracts used by the strategy layer remain consistent with the simulator-facing action model.

## 🔬 Possible Model Extensions

The same decomposition suggests several model extensions:

| Extension                              | What It Would Improve                                      |
| -------------------------------------- | ---------------------------------------------------------- |
| 👥 Multi-agent pressure model           | stronger 4-player survival                                 |
| 📊 Empirical mission-weight calibration | less hand-tuned scoring                                    |
| 🎯 Opponent-specific reaction estimates | better contested-target evaluation                         |
| 🔍 Local search over source allocation  | stronger coordinated attacks                               |
| ⏱️ Late-game value function             | better elimination, blocking, and immediate ship swing     |
| 🧪 Layered benchmark protocol           | clearer separation of route, forecast, and strategy errors |

Each extension preserves the same modeling order:

$$
\text{physics} \rightarrow \text{world model} \rightarrow \text{strategy}
$$

Strategy may query the world model, and the world model may query physics, but physics should not depend on strategic preference parameters.

## ✅ Final Invariants

The final policy should preserve the following invariants:

| Invariant                | Formal Requirement                                                      |
| ------------------------ | ----------------------------------------------------------------------- |
| ☀️ Route legality         | every launch satisfies `Legal(source, target, ships)`                   |
| ⏱️ ETA consistency        | target state is queried at the same `T` implied by the final ship count |
| 🏴 Ownership validity     | offensive missions satisfy `N_i(T, player) <= ships_sent`               |
| 🛡️ Hold validity          | defensive missions satisfy `H_i(tau, L) <= reinforcement_sent`          |
| 📦 Budget validity        | each source sends at most its available budget `B_s`                    |
| 🔁 Commitment consistency | later forecasts include all accepted planned arrivals                   |

In compact form, an accepted launch must satisfy:

$$
\operatorname{Legal}(s,i,n)
\land
\operatorname{BudgetSafe}(s,n)
\land
\left[
N_i(T,p) \le n
\ \lor\
H_i(\tau,L) \le n
\right]
$$

The strategy layer is only allowed to optimize among actions that satisfy these constraints.

<details markdown="1">
<summary>Show the environment-facing wrapper</summary>

```python
def build_world(obs):
    player = _read(obs, "player", 0)
    step = _read(obs, "step", 0) or 0
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0
    raw_init = _read(obs, "initial_planets", []) or []
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*planet) for planet in raw_planets]
    fleets = [Fleet(*fleet) for fleet in raw_fleets]
    initial_planets = [Planet(*planet) for planet in raw_init]
    initial_by_id = {planet.id: planet for planet in initial_planets}

    return WorldModel(
        player=player,
        step=step,
        planets=planets,
        fleets=fleets,
        initial_by_id=initial_by_id,
        ang_vel=ang_vel,
        comets=comets,
        comet_ids=comet_ids,
    )


def agent(obs, config=None):
    start_time = time.perf_counter()
    world = build_world(obs)
    if not world.my_planets:
        return []
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    soft_budget = min(SOFT_ACT_DEADLINE, max(0.55, act_timeout * 0.82))
    deadline = start_time + soft_budget
    return plan_moves(world, deadline=deadline)
```

</details>
