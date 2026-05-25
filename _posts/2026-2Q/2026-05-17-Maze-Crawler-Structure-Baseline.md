---
title: "Maze Crawler: Structure Baseline"
date: 2026-05-17 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, maze-crawler, game-ai, rule-based-agent, pathfinding, simulation]
math: true
pin: false
---

# Maze Crawler: Structure Baseline

Competition link:  
[Maze Crawler](https://www.kaggle.com/competitions/maze-crawler)

Kaggle notebook link:  
[Maze Crawler: Structure Baseline](https://www.kaggle.com/code/pilkwang/maze-crawler-structure-baseline)

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/cover.png" alt="Maze Crawler structure baseline cover" width="90%">
</p>

Maze Crawler looks like a grid game, but the useful mental model is closer to a small control system under partial observation.
The agent is not trying to predict a label.
It is trying to keep a convoy alive while a scrolling loss boundary rises from the south, resources appear under limited vision, and every robot action can interfere with every other robot.

The baseline in this notebook is built around one rule:

$$
\textbf{survive first, convert energy second, avoid self-inflicted losses always.}
$$

That rule explains the structure of the agent.
The implementation is not organized around isolated heuristics such as "move north" or "collect crystal."
It is organized as a layered planner:

| Layer | Responsibility |
|---|---|
| **Mechanics / physics** | Encode movement, walls, cooldowns, upkeep, scroll timing, and collision risk. |
| **World model** | Remember walls, mines, nodes, robot history, visible resources, and fogged facts. |
| **Pathing** | Convert targets into reachable first steps using bounded BFS over trusted map facts. |
| **Strategy** | Choose factory tempo, worker corridor support, scout ferry behavior, and optional mine hooks. |
| **Safety planner** | Normalize actions, reserve cells, handle forced vacates, and block illegal or self-destructive actions. |
| **Entry point** | Return the Kaggle action dictionary through `agent(obs, config)`. |

The important design choice is dependency direction:

```text
rules -> memory -> route candidates -> strategic intent -> safety shield -> action dict
```

The strategy can be tuned.
The mechanics should stay boring, deterministic, and rule-correct.

## **1. The Game In One Page**

Maze Crawler is easiest to read as a **scrolling survival race with an economy tiebreaker**.
The factory must keep moving north while the southern boundary rises.
Energy matters, but only if it is converted back into usable robot energy before the units carrying or storing it are lost.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-01-map-memory.png" alt="Maze Crawler unit roles: factory, scout, worker, and miner" width="88%">
</p>

| Unit | Plain Role | What The Baseline Must Respect |
|---|---|---|
| **Factory** | King | Losing it ends the episode, and every build competes with northward tempo. |
| **Worker** | Engineer | Opens walls and keeps the factory corridor alive. |
| **Scout** | Eyes and courier | Finds crystals quickly, but carried energy must be banked safely. |
| **Miner** | Investment | Can create high value, but mine energy must be harvested before it counts. |

The two main loops are:

| Loop | Main Question | Typical Failure |
|---|---|---|
| **Survival loop** | Can the factory keep enough margin above the scroll boundary? | Too many build, side-step, or idle turns. |
| **Energy loop** | Can collected energy become robot energy before it is lost? | Scout dies full, mine stores stranded energy, transfer creates blockers. |

The factory margin is the first state variable I want to see in a replay:

$$
m_t = \operatorname{row}(\text{factory}_t) - \operatorname{southBound}_t
$$

When \(m_t\) becomes small, almost every other ambition should be demoted.
The agent can still collect value, but only through actions that do not weaken the route floor.

## **2. Why A Greedy Agent Fails**

A greedy baseline is tempting:

```text
move north, collect nearby crystals, build whenever affordable
```

That is enough to produce legal-looking actions, but it breaks as soon as rule side effects accumulate.
The issue is not that greedy choices are always bad.
The issue is that greedy choices rarely price the tempo, liquidity, and collision consequences of the action.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-02-greedy-vs-survival.png" alt="Greedy play versus survival-first play in Maze Crawler" width="88%">
</p>

| Greedy Habit | Hidden Cost | Baseline Response |
|---|---|---|
| Build whenever affordable | Factory loses a north move. | Use move-first gates and margin bands. |
| Chase every crystal | Workers and scouts abandon survival jobs. | Score targets by role and pressure state. |
| Transfer whenever adjacent | Source can become a zero-energy blocker. | Use transfer ledgers and corridor guards. |
| Transform every node | Mine energy may never return to robots. | Price lifetime, carrier ETA, and recovery. |
| Trust only current vision | Fog hides remembered walls, nodes, and mines. | Keep persistent world memory. |
| Side-step every wall | Factory survives locally but loses tempo. | Prefer route search and careful jump use. |

The operational principle is:

$$
\textbf{take useful actions only after pricing their rule side effects.}
$$

That is why the baseline contains more validation than a starter bot.
Most game losses in this kind of environment are not caused by failing to see a fancy tactic.
They are caused by illegal moves, stale memory, friendly blockers, or spending the factory turn on something that should have waited.

## **3. Module Map**

The notebook writes a standalone `main.py` agent.
The code is long, but the architectural map is compact:

| Module | Core Question | Why It Exists |
|---|---|---|
| **Shared setup** | What vocabulary does every layer use? | Unit types, wall bits, directions, action names, profile knobs, and dataclasses. |
| **Mechanics / physics** | What does the game engine allow? | Movement, walls, cooldowns, scroll timing, combat danger, and upkeep affordability. |
| **World model** | What do we know after fog hides cells? | Durable map memory, resources, mines, visible robots, and assignments. |
| **Pathing and targets** | Which useful goals are reachable? | Bounded BFS, factory route probes, frontier discovery, crystals, unload cells, and wall jobs. |
| **Strategy and safety** | Which legal action should each robot take? | Factory floor selection, worker vanguard, scout ferry, mine hooks, reservations, and normalization. |
| **Entry point** | How does Kaggle call the agent? | `compute_actions(...)`, `agent(obs, config)`, and per-player state storage. |

The code intentionally separates **what is true about the board** from **what the current policy prefers**.
This makes tuning less dangerous.
Changing scout thresholds should not accidentally change wall legality.
Changing worker behavior should not rewrite scroll inference.

<details markdown="1">
<summary>Show notebook snippet: shared dataclasses</summary>

```python
@dataclass(frozen=True)
class Robot:
    uid: str
    rtype: int
    col: int
    row: int
    energy: int
    owner: int
    move_cd: int = 0
    jump_cd: int = 0
    build_cd: int = 0

    @property
    def pos(self):
        return (self.col, self.row)


@dataclass
class Target:
    """High-level intent candidate before primitive action commitment."""

    kind: str
    pos: tuple[int, int]
    score: float


@dataclass(frozen=True)
class RouteQuality:
    """Factory route summary used to price tempo before building units."""

    action: str | None
    next_pos: tuple[int, int] | None
    exists: bool
    uses_jump: bool
    north_gain: int
    margin_surplus: int
    blocked: bool


@dataclass(frozen=True)
class SafetyDecision:
    """Pure factory-destination evaluation result."""

    ok: bool
    forced_actions: dict
    forced_reserved_next: dict
    reserved_next: set
    sacrifice_uids: set
    reason: str = ""
```

</details>

## **4. Mechanics / Physics**

In this notebook, "physics" means the deterministic rules that decide whether an action is possible or dangerous.
For Maze Crawler, the crucial mechanics are:

| Mechanic | Consequence |
|---|---|
| **Scrolling boundary** | A locally useful action can still be fatal if it spends factory tempo. |
| **Walls as edge facts** | Movement legality depends on the wall bit of the current cell and sometimes the opposite bit of the neighbor. |
| **Cooldowns** | A good route action is useless if the robot cannot move yet. |
| **Upkeep before action** | A robot that cannot pay upkeep should not propose a non-idle action. |
| **Combat rank** | Ending on an enemy cell can destroy the robot. |
| **Jump cooldown** | Jump can save tempo, but spending it casually can remove the escape option later. |

The danger band is dynamic.
As the scroll speed ramps, a fixed "safe margin" becomes too optimistic.
The baseline therefore expands the caution gap with step progress:

$$
d_t =
d_0
+
\left\lfloor
b \cdot
\min\left(1,\frac{t}{T_{\text{ramp}}}\right)
\right\rfloor
$$

where \(d_0\) is the base danger gap, \(b\) is the ramp bonus, and \(T_{\text{ramp}}\) is the scroll ramp duration.

<details markdown="1">
<summary>Show notebook snippet: scroll and wall mechanics</summary>

```python
def action_ready(cd):
    # Cooldowns tick before action validation.
    return int(cd or 0) <= 1


def dynamic_danger_gap_for_step(step, config):
    ramp = int(cfg(config, "scrollRampSteps", 400))
    progress = min(1.0, max(0.0, float(step) / max(1, ramp)))
    return SCROLL_DANGER_GAP + int(SCROLL_DANGER_RAMP_BONUS * progress)


def step_pos(pos, direction, distance=1):
    dc, dr, _ = DIRS[direction]
    return pos[0] + dc * distance, pos[1] + dr * distance


def known_edge_blocked(world, pos, direction):
    """Return True if either known side of an edge has the wall bit set."""
    if direction not in DIRS:
        return False
    _, _, bit = DIRS[direction]
    wall = world.wall_at(pos)
    if wall is not None and (wall & bit):
        return True
    nxt = step_pos(pos, direction)
    if not world.in_bounds(nxt):
        return False
    opposite_wall = world.wall_at(nxt)
    if opposite_wall is None:
        return False
    opposite_bit = DIRS[OPPOSITE[direction]][2]
    return bool(opposite_wall & opposite_bit)
```

</details>

The key is that mechanics do not rank strategic options.
They simply answer questions such as:

```text
Can this robot move?
Can it pay after upkeep?
Is the edge blocked?
Is the destination occupied?
Would the robot be crushed?
```

That boundary keeps the strategy layer honest.

## **5. World Model**

The raw observation is only the current visible slice of the world.
The agent needs a memory layer because pathing and economy decisions depend on facts that may disappear into fog.

The world model stores:

| Memory | What It Does |
|---|---|
| **Wall memory** | Keeps seen wall bits and synchronizes reciprocal edge facts. |
| **Node memory** | Remembers discovered mineable nodes. |
| **Mine memory** | Keeps remembered mines and estimates hidden generation. |
| **Visible state** | Tracks current crystals, enemies, friendly robots, and visible cells. |
| **Assignment memory** | Reduces repeated target switching. |
| **Robot history** | Helps identify movement patterns and stale units. |

This is the central split:

```text
WorldModel asks: what do we know?
Strategy asks: what should we do with it?
```

The world model also uses symmetry.
When a wall is observed on one side of the map, the mirrored wall can be inserted as a default memory fact.
A direct observation later still wins.
That is a useful compromise: the agent benefits from map structure without pretending that inferred facts are as strong as fresh vision.

<details markdown="1">
<summary>Show notebook snippet: memory update and reciprocal walls</summary>

```python
class WorldModel:
    """Observation normalizer and durable memory."""

    def __init__(self, obs, config, state_store=None):
        self.obs = to_plain_dict(obs)
        self.config = to_plain_dict(config)
        self.state_store = STATE_BY_PLAYER if state_store is None else state_store
        self.player = int(self.obs.get("player", 0))
        self.width = int(cfg(self.config, "width", 20))
        self.height = int(cfg(self.config, "height", 20))
        self.step = self._infer_step()
        self.south = self._infer_south_bound()
        self.north = self._infer_north_bound()
        self.state = self._state_for_player()

        self.own = {}
        self.enemies = {}
        self.own_positions = {}
        self.enemy_positions = {}
        self.visible_crystals = {}
        self.visible_nodes = set()
        self.visible_cells = set()
        self.factory = None

        self.update_memory()

    def update_memory(self):
        self._read_robots()
        self._read_visible_cells()
        self._read_walls()
        self._read_resources()
        self._prune_memory()

    def _read_walls(self):
        walls = self.obs.get("walls") or []
        observed = {}
        for idx, wall in enumerate(walls):
            wall = int(wall)
            if wall < 0:
                continue
            pos = (idx % self.width, self.south + idx // self.width)
            if self.in_bounds(pos):
                observed[pos] = wall
                self.state["known_walls"][pos] = wall

        # A wall is an edge fact. If one side is observed and the opposite
        # cell is remembered, synchronize the reciprocal bit.
        for pos, wall in observed.items():
            for direction, (_, _, bit) in DIRS.items():
                nxt = step_pos(pos, direction)
                if not self.in_bounds(nxt) or nxt in observed:
                    continue
                if nxt not in self.state["known_walls"]:
                    continue
                opposite_bit = DIRS[OPPOSITE[direction]][2]
                neighbor_wall = int(self.state["known_walls"][nxt])
                if wall & bit:
                    neighbor_wall |= opposite_bit
                else:
                    neighbor_wall &= ~opposite_bit
                self.state["known_walls"][nxt] = neighbor_wall
```

</details>

The mine logic is particularly important.
A mine outside vision may still be generating energy, but its `last_seen` should not be refreshed unless the cell is actually visible.
Otherwise the planner starts treating old mine estimates as fresh evidence.

## **6. Pathing And Candidate Targets**

Pathing converts high-level intent into a first primitive step.
The baseline uses bounded BFS because the action timeout matters:

```text
if a target is too expensive to search,
degrade to local safe motion instead of missing the turn deadline
```

The pathing layer produces:

| Pathing Block | Purpose |
|---|---|
| **BFS first step** | Find a legal first direction toward a target. |
| **Factory route probe** | Test whether the factory has a north-gaining route before reserving cells. |
| **Jump-aware route** | Ask whether a jump can preserve tempo when walls block ordinary movement. |
| **Frontier detection** | Find remembered cells adjacent to unknown space. |
| **Target enumeration** | Generate crystals, unload cells, wall jobs, node jobs, mine harvests, and vanguard positions. |
| **Sticky ordering** | Avoid target churn when multiple options are close. |

The pathing function avoids friendly-occupied cells by default.
There is one important exception: a miner that will `TRANSFORM` leaves its current cell before movement, so that cell can be a valid same-turn destination for a carrier.

<details markdown="1">
<summary>Show notebook snippet: bounded BFS</summary>

```python
def bfs_first_step_and_distance(
    world,
    robot,
    goals,
    reserved_next,
    max_nodes=MAX_BFS_NODES,
    allow_occupied_goal=False,
):
    goals = set(goals)
    if not goals:
        return None, None, None
    start = robot.pos
    if start in goals:
        return "IDLE", start, 0

    queue = deque([start])
    first_dir = {start: None}
    distance = {start: 0}
    seen = {start}
    nodes = 0

    while queue and nodes < max_nodes:
        cur = queue.popleft()
        nodes += 1
        for direction, nxt in world.neighbors(cur, allow_unknown_target=False):
            if nxt in seen:
                continue
            if nxt != start and nxt in reserved_next:
                continue
            if allow_occupied_goal and nxt in goals:
                if crush_danger(robot, world.enemy_at(nxt)):
                    continue
                return direction if cur == start else first_dir[cur], nxt, distance[cur] + 1
            if nxt != start and nxt in world.own_positions:
                continue
            if crush_danger(robot, world.enemy_at(nxt)):
                continue
            seen.add(nxt)
            first_dir[nxt] = direction if cur == start else first_dir[cur]
            distance[nxt] = distance[cur] + 1
            if nxt in goals:
                return first_dir[nxt], nxt, distance[nxt]
            queue.append(nxt)
    return None, None, None
```

</details>

This is a good example of the baseline's style.
Pathing does not decide that a crystal is worth chasing.
It only reports whether a route exists and what the first step would be.
Target scoring lives one layer above.

## **7. Resource Conversion: Scout, Transfer, Miner**

Energy is not a score until it becomes useful.
This is why the baseline treats crystal collection, transfer, and mine transform as separate contracts.

### Scout Ferry

Scouts are useful because they expand vision and collect crystals quickly.
But a scout full of energy far from the factory is a liability.
The policy therefore asks whether the scout still has a good forward job.
If not, high carried energy triggers unload behavior.

```text
if scout energy is high and forward value is weak:
    return toward factory unload cells
```

The threshold is deliberately explicit:

| Knob | Meaning |
|---|---|
| `SCOUT_RETURN_ENERGY` | Soft return threshold. |
| `SCOUT_FORCE_RETURN_ENERGY` | Hard return threshold. |
| `UNLOAD_FRACTION` | Fraction of capacity that indicates banking pressure. |
| `ENABLE_MARGIN_SURPLUS_SCOUT` | Allows a scout only when route margin is healthy. |

### Transfer

Transfer looks harmless because no unit moves.
But transfer changes two things at once:

1. where the energy is,
2. whether the source robot becomes an empty blocker.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-03-transfer.png" alt="Unsafe transfer versus safer transfer in Maze Crawler" width="88%">
</p>

A safe transfer must therefore consider liquidity and geometry.
Funding the next factory build can be worth it.
Creating a zero-energy robot directly in the factory corridor is usually not.

The baseline also avoids over-liquidation.
If the factory already committed a build this turn, a later transfer cannot fund that build, and the next build will be delayed by cooldown anyway.
So the transfer check looks at planned actions, not only current energy.

### Miner Transform

A miner is an investment.
Transforming a node into a mine is not the reward.
The reward appears only if mine energy returns to the robot economy before the scroll, distance, or carrier constraints make it useless.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-04-transform.png" alt="Good transform versus risky transform in Maze Crawler" width="88%">
</p>

The active profile keeps `MINER_TARGET_COUNT = 0`, so mine hooks are present but conservative.
That is intentional.
Before adding a stronger mining economy, the survival and route-floor behavior should be measured cleanly.

## **8. Factory Tempo**

The factory is the main budget.
Every non-movement factory action has an opportunity cost because the southern boundary keeps advancing.

This baseline builds two factory candidates:

1. a **floor candidate** that preserves survival,
2. a **complete candidate** that may spend surplus on support or scouting.

The selector is:

$$
a_F =
\begin{cases}
a_{\text{floor}}, & \text{route blocked or margin pressure is high}\\
a_{\text{complete}}, & \text{route healthy and margin surplus available}
\end{cases}
$$

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-05-opening-choice.png" alt="Scout-first opening versus worker-first opening in Maze Crawler" width="88%">
</p>

There is no universal first build.
An open corridor can support scout-first search.
A blocked or low-branching start may need worker-first wall control.
The baseline tries to make that decision from route quality rather than from a fixed opening script.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-06-jump-tradeoff.png" alt="Side-step versus jump north tradeoff in Maze Crawler" width="88%">
</p>

Jump is the same kind of tradeoff.
It can recover north tempo, but it spends cooldown.
The baseline therefore uses jump more freely under emergency pressure and more conservatively when a side route or worker wall job can preserve the long game.

<details markdown="1">
<summary>Show notebook snippet: factory floor versus complete candidate</summary>

```python
def factory_policy(self, robot):
    wall = self.world.current_wall(robot)
    gap = robot.row - self.world.south
    danger = self.dynamic_danger_gap()
    emergency = gap <= danger

    route_quality = self.factory_route_quality(
        robot,
        emergency=emergency,
        apply=False,
    )
    floor = self.factory_floor_candidate(robot, wall, route_quality, emergency)
    if self.factory_floor_pressure(robot, route_quality):
        return floor

    complete = self.factory_complete_candidate(robot, wall, route_quality, emergency)
    if complete is not None:
        return complete
    return floor


def factory_floor_pressure(self, robot, route_quality):
    return (
        route_quality.blocked
        or route_quality.margin_surplus <= FACTORY_FLOOR_PRESSURE_SURPLUS
        or self.front_worker_is_doing_wall_job(robot)
        or (
            route_quality.uses_jump
            and route_quality.margin_surplus <= FACTORY_FLOOR_PRESSURE_SURPLUS + 3
        )
    )
```

</details>

This is the core of the profile named:

```python
PROFILE = "baseline_route_floor_hybrid"
```

The profile does not try to maximize short-term energy.
It tries to keep a reliable route floor, then uses surplus margin to add economy.

## **9. Strategy And Safety Planner**

The strategy layer decides intent.
The safety planner decides whether the intent can actually be committed.

This distinction matters because many failures occur after a good high-level idea has been translated into a bad primitive action.
For example:

| Good Intent | Bad Primitive Failure |
|---|---|
| Move factory north | Friendly worker blocks the destination. |
| Bank scout energy | Scout transfer leaves an empty body in the corridor. |
| Open a wall | Worker cannot pay upkeep plus wall cost. |
| Harvest mine | Carrier path collides with another planned unit. |
| Jump out of danger | Jump cooldown is unavailable or destination is unsafe. |

The planner orders robots by urgency.
Factory and robots near the scroll line are planned first because their reservations define constraints for less urgent units.

<details markdown="1">
<summary>Show notebook snippet: planning order, reservations, and normalization</summary>

```python
def plan(self):
    danger = self.dynamic_danger_gap()
    self.planned_special_actions = self.preplan_special_actions()

    robots = sorted(
        self.world.own.values(),
        key=lambda robot: (
            0 if robot.rtype == FACTORY else 1 if robot.row - self.world.south <= danger else 2,
            0 if robot.rtype == WORKER else 1 if robot.rtype == MINER else 2 if robot.rtype == SCOUT else 3,
            robot.row,
            robot.uid,
        ),
    )

    for robot in robots:
        if robot.uid in self.forced_actions:
            action, next_pos = self.forced_actions[robot.uid]
            self.commit(robot, action, next_pos)
            continue
        if robot.uid in self.planned_special_actions:
            action, next_pos = self.planned_special_actions[robot.uid]
            self.commit(robot, action, next_pos)
            continue
        if self.expired():
            self.commit(robot, "IDLE", robot.pos)
            continue
        action, next_pos = self.policy_action(robot)
        self.commit(robot, action, next_pos)

    self.world.state["last_actions"] = dict(self.actions)
    return self.actions


def commit(self, robot, action, next_pos):
    action, next_pos = self.normalize_action(robot, action)
    self.actions[robot.uid] = action

    if action in DIRS or action.startswith("JUMP_"):
        self.reserved_next.add(next_pos)
    elif action in {"BUILD_SCOUT", "BUILD_WORKER", "BUILD_MINER"}:
        self.reserved_next.add(robot.pos)
        self.reserved_next.add(step_pos(robot.pos, "NORTH"))
    elif action == "TRANSFORM":
        # The miner disappears before movement, so the cell is not reserved.
        pass
    else:
        self.reserved_next.add(robot.pos)
```

</details>

The safety layer also normalizes invalid actions to `IDLE`.
That may sound passive, but it is much better than returning an illegal action dictionary and letting the engine decide the failure mode.

## **10. Action Emitter**

The Kaggle-facing entry point is deliberately small.
The main planner can be tested through `compute_actions(...)`, while `agent(obs, config)` simply supplies the shared state store.

<details markdown="1">
<summary>Show notebook snippet: competition entry point</summary>

```python
def compute_actions(obs, config, started, state_store=None, strategy_cls=None):
    deadline = started + SOFT_ACT_DEADLINE
    try:
        world = WorldModel(obs, config, state_store=state_store)
        strategy_type = Strategy if strategy_cls is None else strategy_cls
        strategy = strategy_type(world, started, deadline)
        actions = strategy.plan()
        return {uid: action for uid, action in actions.items() if uid in world.own}
    except Exception:
        try:
            player = int(to_plain_dict(obs).get("player", 0))
            store = STATE_BY_PLAYER if state_store is None else state_store
            store[player] = fresh_state(player)
        except Exception:
            pass
        return {}


def agent(obs, config):
    return compute_actions(obs, config, time.perf_counter())
```

</details>

Two details are worth keeping:

| Detail | Reason |
|---|---|
| `state_store` is injectable | Tests can isolate memory without changing production behavior. |
| `strategy_cls` is injectable | Experiments can swap policy behavior while keeping mechanics and safety fixed. |

That makes the notebook a baseline platform rather than only a single bot.

## **11. Rule-Level Checks**

Single-game score is noisy, so the notebook includes rule-level checks before score interpretation.
These checks are not meant to prove that the strategy is strong.
They protect invariants that are easy to break while tuning.

| Check Family | What It Guards |
|---|---|
| **Factory survival** | Route-first cooldown behavior, floor-candidate pressure behavior, safe jump handling. |
| **Worker engineering** | Correct wall direction, corridor opening, vanguard movement, guarded refuel. |
| **Scout economy** | One-scout target count, active scout replacement, high-energy return behavior. |
| **Mine economy** | Transform timing, hidden mine energy, carrier reachability, same-turn harvest hooks. |
| **Transfer policy** | No careless zero-energy blockers, planned-build ledger respected, route-critical support. |
| **Standalone file** | Compile success and synchronized `main.py` / `submission.py` source. |

A failed rule check should stop score interpretation.
Otherwise it is too easy to mistake a broken mechanic for a weak strategy.

The notebook also copies the generated agent into a standalone submission file:

```python
Path("submission.py").write_text(Path("main.py").read_text())
```

Then it verifies that the standalone file remains importable and synchronized.

## **12. Evaluation Metrics**

For actual strategy comparison, the right unit is a paired seed comparison.
For each seed:

$$
d_i = R_i^{\text{candidate}} - R_i^{\text{baseline}}
$$

Then summarize:

$$
CI_{95}
\approx
\bar d
\pm
1.96 \frac{s_d}{\sqrt{N}}
$$

But the reward delta is only the first line of evidence.
The replay metrics should explain the failure shape.

| Failure Shape | First Metrics To Check | Likely Adjustment |
|---|---|---|
| Factory dies early | `factory_min_margin`, `factory_death_step`, `factory_build_when_route_exists` | Let the floor candidate dominate more often. |
| Route exists but progress is weak | `factory_idle_when_route_exists`, side-step streaks, route cooldown turns | Improve route scoring or worker vanguard support. |
| Scout underperforms | `scout_build_margin_surplus`, `scout_energy_ge_90_turns`, `scout_transfer_count` | Raise scout surplus threshold or disable scout in floor runs. |
| Worker appears too early | `factory_build_count`, first worker timing | Build worker only when long route is blocked or near a dead end. |
| Worker cannot sustain walls | `worker_energy_below_threshold`, failed wall-removal turns, refuel count | Compare route-critical refuel budgets. |
| Jump is missing later | `noncritical_north_wall_jump_count`, `jump_cd_unavailable_in_danger_count` | Keep noncritical jumps conservative. |
| Transfer hurts score | `factory_worker_refuel_count`, estimated overflow, worker wall action after refuel | Tighten refuel purpose gates or overflow budget. |
| Many actions normalize to idle | `normalized_to_IDLE_count` | Inspect legality, cooldowns, reservations, and target conflicts. |

The diagnosis tree is intentionally simple:

```text
Did factory margin collapse?
    yes -> inspect whether floor candidate was overridden
    no  -> did the worker fail to keep the route open?
        yes -> tune route-critical refuel and worker wall targeting
        no  -> did scout energy return safely?
            no  -> tighten scout surplus / return thresholds
            yes -> consider later economy hooks
```

That is the reason this baseline is structured rather than only heuristic.
If the route floor is weak, tune survival.
If the factory survives but the score is low, inspect resource conversion.
If both are fine but the score plateaus, then it is time to add a stronger economy layer.

## **13. What This Baseline Establishes**

The notebook establishes a practical rule-based foundation for Maze Crawler:

| Foundation | What It Enables |
|---|---|
| **Rule-correct mechanics** | Strategy changes do not need to rediscover cooldown, wall, upkeep, and collision rules. |
| **Durable world memory** | The agent can plan from fogged facts without treating stale estimates as fresh vision. |
| **Bounded pathing** | The planner can search useful routes without risking the action timeout. |
| **Factory floor policy** | Survival has a stable fallback before economy is considered. |
| **Safety shield** | Good high-level intents are prevented from becoming illegal primitive actions. |
| **Rule checks** | Experiments can be filtered before noisy score comparisons. |

The result is not a final grandmaster agent.
It is a structured baseline that makes later improvements measurable.

The next natural experiments are:

1. tune the factory floor versus complete candidate selector,
2. compare scout surplus thresholds on paired seeds,
3. activate miner hooks only after route survival is stable,
4. add replay metrics that separate wall-block losses from resource-conversion losses,
5. test alternative worker vanguard policies without changing mechanics.

The central lesson is the same as the opening rule:

> **In Maze Crawler, economy is real only after survival, pathing, and safety have already done their jobs.**
