---
title: "Maze Crawler: Structure Baseline -KR"
date: 2026-05-17 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, maze-crawler, game-ai, rule-based-agent, pathfinding, simulation, korean]
math: true
pin: false
---

# Maze Crawler: Structure Baseline -KR

대회 링크:  
[Maze Crawler](https://www.kaggle.com/competitions/maze-crawler)

Kaggle 노트북 링크:  
[Maze Crawler: Structure Baseline](https://www.kaggle.com/code/pilkwang/maze-crawler-structure-baseline)

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/cover.png" alt="Maze Crawler structure baseline cover" width="90%">
</p>

Maze Crawler는 겉으로 보면 격자 게임입니다. 하지만 에이전트를 설계할 때는 단순한 격자 이동 문제라기보다, **부분 관측 환경에서 작은 로봇 무리를 끝까지 살려 올리는 운영 문제**로 보는 편이 훨씬 정확합니다.

핵심은 매 턴 그럴듯한 한 수를 고르는 것이 아닙니다. 남쪽에서 올라오는 손실 경계선을 피하면서 factory를 북쪽으로 이동시키고, 제한된 시야 안에서 자원을 발견하고, scout/worker/miner/factory의 행동이 서로 방해하지 않도록 관리해야 합니다.

이 베이스라인의 중심 규칙은 다음입니다.

$$
\textbf{먼저 살아남고, 그다음 에너지를 회수하며, 불필요한 실수로 유닛을 잃지 않는다.}
$$

이 규칙이 전체 구조를 설명합니다. 구현은 “북쪽으로 가기”, “crystal 먹기” 같은 개별 휴리스틱을 모아놓은 형태가 아닙니다. 아래처럼 역할을 나누어 계획기를 구성합니다.

| 층 | 책임 |
|---|---|
| 규칙/물리 처리 | 이동, 벽, cooldown, upkeep, scroll timing, 충돌 위험을 규칙으로 고정합니다. |
| 월드 모델 | 벽, 광산, node, robot history, 보이는 자원, fog 속 기억을 유지합니다. |
| 경로 탐색 | 검증된 지도 정보 위에서 제한된 BFS로 목표를 첫 행동으로 바꿉니다. |
| 전략층 | factory tempo, worker corridor support, scout ferry, 선택적 mine hook을 결정합니다. |
| 안전 계획 | 행동을 정규화하고, 다음 칸을 예약하고, 강제 vacate와 불법 행동을 막습니다. |
| 진입점 | `agent(obs, config)`를 통해 Kaggle이 기대하는 action 딕셔너리를 반환합니다. |

의존 방향은 다음과 같습니다.

```text
규칙 -> 기억 -> 경로 후보 -> 전략 의도 -> 안전 장치 -> action dict
```

전략은 바꿔도 됩니다. 하지만 규칙 처리부는 재미없을 정도로 결정론적이어야 하고, 게임 규칙과 정확히 맞아야 합니다. 이 둘을 섞으면 튜닝은 빨라 보이지만, 나중에 어떤 변경이 점수를 바꿨는지 추적하기 어려워집니다.

## **1. 게임을 한 장으로 보기**

Maze Crawler는 **스크롤되는 생존 레이스에 경제 요소가 붙은 게임**으로 보는 것이 좋습니다. Factory는 계속 북쪽으로 올라가야 하고, 남쪽 경계는 계속 따라옵니다. 에너지는 중요하지만, 그 에너지가 사라지기 전에 로봇이 쓸 수 있는 에너지로 돌아와야만 실제 가치가 됩니다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-01-map-memory.png" alt="Maze Crawler unit roles: factory, scout, worker, and miner" width="88%">
</p>

| 유닛 | 쉬운 역할 | 반드시 지켜야 하는 것 |
|---|---|---|
| Factory | 핵심 유닛 | 잃으면 episode가 끝나고, 모든 build는 북쪽 진행 tempo와 경쟁합니다. |
| Worker | 벽 담당 | 벽을 열고 factory corridor를 살립니다. |
| Scout | 정찰과 운반 | crystal을 빠르게 찾지만, 들고 있는 에너지는 안전하게 회수해야 합니다. |
| Miner | 투자 유닛 | 높은 가치를 만들 수 있지만, mine energy가 회수되기 전에는 점수가 아닙니다. |

두 개의 큰 loop가 있습니다.

| Loop | 핵심 질문 | 흔한 실패 |
|---|---|---|
| 생존 루프 | Factory가 scroll boundary 위에서 충분한 margin을 유지할 수 있는가? | build, side-step, idle이 너무 많아 북쪽 진행이 늦어집니다. |
| 에너지 루프 | 모은 에너지가 사라지기 전에 로봇이 쓸 수 있는 에너지로 돌아오는가? | scout가 에너지를 들고 죽거나, mine energy가 회수되지 못합니다. |

가장 먼저 봐야 할 상태값은 factory margin입니다.

$$
m_t = \operatorname{row}(\text{factory}_t) - \operatorname{southBound}_t
$$

\(m_t\)가 작아지면 거의 모든 욕심을 낮춰야 합니다. 여전히 자원은 먹을 수 있지만, factory route floor를 약하게 만드는 행동은 뒤로 밀려야 합니다.

## **2. Greedy 에이전트가 깨지는 이유**

간단한 greedy 베이스라인은 유혹적입니다.

```text
북쪽으로 이동하고, 가까운 crystal을 먹고, 에너지가 되면 build한다.
```

이 정도만으로도 겉보기에는 그럴듯한 행동을 만들 수 있습니다. 하지만 규칙의 부작용이 쌓이는 순간 쉽게 깨집니다. 문제는 greedy 선택이 항상 나쁘다는 것이 아닙니다. 문제는 greedy 선택이 tempo, 에너지 회수 가능성, 충돌 비용을 거의 계산하지 않는다는 점입니다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-02-greedy-vs-survival.png" alt="Greedy play versus survival-first play in Maze Crawler" width="88%">
</p>

| Greedy 습관 | 숨은 비용 | 대응 |
|---|---|---|
| 에너지가 되면 바로 build | Factory가 북쪽으로 갈 한 턴을 잃습니다. | move-first gate와 margin band를 둡니다. |
| 모든 crystal을 추적 | worker/scout가 생존 임무를 떠납니다. | 역할과 pressure state에 따라 target 점수를 다르게 둡니다. |
| 인접하면 바로 transfer | 에너지를 비운 robot이 길을 막을 수 있습니다. | transfer ledger와 corridor guard를 둡니다. |
| 모든 node를 transform | mine energy가 로봇에게 돌아오지 못할 수 있습니다. | lifetime, carrier ETA, recovery 가능성을 함께 봅니다. |
| 현재 시야만 신뢰 | fog 속에 기억된 벽, node, mine을 놓칩니다. | 지속적인 world memory를 유지합니다. |
| 벽을 만나면 무조건 side-step | 당장은 살지만 tempo를 잃습니다. | route search와 jump 사용을 조심스럽게 결합합니다. |

운영 원칙은 다음입니다.

$$
\textbf{규칙의 부작용을 계산한 뒤에만 유용한 행동을 한다.}
$$

그래서 이 베이스라인에는 기본 봇보다 검증 코드가 많이 들어갑니다. 이런 환경에서 많은 손실은 멋진 전술을 못 봐서가 아니라, 불법 이동, 오래된 memory, friendly blocker, factory turn 낭비에서 나옵니다.

## **3. 모듈 지도**

구현은 `main.py` 하나로 제출 가능한 에이전트 형태입니다. 코드는 길지만, 구조는 비교적 단순합니다.

| 모듈 | 핵심 질문 | 존재 이유 |
|---|---|---|
| 공통 정의 | 모든 층이 공유할 기본 어휘는 무엇인가? | unit type, wall bit, direction, action name, profile 파라미터, dataclass를 정리합니다. |
| 규칙/물리 처리 | 게임 엔진은 무엇을 허용하는가? | 이동, 벽, cooldown, scroll timing, combat danger, upkeep affordability를 다룹니다. |
| 월드 모델 | fog가 가린 뒤에도 무엇을 알고 있는가? | map memory, resource, mine, visible robot, assignment를 유지합니다. |
| 경로와 목표 | 어떤 유용한 목표가 도달 가능한가? | 제한된 BFS, factory route probe, frontier, crystal, unload cell, wall job을 만듭니다. |
| 전략과 안전 계획 | 각 robot은 어떤 합법 행동을 해야 하는가? | factory floor, worker vanguard, scout ferry, mine hook, reservation, normalization을 담당합니다. |
| 진입점 | Kaggle은 에이전트를 어떻게 호출하는가? | `compute_actions(...)`, `agent(obs, config)`, player별 state storage를 제공합니다. |

핵심은 **보드에 대해 참인 사실**과 **현재 정책이 선호하는 행동**을 분리하는 것입니다. 이 분리가 있어야 튜닝이 덜 위험해집니다. Scout 기준값을 바꾸는 일이 벽 판정을 건드리면 안 되고, worker 행동을 바꾸는 일이 scroll 추론을 바꾸면 안 됩니다.

<details markdown="1">
<summary>코드: 공유 dataclass</summary>

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

## **4. 게임 규칙: 먼저 가능 여부를 고정합니다**

Maze Crawler에서 먼저 고정해야 할 규칙은 다음입니다.

| 규칙 | 결과 |
|---|---|
| 스크롤 경계 | 당장 좋아 보이는 행동도 factory tempo를 쓰면 치명적일 수 있습니다. |
| 벽은 간선 정보 | 이동 가능성은 현재 cell의 wall bit와 이웃 cell의 반대쪽 bit에 의존합니다. |
| 쿨다운 | 좋은 route action도 robot이 아직 움직일 수 없으면 무의미합니다. |
| 행동 전 upkeep | upkeep을 낼 수 없는 robot은 idle이 아닌 행동을 제안하지 않아야 합니다. |
| 전투 순위 | enemy cell에서 턴을 끝내면 robot이 파괴될 수 있습니다. |
| Jump 쿨다운 | jump는 tempo를 구할 수 있지만, 무심코 쓰면 나중의 탈출 수단이 사라집니다. |

위험 구간은 고정값이 아닙니다. Scroll speed가 빨라질수록 고정된 안전 여유는 너무 낙관적입니다. 그래서 step progress에 따라 경계 폭을 키웁니다.

$$
d_t =
d_0
+
\left\lfloor
b \cdot
\min\left(1,\frac{t}{T_{\text{ramp}}}\right)
\right\rfloor
$$

여기서 \(d_0\)는 기본 danger gap, \(b\)는 ramp bonus, \(T_{\text{ramp}}\)는 scroll ramp가 끝나는 시점입니다.

<details markdown="1">
<summary>코드: scroll과 wall 규칙</summary>

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

규칙 처리부는 전략의 우선순위를 매기지 않습니다. 단지 다음 질문에 답합니다.

```text
이 robot은 움직일 수 있는가?
upkeep을 낸 뒤에도 행동할 수 있는가?
edge가 막혀 있는가?
도착 칸이 이미 예약되어 있는가?
enemy와 충돌해 파괴되는가?
```

이 경계가 있어야 전략층이 정직해집니다.

## **5. 월드 모델**

매 턴 들어오는 관측값에는 지금 보이는 칸의 정보만 들어 있습니다. 하지만 길을 찾거나 자원 회수를 판단하려면, 이전에 봤던 벽과 node, mine 정보도 계속 써야 합니다. 그래서 현재 관측과 과거 기억을 따로 관리합니다.

월드 모델은 다음을 저장합니다.

| 기억 항목 | 역할 |
|---|---|
| 벽 기억 | 본 wall bit를 저장하고 반대편 edge 정보도 함께 맞춥니다. |
| Node 기억 | 발견한 mineable node를 기억합니다. |
| Mine 기억 | 기억된 mine과 hidden generation 추정을 유지합니다. |
| 현재 시야 | 현재 crystal, enemy, friendly robot, visible cell을 추적합니다. |
| 배정 기억 | target switching을 줄입니다. |
| Robot 기록 | movement pattern과 stale unit을 파악하는 데 도움을 줍니다. |

이 분리가 중요합니다.

```text
월드 모델은 묻습니다: 우리는 무엇을 알고 있는가?
전략층은 묻습니다: 그 지식으로 무엇을 할 것인가?
```

월드 모델은 대칭성도 사용합니다. 한쪽에서 wall을 보면 mirror wall을 기본 기억값으로 넣을 수 있습니다. 하지만 나중에 직접 관측한 정보가 항상 우선합니다. 이렇게 하면 map structure를 활용하면서도 추론한 정보를 새로 본 정보처럼 과신하지 않습니다.

<details markdown="1">
<summary>코드: memory update와 reciprocal wall</summary>

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

Mine 처리도 중요합니다. 시야 밖의 mine은 여전히 energy를 만들 수 있습니다. 하지만 cell이 실제로 보이지 않았는데 `last_seen`을 갱신하면 안 됩니다. 그러면 오래된 mine 추정을 방금 확인한 근거처럼 다루게 됩니다.

## **6. 경로 탐색과 후보 목표**

경로 탐색은 “어디로 가고 싶다”는 의도를 실제 첫 이동으로 바꿉니다. 이 베이스라인은 탐색 범위를 제한한 BFS를 사용합니다. Action timeout이 있기 때문입니다.

```text
목표 탐색이 너무 비싸면,
턴 제한 시간을 넘기기보다 가까운 안전 이동으로 낮춥니다.
```

경로 탐색층은 다음을 만듭니다.

| 경로 탐색 블록 | 목적 |
|---|---|
| BFS 첫걸음 | 목표까지 가는 합법 첫 방향을 찾습니다. |
| Factory route 확인 | cell을 예약하기 전에 factory가 북쪽으로 이득을 보는 route를 갖는지 확인합니다. |
| Jump 고려 route | wall 때문에 일반 이동이 막힐 때 jump가 tempo를 지킬 수 있는지 봅니다. |
| Frontier 탐색 | 기억된 cell 중 unknown space와 맞닿은 곳을 찾습니다. |
| 목표 후보 생성 | crystal, unload cell, wall job, node job, mine harvest, vanguard position을 생성합니다. |
| 목표 고정성 | 비슷한 target 사이에서 계속 바뀌는 현상을 줄입니다. |

기본적으로 경로 탐색은 friendly-occupied cell을 피합니다. 중요한 예외가 하나 있습니다. Miner가 `TRANSFORM`하면 movement 전에 해당 cell에서 사라지므로, carrier가 같은 턴에 그 cell로 들어갈 수 있습니다.

<details markdown="1">
<summary>코드: bounded BFS</summary>

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

이 함수는 베이스라인의 성격을 잘 보여줍니다. 경로 탐색은 crystal을 따라갈 가치가 있는지 결정하지 않습니다. 경로가 존재하는지와 첫 이동이 무엇인지 보고할 뿐입니다. 목표 점수화는 그 위층의 일입니다.

## **7. 자원 전환: Scout, Transfer, Miner**

Energy는 쓸 수 있는 형태로 돌아오기 전까지 점수가 아닙니다. 그래서 crystal 수집, transfer, mine transform은 각각 다른 판단 기준으로 다룹니다.

### Scout Ferry

Scout는 시야를 넓히고 crystal을 빠르게 줍기 때문에 유용합니다. 하지만 factory에서 멀리 떨어진 고에너지 scout는 위험 자산입니다. 앞으로 할 좋은 일이 없다면, 들고 있는 energy가 많을 때 unload 행동으로 전환해야 합니다.

```text
scout energy가 높고 forward value가 약하면:
    factory unload cell 쪽으로 돌아간다.
```

명시적인 threshold를 둡니다.

| 파라미터 | 의미 |
|---|---|
| `SCOUT_RETURN_ENERGY` | 완만하게 복귀를 유도하는 기준입니다. |
| `SCOUT_FORCE_RETURN_ENERGY` | 강제로 복귀시키는 기준입니다. |
| `UNLOAD_FRACTION` | capacity 대비 회수 압력이 얼마나 큰지 나타냅니다. |
| `ENABLE_MARGIN_SURPLUS_SCOUT` | route margin이 건강할 때만 scout를 허용합니다. |

### Transfer

Transfer는 움직이지 않기 때문에 안전해 보입니다. 하지만 실제로는 두 가지를 동시에 바꿉니다.

1. energy의 위치,
2. 보내는 robot이 에너지를 비운 뒤 길을 막는지 여부.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-03-transfer.png" alt="Unsafe transfer versus safer transfer in Maze Crawler" width="88%">
</p>

안전한 transfer는 energy 여유와 위치 구조를 함께 봐야 합니다. 다음 factory build를 가능하게 해 주는 transfer는 가치가 있을 수 있습니다. 하지만 factory corridor에 energy가 빈 robot을 남기는 transfer는 대개 나쁩니다.

또한 에너지를 너무 많이 비워내는 것도 피해야 합니다. Factory가 이미 이번 턴 build를 확정했다면, 이후 transfer는 그 build에 쓸 수 없습니다. 다음 build는 cooldown 때문에 어차피 지연됩니다. 그래서 transfer 검사는 현재 energy만이 아니라 이미 계획된 action도 함께 봅니다.

### Miner Transform

Miner는 투자입니다. Node를 mine으로 바꾸는 것 자체가 보상이 아닙니다. Mine energy가 scroll, 거리, 운반 제약 때문에 쓸모없어지기 전에 robot economy로 돌아와야 보상이 됩니다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-04-transform.png" alt="Good transform versus risky transform in Maze Crawler" width="88%">
</p>

현재 설정은 `MINER_TARGET_COUNT = 0`을 유지합니다. Mine hook은 존재하지만 보수적으로 꺼져 있습니다. 더 강한 mining economy를 넣기 전에 생존과 route-floor 동작이 안정적인지 먼저 측정해야 하기 때문입니다.

## **8. Factory tempo 관리**

Factory는 가장 중요한 예산입니다. Factory가 이동하지 않는 모든 행동에는 기회비용이 있습니다. 남쪽 경계가 계속 올라오기 때문입니다.

이 베이스라인은 factory 후보를 두 개로 나눕니다.

1. 생존을 보존하는 **floor candidate**,
2. 남는 여유를 support나 scouting에 쓰는 **complete candidate**.

선택 규칙은 다음과 같습니다.

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

항상 맞는 첫 build는 없습니다. 열린 corridor에서는 scout-first search가 가능합니다. 반대로 길이 막혔거나 branching이 낮은 시작은 worker-first wall control이 필요할 수 있습니다. 고정 opening script보다 route quality를 보고 결정하는 편이 더 안전합니다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-17-maze-crawler-structure-baseline/fig-06-jump-tradeoff.png" alt="Side-step versus jump north tradeoff in Maze Crawler" width="88%">
</p>

Jump도 같은 절충입니다. 북쪽 tempo를 회복할 수 있지만 cooldown을 씁니다. 그래서 emergency pressure에서는 jump를 더 적극적으로 쓰고, side route나 worker wall job으로 장기 진행을 지킬 수 있을 때는 더 아낍니다.

<details markdown="1">
<summary>코드: factory floor와 complete candidate 선택</summary>

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

이 설정의 이름은 다음입니다.

```python
PROFILE = "baseline_route_floor_hybrid"
```

짧은 에너지 이득을 최대화하려는 설정이 아닙니다. 안정적인 route floor를 지킨 뒤, 남는 margin으로 economy를 붙이는 설정입니다.

## **9. 전략층과 안전 계획**

전략층은 의도를 정합니다. 안전 계획은 그 의도가 실제 행동으로 확정될 수 있는지 결정합니다.

이 구분은 중요합니다. 좋은 전술적 아이디어가 나쁜 단일 행동으로 바뀌면서 실패하는 경우가 많기 때문입니다.

| 좋은 의도 | 나쁜 실행으로 생기는 실패 |
|---|---|
| Factory를 북쪽으로 이동 | Friendly worker가 destination을 막습니다. |
| Scout energy를 회수 | Scout transfer 뒤 corridor에 빈 robot이 남습니다. |
| 벽을 열기 | Worker가 upkeep과 wall cost를 모두 낼 수 없습니다. |
| Mine harvest | Carrier path가 다른 planned unit과 충돌합니다. |
| 위험에서 jump | Jump cooldown이 없거나 destination이 unsafe합니다. |

계획기는 긴급도 기준으로 robot 순서를 정합니다. Factory와 scroll line에 가까운 robot을 먼저 계획합니다. 이들의 reservation이 덜 급한 unit의 제약이 되기 때문입니다.

<details markdown="1">
<summary>코드: planning order, reservation, normalization</summary>

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

안전 계획층은 유효하지 않은 action을 `IDLE`로 정규화합니다. 소극적으로 보일 수 있지만, 불법 action 딕셔너리를 반환하고 engine의 실패 처리를 기다리는 것보다 훨씬 낫습니다.

## **10. Action 반환부**

Kaggle이 호출하는 진입점은 일부러 작게 둡니다. 핵심 계획 로직은 `compute_actions(...)`에서 따로 테스트할 수 있고, `agent(obs, config)`는 공유 state store를 연결해 결과만 돌려주는 얇은 껍질 역할을 합니다.

<details markdown="1">
<summary>코드: 대회 제출용 진입점</summary>

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

두 가지는 특히 유지할 가치가 있습니다.

| 구조 | 이유 |
|---|---|
| `state_store` 주입 | 테스트에서 memory를 격리할 수 있습니다. |
| `strategy_cls` 주입 | 규칙 처리와 safety를 유지한 채 policy만 바꿔 실험할 수 있습니다. |

이 구조 덕분에 단일 bot을 넘어 실험용 기반으로 쓸 수 있습니다.

## **11. 규칙 단위 검증**

한 게임의 점수는 흔들림이 큽니다. 그래서 점수를 해석하기 전에 규칙 단위 검사를 둡니다. 이 검사는 strategy가 강하다는 것을 증명하려는 것이 아닙니다. 튜닝 중 쉽게 깨지는 불변 조건을 보호하기 위한 장치입니다.

| 검사 묶음 | 보호하는 것 |
|---|---|
| Factory 생존 | route-first 쿨다운 처리, floor-candidate 압력, 안전한 jump |
| Worker 작업 | wall 방향, corridor 개방, vanguard 이동, 보호된 refuel |
| Scout 경제 | 한 scout의 목표 수, active scout 보충, 고에너지 복귀 |
| Mine 경제 | transform 타이밍, hidden mine energy, carrier 도달 가능성, same-turn harvest hook |
| Transfer 정책 | 부주의한 zero-energy blocker 방지, planned-build ledger 존중, route-critical 지원 |
| 제출 파일 | compile 성공과 `main.py` / `submission.py` source 동기화 |

Rule check가 실패하면 점수 해석을 멈추는 편이 맞습니다. 그렇지 않으면 깨진 규칙 처리부를 약한 strategy로 착각하기 쉽습니다.

생성된 agent는 독립 실행 가능한 제출 파일로 복사합니다.

```python
Path("submission.py").write_text(Path("main.py").read_text())
```

그리고 제출 파일을 import할 수 있는지, source가 동기화되어 있는지 확인합니다.

## **12. 평가 지표**

전략 비교의 올바른 단위는 같은 seed끼리 맞춰 보는 paired 비교입니다. 각 seed에 대해 다음 차이를 봅니다.

$$
d_i = R_i^{\text{candidate}} - R_i^{\text{baseline}}
$$

그리고 다음처럼 요약합니다.

$$
CI_{95}
\approx
\bar d
\pm
1.96 \frac{s_d}{\sqrt{N}}
$$

하지만 reward delta는 첫 번째 증거일 뿐입니다. Replay metric은 실패의 모양을 설명해야 합니다.

| 실패 모양 | 먼저 볼 지표 | 가능한 조정 |
|---|---|---|
| Factory가 일찍 죽음 | `factory_min_margin`, `factory_death_step`, `factory_build_when_route_exists` | floor candidate가 더 자주 우선되게 합니다. |
| Route는 있지만 진행이 약함 | `factory_idle_when_route_exists`, side-step streak, route cooldown turn | route scoring 또는 worker vanguard support를 개선합니다. |
| Scout 성능이 낮음 | `scout_build_margin_surplus`, `scout_energy_ge_90_turns`, `scout_transfer_count` | scout surplus threshold를 높이거나 floor run에서 scout를 끕니다. |
| Worker가 너무 빨리 나옴 | `factory_build_count`, first worker timing | long route가 막혔거나 dead end에 가까울 때만 worker를 만듭니다. |
| Worker가 wall job을 못 버팀 | `worker_energy_below_threshold`, failed wall-removal turns, refuel count | route-critical refuel budget을 비교합니다. |
| 나중에 jump가 없음 | `noncritical_north_wall_jump_count`, `jump_cd_unavailable_in_danger_count` | noncritical jump를 보수적으로 둡니다. |
| Transfer가 점수를 해침 | `factory_worker_refuel_count`, estimated overflow, worker wall action after refuel | refuel purpose gate 또는 overflow budget을 조입니다. |
| 많은 action이 IDLE로 정규화됨 | `normalized_to_IDLE_count` | legality, cooldown, reservation, target conflict를 확인합니다. |

진단 트리는 단순하게 유지합니다.

```text
Factory margin이 무너졌는가?
    yes -> floor candidate가 override되었는지 본다.
    no  -> worker가 route를 열어두지 못했는가?
        yes -> route-critical refuel과 worker wall targeting을 조정한다.
        no  -> scout energy가 안전하게 돌아왔는가?
            no  -> scout surplus / return threshold를 조인다.
            yes -> 이후 economy hook을 고려한다.
```

이것이 구조적 베이스라인의 이유입니다. Route floor가 약하면 생존 쪽을 조정합니다. Factory는 살아남지만 점수가 낮으면 resource conversion을 봅니다. 둘 다 괜찮은데 정체된다면 그때 더 강한 economy layer를 추가합니다.

## **13. 이 베이스라인의 역할**

Maze Crawler를 위한 실용적인 rule-based 기반은 다음 요소들로 구성됩니다.

| 기반 요소 | 가능해지는 것 |
|---|---|
| 규칙에 맞는 엔진 처리 | 전략 변경이 cooldown, wall, upkeep, collision rule을 다시 발견할 필요가 없습니다. |
| 오래 유지되는 world memory | Fog 속 사실을 계획에 쓰되, 오래된 추정을 방금 본 정보처럼 과신하지 않습니다. |
| 제한된 pathing | Action timeout을 위험하게 만들지 않고 유용한 route를 탐색합니다. |
| Factory floor policy | Economy보다 survival fallback이 먼저 존재합니다. |
| Safety shield | 좋은 의도가 불법 단일 행동으로 바뀌지 않도록 막습니다. |
| 규칙 검사 | 변동이 큰 점수 비교 전에 실험을 걸러낼 수 있습니다. |

이 결과물이 최종 완성형 에이전트라는 뜻은 아닙니다. 대신 이후 개선을 측정 가능하게 만드는 구조적 베이스라인입니다.

다음 실험은 자연스럽게 이어집니다.

1. factory floor와 complete candidate selector를 조정합니다.
2. paired seed에서 scout surplus threshold를 비교합니다.
3. route survival이 안정된 뒤에만 miner hook을 켭니다.
4. wall-block loss와 resource-conversion loss를 분리하는 replay metric을 추가합니다.
5. 규칙 처리부는 유지한 채 worker vanguard policy만 바꿔 테스트합니다.

마지막 교훈은 첫 규칙과 같습니다.

> **Maze Crawler에서 economy는 survival, pathing, safety가 먼저 제 역할을 한 뒤에야 실제 가치가 됩니다.**
