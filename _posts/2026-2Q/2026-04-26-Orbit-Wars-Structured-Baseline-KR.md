---
title: "Orbit Wars: Structured Baseline Methodology -KR"
date: 2026-04-26 19:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, orbit-wars, game-ai, simulation, reinforcement-learning, korean]
math: true
pin: false
---

# Orbit Wars: Structured Baseline Methodology -KR

대회 링크:  
[Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)

Kaggle 노트북 링크:  
[Orbit Wars: Structured Baseline](https://www.kaggle.com/code/pilkwang/orbit-wars-structured-baseline)

관련 벤치마크 노트북:  
[Benchmark: How Strong Is Your Orbit Wars Agent?](https://www.kaggle.com/code/pilkwang/benchmark-how-strong-is-your-orbit-wars-agent)

Kaggle의 **Orbit Wars**는 2026년 4월 16일 UTC에 열린 연속 공간 전략 시뮬레이터 기반 Featured competition입니다. 겉으로는 행성에서 행성으로 함선을 보내는 게임처럼 보이지만, 모델링 관점에서는 꽤 까다로운 온라인 의사결정 문제입니다.

현재 턴에서 발사 여부와 방향, 함선 수를 정하지만 그 결과는 즉시 나타나지 않습니다. 함대가 이동하는 동안 행성은 생산을 하고, 다른 함대도 도착하며, 행성 자체가 회전할 수도 있고, 같은 턴에 여러 플레이어의 함대가 함께 충돌할 수도 있습니다. 따라서 이 문제는 단순한 greedy target selection이 아니라 **지연된 행동 결과를 다루는 online multi-agent control**로 보는 편이 자연스럽습니다.

강화학습 표기법으로 쓰면 다음과 같은 구조입니다.

$$
\pi(a_t \mid s_t), \quad
s_{t+1} \sim P(s_{t+1}\mid s_t,a_t), \quad
G_t = \sum_{k \ge 0}\gamma^k r_{t+k}
$$

여기서 중요한 것은 학습 정책을 바로 넣기 전에, 시뮬레이터가 요구하는 기본 계약을 먼저 명확히 하는 것입니다.

> 합법적인 직선 경로를 찾고, 실제 도착 시점의 목표 상태를 예측한 뒤, 앞서 확정한 발사 계획까지 반영했을 때 여전히 의미가 있는 경우에만 함선을 보냅니다.

이 한 문장이 전체 구조를 결정합니다. **물리 모델**은 행동이 가능한지와 언제 도착하는지를 판단하고, **월드 모델**은 그 도착 시점에 보드가 어떤 상태일지 예측하며, **전략층**은 그 예측 위에서 어떤 종류의 미션을 선택할지 정합니다.

| 핵심 원칙 | 의미 |
|---|---|
| 합법 경로가 먼저입니다. | 경로가 성립하지 않는 움직임은 점수화하지 않습니다. |
| 현재가 아니라 도착 시점으로 봅니다. | 지금 보이는 행성 상태가 아니라 ETA 시점의 소유권과 병력을 평가합니다. |
| 미션마다 계약이 다릅니다. | 점령, 구조, 보강, 재점령, 스나이프, 스웜은 서로 다른 조건을 갖습니다. |
| 확정한 발사는 미래 사실이 됩니다. | 이미 보낸 함대는 이후 판단에서 반드시 도착 예정 이벤트로 반영합니다. |

## **1. 한 턴의 에이전트 루프**

한 턴에서 관측값은 대략 다음과 같이 볼 수 있습니다.

$$
O_t =
\left(
player,\ planets_t,\ fleets_t,\ \omega,\ initial\_planets,\ comets_t
\right)
$$

출력은 발사 명령의 목록입니다.

$$
A_t =
\left[
(source_j,\ \theta_j,\ n_j)
\right]_{j=1}^{k_t}
$$

각 원소의 의미는 다음과 같습니다.

| 기호 | 의미 |
|---|---|
| `source_j` | 출발 행성 id |
| `theta_j` | 발사 각도 |
| `n_j` | 보낼 함선 수 |

정책은 결정론적으로 쓸 수 있습니다.

$$
\pi : O_t \mapsto A_t
$$

하지만 핵심은 어떤 목표를 고르느냐보다, **발사 후보가 실제 행동으로 확정되기 전에 어떤 검증을 거치는가**입니다.

| 객체 / 신호 | 모델에서의 역할 | 무시했을 때 생기는 문제 |
|---|---|---|
| 행성 상태 | 출발지 병력, 목표 가치, 생산량 예측 | 무리한 확장 또는 방어 붕괴 |
| 함대 상태 | 미래 도착 기록 | 낡은 소유권 추정 |
| 태양 | 직선 경로의 합법성 판정 | 태양에 걸려 함대 손실 |
| 각속도 | 회전 행성의 미래 위치 예측 | 현재 위치를 향해 잘못 조준 |
| 혜성 경로 | 짧은 시간 동안만 유효한 이동 목표 | 이미 지나간 기회 추적 |

목표는 다음처럼 볼 수 있습니다.

$$
\max_{\pi}\ \mathbb{E}\left[ R_{\text{final}} \mid \pi,\ O_0 \right]
$$

어려운 점은 delay입니다.

> 턴 `t`에서 선택한 발사는 `t + T`에 해결됩니다. 그 사이에 생산, 이동, 보이는 함대, 다른 플레이어의 도착, 같은 턴 전투가 모두 보드를 바꿉니다.

## **2. 스냅샷만 보면 틀립니다**

스냅샷 기반 휴리스틱은 현재 보이는 행성만 보고 판단합니다. Orbit Wars에서는 이 방식이 쉽게 깨집니다.

| 제약 | 스냅샷 판단이 깨지는 이유 |
|---|---|
| `100 x 100` 연속 공간 | 거리와 각도가 grid step이 아니라 실수값입니다. |
| `500`턴 경기 | 생산과 늦은 도착이 누적됩니다. |
| `2` 또는 `4` 플레이어 | 1대1에서 안전한 움직임이 4인전에서는 처벌받을 수 있습니다. |
| 매우 짧은 행동 시간 | 무거운 탐색은 제한해야 합니다. |
| `[from_planet_id, angle, num_ships]` 행동 | 에이전트가 각도와 함선 수를 직접 정합니다. |
| 회전 행성 | 목표 위치가 도착 전까지 변합니다. |
| 함선 수에 따른 속도 변화 | 보낸 함선 수가 ETA를 바꿉니다. |
| 태양 충돌 | 일부 직선 경로는 불법입니다. |
| 같은 턴 전투 | 여러 도착이 비선형으로 상호작용합니다. |

중심 질문은 이것이 아닙니다.

> 지금 어느 행성이 가치 있어 보이는가?

더 중요한 질문은 이것입니다.

> 이 출발지에서 이 함선 수로 지금 출발하면, 도착 턴의 목표 상태는 무엇인가?

이 시간 이동 때문에 구조는 현재 스냅샷 휴리스틱이 아니라 **앞으로의 상태를 예측하는 월드 모델** 중심으로 잡힙니다.

## **3. 에이전트를 세 모델로 나누는 이유**

구조는 세 층으로 나뉩니다.

$$
\text{physics} \rightarrow \text{world model} \rightarrow \text{strategy}
$$

각 층이 대답하는 질문은 다릅니다.

| 층 | 대답해야 하는 질문 | 담당하지 않는 것 |
|---|---|---|
| 물리 모델 | 이 직선 발사가 합법인가, 언제 도착하는가? | 목표 가치 |
| 월드 모델 | 도착 턴에 목표 행성은 누가 소유하는가? | 전략 선호도 |
| 전략층 | 이 미션에 함선을 쓸 가치가 있는가? | 원시 경로 기하 |

아래 층은 위 층의 선호도를 몰라야 합니다. 물리 모델은 목표가 전략적으로 매력적인지 몰라도 됩니다. 월드 모델은 미션이 좋아 보인다는 이유로 전투 규칙을 바꾸면 안 됩니다. 전략층은 소유권을 추측하지 말고 월드 모델에 물어봐야 합니다.

의사결정 흐름은 다음처럼 요약됩니다.

$$
\text{legal route}
\Rightarrow
\text{arrival-time state}
\Rightarrow
\text{mission score}
\Rightarrow
\text{committed launch}
$$

하나의 발사가 확정되려면 다음 조건이 함께 만족되어야 합니다.

$$
\operatorname{Legal}(s,i,n)
\land
\operatorname{ArrivesUseful}(i,T)
\land
\operatorname{OwnsOrHolds}(i,T,n)
\land
\operatorname{BudgetSafe}(s,n)
$$

## **4. 물리 모델: 존재하지 않는 경로를 만들지 않습니다**

먼저 보드는 정사각 평면으로 봅니다.

$$
p = (x, y), \quad x, y \in [0, 100]
$$

태양은 중앙에 있습니다.

$$
c = (50, 50), \quad R_{\odot} = 10
$$

구현에서는 태양 주변에 safety margin을 둡니다.

$$
R_{\text{blocked}} = R_{\odot} + \epsilon
$$

구현에서는 `epsilon = 1.5`를 사용합니다. 아주 간신히 통과하는 경로는 안전하지 않은 경로로 취급합니다. 전략층은 이 필터를 통과한 후보만 보게 됩니다.

출발 행성 `s`와 목표 행성 `t`에 대해, 거리는 중심에서 중심까지가 아니라 출발 행성 경계에서 목표 행성 원에 처음 닿는 지점까지로 봅니다.

$$
D(s,t) \approx \max\left(0,\lVert p_t - p_s \rVert - r_s - r_t - \delta\right)
$$

방향은 직접 각도입니다.

$$
\theta = \operatorname{atan2}(y_t-y_s, x_t-x_s)
$$

그리고 실제로 검사해야 하는 것은 launch point에서 hit point까지의 선분입니다. 이 선분이 태양 차단 원을 지나면 안 됩니다.

$$
\operatorname{dist}\left(c,\overline{ab}\right) \ge R_{\text{blocked}}
$$

함선 수가 속도를 바꾸기 때문에, 모든 발사 규모에 같은 ETA를 재사용할 수 없습니다.

$$
v(n) =
\begin{cases}
1, & n \le 1 \\
1 + 5 \cdot \left(\operatorname{clip}\left(\frac{\log n}{\log 1000},0,1\right)\right)^{1.5}, & n > 1
\end{cases}
$$

도착 턴은 다음과 같습니다.

$$
T(s,t,n) = \left\lceil \frac{D(s,t)}{v(n)} \right\rceil
$$

이 식 하나가 코드 구조에 큰 영향을 줍니다. `n`이 바뀌면 속도가 바뀌고, 속도가 바뀌면 도착 턴이 바뀌며, 도착 턴이 바뀌면 회전 행성의 위치가 바뀝니다. 그러면 같은 목표라도 경로가 합법에서 불법으로 바뀔 수 있습니다. 그래서 최종 발사 함선 수가 결정된 뒤에는 다시 조준해야 합니다.

회전 행성의 위치는 도착 턴 기준으로 예측합니다.

$$
p_t(T) =
c + \rho
\begin{bmatrix}
\cos(\theta_0 + \omega T) \\
\sin(\theta_0 + \omega T)
\end{bmatrix}
$$

바깥쪽 행성은 사실상 정적 대상으로 취급합니다.

$$
\rho + r \ge 50
$$

혜성은 path array로 미래 위치가 주어지는 임시 이동 행성처럼 다룹니다. 따라서 높은 가치의 혜성도, 도착하기 전에 유효한 경로가 끝나면 의미가 없습니다.

<details markdown="1">
<summary>코드: 속도, 태양 충돌, 직선 경로 검증</summary>

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
```

</details>

이 층의 원칙은 단순합니다.

> 전략층이 함선을 쓰기 전에, 물리 모델이 행동의 존재를 먼저 증명해야 합니다.

낮은 가치의 합법 move가, 높은 가치처럼 보이는 불법 move보다 낫습니다.

## **5. 이동 목표는 도착 시점으로 다시 조준합니다**

움직이는 행성이나 혜성은 현재 위치로 쏘면 안 됩니다. 함대가 도착하는 턴에는 목표가 다른 위치에 있을 수 있습니다. 그래서 미래 턴 후보를 훑으면서, 그 턴의 목표 위치와 ETA가 서로 일관되는 지점을 찾습니다.

<details markdown="1">
<summary>코드: 이동 목표의 일관된 intercept 탐색</summary>

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

여기서 좋은 점은 목표 위치 예측과 ETA 계산이 서로 맞아야 한다는 것입니다. 단순히 “미래 위치를 하나 찍고 쏜다”가 아니라, **그 위치를 향해 쐈을 때 실제 도착 턴이 다시 그 위치의 시점과 맞는지** 확인합니다.

## **6. 월드 모델: 도착 턴의 소유권**

월드 모델이 묻는 질문은 하나입니다.

> 내가 보낸 함대가 도착할 때, 목표 행성은 어떤 상태인가?

이를 위해 보이는 함대들의 도착 기록을 만들고, 생산을 반영하고, 같은 턴 전투를 해결합니다. 같은 턴 전투는 단순한 디테일이 아닙니다. 여러 플레이어의 도착 병력이 소유자별로 묶이고, 가장 큰 두 세력이 먼저 상쇄된 뒤, 살아남은 세력만 행성 주둔군과 싸웁니다.

행성 `i`의 상태는 다음처럼 볼 수 있습니다.

$$
S_i(t) = \left(o_i(t), g_i(t)\right)
$$

`o_i(t)`는 소유자, `g_i(t)`는 주둔 병력입니다. 도착 이벤트가 해결되기 전에, neutral이 아닌 소유자는 생산을 받습니다.

$$
g_i^-(t) =
\begin{cases}
g_i(t-1) + q_i, & o_i(t-1) \ne -1 \\
g_i(t-1), & o_i(t-1) = -1
\end{cases}
$$

그다음 해당 턴에 도착하는 함대를 소유자별로 합칩니다.

$$
A_k(t) = \sum_{f \in \mathcal{F}_{i,t},\ \operatorname{owner}(f)=k} \operatorname{ships}(f)
$$

가장 큰 두 세력이 같으면 둘 다 사라집니다. 다르면 가장 큰 세력이 차이만큼 살아남습니다.

$$
B(t) = A_{k_1}(t) - A_{k_2}(t)
$$

살아남은 세력이 행성 소유자와 같으면 보강입니다.

$$
g_i(t) = g_i^-(t) + B(t)
$$

적이면 주둔 병력에서 빠집니다.

$$
g_i(t) = g_i^-(t) - B(t)
$$

이 값이 음수가 될 때만 소유권이 바뀝니다. 그래서 `현재 병력 + 1` 같은 단순 추정이 자주 틀립니다.

<details markdown="1">
<summary>코드: 같은 턴 전투와 행성 시간 전개 시뮬레이션</summary>

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

## **7. 필요한 함선 수는 시뮬레이션으로 구합니다**

여러 미션은 겉으로는 달라 보이지만, 내부 질문은 같습니다.

| 미션 | 월드 모델 질문 |
|---|---|
| 점령 | 도착 후 내 소유가 될 수 있는가? |
| 구조 | 내 행성이 무너지기 전에 보강이 도착할 수 있는가? |
| 보강 | 특정 기간까지 소유권을 유지할 수 있는가? |
| 재점령 | 구조가 늦었다면 빠르게 다시 되찾을 수 있는가? |
| 스나이프 | 다른 플레이어의 도착이 필요 병력을 줄여주는가? |
| 스웜 | 여러 출발지의 partial send를 같은 유효 시점에 맞출 수 있는가? |

중심 질문은 “지금 병력이 몇 개인가?”가 아닙니다.

> 관련 도착 이벤트까지 고려했을 때, 특정 턴에 내가 소유하려면 몇 척이 필요한가?

이를 수식으로 쓰면 다음과 같습니다.

$$
N_i(T, a) =
\min \left\{
n \ge 0 :
o_i\left(T;\ \mathcal{A}_i \cup \{(T,a,n)\}\right) = a
\right\}
$$

여기서 `i`는 목표 행성, `T`는 도착 턴, `a`는 공격자 소유자, \(\mathcal{A}_i\)는 이미 알려진 도착 기록입니다.

구현에서는 이를 closed-form으로 억지로 풀지 않고, 시간 전개 시뮬레이션과 binary search로 계산합니다. 같은 턴 전투, 생산, 이미 계획한 발사, 여러 소유자의 동시 도착이 섞이면 손으로 만든 shortcut이 틀리기 쉽기 때문입니다.

방어는 더 엄격합니다. 한 순간 소유하면 끝나는 것이 아니라, 일정 기간까지 유지해야 할 수 있습니다.

$$
H_i(\tau, L) =
\min \left\{
n \ge 0 :
\forall t \in [\tau, L],\
o_i\left(t;\ \mathcal{A}_i \cup \{(\tau,p,n)\}\right) = p
\right\}
$$

`tau`는 보강 도착 턴, `L`은 hold-until 턴, `p`는 내 player id입니다. 이것이 구조와 보강의 차이입니다. 구조는 특정 붕괴를 막는 일이고, 보강은 그 뒤 일정 기간까지 버티게 만드는 일입니다.

<details markdown="1">
<summary>코드: 도착 시점 소유권과 유지 조건 질의</summary>

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
```

</details>

## **8. 전략층은 미션 묶음을 고릅니다**

전략층은 합법 경로와 도착 시점 예측을 받은 뒤에야 선호도를 계산합니다. 하나의 큰 score formula로 모든 것을 섞을 수도 있지만, 이 구조에서는 미션 묶음을 분리합니다. 각 미션의 유효 조건이 다르기 때문입니다.

하나의 후보 미션은 다음처럼 볼 수 있습니다.

$$
m = (kind,\ target,\ sources,\ ships,\ T)
$$

유효한 미션 집합은 다음 조건으로 걸러집니다.

$$
m \in \mathcal{M}_{valid}
\quad \Longleftrightarrow \quad
\operatorname{Legal}(m)
\land
\operatorname{TemporalCondition}(m)
\land
\operatorname{OwnershipCondition}(m)
$$

그다음에야 가치와 비용을 비교합니다.

$$
Score(m)
=
V(target, kind, T)
- C(ships, T)
- Risk(m)
$$

| 항목 | 의미 |
|---|---|
| `V` | 생산량, 전략적 위치, 적 압박, late-game swing을 보상합니다. |
| `C` | 함선 비용과 이동 시간을 벌점으로 둡니다. |
| `Risk` | contested timing, 적 반응 window, 출발 행성의 과소비를 반영합니다. |

중요한 점은 score가 feasibility와 ownership check 이후에 나온다는 것입니다. 아무리 좋아 보이는 목표라도, 합법적으로 도착하지 못하거나 도착 시점에 소유할 수 없으면 후보가 아닙니다.

방어 미션은 특히 구분이 필요합니다.

| 방어 미션 | 시간 의미 |
|---|---|
| Reinforce-to-hold | 이미 가진 행성을 일정 기간까지 계속 지킵니다. |
| Rescue | fall turn 전에 보강이 도착해 붕괴를 막습니다. |
| Recapture | 구조는 늦었지만 빠르게 되찾을 가치가 있습니다. |
| Salvage | 출발지가 이미 위험하므로 유용한 함선을 다른 곳으로 뺍니다. |

공격도 같은 방식으로 나뉩니다.

| 공격 미션 | 핵심 아이디어 |
|---|---|
| Single-source capture | 한 출발지만으로 목표를 점령합니다. |
| Hostile pressure | 적 생산 또는 병력 덩어리를 압박합니다. |
| Neutral snipe | 다른 플레이어의 타이밍을 이용해 중립 목표를 빼앗습니다. |
| Swarm | 여러 출발지의 partial send를 맞춰 전환합니다. |
| Crash exploit | enemy-vs-enemy 충돌 뒤의 빈틈을 이용합니다. |
| Late follow-up | 시간이 짧을 때 즉시 병력 swing을 우선합니다. |

이렇게 나누면 모든 규칙이 완벽해지는 것은 아닙니다. 대신 각 전술적 아이디어가 이름과 계약을 갖게 됩니다. 전략층은 물리 모델과 월드 모델을 통과한 미션만 순위를 매깁니다.

출발지별 예산도 필요합니다. 행성 `s`에서 공격에 쓸 수 있는 병력은 다음처럼 둡니다.

$$
B_s = \max(0,\ ships_s - R_s)
$$

reserve는 예측된 hold need와 proactive danger의 최댓값입니다.

$$
R_s = \max\left(
R^{forecast}_s,\ R^{reaction}_s
\right)
$$

즉 확장과 압박은 기존 생산 기반을 지키는 데 필요한 함선을 침식하면 안 됩니다.

## **9. 최종 발사 함선 수는 다시 맞춰야 합니다**

함선 수가 바뀌면 ETA가 바뀝니다. ETA가 바뀌면 도착 시점의 필요 병력도 바뀝니다. 그래서 “필요 병력 계산 -> 발사 함선 수 결정 -> 경로 재확인”을 반복해야 합니다.

<details markdown="1">
<summary>코드: 미션 구조와 최종 발사량 조정</summary>

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

이 부분은 작은 구현 디테일처럼 보이지만 매우 중요합니다. probe size로 찾은 경로를 final send에 그대로 쓰면, 실제 속도와 ETA가 달라져 월드 모델이 본 조건과 실행되는 행동이 달라질 수 있습니다.

## **10. 커밋 루프가 전체를 묶습니다**

후보 move들은 서로 독립적이지 않습니다. 어떤 출발 행성이 40척을 보내면 그 행성에는 더 이상 그 40척이 없습니다. 그 함대가 17턴 뒤 도착한다면, 같은 목표에 대한 이후 예측은 반드시 그 예정 도착을 포함해야 합니다.

확정한 발사는 미래 사실로 들어갑니다.

| 단계 | 작업 | 의미 |
|---|---|---|
| 1 | 미션 선택 | 유효한 후보에서 시작합니다. |
| 2 | 최종 발사 함선 수 재조준 | 함선 수가 속도와 ETA를 바꿉니다. |
| 3 | 출발지 inventory 차감 | 한 행성을 과도하게 쓰지 않습니다. |
| 4 | planned arrival 추가 | 이후 예측에 해당 발사가 반영됩니다. |
| 5 | 뒤의 미션 재평가 | 이후 판단은 업데이트된 세계를 봅니다. |

행성 `i`의 기본 도착 기록은 다음과 같습니다.

$$
\mathcal{A}_i = \{(T, \operatorname{owner}, \operatorname{ships})\}
$$

발사를 확정하면 planned commitment가 추가됩니다.

$$
C_i \leftarrow C_i \cup \{(T_{\text{new}}, player, n_{\text{new}})\}
$$

이후 월드 모델 질의는 원래 도착 기록이 아니라 다음을 읽습니다.

$$
\mathcal{A}_i \cup C_i
$$

따라서 이후 소유권 예측은 같은 미래 도착 기록 위에서 이루어집니다.

$$
S_i(t \mid \mathcal{A}_i \cup C_i)
$$

이 구조가 시간적 일관성을 만듭니다. 또한 split launch가 의도적인 경우에도 의미가 보존됩니다. 함선 수가 속도를 바꾸기 때문에, 작은 동기화 발사 두 개는 큰 발사 하나와 전술적으로 다를 수 있습니다.

<details markdown="1">
<summary>코드: 확정 발사를 미래 도착 기록에 반영하는 루프</summary>

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
```

</details>

## **11. 일관성 검사는 같은 층 구조를 따릅니다**

정책이 의존하는 계약은 다음과 같습니다.

| 계약 | 정의 |
|---|---|
| Route | 직선 발사, 태양 안전 선분, 예측 ETA |
| State | 특정 턴의 예측 소유자와 주둔 병력 |
| Ownership | 특정 턴에 소유하기 위한 최소 병력 |
| Hold | 일정 기간까지 소유권을 유지하기 위한 최소 보강 |
| Commitment | 확정한 발사는 이후 판단의 미래 도착이 됩니다. |

검증도 같은 분해를 따릅니다.

| 검사 | 테스트하는 계약 |
|---|---|
| 태양 안전 routing | 경로 합법성 |
| 이동 목표 예측 | ETA를 반영한 목표 위치 |
| 같은 턴 전투 | 소유자별 도착 해결 |
| hold semantics | reinforce/rescue/recapture 구분 |
| swarm timing | 여러 출발지의 동기화된 소유권 확보 |
| crash window timing | 충돌 뒤 기회 |
| commitment 이후 salvage | 미래 상태 일관성 |

이 검사는 전략 점수가 좋은지보다 더 아래의 질문을 봅니다. 전략층이 믿고 있는 수학적 계약이 실제 행동 모델과 맞는지 확인하는 것입니다.

## **12. 확장 방향**

같은 분해는 이후 개선 방향도 보여줍니다.

| 확장 | 좋아지는 부분 |
|---|---|
| Multi-agent pressure model | 4인전 생존력 |
| Empirical mission-weight calibration | 손으로 맞춘 scoring 의존도 감소 |
| Opponent-specific reaction estimates | contested target 평가 |
| Local search over source allocation | 여러 출발지 coordinated attack |
| Late-game value function | 제거, blocking, 즉시 병력 swing |
| Layered benchmark protocol | route, forecast, strategy 오류 분리 |

이 확장들도 같은 순서를 보존해야 합니다.

$$
\text{physics} \rightarrow \text{world model} \rightarrow \text{strategy}
$$

전략층은 월드 모델에 질의하고, 월드 모델은 물리 모델에 질의할 수 있습니다. 하지만 물리 모델이 전략 선호도에 의존하면 안 됩니다.

## **13. 마지막 불변 조건**

최종 정책은 다음 불변 조건을 지켜야 합니다.

| 불변 조건 | 요구 조건 |
|---|---|
| Route legality | 모든 발사는 `Legal(source, target, ships)`를 만족합니다. |
| ETA consistency | 최종 함선 수가 암시하는 같은 `T`에서 목표 상태를 질의합니다. |
| Ownership validity | 공격 미션은 `N_i(T, player) <= ships_sent`를 만족합니다. |
| Hold validity | 방어 미션은 `H_i(tau, L) <= reinforcement_sent`를 만족합니다. |
| Budget validity | 각 출발지는 가용 예산 `B_s`를 넘겨 보내지 않습니다. |
| Commitment consistency | 이후 예측은 모든 확정 발사를 포함합니다. |

짧게 쓰면, 확정된 발사는 다음을 만족해야 합니다.

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

전략층은 이 조건을 만족하는 행동들 사이에서만 최적화해야 합니다.

<details markdown="1">
<summary>코드: 환경에 노출되는 래퍼</summary>

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

## **14. 정리**

Orbit Wars에서 좋은 baseline은 단순히 “좋아 보이는 행성으로 많이 보낸다”가 아닙니다. 먼저 직선 경로가 실제로 존재해야 하고, 도착 시점의 목표 위치와 소유권을 봐야 하며, 이미 확정한 발사를 이후 예측에 반영해야 합니다.

논리 사슬은 다음과 같습니다.

```text
물리적으로 가능한가?
-> 도착 턴에 어떤 상태인가?
-> 어떤 미션 계약을 만족하는가?
-> 출발지 예산을 넘지 않는가?
-> 확정한 발사를 이후 예측에 반영했는가?
```

이 구조가 있으면 이후에 더 공격적인 전략이나 학습 기반 policy를 얹더라도, 아래층의 계약을 깨지 않고 실험할 수 있습니다. 결국 이 baseline의 가치는 특정 상수 몇 개보다, **물리 모델, 월드 모델, 전략층을 분리해 행동의 합법성과 시간적 일관성을 보존하는 구조**에 있습니다.
