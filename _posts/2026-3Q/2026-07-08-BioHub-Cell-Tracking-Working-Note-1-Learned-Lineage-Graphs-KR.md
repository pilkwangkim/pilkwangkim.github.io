---
title: "BioHub Cell Tracking 작업 기록 1: 학습 기반 계보 그래프와 평가지표를 고려한 복원"
date: 2026-07-11 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, unet, ilp, graph-repair, working-note, korean]
math: true
pin: false
---

# BioHub Cell Tracking 작업 기록 1: 학습 기반 계보 그래프와 평가지표를 고려한 복원

- 대회: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- 공식 평가지표: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- 배경 기사: [Biohub Calls on AI Community to Transform 3D Cell Tracking](https://network.febs.org/posts/biohub-calls-on-ai-community-to-transform-3d-cell-tracking)
- 영문판: [BioHub Cell Tracking Working Note 1: Learned Lineage Graphs and Metric-Aware Repair](https://pilkwangkim.github.io/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)

관련 공개 노트북:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

이 글은 BioHub 세포 추적 대회를 다루는 첫 번째 작업 기록이다.
특정 제출 노트북을 해설하는 데 그치지 않고, 이후 실험이 임의적인 시행착오로 흐르지 않도록 문제의 구조와 판단 기준을 정리하는 것이 목적이다.

핵심은 다음 한 줄이다.

```text
이 문제는 단순한 3차원 분할 문제가 아니라,
평가지표가 풀이의 형태를 강하게 규정하는 세포 계보 그래프 복원 문제다.
```

따라서 가장 유용한 추상화는 다음이다.

$$
G=(V,E).
$$

각 노드 \(v\in V\)는 특정 시점의 세포 중심점이고, 방향 간선 \(e=(u\rightarrow v)\in E\)는 두 시점 사이에서 같은 세포를 잇는다.
세포 분열은 제출 파일에 별도의 레이블로 기록하지 않는다.
하나의 부모 노드에서 두 개의 자식 간선이 나가는 그래프 구조로 표현한다.

이 글의 모든 논리는 이 관점 위에 있다.

---

## 1. 대회와 문제 설정

이 대회에서는 제브라피시 배아의 3차원 형광 현미경 영상에서 세포를 추적한다.
각 샘플은 짧은 3차원 시계열 영상이다. 알고리즘은 시점마다 세포를 검출하고, 같은 세포를 시간축으로 연결하며, 분열 사건을 계보 그래프로 복원해야 한다.

생물학적으로 중요한 이유는 명확하다.
발생생물학에서 중요한 것은 한 프레임에 찍힌 세포의 위치만이 아니다.
세포가 어떻게 이동하고 분열해 하나의 개체를 이루는지를 함께 봐야 한다.
하지만 실제 3차원 현미경 영상은 세포가 조밀하고 잡음이 많으며, 축마다 해상도가 다르다. 세포의 외형도 서로 비슷하다.
세포는 움직이며 모양이 바뀌고, 흐려지거나 다른 세포와 가까워졌다가 둘로 갈라진다. 사람이 전 과정을 추적하기에는 금세 한계가 온다.

입력 영상의 배열 구조와 물리적 축척은 다음과 같다.

```text
sample.zarr/
  0/
    zarr.json
    c/{t}/0/0/0

shape: usually (T, Z, Y, X)
dtype: uint16
voxel scale:
  z = 1.62500 microns / voxel
  y = 0.40625 microns / voxel
  x = 0.40625 microns / voxel
```

학습 샘플에는 GEFF 형식의 희소한 계보 주석이 제공된다.

```text
sample.geff/
  nodes/ids
  nodes/props/t/values
  nodes/props/z/values
  nodes/props/y/values
  nodes/props/x/values
  edges/ids
```

핵심은 **희소하다**는 점이다.
영상에 보이는 모든 세포가 모든 프레임에 표시되어 있지는 않다.
따라서 일반적인 밀집 분할 검증만으로는 모델의 장단점을 제대로 판단하기 어렵다.
영상에서는 타당해 보이는 예측 세포도 희소한 정답 노드와 짝을 이루지 못할 수 있다.
결국 평가는 그래프 단위로 해야 하며, 후처리도 거짓 양성을 억제하는 쪽으로 설계해야 한다.

최종 제출물은 노드와 간선, 두 종류의 행으로 이루어진 CSV 파일이다.

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6_xxxx,node,1,0,32,128,128,-1,-1
1,44b6_xxxx,node,2,1,33,130,125,-1,-1
2,44b6_xxxx,edge,-1,-1,-1,-1,-1,1,2
```

노드 행은 세포 검출 결과를, 간선 행은 시간축 연결을 나타낸다.
평가기는 이 CSV를 하나의 그래프로 해석한다.

---

## 2. 평가지표가 곧 설계 조건이다

평가기는 먼저 각 시점에서 예측 노드와 정답 노드를 중심점 사이의 물리적 거리로 짝짓는다.
복셀의 축척이 방향마다 다르므로, 거리는 반드시 마이크로미터 단위로 계산해야 한다.

$$
d_{\mu m}(i,j)
=
\sqrt{
(1.625\Delta z)^2
+
(0.40625\Delta y)^2
+
(0.40625\Delta x)^2
}.
$$

두 노드의 거리가 다음 한계 안에 있을 때만 짝을 이룰 수 있다.

$$
d_{\mu m}(i,j) \le 7.0.
$$

간선의 양 끝 노드가 모두 정답 노드와 짝을 이루고, 그 정답 노드 사이에 실제 간선이 있을 때만 참 양성으로 인정된다.
기본 간선 점수는 Jaccard 지수다.

$$
J_{\text{edge}}
=
\frac{TP_{\text{edge}}}
{TP_{\text{edge}}+FP_{\text{edge}}+FN_{\text{edge}}}.
$$

노드를 지나치게 많이 예측하면 별도의 감점도 받는다.
이를 단순화한 보정 간선 점수는 다음과 같다.

$$
\tilde J_{\text{edge}}
=
\max
\left(
0,
J_{\text{edge}}
\left[
1-a\frac{T_{\text{pred}}-T_{\text{true}}}{T_{\text{true}}}
\right]
\right).
$$

공식 설명에서 감점 계수는 \(a=0.1\)이다.

분열 점수는 별도로 계산한다.
정답의 분열은 계보 그래프의 갈림점으로 나타난다.
예측 그래프가 분열 직전 구간과 두 자식 계보를 모두 포착해야 분열을 맞힌 것으로 본다.
최종 점수는 다음과 같이 구성된다.

$$
S
=
\tilde J_{\text{edge}}
+
wJ_{\text{division}},
\qquad
w=0.1.
$$

이 수식에서 세 가지 설계 원칙을 바로 얻을 수 있다.

| 평가지표의 압력 | 모델링에 미치는 영향 |
|---|---|
| 간선 Jaccard가 점수의 대부분을 차지한다 | 밝은 점을 많이 찾는 것보다 연결의 정확도가 중요하다. |
| 노드 과다 예측에 감점이 있다 | 고립된 잡음 노드와 짧은 조각은 오히려 손해가 될 수 있다. |
| 분열 항의 가중치는 작지만 0은 아니다 | 거짓 양성을 엄격히 통제할 수 있을 때만 분열 복원이 이득이다. |

따라서 제출 논리는 다음 방향으로 정리되었다.

```text
충분한 노드를 찾는다
간선을 정확히 고른다
기하학적 근거가 있는 그래프 오류만 복원한다
참 간선을 만들 가능성이 낮은 짧은 조각은 제거한다
```

---

## 3. 좌표계와 그래프 표현

구현에서 가장 중요한 원칙은 그래프 연산에는 원본 복셀 좌표를 쓰되, 거리 비교 직전에만 마이크로미터로 변환하는 것이다.

<details markdown="1">
<summary>코드: 물리 좌표계 변환</summary>

```python
import numpy as np

VOXEL_SCALE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

def distance_um(a_zyx, b_zyx):
    a = np.asarray(a_zyx, dtype=np.float32)
    b = np.asarray(b_zyx, dtype=np.float32)
    return float(np.linalg.norm((a - b) * VOXEL_SCALE_UM))

def within_match_gate(a_zyx, b_zyx, gate_um=7.0):
    return distance_um(a_zyx, b_zyx) <= gate_um
```

</details>

이 간단한 규칙만으로도 많은 오류를 막을 수 있다.
\(z\) 축의 간격은 \(x,y\) 축보다 약 네 배 크다.
복셀 인덱스에 유클리드 거리를 바로 적용하면 연결 허용 범위가 실제 공간에서 왜곡된다.

---

## 4. 첫 기준선: 고전적 중심점 검출과 최근접 연결

이 단계의 데이터 구조, 탐색적 분석, 고전적 기준선은 공개 노트북
[Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)에 정리했다.

첫 기준선은 의도적으로 단순하게 만들었다.

```text
3차원 프레임을 읽는다
-> 밝기 범위를 정규화한다
-> 밝은 국소 최댓값을 찾는다
-> 중심점을 보정한다
-> 물리 거리 기반 NMS로 중복을 제거한다
-> 이웃한 프레임을 Hungarian matching으로 연결한다
-> 노드와 간선을 기록한다
```

이 기준선의 목적은 높은 점수가 아니라 평가지표의 성질을 확인하는 것이었다.
노드 수가 급증하면 점수가 떨어졌고, 연결 허용 범위를 넓히면 거짓 간선이 쌓였다.
경계의 봉우리를 그대로 두면 한 프레임짜리 잡음 성분이 많아졌다.

중요했던 보정 단계는 다음과 같다.

1. **밝기 가중 중심점 보정**

   거칠게 찾은 봉우리 주변의 국소 영역 \(W\)에서 중심 좌표를 다시 계산한다.

   $$
   \hat{\mathbf r}
   =
   \frac{
   \sum_{\mathbf r\in W}
   \mathbf r
   \max(I(\mathbf r)-P_{20}(I_W),0)
   }{
   \sum_{\mathbf r\in W}
   \max(I(\mathbf r)-P_{20}(I_W),0)
   }.
   $$

2. **물리 거리 기반 NMS**

   중심점 보정 뒤 같은 세포 주변으로 모인 봉우리 중 가장 강한 하나만 남긴다.

   $$
   d_{\mu m}(i,j) < r_{\text{nms}},
   \quad s_i\ge s_j
   \Longrightarrow
   j\text{를 제거한다}.
   $$

3. **프레임별 후보 수 안정화**

   한 프레임에서 임계값이 무너지면 거짓 노드가 대량으로 생길 수 있다.
   이전 프레임의 개수를 기준으로 남길 후보 수 \(K_t\)에 상한을 둔다.

   $$
   K_t
   =
   \min
   \left(
   N_t,
   \left\lceil \alpha N_{t-1}+\beta \right\rceil
   \right).
   $$

   검출 강도가 높은 순서대로 \(K_t\)개만 유지한다.

화려한 기법은 아니지만, 평가지표에 맞는 출력을 만들기 위한 기본적인 정리 과정이다.

---

## 5. 학습 기반 그래프: Temporal UNet, Transformer, ILP

더 강한 모델에서는 Temporal UNet과 노드 Transformer를 그래프 추정의 중심축으로 삼았다.
전체 흐름은 다음과 같다.

```text
3차원 영상
-> Temporal UNet 중심점 검출기
-> 세포 후보 노드
-> 노드 특징과 학습된 간선 점수
-> ILP 기반 그래프 선택
-> 그래프 복원
-> submission.csv
```

검출기는 각 복셀에 세포 중심이 있을 확률장을 만든다.

$$
p_t(\mathbf r)
=
\sigma
\left(
f_\theta(X_{t-k:t+k})(\mathbf r)
\right),
$$

여기서 \(X_{t-k:t+k}\)는 프레임 \(t\) 주변의 짧은 시간 문맥이다.

검출된 노드 쌍의 연결 로짓(logit)은 다음과 같이 계산한다.

$$
z_{ij}
=
g_\phi(h_i,h_j,\Delta t,\Delta \mathbf r).
$$

\(h_i\)와 \(h_j\)는 두 노드의 학습된 표현이다.
이 로짓을 간선별로 독립 판정하지 않고, 뒤에서 설명할 정규화와 최적화 과정을 거쳐 최종 그래프를 고른다.
계보 그래프에서는 한 노드에 여러 부모가 붙을 수 없고, 분열에 해당하는 갈림도 드물며 물리적으로 타당해야 하기 때문이다.

### 5.1 학습된 간선 점수와 운동 기하의 결합

후처리 연결에서는 모델의 간선 확률만 사용하지 않는다.
출발 노드 \(i\)의 직전 변위가 있으면 다음 위치를 예측한다.

$$
\hat p_i
=
p_i+\lambda(p_i-p_{i-1}).
$$

도착 후보 \(j\)에 대한 배정 비용은 다음과 같다.

$$
C_{ij}
=
\|p_j-\hat p_i\|_2
+0.05\|p_j-p_i\|_2
-\beta q_{ij}.
$$

여기서 \(q_{ij}\)는 학습된 간선 확률이다. \(\beta\)는 비슷한 기하 비용을 가진 후보들 사이에서 모델의 판단을 얼마나 강하게 반영할지 정한다.
현재 기준 설정은 \(\lambda=0.5\), \(\beta=0.75\)다.
물리적 허용 범위를 벗어난 쌍에는 매우 큰 비용을 주고, Hungarian 알고리즘으로 일대일 대응을 고른다.

<details markdown="1">
<summary>코드: 간선 확률을 반영한 운동 기반 배정</summary>

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def assignment_cost(source_pos, target_pos, previous_pos, edge_prob):
    predicted = source_pos
    if previous_pos is not None:
        predicted = source_pos + 0.5 * (source_pos - previous_pos)

    raw_distance = np.linalg.norm(target_pos - source_pos)
    motion_distance = np.linalg.norm(target_pos - predicted)
    return motion_distance + 0.05 * raw_distance - 0.75 * edge_prob

cost = np.full((len(source_ids), len(target_ids)), LARGE_COST)
for i, source_id in enumerate(source_ids):
    for j, target_id in enumerate(target_ids):
        if raw_distance_um(source_id, target_id) <= gate_um:
            cost[i, j] = assignment_cost(
                position_um[source_id],
                position_um[target_id],
                predecessor_position_um.get(source_id),
                learned_edge_prob(source_id, target_id),
            )

rows, cols = linear_sum_assignment(cost)
matches = [
    (source_ids[i], target_ids[j])
    for i, j in zip(rows, cols)
    if cost[i, j] < LARGE_COST
]
```

</details>

이 식은 파라미터 민감도도 설명한다.
\(\beta\)가 너무 작으면 모델을 거의 사용하지 않는 최근접 운동 추적기가 되고,
너무 크면 보정이 덜 된 간선 점수가 타당한 운동 기하를 덮어쓴다.

---

## 6. 그래프 복원이 주된 개선축이 된 이유

학습 기반 그래프와 공백 복원 과정을 실행할 수 있는 공개본은
[Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)에서 볼 수 있다.

기본 모델이 충분히 강해진 뒤에는 검출 범위를 넓히는 변화가 오히려 위험해졌다.
실질적인 개선은 그래프를 직접 고치는 단계에서 나왔다.

```text
짧은 잡음 성분을 제거한다
물리적으로 타당한 이동 단절만 다시 연결한다
중간 지점에 근거가 있을 때만 한 프레임의 공백을 복원한다
엄격한 기하 조건을 만족할 때만 분열 간선을 추가한다
합성 노드가 불필요하게 늘어나지 않도록 제한한다
```

가장 효과가 컸던 복원 단계는 짧은 트랙을 솎아내는 일이었다.
노드가 몇 개뿐인 연결 성분은 참 간선을 거의 만들지 못하면서 노드 수와 거짓 간선 후보만 늘릴 수 있다.

규칙은 다음처럼 쓸 수 있다.

$$
\operatorname{keep}(C)
=
\mathbf 1
\left[
|V_C|\ge L_{\min}
\ \lor\
D_{\text{division}}(C)=1
\right].
$$

코드 형태는 단순하다.

<details markdown="1">
<summary>코드: 짧은 연결 성분 제거</summary>

```python
def keep_component(component_nodes, component_edges, min_track_len=7):
    if len(component_nodes) >= min_track_len:
        return True

    # Plausible forks should not be removed only because they are short.
    out_degree = {}
    for source_id, target_id in component_edges:
        out_degree[source_id] = out_degree.get(source_id, 0) + 1

    has_fork = any(deg >= 2 for deg in out_degree.values())
    return has_fork
```

</details>

이 단순한 규칙이 훨씬 복잡한 추가 모델보다 큰 효과를 냈다.
공개 리더보드에서는 극단적인 제거보다 적당한 길이 기준이 안정적이었다.

| 연결 성분의 최소 길이 | 공개 리더보드에서의 경향 |
|---:|---|
| 4 | 유용하지만 짧은 잡음이 더 남는다. |
| 6 | 공개 테스트에서 매우 강했다. |
| 7 | 이후 최고점 부근에서도 강한 설정이었다. |
| 12 | 쓸모 있는 짧은 트랙까지 지우기 시작한다. |
| 14 | 과도한 제거의 영향이 뚜렷하다. |

이 숫자가 모든 체크포인트에 그대로 적용되는 상수라는 뜻은 아니다.
검출기 체크포인트와 비공개 테스트 분포, 다른 복원 규칙에 따라 최적점은 달라진다.
다만 짧은 트랙 제거 자체가 유효한 개선축이라는 점은 분명했다.

---

## 7. 한 프레임 공백과 세포 분열의 보수적 복원

한 프레임 공백 복원은 다음과 같이 중간 시점의 검출이 빠진 경우를 다룬다.

```text
시점 t의 노드
시점 t+1에서 누락되거나 약하게 검출된 노드
시점 t+2의 노드
```

중간 지점의 초기 후보는 다음과 같다.

$$
\mathbf m
=
\frac{\mathbf r_t+\mathbf r_{t+2}}{2}.
$$

두 끝점 사이의 이동 거리가 물리적으로 타당할 때만 연결을 검토한다.

$$
d_{\mu m}(\mathbf r_t,\mathbf r_{t+2})
\le
2r_{\text{gap}}.
$$

\(\mathbf m\) 근처에 기존 노드가 있으면 새 노드를 만들지 않고 재사용한다.
없다면 합성 중간 노드를 만든 뒤 영상의 국소 밝기로 중심을 보정한다.
보정된 위치가 초기 중간점에서 너무 멀어지면 해당 후보는 버린다.

$$
d_{\mu m}(\hat{\mathbf m},\mathbf m)
\le
r_{\text{shift}}.
$$

합성 노드의 전체 개수에도 상한을 둔다.

$$
N_{\text{synthetic}}
\le
\min
\left(
N_{\text{abs}},
\left\lfloor \rho |V| \right\rfloor
\right).
$$

복원 후보가 기각되면 그래프를 수정 전 상태로 되돌려야 한다.
합성 노드만 남기고 간선을 취소하면 고립 노드 감점이 생길 수 있기 때문이다.

<details markdown="1">
<summary>코드: 실패 시 되돌리는 한 프레임 공백 복원</summary>

```python
def commit_gap_repair(
    source_id,
    target_id,
    source_point,
    target_point,
    t,
    endpoint_distance_um,
    nodes,
    edges,
    state,
):
    midpoint = 0.5 * (source_point + target_point)
    middle_id = find_reusable_isolated_node(midpoint, frame=t + 1)
    middle_reused = middle_id is not None

    if middle_id is None:
        if state.synthetic_added >= state.synthetic_cap:
            return False
        middle_id = next_node_id()
        refined = refine_with_local_intensity(midpoint, frame=t + 1)
        nodes[middle_id] = {
            "t": t + 1,
            "z": refined[0],
            "y": refined[1],
            "x": refined[2],
            "gap_synthetic": 1,
        }
        state.synthetic_added += 1

    needs_center = endpoint_distance_um >= 8.0
    if needs_center and center_score(nodes[middle_id]) < 0.20:
        if not middle_reused:
            nodes.pop(middle_id)
            state.synthetic_added -= 1
        return False

    edges.append((source_id, middle_id))
    edges.append((middle_id, target_id))
    return True
```

</details>

분열 복원은 이보다 더 보수적으로 처리한다.
부모 \(p_t\), 이미 연결된 첫째 자식 \(c^{(1)}_{t+1}\), 둘째 자식 후보 \(c^{(2)}_{t+1}\)가 있을 때 다음 조건을 모두 만족해야 간선을 추가한다.

$$
d_{\mu m}(p,c^{(2)})\le r_{\text{parent}},
$$

$$
d_{\mu m}(c^{(1)},c^{(2)})\le r_{\text{sister}},
$$

또한 둘째 자식 후보에는 기존 부모가 없어야 한다.

거짓 분열 간선은 분열 점수뿐 아니라 기본 간선 점수도 깎는다.
따라서 분열 재현율을 높이는 시도는 추가 간선의 정밀도가 충분히 높을 때만 이득이다.

---

## 8. 희소 라벨에 맞춘 학습 목적함수

Temporal UNet과 노드 Transformer는 검출과 연결을 함께 학습한다.
전체 목적함수는 두 항의 합이다.

$$
\mathcal L
=
\mathcal L_{\text{edge}}
+
\lambda_{\text{det}}\mathcal L_{\text{det}}.
$$

### 8.1 검출 손실

정답 중심이 있는 복셀을 \(y(\mathbf r)=1\), 나머지를 \(0\)으로 둔다.
문제는 주석이 희소하므로 \(y=0\)이 반드시 배경을 뜻하지는 않는다는 점이다.
이를 완화하기 위해 양성과 음성 항을 각각 개수로 정규화하고, 음성 전체에는 작은 계수 \(\eta\)를 곱한다.

$$
w_+=\frac{1}{N_+},
\qquad
w_-=\frac{\eta}{N_-},
$$

$$
\mathcal L_{\text{det}}
=
-\sum_{\mathbf r}
\left[
w_+y(\mathbf r)\log \sigma(s_{\mathbf r})
+
w_-(1-y(\mathbf r))\log(1-\sigma(s_{\mathbf r}))
\right].
$$

이렇게 하면 표시된 중심은 강하게 학습하면서도, 주석에 없다는 이유만으로 밝은 세포 후보를 강한 음성으로 몰아붙이지 않는다.

### 8.2 간선 손실

연속한 두 프레임의 정답 연결 행렬을 \(Y_{ij}\)라 하자.
희소 주석의 영향을 줄이기 위해, 정답 간선에 참여한 행이나 열만 학습 마스크 \(\mathcal M\)에 넣는다.

$$
\mathcal M_{ij}
=
\mathbf 1
\left[
\sum_kY_{ik}>0
\quad\lor\quad
\sum_kY_{kj}>0
\right].
$$

간선 로짓 \(z_{ij}\)는 **출발 노드 축**으로 softmax한다.

$$
q_{ij}
=
\frac{\exp z_{ij}}
{\sum_k\exp z_{kj}}.
$$

따라서 각 도착 노드는 하나의 부모를 선택하도록 경쟁하지만, 같은 출발 노드가 두 도착 노드에 높은 확률을 줄 수는 있다.
즉 merge는 억제하면서 세포 분열은 표현할 수 있다.

손실은 \(\gamma=2\)인 초점 이진 교차엔트로피(focal BCE) 형태다.

$$
p^*_{ij}
=
Y_{ij}q_{ij}+(1-Y_{ij})(1-q_{ij}),
$$

$$
\mathcal L_{\text{edge}}
=
-\frac{1}{|\mathcal M|}
\sum_{(i,j)\in\mathcal M}
(1-p^*_{ij})^2
\left[
Y_{ij}\log q_{ij}
+
(1-Y_{ij})\log(1-q_{ij})
\right].
$$

<details markdown="1">
<summary>코드: 희소 주석을 반영한 간선 손실</summary>

```python
import torch
import torch.nn.functional as F

def sparse_edge_loss(logits, target):
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    mask = active_rows[:, None] | active_cols[None, :]

    probs = torch.softmax(logits, dim=0)
    bce = F.binary_cross_entropy(probs, target, reduction="none")
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    return (((1.0 - p_t) ** 2) * bce)[mask].mean()
```

</details>

### 8.3 ILP가 맡는 역할

신경망이 내놓은 \(q_{ij}\)는 국소적인 연결 가능성일 뿐, 그 자체로 유효한 계보 그래프는 아니다.
추론에서는 이진 변수 \(x_{ij}\)로 간선 선택 여부를 나타내고, 출현·소멸·분열 비용을 더한 정수계획 문제를 푼다.
단순화한 목적함수는 다음과 같다.

$$
\min_{x,a,d,b}
-\lambda_e\sum_{ij}q_{ij}x_{ij}
+\lambda_a\sum_j a_j
+\lambda_d\sum_i d_i
+\lambda_b\sum_i b_i.
$$

핵심 제약은 다음 두 식으로 요약할 수 있다.

$$
\sum_i x_{ij}\le1,
\qquad
\sum_j x_{ij}\le1+b_i,
\qquad
b_i\in\{0,1\}.
$$

첫 식은 하나의 세포에 부모가 둘 이상 붙는 merge를 막는다.
둘째 식은 보통은 자식 하나만 허용하되, \(b_i=1\)인 분열 노드에는 자식 둘을 허용한다.
이 구조 덕분에 네트워크의 국소 점수와 계보 그래프의 전역 제약을 분리해 다룰 수 있다.

### 8.4 보조 Center 모델의 양성-미표기 손실

별도로 학습한 DeepCenterUNet3D는 한 프레임에서 중심점 heatmap만 예측한다.
이 모델도 희소 라벨 문제를 피해야 한다.
정답 중심 확률장을 \(h(\mathbf r)\), 영상 밝기의 40분위수를 \(Q_{0.4}(I)\)라 하면 복셀별 가중치는 다음과 같다.

$$
w(\mathbf r)
=
\begin{cases}
12, & h(\mathbf r)>0.05,\\
1, & I(\mathbf r)<Q_{0.4}(I),\\
0.05, & \text{otherwise}.
\end{cases}
$$

어두운 배경은 정상적인 음성으로 학습하고, 밝지만 주석이 없는 영역은 거의 무시한다.

$$
\mathcal L_{\text{center}}
=
\frac{
\sum_{\mathbf r}w(\mathbf r)
\operatorname{BCEWithLogits}(s_{\mathbf r},h(\mathbf r))
}{
\sum_{\mathbf r}w(\mathbf r)
}.
$$

<details markdown="1">
<summary>코드: 양성-미표기 가중치</summary>

```python
import numpy as np
import torch
import torch.nn.functional as F

weight_map = np.full(target_heatmap.shape, 0.05, dtype=np.float32)
background_cutoff = np.quantile(image, 0.40)
weight_map[image < background_cutoff] = 1.0
weight_map[target_heatmap > 0.05] = 12.0

target = torch.from_numpy(target_heatmap).to(logits)
weights = torch.from_numpy(weight_map).to(logits)
loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
loss = (loss * weights).sum() / weights.sum().clamp(min=1.0)
```

</details>

이 Center 모델은 기본 그래프를 대신하지 않는다.
11절에서 보듯, 시간 모델이 제안한 애매한 복원 후보에 영상 공간의 독립적인 근거를 보태는 역할에 가장 잘 맞았다.

---

## 9. 점수 변화에서 얻은 결론

공개 리더보드 점수는 비공개 검증을 대신하지 못한다.
그래도 한 번에 하나의 구조적 가설만 바꾸면 잡음이 섞인 진단 지표로는 활용할 수 있었다.
2026년 7월 11일까지의 점수 변화는 대략 다음과 같다.

| 단계 | 공개 점수 | 달라진 점 |
|---|---:|---|
| 고전적 검출과 초기 계보 기준선 | 0.68--0.75 | 제출 형식, 물리 거리, 그래프 출력을 검증했다. |
| 규칙 기반 기하 모델 | 0.82--0.86 | 학습 모델 없이도 보수적인 그래프 구조가 강했다. |
| 학습 모델의 첫 재현 | 약 0.81 | 공개 모델을 그대로 재현하는 것만으로는 부족했다. |
| 학습 그래프와 그래프 복원 | 0.844--0.860 | 운동 기반 재연결, 공백 복원, 짧은 트랙 제거가 실제 상승을 만들었다. |
| 체크포인트별 그래프 보정 | 0.885--0.897 | 체크포인트와 후처리를 하나의 시스템으로 맞췄다. |
| UNET400, 공간 TTA, 최소 트랙 길이 6 | 0.900 | 재현율을 무작정 넓히는 것보다 잘 보정된 그래프가 강했다. |
| UNET400 + Center400 조건부 공백 확인 | **0.901** | 보조 모델이 불확실한 수정만 확인할 때 처음으로 이득이 났다. |

### 9.1 학습 횟수는 독립적인 성능 축이 아니었다

125, 200, 250, 300, 400회 학습 결과를 비교하면서 얻은 가장 중요한 결론은,
학습을 오래 한다고 같은 후처리에서 점수가 자동으로 오르지는 않는다는 점이었다.
후기 체크포인트는 내부 손실이 더 낮더라도 검출 확률과 간선 확률의 보정 상태가 달라진다.
따라서 체크포인트는 다음 요소와 한 묶음으로 봐야 한다.

```text
모델 가중치
+ 검출 임계값
+ TTA 구성
+ 운동/간선 배정 비용
+ 복원 개수의 상한
+ 짧은 트랙 제거 기준
```

가중치만 따로 떼어 비교할 수 없다는 뜻이다.
실제로 검출 임계값을 \(0.9700\)에서 \(0.9675\), \(0.9725\)로 바꾼 두 제출은 모두 \(0.899\)였다.
이 축에서는 이미 국소 최적점 부근에 도달했다고 볼 수 있었다.

### 9.2 300ep와 400ep의 오류 구조 비교

같은 학습 영상 199개에서 300ep와 400ep의 예측을 비교한 결과는 다음과 같았다.

| 모델 | 간선 TP | 간선 FP | 간선 FN | 전체 간선 Jaccard | 평균 대리 점수 |
|---|---:|---:|---:|---:|---:|
| UNET300 | 121,669 | 5,212 | 7,214 | 0.907334 | 0.902110 |
| UNET400 | 122,151 | 5,202 | 6,732 | 0.910997 | 0.912574 |

400ep는 참 양성을 482개 늘리고 거짓 음성을 482개 줄였으며, 거짓 양성도 10개 줄였다.
샘플별로 보면 101개에서 좋아지고 86개에서 나빠졌다.
즉 400ep는 단순히 같은 예측에 확신만 더한 모델이 아니었다.
평균 성능은 나아졌지만 오류가 발생하는 위치와 유형이 달라졌으므로, 300ep에 맞춘 임계값을 그대로 옮기면 공개 점수가 오히려 떨어질 수 있었다.

여기서 주의할 점이 있다.
이 분석은 전체 학습 자료로 만든 체크포인트를 같은 자료에 다시 적용한 **표본 내 오류 분석**이다.
폴드별 미사용 자료에 대한 예측이 아니므로 진정한 OOF라고 부를 수 없다.
이 구분은 이후 그래프 복원 정책을 학습할 때 중요해졌다.

---

## 10. 실패한 실험이 알려준 것

실패한 실험을 통해 각 모델의 신호를 어디까지 믿어야 하는지 알 수 있었다.

### 10.1 무리한 재현율 확대와 과도한 TTA

밝기 변환 TTA와 공격적인 검출 후보 확장은 노드 수만 늘렸을 뿐 최고점 그래프를 넘지 못했다.
한 밝기 TTA 계열은 \(0.894\)까지 떨어졌다.
공간 반전과 XY 회전을 이용한 6-view TTA는 유용했지만, 변환의 수를 늘린다고 항상 좋아지지는 않았다.

### 10.2 `edge_predictor`라는 이름과 실제 가중치의 범위는 달랐다

가장 큰 실패 중 하나는 별도의 `edge_predictor_best.pth`를 기존 그래프에 단순히 끼워 넣은 실험이었다.
파일 이름만 보면 간선 점수기만 바뀔 것 같지만, 체크포인트에는 Temporal UNet 검출기를 포함한 전체 모델 상태가 들어 있었다.

| 출력 | 보정된 UNET400 기준 모델 | 보정 없이 교체한 체크포인트 |
|---|---:|---:|
| 노드 행 | 128,535 | 170,860 |
| 간선 행 | 123,988 | 164,603 |
| 공개 점수 | 약 0.900 | 0.861 |

노드와 간선이 약 33% 늘어났다.
새 체크포인트에 기존 검출 임계값 \(0.97\)과 설정을 그대로 적용했기 때문이다.
독립 시드 모델은 가중치만 교체해 제출할 것이 아니라, 따로 보정한 뒤 예측 그래프의 합의와 불일치를 이용해 결합해야 한다.

### 10.3 작은 ILP 변경도 독립적인 실험이어야 한다

분열 가중치를 \(1.0\)에서 \(0.7\)로 낮춘 변형은 \(0.897\)이었다.
같은 노트북에 `pool_kernel_um=2.0` 변경도 있었지만 예측이 끝난 뒤 실행되어 실제 제출에는 반영되지 않았다.
여러 변경을 한 셀에 몰아넣으면 실행 순서 때문에 무엇을 검증했는지조차 불분명해질 수 있다.

### 10.4 Center 모델을 전역 검출기로 합치는 것은 위험했다

DeepCenterUNet3D의 검출 결과를 무조건 합치거나, 모든 합성 노드에 대한 강제 통과 조건으로 사용했을 때는 이득이 없었다.
모든 합성 공백 노드에 Center 확률을 요구한 실험은 \(0.898\)로 떨어졌다.
실제로 누락된 세포는 흐리거나 다른 세포에 가려졌을 가능성이 높다.
Temporal UNet이 놓친 프레임에서는 Center도 낮은 점수를 내는 상관된 거짓 음성이 생긴다.

학습 횟수별 `best.pt`가 서로 다른 모델이라고 생각한 것도 함정이었다.
100--500ep 스냅숏의 `best.pt`는 모두 초기에 기록된 동일한 최적 체크포인트를 가리켰다.
특정 학습 시점의 Center를 비교하려면 각 시점의 마지막 체크포인트를 따로 평가해야 했다.

---

## 11. 보조 중심점 모델이 처음으로 효과를 낸 조건

Center 모델이 처음으로 이득을 만든 것은 전역 검출기로 합쳤을 때가 아니었다.
UNET400 그래프가 이미 제안한 **거리상 애매한 한 프레임 공백**만 확인하도록 제한했을 때였다.
이 실험은 공개 노트북
[Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)에 이어서 발전시켰다.

기본 공백 후보는 \(t\)에서 끝난 트랙과 \(t+2\)에서 시작한 트랙 사이의 중간점이다.

$$
\tilde p_{t+1}=\frac{p_t+p_{t+2}}{2}.
$$

UNET400과 Center400을 결합해 효과를 본 규칙은 다음과 같다.

$$
\operatorname{accept}(d,c)=
\begin{cases}
1, & d<8\ \mu\mathrm{m},\\
\mathbf{1}[c(\tilde p_{t+1})\ge0.20], & 8\le d\le12\ \mu\mathrm{m},\\
0, & d>12\ \mu\mathrm{m}.
\end{cases}
$$

여기서 \(d=\|p_{t+2}-p_t\|_2\)이고, \(c\)는 제안된 중간점 주변의 DeepCenter 확률이다.
이 구성은 \(0.901\)을 기록했다.
반대로 모든 합성 공백 노드에 \(c\ge0.15\)를 요구한 구성은 \(0.898\)이었다.

이 차이는 두 신호의 우선순위를 보여준다.

```text
강한 시간 기하 > 약한 Center 음성 근거
애매한 시간 기하 + Center 양성 근거 > 시간 기하 단독
```

Center 점수가 낮다는 사실만으로 세포가 없다고 단정할 수는 없다.
반면 운동 기하의 근거가 약한 곳에서 Center 점수가 높다면, 복원을 지지하는 독립적인 증거로 쓸 수 있다.

<details markdown="1">
<summary>코드: 거리 조건부 Center 확인</summary>

```python
def accept_gap(span_um, center_prob):
    if span_um < 8.0:
        return True
    if span_um > 12.0:
        return False
    return center_prob >= 0.20
```

</details>

고정 임계값을 거리에 따라 변하도록 일반화하면 다음과 같다.

$$
\tau_c(d)
=
0.12
+
\operatorname{clip}
\left(
\frac{d-8}{4},
0,
1
\right)
\cdot
(0.28-0.12).
$$

양방향 시간 문맥이 있으면 정방향과 역방향에서 예측한 중간점을 비교할 수 있다.

$$
\hat p_f=2p_t-p_{t-1},
\qquad
\hat p_b=2p_{t+2}-p_{t+3},
$$

$$
e_{\text{cons}}
=
\|\hat p_f-\hat p_b\|_2.
$$

\(e_{\text{cons}}\le2.5\,\mu\mathrm{m}\)이면 영상 공간의 낮은 Center 점수보다 시간축의 합의를 우선한다.

<details markdown="1">
<summary>코드: 거리 적응형 Center 조건과 양방향 합의</summary>

```python
import numpy as np

def adaptive_center_threshold(span_um):
    fraction = np.clip((span_um - 8.0) / 4.0, 0.0, 1.0)
    return 0.12 + fraction * (0.28 - 0.12)

def linear_context(previous_id, source_id, target_id, following_id):
    return (
        previous_id is not None
        and following_id is not None
        and out_degree[previous_id] == 1
        and in_degree[following_id] == 1
        and nodes[previous_id]["t"] + 1 == nodes[source_id]["t"]
        and nodes[target_id]["t"] + 1 == nodes[following_id]["t"]
    )

if linear_context(previous_id, source_id, target_id, following_id):
    forward_mid = 2.0 * position_um[source_id] - position_um[previous_id]
    backward_mid = 2.0 * position_um[target_id] - position_um[following_id]
    consensus_error = np.linalg.norm(forward_mid - backward_mid)
else:
    consensus_error = np.inf

if consensus_error <= 2.5:
    accept = True
else:
    threshold = adaptive_center_threshold(endpoint_distance_um)
    accept = center_probability >= threshold
```

</details>

여기서 이어지는 실험은 단순한 전역 임계값 탐색과 성격이 다르다.

1. 거리가 멀어질수록 Center 임계값을 \(0.12\)에서 \(0.28\)까지 높이는 적응형 조건
2. 앞뒤 속도 예측이 합의하면 Center의 거부를 무시하는 양방향 운동 조건
3. 애매한 공백을 합성 노드 생성과 기존 고립 노드 재사용으로 나누어 평가하는 실험

공통 원칙은 보조 모델의 영향력을 넓히는 것이 아니라, **불확실성의 종류에 따라 개입 범위를 제한하는 것**이다.

---

## 12. 모델 다양성과 진정한 OOF

두 번째 모델을 만드는 목적은 하나가 아니다.
다른 시드로 전체 자료를 학습한 모델은 테스트 시점의 앙상블과 모델 간 불일치를 측정하는 데 유용하다.
하지만 같은 학습 샘플에 대한 예측은 OOF가 아니다.
그래프 복원 정책을 학습하려면 각 샘플을 학습에서 제외한 폴드별 모델이 따로 필요하다.

```text
전체 자료로 학습한 독립 시드
-> 테스트 시점의 합의와 불일치 특징

두 폴드의 미사용 자료 예측
-> 진정한 OOF 그래프 수정과 복원 정책 레이블
```

<details markdown="1">
<summary>코드: 두 폴드 OOF 예측 수집</summary>

```python
from collections import Counter

oof_predictions = []
holdout_coverage = Counter()

for fold in (0, 1):
    train_movies = split_manifest[fold]["train"]
    holdout_movies = split_manifest[fold]["test"]

    assert set(train_movies).isdisjoint(holdout_movies)

    weight = weights_root / f"split_{fold}" / "edge_predictor_best.pth"
    predictions = predict_graphs(
        movies=holdout_movies,
        weight_path=weight,
    )
    oof_predictions.extend(predictions)
    holdout_coverage.update(holdout_movies)

expected_movies = set(all_training_movies)
assert set(holdout_coverage) == expected_movies
assert all(count == 1 for count in holdout_coverage.values())
```

</details>

확률을 혼합한다면 두 모델이 충분히 학습된 뒤 기준 모델에 더 큰 비중을 두는 편이 안전하다.

$$
p_{\mathrm{blend}}
=
\alpha p_{\mathrm{anchor}}
+
(1-\alpha)p_{\mathrm{seed}},
\qquad
\alpha\in[0.65,0.75].
$$

그러나 단순 평균이 최종 목표는 아니다.
두 모델이 같은 간선에 동의하는지, 어떤 상황에서 서로 다른 오류를 내는지가 더 중요한 정보다.

OOF 복원 학습표는 다음과 같이 구성할 수 있다.

| 수정 후보 | 레이블을 정하는 기준 |
|---|---|
| 기존 간선 | 짝지어진 정답 노드 사이에 같은 간선이 있는지 |
| 한 프레임 공백 연결 | 정답 그래프에 길이 2의 경로가 있는지 |
| 분열 간선 | 정답 그래프에 같은 갈림이 있는지 |
| 짧은 연결 성분의 유지/제거 | 해당 성분이 짝지어진 참 간선을 만드는지 |

각 간선이나 복원 후보의 특징 벡터는 원본 영상을 그대로 넣기보다, 평가지표와 직접 관련된 값으로 구성할 수 있다.

$$
x_{ij}
=
\left[
q_{ij},
d_{\text{raw}},
d_{\text{motion}},
\operatorname{rank}_{ij},
\deg^+(i),
\deg^-(j),
\rho_i,
\rho_j,
t_{\text{norm}},
c_{ij}
\right].
$$

여기서 \(c_{ij}\)는 필요할 때만 사용하는 Center 근거이고, \(\rho\)는 국소 세포 밀도다.
간단한 복원 분류기는 다음 확률을 출력한다.

$$
P(y_{ij}=1\mid x_{ij})
=
\sigma(w^\top x_{ij}+b).
$$

<details markdown="1">
<summary>코드: 경량 복원 정책의 저장 형식</summary>

```python
FEATURE_COLUMNS = [
    "edge_prob",
    "edge_dist_um",
    "motion_dist_um",
    "candidate_rank_dist",
    "source_out_degree",
    "target_in_degree",
    "source_density_7um",
    "target_density_7um",
    "t_norm",
    "center_support",
]

policy = {
    "feature_columns": FEATURE_COLUMNS,
    "mean": feature_mean.tolist(),
    "scale": feature_scale.tolist(),
    "coef": classifier.coef_[0].tolist(),
    "intercept": float(classifier.intercept_[0]),
    "threshold": float(oof_optimal_threshold),
}
```

</details>

결국 풀어야 할 질문은 다음과 같다.

```text
기하 구조, 모델 간 불일치, 국소 영상 근거가 주어졌을 때
어떤 그래프 수정이 실제 평가지표를 높이는가?
```

향후 실험 방향은 다음과 같이 정리된다.

```text
보정된 UNET400 그래프를 기준으로 유지한다
Center는 애매한 복원 후보를 확인하는 데만 사용한다
독립 시드 모델로 예측 그래프의 불일치를 측정한다
두 폴드 모델로 진정한 OOF 복원 학습표를 만든다
```
