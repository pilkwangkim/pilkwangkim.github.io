---
title: "BioHub Cell Tracking Working Note 1: Learned Lineage Graph와 Metric-Aware Repair"
date: 2026-07-11 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, unet, ilp, graph-repair, working-note, korean]
math: true
pin: false
---

# BioHub Cell Tracking Working Note 1: Learned Lineage Graph와 Metric-Aware Repair

대회 링크:  
[BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)

공식 metric 설명:  
[RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)

배경 기사:  
[Biohub Calls on AI Community to Transform 3D Cell Tracking](https://network.febs.org/posts/biohub-calls-on-ai-community-to-transform-3d-cell-tracking)

English version:  
[BioHub Cell Tracking Working Note 1: Learned Lineage Graphs and Metric-Aware Repair](https://pilkwangkim.github.io/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/)

관련 공개 노트북:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

이 글은 BioHub cell tracking competition에 대한 첫 번째 working note다.
목표는 단순히 제출 노트북 하나를 설명하는 것이 아니라, 이후 실험이 덜 랜덤해지도록 문제 구조를 정리하는 것이다.

핵심은 다음 한 줄이다.

```text
이 문제는 단순한 3D segmentation 문제가 아니라,
metric이 강하게 정의된 lineage graph reconstruction 문제다.
```

따라서 가장 유용한 추상화는 다음이다.

$$
G=(V,E).
$$

여기서 각 node \(v\in V\)는 특정 time point의 cell centroid이고, directed edge \(e=(u\rightarrow v)\in E\)는 시간 방향의 cell link다.
Cell division은 submission 파일에서 별도 label로 제출하는 것이 아니다.
공식 metric notes 기준으로는 하나의 parent node가 정확히 두 개의 outgoing edge를 갖는 graph topology로 나타난다.

이 글의 모든 논리는 이 관점 위에 있다.

---

## 1. Competition Background

이 대회는 zebrafish embryo의 형광 3D microscopy video에서 cell을 추적하는 문제다.
각 sample은 짧은 3D time-lapse sequence이고, 알고리즘은 매 time point마다 cell을 detect하고, 같은 biological cell을 시간축으로 연결하고, division event를 lineage graph로 복원해야 한다.

생물학적으로 중요한 이유는 명확하다.
Developmental biology에서 관심사는 한 frame의 cell 위치만이 아니다.
세포가 어떻게 이동하고, 어떻게 분열하고, 어떻게 organism을 만들어 가는지가 중요하다.
그런데 실제 3D microscopy data는 dense하고, noisy하고, anisotropic하고, cell들이 서로 매우 비슷하게 보인다.
Cell은 변형되고, 흐릿해지고, 다른 cell과 가까워지고, 어느 순간 둘로 갈라진다.
Manual tracking은 빠르게 병목이 된다.

입력 데이터도 toy video가 아니다.
대회 데이터 설명에서 중요한 contract는 다음과 같다.

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

Train sample에는 sparse lineage annotation이 GEFF 형식으로 붙는다.

```text
sample.geff/
  nodes/ids
  nodes/props/t/values
  nodes/props/z/values
  nodes/props/y/values
  nodes/props/x/values
  edges/ids
```

여기서 중요한 단어는 **sparse**다.
모든 visible cell이 모든 frame에서 label되어 있는 것이 아니다.
따라서 일반적인 dense segmentation validation만으로는 이 문제를 제대로 읽기 어렵다.
이미지상으로는 그럴듯한 predicted cell이라도 sparse ground truth node와 match되지 않을 수 있다.
이 구조 때문에 workflow는 자연스럽게 graph-level metric과 conservative post-processing 쪽으로 이동한다.

최종 submission은 두 row type을 가진 하나의 CSV다.

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6_xxxx,node,1,0,32,128,128,-1,-1
1,44b6_xxxx,node,2,1,33,130,125,-1,-1
2,44b6_xxxx,edge,-1,-1,-1,-1,-1,1,2
```

Node row는 cell detection을 정의하고, edge row는 temporal link를 정의한다.
Metric은 이 CSV를 graph로 해석한다.

---

## 2. Metric Is The Design Spec

공식 metric은 먼저 time point별로 predicted node와 ground-truth node를 physical centroid distance 기준으로 match한다.
Voxel scale이 anisotropic하므로 거리 계산은 반드시 물리 단위로 해야 한다.

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

Node match는 다음 gate 안에서만 가능하다.

$$
d_{\mu m}(i,j) \le 7.0.
$$

그 다음 edge는 양 끝 node가 모두 ground-truth node에 match되고, 그 ground-truth node들이 실제 ground-truth edge로 연결되어 있을 때만 true positive가 된다.
기본 edge score는 Jaccard다.

$$
J_{\text{edge}}
=
\frac{TP_{\text{edge}}}
{TP_{\text{edge}}+FP_{\text{edge}}+FN_{\text{edge}}}.
$$

또한 node를 과하게 예측하면 penalty가 붙는다.
단순화하면 adjusted edge score는 다음 형태다.

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

공식 metric notes에서 penalty coefficient는 \(a=0.1\)이다.

Division score는 별도로 계산된다.
Ground-truth division은 lineage graph의 fork다.
Predicted graph가 pre-split stage를 cover하고 두 daughter lineage를 모두 touch해야 한다.
최종 score는 다음 구조다.

$$
S
=
\tilde J_{\text{edge}}
+
wJ_{\text{division}},
\qquad
w=0.1.
$$

이 수식이 거의 모든 의사결정을 이끌었다.
중요한 결과는 세 가지다.

| Metric pressure | Modeling consequence |
|---|---|
| Edge Jaccard가 지배적이다 | 단순히 밝은 점을 많이 찾는 것보다 link quality가 중요하다. |
| Node overprediction에 penalty가 있다 | isolated noisy node와 짧은 fragment는 손해가 될 수 있다. |
| Division weight는 작지만 0은 아니다 | mitosis recovery는 false positive가 잘 통제될 때만 유용하다. |

그래서 notebook의 방향은 점점 다음 형태로 바뀌었다.

```text
충분한 node를 찾는다
edge를 잘 고른다
geometry가 뒷받침되는 graph error만 repair한다
true edge를 만들 가능성이 낮은 fragment는 prune한다
```

---

## 3. Data Model In Code

로컬 코드에서 가장 중요한 convention은 graph operation은 원본 voxel coordinate에서 하고, 거리만 micron으로 변환하는 것이다.

<details markdown="1">
<summary>Show snippet: physical coordinate contract</summary>

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

이 작은 contract가 꽤 많은 실수를 막아준다.
\(z\) 축은 \(x,y\) 축보다 약 네 배 거칠다.
Raw voxel index의 Euclidean distance를 그대로 쓰면 linking gate가 물리적으로 틀어질 수 있다.

---

## 4. First Baseline: Classical Centers And Nearest Links

이 단계의 data contract, EDA, classical baseline은 공개 노트북
[Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)에 정리했다.

처음 유용했던 baseline은 의도적으로 단순했다.

```text
3D frame 하나를 읽는다
-> intensity를 normalize한다
-> bright local maximum을 찾는다
-> centroid를 refine한다
-> physical NMS로 duplicate를 제거한다
-> adjacent frame을 Hungarian matching으로 link한다
-> nodes and edges를 쓴다
```

이 baseline이 이기기 위한 것은 아니었다.
대신 metric이 어떤 압력을 주는지 보여주는 역할을 했다.
Node count가 폭발하면 score가 떨어졌다.
Link gate가 넓으면 false edge가 쌓였다.
Border peak를 방치하면 one-frame noise component가 많아졌다.

중요했던 classical refinement는 다음이다.

1. **Intensity-weighted centroid refinement**

   Coarse peak 주변 local window \(W\) 안에서 좌표를 다시 계산한다.

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

2. **Physical NMS**

   Refinement 후 같은 cell 주변으로 모이는 peak는 하나만 남긴다.

   $$
   d_{\mu m}(i,j) < r_{\text{nms}}
   \Longrightarrow
   \text{keep only the stronger peak}.
   $$

3. **Count stabilizer**

   한 frame에서 threshold가 실패하면 false node가 대량으로 생긴다.
   간단한 방어식은 다음이다.

   $$
   N_t > \alpha N_{t-1}+\beta
   \Longrightarrow
   \text{keep only the strongest candidates in frame }t.
   $$

이런 것들은 화려한 trick이라기보다 metric hygiene에 가깝다.

---

## 5. Learned Graph Model: UNet, Node Transformer, ILP

더 강한 시스템은 official learned-model family를 graph backbone으로 사용한다.
개념적 pipeline은 다음과 같다.

```text
3D image volume
-> temporal UNet center detector
-> candidate cell nodes
-> node features and learned edge probabilities
-> ILP graph selection
-> output graph repair
-> submission.csv
```

Detector는 center probability field를 만든다.

$$
p_t(\mathbf r)
=
\sigma
\left(
f_\theta(X_{t-k:t+k})(\mathbf r)
\right),
$$

여기서 \(X_{t-k:t+k}\)는 frame \(t\) 주변의 짧은 temporal context다.

Candidate node는 learned edge model로 link된다.

$$
p_{ij}
=
\sigma
\left(
g_\phi(h_i,h_j,\Delta t,\Delta \mathbf r)
\right).
$$

여기서 \(h_i\)와 \(h_j\)는 learned node representation이다.
최종 graph는 edge를 독립적으로 전부 받아들이는 방식이 아니라 optimization layer로 선택한다.
Lineage graph에는 구조적 제약이 있기 때문이다.
한 node가 임의로 많은 parent를 가지면 안 되고, division-like fork는 드물고 물리적으로 그럴듯해야 한다.

### 5.1 Learned edge와 motion geometry의 결합

실제 post-link assignment에서는 learned probability만 보지 않는다.
Source node \(i\)의 직전 displacement가 있으면 다음 위치를 예측한다.

$$
\hat p_i
=
p_i+\lambda(p_i-p_{i-1}).
$$

Candidate target \(j\)의 assignment cost는 다음 형태다.

$$
C_{ij}
=
\|p_j-\hat p_i\|_2
+0.05\|p_j-p_i\|_2
-\beta q_{ij}.
$$

여기서 \(q_{ij}\)는 learned edge probability이고, \(\beta\)는 learned evidence가 geometry tie를 얼마나 강하게 풀지 정한다.
현재 anchor는 \(\lambda=0.5\), \(\beta=0.75\)를 사용한다.
Gate 밖의 pair에는 큰 cost를 주고 Hungarian assignment로 one-to-one match를 고른다.

<details markdown="1">
<summary>Show snippet: learned-edge weighted motion assignment</summary>

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

이 식은 parameter sensitivity도 설명해준다.
\(\beta\)를 너무 낮추면 model을 거의 쓰지 않는 nearest-motion tracker가 되고,
너무 높이면 calibration이 불완전한 learned score가 좋은 geometry를 덮어쓴다.

Kaggle notebook에서는 online training을 할 수 없으므로, model은 attached support dataset으로 전달한다.

```text
support_pack/
  ARTIFACT_MANIFEST.json
  repo/
  weights/
    unet_transformer/
      split_0/
        edge_predictor_best.pth
        checkpoint_last.pth
  wheels/
```

<details markdown="1">
<summary>Show snippet: artifact discovery</summary>

```python
from pathlib import Path
import json

ARTIFACT_MANIFEST = Path(
    "/kaggle/input/datasets/pilkwang/"
    "biohub-tracking-support-pack-v1/ARTIFACT_MANIFEST.json"
)

with ARTIFACT_MANIFEST.open() as f:
    manifest = json.load(f)

artifact_root = ARTIFACT_MANIFEST.parent
model_info = (
    manifest.get("models", {}).get("unet_transformer")
    or manifest["model"]
)
weight_path = artifact_root / model_info["weight_path"]
repo_dir = artifact_root / "repo"

assert weight_path.is_file(), weight_path
assert repo_dir.is_dir(), repo_dir
print("weight:", weight_path)
print("sha256:", model_info["weight_sha256"])
print("repo:", repo_dir)
```

</details>

Manifest는 단순한 메타데이터가 아니다.
Internet-disabled Kaggle run에서 notebook이 source code, weights, wheel files, coordinate convention, model role을 복원하는 contract다.

---

## 6. Why Graph Repair Became The Main Lever

Learned graph와 gap-recovery 계열의 실행 가능한 공개본은
[Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)에서 볼 수 있다.

Learned model이 충분히 강해진 뒤에는 broad detection change가 위험해졌다.
가장 유용한 변화는 graph level에서 나왔다.

```text
짧은 noisy component를 제거한다
물리적으로 그럴듯한 motion break만 relink한다
midpoint가 support될 때만 one-frame gap을 복원한다
division edge는 strict geometry 아래에서만 추가한다
node count를 불필요하게 키우는 synthetic growth를 피한다
```

가장 중요했던 repair module은 short-track pruning이었다.
Node가 몇 개 없는 connected component는 true edge를 만들지 못하면서 node count와 edge false-positive surface만 늘릴 수 있다.

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
<summary>Show snippet: short component pruning</summary>

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

이 아이디어 하나가 훨씬 복잡한 추가 모델보다 강하게 작동했다.
관측된 public setting은 극단적 pruning보다 moderate pruning 주변에 몰렸다.

| Minimum connected-component length | Public reading |
|---:|---|
| 4 | 유용하지만 short noise가 더 남는다. |
| 6 | public test에서 매우 강했다. |
| 7 | 이후 current best logic 근처의 강한 설정. |
| 12 | 아직 가능하지만 유용한 short track도 지우기 시작한다. |
| 14 | over-pruning이 보인다. |

정확한 숫자가 영원한 상수라는 뜻은 아니다.
Detector checkpoint, hidden test distribution, 나머지 repair stack에 따라 움직인다.
하지만 short-track pruning이 real signal이라는 점은 꽤 명확하다.

---

## 7. Gap Recovery And Safe Division

One-frame gap recovery는 다음 패턴을 repair하려고 한다.

```text
node at t
missing or weak node at t+1
node at t+2
```

Candidate midpoint는 다음이다.

$$
\mathbf m
=
\frac{\mathbf r_t+\mathbf r_{t+2}}{2}.
$$

Bridge는 displacement가 물리적으로 그럴듯할 때만 고려된다.

$$
d_{\mu m}(\mathbf r_t,\mathbf r_{t+2})
\le
2r_{\text{gap}}.
$$

\(\mathbf m\) 근처에 existing node가 있으면 그것을 재사용한다.
그렇지 않으면 synthetic midpoint를 만들고, image에서 local intensity-weighted centroid로 refine할 수 있다.
단, synthetic node가 너무 멀리 이동하면 버린다.

$$
d_{\mu m}(\hat{\mathbf m},\mathbf m)
\le
r_{\text{shift}}.
$$

Synthetic growth에는 전체 cap도 둔다.

$$
N_{\text{synthetic}}
\le
\min
\left(
N_{\text{abs}},
\left\lfloor \rho |V| \right\rfloor
\right).
$$

구현에서 중요한 부분은 repair가 reject될 때 graph state를 원상복구하는 것이다.
Node만 남기고 edge를 취소하면 isolated-node penalty를 만들 수 있다.

<details markdown="1">
<summary>Show snippet: one-frame gap insertion with rollback</summary>

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

Division recovery는 더 보수적이다.
Parent \(p_t\), existing child \(c^{(1)}_{t+1}\), candidate second child \(c^{(2)}_{t+1}\)가 있을 때, 다음 조건을 만족할 때만 edge를 추가한다.

$$
d_{\mu m}(p,c^{(2)})\le r_{\text{parent}},
$$

$$
d_{\mu m}(c^{(1)},c^{(2)})\le r_{\text{sister}},
$$

그리고 candidate child에는 이미 parent가 없어야 한다.

이유는 metric pressure다.
False-positive division edge는 division term뿐 아니라 edge term도 손상시킬 수 있다.
Division recall은 edit의 precision이 높을 때만 이득이다.

---

## 8. Training And Artifact Workflow

실제 workflow는 두 층으로 나뉘었다.

```text
local or remote training
-> checkpoint snapshots
-> support pack build
-> Kaggle dataset upload
-> notebook commit
-> public LB read
```

Trainer는 의도적으로 strict하게 만들었다.
Continue mode에서는 반드시 기존 checkpoint가 있어야 한다.
Fresh start는 명시적으로 허용해야 한다.
긴 학습을 돌린 뒤 실수로 epoch 1부터 다시 시작하는 것만큼 허탈한 일도 별로 없기 때문이다.

<details markdown="1">
<summary>Show snippet: strict resume command</summary>

```bash
cd /workspace/26_Biohub

# Check the registered method, checkpoint, and scratch layout first.
bash runpod_5090/scripts/biohub_runctl.sh inventory
bash runpod_5090/scripts/biohub_runctl.sh doctor anchor400_legacy

# Continue is fail-closed: a missing checkpoint is an error, not a fresh run.
bash runpod_5090/scripts/biohub_runctl.sh \
  start anchor400_legacy continue \
  --epochs 500
```

</details>

Packaging도 같은 contract를 따른다.

<details markdown="1">
<summary>Show snippet: support pack version upload</summary>

```bash
cd /workspace/26_Biohub

BIOHUB_TRAIN_METHOD=unet_transformer_400ep_snapshot_v1 \
BIOHUB_ARTIFACT_TAG=400ep-snapshot-v1 \
BIOHUB_DATASET_SLUG=biohub-tracking-support-pack-v1 \
BIOHUB_DATASET_VERSION_MESSAGE="Update learned graph checkpoint snapshot" \
bash runpod_5090/scripts/package_and_upload_support_pack_version.sh
```

</details>

Checkpoint path만 확인해서는 충분하지 않다.
같은 filename 아래 dataset version이 바뀔 수 있으므로 epoch와 SHA를 같이 검증한다.

<details markdown="1">
<summary>Show snippet: fail-closed checkpoint identity</summary>

```python
import hashlib
import torch

def sha256_file(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

expected_epoch = 400
actual_epoch = int(checkpoint.get("epoch", -1))
if actual_epoch != expected_epoch:
    raise ValueError(
        f"expected checkpoint epoch {expected_epoch}, got {actual_epoch}"
    )

actual_sha = sha256_file(checkpoint_path)
expected_sha = manifest["model"]["weight_sha256"]
if actual_sha != expected_sha:
    raise ValueError(f"checkpoint SHA mismatch: {actual_sha}")
```

</details>

Public notebook이나 blog에서는 hardware 자체를 이야기의 중심으로 두지 않는 것이 좋다.
중요한 것은 artifact contract다.

```text
weights are trained outside the notebook
the notebook is deterministic given the attached dataset
the manifest records what the notebook is actually using
```

현재 calibrated UNET400 anchor의 `edge_predictor_best.pth` SHA-256은 다음과 같다.

```text
12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771
```

Weight filename보다 이 hash가 실험 lineage를 더 정확하게 식별한다.

---

## 9. Score Progression And What It Taught Me

Public leaderboard 숫자는 private validation이 아니다.
그래도 한 번에 하나의 구조적 가설만 바꾸면 noisy diagnostic으로 쓸 수 있었다.
2026년 7월 12일까지 확인한 public progression은 대략 다음과 같다.

| Stage | Public LB reading | What changed |
|---|---:|---|
| Classical and early lineage baselines | 0.68--0.75 | schema, physical distance, graph output을 검증했다. |
| Strong rule-based geometry | 0.82--0.86 | model artifact 없이도 conservative topology가 강했다. |
| First learned artifact reproduction | about 0.81 | learned model을 재현하는 것만으로는 충분하지 않았다. |
| Learned graph plus graph repair | 0.844--0.860 | motion relink, gap repair, pruning이 실질적인 상승을 만들었다. |
| Checkpoint-specific graph calibration | 0.885--0.897 | checkpoint와 post-processing을 하나의 system으로 맞췄다. |
| UNET400, spatial TTA, min-track 6 anchor | 0.900 | broad recall보다 calibrated graph가 강했다. |
| Conditional UNET400 + Center400 gap confirmation | **0.901** | auxiliary model이 불확실한 edit만 확인할 때 처음으로 이득이 났다. |

### 9.1 Epoch는 독립적인 성능 축이 아니었다

125, 200, 250, 300, 400 epoch를 비교하면서 가장 중요했던 결과는
epoch를 늘리는 것 자체가 같은 post-processing 아래에서 score를 자동으로 올리지 않는다는 점이었다.
Later checkpoint는 internal loss가 더 낮더라도 detection과 edge probability의 calibration이 달라진다.
그래서 checkpoint는 다음 묶음으로 읽어야 한다.

```text
model weights
+ detector threshold
+ TTA contract
+ motion/edge assignment
+ repair caps
+ pruning length
```

Weights alone이 아니다.
실제로 detector threshold를 \(0.9700\) 주변에서 \(0.9675\), \(0.9725\)로 움직인 두 제출은 모두 \(0.899\)였다.
이 결과는 그 축의 local optimum이 사실상 닫혔음을 보여줬다.

### 9.2 300ep와 400ep의 error anatomy

같은 199개 training movie에 대해 300ep와 400ep prediction을 비교한 결과는 다음과 같았다.

| Run | Edge TP | Edge FP | Edge FN | Global edge J | Mean score proxy |
|---|---:|---:|---:|---:|---:|
| UNET300 | 121,669 | 5,212 | 7,214 | 0.907334 | 0.902110 |
| UNET400 | 122,151 | 5,202 | 6,732 | 0.910997 | 0.912574 |

400ep는 TP를 482개 늘리고 FN을 482개 줄이면서 FP도 10개 줄였다.
Sample 단위로는 101개가 좋아지고 86개가 나빠졌다.
즉 400ep는 단순히 더 confident한 같은 모델이 아니었다.
평균적으로는 더 좋지만 error distribution이 바뀌었고, 그 때문에 300ep용 threshold를 그대로 옮기면 public score가 오히려 떨어질 수 있었다.

여기서 주의할 점이 있다.
이 분석은 all-train checkpoint를 training data에 다시 적용한 **in-sample error anatomy**다.
Fold-held-out prediction이 아니므로 true OOF라고 부르면 안 된다.
이 구분은 이후 repair policy를 설계할 때 중요해졌다.

---

## 10. What Did Not Work Yet

실패한 실험은 어떤 model signal에 어느 정도의 권한을 줘야 하는지 알려줬다.

### 10.1 Broad recall과 aggressive TTA

Intensity TTA와 aggressive detection expansion은 node count를 늘렸지만 최고 graph를 넘지 못했다.
한 intensity-TTA 계열은 \(0.894\)까지 떨어졌다.
Spatial flip과 XY rotation을 사용하는 6-view TTA는 유용했지만, 더 많은 view가 자동으로 더 좋은 것은 아니었다.

### 10.2 이름이 `edge_predictor`라고 edge head만 담긴 것은 아니다

가장 큰 실패 중 하나는 별도 `edge_predictor_best.pth`를 기존 graph에 단순 교체한 실험이었다.
파일 이름 때문에 edge scorer만 바뀐다고 생각하기 쉽지만, checkpoint에는 TemporalUNet detector를 포함한 전체 model state가 들어 있었다.

| Output | Calibrated UNET400 anchor | Uncalibrated checkpoint swap |
|---|---:|---:|
| Node rows | 128,535 | 170,860 |
| Edge rows | 123,988 | 164,603 |
| Public score | about 0.900 | 0.861 |

Node와 edge가 약 33% 폭증했다.
새 checkpoint에 기존 \(0.97\) detector threshold와 config를 그대로 적용했기 때문이다.
이 결과는 독립 seed model을 weight swap으로 제출하면 안 되고, calibration 이후 output-level consensus나 disagreement gate로 결합해야 한다는 것을 보여줬다.

### 10.3 작은 ILP 변경도 독립적인 실험이어야 한다

Division weight를 \(1.0\)에서 \(0.7\)로 내린 변형은 \(0.897\)이었다.
같은 notebook에 `pool_kernel_um=2.0` 패치도 있었지만 prediction이 끝난 다음 적용되어 실제 제출에는 영향을 주지 않았다.
여러 변경을 한 셀에 넣으면 실행 순서 때문에 무엇을 시험했는지조차 흐려질 수 있다.

### 10.4 Center를 global detector처럼 쓰는 것은 여전히 위험하다

DeepCenterUNet3D를 blind union이나 모든 synthetic node의 hard gate로 쓰면 이득이 없었다.
특히 모든 synthetic gap node에 Center confidence를 요구한 실험은 \(0.898\)로 떨어졌다.
실제로 누락된 cell은 흐리거나 가려졌을 가능성이 높다.
TemporalUNet이 놓친 같은 frame에서 Center도 낮은 score를 내는 correlated false negative가 생긴다.

또 하나의 artifact trap도 있었다.
100--500ep snapshot의 `best.pt`는 모두 같은 early best checkpoint를 가리켰다.
특정 epoch의 Center를 시험하려면 `checkpoint_last.pt`를 사용하고 checkpoint 안의 epoch를 assert해야 했다.

---

## 11. When The Auxiliary Center Model Finally Helped

Center model이 처음으로 이득을 만든 것은 global detector로 합쳤을 때가 아니었다.
UNET400 graph가 이미 제안한 **거리상 애매한 one-frame gap**에만 veto 권한을 줬을 때였다.
이 계열은 공개 노트북
[Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)에 이어서 발전시켰다.

기본 gap proposal은 \(t\)의 track end와 \(t+2\)의 track start 사이에 midpoint를 둔다.

$$
\tilde p_{t+1}=\frac{p_t+p_{t+2}}{2}.
$$

UNET400 + Center400의 성공한 규칙은 다음이었다.

$$
\operatorname{accept}(d,c)=
\begin{cases}
1, & d<8\ \mu\mathrm{m},\\
\mathbf{1}[c(\tilde p_{t+1})\ge0.20], & 8\le d\le12\ \mu\mathrm{m},\\
0, & d>12\ \mu\mathrm{m}.
\end{cases}
$$

여기서 \(d=\|p_{t+2}-p_t\|_2\)이고 \(c\)는 proposed midpoint 주변의 DeepCenter probability다.
이 구성은 \(0.901\)을 기록했다.
반대로 모든 synthetic gap node에 \(c\ge0.15\)를 요구한 구성은 \(0.898\)이었다.

이 차이가 중요한 이유는 signal hierarchy를 보여주기 때문이다.

```text
strong temporal geometry > weak negative Center evidence
marginal temporal geometry + positive Center evidence > geometry alone
```

즉 Center score가 낮다는 사실은 cell이 없다는 강한 증거가 아니다.
하지만 motion geometry가 이미 약한 곳에서 Center score가 높다면 유용한 독립 증거가 된다.

<details markdown="1">
<summary>Show snippet: conditional Center confirmation</summary>

```python
def accept_gap(span_um, center_prob):
    if span_um < 8.0:
        return True
    if span_um > 12.0:
        return False
    return center_prob >= 0.20
```

</details>

Flat threshold를 거리별로 일반화하면 다음과 같다.

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

양방향 temporal context가 있으면 forward와 backward midpoint도 비교할 수 있다.

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

\(e_{\text{cons}}\le2.5\,\mu\mathrm{m}\)이면 image-space veto보다 temporal consensus를 우선한다.

<details markdown="1">
<summary>Show snippet: adaptive Center gate and bidirectional consensus</summary>

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

이 결과에서 파생되는 다음 실험도 global threshold search와는 다르다.

1. 거리가 멀어질수록 Center threshold를 \(0.12\)에서 \(0.28\)까지 올리는 distance-adaptive gate
2. 앞·뒤 velocity prediction이 합의하면 Center veto를 우회하는 bidirectional motion gate
3. marginal gap을 synthetic node와 observed isolated-node reuse로 분해하는 causal split

공통 원칙은 auxiliary model의 권한을 넓히는 것이 아니라, **불확실성의 종류에 따라 권한을 제한하는 것**이다.

---

## 12. Model Diversity, True OOF, And Current Direction

두 번째 model에는 서로 다른 목적이 있다.
독립 seed로 전체 data를 학습한 model은 test-time disagreement와 ensemble에 유용하다.
하지만 그 prediction은 training sample에 대해 OOF가 아니다.
Repair policy를 학습하려면 fold-held-out model이 따로 필요하다.

```text
all-train independent seed
-> test-time consensus and disagreement features

two-fold held-out models
-> true OOF graph edits and repair-policy labels
```

<details markdown="1">
<summary>Show snippet: true two-fold OOF capture contract</summary>

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

Score blend를 쓴다면 model이 충분히 성숙한 뒤 다음처럼 anchor를 우선하는 형태가 안전하다.

$$
p_{\mathrm{blend}}
=
\alpha p_{\mathrm{anchor}}
+
(1-\alpha)p_{\mathrm{seed}},
\qquad
\alpha\in[0.65,0.75].
$$

하지만 목표는 평균 자체가 아니다.
두 model이 같은 edge에 동의하는지, 어느 상황에서 서로 다른 오류를 내는지가 더 중요하다.

True OOF repair table은 다음 형태가 된다.

| Proposal type | Label source |
|---|---|
| existing edge | matched GT nodes가 같은 edge를 갖는지 |
| one-frame gap bridge | GT에 length-2 path가 있는지 |
| division edge | GT에 같은 fork가 있는지 |
| short-component keep/drop | component가 matched true edge를 만드는지 |

각 candidate edge나 repair action의 feature vector는 raw pixel 전체가 아니라 다음처럼 metric에 가까운 값으로 구성할 수 있다.

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

여기서 \(c_{ij}\)는 optional Center evidence이고, \(\rho\)는 local density다.
Compact policy는 예를 들어 다음 확률을 내보낸다.

$$
P(y_{ij}=1\mid x_{ij})
=
\sigma(w^\top x_{ij}+b).
$$

<details markdown="1">
<summary>Show snippet: exported repair-policy contract</summary>

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

최종적으로 묻고 싶은 질문은 다음이다.

```text
Which proposed graph edits are metric-positive,
given geometry, model disagreement, and local image evidence?
```

현재 방향을 요약하면 다음과 같다.

```text
UNET400 calibrated graph를 anchor로 유지한다
Center는 marginal repair confirmation에만 사용한다
독립 seed는 output-level disagreement를 위해 학습한다
two-fold models로 true OOF repair actions를 만든다
checkpoint, config, split, and SHA를 하나의 artifact contract로 기록한다
```

---

## Appendix: Submission Audit Snippet

최종 output은 결국 CSV 하나다.
제출 전에는 schema audit routine을 작게 유지한다.

<details markdown="1">
<summary>Show snippet: submission schema audit</summary>

```python
REQUIRED_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]

def audit_submission(df, expected_datasets):
    assert list(df.columns) == REQUIRED_COLUMNS
    assert df["id"].tolist() == list(range(len(df)))
    assert set(df["row_type"]).issubset({"node", "edge"})
    assert set(expected_datasets).issubset(set(df["dataset"]))

    node_rows = df["row_type"].eq("node")
    edge_rows = df["row_type"].eq("edge")

    assert (df.loc[node_rows, ["source_id", "target_id"]] == -1).all().all()
    assert (df.loc[edge_rows, ["node_id", "t", "z", "y", "x"]] == -1).all().all()

    for dataset, part in df.groupby("dataset"):
        node_ids = set(part.loc[part["row_type"].eq("node"), "node_id"])
        for source_id, target_id in part.loc[part["row_type"].eq("edge"), ["source_id", "target_id"]].itertuples(index=False):
            assert source_id in node_ids, (dataset, source_id)
            assert target_id in node_ids, (dataset, target_id)
```

</details>

이 코드는 화려하지 않지만, 피할 수 있는 submission failure를 막는다.
Graph competition에서 지루한 validation도 모델의 일부다.
