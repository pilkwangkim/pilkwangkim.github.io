---
title: "BioHub Cell Tracking Working Note 1: Learned Lineage Graph와 Metric-Aware Repair"
date: 2026-07-08 21:00:00 +0900
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

weight_path = ARTIFACT_MANIFEST.parent / manifest["models"]["unet_transformer"]["weight_path"]
repo_dir = ARTIFACT_MANIFEST.parent / manifest["contents"]["repo"]

print(weight_path)
print(repo_dir)
```

</details>

Manifest는 단순한 메타데이터가 아니다.
Internet-disabled Kaggle run에서 notebook이 source code, weights, wheel files, coordinate convention, model role을 복원하는 contract다.

---

## 6. Why Graph Repair Became The Main Lever

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

BIOHUB_RUN_MODE=continue \
BIOHUB_SCRATCH_ROOT=/tmp/biohub_scratch \
BIOHUB_TRAIN_METHOD=unet_transformer_50ep_v1 \
BIOHUB_EPOCHS=400 \
BIOHUB_RESUME=1 \
BIOHUB_OVERWRITE=0 \
bash runpod_5090/scripts/run_5090_50epoch_oneshot.sh
```

</details>

Packaging도 같은 contract를 따른다.

<details markdown="1">
<summary>Show snippet: support pack version upload</summary>

```bash
cd /workspace/26_Biohub

BIOHUB_TRAIN_METHOD=unet_transformer_50ep_v1 \
BIOHUB_ARTIFACT_TAG=400ep-snapshot-v1 \
BIOHUB_DATASET_SLUG=biohub-tracking-support-pack-v1 \
BIOHUB_DATASET_VERSION_MESSAGE="Update learned graph checkpoint snapshot" \
bash runpod_5090/scripts/package_and_upload_support_pack_version.sh
```

</details>

Public notebook이나 blog에서는 hardware 자체를 이야기의 중심으로 두지 않는 것이 좋다.
중요한 것은 artifact contract다.

```text
weights are trained outside the notebook
the notebook is deterministic given the attached dataset
the manifest records what the notebook is actually using
```

---

## 9. Score Progression And What It Taught Me

Public leaderboard 숫자는 private validation이 아니다.
그래도 noisy diagnostic으로는 쓸모가 있었다.

| Stage | Rough public reading | Lesson |
|---|---:|---|
| Classical peak/link baseline | 0.6 to 0.7 range | submission schema와 metric이 제대로 작동하는지 확인했다. |
| Strong rule-based public family | 0.82 to 0.86 range | careful geometry만으로도 꽤 멀리 갈 수 있다. |
| Learned graph artifact | 0.88+ range | edge-aware learning이 강한 backbone이다. |
| Short-track pruning and repair tuning | around 0.89+ | metric-aware graph cleanup이 핵심 lever다. |
| Hard auxiliary center gate | not clearly helpful | second detector는 blind union보다 soft feature로 써야 한다. |

가장 중요한 의외의 결과는 epoch를 늘리는 것 자체가 같은 post-processing 아래에서 자동으로 score를 올리지 않았다는 점이다.
Model checkpoint와 repair threshold는 하나의 pair로 봐야 한다.
나중 checkpoint가 internal loss는 더 좋더라도 detection score와 edge score의 분포가 달라져서 기존 threshold가 빗나갈 수 있다.

그래서 이제 checkpoint는 다음 묶음으로 읽는다.

```text
model weights + detector threshold + edge threshold + repair caps + pruning length
```

Weights alone이 아니다.

---

## 10. What Did Not Work Yet

실패한 실험도 유용했다.

첫째, broad recall expansion은 보기보다 자주 손해였다.
추가 node가 true edge로 이어지면 도움이 되지만, isolated detection은 대부분 metric debt다.

둘째, division recovery는 민감하다.
Division term의 weight는 작고, false-positive fork는 edge error도 만들 수 있다.
좋은 division strategy는 단순히 division을 켜는 것이 아니다.
Geometry가 지루할 정도로 안정적인 fork만 복원하는 것이다.

셋째, auxiliary full-frame center detector는 hard gate만으로는 아직 충분히 유용하지 않았다.
안전한 contract는 다음에 가깝다.

$$
\text{add center } c
\iff
s(c)\ge\tau
\land
d(c,\mathcal N_{\text{existing}})\ge r_{\text{dup}}
\land
\text{graph evidence}(c)=1
\land
\text{frame cap is not exceeded}.
$$

실제로는 이것도 너무 blunt할 수 있다.
더 나은 다음 단계는 auxiliary detector를 repair policy의 feature로 쓰는 것이다.

$$
p_i
=
P(y_i=1\mid x_i,a_i),
$$

여기서 \(a_i\)는 candidate graph edit이고, \(x_i\)는 geometry, model score, local density, track length, auxiliary center evidence 등을 포함한다.

---

## 11. Current Direction

현재 방향은 다음과 같다.

```text
learned graph backbone은 유지한다
short-track pruning signal은 유지한다
checkpoint family별 threshold를 calibrate한다
auxiliary detector는 global union이 아니라 repair feature로 쓴다
sparse ground truth로 OOF-labeled repair action을 만든다
```

다음 글은 아마 이 repair-policy layer를 다루는 것이 좋을 것이다.
자연스러운 offline dataset은 raw pixel table이 아니라 proposed graph edit table이다.

| Proposal type | Label source |
|---|---|
| existing edge | matched GT nodes가 같은 edge를 갖는지 |
| one-frame gap bridge | GT에 length-2 path가 있는지 |
| division edge | GT에 같은 fork가 있는지 |
| short-component keep/drop | component가 matched true edge를 만드는지 |

그 다음 notebook은 hand-tuned repair threshold 대신 compact exported policy를 사용할 수 있다.
이 방향이 더 맞는 이유는 남은 ambiguity를 직접 묻기 때문이다.

```text
Which proposed graph edits are metric-positive?
```

이 질문은 다음 질문보다 낫다.

```text
오늘 public LB에서 맞아떨어지는 global threshold를 하나 더 찾을 수 있을까?
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
