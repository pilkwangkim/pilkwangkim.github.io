---
title: "BioHub Cell Tracking Working Note 1: Learned Lineage Graphs and Metric-Aware Repair"
date: 2026-07-11 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, unet, ilp, graph-repair, working-note]
math: true
pin: false
---

# BioHub Cell Tracking Working Note 1: Learned Lineage Graphs and Metric-Aware Repair

Competition link:  
[BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)

Official metric notes:  
[RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)

Background note:  
[Biohub Calls on AI Community to Transform 3D Cell Tracking](https://network.febs.org/posts/biohub-calls-on-ai-community-to-transform-3d-cell-tracking)

Korean version:  
[BioHub Cell Tracking Working Note 1: Learned Lineage Graph와 Metric-Aware Repair](https://pilkwangkim.github.io/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs-KR/)

Related public notebooks:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

This is the first working note for the BioHub cell tracking competition.
The goal is not only to describe a submission notebook, but to organize the problem in a way that makes the next experiments less random.

The short version is:

```text
This is not just a 3D segmentation problem.
It is a lineage graph reconstruction problem under a very specific metric.
```

The useful abstraction is therefore:

$$
G=(V,E),
$$

where each node \(v\in V\) is a detected cell centroid at one time point, and each directed edge \(e=(u\rightarrow v)\in E\) is a temporal link between cells.
A division event is not a separate label in the submission file. In the official metric notes, it appears as graph topology: one parent node with exactly two outgoing edges.

Everything that follows is built around that view.

---

## 1. Competition Background

The competition asks participants to track fluorescently labeled cells in zebrafish embryo microscopy videos.
Each sample is a short 3D time-lapse sequence.
At every time point, the algorithm must detect cells in a 3D volume, link the same biological cell across time, and identify division events by reconstructing the lineage graph.

That is biologically meaningful because developmental biology is a process problem.
Researchers do not only want to know where cells are in one frame.
They want to know how cells move, split, and build an organism over time.
The hard part is that 3D microscopy data is dense, noisy, anisotropic, and visually repetitive.
Many cells look alike.
Cells may deform, disappear into dim regions, or divide into two daughters.
Manual tracking becomes a bottleneck quickly.

The input data is a real-world microscopy benchmark, not a toy video task.
The public data description gives the main contract:

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

Training samples also include sparse lineage annotations in GEFF format:

```text
sample.geff/
  nodes/ids
  nodes/props/t/values
  nodes/props/z/values
  nodes/props/y/values
  nodes/props/x/values
  edges/ids
```

The important word is **sparse**.
The annotations do not label every visible cell in every frame.
That means ordinary dense segmentation validation can be misleading.
A predicted cell can look reasonable in the image but still not correspond to a sparse ground-truth node.
This pushes the workflow toward graph-level metrics and conservative post-processing.

The final submission is a single CSV with two row types:

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6_xxxx,node,1,0,32,128,128,-1,-1
1,44b6_xxxx,node,2,1,33,130,125,-1,-1
2,44b6_xxxx,edge,-1,-1,-1,-1,-1,1,2
```

Node rows define cell detections.
Edge rows define temporal links.
The metric then interprets this CSV as a graph.

---

## 2. The Metric Is The Design Spec

The official metric first matches predicted and ground-truth nodes per time point using physical centroid distance.
The anisotropic voxel scale matters:

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

A node can match only if this physical distance is within the competition matching gate:

$$
d_{\mu m}(i,j) \le 7.0.
$$

After node matching, an edge is correct only when both endpoints match ground-truth nodes that are connected by a ground-truth edge.
The basic edge score is Jaccard:

$$
J_{\text{edge}}
=
\frac{TP_{\text{edge}}}
{TP_{\text{edge}}+FP_{\text{edge}}+FN_{\text{edge}}}.
$$

There is also a penalty for over-predicting the number of nodes.
In simplified form, the adjusted edge score is:

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
\right),
$$

with penalty coefficient \(a=0.1\) in the official metric notes.

Division scoring is separate.
A ground-truth division is a fork in the lineage graph.
The predicted graph must cover the pre-split stage and touch both daughter lineages.
The final score has the form:

$$
S
=
\tilde J_{\text{edge}}
+
wJ_{\text{division}},
\qquad
w=0.1.
$$

This equation shaped almost every decision in my notebook.
Three consequences matter:

| Metric pressure | Modeling consequence |
|---|---|
| Edge Jaccard dominates | Link quality is more important than merely detecting many bright spots. |
| Node overprediction is penalized | Isolated noisy nodes and short fragments can hurt. |
| Division has low but nonzero weight | Mitosis recovery is useful only when false positives are tightly controlled. |

This is why the notebook gradually moved away from "detect as much as possible" and toward:

```text
detect enough nodes
link them well
repair graph errors only when geometry supports the edit
prune fragments that are unlikely to produce true edges
```

---

## 3. The Data Model In Code

The most important local convention is to keep graph operations in original voxel coordinates and convert only distances to microns.

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

This small contract prevents a surprisingly common error.
The \(z\) axis is about four times coarser than \(x\) and \(y\).
Treating voxel distance as Euclidean distance in raw index space can make the linking gate physically wrong.

---

## 4. First Baseline: Classical Centers And Nearest Links

The data contract, EDA, and classical baseline from this stage are available in
[Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline).

The first useful baseline was intentionally simple:

```text
read one 3D frame
-> normalize intensity
-> find bright local maxima
-> refine centroids
-> remove duplicates by physical NMS
-> link adjacent frames by Hungarian matching
-> write nodes and edges
```

The baseline was not expected to win.
It was useful because it made the metric visible.
When node count exploded, the score fell.
When links were too wide, false edges accumulated.
When border peaks were left unchecked, many components became one-frame noise.

The key classical refinements were:

1. **Intensity-weighted centroid refinement**

   For a coarse peak, refine its coordinate inside a local window \(W\):

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

   Two peaks that collapse to the same physical neighborhood should not both survive:

   $$
   d_{\mu m}(i,j) < r_{\text{nms}}
   \Longrightarrow
   \text{keep only the stronger peak}.
   $$

3. **Count stabilizer**

   A threshold failure in one frame can create a burst of false nodes.
   A simple guard is:

   $$
   N_t > \alpha N_{t-1}+\beta
   \Longrightarrow
   \text{keep only the strongest candidates in frame }t.
   $$

These are not glamorous tricks.
They are metric hygiene.

---

## 5. Learned Graph Model: UNet, Node Transformer, ILP

The stronger system uses the official learned-model family as the graph backbone.
The conceptual pipeline is:

```text
3D image volume
-> temporal UNet center detector
-> candidate cell nodes
-> node features and learned edge probabilities
-> ILP graph selection
-> output graph repair
-> submission.csv
```

The detector produces a center probability field:

$$
p_t(\mathbf r)
=
\sigma
\left(
f_\theta(X_{t-k:t+k})(\mathbf r)
\right),
$$

where \(X_{t-k:t+k}\) is a short temporal context around frame \(t\).

Candidate nodes are then linked by a learned edge model:

$$
p_{ij}
=
\sigma
\left(
g_\phi(h_i,h_j,\Delta t,\Delta \mathbf r)
\right).
$$

Here \(h_i\) and \(h_j\) are learned node representations.
The final graph is selected with an optimization layer rather than by accepting every edge independently.
That matters because a lineage graph has structural constraints.
A node should not have arbitrary many parents.
Division-like forks should be rare and physically plausible.

### 5.1 Combining Learned Edges With Motion Geometry

The post-link assignment does not use learned probability alone.
When the previous displacement of source node \(i\) is available, its next
position is predicted as:

$$
\hat p_i
=
p_i+\lambda(p_i-p_{i-1}).
$$

The assignment cost for candidate target \(j\) is:

$$
C_{ij}
=
\|p_j-\hat p_i\|_2
+0.05\|p_j-p_i\|_2
-\beta q_{ij}.
$$

Here \(q_{ij}\) is the learned edge probability and \(\beta\) determines how strongly learned evidence resolves a geometric tie.
The current anchor uses \(\lambda=0.5\) and \(\beta=0.75\).
Pairs outside the physical gate receive a large cost, and Hungarian assignment selects a one-to-one matching.

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

This equation also explains the parameter sensitivity.
If \(\beta\) is too small, the system becomes almost a nearest-motion tracker.
If it is too large, an imperfectly calibrated learned score can override good geometry.

The Kaggle notebook cannot train online, so the model is shipped as an attached support dataset:

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

The manifest is not just bookkeeping.
It is the contract that lets the notebook recover source code, weights, wheel files, coordinate conventions, and model roles in an internet-disabled Kaggle run.

---

## 6. Why Graph Repair Became The Main Lever

An executable public version of the learned graph and gap-recovery family is
available in [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery).

Once the learned model became strong enough, broad detection changes became risky.
The most useful changes moved to the graph level:

```text
remove short noisy components
relink physically plausible motion breaks
recover one-frame gaps only when the midpoint is supported
add division edges only under strict geometry
avoid synthetic growth that inflates node count
```

The most important repair module was short-track pruning.
A connected component with only a few nodes often contributes no true edges but still increases the node count and edge false-positive surface.

The rule can be written as:

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

In code, the shape is simple:

<details markdown="1">
<summary>Show snippet: short component pruning</summary>

```python
def keep_component(component_nodes, component_edges, min_track_len=7):
    if len(component_nodes) >= min_track_len:
        return True

    # Do not remove a small component if it contains a plausible fork.
    out_degree = {}
    for source_id, target_id in component_edges:
        out_degree[source_id] = out_degree.get(source_id, 0) + 1

    has_fork = any(deg >= 2 for deg in out_degree.values())
    return has_fork
```

</details>

This one idea was stronger than many more elaborate additions.
The best observed public settings clustered around moderate pruning rather than extreme pruning:

| Minimum connected-component length | Public reading |
|---:|---|
| 4 | Useful, but leaves more short noise. |
| 6 | Very strong in public tests. |
| 7 | A strong later setting around the current best logic. |
| 12 | Still viable, but begins to remove useful short tracks. |
| 14 | Over-pruning becomes visible. |

The exact number is not universal.
It depends on the detector checkpoint, the hidden test distribution, and the rest of the repair stack.
The robust conclusion is that short-track pruning is a real signal.

---

## 7. Gap Recovery And Safe Division

One-frame gap recovery tries to repair this pattern:

```text
node at t
missing or weak node at t+1
node at t+2
```

The candidate midpoint is:

$$
\mathbf m
=
\frac{\mathbf r_t+\mathbf r_{t+2}}{2}.
$$

The bridge is considered only if the displacement is physically plausible:

$$
d_{\mu m}(\mathbf r_t,\mathbf r_{t+2})
\le
2r_{\text{gap}}.
$$

If an existing node is close to \(\mathbf m\), the notebook reuses it.
Otherwise, it may create a synthetic midpoint and refine it from the image with a local intensity-weighted centroid.
The synthetic shift is capped:

$$
d_{\mu m}(\hat{\mathbf m},\mathbf m)
\le
r_{\text{shift}}.
$$

Synthetic growth is also capped globally:

$$
N_{\text{synthetic}}
\le
\min
\left(
N_{\text{abs}},
\left\lfloor \rho |V| \right\rfloor
\right).
$$

The important implementation detail is transactional rollback.
If a repair is rejected, leaving its node behind without edges creates exactly
the kind of isolated-node debt the metric penalizes.

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

Division recovery is even more conservative.
For a parent \(p_t\), an existing child \(c^{(1)}_{t+1}\), and a candidate second child \(c^{(2)}_{t+1}\), the edit is allowed only when:

$$
d_{\mu m}(p,c^{(2)})\le r_{\text{parent}},
$$

$$
d_{\mu m}(c^{(1)},c^{(2)})\le r_{\text{sister}},
$$

and the candidate child has no existing parent.

The reason is metric pressure.
False-positive division edges can damage the division term and also create bad temporal edges.
Division recall is useful only when the edit is high precision.

---

## 8. Training And Artifact Workflow

The practical workflow had two parts:

```text
local or remote training
-> checkpoint snapshots
-> support pack build
-> Kaggle dataset upload
-> notebook commit
-> public LB read
```

The trainer was made deliberately strict.
Resume mode must find an existing checkpoint.
Fresh-start mode must be explicitly allowed.
That rule exists because accidentally restarting from epoch 1 after a long run is one of the least fun ways to spend a night.

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

Packaging follows the same contract:

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

A checkpoint path alone is not enough.
The dataset version can change behind the same filename, so both epoch and SHA
should be checked.

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

In the public notebook text, I avoid making the hardware the story.
The story is the artifact contract:

```text
weights are trained outside the notebook
the notebook is deterministic given the attached dataset
the manifest records what the notebook is actually using
```

The SHA-256 of the calibrated UNET400 anchor's
`edge_predictor_best.pth` is:

```text
12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771
```

This hash identifies the experimental lineage more reliably than the weight filename.

---

## 9. Score Progression And What It Taught Me

The exact public leaderboard numbers are not private validation.
They were still useful as a noisy diagnostic when each submission changed one structural hypothesis.
As of July 12, 2026, the progression in this project was roughly:

| Stage | Public LB reading | What changed |
|---|---:|---|
| Classical and early lineage baselines | 0.68--0.75 | Validated the schema, physical distances, and graph output. |
| Strong rule-based geometry | 0.82--0.86 | Conservative topology went surprisingly far without a model artifact. |
| First learned artifact reproduction | about 0.81 | Reproducing a learned model was not enough by itself. |
| Learned graph plus graph repair | 0.844--0.860 | Motion relinking, gap repair, and pruning produced real gains. |
| Checkpoint-specific graph calibration | 0.885--0.897 | Treated the checkpoint and post-processing as one system. |
| UNET400, spatial TTA, min-track 6 anchor | 0.900 | A calibrated graph beat broad recall expansion. |
| Conditional UNET400 + Center400 gap confirmation | **0.901** | The auxiliary model helped for the first time when it checked only uncertain edits. |

### 9.1 Epoch Count Was Not An Independent Performance Axis

Comparing checkpoints around 125, 200, 250, 300, and 400 epochs produced an important result:
more epochs did not automatically improve the public score under fixed post-processing.
A later checkpoint can have a lower internal loss while shifting detection and edge-probability calibration.
I therefore treat a checkpoint as:

```text
model weights
+ detector threshold
+ TTA contract
+ motion/edge assignment
+ repair caps
+ pruning length
```

not as weights alone.
For example, moving the detector threshold from the \(0.9700\) anchor to \(0.9675\) and \(0.9725\) produced \(0.899\) on both sides.
That effectively closed the local detector-threshold axis.

### 9.2 Error Anatomy: 300ep Versus 400ep

Running the 300- and 400-epoch models over the same 199 training movies gave:

| Run | Edge TP | Edge FP | Edge FN | Global edge J | Mean score proxy |
|---|---:|---:|---:|---:|---:|
| UNET300 | 121,669 | 5,212 | 7,214 | 0.907334 | 0.902110 |
| UNET400 | 122,151 | 5,202 | 6,732 | 0.910997 | 0.912574 |

UNET400 gained 482 true-positive edges, removed 482 false negatives, and reduced false positives by 10.
At the sample level, 101 movies improved and 86 worsened.
The 400-epoch model was therefore not merely a more confident copy of the same predictor.
Its average behavior improved, but its error distribution changed, which explains why thresholds tuned for UNET300 could underperform when transferred directly.

There is an important qualification.
Both checkpoints were trained on all training movies, so this was **in-sample error anatomy**, not true OOF evaluation.
That distinction became essential when designing a learned repair policy.

---

## 10. What Did Not Work Yet

The failed experiments clarified how much authority each signal should receive.

### 10.1 Broad Recall And Aggressive TTA

Intensity TTA and aggressive detection expansion increased node counts without beating the best graph.
One intensity-TTA branch fell to \(0.894\).
Six-view spatial TTA using flips and XY rotations was useful, but more views were not automatically better.

### 10.2 An `edge_predictor` Checkpoint Is Not Just An Edge Head

One of the largest failures came from swapping a separate `edge_predictor_best.pth` into the calibrated graph.
The filename suggests an edge scorer, but the checkpoint contains the full model state, including the TemporalUNet detector.

| Output | Calibrated UNET400 anchor | Uncalibrated checkpoint swap |
|---|---:|---:|
| Node rows | 128,535 | 170,860 |
| Edge rows | 123,988 | 164,603 |
| Public score | about 0.900 | 0.861 |

Nodes and edges increased by roughly 33%.
The new checkpoint had inherited the old \(0.97\) detection threshold and configuration without recalibration.
This showed that an independent-seed model should not be submitted as a blind weight swap.
It should be calibrated and combined through output-level consensus or disagreement.

### 10.3 Small ILP Changes Still Need Isolated Experiments

Lowering the division weight from \(1.0\) to \(0.7\) scored \(0.897\).
The same notebook also contained a `pool_kernel_um=2.0` patch, but it ran after prediction and therefore had no effect on the submission.
Bundling changes into one cell can make execution order obscure what was actually tested.

### 10.4 Center Is Still Dangerous As A Global Detector

DeepCenterUNet3D did not help as a blind union or as a hard gate over every synthetic node.
Requiring Center confidence for all synthetic gap nodes reduced the score to \(0.898\).
A genuinely missing cell is often dim or occluded, so the auxiliary detector can make the same false negative in the same difficult frame.

There was also an artifact trap.
The `best.pt` files in the 100--500 epoch snapshots all referred to the same early best checkpoint.
Testing a specific Center epoch required `checkpoint_last.pt` plus an explicit assertion on the stored epoch.

---

## 11. When The Auxiliary Center Model Finally Helped

The Center model first produced a gain when it stopped acting like a global detector.
It received veto authority only over **geometrically marginal one-frame gaps** already proposed by the UNET400 graph.
This line of work extends the public
[Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings) notebook.

For a track ending at \(t\) and another beginning at \(t+2\), the gap proposal is:

$$
\tilde p_{t+1}=\frac{p_t+p_{t+2}}{2}.
$$

The successful UNET400 + Center400 policy was:

$$
\operatorname{accept}(d,c)=
\begin{cases}
1, & d<8\ \mu\mathrm{m},\\
\mathbf{1}[c(\tilde p_{t+1})\ge0.20], & 8\le d\le12\ \mu\mathrm{m},\\
0, & d>12\ \mu\mathrm{m}.
\end{cases}
$$

Here \(d=\|p_{t+2}-p_t\|_2\), and \(c\) is the DeepCenter probability near the proposed midpoint.
This configuration scored \(0.901\).
In contrast, requiring \(c\ge0.15\) for every synthetic gap node scored \(0.898\).

The difference reveals the correct signal hierarchy:

```text
strong temporal geometry > weak negative Center evidence
marginal temporal geometry + positive Center evidence > geometry alone
```

A low Center score is not strong evidence that a cell is absent.
A high Center score is useful independent evidence when motion geometry is already uncertain.

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

The flat threshold can be generalized by distance:

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

With bidirectional temporal context, the forward and backward midpoint
predictions can also be compared:

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

When \(e_{\text{cons}}\le2.5\,\mu\mathrm{m}\), temporal consensus takes priority over a weak image-space veto.

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

The next experiments derived from this result are not another global-threshold sweep:

1. a distance-adaptive gate that raises the Center threshold from \(0.12\) to \(0.28\) as the gap grows;
2. a bidirectional motion gate that bypasses Center when forward and backward velocity predictions agree;
3. a causal split between synthetic gap nodes and reuse of observed isolated nodes.

The shared principle is to restrict auxiliary-model authority according to the type of uncertainty.

---

## 12. Model Diversity, True OOF, And Current Direction

A second model can serve two different purposes.
An independent seed trained on all data is useful for test-time disagreement and ensembling.
Its predictions on training movies are not OOF.
A learned repair policy requires separate fold-held-out models.

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

Once the second model is mature enough, a conservative score blend can remain anchor-heavy:

$$
p_{\mathrm{blend}}
=
\alpha p_{\mathrm{anchor}}
+
(1-\alpha)p_{\mathrm{seed}},
\qquad
\alpha\in[0.65,0.75].
$$

The real objective is not averaging by itself.
It is learning where the models agree and where they make different errors.

A true OOF repair table naturally contains:

| Proposal type | Label source |
|---|---|
| existing edge | whether matched GT nodes have the same edge |
| one-frame gap bridge | whether GT contains the length-2 path |
| division edge | whether GT contains the same fork |
| short-component keep/drop | whether the component produces matched true edges |

The feature vector for a candidate edge or repair action can stay close to the
metric rather than containing raw pixels:

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

Here \(c_{ij}\) is optional Center evidence and \(\rho\) is local density.
A compact policy can then export:

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

The final question is:

```text
Which proposed graph edits are metric-positive,
given geometry, model disagreement, and local image evidence?
```

The current direction is therefore:

```text
keep the calibrated UNET400 graph as the anchor
use Center only for marginal repair confirmation
train an independent seed for output-level disagreement
build true OOF repair actions with two-fold models
record checkpoint, config, split, and SHA as one artifact contract
```

---

## Appendix: Submission Audit Snippet

The final output is still just a CSV.
Before submission, I keep a small audit routine around the schema.

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

This is not exciting code, but it prevents avoidable failures.
For graph competitions, boring validation is part of the model.
