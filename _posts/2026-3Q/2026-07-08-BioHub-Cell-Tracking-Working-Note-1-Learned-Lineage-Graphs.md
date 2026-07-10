---
title: "BioHub Cell Tracking Working Note 1: Learned Lineage Graphs and Metric-Aware Repair"
date: 2026-07-08 21:00:00 +0900
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

weight_path = ARTIFACT_MANIFEST.parent / manifest["models"]["unet_transformer"]["weight_path"]
repo_dir = ARTIFACT_MANIFEST.parent / manifest["contents"]["repo"]

print(weight_path)
print(repo_dir)
```

</details>

The manifest is not just bookkeeping.
It is the contract that lets the notebook recover source code, weights, wheel files, coordinate conventions, and model roles in an internet-disabled Kaggle run.

---

## 6. Why Graph Repair Became The Main Lever

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

BIOHUB_RUN_MODE=continue \
BIOHUB_SCRATCH_ROOT=/tmp/biohub_scratch \
BIOHUB_TRAIN_METHOD=unet_transformer_50ep_v1 \
BIOHUB_EPOCHS=400 \
BIOHUB_RESUME=1 \
BIOHUB_OVERWRITE=0 \
bash runpod_5090/scripts/run_5090_50epoch_oneshot.sh
```

</details>

Packaging follows the same contract:

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

In the public notebook text, I avoid making the hardware the story.
The story is the artifact contract:

```text
weights are trained outside the notebook
the notebook is deterministic given the attached dataset
the manifest records what the notebook is actually using
```

---

## 9. Score Progression And What It Taught Me

The exact public leaderboard numbers are not private validation.
Still, the sequence was useful as a noisy diagnostic.

| Stage | Rough public reading | Lesson |
|---|---:|---|
| Classical peak/link baseline | 0.6 to 0.7 range | The submission schema and metric were working. |
| Strong rule-based public family | 0.82 to 0.86 range | Careful geometry can go surprisingly far. |
| Learned graph artifact | 0.88+ range | Edge-aware learning is a strong backbone. |
| Short-track pruning and repair tuning | around 0.89+ | Metric-aware graph cleanup is the main lever. |
| Hard auxiliary center gate | not clearly helpful | A second detector should be a soft feature, not a blind union. |

The most important surprise was that more epochs did not automatically improve public score under the same post-processing.
The model checkpoint and the repair thresholds form a pair.
A later checkpoint can have better internal loss but shift the distribution of detection scores and edge scores enough that the old thresholds become suboptimal.

That is the reason I now read a checkpoint as:

```text
model weights + detector threshold + edge threshold + repair caps + pruning length
```

not as weights alone.

---

## 10. What Did Not Work Yet

The failures were useful.

First, increasing recall broadly was often worse than it looked.
Extra nodes can help if they become true edges, but isolated detections are mostly metric debt.

Second, division recovery is fragile.
The division term has a small weight, and false-positive forks can also create edge errors.
The right division strategy is not "turn divisions on."
It is "recover only the forks that are geometrically boring."

Third, the auxiliary full-frame center detector did not become useful through a hard gate.
The safe contract is more subtle:

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

In practice, even that can be too blunt.
The better next step is to use the auxiliary detector as a feature inside a repair policy:

$$
p_i
=
P(y_i=1\mid x_i,a_i),
$$

where \(a_i\) is a candidate graph edit and \(x_i\) includes geometry, model scores, local density, track length, and auxiliary center evidence.

---

## 11. Current Direction

The current working direction is:

```text
keep the learned graph backbone
keep the short-track pruning signal
calibrate thresholds per checkpoint family
use auxiliary detectors as repair features rather than global unions
build OOF-labeled repair actions from sparse ground truth
```

The next post should probably focus on this repair-policy layer.
The natural offline dataset is not a table of raw pixels.
It is a table of proposed graph edits:

| Proposal type | Label source |
|---|---|
| existing edge | whether matched GT nodes have the same edge |
| one-frame gap bridge | whether GT contains the length-2 path |
| division edge | whether GT contains the same fork |
| short-component keep/drop | whether the component produces matched true edges |

Then the notebook can replace hand-tuned repair thresholds with a compact exported policy.
That feels like the right direction because it attacks the actual remaining ambiguity:

```text
Which proposed graph edits are metric-positive?
```

That is a better question than:

```text
Can I find one more global threshold that happens to work today?
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
