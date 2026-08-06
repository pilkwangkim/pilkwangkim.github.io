---
title: "ARC-AGI-3 Research Note R1: The Unit Cost of Thinking, and Program-as-Policy"
date: 2026-08-05 21:00:00 +0900
categories: [AI, Kaggle]
tags: [arc-agi, arc-agi-3, agents, world-models, program-synthesis, research-note, kaggle]
math: true
pin: false
---

# ARC-AGI-3 Research Note R1: The Unit Cost of Thinking, and Program-as-Policy — From Design to First Results

> **This is a research note.** The working notes (Parts 1–2) record submissions and measurements; the R series records the study, design and validation of ideas meant to change the game rather than tune it. This post follows one question end to end — *why do improvements inside the current paradigm stall, and what structure changes the token cost of an action by an order of magnitude?* It starts from measurements in our own logs, dissects the code and traces of the two systems that have effectively conquered the public set (EWM and Schema), writes a port design for our constraints (a local 27B, ~58k tokens per scored run), and ends with **the first pilot measurements (P0) of that design**. Everything is as of August 5, 2026.

Previous: [Working Note 2]({{ site.baseurl }}{% post_url 2026-3Q/2026-08-04-arc-agi-3-part-2-where-a-perfect-idea-died %})

## 0. Where tweaking tops out

Part 2's conclusion was "read the system, not the leaderboard" — and the repairs that reading pointed to (the revived memory channel, the experiment ledger, cross-clone transfer) are all built and measured. But those repairs share a ceiling: **they all operate inside the existing cost structure.** The LLM sits inside the loop, spending hundreds of tokens per turn to read an observation and choose an action. However much waste you trim, the order of magnitude of "tokens needed to understand one level" stays fixed.

(That ceiling gets one more empirical confirmation in this post's appendix: a memory mechanism that worked perfectly and moved nothing.)

## 1. Measuring our agent's unit cost of thinking

I analyzed all 2,982 LLM turns from a 56-run competition simulation. How many real actions does one LLM turn execute?

| Actions per turn | Share |
|---|---:|
| 0 (pure deliberation) | **28%** |
| 1 | 43% |
| 2–4 | 15% |
| 5–9 | 9% |
| 10+ | 4% |

Median **1**, mean 2.06, median 52 turns per run, median **600 tokens per action**. Our agent takes at most one step in 71% of its turns and pays 600 tokens per step; with ~58k tokens per run, the step count is pinned near ~100. Compaction and ledgers turn the 600 into 500; what's needed is to move the **distribution itself** to the right.

## 2. Deep read ①: EWM — verification is the substance, the simulator is the shell

I read the full [public repository](https://github.com/astroseger/arc-3-agents-baseline1) of Executable World Models (58.12% public RHAE with GPT-5.5, 15/25 games solved). Four structural facts:

1. **No LLM in the controller.** A plain Python state machine reads the session log and picks a prompt protocol per situation. The explore→model→verify→plan cycle is enforced by prompt discipline *inside one long agentic turn*.
2. **Four fixed interfaces** — engine `world_model_engine(state, action) → (new_state, status)`, initial-state reconstruction, renderer, planner — shipped as empty stubs the LLM must fill. No reward function anywhere.
3. **Verification is exact, full-history replay.** Every recorded step replays through the model; rendered frames must match observations exactly. A mismatch is a blocking event.
4. **Plans are proven inside the model before touching the game**, then executed in lockstep, halting on the first mismatch.

And the follow-up ablation contains the direction's most important twist: **an executable world model *without* verification underperforms a plain textual one** (51.16% vs 58.85%); with fixed interfaces plus replay verification it jumps to 65.6%, then 82%. What must be ported is not "make the model write a simulator" — it is **"make its predictions mechanically accountable to the full recorded transition history."**

## 3. Deep read ②: Schema traces — what a 99% playthrough actually looks like

From Schema's released [trace dataset](https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces) (25 games × 2 models; self-reported 98.98%/95.35%, with a bundled scorer we could re-run locally), I dissected four trajectories:

- **A three-function contract**: `init_state` / `predict` / `is_goal`.
- **notes.md re-injected verbatim every turn** — external memory, not context, is the substrate of remembering.
- **`run_backtest`** replays every recorded transition through `predict`; agents verify "backtest 10/10 green" before committing long plans.
- **`commit_actions` submits a batch plan**; the harness executes step-by-step against predictions and **drops the plan tail on the first mismatch** — the surprise gate that makes long open-loop batches safe.

The economics, measured:

| Trace | LLM turns | Actions | Actions/turn | Tokens (approx.) |
|---|---:|---:|---:|---:|
| ft09 (easy, 100%) | 15 | 78 | **5.2** | ~60k |
| sb26 (mid, 98.6%) | 12 | 135 | **11.2** | ~56k |
| dc22 (hard, 98.7%) | 218 | 1,205 | 5.5 | ~634k |

3–11 actions per turn — a different order of magnitude from our median of 1. And **easy/mid games finish within ~60k content tokens even for frontier models — the same order as our per-run budget.**

## 4. The port: D1 under our constraints

| Piece | Schema/EWM | Ours today | Work needed |
|---|---|---|---|
| Model-code persistence | file system | sandbox has no file IO | **harness-side world-model store**, injected per call |
| Transition record | event logs | `transitions` var already injected | none — backtest runs in-sandbox |
| Batch + surprise gate | commit_actions | `action(list)` + mixed-batch abort graft | near-isomorphic; protocol only |
| Search over the model | run_bfs tool | stdlib BFS in-sandbox | prompt only |
| Determinism | assumed | **35.5% repeated-pair divergence, measured** | settled-frame compare + tolerance + "sim-free" family flag |

The minimal port (v0): ① the harness keeps `{src, notes}` per game — implemented as parent-side code rewriting: any code cell containing `def predict(` is captured automatically, and every later cell gets it injected as the string `world_model_src` (restored via `exec(world_model_src)`); no sandbox protocol changes. ② The prompt switches to model-first discipline (probe sweep → write `predict` → **backtest all transitions** → search over predict → emit the plan in one batch). ③ Families whose mismatch rate stays above 20% get flagged "sim-free" and fall back to today's behavior — the fold of EWM's determinism assumption into our measured 35.5% reality.

Output reliability also resolved during research: our serving stack (the vLLM 0.19 line — the wheelhouse version got corrected along the way) supports **per-request structured outputs with no server flags**; following the literature (Tam et al. 2024 and the projection-tax follow-ups), **constrain the plan JSON, never the Python model** — code gets assistant prefill and a stop token, with an `ast.parse` repair loop.

## 5. The P0 pilot: design and first measurements

The design was implemented as grafts and piloted immediately. Four public games — a deterministic pair (bp35, tn36) and a non-deterministic pair (m0r0, sk48) — under three conditions in one commit, agents tagged by game-id suffix:

| Arm | Contents |
|---|---|
| **baseline** | the current stack, untouched (control) |
| **protocol** | the MODEL-FIRST prompt only |
| **full** | protocol + the world-model store |

One caveat: 12 parallel runs is a richer token regime (~120k tokens/run) than the scored one (~58k). P0 asks *can it be done*; whether it pays at the scored operating point is P1's question.

### Results

> **Correction (2026-08-06).** The first version of this section reported "wrote predict 4/4, ran backtest 4/4" for the protocol/full arms. That was a measurement error: the protocol block itself is printed into the transcript every turn, so a substring search **mistook the prompt's echo for the model's behavior**. Re-measured with a line anchor (`^def predict(`), the number of runs in which the model actually wrote predict is zero across all of P0. Corrected numbers and interpretation below.

| Arm | L1 clears | Actions/turn (med / p90) | Tokens/action | Actually wrote predict (corrected) |
|---|---:|---:|---:|---:|
| baseline | 2/4 | 1 / 6 | 641 | 0/4 |
| protocol | 2/4 | **2 / 13** | **415** | **0/4** |
| full | **3/4** | 1 / 6 | 755 | **0/4** |

**Observation ① (corrected) — the gate was NOT passed.** The model reads the protocol and **complies with its cheap parts only**: it does the probe sweeps and action batching (the substance of Observation ②), and it writes helper functions in abundance (`def find_node`, `def get_progress` and kin in 32/56 files by line anchor) — but it skips the expensive core: writing a forward-dynamics `predict` and backtesting it. As long as each turn's local incentive is "just act", exhortation cannot elicit a multi-turn expensive workflow.

**Observation ② — the economics shift is real.** The protocol arm's 641 → **415** tokens/action (−35%) and p90 6 → **13** are behavioral metrics, immune to the contamination. But their source was the protocol's batching pressure, not model-building.

**Observation ③ — the m0r0 clear is real, but its attribution is now uncertain.** It happened without a predict, so we cannot distinguish systematic-sweep benefit from noise (n=1).

L2 remains zero in every arm; n=4 per arm.

## 6. Appendix: the compaction A/B came back neutral

A 56-run A/B (with vs without the eviction-repair layer: experiment ledger + cross-level carry) also finished. The machinery worked perfectly — the ledger appeared in 56/56 transcripts, 99 "repeatedly inert" warnings fired, the cross-level carry triggered 8 times. The outcomes:

| | Without | With |
|---|---|---|
| L1 (wave 1/2) | 46% / 57% | 50% / 61% |
| L2+ | 3+3 | 1+2 |
| Total levels cleared | 35 | 34 |

+4pp is inside the noise; totals identical. **Preserving knowledge does not convert it into clears within the budget** — the third empirical confirmation of §0's claim, and a restatement of why P0 exists.

## 7. Status of the other digs

- **D2 (offline distillation)**: feasibility resolved — LoRA on the bf16 base → merge → FP8 re-quantization (~28GB, loadable as-is); material = the Schema traces (~8–20M usable SFT tokens); compute is trivial (4–10 H100-hours). Pipeline and validation make it **a 2–4 week project**, booked for after the prompt-side plateaus.
- **D3 (zero-token probes)**: from "Explore Before You Solve" — the budget cap $B_{\max} = \max(5, \min(30, 0.4\,h_1))$ (≈30 probe / 1–3 verify / rest solve at our budget), and, more valuable, their failure: LLM-prior exploration lost four one-press-winnable games → **hard-code a deterministic opening sweep (~10 actions)**. Folded into D1's probe phase.
- **D4 (observation compression)**: Schema's hex-row grid encoding (~1.3k tokens/frame) as the reference point; partially absorbed into D1.
- **D5 (clones × programs)**: promote what the scout publishes from action segments to **world-model source**, riding the verified transfer infrastructure. Once D1 stands, D5 is nearly free.

## 8. P1 results, and the fork in the road

P1 (the 56-run competition sim at the scored regime, ~58k tokens/run, program mode fully on) is also complete. Compliance with the cheap parts of the discipline held, but **P0's economics gain evaporated in the token-starved regime** (562 → 544 tokens/action; actions-per-turn distribution unchanged), and outcomes were flat (L1 by wave 50/46% vs the control's 50/61%; total levels 30 vs 34). The model wrote predict in 1 of 56 runs.

In the corrected picture, the lesson circles back to the EWM read (§2). EWM worked not because its prompts were good but because **a scripted controller mechanically enforced the phases**. The exhortation was decoration; the enforcement was the substance. Our fork in the road:

1. **Mechanical enforcement (P1b)** — close the cheap path at the harness level. For example: after the first N turns of a game, `action()` refuses (with a "write your model first" notice) unless a predict exists in the store or the run has declared itself sim-free; combined with seeding a skeleton predict (identity + TODO) into the store so that extending is cheaper than ignoring.
2. **Distillation (D2)** — behavior that prompting cannot elicit gets baked into the weights. The 2–4 week bet whose feasibility is already resolved (§7); prompt-resistance is exactly the kind of problem SFT fixes.

Either way, one methodological lesson from this series is now settled: **measure compliance separately from prompt echo.** A string appearing in the transcript and the model doing the thing are different propositions.

---

### Sources

- [EWM code](https://github.com/astroseger/arc-3-agents-baseline1) · [paper](https://arxiv.org/abs/2605.05138) · [ablation follow-up](https://arxiv.org/abs/2607.15439)
- [Schema trace dataset](https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces) (self-reported, not ARC-verified)
- [Explore Before You Solve](https://arxiv.org/abs/2605.25931) · [Tycho](https://arxiv.org/abs/2607.28287) · [Tam et al.](https://arxiv.org/abs/2408.02442)
- Our measurements: 2,982 turns from the 56-run sim / the P0 12-run three-arm commit / the 56-run compaction A/B (2026-08-05)
