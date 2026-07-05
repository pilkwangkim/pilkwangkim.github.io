---
title: "Pokémon TCG AI Battle Working Note (Part 1): Meta Intelligence and the Pilot Gap"
date: 2026-07-02 21:30:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, pokemon-tcg, game-ai, reinforcement-learning, behavior-cloning, meta-analysis, benchmark, working-note]
math: true
pin: false
---

# Pokémon TCG AI Battle Working Note (Part 1): Meta Intelligence and the Pilot Gap

> **Working note status: first written 2026-07-02, updated through the 2026-07-04 benchmark pass.**  
> This is the first note in a weekly series on the Pokémon TCG AI Battle competition. The goal is not only to improve a submission, but to build a repeatable research loop: mine the public battle logs, understand the changing field, separate deck strength from pilot strength, benchmark candidates against the actual meta, and eventually use behavior cloning / reinforcement learning to repair the parts that rule-based agents do poorly.

Competition link:  
[The Pokémon Company - PTCG AI Battle Challenge Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)

---

## 1. What the competition really asks

At first glance, this looks like a normal Kaggle agent competition: write an agent, submit it, watch a skill rating move. But the core object is more subtle.

The agent is playing the Pokémon Trading Card Game inside a simulator. Each turn, it receives an observation containing the current board, logs, hidden-information boundaries, and a list of legal options. It returns the indices of the options it wants to take. The simulator only offers legal actions, so the policy problem is not "generate a legal move from scratch." It is:

$$
a_t \in \text{LegalOptions}(o_t)
$$

or, more concretely:

$$
\pi(i \mid o_t, \text{option}_1, \ldots, \text{option}_k)
$$

where each candidate option is already legal, but not equally good.

That makes the challenge a mixture of three problems:

| Layer | Question | Why it matters |
|---|---|---|
| Deck construction | Which 60-card deck should I bring? | A weak deck cannot be saved by clean play. |
| Pilot policy | How should the deck be played from each legal-option menu? | A strong deck can score badly with a bad pilot. |
| Meta adaptation | Which opponents are common today, and which ones are rising? | A deck that beats yesterday's field can be the wrong answer tomorrow. |

The key lesson from the first week is that these layers must not be collapsed into one score. A deck hash can be strong in the public logs, but if I cannot pilot it, it is not yet a submission. Conversely, a locally strong agent can fail live if the benchmark panel is missing the decks that actually matter.

So I use the term:

$$
\text{Strength} = f(\text{deck}, \text{pilot}, \text{field}, \text{seat}, \text{variance})
$$

That last term matters because the engine's stochasticity is not fully seed-controlled from the public interface. Common-random-number paired evaluation would be ideal, but it is not available. The practical replacement is seat balancing, common opponent panels, Wilson intervals, holdout runs, and local-to-live calibration.

---

## 2. Why a working-note pipeline is needed

My early approach was ordinary: inspect public notebooks, build a reasonable rule-based deck/pilot, submit, and watch the score. It was useful, but it quickly hit the wall that many Kaggle game competitions hit:

1. A candidate can look good locally because the local opponent pilots are too weak.
2. A candidate can look good because it was selected from many noisy variants.
3. A candidate can be good into yesterday's dominant deck and weak into today's rising deck.
4. A deck can be discovered from logs before its pilot is ready.
5. A high live score can come from a better pilot, not only from a better deck list.

The workflow therefore moved from "tweak the submission" to "build a meta intelligence system."

The daily loop now looks like this:

```text
Kaggle replay logs
-> compact extraction tables
-> archetype / exact-deck census
-> matchup matrix and strategy profiles
-> field-weighted benchmark panels
-> pilot library and fidelity checks
-> candidate deck / pilot search
-> holdout and local-to-live gap diagnosis
-> submission only when the evidence beats known references
```

That last line is important. A candidate should not be submitted merely because it looks plausible. It should beat the known reference set, or I should honestly say that the evidence is not strong enough yet.

---

## 3. Data layer: what was processed by June 30

By June 30, the local store had processed five days of replay logs. The 2026-06-30 batch alone was
enough to support a useful field snapshot:

| Item | 2026-06-30 value |
|---|---:|
| Raw replay JSON files | 5,734 episodes |
| Player deck rows | 11,468 |
| Winner decklists captured | 133 |
| Strategy profiles | 17 archetypes |
| Exact decklists exported for reuse | 55 ready decks |
| Candidate decision rows extracted | 327,937 |
| Subsampled decision rows kept | 82,182 |

The important point is not the count itself, but the shape of the evidence chain. Replays are converted
into compact analysis tables, every derived artifact is partitioned by source date, and benchmark runs
remain linked to the field snapshot that produced their panel weights.

That makes the research reproducible. A future weekly report should be able to say: "This deck was
selected because it beat the 20260630 field-weighted panel, whose weights came from these replay logs."

---

## 4. The first meta story: Archaludon fell, Alakazam rose, Marnie emerged

The largest visible shift from June 29 to June 30 was a rotation away from a very Archaludon-heavy field into a more Alakazam-heavy field.

| Archetype | 2026-06-29 share | 2026-06-30 share | Delta | 2026-06-30 score rate |
|---|---:|---:|---:|---:|
| Alakazam / Dunsparce | 0.2124 | 0.3306 | +0.1182 | 0.5186 |
| Archaludon | 0.4067 | 0.2892 | -0.1176 | 0.4801 |
| Starmie | 0.0710 | 0.0787 | +0.0076 | 0.4501 |
| Dragapult | 0.0659 | 0.0687 | +0.0028 | 0.4365 |
| Marnie's Impidimp / Munkidori | 0.0126 | 0.0412 | +0.0286 | 0.6469 |
| Cynthia's Power Weight / Cynthia's Gabite | 0.0048 | 0.0367 | +0.0319 | 0.4988 |
| Crustle | 0.0045 | 0.0169 | +0.0124 | 0.5412 |
| Okidogi / Solrock | 0.0079 | 0.0149 | +0.0070 | 0.6140 |

The headline is not simply "Alakazam is best." It is more nuanced:

- Alakazam became the largest share archetype, but its score rate was only modestly positive.
- Archaludon lost share and had a sub-50 field score rate, but tuned Archaludon agents were still very relevant in local benchmarks.
- Marnie's Impidimp / Munkidori was still a small-share archetype, but its score rate and matchup matrix made it the most important rising threat.
- Okidogi / Solrock and Crustle were still small, but they were no longer ignorable.

This is why raw share alone is dangerous. A popular archetype can be only mildly positive because many teams are piloting imperfect versions. A lower-share archetype can be the real pressure if it has strong matchups into the two largest decks.

---

## 5. Matchups explain the shift better than share does

The most useful table is not the field-share table. It is the matchup table.

On June 30:

| Matchup | Games | First archetype win rate |
|---|---:|---:|
| Alakazam / Dunsparce vs Archaludon | 1,026 | 0.6637 |
| Marnie's Impidimp / Munkidori vs Archaludon | 148 | 0.6014 |
| Marnie's Impidimp / Munkidori vs Alakazam / Dunsparce | 134 | 0.6791 |
| Archaludon vs Starmie | 261 | 0.7471 |
| Archaludon vs Dragapult | 251 | 0.6534 |
| Crustle vs Alakazam / Dunsparce | 62 | 0.6774 |
| Okidogi / Solrock vs Alakazam / Dunsparce | 59 | 0.6780 |

This begins to explain the observed field movement.

Alakazam's rise is rational because it is very strong into field Archaludon. If Archaludon occupies 30 to 40 percent of the field, Alakazam has a clean reason to grow.

But the next layer is more interesting. Marnie's Impidimp / Munkidori beats both of the big decks in the observed matrix: about 60 percent into Archaludon and about 68 percent into Alakazam. That is exactly the shape of a rising second-order meta answer.

The field-weighted matchup estimate reinforces this:

| Archetype | Usage share | Field-weighted score | Covered field weight |
|---|---:|---:|---:|
| Marnie's Impidimp / Munkidori | 0.0412 | 0.6623 | 0.9554 |
| Cubchoo / Gravity Gemstone | 0.0094 | 0.6520 | 0.7545 |
| Meowth ex / Mega Kangaskhan ex | 0.0098 | 0.6198 | 0.8403 |
| Okidogi / Solrock | 0.0149 | 0.6073 | 0.8883 |
| Team Rocket Spidops | 0.0106 | 0.5969 | 0.7671 |
| Crustle | 0.0169 | 0.5660 | 0.8309 |
| Alakazam / Dunsparce | 0.3306 | 0.5207 | 0.9980 |
| Archaludon | 0.2892 | 0.4720 | 0.9987 |

This table is one of the first genuinely useful outputs of the project. It says that the strongest-looking 0630 archetype was not the most common one. It was the emerging Marnie/Munkidori family.

But there is a catch: discovering a deck is not the same as owning a submission.

---

## 6. Deck quality and pilot quality are separate

The 0630 logs produced several exact deck discoveries:

| Archetype | Exact deck sample | Games | Score rate | Wilson low |
|---|---:|---:|---:|---:|
| Marnie's Impidimp / Munkidori | rank 2 hash | 137 | 0.6788 | 0.5967 |
| Marnie's Impidimp / Munkidori | rank 1 hash | 246 | 0.6626 | 0.6014 |
| Okidogi / Solrock | rank 1 hash | 141 | 0.6454 | 0.5635 |
| Alakazam / Dunsparce | rank 1 hash | 287 | 0.6272 | 0.5699 |
| Archaludon | rank 1 hash | 144 | 0.6042 | 0.5226 |
| Crustle | rank 1 hash | 193 | 0.5440 | 0.4736 |

This is useful, but it also exposes the biggest gap in the current system.

For a deck like Archaludon, a rule-based pilot can be reasonably effective because the game plan is straightforward: evolve, attach metal resources, attack early, keep tempo, use a small number of high-value attack patterns. For Alakazam, the pilot is harder: the deck has more draw/ability sequencing, bench management, and late payoff timing. For Marnie/Munkidori, the pilot is harder still because the deck needs a high-setup combo pattern before it turns into damage pressure.

So I separate:

```text
deck discovery
pilot acquisition
benchmark validation
submission packaging
```

A strong exact deck can live in the FieldStore as a future candidate while still being blocked from submission because the pilot is not good enough.

This distinction also explains a frustrating early failure mode: some local benchmarks selected a deck correctly but still produced weak live scores. The deck was not necessarily wrong. The pilot may have been a poor approximation of how the deck wins.

---

## 7. Strategy profiles: what the decks are actually doing

The strategy profiler tries to describe a deck by behavior, not just by card names.

| Archetype | Class | First attack turn | Attack cadence | Prize rate | Abilities / turn | Draw-search / turn | Main attack concentration |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alakazam / Dunsparce | combo | 4.049 | 0.4840 | 0.3056 | 1.0336 | 4.7817 | 0.9115 |
| Archaludon | aggro | 2.773 | 0.6601 | 0.3149 | 0.2473 | 5.3600 | 0.6724 |
| Starmie | aggro | 2.719 | 0.6564 | 0.4599 | 0.2275 | 4.1409 | 0.6621 |
| Dragapult | midrange | 2.968 | 0.6250 | 0.4023 | 0.5759 | 8.1486 | 0.7005 |
| Marnie's Impidimp / Munkidori | combo | 4.734 | 0.4962 | 0.3943 | 0.8169 | 7.6289 | 0.9929 |
| Okidogi / Solrock | midrange | 3.971 | 0.5448 | 0.3349 | 0.1089 | 5.7270 | 0.7438 |
| Crustle | combo / wall | 9.546 | 0.3059 | 0.0050 | 0.0908 | 2.1338 | 0.9433 |

The behavioral reading:

- **Archaludon** is the cleanest tempo deck: early first attack, high cadence, lower ability complexity. This makes it rule-friendly.
- **Alakazam / Dunsparce** is an ability/draw combo deck: slower first attack, concentrated payoff through Alakazam's attack pattern, and many sequencing choices.
- **Marnie/Munkidori** is a high-setup combo deck: late-ish first attack, very high setup action count, and almost all payoff through Marnie's Grimmsnarl ex `Shadow Bullet`.
- **Crustle** is not a prize-race deck. It is a wall / collapse pattern. The prize rate is almost zero, so a naive benchmark that only checks prize tempo can misunderstand it.
- **Okidogi/Solrock** is a midrange support deck: it is not simply an attacker pile; its support engine matters.

This gives the pilot search a target. Instead of writing "play Marnie better" as a vague instruction, I can define behavioral constraints:

```text
Does the pilot set up Impidimp -> Morgrem / Grimmsnarl?
Does it preserve the support package instead of attacking too early?
Does it reach the same high-concentration payoff attack pattern?
Does it avoid dead low-value legal options during setup?
```

That is where behavior cloning becomes relevant.

---

## 8. Benchmarking: what was actually tested

The local benchmark is useful only if the opponent panel and pilots are good enough. For this reason, the benchmark uses:

- real CABT simulator games, not a mock;
- seat balancing, because seat and turn order matter;
- common opponent panels, because comparing candidates against different opponents is noisy;
- Wilson intervals, because small matchup slices lie;
- field weights from the daily meta;
- pilot status labels, because weak proxy pilots can inflate local scores.

The first 0630 relevance run favored Archaludon-family candidates:

| Candidate | Games | Raw score rate | Wilson low |
|---|---:|---:|---:|
| `flex_archaludon_0021` | 128 | 0.7500 | 0.6684 |
| `public_archaludon_75wr` reference | 112 | 0.7500 | 0.6624 |
| `flex_archaludon_0018` | 128 | 0.7422 | 0.6601 |
| `archaludon_75wr_vs_starmie` | 128 | 0.7109 | 0.6272 |
| `flex_alakazam_dunsparce_0000_seed` | 112 | 0.6786 | 0.5874 |

Field-weighted, the same run had:

| Candidate | Field-weighted score | Edge vs field |
|---|---:|---:|
| `flex_archaludon_0021` | 0.6929 | +0.2322 |
| `flex_alakazam_dunsparce_0000_seed` | 0.6490 | +0.0896 |
| `public_archaludon_75wr` reference | 0.6213 | +0.1860 |
| `flex_archaludon_0018` | 0.5905 | +0.1298 |

If I had stopped there, the conclusion would have been simple: push Archaludon. But a larger validation changed the reading.

With a GPS=16 validation:

| Candidate | Games | Raw score rate | Wilson low |
|---|---:|---:|---:|
| `public_archaludon_75wr` reference | 224 | 0.7634 | 0.7036 |
| `flex_alakazam_dunsparce_0000_seed` | 224 | 0.7589 | 0.6989 |
| `flex_archaludon_0021` | 256 | 0.6914 | 0.6323 |
| `flex_archaludon_0018` | 256 | 0.6758 | 0.6162 |
| `archaludon_75wr_vs_starmie` | 256 | 0.6289 | 0.5682 |

And field-weighted:

| Candidate | Field-weighted score | Edge vs field |
|---|---:|---:|
| `flex_alakazam_dunsparce_0000_seed` | 0.7430 | +0.1837 |
| `public_archaludon_75wr` reference | 0.6600 | +0.2246 |
| `flex_archaludon_0021` | 0.5674 | +0.1067 |
| `flex_archaludon_0018` | 0.5223 | +0.0616 |

This is the first big methodological lesson:

> A small local benchmark can pick the right family for the wrong reason, or the right candidate on a lucky panel. A second validation run is not optional; it can change the conclusion.

The current reading is not "Alakazam is solved." It is:

- Archaludon remains a strong practical family because it is easier to pilot and has good pressure into Alakazam/Starmie-like panels.
- `flex_alakazam_dunsparce_0000_seed` must be revived as a serious B-candidate track because the larger validation was unexpectedly strong.
- `public_archaludon_75wr` is a valuable reference/oracle, but copying a public notebook is not the objective. The useful part is behavior comparison and controlled derivative testing.
- The Marnie/Munkidori deck family is probably the most valuable field discovery, but it is pilot-gated.

---

## 9. The local-live gap

This project had an uncomfortable but useful failure: some candidates that looked locally plausible did not live-score as expected.

The explanation is not a single bug. The local-live gap can come from several causes:

| Cause | Symptom | Fix |
|---|---|---|
| Weak opponent pilots | Local win rate is inflated because proxies misplay the field decks. | Audit and repair opponent pilots. |
| Missing meta coverage | The benchmark does not include the decks that beat the candidate live. | Refresh field panel from logs. |
| Candidate pilot only imitates field average | The deck is fine, but the pilot does not beat the average pilot. | Optimize for beat-field supremacy, not fidelity. |
| Winner's curse | The best candidate among many variants regresses live. | Holdout validation and shrinkage. |
| Meta drift | Today's live field is no longer yesterday's log field. | Daily weighting and trend reports. |
| Runtime / packaging differences | Local smoke passes but live run behaves differently. | Self-game diagnostics and package validation. |

This led to a policy change in the project:

```text
Do not publish or submit a "meta snapshot" candidate just because it looks reasonable.
It must beat refreshed known references, or the report must say that no verified edge exists.
```

The point is not to be conservative for its own sake. The point is to avoid turning the leaderboard into the validation set.

---

## 10. Project history ledger

Because this series is also meant to support a future Working Note / write-up, I want each post to preserve more than the final conclusion. The failed directions matter. They explain why the current workflow exists.

Here is the project ledger up to the June 30 snapshot.

| Phase | What happened | What it taught | Durable artifact |
|---|---|---|---|
| Competition orientation | Read the competition pages, simulator notes, sample decks, public examples, and local guides. | The unit of submission is not just a deck; it is a deck plus a callable pilot plus packaging discipline. | Guide notes, early `AGENTS.md`, sample deck inventory |
| First baseline era | Built Lucario-style rule-based notebooks, debugged `main.py`, deck CSV, `cg` package inclusion, and notebook cell visibility / standalone behavior. | A clean notebook can submit, but a rule-only pilot without meta context is fragile. Packaging mistakes can masquerade as strategy failures. | Lucario baseline notebooks and submission-format checks |
| Public example reading | Compared high-scoring public notebooks and extracted ideas without treating them as mainline code to copy. | Public notebooks are best used as references, opponents, and behavior oracles; direct cloning destroys the research loop. | Public example inventory and derivative-candidate notes |
| Two-agent portfolio turn | Started thinking in two active submissions rather than one "best" agent. | The objective is portfolio upside and different failure modes, not a single stable average. | Two-submission specs and portfolio benchmark tables |
| Local engine breakthrough | Verified that `kaggle_environments.make("cabt")` and the packaged CABT engine can run full local games. | Mock benchmarks are unnecessary, but the engine does not expose reliable RNG seeding. This rules out CRN-style paired comparisons. | Seat-balanced benchmark runner and engine diagnostics |
| MetaStore / FieldStore buildout | Moved from raw replay reading to date-partitioned compact stores. | Raw replays are evidence, but the analysis layer must be compact, queryable, and tied to source dates. | `Archive/MMDD`, `MetaStore/YYYYMMDD`, `FieldStore/YYYYMMDD` |
| Deck discovery layer | Extracted exact deck hashes, winner decklists, field share, and matchup matrices. | A deck can be discovered before it is playable by my agent. Deck discovery and pilot acquisition must be separate gates. | Ready decks, exact deck summaries, matchup matrices |
| Strategy profiler | Added behavior features: first attack turn, attack cadence, draw/search rate, ability rate, attack concentration, prize rate. | Archetype names are not enough. A deck's mechanism has to be described in behavior space. | Strategy profile reports |
| Pilot audit and repair | Formalized pilot fidelity, weak-proxy detection, local-live gap diagnosis, beat-field pilot targets, and holdout / shrinkage. | A faithful average pilot is useful as an opponent, but the submitted pilot must beat the field, not merely imitate it. | Pilot audit specs and gap diagnosis policy |
| Meta snapshot lessons | Built public meta snapshot notebooks, then tightened the rule that a snapshot recommendation must beat known references or clearly state that no verified edge exists. | Publishing a plausible candidate is not enough. The note must separate evidence, uncertainty, and compromise decisions. | Meta snapshot notebooks and audit manifests |
| June 30 weekly staging | Processed 5,734 new episodes, saw Alakazam rise, Archaludon fall, and Marnie/Munkidori emerge. | The best deck discovery was probably not the easiest current submission. The project must prioritize pilot acquisition. | Daily meta report, deep analysis, benchmark staging report |

The live ladder also provided useful feedback. Some earlier candidates that looked reasonable locally landed in the 600-800 range, while a later Archaludon tempo challenger reached roughly the high-900 band. I do not treat those numbers as final truth, because Kaggle ratings are noisy and time-dependent, but they are strong enough to update the workflow: local benchmark results must be calibrated against known live references, and "looks decent" is not a promotion criterion.

---

## 11. Report-ready artifact map

For a future consolidated report, the important object is not a single notebook. It is an evidence chain.

| Artifact family | What it records | Report question it answers |
|---|---|---|
| `Archive/MMDD` | Raw official replay logs for a day. | What actually happened on the ladder? |
| `MetaStore/YYYYMMDD/daily_meta_report.md` | Daily field share, exact deck hashes, common matchups, strategy snapshots. | What changed today? |
| `MetaStore/meta_trend_report.md` | Cross-day share and score movement. | Which archetypes are rising or falling? |
| `FieldStore/YYYYMMDD` | Ready decklists, field weights, selection reports. | Which decks are worth testing, and how should the panel be weighted? |
| Strategy profile tables | Behavioral summaries by archetype. | Why does a deck win, beyond its card list? |
| `pokemon_benchmark_runs/...` | Local simulator results, weighted scores, validation panels, portfolio reports. | Which runnable agents actually beat the current field proxy? |
| Pilot library / pilot audit outputs | Fidelity, weak-proxy labels, missing archetype coverage, behavior divergence. | Can I trust this benchmark opponent or submitted pilot? |
| Decision extraction / BC datasets | Observation-option-action rows from replays. | Can a learned pilot imitate a discovered deck's real behavior? |
| Meta snapshot notebooks | Public-facing summary plus optional submission package. | What can be shared clearly without losing the evidence trail? |
| `AGENTS.md` and specs | Operating policy and promotion gates. | Why did the workflow make this decision? |

I also want each future note to distinguish evidence levels:

| Level | Evidence type | How it should be used |
|---|---|---|
| L0 | Anecdotal live score or single notebook claim | Useful hint, never enough alone. |
| L1 | Field census / deck frequency | Good for meta weighting, not for submission strength. |
| L2 | Matchup matrix from logs | Good for deck discovery and counter-map reasoning. |
| L3 | Runnable local benchmark with validated pilots | Good for candidate screening. |
| L4 | Holdout / larger validation with shrinkage | Required before serious promotion. |
| L5 | Live calibration against known references | Required to trust the local benchmark over time. |

This is the part I was missing at the beginning. A working note should not only say "candidate A looked good." It should say which evidence level supports that claim, what was not covered, and which artifact lets someone reproduce or dispute it later.

---

## 12. RL and behavior cloning: why it starts with imitation

Pokemon TCG is tempting as a reinforcement learning problem, but starting from pure self-play is expensive and brittle:

- the action space is a variable legal-option list;
- hidden information and stochastic draws make credit assignment noisy;
- many legal moves are obviously bad, but only after understanding card context;
- a deck-specific pilot needs sequencing, not only final win/loss feedback;
- the simulator is fast enough for local games, but not fast enough to waste on blind exploration.

So the first RL direction is behavior cloning from replay logs.

The supervised learning object is:

$$
\pi_\theta(i \mid o_t, c_i)
$$

where \(c_i\) is a legal candidate option from the simulator's `select.option` menu. A masked softmax scores only legal candidates:

$$
P(i \mid o_t) =
\frac{\exp(s_\theta(o_t, c_i))}
{\sum_{j \in \text{legal}(o_t)} \exp(s_\theta(o_t, c_j))}
$$

The label is the option actually selected in the replay. One important engineering detail: in the replay schema, the observation at step \(i\) is paired with the action that appears at step \(i+1\). Getting this alignment wrong silently trains nonsense.

The initial BC playbook has already covered:

- replay schema audit;
- winner-only decision extraction;
- game-level split;
- legal-option candidate feature scaffolding;
- a masked legal-option scorer;
- a minimal behavior-cloning loop.

The early smoke test was intentionally small: about 30 games and 2,120 decision records, just enough to prove that the pipeline connects. It is not a strength claim.

The real use of BC is not "replace the agent with a neural policy tomorrow." The immediate goal is more modest and more useful:

```text
Use replay behavior to build better pilots for discovered decks.
Use behavior profiles to compare our pilot against field pilots.
Use learned scorers as priors or fallback rankers inside a rule-based agent.
Use policy divergence to detect why a local candidate underperforms live.
```

In short: RL is not a magic button here. It is a way to close the pilot gap.

---

## 13. Current deck readings as of June 30

Here is the current strategic map.

### Alakazam / Dunsparce

The most common archetype on June 30. It is strong into field Archaludon and therefore has a clear reason to rise. The best exact deck sample scored around 0.627 over 287 games. The benchmark result for `flex_alakazam_dunsparce_0000_seed` revived the family as a serious second submission track.

The risk: older Alakazam builds underperformed live. That suggests the deck is sensitive to pilot quality and local proxy quality.

### Archaludon

Not the best raw field archetype on June 30, but still the most practical rule-based candidate family. It attacks early, has a clean tempo plan, and local pilots can execute it with fewer brittle combo decisions. It also remains a useful reference because public Archaludon-like notebooks were strong.

The risk: field Alakazam beats field Archaludon hard, so the Archaludon pilot must be better than the average field Archaludon pilot.

### Marnie's Impidimp / Munkidori

The most important discovery. It has meaningful share, the best field-weighted signal among non-tiny archetypes, and strong observed matchups into both Alakazam and Archaludon.

The risk: pilot availability. The deck is a high-setup combo family. A naive pilot can destroy the edge.

### Crustle

A wall/collapse stress deck. It does not look like a normal prize-race strategy. It is valuable even if it is not immediately a submission candidate because it reveals whether our agents can handle non-tempo game plans.

### Okidogi / Solrock

A small but strong midrange family. The logs suggest it is especially relevant as a blind spot into Alakazam/Archaludon panels.

---

## 14. What I would do next

The next week should not be a random search over card swaps. It should be a structured research loop:

1. **Pilot acquisition for Marnie/Munkidori.**  
   Extract winner decisions, build a pilot profile, and benchmark exact rank-1/rank-2 decks only after the pilot passes trace-level sanity checks.

2. **Alakazam pilot audit.**  
   Compare `flex_alakazam_dunsparce_0000_seed`, older weak Alakazam submissions, and the 0630 rank-1 exact deck. Determine whether the gap is deck list, pilot priority, or benchmark proxy.

3. **Archaludon robustness validation.**  
   Keep `0021`, `0018`, and public-Archaludon-like variants in a larger holdout. Do not assume the first GPS=8 signal is stable.

4. **Stress opponent expansion.**  
   Add Crustle, Okidogi/Solrock, Dragapult, and Cynthia/Gabite pilots as soon as they are good enough to be useful. A bad stress pilot is worse than no stress pilot because it teaches the wrong lesson.

5. **Local-live gap accounting.**  
   For every live submission, compare local predicted win rate, field-weighted matchup expectation, and actual rating movement. The objective is not just to score; it is to calibrate the simulator workflow.

6. **BC as pilot repair.**  
   Use behavior cloning first for deck-specific pilot imitation, then consider RL/self-play only after the supervised pipeline is strong enough to produce useful priors.

---

## 15. July 4 update: the meta widened, and the B-slot question changed

The July 3 replay batch and the July 4 benchmark pass changed the interpretation of the week. The earlier story was mostly:

```text
Archaludon falls -> Alakazam rises -> Marnie appears as the second-order answer
```

By July 4, that was no longer enough. The field had widened into a larger pressure map.

The 20260703 extraction processed:

| Item | Value |
|---|---:|
| Episodes | 5,133 |
| Player deck rows | 10,266 |
| Winner decklists captured | 99 |
| Behavior rows | 5,126 |
| Decision rows kept | 60,000 |

The latest field share made the rotation clearer:

| Archetype | 20260703 share | 20260703 score rate | Reading |
|---|---:|---:|---|
| Alakazam / Dunsparce | 0.2572 | 0.4428 | Largest share, but not winning the field overall. |
| Marnie's Impidimp / Munkidori | 0.1784 | 0.5958 | The strongest high-share pressure. |
| Starmie | 0.0924 | 0.5406 | Rebounded sharply; important control check. |
| Cubchoo / Gravity Gemstone | 0.0569 | 0.4058 | High enough share to matter as a stress row, but not a strong field result. |
| Solrock / Okidogi | 0.0512 | 0.5209 | New coverage gap; no longer ignorable. |
| Lucario | 0.0505 | 0.5097 | Stable reference pressure. |
| Archaludon | 0.0486 | 0.3888 | Field share collapsed, but tuned pilots can still be strong. |
| Marnie's Grimmsnarl / Impidimp | 0.0386 | 0.5606 | Smaller than Marnie/Munkidori, still efficient. |
| Crustle | 0.0376 | 0.5052 | Wall / library-out stress shape. |
| Comfey / Chandelure | 0.0351 | 0.5000 | Coverage gap, not yet a clean benchmark row. |
| Cubchoo / Dunsparce | 0.0297 | 0.5475 | New small-share pressure row. |

The trend view is even more important than the one-day table:

| Archetype | 20260629 share | 20260703 share | Move | 20260703 score |
|---|---:|---:|---:|---:|
| Archaludon | 0.4067 | 0.0486 | -0.3581 | 0.3888 |
| Marnie's Impidimp / Munkidori | 0.0126 | 0.1784 | +0.1658 | 0.5958 |
| Alakazam / Dunsparce | 0.2124 | 0.2572 | +0.0448 | 0.4428 |
| Starmie | 0.0710 | 0.0924 | +0.0214 | 0.5406 |
| Cubchoo / Gravity Gemstone | 0.0016 | 0.0569 | +0.0553 | 0.4058 |
| Solrock / Okidogi | 0.0018 | 0.0512 | +0.0494 | 0.5209 |
| Meowth ex / Mega Kangaskhan ex | 0.0054 | 0.0464 | +0.0409 | 0.4958 |
| Marnie's Grimmsnarl / Impidimp | 0.0081 | 0.0386 | +0.0305 | 0.5606 |
| Crustle | 0.0045 | 0.0376 | +0.0331 | 0.5052 |
| Comfey / Chandelure | 0.0037 | 0.0351 | +0.0314 | 0.5000 |

This changed the benchmark problem. A panel that only asks "Can I beat Alakazam and Archaludon?" is no longer representative. It also has to ask:

```text
Can the agent handle Marnie pressure?
Can it survive Starmie speed?
Can it close games against wall / library-out shapes?
Can it avoid being fooled by weak local proxy pilots?
```

### Pilot repair before candidate conclusions

The most important July 4 engineering change was not a new deck. It was a benchmark-panel repair.

The priority repair pass built or restored proxy pilots for Solrock/Okidogi, Crustle, Meowth/Kangaskhan, and Marnie/Grimmsnarl. The useful coverage moved materially:

| Selection | Fidelity-pass panel weight | Attached pilot weight |
|---|---:|---:|
| Previous strict selection | 0.6839 | 0.7962 |
| After priority repair | 0.8577 | 0.8851 |

This matters because a candidate can look excellent against weak proxies and then collapse live. The repaired panel does not make the benchmark perfect, but it moves it from "too incomplete to trust" toward "good enough to rank a short list."

Remaining gaps were still real:

| Archetype | Status | Why it matters |
|---|---|---|
| Comfey / Chandelure | pilot missing | Small but non-trivial share; unusual win mechanism. |
| Cubchoo / Dunsparce | pilot missing | New rising pressure row. |
| Dragapult | needs fidelity | Low share, but high observed score and useful stress behavior. |
| Hop / Trevenant | fidelity fail | Falling share, but still a historical stress pattern. |

### Larger repaired-panel confirmation

After that repair, I reran a larger confirmation benchmark:

```text
Run: fresh_repaired_panel_confirmation_20260704_gps32
Candidates: 5
Opponents: 16
Games per seat: 32
Local games per candidate: about 960-1024
```

Raw score rates:

| Candidate | Games | Raw score rate | Wilson low |
|---|---:|---:|---:|
| live A Archaludon reference | 1,024 | 0.7266 | 0.6985 |
| public0627 v62 Lucario | 1,024 | 0.7236 | 0.6954 |
| i-have-one-rear-card | 960 | 0.7198 | 0.6905 |
| Phantom Dragapult | 960 | 0.7177 | 0.6884 |
| live B Alakazam reference | 1,024 | 0.6621 | 0.6326 |

Field-weighted scores over the repaired panel:

| Candidate | Weighted score | Validated weighted score | Reading |
|---|---:|---:|---|
| public0627 v62 Lucario | 0.7718 | 0.7748 | Best single-agent weighted read, but not necessarily the best portfolio partner. |
| live A Archaludon reference | 0.7715 | 0.7735 | Essentially tied with Lucario; strong A-style anchor. |
| i-have-one-rear-card | 0.7580 | 0.7665 | Not the highest standalone bar, but strong complement shape. |
| Phantom Dragapult | 0.7509 | 0.7509 | Viable pressure challenger, no longer the sole best B-slot answer. |
| live B Alakazam reference | 0.7089 | 0.7164 | Weakest in this confirmation set. |

The pair view was the most useful part:

| Pair | Mean best score | Min member Wilson low | Both below 50% rows |
|---|---:|---:|---:|
| Lucario v62 + i-have-one-rear-card | 0.8281 | 0.6905 | 0 |
| i-have-one-rear-card + Phantom Dragapult | 0.8281 | 0.6884 | 0 |
| live A Archaludon + Phantom Dragapult | 0.8115 | 0.6884 | 0 |
| live A Archaludon + i-have-one-rear-card | 0.8094 | 0.6905 | 0 |
| Lucario v62 + Phantom Dragapult | 0.7906 | 0.6884 | 2 |

This changed the submission reading. Earlier smaller runs made Phantom Dragapult look like the clean B-slot challenger. The larger repaired-panel run made the conclusion more careful:

```text
Phantom Dragapult is viable.
i-have-one-rear-card has the stronger complement signal.
live B Alakazam is weak in this repaired-panel evidence.
Lucario / Archaludon remain the strongest single-agent references.
```

So the robust conclusion is not "submit Phantom immediately." It is:

```text
The current B-slot should be questioned.
The replacement should be judged by portfolio complement, not just standalone weighted score.
```

### What this changed in the research policy

This update reinforced three rules.

First, **deck selection is not enough**. The field may reveal a strong archetype, but until the pilot is faithful enough as an opponent or strong enough as a submission, the deck is only a hypothesis.

Second, **pilot repair can change benchmark conclusions**. The same candidate can move up or down when the opponent panel stops misplaying important archetypes. That is a feature, not a bug: it means the benchmark is becoming less self-deceptive.

Third, **the two-agent objective is a portfolio objective**. The best single-agent bar is not automatically the second submission. A second slot should cover matchups the first slot misses and avoid correlated failure.

As of this update, my practical read is:

| Slot role | Current interpretation |
|---|---|
| A-style anchor | Archaludon / Lucario-style references remain the strongest single-agent anchors. |
| B-slot complement | i-have-one-rear-card has the best current complement signal, with Phantom Dragapult still viable. |
| Known weak point | Alakazam B is familiar and live-tested, but local repaired-panel evidence no longer protects it. |
| Next research target | Marnie, Starmie, Comfey/Chandelure, Cubchoo/Dunsparce, and Dragapult pilot fidelity. |

This is the kind of update I want the working-note series to preserve: not only which candidate was chosen, but why the interpretation changed.

---

## 16. Conclusion

The first week changed my view of the competition.

This is not a simple "find the best deck" contest. It is a three-layer game:

```text
deck discovery
pilot quality
meta timing
```

The June 30 logs made that especially clear. Alakazam became the largest share deck because it answers Archaludon. Marnie/Munkidori emerged because it appears to answer both. Archaludon remained locally useful because a clean, strong pilot can outperform average field Archaludon, even when the archetype itself is no longer the raw field leader.

The practical lesson is that a strong weekly system must be honest about evidence:

- logs can identify decks;
- behavior profiles can explain why they win;
- benchmarks can test runnable agents;
- pilot fidelity determines whether those benchmarks mean anything;
- live scores calibrate the whole loop.

That is the direction for the next note: move from meta discovery to pilot acquisition, especially for the emerging decks that the logs already identify but the local agent cannot yet play well.
