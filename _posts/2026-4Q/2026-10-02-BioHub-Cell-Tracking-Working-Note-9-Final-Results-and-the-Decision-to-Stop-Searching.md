---
title: "BioHub Cell Tracking Working Note 9: A Big Plan and a Weak Start"
date: 2026-10-02 19:00:00 +0900
last_modified_at: 2026-10-03
categories: [AI, Kaggle]
tags: [kaggle, biohub, ai-agents, cell-tracking, competition-retrospective, lineage-reconstruction, microscopy, model-portfolio, oof, research-methods, working-note]
lang: en
slug: BioHub-Cell-Tracking-Working-Note-9-Final-Results-and-the-Decision-to-Stop-Searching
math: true
pin: false
hide: false
published: true
image:
  path: /assets/img/posts/2026-10-02-biohub-working-note-9/cover.png?v=a3ef7423b8db
  alt: "BioHub Cell Tracking Working Note 9: A Big Plan and a Weak Start"
topic: biohub
translation_key: biohub-09
series: biohub
series_order: 9
---

<style>

/* Local to the BioHub manuscripts; labels follow each table's own headings. */
.content .table-wrapper:has(> table.biohub-table) {
  max-width: 100%;
  overflow-x: auto;
  container: biohub / inline-size;
}
.content .table-wrapper > table.biohub-table {
  table-layout: fixed;
  width: 100%;
  min-width: var(--table-min, 0);
  font-size: 0.92rem;
  line-height: 1.6;
  font-variant-numeric: tabular-nums;
}
.content table.biohub-table th,
.content table.biohub-table td {
  padding: 0.6rem 0.7rem;
  white-space: normal;
  overflow-wrap: anywhere;
  vertical-align: top;
}
html[lang="ko"] .content table.biohub-table { word-break: keep-all; }
.content table.biohub-table th:nth-child(1) { width: var(--c1); }
.content table.biohub-table th:nth-child(2) { width: var(--c2); }
.content table.biohub-table th:nth-child(3) { width: var(--c3); }
.content table.biohub-table th:nth-child(4) { width: var(--c4); }
.content table.biohub-table th:nth-child(5) { width: var(--c5); }
.content table.biohub-table th:nth-child(6) { width: var(--c6); }
@container biohub (max-width: 620px) {
  .content .table-wrapper > table.biohub-records {
    display: block;
    min-width: 0;
    border: 0;
  }
  .content table.biohub-records thead {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
  }
  .content table.biohub-records tbody { display: block; }
  .content table.biohub-records tr {
    display: block;
    margin-bottom: 0.9rem;
    border: 1px solid var(--tb-border-color, #9996);
    border-radius: 0.3rem;
  }
  .content table.biohub-records td {
    display: block;
    width: auto;
    border: 0;
    text-align: left !important;
    padding: 0.45rem 0.75rem;
  }
  .content table.biohub-records td:first-child {
    font-weight: 600;
    border-bottom: 1px solid var(--tb-border-color, #9996);
    padding-block: 0.65rem;
  }
  .content table.biohub-records td:last-child { padding-bottom: 0.75rem; }
  .content table.biohub-records td:not(:first-child)::before {
    display: block;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted-color, #6c757d);
    margin-bottom: 0.1rem;
  }
  .content table.biohub-records td:nth-child(2)::before { content: var(--label2); }
  .content table.biohub-records td:nth-child(3)::before { content: var(--label3); }
  .content table.biohub-records td:nth-child(4)::before { content: var(--label4); }
  .content table.biohub-records td:nth-child(5)::before { content: var(--label5); }
  .content table.biohub-records td:nth-child(6)::before { content: var(--label6); }
}

@container biohub (max-width: 575px) {
  .content table.biohub-numeric:has(th:nth-child(4))::before {
    content: "↔ Scroll horizontally to see all columns.";
    display: table-caption;
    text-align: left;
    font-size: 0.78rem;
    color: var(--text-muted-color, #6c757d);
    padding-bottom: 0.35rem;
  }
}
@container biohub (max-width: 703px) {
  .content table.biohub-numeric:has(th:nth-child(5))::before {
    content: "↔ Scroll horizontally to see all columns.";
    display: table-caption;
    text-align: left;
    font-size: 0.78rem;
    color: var(--text-muted-color, #6c757d);
    padding-bottom: 0.35rem;
  }
}
.content mjx-container {
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
}
.content mjx-container[display="true"] { padding-block: 0.25rem; }
.content details { min-width: 0; }
.content summary { cursor: pointer; }
.content code { overflow-wrap: anywhere; }

.content .biohub-figure { display: block; max-width: 100%; border: 0; }
.content picture, .content picture img { display: block; max-width: 100%; }
.content picture img { width: 100%; height: auto; }
.content mjx-container { max-width: 100%; overflow-x: auto; overflow-y: hidden; }
main h1, .content h2, .content h3, .content h4 { word-break: keep-all; overflow-wrap: break-word; }
.content table.biohub-score th:first-child, .content table.biohub-score td:first-child {
  position: sticky; left: 0; z-index: 1;
  background: var(--main-bg, #fff); white-space: nowrap;
}
.content table.biohub-score td:nth-child(3), .content table.biohub-score td:nth-child(4) {
  white-space: nowrap;
}
@media (max-width: 620px) { .content .biohub-fig-03-component-and-composition { aspect-ratio: 760 / 2070; } }
@media (max-width: 620px) { .content .biohub-fig-02-research-verdicts { aspect-ratio: 760 / 1764; } }
@media (max-width: 620px) { .content .biohub-fig-01-private-submissions { aspect-ratio: 760 / 1030; } }

</style>

[Previous: Working Note 8]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two/) · [한국어]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-9-Final-Results-and-the-Decision-to-Stop-Searching-KR/)

> **About this series.** BioHub asks competitors to reconstruct cell lineage graphs from 3D microscopy movies. The 199 labeled training movies come from two embryos; Public covers 29% of the hidden test from unseen embryos, and Private covers the other 71%. In Notes 1–8, I recorded my decisions during the competition. Here, I revisit my research using the final results I checked on October 2, 2026, and the solutions released after the competition.
{: .prompt-info }

I began BioHub with a large plan. I wanted to learn from 3D microscopy, train useful detection and association models, and build a substantially better cell-tracking system. I rented GPU capacity for more than a month and used AI agents to help implement and investigate the ideas. The result fell far short of the performance I had hoped to reach.

What bothers me most is how I used that opportunity. Even with two or three weeks left, I often could not identify the next experiment worth running. I left the GPU idle, or spent time on calculations that did not help me choose a model or improve a prediction. There was still work being done, but I had difficulty connecting it to a credible route toward a much stronger system.

The problem began earlier. I spent too long repairing code before establishing a reliable path from data to scored predictions. I kept improving one pipeline without establishing a sustained development and comparison path for alternatives. As its models, calibration and graph rules became more closely tied together, changing an early assumption required changes elsewhere. By the time I recognized those dependencies, correcting course had become expensive.

This is the thread I want to follow through the retrospective: **how a weakly established starting point made later research harder to redirect, and what I can learn to recognize earlier.** The first eight notes record the project as it unfolded. Here I revisit that history alongside the solutions published after the competition, keeping what I could have known then separate from what I learned afterward.

## 1. The result fell far short of what I had aimed for

The October 2 official account query returned 175 submissions, including 162 COMPLETE submissions with Private scores. I selected v93 and v92, and finished **445th**, with v93's **0.91920** as my final score. That score was far below what I had set out to achieve.

Several changes did help. The Private directions of v93 over v92 and v92 over v90 were positive, so the project did produce useful improvements that transferred to the hidden evaluation. The important question is why those improvements remained within a system whose overall performance was still disappointing.

V94 had the highest Private score among my returned scored submissions. Selecting it would have improved the final result slightly; it would still have left me far from the performance I wanted. I keep the exact comparison in the appendix because it describes a small difference among candidates already built. It does not explain why I had not built a substantially stronger system.

To understand that larger gap, I need to start with how I turned ideas into executable experiments.

## 2. I tried to repair too much before proving the complete path

Early on, I tried to use the agents' help to fix every bug I encountered. After several days of repeated repair, I began to realize that trying to make the whole codebase sound was an open-ended task. I remember spending more than ten days' worth of attention, time and tokens in these loops. The surviving records let me check particular interruptions, but not reconstruct a complete token bill or prove that the entire period was unproductive.

Between July 1 and July 10, the RunPod logs record missing staging prerequisites and an extraction helper, a DataLoader mask-shape failure, launch options the actual Center trainer did not accept, and inconsistencies in run modes and checkpoint names. These were concrete interface problems. Repairing the launcher did not help if the transported trainer expected different arguments; repairing the trainer did not settle whether inference would load the intended checkpoint.

There was real training alongside this work. Offline scoring was available on June 30, TemporalUNet training began on July 2, and substantial training continued during the following days. My mistake was not spending every moment on code instead of models. It was failing to establish one complete execution path that I could use to check both kinds of work.

### The software lesson was about boundaries, not fixing everything

With little software-engineering experience, I did not initially see how many assumptions crossed file boundaries: tensor axes, sampling conventions, command-line arguments, checkpoint layout, coordinate conversion and graph serialization. I could repair one failure without knowing whether the next stage agreed with the repair. A passing local test was reassuring, but it often checked a smaller claim than the experiment needed.

The practical lesson I took from this is to define the smallest complete result first. For this project, that would have meant loading a real training batch, checking its actual shapes and masks, saving and resuming the intended checkpoint, predicting one real movie, writing and reloading its graph, and calling the official scorer. A second movie would check that the path was not accidentally specific to the first. I already had parts of this sequence; I should have connected them before extending parallel launchers and general machinery.

After that, each repair could have answered a concrete question: what prevents the next comparison from running? I could ask an agent to fix that boundary and demonstrate the actual consumer, while leaving healthy training alone. A new model might need an adapter. It would not need to inherit every convenience and general check before producing a useful prediction.

The August 20 snapshot shows the same issue more clearly. **None of 331 planned CPU scientific tasks had launched** while verification and orchestration were being repaired. One verifier took **57 minutes 13.424 seconds** and reread about **696 GB**. Later failures involved a process banner absent from a mock, file-time assumptions after transport, and a historical `bytes` field where a verifier expected `size`. Those defects were real, but resolving the chain had become a prerequisite for starting the scientific worker.

I learned some useful engineering habits through this: trace the exact producer and consumer, test real inputs across the boundary, preserve working checkpoints, and distinguish a repair from a change in the scientific question. That was valuable for someone starting with so little engineering knowledge. The lesson is also a limit on how I should learn it: build the knowledge needed to run and interpret the next experiment, rather than attempt to perfect an expanding system before doing research.

### I also needed to verify what my local score was measuring

An executable path is not enough if it evaluates a different pipeline. As described in [Note 6]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/), the September 2 research-path correction changed the embryo-out replay from **0.6005 to 0.7499** on 199 movies. Both used embryo-out backbone assets; 0.7499 was not the notebook's own hidden output. This was a correction to the comparison instrument, not a new model gain.

I should have checked that alignment earlier: preprocessing, fitted model roles, proposal selection, graph-stage order and serialized outputs. Otherwise a carefully calculated local result could still answer the wrong deployment question. The foundation I lacked was therefore both executable and scientific: a path that ran reliably, and an understanding of what comparison it actually made.

## 3. I needed to define the learning problems before choosing more models

Once a baseline worked, I put considerable effort into improving it. That was a reasonable starting point, but it did not decide where the next large improvement could come from. I needed a more explicit account of what information the system lacked and which model would learn it.

For BioHub, the questions were concrete. Could the detector localize faint nuclei in crowded volumes? Did the representation distinguish nearby cell identities, or mainly cells from background? Did sparse annotation turn an unmarked real cell into a training negative? Did a division crop contain the mother and both daughters? Could a later graph stage erase useful evidence produced upstream?

Those questions connect domain knowledge to model design. They also explain why collecting architecture names was not enough. Two backbones can learn the same easy task and share the same important errors. A smaller model with the right temporal target can provide information that a stronger cell-versus-background detector never learned to represent.

### How much biology and microscopy should I have studied?

I do not think I needed to master developmental biology before making predictions. I did need enough domain understanding to define a training example, recognize an ambiguous label and interpret a failure. In this competition, that included anisotropic voxel spacing, how cells appear in XY and XZ views, occlusion in depth, the timing and appearance of division, stage motion, and what sparse annotations did or did not say about unmarked cells.

I would now learn those points through a small, deliberate error panel. I would examine a correct detection, a faint miss, a dense-neighborhood identity swap, a true division, a false fork, and a depth-ambiguous case. For each, I would put the raw views, annotation, model input and prediction side by side. If I could not explain the target or distinguish uncertainty from an error, I would read the relevant domain material or clarify the annotation convention before designing the learner.

That gives me a useful stopping rule for domain study. I need more of it whenever it could change the target, crop, negative labels, augmentation or interpretation of an error. Once those are coherent, I can concentrate on the machine-learning problem: representation, loss balance, initialization, sampling, optimization and calibration. If those methods fail on an example that should be learnable, I return to the domain assumptions rather than merely trying a larger backbone.

The winner's later account makes this boundary vivid: inspecting images helped reveal a different temporal task, not just supply more annotations. I could not have known that recipe in advance. I could have recognized that my own image inspection should influence the question I asked a model to answer.

### An early portfolio should have protected different questions

I would have started with one reliable baseline and one alternative aimed at a specific missing signal. A targeted error investigation would decide where to invest next. This is manageable for a solo participant using agents; it does not require training every family, fold and seed at once.

I did consider portfolios and interactions at the time. The July 15 structural roadmap warned that a blend tested under the anchor's optimized post-processing did not fairly test the blend's own best configuration. The August 3 work produced several submitted association and division compositions. The principle was present, but I did not consistently turn it into a development schedule that gave a distinct representation sufficient training and a compatible path to predictions.

That gap between planning and execution is central to the story. As I kept improving the baseline, I needed to ask whether the alternative was also learning, whether its output had a usable consumer, and what one measured result would justify its next investment. Without those questions, a large plan could become a sequence of local modifications to the system I already knew how to run.

## 4. A strong model had to be strong at the decision I needed

Before giving an alternative more time, I needed to distinguish a promising learning problem from a model that merely had a convincing name or scalar score.

| Role | Evidence I should have asked for | Evidence that was insufficient by itself |
|---|---|---|
| Detector | Annotated-cell recovery and localization across density, brightness and motion; the consequences of its proposals in the finished graph | More peaks, a lower training loss, or a famous architecture |
| Appearance or association model | Separation of the actual competing parents or successors, including hard nearby alternatives; improvement in completed links | A high AUC on easy pairs or features trained only for cell-versus-background detection |
| Division model | Useful ranking near the operating threshold on production proposals; supported mother–daughter choices that survive graph construction | Good classification of GT-centered crops alone |
| Ensemble member | A measured contribution to a frozen combination at an affordable complete runtime | A different seed, low error correlation, or a standalone score close to the incumbent |
{: #biohub-table-2 .biohub-table .biohub-records style="--c1: 22%; --c2: 41%; --c3: 37%; --label2: 'Evidence I should have asked for'; --label3: 'Evidence that was insufficient by itself';" }
Training maturity is part of this judgment. Before treating a loss as a fair test of a model, I needed to know what the sampler had shown it, whether relevant positives and hard competitors were present, whether its curves were still improving, and whether it could fit a small controlled training problem. Inner-development predictions should measure the same localization or ranking task that the deployed system requires.

An epoch count cannot answer all of that. It depends on the loader's sampling cycle; a long run can repeatedly see easy examples. Nor does a high global AUC guarantee useful ranking among the rare proposals near the operating boundary.

My records contain examples of both premature judgment and genuine negative evidence. CandidatePU G3 took about **13.4 GPU minutes** to train, while admission and exact replay took about **1.89 hours**. Those timings do not prove undertraining; they show why a cheaper comparison of capacity, exposure and learning endpoint should have preceded a large replay of the first formulation. An earlier claim that CandidatePU was undertrained merely because its best checkpoint was at epoch 48 was withdrawn: it had run **57,600 steps per fold**.

Spotiflow's Exposure256 run lasted about **599.9 seconds**, failed detector and candidate-coverage prerequisites, and never reached held-out graph scoring; its association head was not fitted. I could conclude that this training condition had not prepared the intended comparison. I could not treat it as a completed negative graph result for a mature model family.

The clean pseudo A2 comparison was different. Control and student each supplied their own detection and association, under historically selected graph settings. The student recovered more nodes, but its reciprocal four-movie graph score was **−0.0205909080** below control. I had a real reason to reject that composition. Automatically expanding the same losing recipe to 199 movies would not have answered a new question.

So how much time should I have given a promising model? Enough to reach a declared, informative learning endpoint, followed by a fresh investment decision. A first run establishes exposure, throughput and whether the target can be learned. Useful learning earns compatible inference and a small complete comparison. A model that cannot learn needs one diagnosed change to its support or objective, or a concrete reason to defer it. A mature composition that loses needs rejection within the scope actually tested.

This sequence matters because my public weights were useful enough for several stronger solutions to build on. The detector was not worthless. I needed to understand which identity or event information it lacked, then develop a model and consumer for that information. That brings me to the comparison that repeatedly complicated the project.

## 5. Adding a component and developing another pipeline were different experiments

I used the shorthand A+B+C for a detector, an association model and a graph policy. The plus signs describe a pipeline using those components; they are not arithmetic or a claim that the components' score gains add up. To make the relationships explicit, let the established pipeline be **P₀ = (A₀, B₀, C₀)**, with operating choices θ₀ fitted for it. These choices include proposal gates, confidence thresholds, graph costs, repair rules and blend weights.

Let **D** be one specified additional source of evidence or mechanism, with its training recipe and any new settings fixed in advance. Adding it to P₀ while retaining the shared operating choices θ₀ produces a modified pipeline **P_D**. The comparison

$$
\Delta_D(\theta_0)=S(P_D;\theta_0)-S(P_0;\theta_0)
$$

asks whether D helps that pipeline at those settings. S is the score on the stated evaluation population. A zero or negative result is useful: the unchanged system does not benefit from this addition as tested. It does not answer whether a different, compatible use of D could work.

For example, D may add candidate cells, change feature scale or alter the relative confidence of ordinary links and division proposals. A linker fitted on the old candidate population may rank the newcomers badly. A graph policy fitted on the old score distribution may suppress useful proposals. A repair stage may overwrite them. These are specific compatibility hypotheses I could test, not reasons to assume every failed component deserves another attempt.

There are two distinct alternatives to that plug-in experiment. One is to keep the fitted components of P_D but fit a small, declared adaptation θ_D on training-side data—for example, score calibration or graph cost settings. The other is to build **P₁ = (A₁, B₁, C₁)** as a separately developed pipeline. The subscripts identify different versions in the same functional roles; they do not imply that all three components must change. For example, **P₁ = (A₀, B₁, C₁)** retains the detector and replaces only the identity learner and graph policy. Refitting an association head changes B and belongs in this second description, rather than being hidden inside an operating-setting change. If its architecture does not split into these roles, I would describe P₁ directly instead of forcing it into this shorthand.

A separately developed pipeline is neither an automatically stronger version nor simply P₀ with D appended. Its learning objective, proposal population and fitted downstream choices θ₁ have to be stated. The labels P₀, P_D and P₁ identify comparison arms; they say nothing by themselves about which will score higher.

| Frozen arm | Question it answers |
|---|---|
| P₀ at θ₀ | What is the current complete reference? |
| P_D at the unchanged θ₀ | Does D help as a plug-in? |
| P_D with a declared training-side adaptation θ_D | Can a specific compatibility change make D useful? |
| P₁ with its own declared training and operating choices | Does another complete prediction route deserve further investment? |
{: #biohub-table-3 .biohub-table .biohub-records style="--c1: 35%; --c2: 65%; --label2: 'Question it answers';" }
Not every idea needs all four arms. I would choose the smallest set that distinguishes the plausible explanations, before observing held-out results. Adaptation would have a budget and endpoint and use training-side data or explicitly labeled screening evidence. Repeatedly retuning on the evaluated embryos and calling the best result fresh OOF would simply introduce another selection problem.

There is also a maturity issue. A well-developed P₀ beating an early P₁ tells me which one to deploy now. Without evidence about P₁'s exposure, learning trajectory and compatibility, it tells me little about whether a further bounded development step is worthwhile. That investment decision comes before the final adoption comparison; it does not exempt P₁ from eventually competing against the best complete reference.

The July roadmap already recognized this for blending. I needed to carry that distinction through the experimental schedule. Instead, some research remained concentrated on additions at the established operating point, while broader conclusions were drawn about what the alternative could contribute.

{::nomarkdown}
<a class="popup img-link biohub-figure" href="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-03-component-and-composition.png?v=ce142e053151">
<picture>
  <source media="(max-width: 620px)" srcset="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-03-component-and-composition-mobile.png?v=9061502684d4">
  <IMG class="biohub-fig-03-component-and-composition" src="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-03-component-and-composition.png?v=ce142e053151" alt="A plug-in test and a separately developed prediction path" width="1800" height="1310" loading="lazy">
</picture>
</a>
{:/nomarkdown}

_Figure 1. P₀ is the established pipeline; P_D adds D; P₁ is a separately developed pipeline with explicitly stated components. The two routes answer different questions. This is an experiment-design schematic, not a measured optimization landscape._

## 6. My continuation criteria made correcting course harder

The comparison problem affected how I interpreted thresholds and negative results. I needed to distinguish a rule governing predictions from a rule governing the next research investment.

The **candidate gate** determined which actions could exist. In the late division diagnosis, the 12 µm parent gate alone excluded 25 annotated events. A wider gate would admit more true and false proposals; it would still need a useful ranking and graph policy. The gate's current failures did not measure the quality of a learner trained for a different candidate pool.

The **verifier's 0.90 threshold** governed which proposed divisions could change the graph. On the recorded 151-event embryo-out population, 26 events were recovered, while 63 reachable-but-missed events scored below that threshold. This identified a ranking and operating-point problem. It did not show that admitting those proposals would improve the graph: false divisions and ordinary-edge damage had to be measured together.

The **research advance bar** governed continuation or deployment. Late K5 required a pooled gain of at least **+0.004**, nonnegative point estimates in both embryo prefixes, and its other conditions—not +0.004 in each prefix. The local H4 ensemble was **+0.0007159 over H1**, below its separate **+0.002** advance threshold, although it passed K5 against an older reference.

Those can be sensible adoption criteria for a finished candidate. They are incomplete investment criteria for an immature approach. If every learning step must already clear the complete-system bar, a route requiring new supervision, a new consumer and calibration can be rejected before I have established whether it learns useful information. Conversely, evidence of learning alone is not a reason to deploy it. I needed both decisions, in the right order.

### What a negative result actually ruled out

The **+0.000365** GT-assisted result on 3,178 frozen actions in [Note 5]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/) measured a limited filter. It could not create missing events, introduce a representation or enumerate every interacting edit. Larger GT-assisted headroom in another division space was not a learnable gain either. The difference showed that an oracle's scope is defined by the actions it permits.

A training-free native-patch NCC screen had an entry target of **+0.03 AUC** over a distance-only reference at **0.99538784**. That target exceeded one. Other signal and sample-support conditions also failed, so the impossible bar was not the only problem. But the exercise still did not measure a developed contextual appearance learner. I needed to inspect both the entry criterion and the conclusion drawn from it.

GO1/GO2 illustrates the distinction between a small warning and a complete result. The 16-movie panel was pooled **+0.012533**, with one prefix at **−0.001389** and about half the gain from one recovered division. After reopening the question, the 199-movie result was **+0.002100**, both prefixes positive, with a lower bootstrap bound of **−0.000556**. It failed the +0.004 advance bar. This did not reveal a large missed stable improvement, but it did show why a small-panel sign and a full continuation verdict were different measurements.

{::nomarkdown}
<a class="popup img-link biohub-figure" href="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-02-research-verdicts.png?v=29b04b56f94f">
<picture>
  <source media="(max-width: 620px)" srcset="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-02-research-verdicts-mobile.png?v=edceded37e12">
  <IMG class="biohub-fig-02-research-verdicts" src="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-02-research-verdicts.png?v=29b04b56f94f" alt="Different findings call for different next decisions" width="1800" height="1210" loading="lazy">
</picture>
</a>
{:/nomarkdown}

_Figure 2. I need to distinguish a completed losing composition, a model that has not learned under the tested conditions, incomplete execution, and a diagnostic limited to particular actions. Each supports a different next decision._

### The local-minimum analogy describes my process, not an inevitability

The September 17 re-examination described the incumbent as effectively the best system reachable under the remaining constraints. I accepted a conclusion broader than the comparisons established. Genuine negative scores, limited oracles, insufficient training and estimates of implementation cost did not together measure every viable representation or complete alternative.

The useful part of the local-minimum analogy is the path dependence. I had tuned several stages together; an alternative could require temporary regressions while its learner and consumer were developed. My continuation rules made that intermediate work harder to justify. The absence of a ready alternative then became a reason to stop, even though the earlier allocation of time helped explain why it was not ready.

I cannot establish that failure was inevitable, or that a much stronger solution lay just beyond a measurable potential barrier. What I can identify is a process that made revision expensive and then treated its own limited alternatives as evidence that little remained to investigate. Near the deadline, stopping for cost could still be reasonable. Earlier, I needed to keep a specific alternative developing and demand conclusions limited to what I had actually tested.

The postcompetition write-ups help make that corrective work concrete. They show how other participants turned observations about cells, labels and errors into different learning problems, then connected the resulting models to complete predictions.

## 7. What the write-ups helped me see in my own experiments

Reading the solutions after the competition, I recognized many of the questions I had already encountered. I had trained division models and a contrastive temporal representation, tried independent detectors, compared alternative graph hypotheses and joint division decisions, acquired external microscopy, and manually reviewed 1,000 candidate events. Several of those directions were already in my July plans. The difference became clearer when I followed each experiment beyond its first result: what had the model learned, what information was still missing, and how was its output supposed to change tracking?

The strongest examples connected those questions. An observation in the images led to a target; the target determined the supervision and samples; the resulting predictions had a defined role in the graph. When a result disappointed, the next change addressed a particular break in that chain. I had often identified a break too. What I had not developed consistently was the next formulation needed to investigate it.

These are the authors' accounts of methods released after the competition, rather than experiments I independently reproduced. I could not have known their final recipes or which would succeed. What I can compare is their development reasoning with the signals already present in my own project.

### 7.1 First, I needed to understand what the loss was teaching

The [fourth-place author](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744673) describes a detector whose sparse annotated tier contributed less than 0.5% of the total loss. Its probability head collapsed toward zero. The repair normalized positive, background and unknown tiers separately, so voxel counts no longer determined their relative influence. The [14th-place account](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744486) describes a related mismatch in division-flow training: weights intended to balance cells did not balance the actual mass of negative voxels. Both examples begin by examining the learning problem the loss really created.

I had a similar signal in July. My temporal embedding model's center head deteriorated while association continued improving. The record diagnosed domination by dark-background voxels and proposed region-balanced PU loss and head-specific checkpoints. I pursued reuse of association and motion. Looking back through the records, I could not find a completed successor that separately normalized this center objective and developed its detector path. I did investigate other detectors, but I did not find a result answering this particular diagnosis.

That diagnosis was enough to justify a small learning comparison. I could have held examples, initialization and exposure fixed, compared the existing loss with one tier-normalized loss, and checked whether each could fit a tiny training set. On inner-development movies, I would then have compared center recall at a similar candidate load, alongside each tier's contribution and each head's trajectory. Because the new loss could change probability scale, I would have calibrated the threshold inside training.

The outcome would have determined the next investment. A recovered center head with useful inner predictions could justify its association and decoder development. Failure to fit the tiny set would direct attention to masks, normalization or supervision. Learning accompanied by excessive harmful proposals would reject that formulation at its measured exposure. This would have given me a clearer answer than the total loss or a single `best` checkpoint: was the model failing to learn its intended task, or learning a task that did not improve the graph?

### 7.2 Visual review could change the task as well as the labels

The [winner](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744801) spent substantial time on detection and manually painted roughly 1,800 cells, with disappointing detector benefit. The author credits prolonged inspection with a more useful observation: temporal images could reveal a cell's relatives and impending division even when geometry was ambiguous.

The model developed from that observation was query-conditioned. A marker specified the cell to follow; the outputs described cell state and temporal occupancy of its continuation or daughters. Crop jitter discouraged always following the central cell. Temporal-spacing augmentation exposed larger movements. Model-proposed events, checked visually, expanded division support from 151 to 515; a later round included 2,492 hard interphase examples. The two rounds each ran for 100 epochs with roughly 6,600 sampled crops per epoch and explicit class balancing. This was sustained exposure to the chosen task, rather than hundreds of thousands of independent events. A learned linker consumed the predictions while separating division probability from daughter choice.

That account made me revisit what my own visual work had actually taught the system. I had also invested time in reviewing events, and those judgments reached real models and submissions.

#### What my 1,000 judgments actually changed

On September 4–6, I reviewed two batches of 500 proposed divisions. The agents selected the ordinary candidates using the existing verifier's score, with floors of 0.5 for batch 1 and 0.4 for batch 2. Each batch included 50 blinded controls with known GT labels, used to check the judgments rather than add new training examples. I marked candidates `y`, `n` or `s`: positive, negative or uncertain. I excluded the controls and uncertain judgments from the added fitting rows and refitted the existing HistGradientBoostingClassifier division verifier. This supervised a tabular decision on the current proposals; it did not train a new temporal image model.

Different versions consumed different portions of the review. For v85, I used the first 250 judgments from batch 1, adding 136 usable rows: 58 positives and 78 negatives. For v86, I used batch 2 alone, adding 265 rows: 79 positives and 186 negatives. V87 retained that batch-2 table. Later re-review left 486 of the 1,000 judgments uncertain, including judgments on the GT controls. That was a later label collection, distinct from the table already used by v86 and v87. A count of reviewed images was therefore not a count of new examples in a deployed model.

#### What improved locally, and what happened in the submissions

At the same verifier threshold of 0.75, adding the partial batch-1 labels changed the paired local replay from 0.7535 to 0.7573: **+0.0038**. Lowering the threshold to 0.65 added **+0.0010**, reaching 0.7583. The total +0.0048 combined a label change with an operating-point change. These local comparisons shared the upstream pipeline, so they measured a conditional improvement rather than independent hidden-population transfer.

The completed submissions were mixed:

| Version | Verifier change | Public | Private |
|---|---|---:|---:|
| v83 | GT table; threshold 0.75 | 0.94437 | 0.91331 |
| v85 | Partial batch 1; threshold 0.65 | 0.93996 | 0.91373 |
| v86 | Batch 2; threshold 0.90 | 0.94347 | 0.90877 |
| v87 | Same batch 2; three added cues and imputation; threshold 0.90 | 0.94662 | 0.91217 |
{: #biohub-handlabel-scores .biohub-table .biohub-numeric .biohub-score style="--c1: 12%; --c2: 50%; --c3: 19%; --c4: 19%; --label2: 'Verifier change'; --label3: 'Public'; --label4: 'Private'; --table-min: 42rem;" }
V85's Private difference from v83 was **+0.00042**, despite its lower Public score. V87 improved Public but finished **−0.00114 Private** below v83. The table and threshold changed together in these comparisons, so I cannot attribute their entire effect to labeling. The saved submission query also contains aggregate scores, without the hidden graphs needed to identify which divisions improved or deteriorated. My [supporting records]({{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/handlabel-evidence.json) separate these submitted compositions from the local label-only comparison.

The v86-to-v87 comparison changed less. I kept the upstream models, batch-2 labels and threshold of 0.90, then added three cues about an existing nearby detection and raw temporal brightness at the proposed daughter's location, with median imputation for the expanded feature schema. That package improved the local replay by **+0.0018**, Public by **+0.00315** and Private by **+0.00340**. The comparison supports the package under this composition, without separating its individual cues. It recovered much of v86's Private loss but left v87 below v83.

#### What the review revealed about sampling and representation

Using all of batch 1 did not improve on the partial batch's local gain. False positives increased without a comparable increase in recovered true divisions. At that point, further annotation of the same kind looked less attractive. The useful question was why.

A September 5 diagnostic of the full-batch-1 recipe on the 6bba embryo half found missed GT divisions with a median verifier score of **0.084** and a 90th percentile of **0.368**. Both were far below the batch-selection floors. The diagnostic and selection fits differed, so those values cannot count exactly how many events sampling excluded. They nevertheless exposed a weakness in the labeling plan: concentrating on candidates the current ranker already found plausible could leave a separate class of low-scoring true events poorly represented.

The existing features also confused a genuine daughter already attached to another track with an unrelated neighboring cell entering the proposed fork. Additional labels could move the decision boundary, but the refit still had to use the information in that table. It could not acquire a richer temporal identity representation merely from more judgments.

Some judgments themselves required better views. I also had to correct a distance-based explanation of my judgments: distance was not why I rejected those GT-positive cases. In some 3D views, a cell appeared from behind another, making the relationship uncertain. Wider re-review moved 60 judgments from `n` to `s`, with none moved from `n` to `y`. I needed to distinguish uncertainty in the image from a confidently negative biological judgment, and make the annotation convention explicit.

These findings justified deferring thousands more labels for the unchanged recipe. They also pointed toward a different use of a smaller review: low-scoring missed events, matched false forks, clearer depth views and a comparison of what I could see with what the model received. At the time, I was using a Public reading rule under which a sufficiently low v86 score would mean that hand rows hurt hidden performance at any threshold. One submitted composition could not establish that. The final Private values now reinforce the distinction, but the limitation was already visible before Private: a poor threshold, inadequate representation and a different annotation task were separate explanations.

#### The question the daughter models still needed to answer

I also trained strict division models with hard-negative fine-tuning and exact graph comparisons. My September daughter proposal programme ran 12 endpoints of 3,000 steps each. Inner cross-fit readiness gave division AUROC 0.813/0.912, yet both daughters were localized within 3.5 µm only 0.31/0.36 of the time. The diagnosis identified a weak second peak under the shared target. This was a separate development effort from the tabular refits, and it located a different problem: recognizing a division did not mean finding both daughters of the queried parent.

Scarce division support had been apparent in July. By September, the second-peak failure supplied another concrete reason to examine the target. I could have put a fixed panel of missed and false forks in raw XY/XZ temporal views beside the model's inputs and outputs. Did the crop include both daughters? Were labels consistent? Was the model following the nearest bright object instead of the specified parent? Those questions could connect visual inspection to an actual change in learning.

One bounded training-side comparison could have tested the existing shared target against separate continuation/division targets with an explicit query marker. I would have frozen labels and exposure, retaining unknown cases as unknown. Better two-daughter coverage without excessive ordinary-cell errors could justify a compatible linker and broader evaluation. Failure to fit would call for one diagnosed target or support repair; learning without useful inner generalization would end that formulation.

The winner's account helped me articulate that follow-up, rather than supplying a guaranteed recipe. Its optimistic pseudo-label CV and uncertain isolated MAE benefit remain limits. What I needed from my own work was the link between visual evidence, the query, the learned target and the daughters the graph could actually choose. Neither annotation totals nor an epoch schedule could establish that link alone.

### 7.3 Identity had to be learned among the actual competitors

The parent-query problem led me back to ordinary links. A detector can locate a nucleus without learning enough to distinguish it from a nearby competing nucleus. That distinction became explicit in several write-ups.

The [third-place solution](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744484) trained a separate appearance matcher and treated uncertain unannotated cells carefully. The [10th-place solution](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744980) developed an independent contrastive appearance learner and a contextual linker. Before that richer model was ready, a position-only transformer provided a cheaper complete-system comparison. The [sixth-place solution](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744582) trained association around its own detections, including displaced matched centers and detector-generated distractors. The [16th-place account](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744671) updated its edge model when the detector changed its candidate population.

I had proposed identity contrastive learning and hard negatives in July, and trained the temporal model. Association improved while its center head collapsed. What I still needed to establish was whether the representation separated the difficult competitors that mattered to tracking. A later readiness disclosure made the problem concrete: in one readiness fold, **94.35%** of true successors were the nearest detections. Looking at the next frame was legitimate, but a high recovery rate on that population could be explained largely by easy geometry. It did not tell me enough about identity in dense or ambiguous neighborhoods.

A useful next comparison could have started with training-side cases where both the correct and incorrect parents survived the geometric gate. I would have kept them separate from cases where the correct node or link candidate was absent. On that fixed panel, the current representation and one task-specific identity representation could be compared using nearby competitors and careful handling of unknown links. Candidate-relative rankings on inner cases would establish whether the new information was useful. A compatible head and a small serialized-graph comparison would then establish whether the system used it.

That development sequence would have helped me distinguish learning from integration without explaining away a genuine loss. My teacher-clean A2 comparison supplied the student's own detection and association and lost under the historical graph settings. I cannot assign that loss to an old linker being forced onto a new detector. It rejected that measured composition. The identity question concerned other, specifically motivated formulations and the evidence needed before investing in them.

### 7.4 The label had to mean something the inputs could answer

Once I separated detection, identity and daughter choice, the label itself needed closer examination. Two authors changed their targets in apparently opposite directions.

The [14th-place author](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744486) moved a division-site reader from graph-TP labels toward image-site labels. A biological event visible in an image and the success of inserting a particular graph fork were different questions. The [17th-place author](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744930) reports a semantic duplicate classifier with good AUROC whose graph edits were harmful, then changed supervision toward the edit's metric effect. An image reader needed a target answerable from the image; a graph editor needed a target relevant to its action.

That distinction helps explain the limits of my division scores. The endpoints with inner GT-centered AUROC **0.813 and 0.912** scored **0.5752 and 0.6922**, respectively, on the other embryo's verifier candidate pool. Candidate population and evaluation domain/split changed together, so this was not an isolated estimate of crop-centering error. One fold had only **26 training positives**. The deployed question also included ambiguous candidate daughters and competing ordinary edges. I needed to distinguish those mismatches before concluding what another appearance model could learn.

My September 14 S4 review exposed a related problem on ordinary identity. It asked which next-frame cell continued the marked source: the GT successor or the alternative chosen by my pipeline. I reviewed **100** blinded A/B items, comprising **80** suspected swaps and **20** known-link controls, using temporal views and depth projections. On the controls, **15** judgments supported the GT successor and **five** were uncertain; none confidently selected the wrong alternative. Counting uncertainty as non-agreement gave **75%**, below the declared **85%** trust bar, so the diagnostic was inconclusive. I had reported that the yellow source marker often lay between two cells and marked those cases uncertain. The display and annotation question needed attention before my judgments could support a model.

S4 stopped at that diagnosis. I did not put the judgments into a model, fitting table or threshold, so no applied candidate or resulting score change was evaluated. This differed from the September 4–6 division review, which reached refits and submissions. A clearer display and labeling convention, followed by a small repeat, could have established whether I could answer the intended identity question reliably. The five uncertain controls, with no confident wrong choices, did not establish an inability to annotate or settle whether the GT or tracking mechanism was wrong.

The [12th-place solution](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744501) supplies a concrete example of evaluating the decision after these stages are separated. At recovery of **11 out of 109** missed divisions, its reported precision was **5.2%** for a reader alone, **32.4%** with a chooser and **84.6%** with the combined system. Those measurements were conditional on an upstream in-sample graph. They nevertheless described the proposed graph intervention much better than a global AUC: recognizing an event, choosing its daughters and accepting the fork had different error rates.

For my own follow-up, I would have frozen one production-like training proposal pool and treated those as separate questions: image evidence of division, candidate-daughter ranking, and adoption against competing ordinary links. Training or calibration would occur inside training; comparison would inspect precision at a declared recovery budget and damage in the complete graph. That would let each stage learn a target suited to its inputs, while reserving the final contribution decision for the complete system.

### 7.5 Useful proposals could need a different role

The distinction between learning a signal and accepting a graph edit also changed how I read failed detector replacements. A model might supply useful missing candidates while being a poor choice to control the whole graph.

The [second-place author](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744723) developed a detector with stronger faint-cell synthetic augmentation. It improved CV but reduced Public from **0.957 to 0.950** as the primary replacement. The final system retained it as an auxiliary source of tracks. Later recovery required evidence across time and contrast in original-resolution images. The extra detector's role changed from generating the main candidate population to proposing particular missing structures.

I had already tested multi-hypothesis graph union and tracklet admission on July 30. Broad additions hurt the official score, and a learned tabular selector produced only a small gain below its promotion bar. Yet the proposal pool contained **1,704** unique true edges missed by the anchor. Those results located the problem more precisely than a blanket rejection of additional models: useful alternatives existed in the pool, alongside additions the current selector could not safely admit. Its listed inputs summarized source, geometry, topology, persistence and probability, without raw image evidence.

That made an image-supported auxiliary role worth a bounded comparison. I could have kept candidates fixed, compared primary replacement with auxiliary use, and compared the existing selector with one incorporating declared image support. At a similar added-node load, I would have measured unique correct tracks, false scored edges, uncertain additions and the node-count contribution. The purpose would have been to discover whether the source lacked useful candidates or the selection policy lacked the evidence to choose them. If the new role still lost, I would have rejected that composition.

Second place's result does not supply constants I should have copied. Its recovery gains concentrated in a few movies, and nested parameter selection reduced the estimate. Mixed-embryo CV and selected acquisition-specific rules also limited the transfer claim. What the account supplies is a development decision: examine the job an additional model can perform before equating a failed replacement with an unhelpful source.

### 7.6 More supervision still required the right teacher targets

Changing the role of a model naturally raises another question: what should supervise that role? External data and pseudo-labels were possible ways to expand support, but the teacher's outputs needed to match what the student could learn from images.

Sixth place's teacher–student work used topology-optimized associations while preserving raw detector positions for image supervision. GT remained dominant, pseudo examples were weighted, and some targets were restricted to GT. The author reports a gain from the first round, a flat simple second round and further development involving low-contrast augmentation. The successive changes addressed properties of the training task, rather than assuming each additional pseudo-label round would help.

I also completed external-data fine-tuning, synthetic division transfer and teacher-clean student comparisons. Their weak or negative results gave me reasons to reject the tested recipes. To choose a justified successor, I needed to inspect the actual targets: positions, links and event labels; their errors; and the information the student would have to infer from pixels.

A link can be justified by global graph topology while its adjusted coordinate is a poor center for an image crop. Conversely, locating a bright center does not determine parent identity. The teacher can therefore be useful for association and unsuitable for localization, or the reverse. Agreement with the incumbent alone tells me little about whether the student gains a useful new signal.

Before another long run, I could have inspected a fixed training-only sample of targets and compared the existing recipe with one precisely changed target or weighting rule, keeping in-domain exposure and inner comparisons identical. Inadequate targets would call for repair or deferral. A learned target followed by a losing final graph would reject the complete recipe and locate any stated interaction. That sequence would connect the cost of more supervision to a measured reason for expecting it to help. Sixth place's later pseudo-label selection exposed validation too, so I would still need a separate frozen confirmation of my chosen recipe.

### 7.7 The graph had to retain the information the model supplied

Even with a useful target, adequate learning and a suitable role, the last link in the chain could fail. That happened when later graph stages undid an earlier model's decisions.

The [ninth-place account](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744835) used the foundation I had made public and found that motion relinking overwrote ILP division choices. Its final DIVCARRY restored compatible forks near the end of the pipeline, with author-reported gains of **+0.011 Public / +0.013 Private**. The [18th-place solution](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744531) similarly moved a division critic to the completed graph and let event evidence compete with ordinary links there.

I had an unusually similar signal in July. Increasing the hyperedge reward selected more raw ILP forks, but final division TP/FP/FN counts remained unchanged after a later graph filter. Zero-arm parity passed, and the documented conclusion correctly identified the downstream contract as the problem. Much of the intended intervention had disappeared before scoring. The null score could not tell me how useful those changed forks would be in a graph that retained them.

The immediate next step could have been a trace on a few real movies: follow event identities through proposal, optimization, filtering, relinking and serialization, and record the first stage that removed each changed fork. One fork-aware consumer or a later insertion point could then be compared with the unchanged control, measuring ordinary-edge damage and graph validity. This would test a particular compatibility problem. Removing the filter wholesale or increasing the upstream reward repeatedly would leave the same question unresolved.

Third place's sequence also illustrates why the earlier notation separates a fixed addition, **P_D at θ₀**, from its declared adaptation or a separately developed **P₁**. Its matcher-only step barely changed the aggregate, while a later bundle of division evidence, calibration and joint graph optimization improved much more. Several axes changed, so the later gain cannot be assigned to the matcher. The example does show why I needed a declared component-by-base contrast when there was evidence of an interaction. The completed graph, rather than an isolated module score, was where that question had to be resolved.

### 7.8 Time, data and engineering needed to support that development chain

Following the examples from target to final graph changed how I understood the resources I had provided. More data or compute could help only if it addressed a specific learning or integration problem early enough to matter.

The [fifth-place account](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744549) reports pretraining on public ZebraHub microscopy before competition fine-tuning. I eventually staged **four** external embryo datasets with more than **10,000** available division events, trained a synthetic geometry arm and ran real microscopy fine-tuning arms. I also completed the manual review described above. The remaining question was whether those sources and targets developed the information missing from my tracking system.

An earlier pretraining comparison could have tested the same task model from scratch and from external initialization, with equal in-domain exposure. I would have looked for faster convergence or a relevant inner-development improvement, using source and target examples to judge compatibility. Learning the external task without transfer would reject that source/objective or motivate one specific adaptation. A high synthetic AUC would not by itself show that the model had learned the needed visual tracking distinction.

Engineering mattered in the same way. Sixth place reports faster inference through cached encoder features and a more suitable numerical precision, making room for model and view comparisons within the notebook limit. Those changes removed a measured restriction on predictions. In my project, repairing another mock or extending a general verifier could consume time without making the next scientific comparison executable. I needed the engineering endpoint to name the comparison it enabled.

The [seventh-place solution](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744937) also helped separate research breadth from ensemble size. It built a strong single detection/motion architecture with separate event reasoning. My portfolio needed methods capable of answering different unresolved questions, with compatible prediction paths. Their contribution would determine inference budget; the experiments they enabled would determine the value of caches and adapters.

There are clear limits to reading these accounts as a promise of better results. Seventeenth place reports rejecting **seven of 19** design-half gains on a separate confirmation half, although both halves still came from the same **two embryos**. The [Public-12 / Private-95 account](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744912) trained additional models and changed multiple stages yet fell sharply in rank. Fourth place found early cross-embryo detector peaks; seventh place's longer synthetic-pretraining route helped Public and hurt Private. These cases make learning curves and frozen confirmation part of development, rather than reasons to equate more work with inevitable improvement.

What I take from the comparison is a more concrete account of where my research became too narrow. I had useful models, real experiments and several sound diagnoses. I needed to turn a selected diagnosis into its next learning task, provide the samples and compatible graph path it required, and judge that formulation at a defined endpoint. That could have produced another viable system or a better-founded decision to stop. It would have given the ambitious plan a stronger experimental foundation while there was still time to change course.

## 8. How I would turn these lessons into the next research cycle

The write-ups give me a way to make the early plan more concrete. I need to identify the missing information, choose a learner that can acquire it, give that learner meaningful support, and make sure a later stage can use its output. A portfolio should preserve those distinct possibilities long enough to compare them; a model registry with many names does not do that by itself.

| Research route | The information it would develop | First useful decision |
|---|---|---|
| Incumbent temporal detector–linker | A reliable reference and its dominant error anatomy | Which existing bottleneck offers worthwhile improvement? |
| Identity representation or compatible alternative | Evidence distinguishing close competing cells; or a detector–linker pair with different localization and support | Is the alternative learning a relevant task and making usable complete predictions? |
| Temporal event route | Mother–daughter evidence and candidate support that can survive the finished graph | Can it rank the actual event pool without unacceptable ordinary-edge damage? |
{: #biohub-table-4 .biohub-table .biohub-records style="--c1: 22%; --c2: 41%; --c3: 37%; --label2: 'The information it would develop'; --label3: 'First useful decision';" }
I would begin with one baseline and one protected alternative, adding another investigation only when its question and budget are clear. Their distinction should come from the errors they address and the information they learn. Different seeds or architecture names are useful only if they produce a measured contribution or help diagnose a learning problem.

### A concrete follow-up to the representation I already trained

Rather than start with a new model name, I would examine my saved temporal-representation predictions alongside the baseline and annotated links. I would select a fixed training-side panel where the correct parent and a plausible wrong parent both survive the geometric gate. Missing-node and missing-candidate cases would stay separate: an identity learner cannot choose an object absent from its pool.

If that panel exposed a specific appearance distinction the current representation missed, I would compare it with one successor: a task-appropriate pretrained image encoder or a small temporal encoder. I would fix the target, sample nearby competitors as hard negatives, preserve unknown links as unknown, and record actual exposure. The first endpoint would ask whether the new representation separates the relevant alternatives on inner-development cases, not whether it already produces a large complete-system gain.

Useful ranking would earn a compatible association head and a small final-graph comparison on the same node universe. I would inspect changed identity links, ordinary-edge damage and complete runtime. If learning failed, I would assess one evidence-based support or initialization change. If learning succeeded but its information disappeared downstream, I would test the named compatibility problem. If the mature composition lost, I would stop that formulation. Each outcome would make the next decision clearer.

The same sequence applies to division learning. My own labeling showed that labels alone could not supply temporal distinctions absent from the tabular inputs. I would therefore decide separately whether the next hour should improve the annotation views, cover a missed positive class, or support a different temporal target. Those are different experiments and need different outputs.

### Decide the validation boundaries early, then pay for confirmation when it is useful

My earlier idea of developing a portfolio and evaluating it later needs one qualification: I must choose the held-out boundaries before training. Otherwise an all-train teacher, checkpoint selection or fitted features can contaminate a later comparison in ways that a downstream row filter cannot remove.

This does not mean giving every immature idea a full 199-movie campaign. A few real movies establish execution. Training-side checks establish exposure and learning. A small frozen paired screen can use existing imperfect evidence when I identify its limits. Useful fixed recipes then earn reciprocal evaluation and complete notebook validation.

For a complete outer-pure comparison, the held-out embryo must be excluded from detector and linker training, teacher generation, checkpoint selection, stacked features, calibration and learned combinations. Shared all-train assets make the result conditional. Repeated choices on the same two embryos also remain part of the selection history; a later frozen run does not create new independent domains.

I would test deployment parity early enough to influence model choices: do the actual notebook stages preserve the local effect, under the same preprocessing, model roles and graph order? I would also retain credible external systems for an intact comparison when their source, assets and runtime made that feasible. A transplanted constant is a different experiment. If Public and local results disagreed, I would inspect training exposure, proposal populations, graph stages and domain response before deciding what to conclude.

### What I would ask an agent at each decision point

I can use agents to implement trainers, mine cases and trace graph changes without treating a long activity report as a research result. To make the work actionable, I would frame requests around the uncertainty I want to resolve:

- **A learning problem:** “Show a training case the model should fit, an inner case it handles, and one it misses. What relevant examples did each head see, and which curves changed?”
- **A candidate loss:** “What role did I test? Which downstream stages were fitted for these inputs? Show where the useful or harmful change survives to the final graph.”
- **A labeling decision:** “Which missed positives and matched hard negatives would the next batch add? Can I answer the question from these views, and can the model represent the distinction?”
- **A compute decision:** “What observation motivates this experiment, what is its first informative endpoint, what will it cost, and what would I do after positive, null or negative results?”
- **A decision to stop:** “Separate completed negative comparisons, insufficient training, incomplete execution and untested ideas. What is the strongest feasible remaining question, and why is it not worth the remaining budget?”

The answers would let me choose a specific next action: repair an interface, adjust one diagnosed learning condition, confirm a fixed recipe, reject a measured composition, or defer a direction for cost. I would ask for the few artifacts that distinguish the explanations—example panels, learning curves, candidate coverage and changed graphs—rather than a comprehensive new audit at every step.

I would split implementation work between maintaining the reference and developing the alternative, with a focused review when a result affects a real decision. I would retain responsibility for the question, the budget and the interpretation. This is a workable division of labor for my current skills: I do not need to write every trainer or review every line, but I do need to understand what the model is being taught and what result would change my plan.

That also clarifies what I want to learn myself. On the engineering side, I need to recognize data/model/interface boundaries and verify a complete prediction path. On the domain side, I need to identify label ambiguity and mistakes a learner cannot repair from its inputs. On the machine-learning side, I need to distinguish missing support, a poor objective, failed optimization and a genuinely inferior developed system. These are connected skills; building them around real errors is more useful to me than studying them as three unrelated subjects.

### Keep a decision ready while the current experiment runs

A successor is not ready if it depends on an undefined adapter, missing training data or an unspecified interpretation of the current job. While a run is active, I would prepare the feasible next comparison and record its dependency. If no worthwhile task remains, I would release capacity for that concrete reason. I would not fill the GPU simply to make utilization look better.

Each active research day or two should produce a prediction, a candidate decision, a resolved blocker or a runnable next comparison. A useful CPU diagnosis can be the best result of the day. Repeatedly having no such result would prompt me to revisit my allocation of time before more rental time passed.

Near the deadline, the measured cost of another learning-and-evaluation cycle would limit what I could do. That is exactly why the early plan matters. A target, representation and compatible consumer take time to develop; I cannot create them at the end by choosing among similar submissions more carefully.

## Closing

My plan was larger than the foundation I built for it. I learned useful things, trained models others could use, and made some real improvements. But I spent too much time repairing and extending machinery, then allowed an increasingly tuned pipeline to define the terms on which alternatives had to compete. By the time I tried to change that pattern, the remaining development work was difficult to fit into the schedule.

I cannot recover a hypothetical score from a different plan. I can be more precise about what I would do differently: establish one trustworthy prediction path early, inspect errors until I can state the missing learning task, develop a small number of coherent alternatives, and match each continuation decision to the evidence available at that stage. That is the practical knowledge I want to carry forward from this project.

<details markdown="1">
<summary>Appendix: the small difference within my submitted pool</summary>

These are five related submissions from the October 2 account query. The table is a record of complete configurations, not an isolated component ablation.

| Version | Main change or role | Public | Private | Final selection |
|---|---|---:|---:|---|
| v89 | Feature-E composition; diagnostic | 0.94326 | 0.91752 | No |
| v90 | Validity control with feature averaging off | 0.94662 | 0.91217 | No |
| v92 | Registration and motion-repair path | 0.94569 | 0.91418 | **B** |
| v93 | H1 association-head replacement on v92 | **0.94921** | **0.91920** | **A** |
| v94 | H4X two-seed head ensemble; diagnostic | 0.94811 | **0.92050** | No |
{: #biohub-table-1 .biohub-table .biohub-numeric .biohub-score style="--c1: 12%; --c2: 42%; --c3: 15%; --c4: 15%; --c5: 16%; --label2: 'Main change or role'; --label3: 'Public'; --label4: 'Private'; --label5: 'Final selection'; --table-min: 42rem;" }
Let S be the 162 returned COMPLETE submissions with Private scores, and P={v93,v92} my selected pair. Under the recorded best-of-two rule, the selected score was 0.91920. The hindsight difference within that submitted pool was

$$
\begin{aligned}
L_{\mathrm{select}}
&=\max_{v\in S}q(v)-\max_{v\in P}q(v)\\
&=0.92050-0.91920\\
&=0.00130.
\end{aligned}
$$

V93 was the second-best result in the pool. Replacing v92 with v90 or v89 would not change the best-of-two score. V94 was a diagnostic that had not met the recorded advancement conditions; its exact H4X deployment and the local H4 comparison were not identical compositions.

This difference is small relative to the performance gap that motivated the retrospective. I can calculate it after Private was revealed. I cannot calculate how much score a different early plan, better supervision or an unrun model would have added.

{::nomarkdown}
<a class="popup img-link biohub-figure" href="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-01-private-submissions.png?v=7663b97f998c">
<picture>
  <source media="(max-width: 620px)" srcset="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-01-private-submissions-mobile.png?v=5477ac31e139">
  <IMG class="biohub-fig-01-private-submissions" src="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-01-private-submissions.png?v=7663b97f998c" alt="Private scores of five submitted versions" width="1800" height="1030" loading="lazy">
</picture>
</a>
{:/nomarkdown}

_Figure 3. Official aggregate Private scores from the October 2 account query. I selected v93 and v92; v94 led the returned scored history. The plot describes a small difference inside my existing submission pool._

</details>

## Sources and evidence

I took the submission figures from the October 2 official account query. For the research history, I checked the first July RunPod logs, the July 15 structural roadmap, the July 28/30 coupled-pipeline analysis, the August 3 portfolio record, the August 20 execution state, the dated training and experiment receipts, and the September 17 re-examination. The early time/token-cost description is my recollection; the named interruptions are checked against the records. I have not reconstructed a complete utilization or cost ledger.

I recorded the earlier stages and relevant definitions in Notes [1]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/), [2]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics/), [5]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/), [6]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/), [7]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce/) and [8]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two/).

The postcompetition write-ups are primary author accounts. I cite their implementation details and local measurements as reported, alongside their disclosed validation limitations. I have not rerun their training or independently reproduced their comparisons. They help identify development questions that my contemporary evidence could have justified; they do not supply a guaranteed result for a path I did not run.

For the human reviews, I checked the September 4–6 batch selections, exported judgments, ingest reports, verifier comparisons and deployed notebook outputs, together with the September 14 S4 card and completed result. The linked labeling record lists their measurements and sources. I did not rerun training to reconstruct this account, and the aggregate Private scores do not reveal error causes inside hidden graphs.

I have collected the supporting evidence in the [query and experiment summary]({{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/evidence-summary.json) and the [research records and write-ups used here]({{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/research-depth-evidence.json). I reused the series' original cover illustration and created the body figures, crediting material from other sources where needed. In Figures 1–2, I explain the experiment designs and the scope of the conclusions; appendix Figure 3 shows my official submission history. [Figure sources, production and layout checks]({{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/figure-sources.json).
