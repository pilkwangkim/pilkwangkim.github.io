---
title: "Gemma 4 Developer Agent, Part 2: Measuring an Agent While Its Evaluator Moved"
description: "What a month of coding-agent experiments revealed about repeated tool calls, changing evaluators, local-to-hosted comparisons, and the risk of a twelve-hour deadline."
date: 2026-10-23 09:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, gemma-4, coding-agents, evaluation, benchmarking, working-note]
math: true
pin: false
published: true
image:
  path: /assets/img/posts/gemma-4-series/cover-2.webp
  alt: "Part 2 cover: repeated calls, sparse hosted feedback, and completion risk"
  width: 2200
  height: 1240
  fit: contain
  hide_caption: true
topic: gemma-4
lang: en
interactive_labs: [gemma]
translation_key: gemma-4-developer-agent-02
series: gemma-4
series_order: 2
---

<style>
article.technical-article .content table.gemma-eval-setup {
  width: 100%;
  min-width: 0 !important;
  table-layout: fixed;
}
article.technical-article .content table.gemma-eval-setup th,
article.technical-article .content table.gemma-eval-setup td { min-width: 0 !important; }
article.technical-article .content table.gemma-eval-setup th:first-child,
article.technical-article .content table.gemma-eval-setup td:first-child {
  width: 26%;
  word-break: keep-all;
  overflow-wrap: normal;
}
@media (max-width: 600px) {
  article.technical-article .content table.gemma-eval-setup th:first-child,
  article.technical-article .content table.gemma-eval-setup td:first-child { width: 36%; }
}
article.technical-article .content table.quick-summary {
  width: 100%;
  min-width: 0 !important;
  table-layout: fixed;
}
article.technical-article .content table.quick-summary th,
article.technical-article .content table.quick-summary td {
  min-width: 0 !important;
  vertical-align: top;
  white-space: normal;
  word-break: keep-all;
  overflow-wrap: anywhere;
}
article.technical-article .content table.quick-summary th:first-child,
article.technical-article .content table.quick-summary td:first-child { width: 32%; }
article.technical-article .content table.quick-summary code {
  white-space: normal;
  overflow-wrap: anywhere;
}
article.technical-article .content .gemma-functional-figure {
  margin: 1.7rem 0;
  border: 1px solid var(--btn-border-color, #cbd5df);
  border-radius: 0.5rem;
  overflow: hidden;
}
article.technical-article .content .gemma-functional-figure picture,
article.technical-article .content .gemma-functional-figure img {
  display: block;
  width: 100%;
  height: auto;
  margin: 0;
}
article.technical-article .content .gemma-functional-figure figcaption {
  padding: 0.8rem 1rem;
  border-top: 1px solid var(--btn-border-color, #cbd5df);
  color: var(--text-muted-color, #687787);
  background: var(--main-bg, #ffffff);
  font-size: 0.9rem;
  line-height: 1.65;
}

article.technical-article .content .gemma-functional-figure a.img-link {
  display: block;
  width: 100%;
  max-width: 100%;
}
article.technical-article .content .gemma-figure-wide { display: block; }
article.technical-article .content .gemma-figure-narrow { display: none; }
@media (max-width: 600px) {
  article.technical-article .content .gemma-figure-wide { display: none; }
  article.technical-article .content .gemma-figure-narrow { display: block; }
}
</style>


A coding agent must find the relevant code, make an edit, and deliver a patch that passes verification. [Part 1]({{ site.baseurl }}/posts/Gemma-4-Developer-Agent-From-Issue-to-Verified-Patch/) established that workflow. Over the next month, some attempts stopped before a useful edit and others produced patches that failed tests. The host also changed the software running and grading the attempts. A higher score could therefore reflect several changes; identifying an agent improvement required inspecting both the execution record and the evaluation conditions.

The execution traces and submission records showed four distinct problems:

1. **Identical calls cluster into long loops.** In one 22-task comparison without thinking, a call was followed by an identical call with probability 0.056 when its streak length was one. Once three or more identical calls had already occurred, the pooled probability of another was 0.97. A constant repeat rate misses this pattern, although the curve alone does not establish its cause.
2. **Context admission can be checked exactly.** One loop made 55 identical calls before the next request was rejected. Counting the initial call separately from the 54 additional calls makes the token arithmetic match the trace.
3. **The same package can receive different hosted scores.** One byte-identical package scored 0.10 and 0.15 within the same observed host epoch, a period grouped by public artifact changes. That is a recorded spread, not enough data to estimate a universal noise level or a minimum detectable gain.
4. **Completion matters as much as repair rate.** The twelve-hour limit can invalidate a whole submission. A slower agent may solve more individual tasks yet have less chance of producing a scored run. Local timing can expose a risk; it cannot measure the hidden server's remaining headroom.

Our working reference, **E0052**, is the public [Budget-Fit single-agent package][budget-fit]: a set of agent settings and prompts used unchanged, with attribution to its author. It differs from Part 1's unscored teaching example and the recovery packages used for early tool diagnostics. Its highest recorded Public score is **0.18**, and no tested variant has exceeded that score in our hosted records. Those scores alone cannot separate the contributions of task selection, execution variation, and runtime.

The observations have an **October 23, 2026 (KST)** cutoff. Dated leaderboard snapshots and historical comparisons retain their original conditions. An E4B submission made at **09:01 KST on October 23** is pending in that record. The competition ends on **December 2 at 23:59 UTC (December 3, 08:59 KST)**.

## Where things stand

The October 22, 06:15 UTC Public leaderboard snapshot contains 3,428 teams with scores. Four share the top score of 0.24. The highest observed score has stayed at 0.24 since October 2; that does not mean the membership of the leading group has stayed fixed. This team appears at 0.18, rank 398 in that snapshot.

<figure class="gemma-functional-figure">
<div class="gemma-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part2-fig1-leaderboard.png" alt="Numbers of teams at each displayed Public score on October 22" width="2037" height="1021" loading="lazy">
</div>
<div class="gemma-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part2-fig1-leaderboard-mobile.png" alt="Numbers of teams at each displayed Public score on October 22" width="780" height="1996" loading="lazy">
</div>
<figcaption markdown="1">

Figure 1. Team-best Public scores in the October 22, 06:15 UTC snapshot. The second row gives task-count equivalents **if** the Public denominator is 58 and the display truncates to two decimals. Section 7 explains why that is a working interpretation rather than a disclosed host specification. Teal marks our team's score group, matching the Budget-Fit reference in Figure 2.

</figcaption>
</figure>

Under that interpretation, 0.18 corresponds to 11 solves and 0.24 to 14. There are 1,459 teams at the three displayed levels equivalent to 9, 10, or 11 solves. This gap between recorded team-best scores is useful context, but it does not measure a three-task difference between the candidates' expected performance.

## 1. Tracking a changing evaluator

The host distributes the Python packages for running and grading the agent in a wheelhouse. vLLM generates model responses, the harness executes requested tools, and the verifier tests the final patch. The runtime includes this software and its execution settings. A change to any of these components can alter what an unchanged agent does. Since September 25 the host has published ten further wheelhouse versions, re-released the data, and answered open questions.

<p class="table-caption"><span class="table-label">Table 1.</span> Evaluator and rule changes relevant to comparing runs.</p>

| September 25 | October 23 |
|---|---|
| Evaluator `swegemma 0.2.7`, `adk-submission 0.2.11`, vLLM 0.19.1. | `swegemma 0.2.11`, `adk-submission 0.2.13`. vLLM is still labelled 0.19.1, but it was rebuilt twice under that same version number. |
| Half of the graph and embedding files were empty. | Fixed on September 25. Async functions are still missing from the graphs; the host says no hidden task depends on one. |
| An agent's edits to existing tests could survive grading. | In the October 8 v29 release, protected test/config sections are removed from the agent patch before it is applied; the grader also resets tracked test paths. |
| An undeclared or failing tool call ended the task and lost its work. | Since October 8, tool errors are returned to the model and the session continues. |
| `include_thoughts: false` turns thinking off; `thinking_budget` is ignored. | In the inspected current bridge, a positive thinking budget enables thinking and the budget is forwarded. `include_thoughts: false` controls retention of earlier thoughts in the history, independently of enabling thinking. |
| Tool results reached the model encoded twice as JSON; thoughts were dropped between tool calls. | Both fixed on September 30. |
| LoRA adapters were reported to load without effect. | Fixed on September 27. The host reports that adapters up to rank 64 have scored. |
| Only the 31B model. | 31B, 12B, E4B and E2B QAT variants, one base model per submission. A USD 10,000 bonus was added on October 20 for entries using E2B or E4B. |
| Distillation from other models: answer pending. | Allowed within the source models' licences, including trajectories from other open models. |
| Overrunning the 12-hour limit errors the submission. | Unchanged. The host considered scoring unfinished tasks as zero, then said the current rule stands. |
| About 120 hidden tasks. | Staff describe about 60 Public and 60 Private tasks, from private repositories. The published runner processes the splits in one evaluation. Staff retained tasks a strong commercial model solved or missed by one assertion—a solvability filter, not a claim that all tasks are hard for that model. |

A team can also have only one submission pending at a time, and the queue before scoring has ranged from
about 14 to 33 hours.

We attach each score to both the submitted bundle $b$ and an observed **host epoch** $e$, writing it as $S_e(b)$. An epoch is a period bounded by changes to public artifacts. Some boundaries concern documentation; others include executable changes. Comparing the periods therefore starts with identifying what changed. Matching epochs still does not establish identical private serving conditions.

Write $[t_s, t_c]$ for the interval from a submission to the first observation of its score, and $b_k$ for
the publication time of the artifact that starts epoch $k$. The scorer's actual adoption time is not
observable. Our bookkeeping adds a six-hour adoption margin $\delta$; this is a conservative convention, not an observed upper bound on deployment delay. A score is assigned to
epoch $k$ only if

$$
b_k + \delta \le t_s \quad\text{and}\quad t_c < b_{k+1}.
$$

Scores outside that window remain in the history, but we exclude them from epoch-matched bundle comparisons. The rule produces eleven project-defined epochs between September 23 and October 13. Public artifact changes need interpretation too: the last boundary was triggered by recreated file timestamps, yet a subsequent comparison found all 507 bulk files byte-identical and no new task content. An earlier README differed from our old local copy. The eleven groups therefore describe our observation history; they do not establish eleven distinct private scorers.

<figure class="gemma-functional-figure">
<div class="gemma-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part2-fig2-hosted-scores.png" alt="Hosted scores of every submission over time, with host updates marked" width="2100" height="1366" loading="lazy">
</div>
<div class="gemma-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part2-fig2-hosted-scores-mobile.png" alt="Hosted scores of every submission over time, with host updates marked" width="780" height="2527" loading="lazy">
</div>
<figcaption markdown="1">

Figure 2. Twenty-five submission records through October 23, plotted by submission time (UTC): 22 completed, two errored, one pending; errors and pending submissions occupy a separate categorical strip below the score panel. Dashed lines show selected public artifact updates, not observed private deployment times. v29 and v30 share a marker. The two E0053 points identify the same ZIP. The E4B submission is pending; the unchanged early recovery package spans several periods and its scores must not be pooled as same-environment repeats.

</figcaption>
</figure>

The host rebuilt `swegemma 0.2.11` and vLLM 0.19.1 on October 12 without changing their version numbers. We therefore identify local builds by file hashes, and record the server command, package digests, data revision, and compaction settings, which control how earlier conversation is shortened. The combined record is the run's fingerprint.

A trace records what one agent did and what its tools returned. Comparing agents additionally requires knowing which runtime and task inputs produced each trace.

## 2. The expected score as a failure chain

An attempt has to preserve its edits as a patch and that patch has to pass verification. Let $S$ be the raw solve fraction on a fixed set of $N$ tasks. For task $i$, $E_i$ is the event that the working-tree changes are collected and delivered to verification rather than lost through execution or collection failure. The verifier outcome $Y_i\in\{0,1\}$ is 1 for resolved and 0 for unresolved. If a lost-patch path gives no resolution, the expected solve fraction is

$$
\mathbb{E}[S]=\frac{1}{N}\sum_{i=1}^{N}\Pr(E_i)\,\Pr(Y_i=1\mid E_i).
$$

The first probability measures **patch delivery**; the conditional probability measures **repair success given delivery**. A failed edit or missing diff calls for investigation of tools and collection. A delivered patch that fails verification calls for inspection of the changed behavior and the grading tests.

| Recorded symptom | Evidence to inspect | How it can affect the result |
|---|---|---|
| Agent error or interruption | Actual file changes and the collected diff | A useful patch may survive, so the error alone does not establish delivery failure |
| Delivered patch is unresolved | Applied patch and failing verification tests | The repair may miss a requirement, or the verification environment may be wrong |
| Entire run exceeds twelve hours | Preparation plus agent time over all tasks | The submission errors regardless of how many individual tasks were solved |
{: .quick-summary }

An agent error records an execution problem, not necessarily the loss of all work. In the early inspected runtime, an exception could bypass patch capture. The v31 code inspected by October 23 records general run errors and still attempts fallback diff collection. If it recovers a patch that passes verification, the attempt can be resolved despite the error. Timeout and context-error cases therefore require separate inspection of the collected patch and verifier outcome.

The first run on Kaggle's four L4 GPUs had three public tasks. One was resolved. One ended in a context
error after 55 identical malformed calls. In the third, a delegated analysis agent repeated one broken shell
pipeline for about four minutes until the notebook's overall time limit interrupted it, so no edit was
made. Neither reached a useful source edit, so these cases first call for investigation of tool execution and control flow. The next two sections examine how a malformed call can become a long, expensive failure before repair quality is even tested.

## 3. How a tool call becomes the wrong request

A model's request to read a file has to become a call to the actual file-reading tool. The serving **parser** reads the model output and constructs a tool name and arguments; the harness then invokes that tool. Gemma 4 emits a compact format rather than JSON, using the special token written as `<|"|>` to open and close each string:

```text
intended: read_file{filepath:<|"|>fastapi/openapi/utils.py<|"|>,start_line:400,end_line:568}
emitted:  read_file{filepath:<|"|>fastapi/openapi/utils.py\",start_line\":400<|"|>,end_line:568}
```

The second line is reconstructed from recorded arguments because the original output tokens were not retained. Once a string starts, the parser reads until the next delimiter token. An ordinary quote, escaped quote, or backtick at the end of the path does not close it. The parser therefore includes the following `",start_line":400` in the path. The tool correctly reports that this constructed filename does not exist. Searching for another file in response would leave the delimiter error unfixed.

In the 22-task comparison without reasoning, the bundle produced 223 such malformed calls; another 80 calls were rejected for missing arguments. An `edit_file` call can fail similarly when a backtick closes `new_string`: the parser absorbs `old_string` into the previous string. Even if the model has identified the right edit, the tool receives no usable argument for it.

{% include interactive/article-lab.html id="gemma-parser" %}

## 4. Longer streaks have higher continuation rates

A malformed call is recoverable if the model reads the error and corrects its request. The expensive failure was sending the same call again after the same error. Three counting units distinguish an isolated repeat from a long loop.

| Unit | Definition in this analysis |
|---|---|
| Identical call | Same tool name and parsed argument values; argument-key order is normalized |
| Streak length $k$ | Number of identical calls in a row; a changed argument starts a new streak |
| Adjacent call pair | Two neighboring calls; the streak up to the first determines whether the second is a continuation |
{: .quick-summary }

Reading the same file with a different line range counts as a different call. At $k=1$, a call has occurred once; another identical call enters a streak of two. At $k=3$, it has already occurred three times in a row. Define the probability that the next call is also identical as

$$
h(k) = \Pr(\text{next call identical} \mid \text{current streak } k).
$$

Thus $h(1)$ is the chance of sending a call a second time, while $h(3)$ is the chance of sending it a fourth time after three identical calls. A memoryless model assigns the same repeat probability $q$ at every position, so $h(k)=q$ for all $k$. The traces—time-ordered records of model requests, tool results, and edits—from 22 tasks without reasoning contain 1,404 calls and show a different pattern:

<figure class="gemma-functional-figure">
<div class="gemma-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part2-fig3-repeat-hazard.png" alt="Probability that the next call repeats, as a function of the current streak length" width="2045" height="1118" loading="lazy">
</div>
<div class="gemma-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part2-fig3-repeat-hazard-mobile.png" alt="Probability that the next call repeats, as a function of the current streak length" width="780" height="1981" loading="lazy">
</div>
<figcaption markdown="1">

Figure 3. Empirical $h(k)$ for the same 22 tasks without reasoning (blue) and with reasoning capped at 512
tokens per call (orange; buckets with fewer than five observations omitted). The dashed line is the constant
rate that matches the blue average. Counts are numbers of consecutive call pairs.

</figcaption>
</figure>

At a streak length of one, $h(1)=0.056$. For streaks already three calls long, $h(k)$ is between 0.96 and 1.00; pooling the adjacent pairs at $k\ge3$ gives $643/663=0.97$. Starting to repeat was relatively uncommon, but continuing an already long streak was the usual outcome.

Pooling all adjacent call pairs gives an average repeat rate of $\hat q=0.512$. Suppose every position repeated with that probability. For the recorded lengths of the 22 runs, calculating the chance of a streak of at least twenty calls and summing over runs gives an expected $1.5\times10^{-3}$ runs with such a streak. **Eleven** occurred. Matching the average rate misses the long loops. A descriptive model therefore distinguishes starting a repeat from continuing one:

- an **entry probability** $\alpha = h(1)$;
- a **persistence probability** $\beta \approx h(k)$ for $k \ge 3$.

If a streak continues at the same probability $\beta$ at each further call, its expected number of further repeats is $\beta/(1-\beta)$—about 32 for $\beta=0.97$. This expresses how many calls persistence can consume. It does not identify a causal two-state mechanism. Tasks can have different repeat propensities, long-streak positions select runs that have already persisted, and time or call limits hide what would have happened next. In-context copying is one possible mechanism: the inspected loops retained earlier identical calls and unchanged results in history, with no reasoning text between them.

**Context admission.** Each completed call and tool response is included in a later model prompt, even when it repeats an earlier error. This growing input is the model's context. The server admits another request only when the prompt and requested maximum output fit within its limit:

$$
C_{\mathrm{prompt}} + M_{\mathrm{requested}} \le 32{,}768,
$$

$C_{\mathrm{prompt}}$ counts the tokens in the next prompt. $M_{\mathrm{requested}}$ is the configured maximum output allowance, not the length eventually generated; the server checks that allowance when the request arrives. Assume the first request fits this limit, and let $C_0$ be its prompt length. If no compaction shortens the history and each completed call adds $c>0$ tokens, the total admitted calls, including the first, are

$$
n_{\max} = 1 + \left\lfloor \frac{32{,}768 - M_{\mathrm{requested}} - C_0}{c} \right\rfloor.
$$

In the first diagnostic, $M_{\mathrm{requested}} = 8{,}192$, the loop began at $C_0 = 18{,}033$, and each
repeat added $c = 119$ tokens. The floor alone is 54: after the initial call, 54 more fit. Thus $n_{\max} = 55$. The run made 55 identical calls;
the 56th request, at 24,578 prompt tokens, was rejected. In a later run with $M_{\mathrm{requested}} = 4{,}096$,
the rejected request had 28,673 prompt tokens, one more than the 28,672 the limit allows. With $\beta = 0.97$,
a hypothetical constant-continuation streak survives 55 further transitions with probability $0.97^{55} \approx 0.19$. This illustrates the constant-continuation model rather than estimating a task-level failure probability. A finite loop can still exhaust the context available for further investigation.

{% include interactive/article-lab.html id="gemma-context" %}

## 5. Reasoning trades calls for reliability

A small thinking allowance gives the model space to reason before producing its next response or tool call. In the late-September runtime, E0005 had thinking disabled and produced no reasoning tokens. Its paired variant, E0050, enabled thinking with a 512-token cap per model request. Prompts, tools, limits, runtime, and sampling otherwise matched on the same 22 public tasks:

<figure class="gemma-functional-figure">
<div class="gemma-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part2-fig4-reasoning.png" alt="Resolved tasks, long identical-call streaks, malformed calls and rejected calls for the two bundles" width="2021" height="1003" loading="lazy">
</div>
<div class="gemma-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part2-fig4-reasoning-mobile.png" alt="Resolved tasks, long identical-call streaks, malformed calls and rejected calls for the two bundles" width="780" height="2074" loading="lazy">
</div>
<figcaption markdown="1">

Figure 4. The same 22 public tasks, the same runtime and sampling settings. The two bundles differ only in
their reasoning setting.

</figcaption>
</figure>

Resolved tasks increased from 5 to 7. Only two tasks had different binary outcomes, and both favored reasoning. Under equal chances of either bundle winning such a task, the exact two-sided test gives $p=0.5$: these two wins alone provide weak evidence of a quality improvement. The call-level behavior and patch capture differed more visibly:

<p class="table-caption"><span class="table-label">Table 2.</span> Repetition and patch capture in the same 22-task comparison.</p>

| Quantity | Thinking off | Reasoning 512 |
|---|---:|---:|
| Entry probability $\alpha = h(1)$ | 0.056 | 0.018 |
| Persistence $\beta$, pooled over $k \ge 2$ | 0.95 | 0.47 |
| Expected further repeats after two identical calls, $\beta/(1-\beta)$ | 21 | 0.9 |
| Runs with a streak of 20 or more | 11 | 0 |
| Captured patches | 9 | 15 |

Persistence is pooled over $k \ge 2$ because the reasoning bundle has only 17 pairs beyond $k = 1$:
$\beta = 8/17$ with reasoning and $669/701$ without. In this comparison the entry rate was roughly one third as large and the pooled continuation rate was much lower. The expectation row applies the constant-continuation model from Section 4 after entry into repetition. These are repeated call-pair observations, not independent task samples.

Thinking also consumes the session budget. On the local hardware, a model request took 7.2 seconds with reasoning and 3.3 without. A model request asks for the next generated response; a tool call is an operation requested in that response, such as reading a file. They need not occur one for one. If $T_{\mathrm{session}}$ is the session duration and $t_{\mathrm{request}}$ the time per model request, the approximate request count is

$$
K_{\mathrm{calls}} \approx \frac{T_{\mathrm{session}}}{t_{\mathrm{request}}},
$$

so the request latency alone would roughly halve $K_{\mathrm{calls}}$; tool time also consumes the session. The exact trace counts are 1,382 consecutive pairs with 707 repeats without thinking, and 575 pairs with 18 repeats with it. Counting the first call of each run, removing consecutive duplicates leaves **697 versus 579 calls**. Total calls fell by 57%, while this less-redundant-call count fell by about 17%. A non-repeated call is not necessarily useful; it is only a better proxy than raw call count.

More thinking was not uniformly better in the runs examined. A variant whose reasoning was effectively uncapped, with a
five-minute session, timed out on 15 of 22 tasks locally and scored 0.05 on the host. The capped version
scored 0.12 twice, against 0.06 for the unchanged bundle in the same epoch. This is one historical case where the observed local and hosted ordering agreed. It does not establish a transferable optimum for later runtime versions.

## 6. The local instrument must be validated before it measures

A local quality comparison first needs the intended files to reach the agent and the edited files to reach the verifier. Otherwise a solve-rate difference may describe the benchmark route rather than the agent. Several early observations revealed such failures:

<p class="table-caption"><span class="table-label">Table 3.</span> Apparent agent results traced to benchmark or verifier behavior.</p>

| What looked like an agent result | What it actually was |
|---|---|
| Graph tools never helped in the early local batches. | Graph files were deployed under task names, while the official resolver reads files named by repository and commit. The tools had no data in those runs. |
| Some runs hung for 190–600 seconds. | A vLLM admission rule on the smaller rented GPU never admitted long prompts without a cached prefix. A server flag fixed it; the host made the same flag its default shortly afterwards. |
| An empty patch "resolved" every task in one repository. | The tests imported the installed package instead of the edited workspace copy. Running the official sandbox setup fixed it. |
| An empty task list reported a successful run. | A 0-of-0 run counted as complete. |
| A resolved flag was treated as proof that each exact required test had run. | Reproduction against the current verifier found prefix/name matching can accept a required node without that exact test. The flag establishes this verifier's result, not an independent oracle. |

For each task, the unchanged repository should fail verification and the supplied reference patch should pass. These known failure and success controls detect important setup errors. A required-name matching defect can survive both: a targeted negative control must check whether a report containing only a similarly named test is incorrectly accepted. Before a GPU block, a scripted fake model also drives tools and verification with predetermined responses, producing one resolved and one unresolved task.

A verification replay fixes the captured patch and runs its tests again, so the comparison cannot be changed by a new agent decision. The original subprocess route executes commands as host child processes; the second route verifies inside Docker. Under one v28 local condition, all **137 verified replays** agreed with the original subprocess grades; 28 executed replays remained unverified. In the later v29 follow-up, 34 of 35 planned changed-patch cases ran and 28 were verified. Six attempts across two issues then disagreed with their original grades. Protected test/config filtering was consistent with the change. These selected replays do not regrade the full population; the earlier agreement remains tied to its v28 condition.

<p class="table-caption"><span class="table-label">Table 4.</span> Local evaluation setup and controls.</p>

| Part | Present form |
|---|---|
| Hardware | One rented RTX 6000 Ada (48 GB), tensor parallelism 1, serving Gemma in an environment built from a mirrored official wheelhouse. Kaggle's four L4 GPUs share its compute capability but were too limited in weekly quota for repeated runs. |
| Gates | The scripted-model route check and the per-task controls above, required before every block. |
| Identity | Each result carries the fingerprint described in Section 1. |
| Readings | Each checkpoint is recorded with observations and provisional readings that can be revised. |
{: .gemma-eval-setup }

Required-test matching and shell-path rewriting have different implementation scopes. The latter occurs when the local subprocess backend replaces workspace paths even inside shell string literals, potentially changing text written to a file. It has not been established for the hidden Docker sandbox. Recording runtime and backend with each reproduction identifies which results need rechecking.

The local sandbox runs the agent and evaluator as the same user. Reference files may therefore remain readable through shell commands even if a file-reading tool rejects them. The audited traces show no such read, but an event log records tool requests and results, not every file opened indirectly inside a Python command. Recording which data was used does not itself block access.

Repeated development on the same tasks creates a separate issue: prompts can become tuned to those tasks without updating model weights. We keep 25 public tasks outside recorded model runs and track how the others were used for design, analysis, or training. This holdout supports evaluation on tasks not used in those development decisions; it does not certify absence from model pretraining or exclude every indirect file read.

## 7. What a hosted score tells us—and what it does not

The host returns an aggregate score rather than the private per-task results. A score difference can therefore concern three different questions.

| Question | What is compared | What interpretation requires |
|---|---|---|
| How many solves does a displayed score represent? | Solve count versus displayed decimal | Task count and truncation rule |
| Why does the same package score differently on another run? | Repeated executions on the same task set | Separating execution variation from unobserved serving conditions |
| Will Public performance persist on Private? | Solve fractions on different task sets | Assumptions about task selection and dependence |
{: .quick-summary }

For the display question, the observed leaderboard values fit a working grid with 58 Public tasks. If $k$ of those tasks are resolved, truncation to two decimals gives:

$$
\mathrm{displayed}(k) = \frac{\lfloor 100k/58 \rfloor}{100}, \qquad k = 0, 1, \ldots, 14.
$$

This maps 6 solves to 0.10, 9 to 0.15, 11 to 0.18, and 14 to 0.24. The host says about 60 Public tasks. The live competition metadata confirms two-decimal truncation, but does not disclose the exact Public denominator. The 58-task grid is consistent with the snapshot, not a directly observed task roster. Task equivalents below are conditional on that interpretation.

**Recorded variation on nominally fixed tasks.** The same budget-section variant, E0053 (described in Section 8), scored 0.10 and 0.15 from an identical ZIP. Both observations were assigned to HE-11, the eleventh project-defined host epoch. The files and observed public host fingerprints match. This gives a concrete same-package score difference. Without private per-task outcomes, serving logs, or deployment receipts, its cause remains unresolved; we cannot attribute it to random execution variation alone.

Let $\pi_i$ be the probability that the same package solves task $i$ on a repeated execution. Summing the binary outcomes gives the solve count $K_{\mathrm{solves}}=\sum_iY_i$. Its variance, describing how much that count changes across executions, is

$$
\operatorname{Var}(K_{\mathrm{solves}})=\sum_i\pi_i(1-\pi_i)+2\sum_{i<j}\operatorname{Cov}(Y_i,Y_j).
$$

The first term adds each task's own success/failure variation. The covariance term adds effects that make tasks succeed or fail together: a slow server or time already spent on earlier tasks can affect several outcomes. Independent task outcomes would make this term zero, but that independence has not been established here. Two aggregate observations cannot identify the task probabilities, covariance, or a stable standard deviation, and therefore cannot calibrate a two-task noise floor.

**How variation could obscure an improvement.** We can still ask a conditional question: if the variation had a particular size, how reliable would one comparison be? Suppose independent draws of two bundles had equal, known standard deviation $\sigma$, and approximately normal errors. If their expected solve counts differed by $\Delta$, the chance of ordering them correctly would be

$$
\Pr(\text{correct order})\approx\Phi\!\left(\frac{\Delta}{\sigma\sqrt{2}}\right).
$$

<p class="table-caption"><span class="table-label">Table 5.</span> Probability of ordering two bundles correctly under assumed variability.</p>

| Assumed true difference $\Delta$ | Assumed $\sigma = 1$ | Assumed $\sigma = 1.5$ | Assumed $\sigma = 2$ |
|---|---:|---:|---:|
| 1 task | 0.76 | 0.68 | 0.64 |
| 2 tasks | 0.92 | 0.83 | 0.76 |
| 3 tasks | 0.98 | 0.92 | 0.86 |

Here $\Phi$ is the standard normal cumulative distribution function. In this table, $\sigma$ is the standard deviation of solved-task counts, assumed for the calculation rather than estimated from our submissions. Larger variability makes the same expected improvement harder to distinguish in one comparison. Conversely, a comparison without a clear difference does not establish equal expected performance.

**Public versus Private.** Repeated execution on a fixed set and evaluation on a new set are different sources of uncertainty. The Private split contains different tasks. Under a separate hypothetical independent-sampling model, solve rate $p$, Public size 58 and Private size 60 would give

$$
\operatorname{SD}(\hat p_{\mathrm{Public}}-\hat p_{\mathrm{Private}})
\approx\sqrt{p(1-p)\left(\frac{1}{58}+\frac{1}{60}\right)}\approx0.072
\quad (p=0.19).
$$

That calculation concerns differences between task samples, not repeated submissions on one fixed set. The actual split design and task dependence are unknown, so 0.072 is not a forecast or confidence interval for this competition.

## 8. What a local paired comparison can establish

A local paired comparison gives the reference and candidate the same task, records the seed, and alternates execution order. Pairs where both solve or both fail add zero to the win–loss difference. The sign test counts only **discordant pairs**, where one bundle solves and the other fails. Under equal chances of either winning, each discordant outcome acts like a fair coin flip. If these pairs are **independent**, $d$ discordant outcomes with $g$ candidate wins give the two-sided probability of an equally or more imbalanced result:

$$
p=\min\!\left(1,\;2\sum_{j=\max(g,\,d-g)}^{d}\binom{d}{j}2^{-d}\right).
$$

Two wins and no losses give $p=0.5$; six wins and one loss give $p=0.125$. This is the probability of such a win–loss imbalance under equal chances, not the probability that the candidate improved. Repeating the same issue under another seed does not create an independent task, so these attempt counts cannot automatically supply a task-level test. We report resolved attempts below and examine task-level differences separately.

The working reference allows 28 tool calls, eight minutes per session, and a 4,096-token thinking budget, with a short budget section in its prompt. Its identical files received 0.15, 0.15, 0.15, 0.17, and 0.18 across several observed epochs, including ambiguous windows. **Only one of those is a clean HE-11 reference draw.** They are not five same-environment repeats.

<p class="table-caption"><span class="table-label">Table 6.</span> Changes evaluated against the Budget-Fit reference. Counts are resolved attempts; each row retains its own comparison conditions.</p>

| Change to the reference | Local resolved attempts: reference → candidate | Hosted observation |
|---|---|---|
| Remove the 701-byte budget section | 17/44 → 22/44; 22 tasks, two seeds; 1 baseline-only and 6 candidate-only attempts | 0.10 and 0.15, same candidate ZIP, HE-11 |
| Remove earlier thoughts from history | 17/44 → 17/44; 22 tasks, two seeds | Kernel run timeout; no score |
| Allow 36 tool calls instead of 28 | 17/44 → 14/44; 22 tasks, two seeds | Not submitted |
| Greedy decoding (temperature 0) | 14/44 → 16/44; fresh reference run, 22 tasks, two seeds | Not submitted |
| Use a longer public prompt whose author page displayed a best score of 0.22 | 17/40 → 16/40; 40 external tasks, one seed | Our exact package: 0.17; the author's selected best is not a matched control |
| Use 12B with the same settings | 17/44 → 11/44; 22 tasks, two seeds | Not submitted |

The rows are separate comparisons, not arms of one experiment with a shared reference run. Some also differ in runtime. The 40-task result remains provisional rather than a completed confirmatory comparison. Some stopping or advancement rules were fixed before execution; the budget-section heuristic was proposed after three blocks were visible. A rule designed after seeing outcomes can describe an investment decision, but cannot serve as pre-registered evidence.

By the October 17 review, seven candidate designs had used roughly 51 measured pod-hours without selecting a replacement. This included initial screens, small canaries that checked whether a change behaved as intended, and paired comparisons. The total describes the investment through that review, not a common seven-arm test or the later cumulative cost.

**Does a local gain transfer?** The reference's retained public-panel runs resolve about 36% of attempts. Its original two-seed external comparison resolved 31/80 attempts on 40 tasks from six Pallets repositories, or 38.75%. The clean HE-11 hosted reference observation is 0.18; under the inferred 58-task grid that is 11/58, about 19%.

The populations, development exposure, hardware, and protocols differ. The host used a strong model as a solvability check for private-repository tasks; it did not claim to select tasks that were hard for that model. We have not separated the effects of these differences, so the ratio of local and hosted solve rates cannot convert the effect of a new prompt into expected hosted gains.

The budget-section variant illustrates the limit. It resolved 22/44 in one comparison, then 13/44 in a later local run. **Both runtime and seeds changed**, and the latter had no matching E0052 control. The difference therefore cannot identify the runtime's causal effect. Its two hosted scores, 0.10 and 0.15, likewise do not establish dominance or equivalence. The defensible conclusion is that this local gain did not establish a reliable hosted gain, and the unchanged reference remains our working choice.

## 9. The twelve-hour limit couples quality to completion

Section 2 separated successful patch delivery from successful repair within a task. A hosted submission adds an outer condition: the whole generation run must finish within twelve hours to receive a score. That couples every task's time cost to the usefulness of the final submission.

Total generation time includes shared startup, task preparation, and agent work. Let $N_{\mathrm{all}}$ count tasks across Public and Private, so $T=h+\sum_{i=1}^{N_{\mathrm{all}}}(s_i+a_i)$. Here $h$ is shared overhead, $s_i$ is task preparation and other counted overhead, and $a_i$ is that task's agent time. Exceeding twelve hours errors the submission instead of scoring only the completed tasks.

A repair gain has no scored value if the complete run cannot finish. Distinguish the total time, conditional repair fraction, and planning utility:

| Quantity | Scope | Meaning |
|---|---|---|
| Total time $T$ | Both Public and Private | Shared startup plus every task's preparation and agent work |
| Raw Public fraction $S_{\mathrm{Public}}$ | Public tasks | Solved tasks divided by task count, before display truncation |
| Planning utility $U$ | Completion of the whole run and Public quality | Raw Public fraction on completion; zero on a timeout |
{: .quick-summary }

Zero on timeout defines our planning utility; Kaggle records an error rather than assigning that score. Multiplying the chance of completing the entire run by its expected Public solve fraction conditional on completion gives

$$
\mathbb{E}[U]=\Pr(T\le H)\,\mathbb{E}[S_{\mathrm{Public}}\mid T\le H],
\qquad H=720\text{ minutes}.
$$

Assume each task's total time $s_i+a_i$ is independent, with common mean $\mu$ and standard deviation $\sigma$, both measured in time units. A normal approximation to the sum then gives the chance that the complete run fits the limit $H$:

$$
\Pr(T\le H)\approx\Phi\!\left(\frac{H-h-N_{\mathrm{all}}\mu}{\sigma\sqrt{N_{\mathrm{all}}}}\right).
$$

<figure class="gemma-functional-figure">
<div class="gemma-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part2-fig5-time-cliff.png" alt="Probability of finishing 120 tasks within 12 hours as per-task time is scaled" width="2051" height="1063" loading="lazy">
</div>
<div class="gemma-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part2-fig5-time-cliff-mobile.png" alt="Probability of finishing 120 tasks within 12 hours as per-task time is scaled" width="780" height="1372" loading="lazy">
</div>
<figcaption markdown="1">

Figure 5. A hypothetical timing model, not hosted measurements: 120 independent task times, reference SD 2.5 minutes, and 20 minutes of shared startup. Multiplying task time scales both its mean and SD. The three curves use reference means of 4.5, 5.0, and 5.5 minutes.

</figcaption>
</figure>

For the 5.0-minute reference, multiplying all task times by 1.18 reduces the illustrative completion probability from essentially one to about 0.40. Suppose the conditional Public solve fraction also improves from 11/58 to 12/58. Using **raw fractions throughout**, the utility is about $0.40\times(12/58)\approx0.083$, versus $11/58\approx0.190$ for a reference that almost always finishes. To preserve the reference's expected utility, the improved variant must finish with probability at least $11/12\approx0.917$. Completion has to remain very likely even when conditional repair quality improves.

If task times are correlated, the sum has covariance terms and these curves change. We do not know hosted per-task times or shared overhead, so the figure supplies no safe slowdown threshold.

There is a relevant warning in the records. Removing earlier thoughts tied the reference locally at 17/44 but increased measured **agent-session** time by about 18%; its hosted run later timed out. This is consistent with a time risk, not proof of its cause. The ratio was measured for agent sessions on one Ada GPU with tensor parallelism 1 (TP1). Hosted execution uses four GPUs (TP4), and its total also includes preparation overhead, so the local ratio cannot be applied to the whole hosted run. In particular, ratios below 1.05 or 1.10 have not been calibrated as safe.

The reference's eight-minute cap also does not guarantee completion: $120\times8=960$ minutes before setup. Its successful submissions show that the cap need not be reached on every task. They do not disclose how much headroom remained.

{% include interactive/article-lab.html id="gemma-timing" %}

## 10. What the agent's own checks do not establish

Finishing a run and delivering a patch still leaves the repair question. Reasoning and the newer tool-error handling reduced some early failure modes, yet the remaining errors had several possible explanations. In 140 recorded reference attempts on 40 exposed external tasks, five had an error streak of at least five calls; all five were unresolved. This diagnostic pool combines three groups collected under different recording conditions, including attempts outside accepted final comparisons. It helps locate questions for investigation, with a different evidential role from a fresh evaluation sample.

Let $C$ be the behavior checked by the agent and $R$ the behavior required by the evaluator's tests. Checking that a function terminates does not establish that its return value is correct. A useful local check need not imply every hidden assertion, but it should test a relevant requirement on the actual final code.

**52 of 69 verification-failed attempts** edited at least one file also edited by the reference fix, across 19 tasks. File overlap narrows the search for failed repairs after an edit; it does not show that the agent found the cause. It could edit the wrong part of that file, while another valid fix might use different files. In 33 attempts, a later Python- or pytest-like command had a response categorized as non-error. Without the assertion, actual exit status, and evidence that it checked the final patch, that category cannot establish “the agent's test passed but the hidden tests failed.”

A Rich example shows what a direct trace inspection can add. For an issue involving zero-width characters, one agent check established termination while the evaluator checked exact span values for a lone variation selector. The two checks concerned different properties of the output. Existing repository tests provided related format information, but the searched trace did not show those lines being read. That identifies a potential source of evidence the agent missed; whether reading it would have led to a correct repair remains open.

The available specification may also limit what the agent can check. Across four runs on 22 public tasks, statements of at most 120 characters had 7 resolves in 32 attempts; longer statements had 24 in 56. Comparing solve-to-failure odds—$7/25$ and $24/32$, rather than the solve rates—gives the descriptive odds ratio

$$
\mathrm{OR}=\frac{7/25}{24/32}\approx0.37.
$$

Character count is only a proxy for the information available to the agent. There are eight distinct tasks with short statements and fourteen with longer ones; the remaining observations repeat those tasks. They are not 88 independent tasks, and independence among the 22 tasks has not been established. Repository and change type also differ. The ratio describes the association; no causal effect or $p$-value is claimed.

We tried prompting checks that more closely matched the issue. A pinned pre-edit check—write a failing check, edit, and rerun the same check—was fully carried out in only 1 of 8 target opportunities. An issue-context helper was skipped in 7 of 16 attempts. In selected earlier traces, short command examples were copied more reliably than long procedures. Following a command example, however, is a different outcome from selecting a useful check and improving repair quality.

A new instruction needs an observed failure it could change. For example, a model that received a failing assertion and proceeded as if it passed could be instructed to recheck that signal. We searched eight selected cases for a strong contradiction of that kind using information visible to the model, but found none supporting one minimal correction. In six cases, details required by the evaluator were absent from the statement, though some might still be inferred from repository context. Two weaker ideas remained. We therefore parked the proposed issue-scenario check: the 52-case file-overlap tally identified where to investigate, not 52 failures that one prompt could repair.

## 11. Adapters and smaller models: readiness is not a quality result

LoRA offers another way to change behavior: train additional weights, an adapter, and attach them to the base model. Learning the weights, applying them during serving, completing the run, and improving repair quality each require evidence. The host reported a LoRA fix on September 27 and later said adapters up to rank 64 had scored. This supports a serving path in that environment, but does not measure our package's capacity or benefit. Participants have reported serving overhead and reduced capacity in the KV cache, the memory holding attention information from earlier tokens. These reports are leads for measurement, not measurements of our package.

A trajectory records the model requests, responses, and tool executions from the start to the end of one task. On October 22 we built the training dataset **S1** from **65 resolved reference trajectories on 37 public tasks**. Each trajectory contains several model requests. Pairing the actual prompt and response of each request produced **1,239 per-call samples**.

The samples contain **259,576 target tokens and 10,938,513 forward tokens**. Target tokens are the outputs selected for the model to learn to generate; forward tokens also count the input processed to predict those outputs. For example, if the prompt contains the issue, code, and earlier tool results, and the response is an `edit_file` call, both prompt and response count toward forward tokens, while training loss would be computed only on the response tokens. That is output-only supervision. The resulting workload is much larger than the target count alone suggests.

A later model request receives a prompt reconstructed from the earlier conversation, not necessarily a verbatim concatenation of the recorded outputs. Concatenating raw outputs for training could therefore produce context the model never received. S1 preserves the served prompt and output token IDs per call and applies supervision only to output tokens. Decoding those IDs with the tokenizer to check their format, and actually training LoRA on the 31B base, remained outstanding.

Tasks used for training cannot subsequently supply an independent repair-quality evaluation. We designated a subset of the tasks already used in development for training, excluding them from independent performance evaluation. The 40-task external panel **X40** and untouched 25-task holdout were excluded from training. The six-task split inside S1 is for loss-based development, separate from the untouched holdout used to assess repair quality.

A hosted completion check added an all-zero rank-16 adapter to the reference. It errored roughly 25 minutes after submission, **including queue time**. The cause is unknown. A newly checked [staff reply][d751339] acknowledges a concurrent platform incident and promises reruns for submissions affected by it, but no log ties our failure to that incident. Completion, adapter activation, training throughput, and quality are four different questions. No learned-adapter quality comparison has run, and the larger training investment remains parked.

For smaller models, 12B resolved 11/44 attempts against 31B's 17/44 under the tested reference settings. That is not a universal model ranking. The USD 10,000 E2B/E4B bonus creates a separate opportunity, but an eligible entry must occupy one of the team's two final selections. On **October 23 at 09:01 KST**, we submitted an E4B version of the reference, crediting the original package. Its status in the reviewed record is **PENDING**. A dated small-model standings snapshot had a best Public score of 0.06; our new entry has no score yet and has not earned a final slot.

## 12. Part 1's three scenarios, revisited

1. **The model can do more than its execution procedure permits.** Early loops, malformed calls, and context failures support this concern. Bounded reasoning and host fixes helped in particular comparisons; their contributions have not been isolated across the whole history.
2. **A usable procedure still leaves recurring repair errors.** Many failed patches touch reference files, but that does not distinguish incorrect localization, missing requirements, capability limits, or insufficient time. The limited trigger search did not yield a ready intervention.
3. **Local gains may not transfer.** The populations and protocols differ, several observed orderings changed, and same-byte hosted scores varied. These are reasons to qualify a local improvement claim, not a calibrated conversion from local solves to hidden solves.

## 13. The next decisions

The unchanged reference remained the working choice. The bounded prompt and navigation reviews had not connected a specific model-visible failure to a small change worth testing; their proposals were parked or held within those scopes. That is a decision about the next investment, not evidence that further gains are impossible. The evidence needed to reopen each option is:

<p class="table-caption"><span class="table-label">Table 7.</span> Current decisions and the evidence needed to revisit them.</p>

| Question | Evidence needed | Current position |
|---|---|---|
| Does E4B warrant further work? | The completion status and score of the pending exact package, read against dated small-model standings | One submitted observation is pending; no superiority or final-slot decision yet |
| Should adapter work resume? | Verified serving behavior, adapter/base-model compatibility, measured throughput and a priced quality comparison | S1 exists; the zero-adapter check errored for an unknown reason; training remains parked |
| Is another repair intervention justified? | A concrete model-visible failure that a small change could address, plus its costs and failure cases | The bounded scenario, stale-check, and navigation reviews did not establish that bridge |
| Can local timing bound submission risk? | Target-topology measurements including setup and the whole task population | Local ratios are warning signals; no universal safe ratio is known |

## Sources

- [Competition overview, evaluation and rules][competition]; [data and harness guide][data];
  [official evaluation wheelhouse][wheelhouse].
- Host clarifications: [sequential scoring and the 12-hour limit][d743063];
  [distillation][d742807]; [hidden-set size][d744258]; [other open models' trajectories][d746046];
  [additional model sizes][d748287]; [scoring and task filtering][d748428];
  [LoRA submissions][d749191]; [the 12-hour rule stands][d749597]; [small-model bonus][d751104]; [concurrent platform incident][d751339].
- [Budget-Fit single-agent bundle][budget-fit] by Wilmer E. Henao, Apache 2.0, used unchanged as the reference.

[competition]: https://www.kaggle.com/competitions/gemma-4-developer-agent
[data]: https://www.kaggle.com/competitions/gemma-4-developer-agent/data
[wheelhouse]: https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse/data
[budget-fit]: https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent
[d743063]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743063
[d742807]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807
[d744258]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744258
[d746046]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746046
[d748287]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287
[d748428]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748428
[d749191]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749191
[d749597]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749597
[d751104]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/751104

[d751339]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/751339
