---
title: "Gemma 4 Developer Agent, Part 1: Learning to Fix a Codebase, One Verified Patch at a Time"
description: "A Gemma 4 developer agent turns an issue and a repository into a verified patch: competition objectives, evaluation, runtime budgets, an inspectable baseline, and controlled comparisons."
date: 2026-09-25 06:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, gemma-4, adk, coding-agents, evaluation, lora, software-engineering, working-note]
math: true
mermaid: true
image:
  path: /assets/img/posts/gemma-4-series/cover-1.webp
  alt: "Part 1 cover: from issue to verified patch"
  width: 2200
  height: 1240
  fit: contain
  hide_caption: true
pin: false
published: true
topic: gemma-4
lang: en
interactive_labs: [gemma]
translation_key: gemma-4-developer-agent
series: gemma-4
series_order: 1
---

<style>
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


<style>
article .content table:not(.rouge-table) th,
article .content table:not(.rouge-table) td {
  white-space: normal;
  overflow-wrap: anywhere;
}
article .content .mermaid,
article .content .mermaid * {
  font-family: Arial, sans-serif !important;
}
@media (max-width: 600px) {
  article .content .table-wrapper { overflow-x: auto; }
  article .content table:not(.rouge-table) { min-width: 600px; }
  article .content table:not(.rouge-table) th,
  article .content table:not(.rouge-table) td { min-width: 110px; }
}
</style>

[Read in Korean]({{ site.baseurl }}/posts/Gemma-4-Developer-Agent-From-Issue-to-Verified-Patch-KR/)

{% assign gemma_part2 = site.posts | where: "translation_key", "gemma-4-developer-agent-02" | where: "lang", "en" | first %}
> **Update — October 23, 2026.** This guide preserves the September 25 specification and its unscored teaching example. The allowed model sizes, thinking settings, protected-file handling, and error recovery have since changed; the code below is version-specific. Our later working reference is a different public package, not a scored version of this example.{% if gemma_part2 %} [Part 2]({{ site.baseurl }}{{ gemma_part2.url }}) follows those changes and the subsequent experiments.{% endif %}

## The task: turn a new issue into a verified patch

In a familiar Kaggle prediction competition, we build a model that turns an input into a
prediction: a house's attributes into a price, for example. Here, the input is a report
that a program behaves incorrectly, together with that program's source code. The system
must investigate the report and change the code. The location and the repair are both
part of the problem.

Under the September 25 rules of the [Google — Gemma 4 Developer Agent Competition][competition],
every model-backed agent uses the same permitted Gemma base model. We build the procedure around
it: how to find relevant code, make an edit, check the result, and decide what to do next. Kaggle
runs that procedure on hidden issues and evaluates the resulting patches. **We submit a way of
solving new issues, not a set of fixes for the public practice issues.**

**A few useful places to start.** These sources have different jobs; you do not need to
read them all before continuing.

- **What the competition asks:** [Kaggle overview and evaluation][competition], followed by the [data and harness guide][data] and [rules][rules]. These define the actual task and permitted submission.
- **Why an open model is relevant:** Google's [Gemma 4 announcement][gemma-launch]. Read it for the wider motivation; the competition selects one particular model and runtime.
- **Where repository-level evaluation comes from:** the [SWE-bench paper][swebench-paper] and [GitHub repository][swebench-code]. They explain the move from writing a function to repairing an existing project.
- **What a small coding agent looks like:** [mini-SWE-agent's documentation][mini-docs] and [GitHub][mini-code]. Follow one model–tool interaction before studying a large framework.
- **What we will actually configure:** the competition's [official evaluation packages][wheelhouse] and [ADK Agent Config documentation][adk]. The competition's restricted format takes precedence over general ADK examples.

### From predictions to repairs

| Question | A typical prediction competition | This developer-agent competition |
|---|---|---|
| What arrives at evaluation time? | Unseen rows, images, or other examples. | An issue description and a repository snapshot. |
| What must the system produce? | A prediction for each example. | Code changes that address each issue. |
| What do we submit here? | Often a prediction file or inference notebook. | An agent package, `submission.zip`, that generates patches when run. |
| What determines success? | A metric comparing predictions with targets. | The fraction of tasks whose patches pass the evaluator's checks. |

A model can suggest a repair when given the faulty function and its symptoms. A developer agent
must first locate that function, then check whether the repair addresses the issue without
breaking existing behavior. Consider a list API with a `limit` parameter: it sets the maximum
number of items in the response. Suppose this API promises a default of ten when `limit` is
omitted, and an empty result when `limit=0`. If an explicit zero instead returns ten items, the
agent has to find where those two cases were conflated.

This is why a coding agent makes repeated model calls. A search reveals a filename; reading
it reveals a function; a test result can invalidate the first explanation. The useful next
action depends on what the previous action discovered. The model is choosing actions in an
ongoing investigation, rather than receiving every relevant fact in a single prompt.

### The three objects to keep separate

A **repository** contains a project's source files, tests, configuration, documentation, and
version history. An **issue** describes behavior to fix or add. A **patch** records the file
changes that implement the repair; a Git **diff** is their textual representation.

We submit an **agent package** containing configuration, instructions, and any optional
learned adapters. Running it produces a **patch for each issue**. A separate evaluator
applies that patch to a fresh copy and produces a **resolved or unresolved outcome**.
Loading the package, obtaining a nonempty diff, and passing verification are three
different achievements. Only the final outcome contributes a resolved task to the score.


### If the model is fixed, what can we improve?

The model supplies decisions, while tools carry out operations such as reading a file or running a
command. The agent keeps that interaction going. The organizer's harness supplies the task
environment, enforces its limits, and checks the final patch. These roles remain distinct even
when the software runs on one machine.

| Term | Its role in this competition |
|---|---|
| **Model** | Gemma generates the next reasoning step, tool call, or response. |
| **Agent** | The model together with instructions, tools, state, and a procedure for continuing the work. |
| **Tool** | An operation such as reading a file, running a command, or submitting the current changes. |
| **Harness** | The organizer's software that loads the agent, prepares tasks, enforces limits, and evaluates patches. |
| **Patch** | A Git diff describing changes relative to the prepared repository baseline. |
{: .quick-summary }

The fixed base model does not fix the agent's behavior. We can change the prompt, the
registration and use of permitted tools, how tasks are delegated, and the chosen limits.
For example, **localization** means narrowing an issue down to the files and symbols that
matter. A better search procedure can put the right function in front of the same model.
A better editing procedure can turn a correct idea into a valid change instead of a failed
tool call. A better test choice can expose a mistaken repair while time remains to revise it.
These changes can improve resolution without training new weights.

The official evaluator remains the organizer's criterion. Improving our own evaluation
means reproducing its relevant behavior and collecting useful evidence, not changing what
Kaggle accepts. Likewise, an agent's checks during debugging are evidence for its next
decision; the later evaluator verdict determines its score. The hidden verification tests
are not an oracle the agent can query during the attempt.

Optional **LoRA adapters** add learned changes to the permitted base model. They offer a second
route to improvement once we identify a recurring behavior that training could address. More
agents, more retrieval, and more training each consume resources; their value must show up in
repairs that finish within the same evaluation constraints.

### Where the first experiment should begin

The first goal is one attempt we can explain from beginning to end. Three questions put
that experiment in order:

1. **Can we trust the local check?** On a chosen development task, inspect the unmodified
   repository and reference-fix controls. An environment failure is not evidence that the
   agent needs a better prompt.
2. **What does a simple agent actually do?** Record its searches, file reads, edits, test
   results, patch, and elapsed time. This sequence is a **trajectory**. A failed edit and
   a wrong diagnosis call for different changes even if both yield zero points.
3. **Does one change fix the observed failure on other issues?** Compare the changed
   agent with the baseline under matching conditions, including issues held out from
   development. Then consider training or delegation if the remaining failures justify it.

## 1. A one-line bug can require a repository-sized investigation

Suppose a list API has at least ten available items. Its `limit` parameter caps the number
returned: `GET /items` uses the default of ten, while `GET /items?limit=0` should return an empty
list. A request with `limit=5` should return five. This is the contract of this invented API;
another API may define zero differently. The bug report is:

> Setting a limit to zero unexpectedly restores the default limit. An omitted limit should
> use the default, but an explicit zero should remain zero.

The request parser represents an omitted parameter as `None` and an explicit zero as the integer
`0`. It then calls a helper to determine the item limit. The helper returns a number, not the list
itself; later query code uses that number to select the response items:

```python
def normalize_limit(value, default=10):
    return value or default
```

Python treats both `None` and `0` as false in this expression, so either input produces the
default, 10. The query code therefore receives a limit of ten even when the caller explicitly
requested zero. Checking specifically for `None` preserves the distinction:

```diff
 def normalize_limit(value, default=10):
-    return value or default
+    return default if value is None else value
```

In this diff, a line marked `-` is removed and a line marked `+` is added. The unchanged
function signature shows where the edit belongs. An executable test supplies an input
and checks the observed result against the required one.

If both snippets were already in the prompt, this would be a small Python question. In
repository work, the agent starts several steps earlier. Suppose the visible project is:

```text
example_service/
├── api/routes.py          # Receives the request
├── settings.py            # Supplies configured defaults
├── query/options.py       # Converts request options
├── query/limits.py        # Normalizes the limit
└── tests/test_options.py  # Exercises the public behavior
```

An agent could search for `limit`, obtain dozens of matches, and edit the first promising
one. It might insert a special case in `routes.py`. That could fix the web endpoint while
leaving the command-line entry point broken, because both eventually call `limits.py`.
Alternatively, it could change the shared helper too broadly and alter how empty strings
are handled, even though the report says nothing about strings.

The useful investigation follows the value. Where does the explicit zero enter? Where is
it converted? At what point does it become ten? Which layer owns the distinction between
an omitted value and a supplied value? The code search is serving a hypothesis about the
program, rather than producing a pile of text for the model to read.

A small behavioral table clarifies what the issue actually promises:

| Parsed input to the helper | Current limit | Required limit |
|---|---:|---:|
| `None` | 10 | 10: the default applies. |
| `0` | 10 | 0: the explicit value must survive. |
| `5` | 5 | 5: ordinary values must keep working. |

<figure class="gemma-functional-figure">
<div class="gemma-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part1-fig1-issue-to-fix.png" alt="An explicit zero becomes ten in the shared helper; an explicit None check repairs the bug and preserves all three input cases" width="2240" height="1650" loading="lazy">
</div>
<div class="gemma-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part1-fig1-issue-to-fix-mobile.png" alt="An explicit zero becomes ten in the shared helper; an explicit None check repairs the bug and preserves all three input cases" width="780" height="3020" loading="lazy">
</div>
<figcaption markdown="1">

Figure 1. This list API must return an empty list for limit=0. The old helper changes zero to the default of ten. Using the default only for None preserves omitted, zero and ordinary inputs as separate cases.

</figcaption>
</figure>

The three rows have different jobs. Zero reproduces the defect. `None` checks that the
repair preserves the fallback. Five checks that an ordinary request has not changed.
If the repository allows other input types, their contract needs separate investigation;
the agent should not invent it from these three examples.

This also explains what makes a test useful. A check of `normalize_limit(5)` passes before
and after the repair. It provides a little regression evidence, but it cannot demonstrate
that the reported bug was fixed. The zero case should fail on the old implementation and
pass on the new one. A **regression test** records the behavior that must remain correct
after future changes.

The agent then needs to inspect the final diff. Did it change the helper it intended to
change? Did an unsuccessful edit leave the file untouched? Did it accidentally include a
temporary script? A final natural-language message saying "fixed" cannot answer those
questions. The patch is what another environment will receive.

A useful trajectory makes those responsibilities visible: what the model requested, what the tool
actually did, and what observation reached the next model call. A correct plan can fail at any of
those interfaces.

A **trajectory** is the recorded sequence of messages, actions, and observations during
that process. It might show a search, two file reads, a failed reproduction, an edit, and a
successful test. When an agent fails, the trajectory lets us ask where its understanding
stopped matching the actual state of the repository.

That is why this competition is an interesting meeting point between machine learning and
software engineering. It rewards a system that repeatedly turns uncertain information
into a working change.

To see how the model uses a file tool, here is a shortened, **invented interaction**, not
a measured trajectory or the exact wire format:

```text
Input to the agent: the zero-limit issue and the current repository context.
Model requests: read_file(filepath="query/options.py", start_line=1, end_line=40).
Harness: reads that file in the sandbox and returns its text.
Tool observation includes: return normalize_limit(raw_limit)
Model requests: read_file(filepath="query/limits.py", start_line=1, end_line=40).
Tool observation includes: return value or default
Next useful action: check that this expression reproduces the zero case.
Agent runs a permitted focused check: normalize_limit(0) returns 10.
Agent requests a precise edit replacing the faulty return expression.
Tool reports success; a subsequent file read confirms the saved replacement.
Agent checks the zero, omitted-value and ordinary-value cases.
Agent requests submit_patch(); the harness extracts the code changes.
Separate verifier: applies that patch to a fresh copy and checks the task.
Illustrative outcome: resolved, if those independent verification checks pass.
```

The model does not directly inspect the disk. It emits a structured request that the
harness interprets. The resulting observation becomes part of the input for its next
decision. A tool error is also an observation; silently ignoring it breaks this feedback loop.

{% include interactive/article-lab.html id="gemma-bug" %}

## 2. Why Google is asking for an open-model developer agent

A language model stores learned numerical parameters, often called **weights**. Its current input
is the **context**: instructions, the issue, and any code or observations supplied so far.
**Inference** uses the weights and context to generate a response or choose an action. Text is
processed as tokens, which can be word fragments, punctuation, or pieces of a code identifier.

Reading a new file changes the context for the next decision. **Training**, in contrast, changes
learned parameters using examples or a learning signal. An agent can therefore investigate an
issue over many tool calls without retraining each time it reads another file.

| Concept | What changes, or what it does |
|---|---|
| Weights | Learned parameters reused across model calls. |
| Context | The information available for the current call; tool results can add to it. |
| Inference | Generates the next response using the weights and current context. |
| Training | Updates learned parameters so future calls can behave differently. |
{: .quick-summary }

This distinction gives us two paths to improvement. A prompt or tool change can help the
same weights use better evidence. Training can teach different behavior across many such
situations. We need to see which limitation is present before deciding between them.
The optional **LoRA adapter** introduced later is a compact set of learned weight changes,
not a replacement repository or a collection of answer patches.

An open-weight model makes those learned parameters available for running and adaptation.
This gives developers a different set of deployment choices from an application that can
only send requests to a hosted service. Google's April 2, 2026 [Gemma 4 announcement][gemma-launch]
presents an Apache 2.0 model family with reasoning and agent-oriented capabilities,
including function calling and structured output. The family includes several sizes;
the September 25 competition specification chooses one specific 31B variation from that broader release.

Here, **31B** describes the approximate parameter scale, not a number of files the model
can understand or steps it can take. **Instruction tuning** prepares a model to respond
to instructions. **Post-training** is further adaptation after the base model's initial
training; it can target a behavior such as producing valid tool calls or following a
useful debugging procedure. Instruction tuning is one form of post-training. These terms
describe aspects of model preparation, and none alone guarantees successful repository repair.

### 2.1 Local execution changes the possibilities, not the definition of success

Why care about running the model near the code? A developer may need control over where
source files are processed, want a reproducible model version, need to operate without a
reliable connection, or want to specialize the system for a particular workflow. Those
are practical motivations for a local agent. They are not automatic guarantees of privacy,
low cost, or speed: the tools and logs still determine where data goes, while hardware
and workload determine runtime.

Google's [AI Edge article][gemma-edge] illustrates the broader ambition with small Gemma
models and on-device actions. Its E2B/E4B examples should not be read as measurements of
this competition's 31B system. Likewise, the [general model card][gemma-card] describes
a 256K context for the 31B model, while Kaggle's serving contract uses 32,768 tokens. The
article about the family tells us why the technology is interesting; the competition
contract tells us what will actually be evaluated.

The organizers' stated purpose is to advance open agents that navigate code and draft
repairs, with human developers retaining roles such as architecture and review. The
competition promotes fine-tuning and reinforcement learning, but **an adapter is optional
in the submission format**. A no-training agent is therefore a useful starting point:
it tells us what the fixed model can already do before we attribute a later gain to
learning. [Competition overview and model rule][competition]

Holding the base model fixed also makes a research question unusually clear. We can change
what information the model sees, which procedure it follows, and what allowed adaptation
it uses. We can then ask whether those changes recover issues within a shared deployment
budget. Training resources can still differ across teams, so the contest is not a guarantee
of equal experimental resources. The restriction makes the deployed system more specific.

### 2.2 A patch score is one useful measurement of a larger ambition

It is easy to slide from "this agent passed more tests" to "this agent makes every developer
faster." Those are different measurements. A benchmark supplies a defined task and
automated criterion. A developer also spends time interpreting requirements, reviewing
changes, coordinating with others, and maintaining the software later.

METR's [July 2025 study][metr-2025] makes the measurement problem concrete. In its particular
randomized study of 16 experienced open-source developers and 246 tasks, access to the
early-2025 tools studied increased completion time by 19%. That was a result for a defined
population, tool generation, and workflow, not a permanent verdict on AI assistance.

The same research group's [February 2026 update][metr-2026] is important context. It says
its newer experiment was affected by selection problems, including developers and tasks
being withheld from conditions that disallowed AI. The authors considered their newer
estimate an unreliable measure of the current productivity effect. Taking either headline
as a universal answer would erase what the studies actually measured.

A resolved issue establishes that the submitted system produced a patch accepted by this
evaluator. Raising that rate is meaningful progress toward a dependable coding assistant.
Establishing its effect on human engineering work requires additional evidence.

## 3. Follow one issue through the evaluator

A public development task identifies a repository and a `base_commit`: the historical
version of that repository immediately before the reference fix. Its `problem_statement`
describes the work to do. Optional `hints_text` can supply additional issue context.

The agent works in a prepared repository at `/workspace`. The snapshot retains enough Git
state for ordinary inspection and diff generation, while future commits containing the fix
have been removed. The public development records also contain a reference `patch` and
`test_patch`; these support training and local verification. **They are not answers to pass
into the agent during a held-out evaluation.** The hidden scoring tasks do not expose their
reference fix or verification tests to the agent. [Dataset specification][data]

The documented evaluation uses separate environments for producing and checking a patch:

<figure class="gemma-functional-figure">
<div class="gemma-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part1-fig2-agent-verifier.png" alt="Separate model serving, agent execution, editable workspace and fresh verification, connected by the extracted Git diff" width="2240" height="1770" loading="lazy">
</div>
<div class="gemma-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part1-fig2-agent-verifier-mobile.png" alt="Separate model serving, agent execution, editable workspace and fresh verification, connected by the extracted Git diff" width="780" height="3180" loading="lazy">
</div>
<figcaption markdown="1">

Figure 2. The model server generates tool calls; the harness executes file operations and commands in the agent sandbox. Verification starts from a fresh copy of the same base code, applies the diff, and runs evaluator-held tests. Running processes and in-memory state do not transfer with the patch.

</figcaption>
</figure>

The agent repeats a practical loop: inspect the code, form a hypothesis, edit, run a
focused check, and use the observation to decide what to do next. That loop belongs to the
patch-generation attempt. The fresh verifier later asks whether the extracted patch works
without the agent's running process or temporary environment.

The three stages produce different outputs. Repository state is useful during debugging; only the
extracted patch is transferred to verification.

| Stage | Input → output |
|---|---|
| Agent attempt | Issue and editable repository → file changes, checks, and a trajectory. |
| Patch extraction | Working-tree changes against the prepared baseline → a Git diff. |
| Fresh verification | Baseline, agent diff, and evaluator tests → a resolved or unresolved result. |
{: .quick-summary }

In the inspected `swegemma 0.2.7`, environment setup records the prepared file state in a baseline commit and points the tag
`_swegemma_baseline`—a name for that version—to it. The agent then edits files in `/workspace`. Git already tracks
existing files such as `query/limits.py`, so an edit to that file appears in a diff. A newly
created file such as `repro_limit.py` is initially **untracked** and does not appear in an
ordinary diff. The released `submit_patch()` tool handles that distinction with these commands:

```bash
git add -N .
git diff --binary _swegemma_baseline
```

The two commands run in sequence, but perform different jobs.

| Command | What it does |
|---|---|
| `git add -N .` | Registers new file paths so Git includes them in the comparison. It does not save their contents in a commit. |
| `git diff --binary _swegemma_baseline` | Compares current files with the baseline version and emits the changes as patch text, including the newly registered files. |
{: .quick-summary }

`-N` means **intent to add** and applies to new, non-ignored files under the current directory.
`--binary` allows binary changes in the patch too. If the baseline tag is unavailable, the
tool falls back to `git diff --binary HEAD`; `HEAD` denotes the current commit.

For example, editing `query/limits.py` and leaving a new `/workspace/repro_limit.py` produces a
patch with both the repair and the reproduction script after `git add -N .`. The script need not
have been committed. Remove scratch files from the repository before extraction, or create them
outside it, such as under `/tmp`, when the task permits. New implementation files that belong to
the repair should remain. [Released patch-extraction source][wheelhouse]

In the second environment, the harness applies that patch to a fresh baseline, attempts to
restore protected test and runner-configuration files, applies the evaluator's `test_patch`,
and runs the selected tests. The released verification code derives its
pytest targets from the test patch. The dataset's account of how reference solutions were
curated also discusses whole-suite checks; that curation step and the per-task grading
procedure are distinct. The released verifier also checks that the test run produced a valid,
nonempty test result; a successful process exit alone is insufficient.
[Harness guide][data] and [released verification source][wheelhouse]

It helps to distinguish the checks the agent can use from the checks that assign the
score. **Existing repository tests** describe behavior already covered by the project.
The agent may inspect them and, where the task's execution rules allow, run them. It can
also make a **focused check** for its current hypothesis: in our zero-limit example, an
assertion that an explicit zero remains zero. These checks help the agent choose a repair.

The **evaluator's verification tests** answer a different question: does the final patch
satisfy the held-out grading criterion? The harness supplies them to the fresh verification
environment. They are not a test oracle the hidden-task agent can repeatedly query while
choosing its patch. In public development, we can study the supplied reference material
for tasks assigned to development; we keep it out of the agent's inputs when measuring
performance on tasks reserved for evaluation. Passing an agent's own check is encouraging,
but the later evaluator outcome is what counts toward the score.

The restoration step needs an implementation qualification. In the inspected
`swegemma 0.2.7`, it uses one `git checkout HEAD -- ...` for a list that can include missing
paths, and suppresses the command's failure. A reproduction with Git 2.54 leaves modified
tracked tests in place when that bulk checkout fails; the separate clean removes named
untracked files. We therefore cannot assume that every test edit is discarded. This is a
limitation of the inspected local source, not proof of the hidden scorer's behavior.

Two practical consequences follow. First, keep the agent focused on the implementation;
changing a test's expectations does not demonstrate that the reported behavior was repaired.
Second, success should be reproducible from the extracted patch alone. An in-memory change,
a package installed interactively, or an unrecorded environment adjustment is not a reliable
solution artifact.

{% include interactive/article-lab.html id="gemma-patch" %}

## 4. What earns a point

For an evaluation split with $N$ tasks, define $r_i=1$ when the submitted patch resolves task
$i$ under the verifier, and $r_i=0$ otherwise. The score is the resolution rate:

$$
\mathrm{Score}=\frac{1}{N}\sum_{i=1}^{N}r_i.
$$

A task is resolved when its verification run succeeds. There is no separate leaderboard
reward for a convincing explanation, a useful intermediate observation, or passing most of
the selected tests. Multiplying the fraction above by 100 expresses it as a percentage.
[Evaluation definition][competition]

For example, on an **illustrative** split of 60 tasks, 9 resolved tasks give
$9/60=0.15$. Resolving one additional task changes that score by about 0.0167. The organizers
describe approximately 120 hidden tasks split evenly between Public and Private; the
60-task example is arithmetic, not a claim that the final split size has been fixed at 60.

This metric makes reliability a direct source of performance. Suppose an agent locates a
bug correctly but a malformed edit request prevents the change from reaching disk. Improving
the edit procedure can recover that task without changing the model's understanding.
Likewise, excessive investigation on one task can reduce the time available for later tasks.

It helps to distinguish three observations in a local run:

| Observation | What it establishes |
|---|---|
| The agent session ended | The control loop stopped. |
| A nonempty patch was extracted | There are changes that can be sent to verification. |
| Verification passed | This task earned a resolved result under this evaluation. |

Only the last observation is the competition objective. The other two explain where work
was lost. An agent timeout may still leave a patch that the local harness recovers and
verifies, so runtime status and resolved status should both be retained.

The Public leaderboard uses roughly half the hidden test data, while final placement uses the
other half. Repeatedly selecting changes from Public scores can favor configurations that do not
transfer to the final set. Development-task traces provide a way to investigate why a change helps
before spending a limited leaderboard submission to test its transfer.

### 4.1 Reliability accumulates across the whole attempt

A repair can fail during search, editing or checking even when the other steps look
good. The final score depends on making the whole sequence work. The calculation below
puts that familiar idea into probabilities; it is optional for following the first run.

<details markdown="1">
<summary>Optional: why several fairly reliable steps can still produce a low success rate</summary>

Consider an illustrative four-stage agent: it finds the relevant location, makes an
appropriate change, carries out the edit correctly, and produces a patch accepted by
fresh verification. Let $A_j$ mean that stage $j$ succeeds and $S$ mean that all four succeed.
The probability of completing this workflow can be written as:

$$
\begin{aligned}
\Pr(S)={}&\Pr(A_1)\,\Pr(A_2\mid A_1)\\
&\times\Pr(A_3\mid A_1\cap A_2)\\
&\times\Pr(A_4\mid A_1\cap A_2\cap A_3).
\end{aligned}
$$

The vertical bar means "given that": each factor measures success **among attempts that
completed the preceding stages**. This is the probability chain rule, so it does not
assume that failures at different stages are independent. The stages are a diagnostic
model of this illustrated workflow; Kaggle does not award separate points for them.

Suppose each conditional success rate is an illustrative 0.8. Then $0.8^4=0.4096$,
or about 41%. Out of 100 starting attempts, about 80 complete the first stage, 64 the
second, 51 the third, and 41 all four.

The example shows why a model that often seems to understand the issue can still have a
disappointing final score. Several opportunities to fail sit between understanding and
verification. It also explains why traces matter: without them, a lower score does not
tell us whether to improve localization, edit syntax, testing, or stopping.

</details>

This is a **behavioral** test, not a requirement to reproduce the reference patch byte for
byte. A different implementation can pass if it satisfies the verifier. The converse
boundary matters too: an automated test suite checks the behaviors it exercises, not
every property a human maintainer might care about. We should aim for a faithful repair
of the issue rather than treating the tests as permission to damage untested behavior.

Repeated attempts also need careful interpretation. If a local experiment generates ten
patches and uses the reference tests to select the passing one, it has given the selector
information that the competition agent does not have. Trying alternatives inside a task
can be useful, but it must fit the real time budget and choose using information available
to the agent. This is why a paper's multi-sample result cannot simply be adopted as this
agent's expected resolution rate.

## 5. How code generation became repository work

Repository repair combines several research problems: checking whether generated code works,
choosing actions from tool observations, navigating unfamiliar code, and learning a useful
procedure. The following studies isolate different parts of that system. Their models, datasets,
and budgets differ, so their headline scores provide design context rather than estimates of this
competition's performance.

### 5.1 First, judge code by running it

In *Evaluating Large Language Models Trained on Code* (2021), Chen and colleagues introduced
the code-trained Codex model and HumanEval. HumanEval gives a model a Python function
specification and checks its generated implementation with tests. The important idea for
our purposes is **functional correctness**: two pieces of code can look different and
still compute the right result. Conversely, code that looks convincing can fail when run.
[Paper][humaneval-paper] · [HumanEval GitHub][humaneval-code]

The paper also considers multiple sampled answers. Finding a correct answer somewhere
among several candidates is a different achievement from producing a correct first
answer. That distinction also explains the retry and candidate-selection choices discussed earlier.
But a function-completion problem still tells the system roughly where its answer belongs.
Our opening bug report did not. Someone first had to discover the helper.

### 5.2 Then, let an action change the next decision

*ReAct: Synergizing Reasoning and Acting in Language Models* appeared as a 2022 preprint
and an ICLR 2023 paper. Its central pattern alternates reasoning, actions, and observations:
the model acts, learns something from the result, and decides what to do next. Its original
experiments were not this repository-repair task, but the mechanism gives us a useful way
to understand a coding agent. [Paper][react-paper] · [Authors' examples][react-project]

In our limit example, reading the request handler could reveal that zero survives parsing.
The next search should then move downstream. If the agent continues rewriting the parser,
it is ignoring the observation that should have changed its diagnosis. The value of the
loop comes from updating the plan with evidence. A long explanation between every pair of
commands is not the essential ingredient.

### 5.3 Move the problem into an existing repository

*SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* began as a 2023 preprint
and was presented at ICLR 2024. It constructs tasks from issues, historical repository
states, and human changes, then uses execution to evaluate generated repairs. The system
must understand and modify software that already exists. The challenge now includes
finding the relevant code and respecting its surrounding behavior.
[Paper][swebench-paper] · [Benchmark GitHub][swebench-code]

That structure explains the Gemma data bundle: an issue alone is insufficient without the
right repository version, and a patch alone is insufficient without a way to verify it.
The competition explicitly invokes a SWE-bench-like pass/fail approach. Its particular
tasks and verification procedure still come from Kaggle's own specification; importing a
score or test rule from another SWE-bench variant would answer a different question.

### 5.4 Make the computer interface easier for the model to use

*SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering* (2024) studies
the interface between model and computer. Search summaries, bounded file views, editing
feedback, and history management affect whether an agent can carry out a repair. An
excellent diagnosis is of little help if the next edit is malformed or the useful search
match disappears in a huge observation. [Paper][sweagent-paper] · [Project GitHub][sweagent-code]

This is directly relevant to the questions we can test here. Should a tool return a whole
file or a focused range? Should the agent edit a short matching string or rewrite a large
module? What does it see after a command fails? The historical paper does not choose the
best settings for Gemma; it shows why interface choices belong in the experiment.

### 5.5 Separate decisions from the environment that executes them

The OpenHands paper, first released in 2024 and accepted at ICLR 2025, describes a broader
platform with an agent, runtime, actions, observations, skills, and delegation. Its
architecture makes a useful distinction: deciding to run a command and actually executing
that command are separate responsibilities. [Paper][openhands-paper] · [Project GitHub][openhands-code]

Gemma's competition harness is its own system, not OpenHands. Nevertheless, this separation
helps diagnose failures. A model can request an appropriate test while the runtime cannot
import a dependency. It can also run every tool successfully and repair the wrong behavior.
Those failures need different interventions. OpenHands' broader browser and development
capabilities do not automatically exist in this competition's offline sandbox.

### 5.6 Ask how much machinery is really helping

The mini-SWE-agent project, publicly released in 2025, makes a small, inspectable control
loop a deliberate design goal. Its documentation emphasizes a simple interaction pattern
and readable trajectories. It provides a useful implementation to study when a complex
agent framework makes it difficult to see what the model actually received and did.
[Documentation][mini-docs] · [GitHub][mini-code]

There is a related lesson in *Agentless: Demystifying LLM-based Software Engineering
Agents* (2024). It explores a prescribed sequence of localization, repair, and validation
rather than letting the model freely choose every next action. The two projects are not
the same design. Together, they give us a productive question: which decisions need model
judgment, and which can be made reliably by a simpler procedure?
[Agentless paper][agentless-paper] · [GitHub][agentless-code]

None of these projects is a drop-in Kaggle submission. Their value is to make the design
space visible. A baseline can be simple while still being a serious research control.
Additional agents, memory, retrieval, or training should earn their place through the
tasks they help resolve.

## 6. What the development data gives us

The word **public** is used in two different ways here. Downloadable development data
is public material we can inspect. The **Public leaderboard** displays a score, but that
does not make its underlying issues or answers available as development examples.

| Name | What the participant can see | What it is for |
|---|---|---|
| Public development data | Released issues, repository snapshots and reference material. | Build and debug the agent; reserve local evaluation tasks before tuning. |
| Public leaderboard split | A score from part of the hidden evaluation; not its task answers. | Limited feedback during the competition. |
| Private leaderboard split | Hidden evaluation used for final ranking. | Judge the submitted design on the final scoring split. |

Within the development data, we also need a local separation of roles. We may use a
training portion to learn weights, a development portion to choose prompts or settings,
and a **holdout** kept aside for a later check. Even if we train no weights, repeated prompt
selection is still development: we are learning which design works on the examples we see.
A holdout is useful because it asks whether that choice works on examples we did not use
to make it. It becomes less informative when we keep consulting it and adjusting to it.

The released repairs provide examples of a procedure that must work on other issues: understand
the project, locate the fault, edit the implementation, and check the result. The distribution of
those examples determines which behaviors development results can meaningfully test.

The data page lists **129 public development tasks**, 782 files, and approximately 22.42 GB.
The repositories are FastAPI, Rich, Requests, and HTTPX. Each task has a frozen repository
snapshot, with code graphs and node embeddings supplied to assist navigation. [Data page][data]

Reading the downloaded `tasks.jsonl` gives the following distribution:

| Repository | What the project does | Tasks | Share |
|---|---|---:|---:|
| `fastapi/fastapi` | [A framework for building web APIs][fastapi-docs]. | 67 | 51.9% |
| `Textualize/rich` | [Formats text, tables, and other output in a terminal][rich-code]. | 48 | 37.2% |
| `psf/requests` | [An HTTP client for making web requests from Python][requests-docs]. | 13 | 10.1% |
| `encode/httpx` | [An HTTP client supporting synchronous and asynchronous use][httpx-docs]. | 1 | 0.8% |
| **Total** | | **129** | **100.0%** |

FastAPI and Rich account for 115 of the 129 tasks, about 89%. The four repository names
therefore do not describe four equally represented validation populations. Even a reader
who has never used these projects can see the consequence: a method that happens to suit
the two largest groups can dominate the average while adding little evidence about the
smaller groups.

<figure class="gemma-functional-figure">
<div class="gemma-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part1-fig3-task-composition.png" alt="Public development task counts: FastAPI 67, Rich 48, Requests 13, HTTPX 1." width="2000" height="1040" loading="lazy">
</div>
<div class="gemma-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part1-fig3-task-composition-mobile.png" alt="Public development task counts: FastAPI 67, Rich 48, Requests 13, HTTPX 1." width="919" height="1340" loading="lazy">
</div>
<figcaption markdown="1">

Figure 3. Public development task composition. FastAPI and Rich account for 115 of 129
tasks (89.1%). This distribution does not describe the hidden repositories. Source: the
released task inventory checked on September 25, 2026. [Dataset][data]

</figcaption>
</figure>

For someone coming from a conventional Kaggle competition, the data format is another
important change. A row is an executable work assignment. It does not just contain input
features and a target value. To evaluate the row, we need its historical program, the
dependencies needed to run it, and the tests that distinguish a repair from a failure.
That is why a dataset with only 129 tasks can contain many gigabytes of files.

### 6.1 A real development issue: the order of two parameters

The public record `fastapi_11194`, also used in the later command example, concerns an
endpoint accepting both a file and a form field. A web **endpoint** is a function that
handles a particular kind of request. A form field might contain a name; a file field
might contain an uploaded document.

The supplied description reports that putting the `Form` parameter before the `File` parameter
causes a validation error with response code 422, whereas declaring the file first avoids it. It
also calls for coverage of multiple files and parameter-order independence. The [public pull
request][fastapi-example] corroborates that description. The following investigation is a proposed
approach to that task, rather than an observed agent trajectory.

The desired behavior is easy to explain to a user: rearranging the declaration of these
inputs should not change whether an otherwise valid upload works. The implementation
question is more subtle. Does the ordering affect the description of the request body,
the extraction of individual fields, or validation after extraction? Which assumption
about the first parameter could incorrectly affect later parameters?

A first investigation would read the existing file/form tests and the code that turns endpoint
parameters into request-processing behavior. A compact reproduction would compare the two
parameter orders with equivalent input. The returned validation details can then help locate which
processing layer to inspect. This derives a search strategy from the symptom instead of using the
reference patch to reveal the repair location.

`fastapi_11194` belongs in the development partition once its description and reference material
have informed the design. A later success on it is then a result on a familiar task, rather than
untouched holdout evidence. Evaluation should still use the dataset's repository snapshot; the
upstream source may contain changes made long after that snapshot.

### 6.2 What is inside the task bundle

| Resource | What to use it for |
|---|---|
| `tasks.jsonl` | Issue descriptions, repository identities, base commits, reference fixes, and verification patches. |
| `snapshots/<instance_id>.tgz` | Recreate the exact code the agent is supposed to repair. |
| `graphs/` | Look up relationships among indexed code symbols. |
| `embeddings/` | Retrieve symbols similar to an existing indexed symbol. |
| `wheels/` | Install historical repository dependencies without network access inside the sandbox. |
| `docker/` and `sandbox/` | Prepare the repository execution environment. |
| `sample_submission/` | Inspect the organizer's configuration conventions. |
| `HARNESS_README.md` | Understand the intended execution and submission contract. |

The `.jsonl` suffix means one JSON object per line. A development record contains several
different kinds of information, and separating them early prevents evaluation mistakes:

| Information | Examples | Role |
|---|---|---|
| Task input and identity | `instance_id`, `repo`, `base_commit`, `problem_statement`, `hints_text` | Identify and present the issue. |
| Reference solution | `patch` | Study or train on solutions within the chosen training partition. |
| Verification material | `test_patch` | Check whether a generated solution resolves the task. |
| Contextual metadata | `created_at` | Analyze chronology and construct development splits. |

Reference fixes can support training and error analysis. For held-out evaluation, choose
the partition before using its patches or trajectories to tune the agent, and keep that
solution material on the evaluator side. In the downloaded public records, every
`hints_text` is empty, even though the schema permits hints; the baseline should work from
the issue and repository without depending on extra comments.

The target population also matters. The hidden tasks were curated from **private
repositories**, whereas the development tasks come from four public projects. Learning
general repository navigation and debugging behavior is more likely to transfer than
encoding assumptions about one familiar project's directory layout.

Here, **generalization** means that a useful behavior survives that change of setting.
"Trace the input through its callers before editing a shared helper" is a procedure that
could apply to another project. "The answer to this type of issue is always in this named
file" may only exploit familiarity with a public repository. A benchmark improvement is
more persuasive when the evidence suggests the first kind of learning.

### 6.3 Graphs provide a view of the code

An abstract syntax tree, or AST, represents the syntactic structure of a program. A code
graph adds relationships among entities such as functions and classes. These can help an
agent move from a suspicious helper to its callers without reading every source file.

The three graph tools are `get_code_neighbors`, `get_code_subgraph`, and
`search_similar_code`. The last name deserves attention: in the supplied implementation,
the query is resolved against **existing symbol keys or suffixes** in the embedding archive.
It does not run a fresh neural embedding model on an arbitrary English issue description.
Start with a symbol such as `HTTPConnection`, then retrieve similar indexed code.

In the invented limit example, a graph might connect `parse_options` to `normalize_limit`
and show several other callers of the same helper. That would make the risk of a shared
change visible. An **embedding** represents an indexed code item as a vector of numbers;
similarity search compares such representations to find nearby items. Similarity does
not establish that an item causes the bug. It suggests where to inspect next.

For example, after text search identifies an indexed function, its stored vector
$\mathbf{q}$ can be compared with another indexed symbol's vector $\mathbf{z}_j$.
For nonzero vectors, cosine similarity is

$$
s_j=\frac{\mathbf{q}^{\mathsf T}\mathbf{z}_j}
{\lVert\mathbf{q}\rVert\,\lVert\mathbf{z}_j\rVert}.
$$

A larger value means the vectors point in a more similar direction; it is not a probability
that the second symbol contains the bug. The supplied tool starts from an existing symbol.
Searching from an arbitrary issue description would require a compatible query-embedding
method; the formula alone does not supply one.

These tools answer different navigation questions. A hybrid procedure can begin with a literal
name from the issue and expand only after identifying relevant source or an indexed symbol.

| Question during investigation | Suitable lookup |
|---|---|
| Where is this API name, error text, or field used? | Text search in source files; no graph entry is required. |
| Which indexed symbols are connected to this one? | `get_code_neighbors`; inspect the returned relationships. |
| What is the connected code around these indexed symbols? | `get_code_subgraph`; select a bounded region. |
| Which indexed code items resemble a known symbol? | `search_similar_code`; start from an existing embedding key or suffix. |
{: .quick-summary }

If a lookup returns nothing, inspect the source and continue with text search. A missing graph
entry does not establish that the implementation is absent. Compare whether each added lookup
finds useful code within the time and context budgets before introducing a more elaborate
retrieval policy.

There is also a difference between a static map and a running program. Python can choose
functions dynamically, generate attributes, or dispatch through wrappers. A graph can
therefore guide navigation without containing every relationship that matters at runtime.
The source and an appropriate execution check remain part of the investigation.

The September 25 competition file listing reports 129 zero-byte entries in `graphs/` and another
129 in `embeddings/`. A [participant investigation][graph-discussion] attributes the empty copies
to the handling of task-name and commit-name hard links, and also reports missing async functions.
The zero-byte file sizes were checked against that listing; the async coverage measurements are
the participant's findings. At that date, the host had acknowledged the report and was
investigating.

That makes a useful first design requirement: **the agent must still navigate when graph
lookup fails or an expected symbol is absent.** Ordinary file inspection, text search, and
Python's own AST parser remain useful. Preserve the original downloaded data when diagnosing
an issue, and keep any local repair separate so the comparison records which data it used.

## 7. The constraints become part of the algorithm

The [competition model specification][competition] checked on September 25 permits one specific
model:

```text
gemma-4-31b-it-qat-w4a16-ct
```

Every model-backed agent and subagent must use that model. The harness registry contains
other model aliases, but the existence of an alias in a library does not make it eligible
for this competition. The stated variant uses four-bit weights and sixteen-bit activations;
the organizer supplies the base model during hosted evaluation.

The long checkpoint name is worth decoding once. `it` denotes instruction tuning; `qat`
denotes quantization-aware training; `w4a16` describes the four-bit weight and sixteen-bit
activation format; `ct` refers to compressed-tensors packaging. **Quantization** stores
some numerical values at lower precision to reduce resource requirements. It is a
representation choice, not permission to swap in a different small model. The exact
checkpoint still matters when loading an adapter or reproducing an experiment.
[Permitted model files][model]

| Constraint | Consequence for the first implementation |
|---|---|
| Four L4 GPUs, 96 GB aggregate VRAM | The hosted runtime is a specific GPU environment; local timing on other hardware needs separate interpretation. |
| 32,768-token context | Instructions, observations, reasoning, and output must share the available context. Read selectively. |
| Total unpacked submission below 3 GiB | Package configurations and optional adapters; do not include a full base-model download. |
| Offline task sandbox | Depend on the supplied environment and wheels, not a network install during a task. |
| Docker sandbox: 4 GiB RAM and 2 vCPUs | Broad test runs and large analysis processes can exhaust resources independently of model inference. |
| Restricted declarative agent configuration | Use registered tools and sandboxed skills instead of an arbitrary host-side Python entrypoint. |
| Twelve hours for all patch generation | Budget across tasks, including sandbox setup; verification time is excluded from this stated limit. |

These are the September 25 conditions in the [harness guide][data]. Some defaults describe the
reusable local library, while the hosted scorer has its own integration. The distinction matters
when reproducing a run: a valid local configuration does not establish which limits the hosted
integration applies.

GPU memory and the repository sandbox's RAM are separate limits. The model server performs
inference on the GPUs. The agent's command that imports a package or runs pytest executes
inside the repository environment. Giving the model server more memory does not make a
memory-hungry test process fit in its four-GiB sandbox.

The model also needs memory beyond its stored weights. While generating a response, it
keeps attention state for the current sequence, commonly called the **KV cache**. The
PagedAttention paper explains how managing that growing state in blocks can reduce memory
waste; vLLM provides a serving system around this class of problem. This is why "the
weights fit" does not imply "any history length or number of simultaneous requests fits."
[PagedAttention paper][pagedattention-paper]

For our agent, context is its current working material: instructions, issue text, relevant
code, observations, and room for the next output. An old file dump occupies space even
after it stops helping the investigation. Good context management means preserving the
facts that change the next decision, such as the current call path or a disproved hypothesis.

### 7.1 There are two clocks

The local agent-session timer begins after preparation of the task environment. The hosted
12-hour limit includes that preparation. Reducing the per-task agent timeout therefore
controls only part of the total duration.

A [host reply checked on September 25][runtime-discussion] states that hidden tasks run
sequentially. Under the behavior described in that reply, exhausting the global limit produces a
submission error. The host planned to assign zero to unfinished tasks instead, but the reply did
not establish that this change had been deployed.

A useful accounting equation for sequential patch generation is:

$$
\begin{aligned}
T_{\mathrm{gen}}&=h+\sum_{i=1}^{N}(s_i+a_i)\\
&\leq720\ \text{minutes}.
\end{aligned}
$$

The right side is the twelve-hour generation budget. The left side separates the time
we need to measure:

| Symbol | Meaning | What to record |
|---|---|---|
| $N$ | Number of tasks in the run. | The fixed task manifest, including failures. |
| $a_i$ | Actual agent-session time for task $i$. | Session start and end; also retain the configured timeout. |
| $s_i$ | Preparation and other counted overhead belonging to task $i$. | Setup and cleanup intervals, with retries where applicable. |
| $h$ | Shared overhead counted by the global timer. | Count it once, outside the task intervals. |
| $T_{\mathrm{gen}}$ | Total counted patch-generation time. | The complete run's timer, checked against the component records. |

Intervals should not overlap in the accounting. Verification time is outside this stated
generation budget; adding it to the left side would answer a different timing question.

For a planning calculation with 120 tasks, $720/120=6$ minutes per task **including setup**.
This is an average allocation, not an official per-task allowance. A three-minute agent
limit would account for up to 360 minutes of agent-session time across those 120 tasks,
leaving a nominal 360 minutes for setup and other counted overhead. Actual startup, retries,
cleanup, task count, and timeout enforcement must still be measured. A per-task setting
alone cannot certify that the complete submission fits.

<figure class="gemma-functional-figure">
<div class="gemma-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part1-fig4-generation-budget.png" alt="Illustrative twelve-hour budget: one to six agent minutes per task consume two to twelve hours across 120 tasks; three minutes leaves six hours for setup and overhead." width="2000" height="1120" loading="lazy">
</div>
<div class="gemma-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part1-fig4-generation-budget-mobile.png" alt="Illustrative twelve-hour budget: one to six agent minutes per task consume two to twelve hours across 120 tasks; three minutes leaves six hours for setup and overhead." width="919" height="1440" loading="lazy">
</div>
<figcaption markdown="1">

Figure 4. Agent work and setup share the generation budget. Each bar assumes 120
sequential tasks using their full agent allowance. At three minutes per task, 360 minutes
remain for setup and other counted overhead. At six minutes, none remain. Verification is
excluded. This is planning arithmetic, not a runtime measurement.

</figcaption>
</figure>

<details markdown="1">
<summary>Optional: express the time-allocation problem mathematically</summary>

The general optimization problem is to spend effort where it increases expected resolved
tasks. If $p_i(t_i)$ is the probability of resolving task $i$ after spending $t_i$ time and
$s_i$ is its counted per-task overhead, a useful planning model is:

$$
\begin{gathered}
\max_{t_1,\ldots,t_N}\quad \sum_i p_i(t_i)\\
\text{subject to}\\
h+\sum_i(s_i+t_i)\leq720\ \text{minutes}.
\end{gathered}
$$

Here $t_i$ is a planned agent-time allocation, while $a_i$ above is time actually observed.
The probability curve belongs to a particular agent and runtime; it is not a property of
the issue alone. We do not know these curves at the start. Fixed, moderate limits are a
measurable baseline. More elaborate decisions about continuing or stopping should come from observed
trajectories and their marginal value.

</details>

In ordinary language, the question is whether the next minute is likely to buy a useful
observation or repair. An agent one targeted test away from confirming a plausible change
is in a different position from one repeating a search that has already failed. The
challenge is recognizing the difference from information available during the run.
We should not label a task "easy" using a reference answer that the deployed agent cannot see.

### 7.2 Tool outputs are also a budget

Before a request reaches generation, the serving limit must accommodate both its input
and the **requested output allowance**:

$$
C_{\mathrm{prompt}}+M_{\mathrm{requested}}\leq32768.
$$

Here $C_{\mathrm{prompt}}$ includes instructions, issue text, tool schemas, observations,
and template overhead. With the example's requested output allowance
$M_{\mathrm{requested}}=8192$, the input can use at most 24,576 tokens. A response that
would eventually be short does not remove this admission constraint. Leave room below the
boundary rather than filling the entire advertised context with source code.
[Serving request validation][vllm-context]

The guide documents default command output truncation at 5,000 characters and `read_file`
limits of 150 lines and 10,000 characters. Printing an entire repository does not put that
repository into context. It produces a truncated observation and consumes time.

An agent should narrow the search, inspect the relevant slice, and ask for the next slice
when necessary. Its internal summary should preserve file locations, the current hypothesis,
changes made, and checks already run. Automatic context compaction can help a long session
continue, but it cannot choose which debugging facts the agent should have made explicit.

## 8. Before the first model run: learn what one task asks of you

Several prerequisites can be checked before loading model weights. Read a practice issue as a
developer, identify what would distinguish a repair from the original behavior, and verify that
the local evaluator can detect that difference. This separates a failure to reconstruct the
environment from a failure by the model to solve the issue.

### 8.1 Choose the practice issue before looking at solutions

First reserve a small set of tasks for hands-on exploration. These become development
examples: once we inspect their answers or repeatedly adjust a prompt around them, their
later success is no longer an independent test of the agent. Keep a separate holdout for
later evaluation. A **holdout** is simply a group of issues whose solutions and results
are kept out of the design loop until a candidate is ready to assess.

Select practice issues using their identities and available descriptions, not a baseline's
eventual successes. Check that their snapshots and dependencies exist. With only one HTTPX
task in this release, a supposedly balanced plan of three tasks from each repository is
impossible. Let the actual inventory determine the plan, and record which repositories
the practice set represents.

Read one practice issue without opening its reference patch. Write down three things in
plain English: what currently happens, what should happen, and what observation would
distinguish the two. If those are unclear, the model will face the same ambiguity. Some
issues request a new behavior instead of reporting a simple exception; an agent needs to
infer the intended API change from the task, not assume every issue is a crashing test.

Then inspect the repository's top-level layout, dependency metadata, and nearby test style.
A stack trace gives a route into code. An API name gives a search term. A prose description
of formatting may require finding the rendering path. Record how you located the relevant
implementation. This short manual exercise reveals what useful model observations should
look like before any prompt is tuned.

### 8.2 Establish a negative and a positive control

An experiment needs a check that should fail and a check that should pass. On a selected
development issue, the unchanged repository and the supplied reference repair provide
useful starting controls. Use the official task-verification path to examine both, with
the reference material kept on the grader side.

| Control | Intended observation | What an unexpected result would make me inspect |
|---|---|---|
| Unchanged repository, with the issue's verification tests | The issue is not resolved. | Whether the tests exercise the reported behavior and whether the correct snapshot was loaded. |
| Reference patch on a fresh copy, with the same verification tests | The issue is resolved. | Dependency versions, patch application, test selection, and environment reconstruction. |
| Agent patch on another fresh copy | An independently determined pass or fail. | The actual changed code and the verifier output. |

These are diagnostic expectations, not a guarantee that every public record will behave
perfectly in every local setup. If the reference repair cannot pass, preserve the failure
and investigate it before treating that task as evidence against the model. If an unchanged
repository passes unexpectedly, a later agent pass might not mean what we think it means.
Controls do not repair evaluation defects automatically; they reveal where to look.

Only after your unaided exploration should you open the practice issue's reference patch.
Compare its location and scope with your hypothesis. The valuable lesson may be that the
change lives in a shared abstraction rather than the visible caller. Keep that lesson
general when designing the agent. A prompt that embeds the answer's exact filename has
turned this practice task into an answer lookup.

### 8.3 Separate three kinds of preparation

| Preparation level | What it establishes |
|---|---|
| CPU machine | Rules, task inventory and split are recorded; agent files and archive layout are inspectable. |
| Docker with prepared task data | The historical repository and dependencies can be reconstructed; unchanged/reference controls can be checked without inference. |
| Compatible GPU host | Gemma serves model requests and the full agent–tool–verification path can be exercised. |
{: .quick-summary }

This order catches missing snapshots and dependency failures before an expensive model run. The
repository's tests execute separately from inference, so access to four GPUs is not a prerequisite
for checking the first two levels.

The first GPU session should answer operational questions: can the model call the tools,
can the harness extract the patch, and how much time does each stage take? Start with one
issue, then two or three preselected development issues whose traces can all be read.
Once the path works, a panel of roughly 8–12 issues can expose timing and failure patterns
before a broader quality comparison. These are planning sizes, conditional on the frozen
split and available compute, not statistically reliable evaluations.

## 9. Build a baseline whose behavior is easy to inspect

The baseline prompt specifies the following loop. Its trace is needed to determine whether the
agent actually follows it:

1. Translate the issue into a concrete expected behavior.
2. Locate the implementation and relevant existing tests.
3. Reproduce the failure with a focused check when feasible.
4. Make the smallest change supported by the evidence.
5. Test the changed behavior and nearby behavior likely to regress.
6. Review the diff, remove scratch files, and submit the patch.

Google's Agent Development Kit, or **ADK**, supplies the agent abstractions used by the
competition's restricted configuration compiler. An `LlmAgent` is its model-backed agent
type. One `LlmAgent` is sufficient to express this loop. Starting with one model-backed worker
also makes it possible to identify whether failures originate in navigation, reasoning,
editing, testing, or stopping. A team of agents can be introduced after there is a baseline
against which to measure its additional work.

The initial package below has no adapter declaration. This matters because the supplied
sample demonstrates multi-LoRA routing and names example adapters. Copying those declarations
without providing the corresponding weights would not produce a clean no-adapter baseline.

Each configuration file controls a different part of the loop. **YAML** is a text format for
settings and nested objects; here it tells the harness which supported agent and tools to
construct. A prompt file supplies instructions for model calls. The harness reads both, rather
than running them as independent host-side programs.

| Design choice | Where the example expresses it | What to look for in the run |
|---|---|---|
| Ask for evidence before editing. | `agent/prompts/system.md` | Does the trace connect the proposed change to a reproduced or inspected behavior? |
| Give the agent ways to inspect and change files. | The registered tools in `agent/agent.yaml` | Do tool calls succeed, and does the agent use their returned observations? |
| Bound the work spent on each issue. | `agent/eval_config.yaml` for hosted evaluation; explicit flags for the local CLI. | Actual tool counts and elapsed time, not just the written settings. |
| Adapt behavior through learning, if later justified. | An optional adapter declaration and its weight files. | A held-out comparison with and without that adapter under the same runtime. |

A **baseline** is the reference version against which we judge later changes. Its first
job is to be understandable and repeatable. If we add a second agent, a graph tool and an
adapter all at once, a changed result will not tell us which addition helped. The small
package below gives us a starting point from which those questions can be tested separately.

### 9.1 Create the files on a CPU machine

This part needs a text editor and Python 3. It does not load model weights. Create a new
project directory:

```bash
mkdir -p gemma4-start/agent/prompts gemma4-start/data gemma4-start/results
cd gemma4-start
```

All later relative paths are measured from this project directory. Its layout will be:

```text
gemma4-start/
├── agent/
│   ├── agent.yaml
│   ├── eval_config.yaml
│   └── prompts/
│       └── system.md
├── build_submission.py
├── data/                 # Development data; keep outside the agent archive
└── results/              # Local evaluation results; keep outside the archive
```

Save the following as `agent/agent.yaml`:

```yaml
name: repository_fixer
model: gemma-4-31b-it-qat-w4a16-ct
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
generate_content_config:
  temperature: 0.2
  max_output_tokens: 8192
  thinking_config:
    include_thoughts: false
```

YAML uses indentation to express structure. `instruction` loads a text file relative to the
configuration file containing `!include`. The six tools here are the basic workspace and
lifecycle tools. Graph tools can be added as a controlled variation after the basic run works.

| Tool | What the baseline uses it for |
|---|---|
| `run_command` | Run shell commands, searches, and focused tests in the sandbox. |
| `read_file` | Read a selected line range from a workspace file. |
| `edit_file` | Replace a matching string in an existing, nonempty file. |
| `write_file` | Create or overwrite a file inside the workspace. |
| `get_status` | Inspect consumed and remaining task budgets and patch status. |
| `submit_patch` | Capture the diff and mark the task ready for verification. |

The sampling settings are **illustrative baseline choices**. `temperature` controls sampling
randomness; lower values generally concentrate selection on more likely continuations.
The output limit is per model response, not the total token allowance for a task.

One version-specific detail changes how we configure reasoning. In the inspected
`adk-submission 0.2.11` bridge, `include_thoughts: false` sends
`enable_thinking: false` to the model's chat template. It disables the separate thinking
mode in that request; it is not merely a request to hide displayed thoughts. This names a
generation mode, not a claim that the model can no longer make reasoned decisions. A numeric
`thinking_budget` is accepted by the schema but is not forwarded by this bridge or the
inspected `google-adk 1.36.1` completion-parameter path. We therefore do not treat it as an
enforced reasoning-token cap. [Released generation bridge and ADK source][wheelhouse]

The teaching baseline starts with thinking disabled. A later comparison can explicitly
enable it with `include_thoughts: true`, inspect the actual server request, and measure
resolved issues, output truncation, and runtime. This is a starting control, not evidence
that reasoning mode helps or hurts. YAML validation alone cannot show which settings the
server received.

Save `agent/prompts/system.md`:

```text
You are repairing a Python repository for the issue in the user message.

Work from the repository at /workspace and the supplied issue description.
Inspect the relevant implementation and nearby tests before editing.
State a short, concrete hypothesis about the cause, then check it with code
inspection or a focused reproduction.

Make a small implementation change that addresses the requested behavior.
Preserve unrelated behavior and follow the repository's existing conventions.
Use short, incremental edit requests. Read the result of each tool call and
respond to errors before building further changes on it.

Dependencies are prepared by the harness. Work offline. Follow the task's
execution rules. Run focused checks and relevant existing tests when permitted
and time allows. If pytest or unittest is disabled, use targeted inline Python
assertions instead. Use run_command to create and run scratch scripts in /tmp;
workspace file tools operate inside /workspace.
Do not change test expectations to make the issue appear resolved, and do not
alter harness-generated pytest.ini or conftest.py.

Check get_status during the task. Reserve time to review git diff, remove any
temporary files from the repository, and verify the final change. Do not commit
your edits. Call submit_patch when the patch is ready; treat it as the final
tool action. Keep your final message brief and factual.
```

The prompt connects each action to evidence the agent should inspect. Check selection still
depends on the task's execution rules. The released local configuration enables sandbox testing by
default, while the generated task prompt also supports environments that disable pytest and
unittest and request inline assertions. Read that prompt before choosing a check. Keeping edits
uncommitted simplifies working-tree diff review; final extraction compares against the harness's
prepared baseline.

### 9.2 Set the hosted task limits explicitly

Save `agent/eval_config.yaml`:

```yaml
evaluation:
  timeout_seconds: 300
  max_tool_calls: 40
  max_time_minutes: 3
  max_turns: 80
```

The **`evaluation:` wrapper is taken from the official sample file**. These four fields
control the command timeout, tool-call count, agent-session minutes, and model turns,
respectively. The numerical choices here are proposed starting limits; the public sample
uses a smaller smoke-test budget.

The command timeout also applies to the final pytest invocation in the released local
verifier. We use 300 seconds so the example does not quietly give that check a shorter
60-second cap. This is separate from the three-minute agent session: an agent command is
also bounded by the remaining session time, while fresh verification occurs afterward.
Checking that the unchanged and reference controls produce their expected outcomes helps
establish whether the chosen timeout is adequate for a selected task. A verifier timeout should not be diagnosed as a reasoning failure.
[Released evaluator and verification source][wheelhouse]

The [host clarification][runtime-discussion] says those four fields are the ones the hosted
scorer reads, and describes omitted hosted limits as unlimited. In contrast, the released
local CLI explicitly defaults its time limit to 60 minutes. Moreover, its `eval` command
does not read this YAML file. **For a local run, pass the corresponding CLI flags.**

### 9.3 Build a ZIP with the correct root

The archive is a delivery format. It should contain the configuration and its resources
at the expected paths, so the evaluator can reconstruct the same agent. The recipe below
also prints a hash, which acts as a fingerprint of the exact file sent to Kaggle.

<details markdown="1">
<summary>Copyable recipe: build and inspect the three-file submission archive</summary>

Save this small standard-library script as `build_submission.py` in the project directory:

```python
from pathlib import Path
import hashlib
import stat
import zipfile

root = Path("agent")
expected = {
    "agent.yaml",
    "eval_config.yaml",
    "prompts/system.md",
}
paths = list(root.rglob("*"))
if any(p.is_symlink() for p in paths):
    raise SystemExit("Remove symlinks from the agent package.")
files = {p.relative_to(root).as_posix(): p for p in paths if p.is_file()}
if set(files) != expected:
    raise SystemExit(f"Unexpected package contents: {sorted(files)}")
if sum(p.stat().st_size for p in files.values()) >= 3 * 1024**3:
    raise SystemExit("The unpacked package must be below 3 GiB.")

archive = Path("submission.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for name in sorted(files):
        info = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 0, 0, 0))
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, files[name].read_bytes())

with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None or set(z.namelist()) != expected:
        raise SystemExit("Archive verification failed.")

print("Archive:", archive)
print("Files:", ", ".join(sorted(expected)))
print("SHA256:", hashlib.sha256(archive.read_bytes()).hexdigest())
```

Run it from `gemma4-start/`:

```bash
python3 build_submission.py
python3 -m zipfile -l submission.zip
```

The listing must show `agent.yaml` at the archive root, not `agent/agent.yaml`. The fixed
timestamps make repeated packaging of unchanged files stable within the same build
environment. Record the printed SHA-256 hash with each experiment so the scored archive can
be identified later.

This script deliberately accepts only the three files of this minimal example. Extend it
when adding a real skill or adapter. The harness still needs to compile and execute this
configuration; archive checks alone do not establish that behavior.

</details>

## 10. Prepare local evaluation in the right environment

Local evaluation needs the task data, a model-serving process, and the harness that coordinates
the agent. The harness also starts repository sandboxes for tool execution and fresh verification.
Installing `swegemma` supplies evaluation software; it does not download the tasks or start the
model server.

The package can be written on a CPU machine. The inference commands below target a **Linux NVIDIA
GPU machine** with CUDA and Docker; copy the project directory there if it was created elsewhere.
The September 25 hosted setup uses four GPUs with tensor parallelism of four, dividing the model
across those devices. Other hardware requires its own serving and memory checks. The commands are
examples checked against the pinned source, with end-to-end GPU execution left to validation on
that host.

“Local evaluation” means running the evaluation machinery under our control; it can take place on
a GPU server rather than a laptop. Within that machine, the following components have separate
responsibilities and failure modes.

| Running component | What it does | What should become observable |
|---|---|---|
| Model server on the GPU host | Loads Gemma and answers model requests. | The expected model name and a successful request/response. |
| Local evaluator process | Loads the agent package, prepares tasks and coordinates tool calls. | A recorded issue attempt with its full trace and patch. |
| Repository sandbox | Holds the task's code and executes permitted commands. | File changes, command output and resource/time errors. |
| Fresh verification sandbox | Reconstructs the task and checks the extracted patch. | Verification logs and a resolved/unresolved outcome. |

Prepare these components in dependency order. In particular, the reference controls can check the
evaluator before any model inference is needed.

| Step | Required result before continuing |
|---|---|
| Download task data | Task records, snapshots and dependency files are available. |
| Install host packages | The pinned harness APIs can load the agent configuration. |
| Build the sandbox and run controls | Logs explain whether unchanged and reference-patched code behave as expected. |
| Start the model server | The intended model name answers a request at the selected endpoint. |
| Run one agent task | Trace, extracted patch, verification result and timings are retained together. |
{: .quick-summary }

Model weights, task data, control scripts, and result logs are development resources. Keep them
outside the submitted agent directory; the submission contains only the configuration and
resources the agent itself needs.

### 10.1 Get the data and keep a record of its version

Join the competition and accept its rules through Kaggle before downloading restricted
competition files. Configure the [official Kaggle CLI][kaggle-cli] using its current
authentication instructions; credentials should remain outside the project and submission.

You can begin with the task records and guide:

```bash
mkdir -p data
kaggle competitions download gemma-4-developer-agent \
  -f tasks.jsonl -p data
kaggle competitions download gemma-4-developer-agent \
  -f HARNESS_README.md -p data
```

If a download is wrapped in a ZIP, extract that file into `data/` before continuing. Full
local evaluation also needs repository snapshots, wheels, and the sandbox files. The
complete download is about 22.42 GB, and unpacking and building containers require additional
disk space. When preparing that full local environment, use:

```bash
mkdir -p downloads
kaggle competitions download gemma-4-developer-agent -p downloads
python3 -m zipfile -e downloads/gemma-4-developer-agent.zip data
```

Keep the downloaded archive as the raw copy, and use a separate working copy if
investigating the graph-file issue.

After extraction, organize the files as follows:

```text
data/
├── tasks.jsonl
├── snapshots/
├── graphs/
├── embeddings/
├── wheels/
├── docker/
└── sandbox/
```

The README sometimes shows a `published/` prefix. Use the directory that actually contains
your `tasks.jsonl`; the examples here call that directory `data/`.

Read the task inventory without printing the reference solutions:

```python
import json
from collections import Counter
from pathlib import Path

tasks = [json.loads(line) for line in Path("data/tasks.jsonl").read_text().splitlines()
         if line.strip()]
print("Tasks:", len(tasks))
print("Repositories:", dict(Counter(t["repo"] for t in tasks)))
first = tasks[0]
print("Example identity:", first["instance_id"], first["repo"], first["base_commit"])
```

Also save the package versions and download date. A change to the harness or dependency
bundle can change results even when the agent prompt is identical.

### 10.2 Install the released evaluation packages

The [official wheelhouse][wheelhouse] checked on September 25 contains these core versions:

| Package | Version |
|---|---:|
| `swegemma` | 0.2.7 |
| `adk-submission` | 0.2.11 |
| `adk-eval-core` | 0.1.0 |
| `google-adk` | 1.36.1 |
| `vllm` | 0.19.1 |

The released harness requires Python 3.12 or newer. Its host Python environment and the
Python 3.13 repository sandbox are separate environments.

Download and extract the host package wheelhouse, about 880.83 MB in the inspected version:

```bash
mkdir -p wheelhouse
kaggle datasets download metric/gemma-4-developer-agent-wheelhouse \
  -p wheelhouse --unzip
```

Use a fresh environment on the GPU host. Install the organizer's packages with dependencies resolved for that host;
do not install every wheel indiscriminately, since several are tied to a Python ABI and
Linux architecture. A host-side installation outline is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --find-links wheelhouse \
  swegemma==0.2.7 adk-submission==0.2.11 adk-eval-core==0.1.0 \
  google-adk==1.36.1 vllm==0.19.1
swegemma eval --help
```

Here `--find-links` adds the downloaded wheels as package candidates; it does not disable
the package index. Initial host setup may require internet access. This is separate from
the offline repository environment used during an evaluated task. Inspect the resolved
versions with `python -m pip freeze` and retain them with the run.

Before loading the model, check the directory and root configuration with the released
package APIs. Save this as `check_agent.py` and run `python check_agent.py`:

```python
from pathlib import Path
from adk_submission import validate_directory
from adk_submission.schema import SandboxedAgentConfig
from adk_submission.yaml_loader import load_yaml
from swegemma.config import build_submission_limits
from swegemma.models import validate_single_declared_model

root = Path("agent").resolve()
limits, _ = build_submission_limits()
layout = validate_directory(root, limits)
SandboxedAgentConfig.model_validate(
    load_yaml(layout.config_path, layout.root_dir, limits=limits)
)
assert validate_single_declared_model(root) == "gemma-4-31b-it-qat-w4a16-ct"
print("Directory and root schema checks passed.")
```

This configuration check needs the installed host dependencies but makes no model request. Fix
schema errors at this stage. The subsequent evaluator run exercises registry compilation and
actual tool execution.

### 10.3 Prepare the repository sandbox and check the controls

The repository sandbox is prepared separately. `Dockerfile.public` expects `imp.py`,
`telnetlib.py`, and `wheels/` in its build context. With the data layout above, construct
that context explicitly:

```bash
mkdir -p sandbox-build/wheels
cp data/docker/Dockerfile.public sandbox-build/Dockerfile
cp data/docker/imp.py data/docker/telnetlib.py sandbox-build/
cp -R data/wheels/. sandbox-build/wheels/
docker build -t swebench-sandbox:latest sandbox-build
```

Building the image uses the network to prepare packages. Evaluated Docker tasks then run
with networking disabled. Do not confuse an online image build with permission for the
agent to install arbitrary packages during an issue.

Once the host dependencies, task data, and Docker image are ready, the verifier can run without a
model server. Both controls below use the same practice task and command timeout through the
`swegemma 0.2.7` verification path. They check environment reconstruction and grading, rather than
the agent's repair ability.

<details markdown="1">
<summary>Copyable controls: verify the unchanged repository and the reference patch without inference</summary>

For the unchanged repository, `--skip-agent-patch` bypasses model execution and proceeds
to verification with an empty agent patch:

```bash
swegemma eval \
  --tasks data/tasks.jsonl \
  --snapshots-dir data/snapshots \
  --submission-dir agent \
  --results-dir results/control-unchanged-001 \
  --sandbox docker \
  --image swebench-sandbox:latest \
  --task-id fastapi_11194 \
  --skip-agent-patch \
  --timeout-seconds 300 \
  --concurrency 1 \
  --display quiet
```

The CLI still requires the submission-directory argument and constructs its model registry,
but this route makes no inference request. The expected diagnostic result is an unresolved
task because the reported defect remains.

For the reference repair, save this as `check_reference.py` **beside** `agent/`, not inside
it. The script calls the released verifier directly and keeps reference data on the
evaluation side:

```python
import asyncio
import json
import time
from pathlib import Path

from adk_submission import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks
from swegemma.results import append_task_result
from swegemma.sandbox import ContainerConfig, ContainerManager

task_id = "fastapi_11194"
cfg = EvalConfig(
    tasks_path=Path("data/tasks.jsonl"),
    snapshots_dir=Path("data/snapshots"),
    submission_dir=Path("agent"),  # Required field; no agent is run.
    results_dir=Path("results/control-reference-001"),
    models=ModelRegistry(),       # Empty: no inference is needed.
    timeout_seconds=300,
)
task = next(t for t in load_tasks(cfg.tasks_path)
            if t.instance_id == task_id)
assert task.patch.strip() and task.test_patch.strip()

snapshot, base_snapshot, snapshot_patch = resolve_task_snapshot_paths(
    cfg.snapshots_dir, task.instance_id, task.repo
)
assert snapshot.is_file()

sandbox = ContainerManager(ContainerConfig(
    image=cfg.image,
    timeout_seconds=cfg.harness.command_timeout_seconds,
))
try:
    result = asyncio.run(verify_task(
        sandbox, cfg, task, snapshot,
        base_snapshot_path=base_snapshot,
        patch_path=snapshot_patch,
        agent_patch=task.patch,
        start_time=time.perf_counter(),
    ))
    append_task_result(result, cfg.results_dir, running_results=[result])
    print(json.dumps({
        "control": "reference",
        "instance_id": result.instance_id,
        "resolved": result.resolved,
        "test_exit_code": result.test_exit_code,
        "error": result.error,
    }, indent=2))
finally:
    sandbox.close()
```

```bash
python check_reference.py
```

The expected positive-control result is `resolved: true`. Inspect the test log if either
control behaves unexpectedly. A 300-second command timeout is a starting setting and may
itself cause a failure; if it changes, repeat both controls under the same new conditions.
The agent's `max_time_minutes` setting does not limit the entire verifier process.

This uses an internal verifier API from the pinned version, so recheck its signature when
updating the harness. It tests the local reconstruction and verification path, not the
entire hosted scorer. Reference patches and this script never belong in the submitted
agent or in an evaluation prompt.

</details>

### 10.4 Start the model server

The model server exposes an API: the agent sends messages and receives generated text or
tool-call requests. The [vLLM project][vllm-code] supports an OpenAI-compatible interface;
that describes the request format, not the company or location performing the inference.
In the local setup below, the endpoint serves Gemma on the GPU host. A tool call returned
by that server is then executed by the harness in the repository sandbox, and its output
becomes a later model observation.

The released CLI connects to a model endpoint; it does not start the server or download the base
model. On the [official model page][model], select the permitted variation, fulfill its access
requirements, and use **Download** to obtain the files. Extract them under a directory such as
`models/` and locate `config.json` with the model weights. That directory is the model path below.
On September 25, the page listed Version 2 and 23.3 GB of files, including `model.safetensors`.
Record the downloaded version as well as the variation name.

On a compatible four-GPU host, the following is a serving template matching the documented
model name, parsers, tensor parallelism, and context length:

```bash
# Replace this with the directory containing the permitted model's config and weights.
GEMMA_MODEL_DIR=/absolute/path/to/gemma-4-31b-it-qat-w4a16-ct

vllm serve "$GEMMA_MODEL_DIR" \
  --host 127.0.0.1 --port 8000 \
  --served-model-name gemma-4-31b-it-qat-w4a16-ct \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```

Run this in its own terminal and wait for startup to complete. Open a second terminal,
return to the same project directory on the GPU host, and activate the environment:

```bash
cd /absolute/path/to/gemma4-start
source .venv/bin/activate
curl --fail http://127.0.0.1:8000/v1/models
```

Replace the `cd` path with the project directory created earlier. The response should contain
`gemma-4-31b-it-qat-w4a16-ct`, matching `agent.yaml`.
A process that is still loading weights is not ready for an agent run. LoRA serving needs
additional adapter registration when adapters are introduced; this template is for the
no-adapter baseline only.

### 10.5 Run one task with explicit limits

Use the second terminal in the project directory with its host environment active. Pin the
local endpoint explicitly: the registry gives `MODEL_PROXY_URL` precedence and also loads
`.env` files. Exporting the intended values prevents a saved setting from silently selecting
another model server:

```bash
export MODEL_PROXY_URL=http://127.0.0.1:8000/v1
export MODEL_PROXY_API_KEY=EMPTY

swegemma eval \
  --tasks data/tasks.jsonl \
  --snapshots-dir data/snapshots \
  --submission-dir agent \
  --results-dir results/baseline-smoke-001 \
  --sandbox docker \
  --task-id fastapi_11194 \
  --timeout-seconds 300 \
  --max-tool-calls 40 \
  --max-time-minutes 3 \
  --max-turns 80 \
  --concurrency 1 \
  --display single
```

`fastapi_11194` is an example task named in the public data documentation. Confirm it is
present in your downloaded task file, or replace it with an `instance_id` from your
inventory. The local CLI flags intentionally repeat the hosted YAML limits so that the two
configurations do not diverge unnoticed.

Keep the first run small. It should establish that the agent loads, the intended model
responds with usable tool calls, the repository environment initializes, a patch can be
extracted, and verification produces a result. A failed task can still be an informative
smoke test if all those stages execute and the failure is visible.

<details markdown="1">
<summary>Version note: the local evaluator and hosted scorer</summary>

The guide and the public wheel do not expose exactly the same surface. In particular, the
`swegemma 0.2.7` wheel inspected here does not contain the hosted `swegemma.metric` module
described in the README. The local CLI is a development evaluator; it is not a complete
copy of Kaggle's scoring integration. Record this version boundary when diagnosing a
difference between local and hosted behavior.

</details>

## 11. Read the result before changing the agent

The harness writes results, patches, verification output, and traces beneath the chosen
results directory. The guide describes `summary.json`, `task_results.jsonl`, `patches/`,
`test_outputs/`, and `traces/`. Start with the actual files created by your installed
version, then follow one task from its recorded outcome back through its patch and trace.

Read the result in this order:

1. **Initialization:** Did the intended model and repository environment start?
2. **Localization:** Which implementation did the agent inspect, and why?
3. **Change:** Does the extracted patch address the issue that was supplied?
4. **Verification:** Did the patch apply, and which check passed or failed?
5. **Resources:** How much setup time, agent time, and tool work did that attempt consume?

That order keeps environment problems from being misdiagnosed as model weakness. It also
avoids the opposite mistake: a clean program exit does not establish that the patch was right.

| Failure seen in the run | First thing to inspect |
|---|---|
| Connection refused or unknown model | Whether the server is ready, the endpoint is correct, and the served model name matches. |
| Repository setup failed | Snapshot path, offline wheels, image build context, and dependency logs. |
| Repeated tool errors | Tool arguments and whether the agent adapts after an error. |
| No patch | Whether edits reached disk, patch extraction succeeded, or the session ended prematurely. |
| Patch does not apply | The actual diff and the baseline used for generation. |
| Tests fail after a valid patch | The implementation hypothesis, missed cases, and regressions. |
| Task consumes its entire budget | Search scope, repeated reasoning, oversized outputs, and test duration. |
{: .quick-summary }

For a local panel, save the task identifier, repository, resolved status, error category,
setup and agent times where available, token and tool usage, patch path, and verification
log path. Preserve the trace even when a run fails: the failed attempt is usually where
the next experiment comes from.

The smoke test is ready to give way to an agent comparison when the intended model served
the request, the task environment was prepared, patch extraction and verification both
completed, and the outcome and timings were retained. The resolved count can still be zero.
If one of those stages did not run, repair that stage before interpreting a prompt change.

### 11.1 Read a failed attempt as a chain of decisions

Return to the invented zero-limit issue. Suppose an agent searches for `limit`, reads
`routes.py`, adds a zero special case there, runs a test that only uses five, and submits.
The verifier later fails. The trace would support a specific diagnosis: it did not trace
the value into the shared helper, and its test did not distinguish the bug from ordinary
behavior. "The model is bad at Python" would be a much less useful account.

The next candidate might require the agent to identify the failing input and inspect the
shared implementation before editing a caller. Its evaluation should check whether those
instructions actually change behavior on other issues, and whether the extra reading
costs more tasks elsewhere than it recovers here.

Now imagine a different trace. The agent finds `normalize_limit`, proposes the correct
change, and sends an edit request whose matching text does not exist. The tool reports
failure. The agent then runs the old code and submits an empty diff. More domain training
may be irrelevant to this failure. The immediate question is whether clearer feedback
handling and shorter edit requests prevent the agent from reasoning about an edit that
never happened.

Finally, an otherwise sensible trace might spend its entire allowance running a broad
test suite. That raises a test-selection and time-allocation question. The same unresolved
outcome can therefore arise from three different mechanisms. A useful first report contains
examples of those mechanisms, alongside the total number of resolved tasks.

## 12. Turn the baseline into an experiment

Once one task travels through the full pipeline, the next step is a fixed development panel.
Run the baseline and a candidate on the same issues, under the same limits and environment.
For a panel of $M$ tasks, a paired difference is:

$$
\Delta=\frac{1}{M}\sum_{i=1}^{M}
\left(r_i^{\mathrm{candidate}}-r_i^{\mathrm{baseline}}\right).
$$

Retain the task-level outcomes behind that mean. A candidate might gain two tasks and lose
two different tasks, leaving the same overall score but revealing a meaningful change in
behavior. Timing matters as well: a larger local resolution rate does not help if its full
hosted run exceeds the deadline.

For example, imagine the following **invented** comparison on 12 issues:

| Outcome on the same issue | Issues | Effect on the comparison |
|---|---:|---|
| Both agents resolve it | 2 | No change. |
| Only the baseline resolves it | 2 | Two regressions, $L=2$. |
| Only the candidate resolves it | 3 | Three gains, $G=3$. |
| Neither resolves it | 5 | No change. |

Let $G$ count issues resolved only by the candidate and $L$ issues resolved only by the
baseline. The tasks where both succeed or both fail cancel in the paired difference:

$$
\Delta=\frac{G-L}{M}=\frac{3-2}{12}
\approx0.0833.
$$

The baseline resolves four and the candidate five: 33.3% versus 41.7%, a difference of
**8.33 percentage points** before rounding. Calling this simply "one more solve" hides
five changed outcomes: three gains and two regressions. Read those five traces.
Perhaps a stronger testing instruction recovers the gains but causes timeouts on the
regressions. That suggests a more selective testing policy, not a conclusion that the
new prompt is uniformly better. In a real comparison, keep the manifest fixed and
account for every task; silently dropping failed attempts changes the denominator.

<figure class="gemma-functional-figure">
<div class="gemma-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part1-fig5-paired-outcomes.png" alt="Invented paired results: both agents resolve 2 tasks, baseline only 2, candidate only 3, neither 5; totals are 4 versus 5 out of 12." width="2000" height="1200" loading="lazy">
</div>
<div class="gemma-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/gemma-4-series/part1-fig5-paired-outcomes-mobile.png" alt="Invented paired results: both agents resolve 2 tasks, baseline only 2, candidate only 3, neither 5; totals are 4 versus 5 out of 12." width="919" height="1460" loading="lazy">
</div>
<figcaption markdown="1">

Figure 5. One net gain can hide five changed outcomes. Read each cell as the two agents'
results on the same issues: the blue cell contains the candidate's three gains; the warm
cell contains its two regressions. All numbers are invented to explain a paired comparison.

</figcaption>
</figure>

Begin with questions that the panel can actually answer:

| Comparison | The question it tests |
|---|---|
| Base prompt vs. a more explicit reproduction step | Does reproducing the issue improve the final fix enough to justify its time? |
| Workspace tools vs. workspace plus graph tools | Does indexed navigation find useful code sooner, including on incomplete graphs? |
| Thinking disabled vs. enabled, with the request verified | Does explicit reasoning recover tasks after accounting for output truncation and runtime? |
| One agent vs. a read-only analyzer | Does delegated investigation improve localization after accounting for extra inference? |
| No adapter vs. a trained LoRA | Does the learned behavior improve held-out resolution under the same runtime budget? |

Use a small smoke panel to find configuration and runtime failures, then a broader fixed
panel to compare behavior. Keep a final local holdout outside prompt selection and training.
Run repeats when sampling variability could change the decision; a single stochastic
success should not become a general claim.

An **ablation** is a comparison that removes or changes one component to find out what it
contributes. Turn graph tools on while keeping the prompt, base model, task set, and budget
fixed, and the result can say something about graph access. Change all four at once, and
even a better score gives little guidance about which change helped. If the budget itself
is the experimental variable, report the quality and time tradeoff explicitly rather
than describing the conditions as identical.

{% include interactive/article-lab.html id="gemma-paired" %}

### 12.1 Follow one failure all the way to a decision

Return to the invented failed edit from the previous section. The model identified
`normalize_limit`, but its replacement text did not match the file. The tool reported the
failure; the agent then behaved as though the edit had happened. That is the observation.
Our hypothesis is narrower than "the model needs more coding knowledge": making recovery
from an unsuccessful edit explicit may prevent decisions based on code that was never saved.

For a candidate experiment, add the following instruction to `agent/prompts/system.md`:

> After an edit tool reports failure, read the affected lines again. Use the current file
> contents to make a smaller valid edit, then confirm the saved change before testing it.

Keep the base model, tools, issue list, runtime, sampling settings, and time limits unchanged. The
prompt addition is the intervention. Check whether it changes the next actions after a failed
edit; the written instruction alone does not establish that the agent recovered.

The first measurement is behavioral: when an edit fails, does the next action re-read the
file, and is the intended change eventually saved? The second is the actual objective:
does the patch resolve the issue, and how much time was spent? A model can become better at
handling an edit error while still misunderstanding the bug. We therefore keep both kinds
of observation instead of substituting the easier intermediate measure for the score.

Now imagine that this change produced the **same invented twelve-task result above**:
three gains and two regressions. If the gains show recovered edits but the losses show
repeated re-reading until timeout, we have a useful direction rather than a universal
winner. A next candidate could test a bounded recovery step. If the supposed gains do not
show the predicted recovery behavior, we should revisit the mechanism before crediting the
instruction. If setup differed between versions, first repair the comparison. These are
three different decisions supported by three different kinds of evidence.

A promising development comparison earns a test on issues kept out of that design process.
It does not yet establish hidden-set performance. The research question has now become
concrete: **can this recovery policy turn a particular class of failure into more verified
repairs without spending away the gains?** The same pattern applies later to a graph tool,
an extra agent or a trained adapter.

### 12.2 Split by the dependency that could leak

Two tasks can share a repository snapshot. The downloaded task file has 127 distinct
`(repo, base_commit)` groups: `rich_3882` and `rich_3894` share one group, and
`requests_6589` and `requests_6629` share another. Randomly dividing rows can therefore place
related code states on both sides of a split. Keep these groups together, and inspect closely
related issues before treating them as independent evidence.

Repository holdout asks a stronger question: can an approach developed on some projects
work on another project? That is relevant because the hidden repositories differ from the
public ones. However, HTTPX has only one task in this release. Its result is useful as an
individual case, but cannot support a stable repository-level estimate. Report counts
alongside rates rather than averaging four repository percentages as if they were equally
informative.

Use both task-weighted resolution rate and repository-level breakdowns. The first reflects
the panel as sampled; the second reveals whether a result comes almost entirely from one
codebase. Neither makes the public data identical to the hidden population.

## 13. Where improvements could come from

The baseline creates several places to intervene. The useful question is which intervention
changes a task outcome or the time needed to obtain it.

**Localization.** Start from concrete issue terms, relevant tests, stack traces when supplied,
and recognizable API names. Add graph traversal when an indexed symbol exists. When it
does not, continue through source search. A retrieval method should be judged by the code it
helps the agent reach and the fixes that follow.

**Tool discipline.** Short edits are less likely to be truncated than a rewrite of a whole
module. A failed command should change the next action. After editing, read or inspect the
diff so subsequent reasoning uses the code that is actually on disk.

**Testing and stopping.** Run the checks that discriminate between the current hypothesis
and its alternatives. Repeating an unchanged passing test contributes little. Once the
implementation and relevant behavior are checked, reserve enough time to clean up and call
`submit_patch()` explicitly. The harness's fallback recovery is useful, but deliberate
completion makes the trajectory easier to understand.

### 13.1 Skills make a procedure reusable

A tool exposes an operation. A **skill** packages reusable instructions for when and how to
perform a task, optionally with resources and helper scripts. A repository-navigation skill,
for example, might explain how to inspect package metadata, locate public entry points, and
summarize a call path. A helper script could extract a compact symbol inventory.

In the competition format, each skill is a directory with a `SKILL.md` manifest containing
YAML front matter such as `name: repo_navigation`. The agent configuration names the skill
directory. Resources are read through the skill interface, and permitted scripts execute
inside the task sandbox through `run_skill_script`, using the same task budget and workspace
as command execution. They do not add unrestricted Python execution in the host process.
[Skill contract][data]

The released submission limits allow `.py` helper scripts; they do not list `.sh` as an
accepted archive extension, even though the overview discusses shell execution. For a first
skill, use the supported Python-script path and validate its package. Add a skill because
its procedure improves an observed task, then test whether the model actually invokes it.

### 13.2 A second agent needs a specific job

The configuration language supports sequential, parallel, and looping workflows, as well
as a model-backed tool called an `AgentTool`. A useful first role for a second agent is
read-only investigation: identify likely files, explain a call path, and return a concise
finding to the agent that edits.

An `AgentTool` with `skip_summarization: true` is illustrated in the organizer's `sample_submission`. It can keep
intermediate exploration out of the main agent's conversation history. It also introduces
another model invocation and another chance to misunderstand the task. Restrict its tools
to its intended role, use the same permitted base model, and compare it against the one-agent
control before increasing the hierarchy.

The availability of `ParallelAgent` does not change the host's statement that hidden tasks
are evaluated sequentially. Workflow concurrency within one task and concurrency across
benchmark tasks are separate concepts.

### 13.3 LoRA changes the model's behavior

Low-rank adaptation, or LoRA, trains a compact update to selected model weight matrices.
Consider one frozen matrix $W$ with $d_{\mathrm{out}}$ rows and $d_{\mathrm{in}}$ columns.
Instead of learning a separate adjustment for every entry, LoRA learns two thinner matrices:

$$
\begin{aligned}
W'&=W+\gamma BA,\\
B&\in\mathbb{R}^{d_{\mathrm{out}}\times r},\qquad
A\in\mathbb{R}^{r\times d_{\mathrm{in}}}.
\end{aligned}
$$

The shared inner dimension $r$ is the **rank setting**. Multiplying the two thin matrices
reconstructs an update with the same shape as $W$, but with rank at most $r$. The scalar
$\gamma$ sets its strength; standard LoRA commonly uses $\gamma=\alpha/r$. During this
form of adapter training, the original $W$ stays frozen while $A$ and $B$ are learned.
[Original LoRA paper][lora-paper]

Counting entries shows why this can be compact:

$$
\begin{aligned}
N_{\mathrm{full}}&=d_{\mathrm{out}}d_{\mathrm{in}},\\
N_{\mathrm{LoRA}}&=r(d_{\mathrm{out}}+d_{\mathrm{in}}),\\
\frac{N_{\mathrm{LoRA}}}{N_{\mathrm{full}}}
&=r\left(\frac{1}{d_{\mathrm{out}}}+\frac{1}{d_{\mathrm{in}}}\right).
\end{aligned}
$$

For an **illustrative** $4096\times4096$ matrix and $r=16$, the arithmetic is:

| What is learned | Shape or calculation | Trainable parameters |
|---|---|---:|
| A full update to the matrix | $4096\times4096$ | 16,777,216 |
| LoRA matrix $A$ | $16\times4096$ | 65,536 |
| LoRA matrix $B$ | $4096\times16$ | 65,536 |
| Both LoRA matrices | $16(4096+4096)$ | **131,072**, about **0.78%** of a full update. |

This example counts parameters for one hypothetical matrix. It is not a Gemma adapter-size
or GPU-memory estimate. Targeted layers, storage precision, metadata, activations, and
optimizer state affect actual resource requirements. The adapter stores the learned update
rather than a full copy of the base model. In this competition, adapters are optional, and
different agents may use different adapters while sharing the permitted base model.

The documented server limits include at most eight loaded LoRAs and rank at most 128. The
entire unpacked submission, including every adapter, must still fit below 3 GiB. An adapter
directory contains `adapter_config.json` and `adapter_model.safetensors`, and an agent names
it using the `adapter` field. [Adapter contract][data]

Training needs a target behavior and a source of feedback. Supervised fine-tuning can teach
verified sequences of investigation, tool calls, and edits. Reinforcement learning can use
executable outcomes as a reward, but also requires producing and checking trajectories. A reward
for short episodes can encourage quitting; a reward for apparent success can reinforce a broken
verification process. The evaluator and the examples therefore matter as much as the optimization
method.

LoRA, supervised fine-tuning, and reinforcement learning describe different choices, so they can
be combined.

| Choice | Question it answers |
|---|---|
| LoRA | Which compact weight update is learned while the base weights stay frozen? |
| Supervised fine-tuning (SFT) | Which demonstrated responses or actions should the model learn to produce? |
| Reinforcement learning (RL) | Which outcomes receive reward, and how should the policy change to obtain them? |
{: .quick-summary }

For example, SFT can train a LoRA adapter from verified recovery episodes. LoRA determines the
form of the parameter update; the supervised objective supplies its learning signal. [Original
LoRA paper][lora-paper]

In supervised fine-tuning, an example can pair a situation with a desirable next action.
For a coding agent, that situation includes the issue and previous observations. A useful
episode may demonstrate that after a command reports "file not found," the next action
inspects the directory instead of pretending the command succeeded. The training teaches
the model to make particular decisions in particular contexts, not merely to produce the
sentence "always verify your work."

This makes the choice of training examples important. If every episode begins with the
correct file already identified, it cannot demonstrate how to find that file. If every
episode succeeds without tool errors, it offers little evidence about recovering from
them. If long, repetitive episodes dominate, the model may learn an expensive procedure.
Before collecting thousands of examples, decide which recurring failure the data should
teach the model to overcome.

*Training Software Engineering Agents and Verifiers with SWE-Gym* (2024 preprint, ICML
2025) is a relevant research example. It provides executable repository tasks and uses
agent-environment trajectories to train software engineering agents and verifiers. Its
contribution makes a key requirement tangible: training an agent needs environments where
actions have observable consequences and resulting repairs can be checked. The released
[paper][swegym-paper] and [GitHub repository][swegym-code] are useful reading before planning
trajectory collection. Their models, task distributions, and reported gains do not predict
a Gemma competition result, and their data should not be admitted into a training plan
without checking overlap and the applicable terms.

Reinforcement learning adds another difficulty. The final pass/fail reward may arrive
after many searches, reads, and edits. Which decisions deserve credit for the success,
and which merely consumed time? This is the **credit-assignment problem**. A reward based
on a weak intermediate signal can train the wrong behavior: a model rewarded for finishing
quickly could stop early, while one rewarded for a test process exiting cleanly could
benefit from a run that executed no useful tests.

A sensible training pilot therefore needs a trustworthy evaluator and a specific failure
pattern. For example: does a small set of verified recovery episodes reduce repeated
invalid edit calls on held-out issues? Measure the resulting repair rate and runtime, not
only the training loss. **Training loss** describes how well the model fits its training
objective; it does not directly measure whether new repository issues are resolved.

Reference patches tell us what changed. They do not directly contain the reasoning and tool
trajectory that discovered the change. Creating useful training episodes requires a method
for producing those steps and verifying the resulting patch. Keep failed attempts as well
as successful ones when diagnosing what the training signal teaches.

The baseline's recurring failures should guide adapter rank, training data, and optimization
choices. A valid adapter and a completed training run are intermediate artifacts; held-out patch
resolution remains the test.

### 13.4 External teachers are an unresolved rule question

The [rules][rules] permit external data subject to accessibility and other conditions. A
[discussion][distillation-discussion] asks whether proprietary API models may generate patches and
trajectories for distillation, and how their terms interact with the winner's release obligations.
In the host reply checked on September 25, an answer was still being prepared.

That leaves a concrete dependency for a teacher-based training plan. Obtain the host's
clarification, check the chosen data and provider terms, and establish what can be released
before investing in that dataset. This does not prevent building and evaluating the
no-adapter agent now.

## 14. Research paths suggested by the opening conditions

The opening conditions on September 25 suggest three plausible research paths: recover failures in
the agent's procedure, train a recurring missing skill, or improve transfer beyond familiar
repositories. These are conditional forecasts based on the fixed model, executable tasks, small
development set, and total time budget.

### 14.1 The model knows more than its current procedure lets it use

In this scenario, many failures come from losing relevant information, carrying out an
edit incorrectly, or spending too long on an unhelpful action. Better interfaces and
procedures continue to recover issues without changing the model weights. A comparatively
small agent could remain competitive because it gives the model useful observations
and keeps the path to verification short.

The evidence would be visible in paired traces: the same model reaches the right code
more often or completes a repair it previously abandoned. The strongest support would be
recoveries across several kinds of issues, not merely shorter episodes. If two candidates
finish faster but still misunderstand the same requirements, this scenario is becoming a
less useful explanation of the remaining failures.

The practical implication is to keep the no-adapter control alive. Training should have
to outperform the best measured procedure, not just an initial prompt that was never
improved. Research on interfaces and simple agents makes this path plausible; it does
not establish where Gemma's ceiling will be.

### 14.2 The procedure becomes reliable, but a recurring skill is still missing

Here, traces show that the model consistently gets the necessary evidence and uses the
tools properly, yet repeatedly makes the wrong decision. It might misunderstand a pattern
of asynchronous control flow, fail to preserve an API contract, or struggle to recover
from a particular class of failed repair. Rephrasing the same instruction may eventually
stop helping.

This would make targeted training more interesting. The test is whether examples teach
that missing behavior on fresh issues, under the deployment format and runtime that will
actually be used. A lower training loss, more polished explanations, or a successful
adapter load would not establish that result. A useful gain would persist when the answer
location and reference patch are unavailable to the agent.

A specialized analysis worker is another hypothesis in this scenario. It could give a
hard subproblem a cleaner context, but it could also repeat the same misunderstanding
and consume additional inference. Its value needs to appear after accounting for that
extra work. More workers do not create more time in the global budget.

### 14.3 Local progress is real, but it does not transfer cleanly

In this scenario, both prompt work and training improve development results, yet the
improvement is concentrated in familiar repositories or issue patterns. A hidden task
uses different conventions, and the policy's shortcuts fail. A small Public score may
also move enough to encourage choices that do not survive the final Private evaluation.

The warning signs would include gains clustered in the dominant development repositories,
regressions on repository holdouts, and a large gap between directly rehearsed tasks and
issues reserved before tuning. No single diagnostic recreates the hidden population,
but these observations would argue for broadening the evidence before increasing model
or workflow complexity.

Final selection would then favor a balance between observed repair rate, transfer evidence,
and the ability to finish the whole workload. A candidate that solves more isolated issues
but exceeds twelve hours is not a usable submission. A fast candidate that repeatedly
stops before a promising repair is complete may be leaving quality on the table.

### 14.4 What could change the plan early

The host's runtime clarification, investigation of graph issues, and pending answer on
external-model distillation are concrete events to watch. A timeout change could alter
how unfinished work is scored. A data update could change which graph comparisons are
meaningful. A teacher-model ruling could admit or rule out a proposed source of training
trajectories. [Runtime discussion][runtime-discussion] · [Graph discussion][graph-discussion]
· [Distillation discussion][distillation-discussion]

Versioned agents, fixed task sets, and recorded runtimes preserve the comparison when host
conditions change. Retain the old evidence, identify which part of the experiment an update
affects, and test that implication before crediting the resulting score change to the agent.

## 15. Submit an artifact you can identify

After local validation, rebuild `submission.zip` from the exact agent directory used in the
chosen experiment. Save its hash, configuration, model and adapter identities, package
versions, task panel, and measured results together. A notebook that constructs the archive
is useful, but the archive is the artifact whose contents must be known.

The submission dialog inspected on September 25 offers **File Upload**, **Notebook**, and
**MCP** tabs. The direct-file route is:

1. Open the competition page and choose **Submit Prediction**.
2. Select **File Upload**, then **Browse Files**.
3. Choose the archive named **`submission.zip`**.
4. Enter a description that identifies the agent version and complete the submission flow.

The ZIP contains the agent package. Kaggle runs it on hidden tasks to generate and score patches.
The evaluated artifact is therefore the packaged procedure, rather than a table of fixes to public
development tasks.

Submitting consumes a submission opportunity and queues the hosted evaluation. Retain the eventual
error or score alongside the submitted ZIP's identity. The rules checked on September 25 allow at
most **one submission per day** and up to **two final submissions**. [Submission limits][rules]

A hosted run can reveal compatibility or population differences that a local panel missed.
Use that result to form a specific question, then investigate it locally. Repeatedly changing
several components to chase a small Public movement makes it difficult to learn what actually
improved.

## 16. Dates, research track, and the first milestone

The main competition opened on September 23, 2026. Its published deadlines are at 23:59 UTC
on the dates below; Korea is nine hours ahead:

| Milestone | UTC | Korea Standard Time |
|---|---|---|
| Optional paper submission | November 12, 2026, 23:59 | November 13, 2026, 08:59 |
| Main entry and team merger | November 25, 2026, 23:59 | November 26, 2026, 08:59 |
| Main final submission | December 2, 2026, 23:59 | December 3, 2026, 08:59 |

The [main competition][competition] awards USD 65,000 across the top three places: USD 37,000,
USD 18,000, and USD 10,000. Teams may have up to five members. The [separate Paper Track][paper]
has USD 35,000 in awards and accepts a research writeup of up to 3,000 words. Its criteria
include novelty, quality, relevance, verifiability, and clarity; participation in the main
prediction competition is not required. The organizers may update the schedule.

The winner rules also require reproducible code and documentation, with the stated
open-source licensing obligations. Recording training data provenance, dependencies,
environment details, and evaluation decisions from the beginning makes a later research
report and release substantially easier. [Official rules][rules]

The Paper Track gives the project a second possible contribution. A useful research
result could explain when an existing-symbol graph helps navigation, which observation
summaries preserve useful debugging information, or how a testing policy trades additional
checks against repair time. The organizers also invite work on tuning, code comprehension,
tasks and benchmarks, and graph reasoning. A focused, reproducible finding can be worth
reporting even when it is not a claim to the highest leaderboard score.

The paper deadline arrives before the main competition deadline, so a research writeup needs a
bounded question and evidence collected early enough to analyze. One well-supported comparison can
explain its failures and limitations; an uncertain evaluation pipeline cannot support a general
claim about agent quality.

### 16.1 A realistic first week is a sequence of answers

The following is a proposed progression, not a schedule that assumes uninterrupted GPU
access. Advance when the current question has an answer; do not launch the next stage
simply because a day has passed.

| Work session | Question to answer | Concrete result to keep |
|---|---|---|
| Understand the contract and inventory | What does the agent receive, what must it produce, and which files are usable? | A task inventory, recorded source versions, and explicit information boundaries. |
| Explore selected development issues | What observations locate the behavior that needs changing? | Short manual investigation notes, with the tasks marked as used for development. |
| Check the environment and controls | Does the verifier distinguish an unchanged repository from its reference repair? | Patch-application and test logs for the selected controls. |
| Run the minimal agent | Can the fixed model complete the tool/edit/patch/verification path? | Full traces, patches, resolved outcomes, and setup-inclusive times. |
| Read every failure in the small panel | Is the bottleneck localization, understanding, editing, testing, or time? | A failure classification supported by specific trace events. |
| Change one thing | Does a targeted intervention recover more issues than it loses? | A paired comparison with gains, regressions, and measured runtime. |

Before requesting a substantial compute run, estimate its duration from the small pilot
on the intended hardware. Decide the spending limit, what failure would stop the run,
and how completed task results will be preserved if it is interrupted. A configuration
that repeatedly fails before the first useful action should be repaired in a small run;
repeating that failure across the full panel does not measure reasoning quality.

The first milestone is an inspectable baseline: an identifiable package evaluated end to end on a
small fixed panel, with per-task traces, patches, test logs, and timings. The next milestone is a
controlled change that improves outcomes on tasks held out from its design.

The first empirical question is therefore specific: **where does the permitted model lose
tasks in this environment, and which change recovers those tasks within the same time
budget?** That question gives the next experiment a purpose.

## Reading guide and source notes

You do not need to read every paper before trying the baseline. Choose the next source
according to the question you are trying to answer. For a first pass through a paper,
read its task definition, system diagram, and evaluation conditions before concentrating
on the headline result. On GitHub, start with the control loop and one recorded example
of an action and observation; that often explains more than the feature list.

| Your next question | Suggested primary reading | What to look for |
|---|---|---|
| How did code evaluation become repository evaluation? | [HumanEval/Codex paper][humaneval-paper] (2021), then [SWE-bench][swebench-paper] (2023/ICLR 2024). | What the system receives, what it must generate, and what execution actually checks. |
| What makes a model into an agent? | [ReAct][react-paper] (2022/ICLR 2023), then [mini-SWE-agent's code][mini-code] (project released 2025). | The next-action loop and how an observation changes subsequent input. |
| Does a more complicated agent necessarily work better? | [SWE-agent][sweagent-paper] and [Agentless][agentless-paper] (2024). | Which interface or workflow components were changed in a comparison. |
| How are model decisions connected to a runtime? | [OpenHands][openhands-paper] (2024/ICLR 2025) and its [repository][openhands-code]. | Boundaries among actions, observations, state, and execution. |
| What would adapter training require? | [LoRA][lora-paper] (2021), followed by [SWE-Gym][swegym-paper] (2024/ICML 2025) and its [code][swegym-code]. | Parameter updates versus training signals, and why executable training episodes matter. |
| What does "local Gemma" mean in practice? | Google's [launch article][gemma-launch], [model card][gemma-card], and [AI Edge article][gemma-edge] (2026). | The particular model size, format, hardware, and demonstrated workload. |
| Why does serving need more memory than the weights? | [PagedAttention][pagedattention-paper] (2023) and [vLLM][vllm-code]. | Persistent model weights versus sequence-dependent attention state. |
| How does benchmark success relate to developer productivity? | METR's [2025 study][metr-2025] together with its [2026 update][metr-2026]. | Who was studied, how time was measured, and which conclusions the design supports. |

These sources explain the background. For implementation and eligibility, the following
competition sources govern the submission:

- [Competition overview, evaluation, model rule, and timeline][competition].
- [Development data, starter files, and `HARNESS_README.md`][data].
- [Competition rules: submissions, teams, external resources, and winner obligations][rules].
- [Official Gemma 4 model variant][model].
- [Official evaluation wheelhouse][wheelhouse]. Source versions checked on September 25:
  `swegemma 0.2.7`, `adk-submission 0.2.11`, and `adk-eval-core 0.1.0`.
- [Host clarification on sequential tasks, evaluation fields, and the global timeout][runtime-discussion].
- [Graph and embedding issue report, with host acknowledgment][graph-discussion].
- [External-model distillation question and pending host response][distillation-discussion].
- [Host announcement of the local evaluation wheelhouse][harness-discussion].
- [Google ADK Agent Config documentation][adk]. The competition uses a restricted compiler;
  generic ADK examples are not automatically valid competition submissions.
- [Official Kaggle CLI documentation][kaggle-cli].
- [Gemma 4 Developer Agent Paper Track][paper].

[competition]: https://www.kaggle.com/competitions/gemma-4-developer-agent
[data]: https://www.kaggle.com/competitions/gemma-4-developer-agent/data
[rules]: https://www.kaggle.com/competitions/gemma-4-developer-agent/rules
[model]: https://www.kaggle.com/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct
[wheelhouse]: https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse/data
[runtime-discussion]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743063
[graph-discussion]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742911
[distillation-discussion]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807
[harness-discussion]: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742882
[adk]: https://adk.dev/agents/config/
[kaggle-cli]: https://github.com/Kaggle/kaggle-cli
[paper]: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper
[humaneval-paper]: https://arxiv.org/abs/2107.03374
[humaneval-code]: https://github.com/openai/human-eval
[react-paper]: https://arxiv.org/abs/2210.03629
[react-project]: https://react-lm.github.io/
[swebench-paper]: https://arxiv.org/abs/2310.06770
[swebench-code]: https://github.com/SWE-bench/SWE-bench
[sweagent-paper]: https://arxiv.org/abs/2405.15793
[sweagent-code]: https://github.com/SWE-agent/SWE-agent
[openhands-paper]: https://arxiv.org/abs/2407.16741
[openhands-code]: https://github.com/All-Hands-AI/OpenHands
[mini-docs]: https://mini-swe-agent.com/latest/
[mini-code]: https://github.com/SWE-agent/mini-swe-agent
[agentless-paper]: https://arxiv.org/abs/2407.01489
[agentless-code]: https://github.com/OpenAutoCoder/Agentless
[gemma-launch]: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
[gemma-card]: https://ai.google.dev/gemma/docs/core/model_card_4
[gemma-edge]: https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/
[lora-paper]: https://arxiv.org/abs/2106.09685
[pagedattention-paper]: https://arxiv.org/abs/2309.06180
[vllm-code]: https://github.com/vllm-project/vllm
[swegym-paper]: https://arxiv.org/abs/2412.21139
[swegym-code]: https://github.com/SWE-Gym/SWE-Gym
[metr-2025]: https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/
[metr-2026]: https://metr.org/blog/2026-02-24-uplift-update/
[fastapi-docs]: https://fastapi.tiangolo.com/
[rich-code]: https://github.com/Textualize/rich
[requests-docs]: https://requests.readthedocs.io/en/latest/
[httpx-docs]: https://www.python-httpx.org/
[fastapi-example]: https://github.com/fastapi/fastapi/pull/11194

[vllm-context]: https://docs.vllm.ai/en/v0.19.1/api/vllm/entrypoints/openai/engine/serving/
