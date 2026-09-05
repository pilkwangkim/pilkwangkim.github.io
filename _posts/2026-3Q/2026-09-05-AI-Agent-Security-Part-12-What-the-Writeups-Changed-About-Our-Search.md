---
title: "AI Agent Security (Part 12): What the Write-ups Changed About Our Search"
date: 2026-09-05 09:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, ai-agent-security, competition-retrospective, experimental-design, evaluation, throughput, transfer]
math: true
pin: false
hide: false
published: true
image:
  path: /assets/img/posts/2026-09-05-ai-agent-security-part-12/cover.png
  alt: "Five published solutions examined together after the competition"
---

<style>
.content .table-wrapper > table {
  table-layout: fixed;
  width: 100%;
  min-width: 36rem;
}
.content .table-wrapper > table th,
.content .table-wrapper > table td {
  white-space: normal;
  overflow-wrap: break-word;
  vertical-align: top;
}
</style>

I wrote [Part 11]({{ site.baseurl }}/posts/AI-Agent-Security-Part-11-When-the-Mechanism-Did-Not-Transfer/) just after the final leaderboard appeared. Our team had fallen from 8th on the public board to 115th overall, finishing with a silver medal. Several direct HTTP configurations that had raised our public score earned zero on private evaluation; the mail family still earned points. My conclusion then was simple: improving throughput pays off only if the method survives the final evaluation.

The write-ups published since then have made me look further back than our final selection. We already had a pure mail submission that scored **26.010** privately. Had we given it enough development time? What evidence might have persuaded us to shift more effort toward it before the deadline?

I read the **1st-, 4th-, 5th-, 6th-, and 11th-place write-ups published on September 3–5, 2026** with those questions in mind. The five participants approached the competition differently: how they interpreted clues about private evaluation, reduced each model's execution costs, checked their results, and chose their final two submissions. Their research histories explain choices that the rankings alone cannot.

The competition was an authorized offline benchmark using synthetic fixtures and simulated tools. The scores and experiments discussed here are reported in the linked write-ups and in [Part 11]({{ site.baseurl }}/posts/AI-Agent-Security-Part-11-When-the-Mechanism-Did-Not-Transfer/).

Earlier articles:

- [Part 1: The Replay Benchmark and Trajectory-Search EDA]({{ site.baseurl }}/posts/AI-Agent-Security-Part-1-The-Replay-Benchmark-and-Trajectory-Search-EDA/)
- [Part 2: The Linear Score Law, the Replay Ceiling, and What Survives the Private Guardrail]({{ site.baseurl }}/posts/AI-Agent-Security-Part-2-The-Linear-Score-Law-and-the-Replay-Ceiling/)
- [Part 3: The v3.1.2 Reset and the Throughput Wall]({{ site.baseurl }}/posts/AI-Agent-Security-Part-3-Multi-Predicate-Stacking-and-the-Broken-Ceiling/)
- [Part 4: Past the Framing Plateau]({{ site.baseurl }}/posts/AI-Agent-Security-Part-4-Past-the-Framing-Plateau/)
- [Part 5: The Raw Wall from the Inside]({{ site.baseurl }}/posts/AI-Agent-Security-Part-5-The-Raw-Wall-from-the-Inside/)
- [Part 6: Throughput as an Experimental System]({{ site.baseurl }}/posts/AI-Agent-Security-Part-6-Throughput-Working-Note-Established-Closed-Open/)
- [Part 7: The Transfer Game]({{ site.baseurl }}/posts/AI-Agent-Security-Part-7-The-Transfer-Game-New-Surfaces-and-Private-Safe/)
- [Part 8: The Evaluation Reset]({{ site.baseurl }}/posts/AI-Agent-Security-Part-8-The-Evaluation-Reset-and-the-Search-for-a-Discrete-Lever/)
- [Part 9: From Firing to Density]({{ site.baseurl }}/posts/AI-Agent-Security-Part-9-From-Firing-to-Density-Chains-Gates-and-the-Per-K-Frontier/)
- [Part 10: From Public Density to Private Transfer]({{ site.baseurl }}/posts/AI-Agent-Security-Part-10-From-Public-Density-to-Private-Transfer/)
- [Part 11: When the Mechanism Did Not Transfer]({{ site.baseurl }}/posts/AI-Agent-Security-Part-11-When-the-Mechanism-Did-Not-Transfer/)

Korean version: [공개된 해법으로 다시 읽는 대회]({{ site.baseurl }}/posts/AI-Agent-Security-Part-12-What-the-Writeups-Changed-About-Our-Search-KR/)

## 1. Five approaches to the same problem

In this competition, the submission algorithm returned fixed candidate messages. The evaluator then ran those messages through a model and simulated tools—a process called replay. A guardrail decided whether an action could proceed, and a separate evaluator identified scored events in the execution trace. A **hop** was one model invocation; a **score cell** grouped outcomes according to the evaluator's rules. Calling the model did not necessarily produce a scored event.

Each participant concentrated on a different part of that process.

| Account | Main question | Approach | What remained uncertain |
|---|---|---|---|
| [1st · xz][w1] | How much cost remained in a viable method? | Indirect evaluation evidence, gradient-guided search, checks on the actual quantized model | The GPT-OSS improvement worked in BF16 but did not transfer to GGUF |
| [4th · ISAKA Tsuyoshi and teammates][w4] | How reliable and efficient could a simple configuration become? | Model-specific cost measurements, small checks followed by full runs, specific human hypotheses | The local proxy was an assumption about the private defense, not its recovered implementation |
| [5th · Giovanny Rodríguez][w5] | Could methods developed on the public task help another tool family? | Follow costs through inputs, outputs, parsing, and reconstructed history | Historical timing changes did not isolate the effect of each change |
| [6th · poijio][w6] | Could a search work backward from a desired output to a useful submission? | Define the target output, then check candidates from the final submission in a fresh environment | Making one call structure mandatory narrowed the search |
| [11th · Mohammad Shadab Alam][w11] | How should measured speed and uncertain defense risks affect the final choices? | Repeated controls, checks for valid behavior, different roles for the main submission and its hedge | Hypothetical defenses and later proposals were not verified private rules or demonstrated gains |

We had studied many of these problems ourselves. Earlier articles covered output tokens, replay costs, differences between GPT and Gemma, and the limits of local testing. The 5th-place author even cites our public throughput analysis. That reference made the comparison particularly close to home. We had spent much of our effort applying those methods to our strongest public approach. **Knowing how to measure performance did not settle where to invest our research time.** [5th-place write-up][w5]

## 2. Useful clues from an unknown defense

We could not inspect the private guardrail or confirm its rules. We could still look for reasons to prefer one candidate over another.

xz compared candidate families using indirect signals from evaluation. The 4th-place team examined completion information and the duration of small submissions, then built a local proxy consistent with its interpretation. The 11th-place author also used limited evaluation evidence when choosing two mail submissions for the final slots. These observations helped the participants make decisions without revealing the complete private implementation. [1st-place write-up][w1] · [4th-place write-up][w4] · [11th-place write-up][w11]

Each source of evidence answered part of the question.

| Evidence | What it could support | What remained uncertain |
|---|---|---|
| Published source and rules | How the visible evaluation worked | The undisclosed guardrail implementation |
| Evaluation metadata | Reasons to prefer some candidate families | Runtime alone could not identify a blocking rule or each candidate's contribution |
| A local proxy guardrail | Behavior under a stated defense assumption | Whether the assumption matched private evaluation |

Runtime depends on model cost, candidate count, cache state, and where an execution fails, among other things. An indirect signal can have several explanations, so the participants' inferences were not automatically correct. Still, uncertainty about the implementation did not make all candidates equally promising.

poijio, who finished sixth, also chose the mail family partly by interpreting the published evaluation structure. That interpretation gave the author a reason to invest in the alternative without claiming to know the private defense. [6th-place write-up][w6]

I had given considerable weight to what we could not know about the private implementation. Reading these choices, I think I should have asked more often whether the evidence we did have justified spending time on an alternative. I did not need certainty to make that decision.

## 3. Compare scores from the same submission

The 11th-place author's public best of **136.450** and final private score of **30.040** came from different submissions. The public-best submission's private score is not reported. The final main submission, NB1, scored **30.055 → 30.040**. Comparing 136.450 directly with 30.040 would mix the effect of changing the submission with the effect of changing the evaluation conditions. [11th-place write-up][w11]

The following figure connects public and private results for the same submission, using the pairs reported by each author and our pure mail result from [Part 11]({{ site.baseurl }}/posts/AI-Agent-Security-Part-11-When-the-Mechanism-Did-Not-Transfer/).

![Reported public/private score pairs for the same submission. A public best is not connected to a different final submission.]({{ site.baseurl }}/assets/img/posts/2026-09-05-ai-agent-security-part-12/fig-01-paired-scores.png)
_Figure 1. Each line connects the public and private scores of one submission. The 4th-place result has no public counterpart, and the 1st-place account gives approximate scores, so both are omitted. Our pure mail submission was not selected for final scoring. Sources: [5th][w5], [6th][w6], [11th][w11], and Part 11 of this series._

The 5th-place author reports **115.240 → 0.000** for public-best submission v50 and **40.155 → 40.365** for final mail submission v57. The 6th-place author reports **147.530 → 0.000** for the public-leading submission and **37.530 → 37.690** for the submission that earned gold. In both cases, the stronger public submission failed privately. But the 5th-place author reused execution-analysis methods developed on that public approach when building the mail submission. The behavior failed to transfer; some of the knowledge gained from developing it remained useful. [5th-place write-up][w5] · [6th-place write-up][w6]

Our **25.695 → 26.010** pair shows that we, too, had a starting point that survived. The higher mail scores achieved by other participants do not tell us which single change would have closed our gap. Their candidates, model branches, execution environments, and research histories differed from ours.

## 4. First place: reducing post-tool generation

### Choosing where to focus

xz began by investigating the hidden defense. The author read agent-security papers and examined the competition repository with Codex, considering classifiers, model-based judgments, and execution provenance as possible explanations. Eventually, evaluation observations proved more useful for choosing a direction than further speculation. They persuaded xz to concentrate on direct mail, the family the author could score reliably at scale. The exact private defense remained unknown. [1st-place write-up][w1]

xz expected other participants to find the same broadly viable family. To gain an advantage, the author looked closely at the cost of a working interaction. Some of that cost came after the useful action had already finished.

### Ending the closing response immediately

In xz's single-action configuration, a successful candidate required two model calls. The first produced the tool call; the second generated a closing response after the tool returned. The score had already been recorded by then. Generating the response used evaluation time without earning more points, and that small cost recurred across many candidates.

xz wanted to preserve the valid first action and end the next generation immediately. **The second invocation remained, but its first generated token became an end-of-generation token.** The author clarified this in the comments: the improvement reduced generation within the second call. It did not remove the call itself or all of its overhead. [1st-place write-up and comments][w1]

The search used GCG, which draws on model gradients to guide changes to discrete inputs. Both stages had to work together: the first still had to produce the correct action, and the second had to end sooner. Breaking the action would defeat the purpose of saving time. Despite the short target output, the search was difficult because the models strongly preferred their usual continuation after a tool call.

### One model improved; the other did not transfer

The competition ran quantized GGUF models. For the search, xz used higher-precision BF16 checkpoints on which gradients could be calculated. An improvement on a research checkpoint therefore needed another check on the corresponding competition model.

xz started this phase with about three days left. After improving the initial inputs, the search obtained the intended closing behavior on Gemma. GPT-OSS was more frustrating: the change worked on BF16 but did not carry over to its competition GGUF. As the deadline approached, xz stopped pursuing GPT-OSS and focused on making the Gemma result more reliable. [1st-place write-up][w1]

| Model | Reported result | Use in the working submission |
|---|---|---|
| Gemma | The shorter closing behavior held on the competition model | Used the improvement and checked its stability further |
| GPT-OSS | The BF16 result did not hold on the corresponding GGUF | Kept the conventional minimum-length version |

With the deadline close, xz used the validated Gemma improvement and kept the conventional GPT-OSS version.

### From a working candidate to a reliable submission

The first working submission scored approximately **44.5**. xz suspected that variation across candidates and differences in the inference stack were still causing losses. Further comparisons showed that the runtime version and the conditions of consecutive evaluations could change the preferred output. A result that relied on a narrow gap between possible continuations was vulnerable to those differences.

After improving consistency across the candidate collection, xz reports approximately **46.5**. Several changes contributed to this stabilization, so the score increase cannot be assigned to any single intervention. [1st-place write-up][w1]

xz also proposed clearer threat models and different rewards for repeated findings in future competitions. The submission itself reduced a remaining generation cost on Gemma and made that reduction dependable enough to use. I see three separate milestones in that work—success on the research model, success on the actual model, and reliable behavior across the complete submission.

## 5. Fourth place: making a simple method reliable

### Why the team chose mail

The fourth-place team describes feeling relieved when the private result appeared. The team had confidence in its solution but had never ruled out a zero. Its final score was **41.325**. Knowing the outcome now makes the choice look easier than it was at the time. [4th-place write-up][w4]

The final approach used a simple direct mail task, without first reading secret or external material, and separate inputs for the two models. Evaluation observations and clues in the available implementation informed a local proxy for the private defense. The proxy matched those observations, although the team did not claim it matched the hidden implementation exactly. That evidence was enough to justify concentrating development on mail.

With a working interaction in hand, the team could investigate its costs in detail. The aim was to complete valid work reliably within the available time.

### The two models had different costs

For GPT-OSS, the team reduced unnecessary generated reasoning while preserving the required tool-call format. For Gemma, it examined avoidable overhead in the output representation. The Gemma change produced the team's largest reported local improvement: approximately **19–20% in raw score per second**. Raw score is the score before conversion to the final scale. That is a local result for one model, not the percentage increase in the final leaderboard score. [4th-place write-up][w4]

Repeated execution made some savings more valuable than others. According to the team, much of the common input benefited from caching, while output generation still cost substantial time. Cutting a little shared input could therefore achieve less than its size suggested. Prompt length alone did not tell the team where to spend its effort.

### Ten candidates were only the start

The team checked output correctness on roughly ten candidates, timed promising variations, and then expanded them to the full **2,000 candidates**. It also checked reproducibility across multiple GPU instances. Small runs rejected obvious failures quickly, but rare malformed outputs could still appear in the larger set and lower its score. [4th-place write-up][w4]

The team compared valid score earned per unit of time. Both quantities needed attention: a run could end sooner while also earning fewer points.

### What did not pay off

Several unsuccessful attempts helped define the final method.

| Attempt | Reported result | What the team learned |
|---|---|---|
| Eliminate the response after tool execution | Unsuccessful | This cost remained in the final method |
| Combine more actions in one interaction | Grouping earned less raw reward in the team's calculation | The same number of actions could earn different rewards depending on their grouping |
| Remove a small amount of shared input | The local gain was smaller than hosted timing variation | The team could not make a confident claim about a hosted gain |

Codex helped the team inspect the execution stack, generate variations, and summarize measurements. Specific questions about a suspected cost proved more productive than broad requests for a faster solution. The latter often produced changes too small to distinguish from measurement noise. [4th-place write-up][w4]

The team also questioned the benchmark's authorization test and its rewards for repetition, both of which influenced what competitors optimized. It proposed better intent assessment and less emphasis on repeated findings. For our own work, the development history is just as relevant: choosing a simple method left plenty to investigate. Model-specific costs, reliability at scale, and the credibility of small gains all affected whether that method was worth keeping.

## 6. Fifth place: measuring the whole execution

### Learning from the public method

Giovanny Rodríguez already had a public HTTP method that worked reliably. The immediate problem was its runtime. Although the author did not expect the family to be dependable privately, it remained a stable test case for studying execution costs. Using it to develop measurement methods was a separate decision from choosing a final submission. [5th-place write-up][w5]

Rodríguez changed only the submitted user messages; the model weights, server, parser, and evaluator stayed fixed. The write-up describes this existing stack as a “replay compiler.” A message passes through a model-specific conversation template, generated output, a parsed tool action, and reconstructed history for the next generation. Rodríguez analyzed those transformations rather than adding new software to the evaluator.

Reconstructing the history made a difference. The model did not necessarily continue from the exact short text it had just generated: the runtime could normalize the tool call before placing it back in the conversation. Reducing the first output could then affect the next input's cost in an unexpected way.

### The work continued after the scored action

Rodríguez divided total elapsed time into three parts:

$$
T = T(1) + T(2) + T(\mathrm{rest}).
$$

The first term covers model processing before the tool call, the second covers the closing stage after execution, and the last covers everything else. That residual includes runtime handling and unmeasured work. It is not a separate measurement of tool execution time alone.

| Share of total runtime in the selected comparison | GPT-OSS | Gemma |
|---|---:|---:|
| Both model-processing stages | 95.4% | 98.2% |
| Closing stage alone, included above | 27.2% | 40.3% |

The scored event was already recorded while a substantial part of the runtime still lay ahead. That made the closing stage worth investigating. These percentages describe the timing breakdown within each model's run; they do not compare hardware efficiency between the models. [5th-place write-up][w5]

Every timed batch had to retain **200 findings, 200 distinct score cells, and normal completion**. Two hundred successful tool calls were insufficient if they all mapped to one cell. A run that skipped the required action and ended early also failed the comparison. The question was how quickly each configuration completed the same valid work.

### Different models, different costs

For GPT-OSS, Rodríguez examined how three formats interacted: the conversation structure learned by the model, the output accepted by the parser, and the standard history reconstructed after execution. Requests to use less reasoning did not reliably reproduce the measured improvement. These results did not establish that hidden reasoning had been switched off.

In one unsuccessful cleanup, removing apparently redundant text slowed execution and changed a destination. The text had looked unnecessary, but removing it changed the behavior. Later comparisons covered both generations and found smaller gains that preserved the required action. The final GPT-OSS configuration averaged **84.358 seconds**, against the preceding control's **86.666 seconds**—a **2.66%** reduction. This was one comparison within a longer development history, not the whole cumulative gain. [5th-place write-up][w5]

Gemma started elsewhere. With thinking disabled, the tested template already supplied an empty thought channel. Another instruction to avoid thought could not remove a reasoning stage that was already absent from that configuration. The changes that helped concerned the tool-call output's length and consistency, along with the closing behavior afterward.

One balanced local experiment interleaved 100 candidates using a shorter representation with 100 using the standard representation. All remained valid, with mean times of **1.603 versus 1.683 seconds per candidate**. Interleaving brought the two configurations closer to the same conditions than separate runs before and after development, although the result still belonged to that local experiment. In the final retained Gemma comparison, the mean fell from the preceding control's **282.324 seconds** to **266.998 seconds**, a **5.43%** reduction. [5th-place write-up][w5]

The models responded differently to the changes. Something that helped one could hurt the other, and a shorter accepted output could still become a longer reconstructed conversation. Input length alone could not explain those effects.

### A faster first stage, but a slower full run

Across the development history, Rodríguez reports the following runtimes for 200 candidates. Every run in this comparison retained 200 valid findings and 200 scoring cells.

| Model | Initial historical run | Final selected configuration, mean | Time reduction |
|---|---:|---:|---:|
| GPT-OSS | 109.373 s | 84.358 s | 22.9% |
| Gemma | 396.973 s | 266.998 s | 32.7% |

![Initial and final retained historical runtimes for each model, as reported by the 5th-place author]({{ site.baseurl }}/assets/img/posts/2026-09-05-ai-agent-security-part-12/fig-02-historical-time.png)
_Figure 2. Each initial value comes from one historical run; each final value is the mean of two runs of the selected configuration. The comparison shows cumulative progress, without isolating any one change or comparing hardware efficiency between models. Source: [5th-place write-up][w5]._

Time reduction is `1 - later time / earlier time`; throughput increase is `earlier time / later time - 1`. The denominators differ, so the percentages differ even when both runs complete the same amount of work.

One early failure explains why Rodríguez measured both stages. In a separate Gemma experiment with a different GPU offload configuration, the first stage became faster but the closing stage grew enough to make the complete run slower.

| Configuration | First stage | Closing stage | Residual time | Total |
|---|---:|---:|---:|---:|
| English control, mean of 2 runs | 176.879 s | 121.671 s | 4.657 s | 303.207 s |
| Early variant with a faster first stage | 171.079 s | 133.613 s | 4.778 s | 309.470 s |
| Variant − control | **−5.800 s** | **+11.942 s** | +0.121 s | **+6.263 s** |

![The first-stage saving of 5.800 seconds was outweighed by an 11.942-second increase in the closing stage]({{ site.baseurl }}/assets/img/posts/2026-09-05-ai-agent-security-part-12/fig-03-phase-tradeoff.png)
_Figure 3. Residuals and differences are calculated from the reported values. This early variant is distinct from the final retained configuration in Figure 2. Source: [5th-place write-up][w5]._

The reported times are enough to check the calculation:

```python
from decimal import Decimal as D

# Stage timings in seconds from the fifth-place write-up.
control = {"first": D("176.879"), "post": D("121.671"),
           "total": D("303.207")}
variant = {"first": D("171.079"), "post": D("133.613"),
           "total": D("309.470")}

for row in (control, variant):
    row["residual"] = row["total"] - row["first"] - row["post"]

delta = {key: variant[key] - control[key] for key in control}
assert delta["total"] == sum(delta[k] for k in ("first", "post", "residual"))
print(delta["total"])  # 6.263 seconds slower overall
```

The first-stage saving was real. But Rodríguez reports that the early variant brought back a closing response after the tool call, and the extra time afterward exceeded that saving. Counting only the first output would have missed the regression. This comparison does not establish a general advantage or disadvantage for the language used in the message.

### Adapting the analysis to mail

Rodríguez later applied the same analysis to mail. The new tool had different arguments, and its result changed the history passed to the next generation. Reusing a message directly was insufficient: some variants selected the wrong tool or copied an unintended argument. The analysis carried over, but the new candidates still needed to be checked. [5th-place write-up][w5]

Three GPT-OSS mail runs took **86.500, 85.590, and 84.739 seconds**, each retaining 200 findings and cells. A selected Gemma group retained the same counts and averaged **263.427 seconds** across three runs. Elsewhere in the records, identical messages produced substantially different runtimes. The adaptation worked locally; the size of its gains on the hosted evaluator remained a separate question.

Public-best HTTP submission v50 ultimately scored **115.240 → 0.000**, while selected mail submission v57 scored **40.155 → 40.365**. The mail submission succeeded, though its score does not tell us how much each local change contributed. Rodríguez also could not confirm that the code available later was exactly the selected version. [5th-place write-up][w5]

### Failed changes and inconclusive experiments

Other experiments found different ways for a promising change to disappoint. Shorter requests sometimes brought back longer output. Changing a destination could preserve 200 successful calls but lose distinct cells. Warming up the first candidate could speed it up while slowing the whole portfolio. A multi-action sequence could execute correctly without earning more score per second. Hardware failures left some comparisons inconclusive; they said nothing about whether the message had become worse.

I would not write off the time spent on a public-only family simply because it failed privately. That work could still teach us something useful elsewhere. Rodríguez learned how the full execution behaved, then did the additional work needed to apply that knowledge to another family. Our throughput analysis had identified part of the same problem. We also needed to decide where else to use it.

## 7. Sixth place: when a useful constraint became a limit

### Two submissions, two outcomes

poijio joined about three weeks before the deadline and finished sixth with a solo gold medal. One submission reached first place on the public board at **147.530**; a mail alternative scored **37.530**. Their private results were **0.000** and **37.690**, respectively. The mail submission earned the medal. [6th-place write-up][w6]

poijio chose mail partly because it seemed more likely to survive private evaluation. The hosts had warned that public performance did not guarantee held-out performance, and the published evaluation code gave the author reasons to investigate another family. Those clues did not reveal the private implementation. poijio cites this interpretation and the remaining development time as reasons for the choice.

The public-leading approach placed eight scored actions in one candidate. The actions shared setup costs that would otherwise recur across separate candidates. poijio carried this structure into the mail research. In hindsight, the author questions whether it remained economical once the scoring family changed.

### Working backward from the output

poijio's “output-first” approach started with a short, valid execution and examined where ordinary generation departed from it. This gave the search a specific target. At first, however, that execution served as a diagnostic reference; it had not yet been produced by the submitted input.

Prescribing an output in a diagnostic run could confirm that the parser accepted it. It could not show that the model would generate the sequence naturally. Earlier outputs and tool results also changed the state for each subsequent response. poijio found that rebuilding an apparently equivalent history as text did not always reproduce behavior on the live execution path. Successful intermediate tests could not simply be combined into a successful full sequence.

Within the chosen mail structure, the search substantially reduced generated output.

| Model | Initial completion tokens | Final completion tokens | Final cumulative input tokens |
|---|---:|---:|---:|
| GPT-OSS | 506 | 149 | 9,624 |
| Gemma | 174 | 140 | 10,240 |

*The author reports these counts for an eight-action sequence. Completion tokens cover all generated output; cumulative input tokens sum the inputs across eight generations, including repeated history. The last column is not the length of the initial message.* [6th-place write-up][w6]

The input totals help explain why output length was only part of the cost. Earlier content reappeared in later inputs, and cache reuse determined how much needed processing again. Setup and termination also took time. poijio used an approximate token-cost model to guide the research, then total elapsed time to decide which changes to keep. The final counts describe the chosen configurations, not proven minima across all possible solutions.

### Results changed when candidates were isolated

Some promising candidates failed when evaluated under different conditions. A candidate that appeared to complete eight actions in a shared process might complete only one or seven in isolation. One small cohort fell from **12/12 to 5/12**. These tests exposed dependence on the earlier conditions; they did not estimate the success rate of the complete final candidate set. [6th-place write-up][w6]

A good aggregate search metric could also conceal a decisive output error. Looking close to the reference was insufficient if a required part failed to appear during natural generation. Shorter output could likewise be a regression when execution stopped before completing the scored work. For the final checks, poijio used candidates emitted by the finished submission code, rather than relying on an input retained from the search.

Verification took substantial effort because the search experiment and the submitted program could look equivalent while behaving differently. A local gain counted as a submission improvement only if it held in the finished program.

### The reward changed; the requirement stayed

Near the deadline, poijio made eight successful actions a mandatory requirement. That choice limited the search even as validation grew more thorough.

The evaluator rewarded both scored events and distinct score cells. Repeating an event within the same cell kept adding event reward, but no further cell bonus. poijio's comparison illustrates how the balance changed with the event's reward.

| Author's reward comparison | Eight single-action candidates, distinct cells | One eight-action candidate, one cell | Raw reward reduction |
|---|---:|---:|---:|
| Earlier, higher-severity family | 144 | 130 | 9.7% |
| Lower-severity mail family | 48 | 34 | 29.2% |

*This is the author's reward calculation, not a measured throughput comparison. The single-action candidates are assumed to occupy distinct cells. Cell bonuses are deduplicated across all candidates evaluated for each model/guardrail combination, so separately calculated candidate totals cannot simply be added if later candidates reuse a cell.* [6th-place write-up][w6]

poijio had already tested alternatives. A repeated-action variant that produced different cells worked locally, but needed **305–341 GPT-OSS completion tokens**, against 149 for the final configuration. A single-action mail submission had also been tested. The choice involved a real tradeoff: the shorter repeated structure earned less reward from distinct cells, while the more varied structure cost more to generate.

Once eight successful actions became mandatory, the search focused on shortening that structure. In poijio's local validation, the final candidates reused the same cell within each model's evaluation, limiting the accumulated bonus. The table alone cannot establish that another configuration would have earned a higher private score; its full execution cost still needed comparison. But the change in rewards gave the author a reason to reconsider a structure inherited from the earlier family.

### Choosing the constraints was still a human decision

poijio reports that Codex wrote the implementation, evaluation, and submission code, while the author set the strategy and designed the experimental loop. The repository accumulated **1,268 experiment directories**—explicitly not 1,268 discoveries. Choosing the alternative family and reframing the search were important human interventions. [6th-place write-up][w6]

Extensive measurement did not prevent the objective from becoming too narrow. The experiments improved candidates within the requirements, but could not compare alternatives the requirements excluded. I had sometimes made the opposite mistake: treating one implementation's failure as a reason to dismiss an entire direction. poijio's experience reminds me to revisit successful structures too, when the reasons for choosing them change.

## 8. Eleventh place: measuring gains and choosing a hedge

Mohammad Shadab Alam pursued three lines of work. One produced the public-best score of **136.450**. The other two became the final selections: **NB1**, built for measured throughput, and **NB2**, which gave up some speed to behave differently under hypothetical defenses. Alam finished ninth publicly and eleventh privately. The submission with the largest public score did not determine the final rank. [11th-place write-up][w11]

### Choosing the final two

Alam did not select the public-best submission for private scoring, and its private score is not reported. NB1 scored **30.055 → 30.040** and NB2 scored **19.340 → 19.325**. Both used the mail family.

Indirect evidence from evaluation influenced that choice. Alam did not claim to know the private implementation, but had enough reason to prefer other candidates for the final slots. The public approach still helped the author study execution costs.

### The cost model depended on the environment

Alam worked locally on a 96 GB RTX PRO 6000, running the competition's quantized models through llama.cpp. In the configurations analyzed, decoding accounted for approximately **79–89%** of elapsed time, even though input tokens were far more numerous. Token counts alone gave a misleading picture of where the time went.

Alam fitted a cost per model invocation and a cost per generated token. Two runs with different invocation and generated-token counts took **36.80** and **5.84 seconds per candidate**. Setting the constant term to zero allowed those measurements to determine the two coefficients exactly. That fit depended on the assumption: it did not independently show that fixed overhead was absent or that the coefficients would hold for other configurations.

Caching also changed what the local measurements captured. Prefix caching avoided repeated input-processing work; when the comparisons required a full input pass, runtime increased by **90% for GPT** and **122% for Gemma**. Alam reports that an approximately **35-fold** factor between local and hosted times matched three hosted scores without refitting. The calibration helped explain why work that seemed comfortably within budget locally could hit the evaluation time limit.

Those measurements came from a particular setup. Changes to caching, runtime, hardware, or generated behavior can affect both absolute times and relative gains. The local environment had made an important evaluation cost easy to overlook.

### Fewer tokens could hide a failure

Speed alone could also mislead. A candidate that failed to perform the scored action might finish quickly and look attractive in a token comparison. Alam's local measurements show how short output could conceal a loss of valid work:

| Configuration | Decoded tokens per candidate | Raw score per candidate |
|---|---:|---:|
| Control | 35.25 | 6.000 |
| Shorter-output variant | 31.00 | 5.25 |
| Variant that usually failed to act | 2.31 | 0.210 |

The third variant usually failed to produce the intended action. It also belonged to a four-member wording pool, where the defect went unnoticed for weeks. Those quick failures kept the average token count looking healthy while average raw score fell to **4.52** per candidate, below the intended **6.00**. Correcting the pool reportedly improved raw score per second by about **4.5%**. [11th-place write-up][w11]

Each member of a wording pool needed its own checks. In one collection, **194** alternatives became **54** after initial checks and **39** after larger checks; seven were ultimately retained. Those counts document the filtering in that experiment. They do not tell us how many tests or variants would be enough elsewhere.

Alam also questioned what a gain from variety actually meant. A screened collection might beat one fixed value simply because the original value was weak. Some changes added variety at almost no measured cost; others reduced valid behavior. Counting distinct strings could not explain either outcome.

### More candidates did not replace repeated measurements

With identical input bytes, greedy decoding reproduced token counts across local runs, yet the same configuration's throughput varied by as much as **16.6%**. Five conclusions drawn from separate comparisons reversed in one day when Alam repeated the measurements in **A–B–B–A** order. Increasing candidates per block by **25 times** reduced the reported standard deviation only from **1.968% to 1.830%**. Larger blocks did little to address the dominant variation between runs.

ABBA spreads the configurations across time and can reduce some effects of drift. It does not automatically remove warm-up, nonlinear drift, or carryover from earlier runs. Alam also reports overstating the best configuration's performance by **10%** after choosing one favorable mid-run block to represent it.

Repeating a measurement does not fix the wrong denominator. Overall throughput is total valid work divided by total elapsed time:

$$
R = \frac{\sum q}{\sum t}.
$$

```python
# Synthetic accounting example: completed valid items and elapsed seconds.
blocks = [(100, 10.0), (100, 100.0)]

pooled = sum(q for q, _ in blocks) / sum(t for _, t in blocks)
unweighted = sum(q / t for q, t in blocks) / len(blocks)

print(round(pooled, 3))     # 1.818 valid items / second overall
print(round(unweighted, 3)) # 5.500: a different estimand
```

In this synthetic example, the first calculation measures output per unit of total elapsed time. The second gives each block's rate equal weight. Both calculations are valid, but answer different questions. Neither aggregate provides an uncertainty estimate without the underlying repeated measurements.

The control itself could fail, too. One candidate reportedly went from **500/500 successes to 0/100 after a reboot**, despite identical input bytes. Comparisons relying on its earlier behavior were no longer trustworthy, although the observation did not identify the cause. Separately, a submission containing three simultaneous changes lost **69 points**. Alam eventually traced that regression to a model-detection exception that had been silently ignored, sending one model the other model's candidates. The intended research idea had not received a valid test.

### The purpose of the second submission

The two final notebooks pursued different priorities.

| Property | NB1 v11b | NB2 v14 |
|---|---|---|
| Public score | 30.055 | 19.340 |
| Private score | 30.040 | 19.325 |
| GPT sentence structure | A common frame with varying fields | 503 sentence templates |
| Gemma sentence structure | A common frame | 53 sentence templates |
| Intended role | Main submission emphasizing throughput | Slower alternative with different structural dependencies |

Both notebooks varied their messages, but NB1 changed fields within a common frame while NB2 changed whole sentences. NB2 also removed a feature used by NB1's GPT branch. Alam tested whether they failed differently under hypothetical defenses. Those tests explain why the hedge was designed that way; they do not establish the rules of private evaluation. Some stronger authorization checks in the local scenarios rejected both notebooks.

NB1 determined the final score. NB2 added no points in that evaluation. Because the better of the two scores counted, NB2's lower score did not harm NB1's result. It did, however, use a submission slot and development time.

Before reserving a second submission, I want to be as clear about its purpose: what performance does the main submission already provide, and which uncertainty makes the alternative worth keeping? Calling it insurance does not answer either question.

## 9. Three ways a result can fail to transfer

Across these experiments, results failed to transfer when three different kinds of conditions changed.

![Conceptual diagram showing that local behavior, execution performance, and private results require different evidence]({{ site.baseurl }}/assets/img/posts/2026-09-05-ai-agent-security-part-12/fig-04-evidence-layers.png)
_Figure 4. This is a conceptual diagram, not a quantitative estimate. Success at one stage does not guarantee success at the next._

First, **the model representation could change**. xz obtained the desired GPT-OSS behavior in BF16 but could not retain it in GGUF. Second, **the execution environment could change**. Even when behavior stayed the same, hardware, runtime, caching, and measurement boundaries could change elapsed time and relative gains. Third, **the evaluation conditions could change**. An action that earned points under the public guardrail could earn none privately.

| Question | Evidence needed | What it does not tell us |
|---|---|---|
| Does the candidate behave as intended? | The final candidate and records of generation, parsing, and execution | Whether its overall throughput is also strong |
| Did cost fall in the intended environment? | Valid behavior, complete elapsed times, and repeated controls | Whether the same percentage gain holds on other hardware |
| Did it survive private evaluation? | Results tied to the same submission | How the complete hidden policy works |

Rodríguez and Alam measured different configurations under different conditions. Rodríguez emphasizes the complete execution and reconstructed inputs; Alam attributes much of the measured cost to decoding. Neither breakdown can be applied unchanged to the other environment. [5th-place write-up][w5] · [11th-place write-up][w11]

## 10. What I would change

I cannot know whether adopting a winner's method would have reproduced the score. I can reconsider choices that were ours to make: where to spend research time, what to compare, and when to question an assumption.

| Our choices at the time | What the other write-ups add | What I would change |
|---|---|---|
| We had a mail family but treated it mainly as insurance | The 4th-, 5th-, 6th-, and 11th-place authors achieved higher final scores in the same broad family | Give the alternative its own measurement and development time |
| We emphasized our inability to inspect the private policy | The 1st-, 4th-, and 11th-place authors made choices using limited observations | Ask what the available evidence says about candidates, even when the implementation is unknown |
| We had built substantial local checks and tooling | The 5th- and 6th-place authors checked how the final submission actually executed | Give each check a clear decision to inform |
| I generalized some implementation failures to whole research directions | The 6th-place author describes the opposite problem: keeping one structure fixed | Distinguish a failed configuration from a reason to abandon the whole direction |
| I sometimes treated differences between runs as matched comparisons | The 11th-place author reversed conclusions after repeating controls and checking baselines | Match conditions, recheck controls, and use a consistent denominator |
| We blurred string and cell diversity with the risk of private failure | The 11th-place author gave the main submission and hedge different failure assumptions | Explain why the candidates might fail differently, and what that protection costs |

I would first change how we allocated research time. Rodríguez applied measurement methods learned on the public approach to another tool family. We had relevant methods and a mail candidate of our own. We did not know how far it could improve, but we had a concrete alternative to develop.

For the final selection itself, Part 11 provides a calculation grounded in an existing result. Choosing our pure mail submission would have raised the private score from **21.340 to 26.010**, taking us from **115th to 50th** on the final leaderboard recorded then. That would still have left us below the recorded gold-medal boundary of **29.230**. We can calculate this difference because the alternative submission's score is known.

Earlier investment might have improved the candidate further; I cannot put a score on that possibility. The 11th-place score of 30.040, 6th-place score of 37.690, 5th-place score of 40.365, and 4th-place score of 41.325 show what other participants achieved in the same broad family. Their candidates and development histories were their own.

## 11. What the score measured

Reading the results as security research also requires asking what earned the points. The guardrail decided whether an action could proceed, while a separate judge recognized synthetic events for scoring. A mail event labeled `CONFUSED_DEPUTY` satisfied the benchmark's predicate. To establish an authorization violation, we would also need to know what the legitimate user had authorized and what the adversary was allowed to control.

xz discusses this tension in the comments: the guardrail could permit an action that the judge labeled an authorization failure. Their disagreement did not establish which component was right. The fourth-place team likewise describes how predicate definitions and rewards for repetition shaped its work. A higher score showed success against those rules without necessarily demonstrating a broader security failure. [1st-place write-up and comments][w1] · [4th-place write-up][w4]

During the competition, I needed to understand what the evaluator rewarded. When drawing conclusions about security, I also need to consider whether those rewards captured the failures the benchmark intended to measure.

A malicious user instructing an agent has different control from an adversary changing external content during a legitimate user's task. Making that threat model explicit, along with the reward for repeated events, would make the ranking easier to interpret. xz's proposals to clarify the threat model and reconsider duplicate rewards address both issues. [1st-place write-up and comments][w1]

---

## References

1. xz, [*1st place solution*][w1], 2026-09-03.
2. ISAKA Tsuyoshi, yuto083, 4eta, Kohei, Rick, [*4th Place Solution — Optimizing a Simple email.send Attack*][w4], 2026-09-03.
3. Giovanny Rodríguez, [*5th: Compiling User Messages into Faster Tool Calls*][w5], 2026-09-05.
4. poijio, [*6th Place Solution: From Ideal Output Back to Input*][w6], 2026-09-04.
5. Mohammad Shadab Alam, [*11th place solution: Measure What You Can, Survive What You Cannot*][w11], 2026-09-04.
6. [Part 11: When the Mechanism Did Not Transfer]({{ site.baseurl }}/posts/AI-Agent-Security-Part-11-When-the-Mechanism-Did-Not-Transfer/), this series, 2026-09-02.

## 12. What I learned

I need to judge an improvement by the complete result. A lower token count, a faster generation stage, or success on a research model can each be useful, but none settles the whole question. Rodríguez's slower full run after a faster first stage and xz's model mismatch are concrete reminders. I need to check that the final configuration preserves the intended behavior and finishes the full task sooner in the environment where it will run. Repeated comparisons can establish a credible gain there; a different evaluation still needs its own evidence.

I also need to give promising alternatives time to develop. We had a mail submission that survived privately, yet concentrated development on our stronger public approach. Choosing the existing alternative would have improved our final position. I cannot know how far earlier investment would have taken it. Next time, I want to reserve sustained effort for useful alternatives, revisit requirements that have become habits, and let relevant evidence change our priorities before the deadline leaves no room to act.

[w1]: https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/writeups/1st-place-solution
[w4]: https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/writeups/4th-place-solution
[w5]: https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/writeups/compiling-user-messages-into-faster-tool-calls
[w6]: https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/writeups/6th-place-solution
[w11]: https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/writeups/11th-place-solution-measure-what-you-can-survive
