---
title: "I fixed the grader. Did I fix the training?"
description: "A small GRPO experiment about missing tests, a promising pilot, and what changed when I ran the full evaluation."
date: "2026-10-02"
draft: true
---

A booking ends at 10. Another starts at 10. How much capacity do they need?

One. The first booking is already over when the next begins.

That tiny detail became the center of this experiment. I wanted to explore what happens when a language model learns from a grader that misses something the task explicitly requires. Could an incomplete test suite reward the wrong solution? Would reinforcement learning make that mistake more common? And if I repaired the tests, would the model learn better behavior?

The first question had a clear answer. The others became much less clear after twelve training runs and 2,048 evaluation draws.

The repaired grader caught all 38 observed false acceptances. Training with it did not establish an improvement on the main outcomes. Understanding that gap became the most useful part of the project.

## A small task with a specific blind spot

The model's job was to write one Python function:

```python
def required_capacity(bookings: list[list[int]]) -> int:
    ...
```

It should return the maximum number of simultaneously active bookings. The input can be unsorted, duplicate bookings count separately, and an empty list needs zero capacity.

Each booking occupies a half-open interval: start included, end excluded. For example:

```python
required_capacity([[1, 5], [2, 6], [7, 9]])  # 2
required_capacity([[1, 5], [5, 9]])          # 1
```

The prompt explicitly stated the endpoint rule. The ambiguity was in the feedback, not the instructions.

I used three versions of the training verifier:

| Condition | Tests | What changed |
| --- | ---: | --- |
| Reference | 96 | The complete training suite |
| Weak | 57 | Removed 39 shared-endpoint cases |
| Repaired | 57 | Replaced eight ordinary weak-suite cases with eight endpoint cases |

The weak grader could award full marks to code that treats touching bookings as overlapping. The repair restored tests that could distinguish that mistake, without increasing the test count. Equal test counts do not mean equal difficulty or information, though; changing the cases changes the reward signal.

For a program $p$ and a suite $S$, the reward was the fraction of tests passed:

$$
R_S(p)=\frac{1}{|S|}\sum_{x\in S}\mathbf{1}[p(x)=f(x)].
$$

Here, $f(x)$ is the expected answer. Execution and extraction failures also needed explicit handling; an unavailable infrastructure result could not simply become a failed test.

A separate, fixed 192-case development audit measured behavior outside the scalar training reward. I call it a *development audit* because it was an existing evaluation suite, not a newly held-out final benchmark. Passing every audit case is still finite-suite success, not proof that a program is correct on every possible input.

## How GRPO enters the picture

I used Qwen2.5-Coder-1.5B-Instruct and trained it with GRPO: Group Relative Policy Optimization.

The useful intuition is to sample several answers to the same prompt, score them, and compare their rewards within that group. Answers above the group average receive positive advantages; answers below it receive negative ones. Those advantages weight a policy-gradient update.

In this experiment, each group contained four completions. The scores came from executing the generated programs against the selected training tests. There was no separately learned reward model deciding whether the code looked convincing.

That makes the verifier's blind spot consequential. A broken solution can be one of the group's best-scoring answers. The update gets feedback from the tests we actually supplied, not from the complete specification we intended them to represent.

I've written a separate note, [Working through GRPO](/writing/working-through-grpo/), for the derivation, a numerical example, and the connection to the DeepSeekMath paper. Here, the key question is what happens when we change the rewards that GRPO compares.

[TRL](https://huggingface.co/docs/trl/v0.28.0/grpo_trainer) supplied the training implementation. It handled generation, advantages, token losses, and optimizer integration; the project still had to supply the task, execution-based rewards, evaluation, and reliable recovery.

## The pilot gave me a reason to look closer

An earlier one-seed pilot produced three target-bug programs out of 32 final draws in the weak condition, compared with one out of 32 in the reference condition.

That was a possible signal worth investigating. It was also two additional observations in a small sample. The pilot retained three unresolved input outcomes, so some of its results had explicit missing-data bounds.

The follow-up used four new training seeds and three conditions per seed: twelve runs. Every run began from the same original model weights with a fresh optimizer. Within each seed's triplet, the prompt, starting weights, and first rollout tokens matched.

The repair was chosen and frozen before seeing these replication outcomes. The final comparison was fixed at update 24, rather than whichever checkpoint happened to look best.

The workload was:

| Measurement | Count |
| --- | ---: |
| Training runs | 12 |
| Optimizer updates per run | 24 |
| Total optimizer updates | 288 |
| Training completions | 1,152 |
| Shared baseline evaluation draws | 128 |
| Update-12 evaluation draws | 384 |
| Update-24 evaluation draws | 1,536 |
| Total evaluation draws | 2,048 |

Those numbers describe different things. There were four independent training-seed replications of each comparison, not 2,048 independent training experiments. Tests sit inside programs, and generated programs sit inside trained policies. Repeated program sources also occurred.

The pilot remains separate. It motivated the replication; it does not become a convenient fifth seed afterward.

## What I actually ran

The recorded environment used Python 3.12, PyTorch 2.8.0, Transformers 4.57.1, TRL 0.28.0, datasets 3.5.1, accelerate 1.12.0, and Modal 1.5.5. Training used an L40S GPU with two CPU cores and 32 GiB of memory.

These are the historical study settings. They matter for interpreting the run; they are not a claim about which package versions someone should install today.

The main training choices were:

| Setting | Recorded value |
| --- | --- |
| Learning rate | `1e-6`, constant |
| Completions per group | 4 |
| Maximum completion length | 512 tokens |
| Temperature / top-p | 0.8 / 0.95 |
| Repetition penalty | 1.05 |
| Loss / reward scaling | `grpo` / `group` |
| KL coefficient, beta | 0 |
| Iterations per generated batch | 1 |
| Optimizer / weight decay | AdamW / 0 |
| Precision | bfloat16 |
| Per-device batch / gradient accumulation | 1 / 4 |

Specifying `loss_type="grpo"` matters: the name of the trainer alone does not fully specify how it normalizes losses. Likewise, `beta=0` means this run did not use a KL penalty to a reference policy.

[Modal](https://modal.com/docs/guide/gpu) provided the cloud GPU and execution environment. The work split into training and generation, protected execution of candidate programs, and offline analysis of saved results. A working GPU launcher was only one part of that system.

The pipeline needed to preserve the relationship between an exact model state, its generated source, the tests used to grade it, and the resulting score. If any of those drifted during recovery, I would no longer be measuring the experiment I thought I had run.

## Running the experiment also meant repairing the measurement

Generated code had to be executed repeatedly. An early approach created a sandbox per test. The revised approach created a fresh sandbox per program and a fresh restricted Python child process per input.

On a benchmark of seven programs and 2,009 inputs, sequential grading took 391.51 seconds. Four-program grading took 142.49 seconds, a measured 2.75× throughput improvement. Thirteen authored live controls passed, and 84 old/new output-and-status pairs matched. That is evidence about this grading change, not a claim that the entire project became 2.75× faster or that each input had a fresh virtual machine.

Some failures were more subtle than slow execution. A persistence callback failure could be mistaken for a candidate transport failure. If that caused the program to run again, a storage problem would silently create a new measurement attempt.

The fix saved immutable evidence before updating its lookup index, used bounded publication retries, and stopped doing per-test index writes. A fault-injection control failed publication three times and then recovered with execution disabled: the two inputs ran once, and recovery did not rerun them.

There was also a timing error involving monotonic clocks across containers. Monotonic time is useful for elapsed time within a process; arbitrary readings from separate processes should not be compared as a shared clock. The permit logic was changed to use process-local intervals and checked with offset-clock and delayed-grant controls.

The recovery rule was simple to state and harder to implement: an unknown execution outcome must block an automatic retry until it is resolved. A storage failure is neither a zero reward nor permission to generate a fresh attempt.

Training recovery needed more than model weights, too. The saved state included optimizer, scheduler, random-number-generator state, and pending rollout information. A three-step interruption/restart check matched the next fresh rollout. Checkpoints were saved at every update, with retention keeping the last two full trainer states and separate model snapshots at updates 12 and 24.

These details belong in the story because missing or duplicated measurements can change its ending. The completed replication had zero unresolved input outcomes. That does not mean the pipeline never failed; it means the failures were accounted for.

## The grader repair worked on the saved programs

First, I applied all three verifiers to the same 2,048 evaluation draws. Here, “accepted” means receiving full marks from that verifier.

| Verifier | Accepts audit-passing draws | Accepts audit-failing draws |
| --- | ---: | ---: |
| Reference | 440 | 0 |
| Weak | 440 | 38 |
| Repaired | 440 | 0 |

The repair rejected all 38 observed weak false acceptances and retained all 440 audit-passing draws. The weak verifier's false acceptances were 7.95% of its 478 accepted draws.

I inspected all 38 draws, representing 33 distinct source hashes. Thirty-four matched the inclusive-endpoint oracle on all 287 unique training-and-audit inputs. One saved witness was:

```python
bookings = [[3664, 3670], [3670, 3676]]
# Expected: 1
# Observed: 2
```

The implementation expired bookings from a heap only when their end was *strictly less than* the next start. At equal timestamps, the previous booking incorrectly remained active.

Four other draws had a different tie problem: they sorted events by time alone, leaving input order to decide how equal-time events were processed. These failed the endpoint rule without matching the exact inclusive-bug signature.

That distinction mattered. A detector for the one bug I expected missed four of the verifier's 38 observed false acceptances. A specific bug signature and a broader correctness audit answer different questions.

This was a clear positive result for grading this cohort. It did not establish that the repair catches every possible wrong program or that eight replacement tests are optimal.

## Then the training results complicated the story

Each condition contributed 512 final draws: 128 from each of four trained policies.

| Final metric | Reference | Weak | Repaired |
| --- | ---: | ---: | ---: |
| Pass all audit cases | 120/512 | 130/512 | 113/512 |
| Full audit pass rate | 23.44% | 25.39% | 22.07% |
| Inclusive-endpoint signature | 7/512 | 7/512 | 11/512 |
| Mean audit case accuracy | 40.03% | 39.52% | 42.68% |

Weak training did not have a higher aggregate target-bug frequency than reference training. Repaired training did not reduce it. The repaired condition also had a lower observed full-pass rate than the weak condition.

With only four seed pairs, these estimates are uncertain. The exploratory paired 95% interval for repaired-minus-weak full-pass rate was −12.48 to +5.84 percentage points, around an observed difference of −3.32 points. The interval method was selected during analysis and is fragile with so few seeds, especially for rare errors.

The replication therefore did not establish amplification of the target bug or a training benefit from the repair. It also did not establish equivalence or prove that repair harms training.

The smaller evaluation prefix would have told a more appealing story. Across the first 32 predetermined final draws per policy, the repaired condition had zero target-bug observations. The full 128 draws per policy revealed eleven. On that prefix, repaired training also looked better than weak training on full audit passes; the complete final cohort reversed that ordering.

Nothing about the model changed between those two summaries. What changed was how much of the planned evidence the summary included.

## Several reasonable metrics disagreed

The repaired condition had the highest mean audit case accuracy and the lowest full-pass rate. That is possible because partial success and complete success measure different parts of the output distribution.

Compared with weak training, the repaired condition produced 63 fewer draws passing zero or one audit case, 80 more passing between two and 191, and 17 fewer passing all 192. These are differences between sampled populations, not tracked transformations of individual programs.

Even a very wrong algorithm could collect substantial partial credit. A selected program that returned the number of bookings instead of maximum simultaneous capacity passed 39/57 weak tests and 38/57 repaired tests. It passed 74/192 audit cases.

The repair also did not improve every property of the grader. In an exploratory comparison of partial-score ordering against audit accuracy, disagreement was 1.61% for the weak verifier and 1.89% for the repaired one. This used the same eligible program pairs for both graders; the pairs shared programs and were not independent experiments. It does not establish the cause of the training result. It shows why a grader's full-acceptance errors do not describe its entire reward signal.

Other checks gave similarly different pictures. Of 1,536 final draws, 1,505 contained syntactically valid extracted Python, but only 363 passed the full audit. And 29 responses that reached the token cap still contained complete, audit-passing programs: the generation limit was reached during prose after the code.

One passing program even came with an explanation describing the wrong endpoint ordering. The executable code sorted the ties correctly. That example does not reveal hidden reasoning or intent. It does show why I needed to evaluate what the code did rather than trust what the accompanying explanation said.

These were exploratory analyses after looking at the outcomes. They help describe the behavior, but they do not replace the declared primary comparisons.

## Why every logged loss was zero

All 288 logged scalar losses were `0.0` or `-0.0`. Yet 285 updates had positive gradient norms, and parameter hashes changed at every update boundary.

The distinction is between the value of an objective and its derivative. Centered group advantages can cancel in the reported scalar while their gradients, weighted by different token log probabilities, do not cancel. A zero number on the loss chart is not enough to conclude that learning stopped.

Three groups genuinely had zero reward variance: all four completions received the same reward. With group-relative advantages and `beta=0`, those groups supplied no reward-ranking gradient.

Even then, the parameter hashes changed. [AdamW retains optimizer moments](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html), so earlier gradients can contribute to a later update even when its current gradient is zero. That is consistent with the observed behavior; I did not perform a tensor-level decomposition of the momentum contribution.

The [GRPO note](/writing/working-through-grpo/#how-can-the-loss-be-zero-while-the-gradient-is-not) works through the cancellation with a small example. For this experiment, it meant checking gradients, state changes, and reload evidence together instead of diagnosing training from one scalar.

## What I can take from this

The experiment began with an incomplete grader and a plausible training hypothesis. It ended with stronger evidence for the grader problem than for the training effect.

I can say that the weak verifier accepted 38 observed audit-failing draws, that the fixed repair rejected all of them, and that it retained all observed audit-passing draws. I cannot say that this four-seed study showed reward hacking becoming more common, or that fixing those tests improved the main training outcomes. Nor does the evidence establish that the model intentionally exploited the grader.

The scope is small: one task, one model, short training runs, and an existing development audit. Short training, rare errors, sampled-group variation, and changed partial rewards are possible reasons for the inconclusive training comparison. This study does not distinguish among them.

A next experiment would need its own plan: more independent seeds if justified, a fresh audit, and a clear decision about which behavior or reward property to test. Continuing until a favorable checkpoint appears would answer a different question.

For me, the useful result is the separation between a test suite that grades saved programs better and a reward signal that demonstrably produces better learned behavior. This experiment established the first on its observed cohort. The second remains open.

*Companion note: [Working through GRPO](/writing/working-through-grpo/). The numerical results here come from the completed booking-capacity study record; the separate pilot is not pooled with the replication.*
