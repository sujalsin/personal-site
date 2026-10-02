---
title: "When passing tests becomes the reward"
description: "How I built an execution-based RL evaluation pipeline, repaired a verifier blind spot, and measured whether the repair improved training."
date: "2026-10-02"
draft: true
---

I wanted to understand what happens when the tests used to train a coding model leave out part of the specification.

A test suite can be useful without being complete. But once its score becomes a reinforcement-learning reward, every omission becomes part of the feedback the model learns from. A program can receive full reward while implementing the wrong behavior. Repairing that omission raises a second question: does a better grader produce a better trained policy?

I built an execution-based training and evaluation pipeline around Qwen2.5-Coder-1.5B-Instruct and ran twelve GRPO training jobs: three verifiers across four paired seeds. My work covered the task and test suites, reward integration, sandbox execution and recovery, experiment orchestration, and offline analysis. TRL supplied the GRPO trainer; Modal supplied the cloud compute and sandbox primitives.

Three results organize the story:

- **A measurable grader repair.** Replacing eight tests rejected all 38 observed false acceptances across 2,048 evaluation draws, while retaining all 440 draws that passed a separate 192-case audit.
- **A separate test of learning.** The four-seed comparison did not establish better training outcomes. Inspecting all false acceptances exposed a second failure mechanism, and the full evaluation changed the picture suggested by a shorter sample.
- **A traceable execution system.** Fault-injection checks verified recovery without repeating completed candidate executions. A controlled benchmark measured 2.75× grading throughput, and a full-state restart control matched the next fresh training rollout.

The distinction between the first two results is visible below. Improving how a grader classifies saved programs is one claim; improving the policy trained with that grader requires its own evidence.

![Two comparisons. On the same 2,048 evaluation draws, reference and repaired graders accepted zero audit-failing draws, while the weak grader accepted 38; all accepted 440 audit-passing draws. At the final training checkpoint, mean audit full-pass rates were 23.44 percent for reference, 25.39 percent for weak, and 22.07 percent for repaired, with four paired training seeds.](../../assets/booking-study-results.svg)

*Top: all three graders applied to the same saved outputs. Bottom: policies trained with each grader; gray lines connect the four paired seeds and diamonds mark condition means. Each policy contributes 128 final draws. These are descriptive measurements; the paired comparison and uncertainty appear in the [training results](#evaluate-the-learned-policy-separately).*

The [failure investigation](#trace-full-reward-back-to-the-failing-behavior), [execution and recovery design](#build-recovery-around-the-identity-of-an-observation), and [training results](#evaluate-the-learned-policy-separately) can each be read on their own. The story starts with the reward mismatch.

## Make the reward mismatch inspectable

The task was to implement a booking-capacity function:

```python
def required_capacity(bookings: list[list[int]]) -> int:
    ...
```

It returns the maximum number of simultaneously active bookings. Bookings may be unsorted, duplicates consume separate capacity, and an empty input requires zero capacity.

The important rule is that intervals are half-open: a booking occupies capacity from its start up to, but excluding, its end. A booking ending at 10 and another beginning at 10 need one unit of capacity.

```python
required_capacity([[1, 5], [2, 6], [7, 9]])  # 2
required_capacity([[1, 5], [5, 9]])          # 1
```

This task made the intervention concrete. I could remove the cases that distinguish inclusive from half-open endpoints, author tests that restore that distinction, and inspect a failing program's behavior on exact witness inputs. The prompt continued to state the correct rule in every condition.

I constructed three training verifiers:

| Condition | Scored tests | Intervention |
| --- | ---: | --- |
| Reference | 96 | Complete training suite |
| Weak | 57 | Remove 39 shared-endpoint cases |
| Repaired | 57 | Replace eight ordinary weak-suite cases with eight endpoint cases |

For a program $p$ and a suite $S$, the reward was its fraction of passing tests:

$$
R_S(p)=\frac{1}{|S|}\sum_{x\in S}\mathbf{1}[p(x)=f(x)].
$$

Here, $f(x)$ is the expected answer. The repair held the test count at 57 while changing which behaviors the reward distinguished. It matched the grading budget in test count; difficulty and information content necessarily changed with the cases.

A separate 192-case development audit supplied a common yardstick across conditions. Its cases never supplied the scalar training reward. Throughout this post, “audit-passing” means passing all 192 cases.

That gave me two comparisons to keep separate: apply different graders to the **same saved programs**, and train **different policies** with those graders before evaluating them against the common audit.

## Turn a pilot into a controlled comparison

An earlier one-seed pilot produced three target-bug programs in 32 final weak-condition draws, compared with one in 32 reference-condition draws. That observation motivated a replication with four new seeds and a fixed repair.

Within each seed, I ran reference, weak, and repaired training from the same Qwen2.5-Coder-1.5B-Instruct weights with fresh optimizers. The prompt and first rollout tokens matched across the three conditions. These matches were checked against saved identities and checkpoint receipts.

I froze the repair before seeing the replication outcomes and fixed the primary comparison at update 24. Each run used four completions per update, giving 288 optimizer updates and 1,152 training completions across twelve runs.

Evaluation used one shared 128-draw baseline, 32 draws per policy at update 12, and 128 per policy at update 24:

| Cohort | Policies evaluated | Draws per policy | Total draws |
| --- | ---: | ---: | ---: |
| Original model | 1 | 128 | 128 |
| Update 12 | 12 | 32 | 384 |
| Update 24 | 12 | 128 | 1,536 |
| **Total** | | | **2,048** |

Evaluation sampling seeds were shared across policies. Training effects were compared within seed, giving four paired replications. The thousands of generated answers and their individual test outcomes provide behavioral detail; they do not increase the number of independent training seeds.

The pilot remained separate, including its three unresolved input outcomes and corresponding missing-data bounds. The replication completed all planned evaluation batches with zero unresolved input outcomes.

## What the model learns from the verifier

I used GRPO—Group Relative Policy Optimization—with [TRL](https://huggingface.co/docs/trl/v0.28.0/grpo_trainer) providing the trainer. For each prompt, the model generated four completions. The execution system scored their extracted programs, and the trainer compared those rewards within the group.

The group-relative advantage has the form:

$$
A_i=\frac{R_i-\overline R}{\sigma_R+\delta}.
$$

The numerator measures how a completion scored relative to its companions; the denominator scales by group reward variation, with a numerical stabilizer $\delta$. Those advantages weight the policy update. The tests themselves do not need to be differentiable: the gradient flows through the model's token log probabilities.

This is where an omitted test can matter. A program violating the endpoint rule can still be one of the group's highest-reward answers. Whether that incentive produces a measurable increase in the behavior is an empirical question—the reason for running the paired training comparison.

The recorded configuration used `loss_type="grpo"`, group reward scaling, and `beta=0`. That last setting removes the reference-policy KL penalty. I used a constant learning rate of `1e-6`, one optimization iteration per generated batch, and AdamW with zero weight decay.

My companion note, [Working through GRPO](/writing/working-through-grpo/), develops the objective, clipping, and a worked numerical example, with links to the DeepSeekMath paper. The configuration table at the end of this post records the remaining training settings.

## Trace full reward back to the failing behavior

Applying all three verifiers to the same 2,048 evaluation draws isolated the grading question from the training question:

| Verifier | Accepts audit-passing draws | Accepts audit-failing draws |
| --- | ---: | ---: |
| Reference | 440 | 0 |
| Weak | 440 | 38 |
| Repaired | 440 | 0 |

An acceptance here means full marks from that verifier. The weak verifier accepted 478 draws, of which 38—**7.95%**—failed the audit. Replacing eight tests removed every one of those observed false acceptances without rejecting an audit-passing draw or increasing the 57-test budget.

I investigated the entire false-acceptance set: 38 draws representing 33 distinct source hashes. The analysis revalidated 10,906 saved input outcomes across the 287 unique training-and-audit inputs for those draws. This used their original execution evidence rather than rerunning the candidates during analysis.

The source and witness inputs revealed two mechanisms.

### A comparison operator changes the interval semantics

Thirty-four draws matched the authored inclusive-endpoint oracle on all 287 inputs. One saved witness was:

```python
bookings = [[3664, 3670], [3670, 3676]]
# Expected capacity: 1
# Observed capacity: 2
```

The implementation maintained active bookings in a heap, but removed an old booking only when its end was strictly less than the next start. At equal timestamps, it incorrectly kept that booking active. The half-open specification requires expiring it when the end is less than **or equal to** the next start.

These programs passed all 57 weak tests, but only 127 of the 192 audit cases. The scores were evidence of a specific reproducible mismatch, and the witness made the cause inspectable.

### Input order decides an event tie

Four more draws passed the weak verifier while failing the reference and audit. Their code sorted start and end events by time alone, allowing input order to determine which event was processed first at a shared timestamp.

One saved witness was:

```python
bookings = [[5444, 5476], [5420, 5444]]
# Expected capacity: 1
# Observed capacity: 2
```

These draws passed 145 audit cases and matched the inclusive oracle on only 235 of the 287 inputs. They violated the endpoint rule without matching the original exact bug signature.

The broader audit therefore exposed four verifier errors that the target-bug metric missed. I retained the original signature as the primary metric and reported the additional mechanism as follow-up analysis. Together, the two views connected the numerical result to the actual programs that produced it.

## Build recovery around the identity of an observation

Making the experiment traceable required more than calling a trainer and collecting scores. Each observation needed to stay bound to its model checkpoint, generated source, test suite, and execution result—even when a cloud operation failed.

I used [Modal](https://modal.com/docs/guide/gpu) for GPU training and candidate execution, with offline analysis over saved evidence. Three engineering problems became particularly important.

### Increase grading throughput while checking behavior

The initial executor created a sandbox for every test. I changed the execution unit to a fresh sandbox per program, with a fresh restricted Python child process for each input.

On the same seven-program, 2,009-input benchmark, sequential grading took 391.51 seconds and four-program grading took 142.49 seconds: **2.75× measured grading throughput**. Thirteen authored live controls passed, and 84 old/new output-and-status pairs matched. The subsequent evaluation scheduler used four batch workers and up to sixteen program sandboxes.

That measurement describes the grading benchmark. The isolation boundary remained a program sandbox plus per-input child processes, validated for the restricted pure-function task.

### Recover a saved result without creating a new trial

A persistence callback failure could be misclassified as a candidate transport failure. An automatic retry could then execute a program again even though its original result already existed.

I changed the persistence order: commit immutable execution evidence first, then publish its lookup index with bounded retries. I also removed per-test index writes. Recovery could reconstruct the published result from saved evidence.

I tested that property by injecting three publication failures, then recovering with the executor disabled. The two control inputs executed once; recovery performed zero reexecutions. During research recovery, 25 original outcomes were retained, and only the 71 inputs proven not to have been submitted were run.

This made the retry boundary explicit. If execution status was unknown, automatic replay stopped. Infrastructure failures could not silently become zero rewards or extra candidate attempts.

### Resume the same training process

Recovering training required model weights, optimizer and scheduler state, random-number-generator state, and pending rollout information. A three-step interruption/restart control checked those states and matched the subsequent fresh rollout.

I also repaired permit timing that incorrectly compared monotonic-clock readings across containers. Each process instead measured its own elapsed intervals. Four concurrent control batches exercised clock offsets of up to an hour and an injected six-second grant delay before research resumed.

Two completed batch receipts were reconstructed without executing candidates again. Recovery retained 330 evaluation batches and completed the remaining 150, preserving the planned 480-batch population.

The offline analyzer then checked all 2,048 score rows, source bindings, fixed populations, paired initializations, and checkpoint receipts. Its release checks included 38 focused tests covering evidence validation, changed-source rejection, population guards, and seed-level statistics.

These controls gave the final counts a concrete meaning: they referred to a defined set of observations with recoverable evidence.

## Evaluate the learned policy separately

The final evaluation contributed 512 draws per training condition: 128 from each of four policies.

| Final metric | Reference | Weak | Repaired |
| --- | ---: | ---: | ---: |
| Pass all audit cases | 120/512 | 130/512 | 113/512 |
| Full audit pass rate | 23.44% | 25.39% | 22.07% |
| Inclusive-endpoint signature | 7/512 | 7/512 | 11/512 |
| Mean audit case accuracy | 40.03% | 39.52% | 42.68% |

The target-bug counts were equal in the reference and weak conditions. Repaired training had more observed target-bug draws and a lower full-pass rate than weak training. The four paired seeds did **not establish amplification of the target bug or a training benefit from the repair**.

For repaired-minus-weak full-pass rate, the observed difference was −3.32 percentage points, with an exploratory paired 95% interval of −12.48 to +5.84. That interval does not establish harm or equivalence either. It comes from four seed differences; the method was selected during analysis and is especially uncertain for rare-event metrics.

Completing the planned evaluation mattered. In the first 32 predetermined final draws per policy, the repaired condition had zero target-bug observations and a higher full-pass rate than weak training. Extending to all 128 planned draws per policy revealed eleven target-bug draws and reversed the full-pass ordering.

Both summaries describe the same trained policies. The difference is evaluation coverage. Reporting the complete cohort prevented an encouraging prefix from becoming the conclusion.

## Look beyond a single definition of improvement

The final table raised another question: why did repaired training have the highest mean case accuracy but the lowest full-program pass rate?

An exploratory breakdown made the difference visible:

| Final outcome | Reference | Weak | Repaired |
| --- | ---: | ---: | ---: |
| Pass zero or one audit case | 182 | 201 | 138 |
| Pass 2–191 audit cases | 210 | 181 | 261 |
| Pass all 192 audit cases | 120 | 130 | 113 |

Relative to weak training, the repaired condition had 63 fewer near-total failures, 80 more partial solutions, and 17 fewer full passes. These are population comparisons, not tracked changes to individual programs. They explain how average test accuracy and complete success can rank the same conditions differently.

I also examined the graders' ordering of imperfect answers. On the same eligible program pairs, partial-score disagreement with audit accuracy was 1.61% for the weak verifier and 1.89% for the repaired verifier. The repair improved full-acceptance decisions without improving this partial-ordering measure. Because the pairs share programs and this analysis was exploratory, it supplies a diagnostic rather than an independent training-effect estimate.

A concrete example helps: one generated program returned the number of bookings instead of maximum concurrent bookings. That wrong algorithm still earned 39/57 on the weak suite and 38/57 on the repaired suite. Repairing the endpoint omission left other sources of partial reward intact.

I checked presentation-level signals as well. Of 1,536 final responses, 1,505 contained syntactically valid extracted Python, while 363 passed the full audit. Twenty-nine responses reached the token cap after producing complete, audit-passing code. One passing program's prose described the wrong endpoint ordering even though its executable code handled ties correctly.

Those observations motivated separate checks for syntax, extraction, functional behavior, and response format. A single proxy would have obscured distinctions that were visible in the saved execution evidence. These analyses followed inspection of the outcomes; the fixed primary comparisons remained the training results above.

## Diagnose the update, not just the loss chart

All 288 logged scalar losses were `0.0` or `-0.0`. Taken alone, that could look like training had stalled. But 285 updates had positive gradient norms, and parameter hashes changed at every update boundary.

With centered group advantages, the scalar objective can cancel while gradients weighted by different token log probabilities remain nonzero. I checked gradient norms, parameter changes, and reload evidence together. The [GRPO companion](/writing/working-through-grpo/#how-can-the-loss-be-zero-while-the-gradient-is-not) works through the math.

Three groups actually had zero reward variance. With identical rewards, group-relative advantages were zero; with `beta=0`, those groups supplied no reward-ranking gradient. Their parameter hashes still changed, consistent with [AdamW's retained optimizer moments](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html). I did not decompose those particular tensor updates to attribute their movement quantitatively.

This distinction mattered operationally: objective value, current gradient, optimizer state, and parameter displacement each describe a different part of an update.

## What this experiment establishes—and what comes next

The project produced an audited verifier intervention, a source-backed taxonomy of its observed failures, and a completed paired training comparison. The execution system supported fault-tested persistence recovery, full-state training resume, and offline validation of the evaluation population.

The empirical result is precise: replacing eight tests corrected every observed full-score false acceptance while retaining every observed audit-passing draw. The training comparison remained unresolved, and the follow-up analyses identified specific ways acceptance decisions, partial rewards, and complete program success can disagree.

The scope is one booking task, one 1.5B model, four training seeds, and 24 updates per run. The 192-case audit is an existing development suite, so its full-pass metric is a finite behavioral check. The findings establish neither universal verifier reliability nor intentional exploitation by the model.

A useful follow-up would separate two interventions: adding endpoint coverage and improving the ordering of partially correct programs. I would specify those comparisons before collecting outcomes, use a fresh audit, and budget additional independent seeds and enough evaluation draws to measure rare endpoint errors. That would test which property of the reward matters for learning, rather than treating the repaired verifier as a single undifferentiated change.

What I take forward is a way to investigate a reward failure end to end: construct a precise mismatch, trace it to executable behavior, intervene on the verifier, preserve observations through system failures, and measure the learned policy independently. The result is an experiment whose conclusions can be followed all the way back to the code and evidence that produced them.

## Recorded setup

Training used an L40S GPU, two CPU cores, and 32 GiB memory. The pinned environment was Python 3.12, Modal 1.5.5, PyTorch 2.8.0, Transformers 4.57.1, TRL 0.28.0, datasets 3.5.1, and accelerate 1.12.0.

| Setting | Recorded value |
| --- | --- |
| Starting model | Qwen2.5-Coder-1.5B-Instruct |
| Training conditions / seeds | 3 / 4 |
| Updates per run | 24 |
| Learning rate | `1e-6`, constant |
| Completions per group | 4 |
| Maximum completion length | 512 tokens |
| Temperature / top-p | 0.8 / 0.95 |
| Repetition penalty | 1.05 |
| Loss / reward scaling | `grpo` / `group` |
| KL coefficient | `beta=0` |
| Iterations per generated batch | 1 |
| Optimizer / weight decay | AdamW / 0 |
| Precision | bfloat16 |
| Per-device batch / gradient accumulation | 1 / 4 |

Checkpoints were saved after every update; retention kept the last two full trainer states and separate model snapshots at updates 12 and 24. Thirty-one evaluation draws failed source extraction and remained in the denominator as failures; the other 2,017 had sandbox provenance. Repeated generated sources were retained as draws rather than treated as independent programs.

The numerical results come from the completed booking-capacity study record. Final analysis used a 13.68 MiB compact result/provenance bundle and 19.20 MiB of targeted failure evidence, with model and optimizer checkpoints retained in Modal. That analysis required no new model calls or candidate execution.

*For the algorithm behind the training loop, see [Working through GRPO](/writing/working-through-grpo/).*
