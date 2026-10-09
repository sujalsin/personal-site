---
title: "Attention Residuals against the clock"
description: "Training two small language models from scratch led me from a mixed result to a memory investigation, a faster mixer, and a fresh comparison."
date: "2026-10-09"
draft: false
---

**[Experiment notebook](https://colab.research.google.com/drive/1cxxps-0qZ8KPiem91ptMx38u-jvnQUYF) · [Attention Residuals paper](https://arxiv.org/abs/2603.15031v1) · [Results](#the-original-result-depended-on-which-axis-i-used)**

I wanted to understand whether a Transformer could learn more effectively by choosing which earlier layers to read from. The Attention Residuals paper gave me a concrete way to test that idea: replace the usual accumulation of layer outputs with a learned, content-dependent mixture.

That led to the question I could investigate on a Colab GPU: **would better access to earlier representations improve a language model within a fixed training-time budget?**

I implemented a standard GPT and a Full Attention Residuals variant, trained both from random weights, and compared them across three paired seeds. The first result split in two. Attention Residuals achieved lower validation loss at the **40.96M-token milestone** in every pair, but took about **70% more training time per token**. After twenty measured training minutes, its advantage was inconsistent.

The investigation led into the tensors my mixer was saving for backward. An algebraic rewrite reduced time per token by **16.5% relative to my original implementation**. In a separate exploratory comparison with fresh baselines, the rewritten version then achieved lower fixed-time validation loss in all three pairs, by **0.0141 nats/token on average**.

The interesting part was connecting those observations: what the architecture learned, what its implementation cost, and how the cost changed the answer to the original question.

## Why I trained the models from scratch

A pretrained model would have been a convenient starting point for testing downstream behavior. For this question, however, it would have brought a substantial training history into the comparison.

Its attention layers and MLPs would already have learned to work with the standard residual stream. Replacing that stream with learned mixtures would change the inputs those components receive. Any subsequent loss difference would combine the effect of the architecture with the model's adaptation to that change. Leaving the original model untouched would give one condition the benefit of an architecture it had already trained with.

I wanted to observe **how the two architectures learned under a shared training recipe, starting at initialization**. Training from scratch let me control the data, objective, optimizer, initialization of shared parameters, and order of sampled batches. It also let me inspect the entire forward and backward computation when something looked wrong.

I used [nanoGPT](https://github.com/karpathy/nanoGPT) as a reference and implemented a compact decoder-only Transformer in PyTorch. Both variants used learned token and position embeddings, causal self-attention, pre-normalized blocks, GELU MLPs, and tied input/output embedding weights. I reused the GPT-2 tokenizer through `tiktoken`; training the model from scratch did not require inventing another tokenizer.

The main model had eight blocks, six attention heads, and width 384: about **33.7 million parameters**. That made complete paired runs and implementation experiments affordable enough to repeat. The purpose of the model was to make the architectural comparison inspectable.

Within each pair, every shared initial weight matched exactly and both models saw the same sampled token prefix. That did **not** make their initial predictions identical: the residual sum and the AttnRes mixture compute different functions even with matching Transformer weights.

## What changes inside the residual stream

For a standard residual sublayer, the update is

$$
h_{\ell+1}=h_\ell+f_\ell(h_\ell).
$$

Expanding the additions gives an embedding plus the outputs of preceding sublayers. Each contribution enters that sum with coefficient one.

In my Full AttnRes implementation, I retained the embedding and each attention/MLP output as separate source vectors. Before a later sublayer, a mixer scored those sources and formed a weighted combination. At one token position,

$$
s_i=q^\top\operatorname{RMSNorm}_g(v_i),\qquad
\alpha_i=\frac{\exp(s_i)}{\sum_j\exp(s_j)},\qquad
h=\sum_i\alpha_i v_i.
$$

Here, $v_i$ is an earlier source, $q$ is that mixer's learned query vector, and $g$ is the learned normalization gain. The query is a parameter; the weights still depend on the input because the source representations change with the tokens.

The softmax runs across **depth sources at the same token position**. Causal self-attention continues to handle mixing across token positions. I normalized the sources for scoring, while the weighted sum used their original, unnormalized values.

Queries started at zero, giving uniform source weights. The first attention sublayer needed no learned mixer because only the embedding existed. Across eight blocks and the final output mixture, there were sixteen learned mixers, adding **12,288 parameters, about 0.0365%** over the baseline.

That small parameter increase was a poor guide to the eventual execution cost.

## First, make the comparison worth running

I started with Tiny Shakespeare and CPU checks. Before asking whether AttnRes helped, I checked that future tokens could not affect earlier predictions, gradients reached the intended parameters, and saving/restoring a checkpoint reproduced the next update. A tiny selection task checked that the mixer could learn to prefer different sources at different token positions.

The first GPU pilot trained each architecture for 500 updates, or 1.024 million processed tokens. Both learned, but AttnRes finished slightly worse on validation: **5.4017 versus 5.3899 nats/token**. It also took about **1.32× as much training time per token**. That was enough to establish a working training path and an overhead worth measuring.

I then moved to a fixed FineWeb-Edu subset and the larger 33.7M configuration. Short timing trials exposed a bigger gap: around **125,500 tokens/s for the baseline and 73,800 for AttnRes**. Reversing the trial order produced similar measurements.

A five-minute learning pilot on a 10M-token training file also favored the baseline at equal time: **5.3405 versus 5.5138 validation loss**. But the baseline had processed 37.6M tokens while AttnRes had processed 22.1M. These were repeated random-window exposures to the file, not that much distinct text.

Those results sharpened the question. Was the architecture learning less effectively, or was the slower implementation giving it too few learning opportunities within the budget?

## Freeze two ways of measuring progress

For the main comparison, I expanded the fixed corpus to **100M training tokens, 1M validation tokens, and 1M test tokens**. Documents were assigned to splits before tokenization using normalized-text hashes. The preparation rejected duplicate IDs and normalized exact-text duplicates; it did not perform a separate near-duplicate audit.

I fixed three paired seeds (2027, 2028, and 2029) and two endpoints before training:

| Endpoint | What it asks |
| --- | --- |
| **Primary: 1,200 measured training seconds** | Which model reaches lower validation loss within the time budget? |
| **Secondary: exactly 40.96M processed tokens** | Which model learns better from the same sampled token prefix? |

Both models used batch four, context length 512, FP32, AdamW, and a constant learning rate of `3e-4`. I alternated which architecture ran first between pairs. With three pairs, that order was only partially balanced.

The clock included batch sampling, host-to-device transfer, forward/backward passes, gradient clipping, the optimizer update, and CUDA synchronization. Setup, validation, logging, and backups were outside the budget. Every training update counted, and a run stopped at the first completed update reaching 1,200 seconds. **Equal measured training time is the resource constraint here; FLOPs and total Colab billing were not held equal.**

Validation covered all 999,999 next-token targets, including the last short window, with loss weighted by target count. I compared the predefined endpoints rather than choosing each model's best intermediate checkpoint. Source snapshots, dataset fingerprints, update histories, and verified checkpoint backups kept the measurements tied to the runs that produced them.

## The original result depended on which axis I used

Lower cross-entropy means the model assigns more probability to the correct next tokens. In the following table, differences are **AttnRes minus baseline**, in nats/token: negative favors AttnRes.

| Paired seed | At 40.96M tokens | At 1,200 training seconds |
| --- | ---: | ---: |
| 2027 | −0.062132 | +0.006089 |
| 2028 | −0.066659 | −0.018244 |
| 2029 | −0.060954 | +0.021406 |
| **Mean** | **−0.063248** | **+0.003083** |
| Sample SD across seeds | 0.003012 | 0.019995 |

At equal tokens, AttnRes improved all three pairs by similar amounts. At equal time, it improved one pair and lost two. The held-out test, evaluated once for each final checkpoint after the main training and validation analysis, was mixed too: paired differences of **−0.001451, −0.014777, and +0.023399**, with mean **+0.002390**.

The baseline processed about **150.3M tokens per run** in the time budget; AttnRes processed **88.4M**. Reaching the 40.96M-token milestone took roughly 327 seconds for the baseline and 556 for AttnRes. Better learning at that milestone had not established a consistent advantage against the clock.

The figure includes the later follow-up as well. Each comparison uses its own paired baseline; the rewritten model was compared with newly trained controls.

![Validation differences for three paired seeds and their means. At 40.96 million tokens, original and rewritten Attention Residuals both improve all three pairs. At 1,200 training seconds, the original improves one pair and the rewrite improves all three. Mean differences are minus 0.063248 and minus 0.059008 at equal tokens, and plus 0.003083 and minus 0.014091 at equal time.](../../assets/attention-residuals-results.svg)

*Dots show individual paired seeds; diamonds show means. These are observed differences, not confidence intervals. The rewrite comparison is exploratory validation evidence and has no follow-up test evaluation.*

## Why this sent me back to the implementation

The result did not match the practical improvement I had hoped to see from the [paper](https://arxiv.org/pdf/2603.15031v1). Its scaling experiments report favorable loss versus compute, and its discussion of ordinary backpropagation argues that earlier layer outputs are already retained, limiting the extra storage needed for Full AttnRes.

But my experiment was not a reproduction of that setup. The paper's scaling comparison uses larger Kimi Linear models, an 8,192-token context, a cosine learning-rate schedule, and a FLOP-based compute axis. My small dense GPT used constant learning rate and measured accelerator time. The paper also develops Block AttnRes and infrastructure optimizations; I had implemented Full AttnRes in eager PyTorch.

The actionable discrepancy was in my own measurements: **an extra 0.0365% in parameters had accompanied 70% more time per token and about 58% more peak allocated memory**. I needed to account for the actual operations and tensors behind those numbers.

## The cost was hiding in the mixer intermediates

I profiled fresh disposable models on training data, leaving the completed study checkpoints alone. Ordinary, unprofiled timing trials measured throughput; separate profiler traces and an autograd saved-tensor inventory helped explain the computation. Keeping those measurements separate mattered because profiling itself changes execution and can retain tensors.

At batch four, context 512, and width 384, one FP32 source tensor occupies **3 MiB**. The sixteen mixers stack progressively larger source lists, from two sources through seventeen. Summed across the forward pass, those stacks account for **456 MiB** of allocation volume.

The original scoring expression then created large normalized and gain-scaled intermediates:

```python
keys = values * inverse_rms * gain
scores = (keys * query).sum(dim=-1)
```

The inventory found **783.31 MiB** of non-parameter storage saved for backward in the baseline and **2,153.68 MiB** in AttnRes. Almost the entire difference (**1,370.38 MiB**) was first saved inside the mixers. This accounted for more than simply retaining the earlier layer outputs.

The opportunity was in the score calculation. For one source vector, let $r$ be its scalar inverse RMS:

$$
\sum_c ((v_c r)g_c)q_c
=r\sum_c v_c(g_cq_c).
$$

I could combine the two small learned vectors first, contract the source's channel dimension, and apply the normalization factor afterward:

```python
direction = gain * query
scores = torch.matmul(values, direction) * inverse_rms
```

These snippets show the scoring change; the implementation also handles shapes and dtypes. The rewrite kept the source stack, raw-value mixture, parameters, epsilon, and training recipe. It avoided two large intermediates in the scoring path.

The saved-storage reduction was **910.789 MiB**, matching the accounting prediction. In the separate short performance probe, time per token fell **16.5%** relative to the original mixer. The rewritten version still took **1.415× baseline time per token**. This was a measured improvement to my implementation, not a speedup over the paper authors' implementation.

## The rewrite still had to earn my trust

The two expressions are identical in real arithmetic. Standalone FP64 checks of mixer outputs, weights, and gradients passed across sixteen source-count/input-scale cases, including nonzero queries. FP32 changes the order of rounding, so I also compared the full models before treating the rewrite as a performance improvement.

The original pointwise logit check **failed with nonzero mixer queries**. The saved audit showed 787 mismatches out of about 103 million logits under the original tolerances. The largest absolute difference was `1.344e-5`; relative L2 difference was `1.390e-6`. Loss, depth-weight, and gradient comparisons passed their original criteria.

After the audit, I documented a revised logit criterion for the performance probe: maximum absolute difference at most `2e-5` and relative L2 difference at most `1e-5`. I retained the original failed result and did not relax the loss, weight, or gradient criteria. Passing that calibrated check supported benchmarking the rewrite; it did not guarantee an identical long training trajectory.

There was a second measurement issue. The probe initially printed a baseline peak of 6.788 GiB, much higher than the earlier 2.31 GiB. Reading the saved memory records showed that every trial had begun with **4.480 GiB already allocated** in the notebook.

Subtracting that starting allocation was useful bookkeeping, but could not establish clean-runtime model peaks without knowing those tensors' lifetimes. I therefore used the probe for its direct timing and saved-storage evidence, then required zero live PyTorch CUDA allocator bytes before each run in the learning follow-up.

## Ask the learning question again, with fresh controls

Faster updates made a new experiment worthwhile: could the rewritten implementation now turn its token-level advantage into lower loss within the same time budget?

I froze a separate six-run plan: three fresh baselines paired with three factored AttnRes models. It kept the data, dimensions, optimizer, precision, learning rate, paired seeds, run order, 1,200-second budget, and 40.96M-token milestone. The original baseline runs remained historical results; they were not reused as controls.

The queue's memory check also caught 17.25 MiB of retained BLAS workspace after the first baseline. I recorded and applied workspace cleanup between model invocations, preserved the completed baseline, and checked zero allocator bytes before every run. All six training runs completed uninterrupted.

| Paired seed | Fresh baseline validation | Rewritten AttnRes validation | Difference at 1,200 seconds |
| --- | ---: | ---: | ---: |
| 2027 | 5.070846 | 5.048497 | −0.022349 |
| 2028 | 5.084349 | 5.068566 | −0.015783 |
| 2029 | 5.063196 | 5.059056 | −0.004140 |
| **Mean difference** | | | **−0.014091** |
| Sample SD of differences | | | 0.009222 |

This time, AttnRes finished with lower validation loss in **all three pairs**. The mean difference corresponds to about **1.4% lower perplexity**, using the geometric mean of the paired perplexity ratios. The equal-token advantage also remained: **−0.059008 nats/token on average**.

The rewritten model processed about **104.9M tokens per run**, while its fresh baseline processed 149.9M. Its remaining cost was measurable:

| Follow-up measurement | Fresh baseline | Rewritten AttnRes |
| --- | ---: | ---: |
| Pooled training tokens/s | 124,954 | 87,419 |
| Training time per token, relative to baseline | 1.000× | 1.429× |
| Maximum peak allocated training memory | 2.313 GiB | 2.773 GiB |

These later runs are consistent with the interpretation that avoidable overhead had been limiting the practical benefit of the architecture. They do not isolate recovered speed as the sole cause: the rewrite also changed FP32 operation order.

This is **exploratory validation evidence**. I reused data and seeds after inspecting the original results, including the original test results, and performed no follow-up test evaluation. An independent confirmation would need a newly reserved held-out set and a frozen implementation. The original mixed test result remains part of the record.

## What I would carry into the next experiment

Training from scratch made it possible to follow the question all the way from residual mixing to learning curves and backward storage. A tiny parameter change altered runtime enough to change what the same training budget could buy. Profiling identified an avoidable cost; the rewrite produced the predicted storage reduction; fresh paired training then showed a modest, consistent validation improvement in the observed runs.

The remaining questions are concrete. I would reserve new evaluation data, use new paired seeds, and test whether the result persists with longer training. A separate comparison could measure Block AttnRes, while batch size, mixed precision, and compilation would test how much the cost depends on this execution setup. Changing those factors together would make the next result harder to interpret, so each needs a defined comparison.

For this experiment, I can say where the original implementation spent extra memory, which computation I changed, how much faster it became, and what happened when I trained fresh models with it. That is a more useful answer to the starting question than either the first encouraging token-level result or the first disappointing time-budget result on its own.

## Experiment details and saved evidence

| Setting | Main study and learning follow-up |
| --- | --- |
| Model | Eight blocks; width 384; six heads; context 512 |
| Parameters | Baseline 33,691,776; AttnRes 33,704,064 |
| Tokenizer | GPT-2 BPE; vocabulary 50,257; `tiktoken 0.12.0` |
| Data | Fixed FineWeb-Edu subset; 100M / 1M / 1M train/validation/test tokens |
| Sampling | Random contiguous training windows with replacement; paired sampler seeds |
| Optimization | AdamW; constant LR `3e-4`; betas `(0.9, 0.95)`; weight decay `0.1`; gradient clip `1.0` |
| Execution | Batch four; FP32; dropout zero; no AMP or compilation |
| Hardware/software | NVIDIA RTX PRO 6000 Blackwell Server Edition; PyTorch `2.11.0+cu130`; CUDA `13.0` |
| Evaluation | Full 999,999-target split; context reset every 512 targets; target-weighted loss |

Three seeds and one shallow, short-training configuration do not establish a general architecture ranking. The study did not tune hyperparameters separately for the two models, measure convergence, or isolate the causes of the original seed variation.

The [Colab notebook](https://colab.research.google.com/drive/1cxxps-0qZ8KPiem91ptMx38u-jvnQUYF) contains the implementation, frozen plans, and saved outputs. Useful entry points are the [main paired results](https://colab.research.google.com/drive/1cxxps-0qZ8KPiem91ptMx38u-jvnQUYF#scrollTo=L8h23LE1h60T), [original held-out test](https://colab.research.google.com/drive/1cxxps-0qZ8KPiem91ptMx38u-jvnQUYF#scrollTo=Az_J42u0BO_C), [profiler and storage inventory](https://colab.research.google.com/drive/1cxxps-0qZ8KPiem91ptMx38u-jvnQUYF#scrollTo=prYTwYZ_Z9Hu), [rewrite and numerical audit](https://colab.research.google.com/drive/1cxxps-0qZ8KPiem91ptMx38u-jvnQUYF#scrollTo=gVDlIrBEFRS6), and [follow-up queue and results](https://colab.research.google.com/drive/1cxxps-0qZ8KPiem91ptMx38u-jvnQUYF#scrollTo=kNnzQLZYaVbT).

Tables and the figure reproduce saved output measurements; they are not a fresh training run or an independent audit of the remote checkpoints. Differences use the saved unrounded values, so subtracting displayed losses can differ in the final decimal place. The website source includes the figure's data and plotting script.
