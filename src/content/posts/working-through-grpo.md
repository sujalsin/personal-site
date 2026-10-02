---
title: "Working through GRPO"
description: "From four scored answers to a policy update: my notes on group-relative advantages, clipping, and a loss that can be zero while learning continues."
date: "2026-10-02"
draft: true
---

In my [booking-capacity experiment](/writing/fixing-the-grader/), the training loop sampled four programs, ran tests, and used their scores to update a language model.

I wanted to understand what happened between those test scores and the weight update. What does it mean to compare answers within a group? How does a nondifferentiable Python test produce a gradient? And why could the logged loss be zero while the model's parameters were changing?

This is my working explanation of outcome-supervised GRPO, with a small numerical example. The paper to start with is [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), especially section 4.1. It introduced Group Relative Policy Optimization as a PPO variant that uses group scores to estimate a baseline without training a separate critic.

## Start with a group of answers

Let $q$ be a prompt. A behavior policy, $\pi_{\mathrm{old}}$, generates $G$ completions. In the booking experiment, $G=4$ and each completion contains a candidate Python program.

The verifier assigns one reward to each completion. For this illustrative group, suppose the rewards are:

$$
(R_1,R_2,R_3,R_4)=(0,0.5,0.5,1).
$$

These are toy scores, not a saved training group. Their mean is $0.5$. For easy arithmetic, use the population standard deviation here:

$$
\sigma_R=\sqrt{\frac{(0-0.5)^2+0+0+(1-0.5)^2}{4}}
=\sqrt{0.125}\approx0.354.
$$

A group-normalized advantage is:

$$
A_i=\frac{R_i-\bar R}{\sigma_R+\delta},
$$

where $\delta$ is a small numerical stabilizer. Ignoring it for this example gives:

| Completion | Reward | Approximate advantage |
| --- | ---: | ---: |
| A | 0 | −1.414 |
| B | 0.5 | 0 |
| C | 0.5 | 0 |
| D | 1 | +1.414 |

An implementation using sample standard deviation gives different magnitudes: approximately −1.225 and +1.225 at the extremes. This table explains the idea; it does not reproduce TRL's exact numerical convention.

The advantage says how the answer scored relative to its companions. It is not an absolute correctness label. A reward of 0.5 can be above average in one group and below average in another.

If every answer scores 1, all centered rewards are zero. If every answer scores 0, the same thing happens. The outcomes are very different, but neither group gives this objective a preference among its sampled answers.

## Where the gradient comes from

A program's test results are discrete. There is no useful derivative through “this Python function returned the right integer.”

The differentiable quantity is the probability the model assigns to generated tokens. Treat the scores and advantages as fixed feedback for an update. Increasing the log probability of a sampled completion with positive advantage increases a simple policy-gradient objective; a negative advantage reverses that incentive.

For intuition, ignoring clipping and regularization, consider:

$$
J_{\mathrm{simple}}(\theta)
=\frac{1}{G}\sum_i A_i\log\pi_\theta(o_i\mid q).
$$

Its gradient is:

$$
\nabla_\theta J_{\mathrm{simple}}
=\frac{1}{G}\sum_i A_i\nabla_\theta\log\pi_\theta(o_i\mid q).
$$

We differentiate the log probabilities, not the tests. The full GRPO surrogate adds a probability ratio, clipping, and a choice of normalization. This simplified expression is a stepping stone, not the exact loss used in the experiment.

It also makes the reward problem concrete. If an incorrect program earns the group's highest reward, its advantage can be positive. The gradient cannot recover a missing test specification from the reward alone. That does not prove that every such update makes the particular bug more likely; model parameters are shared, and the eventual behavior has to be measured.

## Why compare two policies?

During optimization, distinguish the policy that generated the data from the policy being updated. For token $t$ in completion $i$, define:

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})}.
$$

A ratio of 1 means the policies give that sampled token the same probability at that prefix. A ratio of 1.2 means the current policy gives it 20% more probability. It does not mean the whole answer has become 20% more likely; this is a token-level ratio.

[PPO](https://arxiv.org/abs/1707.06347) uses a clipped surrogate to limit the objective's incentive to move too far in the favorable direction on sampled data. The corresponding token term is:

$$
s_{i,t}=\min\left(
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right).
$$

Take $A_i=1$, $\epsilon=0.2$, and $\rho=1.4$. The two terms are 1.4 and 1.2, so the surrogate takes 1.2. Increasing that ratio further gives no additional benefit from this term.

For $A_i=-1$ and $\rho=0.6$, the terms are −0.6 and −0.8. The minimum is −0.8: the objective stops rewarding further movement in that direction past the lower clipping threshold.

Clipping is a property of this objective. It is not a hard guarantee that every probability ratio stays within the interval after an optimizer step.

## Putting the terms together

A sampled outcome-supervised GRPO objective can be written as:

$$
\widehat J(\theta)=
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}
\sum_{t=1}^{T_i}\left[s_{i,t}-\beta k_{i,t}\right].
$$

Here $T_i$ is the number of included completion tokens, and $k_{i,t}$ denotes a KL-penalty estimate relative to a reference policy. Training minimizes the negative objective.

The reference policy and the old policy have different roles. The old policy supplies the denominator associated with the sampled data. The reference policy provides an anchor for regularization. They should not be treated as interchangeable just because both appear in a training implementation.

In my booking study, $\beta=0$, so the KL term contributed nothing. The recorded setup used group-scaled rewards and `loss_type="grpo"`, which normalizes token losses within each completion before averaging completions. Other loss variants change that weighting.

I use the [TRL 0.28.0 documentation](https://huggingface.co/docs/trl/v0.28.0/grpo_trainer) to interpret the saved configuration. “I used GRPOTrainer” is not enough to reconstruct a run: the loss choice, reward scaling, token masks, and number of iterations also matter.

## How can the loss be zero while the gradient is not?

This was the diagnostic that made me slow down. The study logged zero scalar losses at all 288 update boundaries, but 285 had positive gradient norms.

Consider the unclipped, unregularized objective at a point where every sampled token ratio is 1. With the sequence averaging above, each completion contributes its advantage. Since the advantages are centered:

$$
\widehat J=\frac{1}{G}\sum_i A_i=0.
$$

But its gradient at that point is:

$$
\nabla_\theta\widehat J
=\frac{1}{G}\sum_i\frac{A_i}{T_i}
\sum_t\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

The advantages sum to zero. The weighted log-probability gradients need not. The answers contain different tokens in different contexts, so their derivative vectors differ.

A one-dimensional analogy helps. The functions $f(x)=x$ and $g(x)=2x$ both have value zero at $x=0$. Their difference is zero there, but its derivative is −1. Cancellation of values at one point does not imply cancellation of slopes.

There is also an autograd detail worth making explicit. A ratio can be numerically 1 while only its numerator carries gradients:

```python
# A scalar autograd illustration, not a GRPO implementation.
import torch

x = torch.tensor(2.0, requires_grad=True)
ratio = x / x.detach()
ratio.backward()
# ratio.item() == 1.0
# x.grad.item() == 0.5
```

The detached denominator is treated as a constant. Replacing this expression with the constant `1.0` would preserve its value and destroy its gradient.

This explains how zero reported loss and a nonzero update signal can coexist. It does not make zero loss a universal sign of healthy training. In this study, gradient norms, parameter hashes, and reload checks supplied additional evidence.

## Equal rewards are a different case

If all four rewards are identical, every advantage is zero. With no KL penalty, that group supplies no reward-ranking gradient. Three updates in the study had exactly this situation.

Yet even those updates changed parameter hashes. The optimizer was AdamW, which retains moving averages of past gradients. Its first moment follows:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t.
$$

When $g_t=0$, a nonzero $m_{t-1}$ can leave $m_t$ nonzero. The optimizer can therefore move parameters even with a zero current gradient and zero weight decay. The [PyTorch AdamW algorithm](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html) makes that state explicit.

This is a plausible explanation consistent with the saved diagnostics, not a tensor-by-tensor reconstruction of those three updates. “Zero advantage,” “zero gradient,” and “unchanged parameters” describe different stages of the training process.

## The question I return to

GRPO turns relative rewards into an update signal. In the booking experiment, that makes the composition of the test suite part of the learning problem.

The weak suite could give full reward to the wrong endpoint behavior. The repaired suite removed every observed full-score false acceptance. But the four-seed comparison did not establish better final training outcomes. A better acceptance decision on saved programs and a better learning trajectory are separate empirical claims.

That is the connection I want to keep in mind while reading the equations: the advantage is only as meaningful as the comparison the reward function creates. Whether a proposed change improves the model still needs an experiment.

Return to [When passing tests becomes the reward](/writing/fixing-the-grader/) for the complete study story, setup, and results.

## Reading alongside this note

- [DeepSeekMath, section 4.1](https://arxiv.org/html/2402.03300v3#S4.SS1): the original GRPO formulation and outcome/process supervision distinction.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347): the clipped surrogate behind this part of the objective.
- [TRL 0.28.0 GRPOTrainer](https://huggingface.co/docs/trl/v0.28.0/grpo_trainer): the implementation documentation corresponding to the recorded experiment.
- [PyTorch 2.8 AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html): optimizer state and update equations.
