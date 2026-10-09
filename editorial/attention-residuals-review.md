# Attention Residuals article

Article: [Attention Residuals against the clock](../src/content/posts/attention-residuals.md).
Publication and merge were approved by Sujal on October 9, 2026. The post uses
`draft: false` and is included at `/writing/attention-residuals/` in production.

The narrative follows the architectural question, the reason for random-weight
training, correctness checks and pilots, the frozen original comparison, mixer
profiling, the factored score, numerical and memory accounting, and the fresh
paired follow-up. First-person framing describes the documented research choices;
no personal anecdote, emotion, or undocumented chronology was invented.

## Sources and boundaries

- User-provided Colab notebook `1cxxps-0qZ8KPiem91ptMx38u-jvnQUYF`, retrieved October 9,
  2026, and the supplied project report. Notebook source and saved text outputs
  were read without executing cells. Its downloaded SHA-256 is in the figure data.
- [Attention Residuals, arXiv v1](https://arxiv.org/pdf/2603.15031v1), especially
  sections 3.1, 4.1, and 5.1. The article distinguishes the paper's experimental and
  infrastructure setup from this small eager-PyTorch Full AttnRes implementation.
- Notebook code confirms sixteen learned mixers (including the final mixer),
  matching common initial parameters, raw-value mixing, and the score rewrite.
- Original paired values, follow-up paired values, and original test differences
  are extracted from the successful saved output cells, at their printed full
  precision. Throughput and some losses are printed with less precision.
- This is not independent verification of Drive checkpoints, raw training logs,
  dataset ZIPs, or checkpoint receipts. The notebook reports those checks; the
  website renders their saved outcomes.

## Claim-to-evidence map

| Claim | Notebook cell ID / stage |
| --- | --- |
| Model definitions and fixed recipe | `NHda-UFYS8jJ`, `fvhkXRxMTF53` |
| Shakespeare equal-token pilot | `8mVHYFeYcmje` |
| FineWeb timing and learning pilots | `S2tmy4vU2fpx`, `sn0EqEfkCDoK` |
| Original three paired results | `L8h23LE1h60T` |
| Original final test | `Az_J42u0BO_C` |
| Mixer saved-storage inventory | `prYTwYZ_Z9Hu` |
| Rewrite, failed original logit gate, calibrated check, and timing | `gVDlIrBEFRS6` |
| Pre-existing probe allocations | `4AcOGPD0NJaF` |
| Fresh paired follow-up and workspace cleanup | `kNnzQLZYaVbT` |

## Interpretations preserved

- Fixed budget means measured synchronized training time, not equal FLOPs, total
  elapsed session time, or billed units. Setup, evaluation, and backups are outside.
- Equal-token results use a shared sampled prefix at exactly 40.96M processed
  tokens. Exposure can exceed the number of distinct tokens in the dataset file.
- Initial shared weights match, but different residual functions give different
  predictions. The GPT-2 tokenizer is reused; model weights start from scratch.
- 16.5% lower time per token compares the rewrite with the user's original mixer,
  not with the paper authors' implementation. Both remain slower than the baseline.
- 910.789 MiB is saved-for-backward storage reduction, not a clean peak-memory
  reduction. The short probe starts with existing allocations; the follow-up
  supplies the separate zero-start peak-memory measurements.
- The original pointwise logit gate failed. The calibrated performance criterion
  is disclosed; loss/depth-weight/gradient tolerances were unchanged. No claim of
  bitwise equivalence or identical long-run learning is made.
- Follow-up uses fresh baselines and a separately frozen plan, but reuses examined
  data/seeds and changes FP32 operation order. It is exploratory validation-only
  evidence, does not replace the original held-out test, and does not identify the
  cause of the original seed variation.
- No independent research repository URL was supplied for this project. The post
  links directly to the user's notebook and its observed cell IDs. No unknown
  repository, reproduction command, or access to private artifacts is invented.

## Figure and local preview

`editorial/data/attention-residuals-summary.json` contains the values and provenance.
`python scripts/render-attention-residuals-figure.py` regenerates the SVG, requiring
matplotlib. `--png /tmp/attention-results.png` optionally creates a review image.
The script checks the displayed means and sample SD against the seed differences.
It does not run research code, training, or checkpoint evaluation.

Run `npm run dev` for a local preview, or `npm run build` to build the published
article. The existing design and other published posts are unchanged. The article
date is its publication date. The paper discussion describes a motivating
discrepancy and an implementation investigation, not a failed controlled replication.

## Publication review

Rechecked the saved notebook's identity, all four sets of paired validation
differences, their means and sample SD, and the derived headline percentages.
The opening now specifies the 40.96M-token milestone. The numerical discussion
also identifies the sixteen standalone FP64 mixer cases that passed before the
full-model FP32 comparison. The original failed pointwise logit criterion and
the subsequent calibrated criterion remain disclosed. No new research runs or
checkpoint evaluations were performed.
