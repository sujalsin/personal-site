# First research posts — editorial review

Both posts are unpublished (`draft: true`). This branch is intended for review. The live site has not been changed.

- [Main story](../src/content/posts/fixing-the-grader.md): “When passing tests becomes the reward,” following the research question, controlled intervention, failure mechanisms, reliable execution, and measured policy outcomes.
- [GRPO companion](../src/content/posts/working-through-grpo.md): roughly 1,500 words, with a worked example, objective, zero-loss explanation, and primary references.

## Review before publishing

1. Confirm the first-person framing sounds like you. The questions and sequence are grounded in the supplied study record; no anecdotes about your background, emotions, or time spent were invented. Personal details from your actual experience would make the opening more distinctive.
2. Review the evidence snapshot at `c698e7970f0e57729cd9c5142ee63002b1036fc8` in [sujalsin/verifier-rl](https://github.com/sujalsin/verifier-rl/tree/c698e7970f0e57729cd9c5142ee63002b1036fc8) and [research fix PR #1](https://github.com/sujalsin/verifier-rl/pull/1) before publication. The post now uses verified commit-pinned links and a tested portable reproduction command. The large study record remains in the research repository.
3. Decide whether the Modal discussion should become a separate reproducibility tutorial. This draft explains the actual environment and architecture, but does not pretend the website contains a runnable training launcher or the research checkpoints. A tutorial should use your research source and recovery controls.
4. Set the intended publication dates. `2026-10-02` is the draft preparation date, not an inferred date for all experiments. Seed identifiers such as `20261011` are not dates of execution.
5. Publish the linked pair together, or remove links to any companion that remains unpublished. Set each approved post to `draft: false`, build, then merge when ready.

## Evidence checks used in drafting

The supplied complete booking study record is the source for experimental claims. Its opening research narrative, saved trainer configuration, operational controls, and exploratory findings support the text. No new training or candidate execution was performed for drafting.

- Reference/weak/repaired training suites: 96/57/57; fixed development audit: 192.
- Replication: four seeds × three conditions × 24 updates × four completions = 1,152 training completions.
- Evaluation: 128 + (12 × 32) + (12 × 128) = 2,048 draws. Final cohort: 1,536. Extraction failures remain in the original denominators.
- Offline grader repair: 38/38 observed weak false acceptances rejected; 440/440 audit-passing draws retained. Finite-cohort result, not a universal correctness claim.
- Final full passes: reference 120/512, weak 130/512, repaired 113/512. Target signature: 7/512, 7/512, 11/512.
- Four training seeds are the replication units. The pilot stays separate; the audit is not described as untouched. Exploratory findings are identified as such.
- Recorded `loss_type="grpo"`, `scale_rewards="group"`, and `beta=0` are preserved. The companion's reward table explicitly uses toy population-standard-deviation arithmetic, not an exact saved TRL group.
- No actual spending total is claimed. The source's application compute guard is not a provider invoice or hard billing cap.

## Local review

Run `npm ci` and `npm run dev`, then open both `/writing/` routes above. Drafts are included only in the development server and excluded by `npm run build`.

Equations use the Markdown math pipeline and bundled KaTeX styles. The existing portfolio design is preserved.

## Revision: make the research and engineering contribution visible

The main post now opens with the research question, implemented system, and measured contribution. It explains why the task permits an inspectable intervention, makes experimental controls explicit, traces both failure mechanisms to witnesses, and describes recovery through the invariants and fault-injection evidence it preserves. The full training comparison remains prominent, while scope limitations are consolidated instead of repeated throughout the narrative. Historical package settings sit at the end so they do not interrupt the research story.

The intended audience is a technical researcher or research-engineering reviewer. The post does not claim novelty, frontier scale, improved trained-policy performance, or a guaranteed hiring outcome. Commit-pinned implementation, tests, reports, and reproduction links have now been added.

## Target-role review, 2026-10-02

Compared against the current [Research Engineer, Frontier Evals & Environments posting](https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/). The following is an editorial assessment of the evidence, not an OpenAI hiring decision.

| Role emphasis | Evidence this project can support |
| --- | --- |
| Measurement reliability and variance | Paired training seeds, complete evaluation cohorts, separate audit, and a visible comparison of grading versus training results |
| Behavioral investigation | All 38 false acceptances inspected; witness inputs distinguish two mechanisms |
| Evaluation systems | Execution batching, provenance checks, persistence fault injection, and validated restart controls |
| Experiment ownership | A defined behavioral question carried through a frozen intervention, training, analysis, and a concrete follow-up |

The current post is strongest as evidence of careful research engineering. It does not yet demonstrate ambitious multi-task or long-horizon environments, large training-run impact, a novel RL algorithm, or an automated self-improvement system. Those gaps should be acknowledged in an application assessment, not filled with unsupported language in the post.

The opening now identifies project-specific implementation versus TRL/Modal, summarizes three contributions, links to the main technical sections, and includes a figure with the four seed pairs. The SVG is generated from `editorial/data/booking-study-summary.json` by `scripts/render-booking-figure.py` (requires Python, matplotlib, and numpy). These are transcribed summary measurements, not a substitute for the underlying execution records. They are included in the source for inspection.

The post now links the experiment configuration, reward/executor implementation, source-bound failure artifacts, and analysis command with its required inputs. Links lead to an inspected research commit; the personal-site source is used only for presentation.

## Public repository verification

Inspected public research `main` at `aacf98d3b49c61a1bb65c20f699607840d560f59`. Source review confirmed the fixed repair, full training-input execution with subset scoring, unknown-result reward gate, recovery hooks, and associated regression tests.

The portable CSV had LF bytes while its unchanged source manifest expected CRLF. The original command failed at the checksum guard. [Research draft PR #1](https://github.com/sujalsin/verifier-rl/pull/1) restores exactly the manifest-matching CSV bytes and adds a file-specific `-text` attribute. No values, source manifests, historical reports, or research algorithms were changed. Snapshot `c698e7970f0e57729cd9c5142ee63002b1036fc8` includes that fix; its remote CSV Git blob was verified as `328df0afacf98c0b1defe6d6287d7d683162d1bf`.

The unchanged portable analyzer then completed over all 2,048 rows. Website figure counts and every training-seed pair matched the recomputation. This was an arithmetic reproduction, not new training, candidate execution, or a full private-evidence audit.

The article was also corrected to match published clarifications: the training/audit suites share the empty input; all conditions execute 96 training inputs and score 96/57/57; the model uses FP32 weights with BF16 autocast.

The research PR and website PR remain drafts. The reproduction command pins the corrected research snapshot so it does not depend on when the fix reaches main.
