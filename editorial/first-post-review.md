# First research posts — editorial review

Both posts are unpublished (`draft: true`). This branch is intended for review. The live site has not been changed.

- [Main story](../src/content/posts/fixing-the-grader.md): “When passing tests becomes the reward,” following the research question, controlled intervention, failure mechanisms, reliable execution, and measured policy outcomes.
- [GRPO companion](../src/content/posts/working-through-grpo.md): roughly 1,500 words, with a worked example, objective, zero-loss explanation, and primary references.

## Review before publishing

1. Confirm the first-person framing sounds like you. The questions and sequence are grounded in the supplied study record; no anecdotes about your background, emotions, or time spent were invented. Personal details from your actual experience would make the opening more distinctive.
2. Supply the public research repository or evidence URL you want readers to use. The draft does not invent a source link or copy internal report paths that would be broken on the website. The large study attachment is not committed to this website repository.
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

The intended audience is a technical researcher or research-engineering reviewer. The post does not claim novelty, frontier scale, improved trained-policy performance, or a guaranteed hiring outcome. Links to inspectable research code and evidence are the most important remaining publication addition.
