# Round 1 Refinement

## Problem Anchor

- Bottom-line problem: The course project needs a defensible answer to how different constraint families affect offline-to-online RL under low-quality D4RL replay data, without turning the whole project into an overclaimed ATLAS-only algorithm paper.
- Must-solve bottleneck: Low-quality replay contains many suboptimal or misleading actions. Weak behavior regularization underuses useful data, but strong teacher/action constraints can over-constrain online adaptation. SSAR exposes a strong IQL-qv trusted-action signal, but that signal is expensive and not automatically safe to carry into online fine-tuning.
- Non-goals: Do not claim ATLAS is SOTA; do not run a 6-env x 3-seed benchmark before the mechanism is clear; do not vendor no-license third-party code; do not make broad PRDC/A2PR expansion the main contribution.
- Constraints: Course-project timeline; AutoDL budget should stay exploration-first; one or two replay environments and selective seeds are acceptable; A/B-line teammate results are still pending; C-line evidence is mostly smoke-to-mechanism validation, not final benchmark proof.
- Success condition: A coherent workshop-style project where A/B/C lines compare value conservatism, non-conservative contrast, and policy/trusted-action regularization under one protocol; C-line contributes a clear mechanism finding about trusted-action selection and online constraint release; remaining TODOs are narrow enough for teammates to fill.

## Anchor Check

- Original bottleneck: Which constraints are useful under low-quality offline replay, and when do those constraints become harmful during online fine-tuning?
- Why the revised method still addresses it: The revision keeps the group-level A/B/C comparison but makes the C-line mechanism precise: trusted labels help offline, then a release schedule tests whether the same constraint should be relaxed online.
- Reviewer suggestions rejected as drift:
  - Do not turn the project into an ATLAS-only SOTA claim.
  - Do not expand to a broad walker2d control suite before the group-level A/B rows are filled.
  - Do not add a learned online confidence gate yet; first test the simpler linear release schedule.

## Simplicity Check

- Dominant contribution after revision: A controlled diagnosis of trusted-action constraints in offline-to-online RL: aligned teacher labels improve offline learning, but online fine-tuning needs release rather than fixed teacher regularization.
- Components removed or merged:
  - Removed state/action threshold gate from P0.
  - Merged online evaluation into a clean 2x2 design: `{TD3+BC, ATLAS} x {fixed, linear-decay}`.
  - Kept continuous advantage regression as a stretch ablation, not the default method.
- Reviewer suggestions rejected as unnecessary complexity:
  - Extra selector-capacity sweep.
  - Full second-env shuffled-label run before the paper has A/B-line rows.
- Why the remaining mechanism is still the smallest adequate route: It reuses the existing teacher cache and TD3+BC loop, adds one selector and one release schedule, and directly tests the observed offline-to-online failure.

## Changes Made

### 1. Pinned ATLAS Selector And Actor Loss

- Reviewer said: method specificity is too loose.
- Action: fixed the selector interface, default architecture, loss, label source, and actor BC weighting.
- Reasoning: the paper cannot defend ATLAS if `g_phi` and its use in TD3+BC remain ambiguous.
- Impact on core method: ATLAS becomes an implementable controlled probe rather than a name for arbitrary weighted BC.

### 2. Promoted Constraint Release To P0

- Reviewer said: release schedule is implied by the thesis but treated as optional.
- Action: made linear release the required next C-line experiment if we run one more experiment.
- Reasoning: without release, the paper is observational; with release, even a negative result becomes a sharper test.
- Impact on core method: the C-line now tests both offline trust and online release under the same mechanism.

### 3. Added Frontier Positioning Targets

- Reviewer said: the proposal needs direct offline-to-online related work.
- Action: added a verified citation-target list for PROTO, ENOTO, SUF, Adaptive BC Regularization, Cal-QL, IQL/AWR-style advantage weighting.
- Reasoning: we should not pretend constraint-release is newly discovered.
- Impact on core method: novelty is narrowed to controlled label-quality decomposition plus post-cache trusted-action distillation.

## Revised Proposal

# Research Proposal: Trusted Constraints Need Release in Offline-to-Online RL

## Problem Anchor

- Bottom-line problem: The course project needs a defensible answer to how different constraint families affect offline-to-online RL under low-quality D4RL replay data, without turning the whole project into an overclaimed ATLAS-only algorithm paper.
- Must-solve bottleneck: Low-quality replay contains many suboptimal or misleading actions. Weak behavior regularization underuses useful data, but strong teacher/action constraints can over-constrain online adaptation. SSAR exposes a strong IQL-qv trusted-action signal, but that signal is expensive and not automatically safe to carry into online fine-tuning.
- Non-goals: Do not claim ATLAS is SOTA; do not run a 6-env x 3-seed benchmark before the mechanism is clear; do not vendor no-license third-party code; do not make broad PRDC/A2PR expansion the main contribution.
- Constraints: Course-project timeline; AutoDL budget should stay exploration-first; one or two replay environments and selective seeds are acceptable; A/B-line teammate results are still pending; C-line evidence is mostly smoke-to-mechanism validation, not final benchmark proof.
- Success condition: A coherent workshop-style project where A/B/C lines compare value conservatism, non-conservative contrast, and policy/trusted-action regularization under one protocol; C-line contributes a clear mechanism finding about trusted-action selection and online constraint release; remaining TODOs are narrow enough for teammates to fill.

## Technical Gap

The broad claim "conservatism affects online fine-tuning" is already known. Our project should make a narrower and more defensible claim:

> In low-quality replay data, the action-level trust signal matters. Aligned trusted-action labels can improve offline behavior-regularized learning, but carrying that same teacher constraint into online fine-tuning can block adaptation unless the constraint is released.

This gap is operational. It asks three concrete questions:

1. Does trusted-action selection explain the gap between weak TD3+BC-style regularization and strong SSAR-style behavior?
2. Can a cached teacher signal be distilled into a small selector without claiming a new SOTA algorithm?
3. During online fine-tuning, should the trusted-action constraint remain fixed, or should it decay?

## Method Thesis

- One-sentence thesis: Trusted-action labels are useful offline supervision under low-quality replay, but they should be treated as an initialization constraint and released during online fine-tuning.
- Smallest adequate intervention: reuse TD3+BC plus a cached SSAR/IQL-qv teacher; add one supervised selector and one linear release schedule.
- Current-era positioning: this is an offline-to-online constraint-transfer study, not a generic offline RL benchmark. It should be positioned against policy regularization, adaptive BC regularization, unconstrained/stabilized fine-tuning, and Q-ensemble O2O work.

## Contribution Focus

- Dominant contribution: controlled evidence that the quality and persistence of behavior constraints matter separately: aligned trust labels help offline, while fixed trust labels can hurt online.
- Supporting contribution: ATLAS, a lightweight post-cache selector that makes SSAR/IQL-qv trusted-action information reusable for controlled ablations.
- Explicit non-contributions:
  - no SOTA claim;
  - no complete D4RL benchmark;
  - no claim that ATLAS is cheaper than SSAR from scratch;
  - no claim that PRDC/A2PR fail generally.

## Proposed Method

### Complexity Budget

- Frozen / reused:
  - D4RL MuJoCo replay tasks.
  - TD3+BC-style actor/critic loop.
  - Existing CQL/IQL compact baselines.
  - External SSAR source as teacher/cache reference.
- New:
  - ATLAS selector `g_phi(s,a)`.
  - Linear online release schedule for the ATLAS/BC regularization weight.
- Excluded:
  - learned online gate;
  - selector-capacity sweep;
  - diffusion/model-based policy;
  - broad multi-env/multi-seed sweep before A/B merge.

### System Overview

```text
D4RL low-quality replay
        |
        +-- A-line: value conservatism          -> CQL / Cal-QL row
        +-- B-line: non-conservative contrast   -> PPO / SAC / vanilla TD3-style row
        +-- C-line: behavior regularization
               |
               +-- TD3+BC / ReBRAC anchors
               +-- SSAR/IQL-qv teacher cache
               +-- ATLAS selector g_phi(s,a)
               +-- fixed vs linear-release online fine-tuning
        |
        v
compare offline final/best, online delta, and constraint failure mode
```

### ATLAS Selector

- Input: normalized D4RL state `s` and dataset action `a`.
- Output: trust score `g_phi(s,a) in [0,1]`.
- Default architecture: MLP over concatenated `(s,a)` with two hidden layers, ReLU activations, and sigmoid output. Use the repo default hidden width if already defined; otherwise use 256.
- Label source: cached SSAR/IQL-qv teacher export.
- Default supervision: binary cross entropy on teacher-derived trust labels.
- Stretch ablation: continuous IQL-qv advantage regression, only if the binary selector/release story needs more depth.

### Weighted TD3+BC Objective

The actor objective is:

```text
L_actor = L_Q + lambda(t) * mean_i [ w_i * || pi_theta(s_i) - a_i ||_2^2 ]
```

where:

```text
w_i = stopgrad(clip(g_phi(s_i, a_i), 0, 1))
```

The shuffled-label control keeps the weight distribution but breaks `(s,a)` alignment. If shuffled labels collapse while aligned labels work, the evidence supports label quality rather than generic weighting.

### Online Release

The required P0 online test is:

```text
lambda(t) = lambda_0                         fixed condition
lambda(t) = lambda_0 * max(0, 1 - t / K)      linear-release condition
```

Run the clean 2x2:

```text
{TD3+BC, ATLAS} x {fixed, linear-release}
```

The release schedule is not a separate algorithmic contribution yet. It is the minimal intervention needed to test whether the offline-useful trusted constraint should be relaxed online.

### Online Protocol

Use the existing minimal O2O protocol unless deliberately changed:

- environment: `hopper-medium-replay-v2`;
- seed: 0 first;
- offline pretraining: 50k for O2O slice;
- online fine-tuning: 10k first;
- eval: every 1k online steps with the existing eval episode count;
- report: offline final, online final, online best, and online delta.

If final writing needs stronger evidence, extend only one dimension at a time: longer online horizon or higher eval episodes, not a full sweep.

## Related-Work Positioning Targets

These citation targets must be verified before entering the final bibliography:

- PROTO: iterative policy-regularized offline-to-online RL.
- ENOTO: offline-to-online RL with Q-ensembles.
- SUF: stabilized unconstrained fine-tuning.
- Adaptive behavior cloning regularization for stable offline-to-online RL.
- Cal-QL and conservative value methods.
- IQL / AWR-style advantage-weighted policy extraction.
- ReBRAC, PRDC, A2PR, and SSAR as policy-regularization / action-selection neighbors.

Positioning sentence:

> Prior O2O work studies how to regularize or unregularize fine-tuning. Our narrower contribution is a controlled label-quality decomposition: SSAR-style action-level trust labels are useful offline, shuffled trust weights fail, and fixed teacher regularization can obstruct online adaptation.

## Claim-Driven Validation

### Claim 1: Aligned trusted-action labels matter offline

- Minimal evidence already available:
  - ATLAS seed0 100k final/best: 69.97 / 69.97.
  - ATLAS seed1 100k final/best: 68.11 / 68.11.
  - shuffled-label ATLAS 50k final/best: 18.78 / 19.35 versus aligned ATLAS 50k final/best: 45.29 / 45.29.
- Required comparison:
  - show aligned-vs-shuffled gap explicitly.
- Threshold:
  - aligned labels should beat shuffled labels by at least 10 normalized-score points on the diagnostic env. Current gap is much larger.

### Claim 2: SSAR's strong offline behavior depends on trusted selection, not generic regularization

- Minimal evidence already available:
  - SSAR cached IQL-qv seed0 100k final/best: 92.44 / 100.98.
  - cheap SSAR without IQL selection 100k final/best: 25.48 / 30.34.
  - ReBRAC-lite 100k final/best: 36.54 / 54.36.
- Required caution:
  - report SSAR cached result as a high-variance upper anchor, not a stable mean.

### Claim 3: Fixed offline trust does not automatically transfer online

- Minimal evidence already available:
  - TD3+BC decay: offline 22.43 -> online final 39.64.
  - ATLAS decay: offline 45.29 -> online final 37.97.
  - ATLAS fixed: offline 45.29 -> online final 29.22.
- Failure definition:
  - ATLAS online final is below its own offline endpoint and below TD3+BC online final.

### Claim 4: Constraint release is the next decisive C-line test

- Minimal experiment:
  - clean 2x2 `{TD3+BC, ATLAS} x {fixed, linear-release}` under one O2O protocol.
- Outcome interpretation:
  - if ATLAS release improves online final, the project has a positive fix;
  - if it does not, the project has a stronger negative finding that teacher-label trust is an offline stabilizer but not a sufficient O2O mechanism.

### Claim 5: Group-level paper still needs A/B rows

- A-line should provide one value-conservative result row under the shared protocol.
- B-line should provide one non-conservative or weakly constrained online contrast row.
- Without A/B, final title should shift toward C-line mechanism; with A/B, keep the original family-comparison framing.

## Experiment Handoff Inputs

P0:

1. Do not run broad baseline sweeps.
2. Fill A/B-line rows under the shared table.
3. If Tree wants one more C-line run, run the clean release 2x2 or the missing half of it, not PRDC/A2PR expansion.
4. Write the paper as a family-comparison scaffold plus C-line mechanism evidence.

P1:

1. Add one verified related-work paragraph.
2. Add a cost/time table: full SSAR IQL-qv preselection versus ATLAS post-cache selector training/reuse.
3. Add one extended O2O horizon only if the current 10k slice is challenged.

## Compute & Timeline Estimate

- No broad new sweep.
- One release-condition C-line run is acceptable if the AutoDL instance is retained.
- A/B-line rows should be one env, one seed, controlled protocol first.
- Final paper writing should wait for A/B results, but the current draft can already be rewritten around the refined claims.
