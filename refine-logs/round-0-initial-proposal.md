# Research Proposal: When Trusted Constraints Help and Hurt in Offline-to-Online RL

## Problem Anchor

- Bottom-line problem: The course project needs a defensible answer to how different constraint families affect offline-to-online RL under low-quality D4RL replay data, without turning the whole project into an overclaimed ATLAS-only algorithm paper.
- Must-solve bottleneck: Low-quality replay contains many suboptimal or misleading actions. Weak behavior regularization underuses useful data, but strong teacher/action constraints can over-constrain online adaptation. SSAR exposes a strong IQL-qv trusted-action signal, but that signal is expensive and not automatically safe to carry into online fine-tuning.
- Non-goals: Do not claim ATLAS is SOTA; do not run a 6-env x 3-seed benchmark before the mechanism is clear; do not vendor no-license third-party code; do not make broad PRDC/A2PR expansion the main contribution.
- Constraints: Course-project timeline; AutoDL budget should stay exploration-first; one or two replay environments and selective seeds are acceptable; A/B-line teammate results are still pending; C-line evidence is mostly smoke-to-mechanism validation, not final benchmark proof.
- Success condition: A coherent workshop-style project where A/B/C lines compare value conservatism, non-conservative contrast, and policy/trusted-action regularization under one protocol; C-line contributes a clear mechanism finding about trusted-action selection and online constraint release; remaining TODOs are narrow enough for teammates to fill.

## Technical Gap

Existing offline-to-online work already studies whether conservatism helps or hurts fine-tuning. A course project that only says "conservatism matters" would be weak. The useful gap in our current evidence is more specific:

1. In low-quality replay data, generic behavior regularization is not enough.
2. SSAR can be very strong, but its strength appears tied to expensive IQL-qv trusted-action selection rather than ordinary BC regularization.
3. Distilling that trusted-action signal into ATLAS improves offline learning and survives a seed check plus a second replay environment, but the same constraint hurts the first online fine-tuning slice.
4. Therefore the real paper question is not "did we invent a stronger offline RL algorithm?" but "which constraints stabilize offline learning, and which must be released or gated during online adaptation?"

Naive fixes are insufficient. More baselines would consume budget without clarifying the mechanism. More seeds before a mechanism-level claim is stable would mostly measure variance. A larger ATLAS module stack would also blur the paper: the current data says the key is label quality and online release, not model capacity alone.

## Method Thesis

- One-sentence thesis: Under low-quality replay data, trusted-action selection is a powerful offline constraint, but offline-useful constraints must be relaxed or gated during online fine-tuning; ATLAS is a lightweight probe for this constraint-transfer failure mode.
- Why this is the smallest adequate intervention: We reuse TD3+BC/ReBRAC/SSAR/IQL/CQL anchors and add only a cached-teacher trust selector plus minimal online release ablations.
- Why this route is timely: It connects recent behavior-regularized offline RL and state-adaptive regularization to the practical offline-to-online question of when constraints should be trusted, amortized, or released.

## Contribution Focus

- Dominant contribution: A constraint-transfer diagnosis for low-quality offline-to-online RL: trusted-action constraints can strongly improve offline learning, but naive persistence of the same constraints can block online adaptation.
- Optional supporting contribution: ATLAS as a lightweight post-cache distillation probe showing that aligned SSAR/IQL trust labels matter; shuffled labels collapse, and the signal transfers beyond one hopper seed.
- Explicit non-contributions: ATLAS is not claimed as SOTA; PRDC/A2PR are not treated as failed papers from one smoke run; broad benchmark leadership is not claimed.

## Proposed Method

### Complexity Budget

- Frozen / reused backbone:
  - D4RL MuJoCo replay environments.
  - Shared TD3+BC-style training/evaluation utilities.
  - Existing CQL/IQL compact baselines.
  - External SSAR source for teacher/cache reference only.
- New trainable components:
  - ATLAS selector `g_phi(s, a) -> [0, 1]`, trained from cached SSAR/IQL-qv trusted-action labels.
  - Optional release gate or schedule for online fine-tuning, only if time permits.
- Tempting additions intentionally not used:
  - New world model, diffusion policy, offline data relabeler, or multi-module planner.
  - Large hyperparameter sweep.
  - Full PRDC/A2PR source migration into the public repo.

### System Overview

```text
low-quality D4RL replay
        |
        v
offline pretraining under three constraint families
        |
        +-- A-line: value conservatism          -> CQL / Cal-QL row
        +-- B-line: non-conservative contrast   -> PPO / SAC / vanilla TD3-style row
        +-- C-line: behavior/trusted regularization
              |
              +-- TD3+BC / ReBRAC anchors
              +-- SSAR/IQL-qv teacher cache
              +-- ATLAS selector distills trust labels
        |
        v
minimal online fine-tuning slice
        |
        v
compare offline score, online delta, final/best score, and constraint failure modes
```

### Core Mechanism

- Input / output: ATLAS consumes `(state, action)` pairs and outputs a trust score used to weight TD3+BC-style behavior regularization.
- Architecture or policy: Keep the selector small and separate from the actor/critic. Treat it as an amortized teacher-label model, not as a new policy class.
- Training signal / loss: Supervise the selector with SSAR/IQL-qv trusted-action labels or scores exported from the cached teacher. The essential ablation is aligned labels versus shuffled labels.
- Why this is the main novelty: It isolates whether SSAR's useful signal is the action-trust assignment itself. The shuffled-label collapse shows that arbitrary weighting is not enough.

### Optional Supporting Component

- Only include if truly necessary: online constraint release.
- Input / output: a schedule or gate controlling how much ATLAS/BC regularization applies after online interaction begins.
- Training signal / loss: no new model unless the simple schedule fails; first test fixed versus decayed constraint and possibly a state/action confidence threshold.
- Why it does not create contribution sprawl: It directly tests the observed O2O failure mode rather than adding a separate algorithmic story.

### Modern Primitive Usage

- Which RL-era primitive is used: teacher distillation from an IQL-qv / SSAR value-based selector.
- Exact role in the pipeline: SSAR/IQL-qv acts as a teacher for which dataset actions should be trusted; ATLAS amortizes this signal into a cheap selector.
- Why it is more natural than an old-school alternative: The failed return-ranked, online Q-gap, and behavior-consistency selectors suggest that simple heuristics do not recover the right action-level signal.

### Integration into Base Generator / Downstream Pipeline

ATLAS attaches only to the C-line TD3+BC-style training loop. It does not change the shared evaluation protocol. During offline training, the actor loss receives a trust-weighted BC component. During online fine-tuning, the key experimental choice is whether this trust-weighted component remains fixed, decays, or is gated.

### Training Plan

1. Use cached SSAR/IQL-qv teacher outputs where available.
2. Train ATLAS selector on exported trust labels.
3. Run offline weighted TD3+BC with the selector on `hopper-medium-replay-v2` and `walker2d-medium-replay-v2`.
4. Run a minimal online slice from a fixed offline checkpoint.
5. If the online slice remains negative, do not claim O2O improvement; instead report the constraint-transfer failure and test one simple release schedule only if time permits.

### Failure Modes and Diagnostics

- Failure mode: ATLAS improves offline but degrades online.
  - How to detect: offline final is higher than TD3+BC, but online final falls below its offline endpoint or below TD3+BC online.
  - Fallback or mitigation: frame as constraint-transfer failure; test decay/release before adding capacity.
- Failure mode: ATLAS only works on one seed or one environment.
  - How to detect: seed1 or walker check collapses.
  - Fallback or mitigation: reduce claim to hopper-only mechanism evidence and avoid generalization language.
- Failure mode: gains come from arbitrary weighting.
  - How to detect: shuffled-label or constant-label control performs similarly.
  - Fallback or mitigation: current shuffled-label result already rejects this for hopper 50k.
- Failure mode: SSAR teacher cost dominates.
  - How to detect: full IQL-qv preselection time exceeds selector training savings.
  - Fallback or mitigation: claim ATLAS as post-cache amortization, not from-scratch cheaper training.

### Novelty and Elegance Argument

The closest conceptual work already includes CQL/Cal-QL, ReBRAC, PRDC, A2PR, SSAR, and IQL. Therefore the paper should avoid broad novelty claims. The focused novelty is the empirical and mechanistic connection between trusted-action selection and online constraint transfer:

- SSAR indicates that the right dataset actions matter.
- ATLAS shows that aligned trust labels are useful and reusable after caching.
- O2O results show that offline-useful trust can become online-harmful if not released.

This is a cleaner paper than an ATLAS-only SOTA claim because it explains both the positive offline result and the negative online result under one mechanism.

## Claim-Driven Validation Sketch

### Claim 1: Trusted-action selection is a key offline mechanism on low-quality replay

- Minimal experiment: Compare TD3+BC, ReBRAC-lite, SSAR full/cached, cheap SSAR without IQL selection, ATLAS, and ATLAS shuffled-label control on `hopper-medium-replay-v2`.
- Baselines / ablations: TD3+BC, ReBRAC-lite, IQL compact, CQL compact, no-IQL SSAR, shuffled labels.
- Metric: final and best normalized D4RL score; label-control gap.
- Expected evidence: aligned ATLAS beats simple anchors and shuffled labels collapse; no-IQL SSAR collapses relative to full SSAR.

### Claim 2: Offline trusted constraints do not automatically improve online fine-tuning

- Minimal experiment: 50k offline + 10k online slice comparing TD3+BC decay/fixed and ATLAS decay/fixed.
- Baselines / ablations: TD3+BC O2O decay/fixed; ATLAS O2O decay/fixed; optional one release schedule.
- Metric: online final, online best, and online delta from offline checkpoint.
- Expected evidence: current evidence already shows TD3+BC improves online while ATLAS loses offline advantage; if a release schedule fixes this, it becomes a stronger method angle.

### Claim 3: The final group paper compares constraint families, not only C-line

- Minimal experiment: Add one A-line value-conservative row and one B-line non-conservative row under the same environment/protocol.
- Baselines / ablations: CQL/Cal-QL for A; PPO/SAC/vanilla TD3-style online for B.
- Metric: comparable final/best normalized score and online delta.
- Expected evidence: enough to position C-line results inside the original course project question.

## Experiment Handoff Inputs

- Must-prove claims:
  - trusted labels matter;
  - offline gains can fail to transfer online;
  - group-level family comparison remains the final paper backbone.
- Must-run ablations:
  - already done: shuffled ATLAS labels, no-IQL SSAR, TD3+BC O2O fixed/decay;
  - optional P0 if time: one ATLAS label-use/release ablation.
- Critical datasets / metrics:
  - `hopper-medium-replay-v2` as the diagnostic environment;
  - `walker2d-medium-replay-v2` as second C-line sanity check;
  - normalized D4RL final/best, online delta, eval episodes, wall-clock/cost.
- Highest-risk assumptions:
  - SSAR cached results have high variance and should not be treated as stable final mean;
  - ATLAS is post-cache cheaper, not necessarily cheaper from scratch;
  - A/B-line results may change the final emphasis.

## Compute & Timeline Estimate

- Estimated GPU-hours:
  - No broad new sweep by default.
  - One cheap ATLAS release/label ablation: roughly one 50k/100k C-line run after labels, expected under one retained AutoDL session if environment remains intact.
  - A/B-line rows should each be one controlled smoke first, not multi-seed.
- Data / annotation cost: no human annotation; D4RL data and teacher-label cache only.
- Timeline:
  - Immediate: use this refined plan to rewrite paper framing.
  - Next group sync: ask teammates to fill A/B-line rows under the exact table schema.
  - After A/B merge: decide whether one additional ATLAS release ablation is necessary for the final story.
