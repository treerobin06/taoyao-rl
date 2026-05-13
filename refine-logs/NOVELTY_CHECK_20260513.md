# Novelty Check: Original Mainline vs ATLAS

Date: 2026-05-13
Scope: C-line offline-to-online RL project, current ATLAS/SSAR exploration results.

## Executive Verdict

The original course mainline is still the safest project direction:

> Low-quality offline data under D4RL MuJoCo v2: how do different conservatism / regularization designs affect offline-to-online fine-tuning efficiency?

ATLAS should not replace this mainline as the whole project. ATLAS is best positioned as a mechanism-driven extension:

> SSAR is strong but expensive because it relies on IQL-qv trusted action selection. ATLAS tests whether this trusted-action signal can be distilled into a lightweight selector and reused by a simpler TD3+BC-style learner.

Recommended project structure:

1. Main paper body: empirical study of conservatism / regularization under low-quality offline data and online fine-tuning.
2. C-line contribution: mechanism diagnosis of SSAR plus ATLAS as a lightweight amortized trusted-action selector.
3. Claim discipline: do not claim ATLAS is a new SOTA algorithm unless it beats SSAR or strong public baselines across more seeds/envs.

## Claim-Level Novelty Scores

| Candidate claim | Novelty | Course value | Risk | Verdict |
|---|---:|---:|---:|---|
| Compare offline-to-online transfer across data quality and conservative algorithms | 3/10 | 8/10 | low | Keep as project backbone |
| Regularization decay during online fine-tuning | 4/10 | 7/10 | medium | Good simple ablation if time permits |
| "SSAR works mainly because of IQL-qv trusted action selection" | 5/10 | 8/10 | medium | Good mechanism claim |
| ATLAS: distill SSAR/IQL trusted labels into a lightweight action selector | 5-6/10 | 8/10 | medium-high | Keep as C-line extension |
| ATLAS as SOTA replacement for SSAR | 2/10 currently | 4/10 | high | Do not claim now |

## Why The Original Mainline Still Matters

The original PDF defines the project as an offline pretraining -> online fine-tuning -> ablation -> strategy optimization pipeline. It explicitly asks how conservatism design affects fine-tuning under low-quality offline data, with metrics such as offline normalized score, online convergence speed, final asymptotic performance, multi-seed variance, IQM, and confidence intervals.

This is not a "must invent one algorithm" project. It is an empirical course project. The right contribution standard is:

- use recent and representative methods;
- keep baselines comparable;
- explain when conservatism helps or blocks online fine-tuning;
- provide enough ablation to support the explanation.

So the original mainline is meaningful for the course. It is not highly novel as a standalone research-paper topic unless we add a sharper mechanism or method component.

## Closest Prior Work And Implication

### Offline-to-online conservatism is already a known topic

Cal-QL studies offline pretraining for efficient online fine-tuning, arguing that existing offline RL initializations can fine-tune poorly and proposing calibrated conservative Q-learning.

Adaptive BC regularization directly studies the tension that BC-style constraints stabilize offline-to-online transfer but can slow fine-tuning by keeping the policy close to behavior data.

Implication: "conservatism affects online fine-tuning" is not a new claim. It is a valid empirical axis, but it needs a clear dataset/method comparison and not overclaimed as novel.

### Policy / behavior regularization is a crowded family

ReBRAC shows that a minimal TD3+BC-style algorithm with careful design choices can be very strong across D4RL and offline-to-online settings.

PRDC argues that regularizing toward nearest dataset state-action pairs is less overly conservative than constraining toward the exact behavior action.

A2PR specifically targets suboptimal data by selecting high-advantage actions from a VAE-augmented behavior policy.

SSAR is even closer to our current ATLAS direction: it introduces state-adaptive regularization and selectively applies constraints on high-quality actions, and it reports strong offline and offline-to-online D4RL performance.

Implication: a generic "adaptive regularization / high-quality action selection" claim is not enough. ATLAS must be framed narrowly as amortizing a specific expensive SSAR/IQL trusted-action signal.

### Distillation / value-weighted imitation also exists

IQL extracts policies through advantage-weighted behavior cloning. Recent offline behavior distillation work also uses action-value weighted objectives and D4RL experiments.

Implication: "we weight BC using a learned score" is not novel by itself. The defensible novelty is the specific teacher signal, mechanism diagnosis, and empirical evidence that label alignment matters.

## What Current Results Support

Current local evidence supports these statements:

1. SSAR is a strong but expensive anchor.
   - Hopper seed0 cached SSAR: final 92.44, best 100.98.
   - Hopper seed1 eval20 SSAR: final 60.88, best 99.22.
   - Walker seed0 SSAR: final 94.28, best 94.60.
   - Interpretation: SSAR can spike very high, but hopper tail/final score is unstable.

2. The IQL-qv trusted-action signal is important.
   - Cheap SSAR without IQL action selection: final 25.48, best 30.34.
   - IQL compact baseline: final 45.27, best 81.28.
   - CQL compact baseline: final 39.81.
   - Interpretation: SSAR should not be compared only against weak TD3+BC. IQL is a strong part of the story.

3. ATLAS is not just random weighted BC.
   - Hopper ATLAS seed0 100k: final 69.97.
   - Hopper ATLAS seed1 eval20 100k: final 68.11.
   - Shuffled-label ablation: final 18.78 vs aligned 45.29 at 50k.
   - Walker ATLAS seed0: final 71.26, best 77.86.
   - Interpretation: the teacher-label alignment matters, and the method transfers beyond one hopper seed.

4. ATLAS is still below SSAR on walker and does not reproduce SSAR's near-100 spikes.
   - Interpretation: ATLAS is a promising lightweight approximation, not a finished SOTA method.

## Recommended Final Project Framing

Use this as the group-level title/framing:

> Conservatism Under Low-Quality Offline Data: An Empirical Study of Offline-to-Online RL Transfer and Lightweight Trusted-Action Selection

Use this as the C-line specific claim:

> In low-quality replay datasets, fixed behavior regularization is often insufficient. SSAR indicates that trusted-action selection is a key mechanism, but its IQL-qv preselection is expensive. We propose ATLAS as a lightweight distillation of the trusted-action signal, and show preliminary evidence that aligned teacher labels produce stable gains while shuffled labels collapse.

Do not use:

> ATLAS is a new state-of-the-art offline RL algorithm.

## What To Do Next

### Minimum path to a defensible course project

1. Keep the original empirical backbone.
   - Environment focus: `hopper-medium-replay-v2` and `walker2d-medium-replay-v2`.
   - Baselines: TD3+BC / ReBRAC-lite / IQL / CQL / SSAR / ATLAS.
   - Report 1 seed for exploration, add seed only where a claim depends on it.

2. Add one online fine-tuning slice.
   - The original project is offline-to-online, so we need at least one clear online fine-tuning curve.
   - Do not run a 6-env x 3-seed sweep.
   - Run one or two representative methods from offline checkpoint into online fine-tuning, then compare sample efficiency.

3. Run only 1-2 cheap ATLAS ablations.
   - soft vs hard teacher labels;
   - `label_min_weight`;
   - selector capacity.
   Stop if these do not improve or clarify the mechanism.

4. Write results conservatively.
   - Main conclusion: multi-seed broad baseline sweeps are too costly in exploration; mechanism-first smoke tests are better.
   - Empirical finding: SSAR/IQL trusted-action selection is a high-value mechanism.
   - Method finding: ATLAS can partially amortize the signal but is not yet a full SSAR replacement.

### If time is short

Do not chase more algorithms. Finish:

- one clean result table;
- one curve figure;
- one mechanism ablation figure;
- one cost/time discussion;
- one limitations paragraph.

## Source Anchors

- Cal-QL: https://openreview.net/forum?id=Ye9feH28TF
- Adaptive BC Regularization: https://arxiv.org/abs/2210.13846
- ReBRAC: https://arxiv.org/abs/2305.09836
- PRDC: https://proceedings.mlr.press/v202/ran23a.html
- A2PR: https://icml.cc/virtual/2024/poster/34532
- SSAR: https://icml.cc/virtual/2025/poster/44640
- IQL: https://arxiv.org/abs/2110.06169
- Offline Behavior Distillation: https://arxiv.org/abs/2410.22728
- Value-learning bottleneck discussion: https://arxiv.org/abs/2406.09329
