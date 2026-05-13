# Workshop Draft: Conservative and Regularized Offline-to-Online RL under Low-Quality Replay Data

Status: living workshop-style draft  
Date: 2026-05-13  
Scope: C-line evidence is filled in; A/B-line results are placeholders.

## Draft Control Notes

This is not a final paper. It is a shared scaffold for the course project.

Current framing:

- Main project question: how do different forms of conservatism and regularization affect offline-to-online reinforcement learning under low-quality replay data?
- A-line placeholder: value conservatism, e.g. CQL / Cal-QL.
- B-line placeholder: normal or non-conservative contrast, e.g. PPO / SAC / vanilla TD3-style online fine-tuning.
- C-line filled: policy / behavior regularization, trusted-action selection, SSAR, and ATLAS.

Claim boundary:

- We can claim a mechanism finding: trusted-action selection is important for behavior regularization on low-quality replay data.
- We can claim ATLAS is a useful C-line mechanism probe and post-cache distillation attempt.
- We cannot claim ATLAS is SOTA.
- We cannot claim ATLAS improves offline-to-online fine-tuning under the current evidence.

Citation policy:

- Citations are left as `[CITATION NEEDED: ...]` placeholders until verified programmatically.
- Do not invent BibTeX entries from memory.

## Title Candidates

1. Conservative or Constrained? A Small Empirical Study of Offline-to-Online RL under Low-Quality Replay Data
2. When Regularization Helps and Hurts: Offline-to-Online RL on Low-Quality Replay Data
3. Trusted Actions Matter: Mechanism Evidence for Behavior-Regularized Offline RL

## Abstract

Offline-to-online reinforcement learning promises to reuse logged experience before interacting with the environment, but the benefit of offline pretraining depends strongly on the quality of the replay data and on how the learning algorithm constrains its policy. This project studies a focused D4RL MuJoCo replay setting and compares three design families: value conservatism, normal or weakly regularized online learning, and policy/behavior regularization. Our current C-line results show that naive behavior regularization is insufficient on `hopper-medium-replay-v2`: TD3+BC reaches only 22.43 normalized score after 50k offline steps, while a ReBRAC-style baseline reaches 34.48 and SSAR reaches 38.56 final / 43.97 best under the same 50k screen. Further mechanism experiments suggest that SSAR's IQL-qv trusted-action selection is a key driver: removing trusted selection collapses performance, and an aligned teacher-label selector, ATLAS, reaches 69.97 and 68.11 final score on hopper seed0/seed1 at 100k steps, while shuffled labels collapse to 18.78. However, a minimal offline-to-online slice shows a limitation: TD3+BC improves from 22.43 to 39.64 after 10k online steps, whereas ATLAS loses its offline advantage under the current online regularization. These results support a cautious conclusion: trusted-action regularization can substantially improve offline learning under low-quality data, but carrying the same constraint into online adaptation can over-constrain policy improvement.

TODO-A: add value-conservatism result summary.  
TODO-B: add non-conservative online contrast result summary.  
TODO-CHECK: shorten to venue-specific abstract limit after A/B results arrive.

## 1. Introduction

Offline-to-online reinforcement learning is attractive because it can bootstrap an agent from existing replay data and then continue improving through interaction. In practice, however, low-quality offline replay data creates a tension. A learner that trusts the dataset too much can inherit suboptimal behavior. A learner that ignores the dataset can waste online samples or become unstable. The central question is therefore not simply whether offline pretraining helps, but which type of constraint helps during offline learning and which constraints should be relaxed during online fine-tuning.

This course project studies that question through three complementary algorithm families. The A-line studies value conservatism, such as conservative Q-learning methods. The B-line provides a normal or non-conservative contrast, such as PPO, SAC, or vanilla TD3-style online fine-tuning. The C-line studies policy and behavior regularization, including TD3+BC, ReBRAC-style regularization, SSAR, and our ATLAS mechanism probe.

The current draft is written before A/B-line results are complete. We therefore use the C-line evidence to establish the main empirical tension. On `hopper-medium-replay-v2`, simple TD3+BC is weak, stronger behavior regularization helps, and SSAR is a strong modern teacher. But the strongest part of SSAR appears to be not merely "more regularization"; rather, it is the selection of trusted dataset actions through IQL-qv information. This motivates ATLAS, a lightweight selector that distills trusted-action labels from a cached SSAR/IQL-qv teacher into a reusable `(state, action) -> trust score` model.

Our results give a mixed but useful picture. ATLAS improves offline performance and passes a seed check, a shuffled-label control, and a second-environment check. At the same time, its offline advantage does not automatically transfer to a short online fine-tuning slice. This negative result is important for the project: it suggests that constraints that are useful for offline stabilization may need to be decayed, gated, or released during online adaptation.

Our intended contributions are:

1. A small but controlled empirical comparison of algorithm families for low-quality offline-to-online RL.
2. Mechanism evidence that trusted-action selection is a high-value component of behavior-regularized offline RL.
3. A lightweight ATLAS probe showing that teacher-aligned trust labels can improve offline learning, while shuffled labels fail.
4. A cautionary offline-to-online finding: teacher-label regularization can over-constrain online adaptation if it is carried forward naively.

## 2. Background and Setup

### Offline-to-Online RL

In offline-to-online RL, the agent first learns from a fixed offline dataset and then improves through online interaction with the environment. The offline stage can improve sample efficiency, but it can also introduce bias if the offline data is low quality or narrow. This project focuses on replay-style D4RL MuJoCo environments, where the dataset contains broad but noisy experience rather than expert demonstrations. [CITATION NEEDED: D4RL]

### Three Design Families

We organize the project around three families.

Value conservatism attempts to avoid overestimating out-of-distribution actions by pessimistically regularizing value learning. This family includes CQL-style methods and related conservative value algorithms. [CITATION NEEDED: CQL] [CITATION NEEDED: Cal-QL]

Normal or non-conservative online learning acts as a contrast. PPO, SAC, or vanilla TD3-style online fine-tuning do not directly encode offline pessimism or trusted-action constraints. This line is useful even when raw scores are lower, because it shows what happens when explicit conservatism or behavior regularization is removed. [CITATION NEEDED: PPO] [CITATION NEEDED: SAC] [CITATION NEEDED: TD3]

Policy and behavior regularization constrains policy updates toward dataset actions or trusted subsets of dataset actions. TD3+BC is a simple anchor, ReBRAC improves the regularization design, and SSAR introduces a stronger trusted-action mechanism. [CITATION NEEDED: TD3+BC] [CITATION NEEDED: ReBRAC] [CITATION NEEDED: SSAR]

### Evaluation Protocol

The current shared protocol uses normalized D4RL score:

```text
normalized_score = env.get_normalized_score(raw_return) * 100
```

The minimum comparable record for each run is:

- method;
- method family;
- environment;
- seed;
- offline steps;
- online steps if any;
- evaluation episodes;
- final normalized score;
- best normalized score;
- curve or log path;
- one-sentence interpretation.

## 3. C-Line Mechanism: Trusted-Action Selection

The C-line began as a policy-regularization track. Initial smoke experiments showed that plain TD3+BC was a weak anchor on `hopper-medium-replay-v2`, while ReBRAC-lite and SSAR were stronger. The key question became: what exactly is SSAR buying?

The mechanism evidence points to trusted-action selection. A cheap SSAR variant without IQL-based trusted selection collapses relative to full SSAR. Naive substitutes, such as return-ranked trust, online Q-gap trust, and behavior-consistency trust, do not reliably solve the problem. This suggests that the teacher signal is not merely "regularize more"; it is selecting which dataset actions should be trusted.

ATLAS is our C-line probe for this mechanism. It exports SSAR/IQL-qv teacher labels from a cached teacher and trains a selector that maps `(state, action)` pairs to a trust score. The selector is then used for weighted TD3+BC-style learning. The goal is not to replace SSAR as a new SOTA algorithm. The goal is to test whether the trusted-action information can be distilled into a cheaper reusable component once the expensive teacher cache exists.

## 4. Experiments

### 4.1 Experimental Questions

Q1. Under low-quality replay data, do policy/behavior regularization methods outperform a weak TD3+BC anchor?

Q2. Is SSAR's gain plausibly tied to trusted-action selection rather than generic regularization?

Q3. Can ATLAS distill the trusted-action signal into a reusable selector?

Q4. Does the offline advantage persist during online fine-tuning?

TODO-A: add A-line question after CQL/Cal-QL results arrive.  
TODO-B: add B-line question after PPO/SAC/vanilla online results arrive.

### 4.2 Main Offline and Mechanism Results

Environment: `hopper-medium-replay-v2`.

| Method | Seed | Steps | Eval eps | Final | Best | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| TD3+BC | 0 | 50k | 5 | 22.43 | 22.43 | weak anchor |
| ReBRAC-lite | 0 | 50k | 5 | 34.48 | 34.48 | stronger regularized baseline |
| PRDC official | 0 | 50k | 5 | 23.54 | 23.54 | no clear uplift in smoke |
| A2PR official | 0 | 50k | 5 | 22.31 | 22.81 | no clear uplift in smoke |
| SSAR full IQL-qv | 0 | 50k | 5 | 38.56 | 43.97 | strongest 50k modern baseline |
| SSAR cached IQL-qv | 0 | 100k | 5 | 92.44 | 100.98 | high-variance upper anchor |
| SSAR full IQL-qv | 1 | 100k | 20 | 60.88 | 99.22 | near-100 spike repeats; final unstable |
| IQL compact | 0 | 100k | 5 | 45.27 | 81.28 | strong value-style baseline; implementation-sensitive |
| CQL compact | 0 | 50k | 5 | 39.81 | 39.81 | conservative value anchor |
| ATLAS | 0 | 100k | 5 | 69.97 | 69.97 | positive offline signal |
| ATLAS | 1 | 100k | 20 | 68.11 | 68.11 | seed stability check |
| ATLAS shuffled labels | 0 | 50k | 5 | 18.78 | 19.35 | label alignment matters |

Interpretation. The 50k screen suggests that stronger regularization can outperform TD3+BC, but not every recent method helps in this setup. SSAR is the strongest 50k modern baseline, while ATLAS achieves a stronger 100k offline endpoint than the simple baselines. The shuffled-label control is important: it shows that ATLAS is not merely benefiting from arbitrary weighting; aligned teacher labels matter.

### 4.3 Second Replay Environment

Environment: `walker2d-medium-replay-v2`.

| Method | Seed | Steps | Eval eps | Final | Best | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| SSAR full IQL-qv | 0 | 100k | 10 | 94.28 | 94.60 | strong teacher anchor |
| ATLAS | 0 | 100k | 10 | 71.26 | 77.86 | signal survives, but gap remains |

Interpretation. ATLAS is not purely a hopper artifact, but the gap to SSAR remains large. This supports the more cautious claim that ATLAS partially distills trusted-action information, rather than replacing the full teacher mechanism.

### 4.4 Minimal Offline-to-Online Slice

Environment: `hopper-medium-replay-v2`, seed0, 50k offline + 10k online.

| Method | Offline final | Online final | Online best | Interpretation |
|---|---:|---:|---:|---|
| TD3+BC decay | 22.43 | 39.64 | 39.64 | online fine-tuning works; decay slightly better than fixed |
| TD3+BC fixed | 22.43 | 36.78 | 36.78 | online improves but less than decay |
| ATLAS decay | 45.29 | 37.97 | 38.26 | offline advantage does not persist |
| ATLAS fixed | 45.29 | 29.22 | 32.27 | fixed teacher regularization over-constrains online adaptation |
| ATLAS decay, online fraction 0.25 | 45.29 | 31.46 | 43.73 | lower online replay fraction does not fix final degradation |

Interpretation. This slice keeps the project aligned with offline-to-online RL. It also weakens any over-strong ATLAS narrative. ATLAS improves the offline endpoint, but under the current online runner its teacher-label constraint does not translate into better online fine-tuning. This suggests that trusted-action constraints may need a release or gating mechanism during online adaptation.

## 5. Preliminary Cross-Line Table

TODO-A and TODO-B rows should be filled when teammates finish their runs.

| Track | Family | Representative method | Env | Seed | Offline steps | Online steps | Final | Best | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| A | Value conservatism | CQL / Cal-QL | `hopper-medium-replay-v2` | 0 | TODO-A | TODO-A | TODO-A | TODO-A | waiting |
| B | Non-conservative contrast | PPO / SAC / vanilla TD3-style | `hopper-medium-replay-v2` | 0 | TODO-B | TODO-B | TODO-B | TODO-B | waiting |
| C | Policy regularization | TD3+BC O2O decay | `hopper-medium-replay-v2` | 0 | 50k | 10k | 39.64 | 39.64 | done |
| C | Trusted-action selection | ATLAS offline | `hopper-medium-replay-v2` | 0 | 100k | 0 | 69.97 | 69.97 | done |
| C | Trusted-action selection | ATLAS offline | `hopper-medium-replay-v2` | 1 | 100k | 0 | 68.11 | 68.11 | done |

## 6. Discussion

The current evidence suggests a useful distinction between offline stabilization and online adaptability. Trusted-action selection can stabilize learning from low-quality replay data, but the same constraint can become harmful once the agent has access to fresh online data. This is consistent with the ATLAS O2O slice: ATLAS begins from a stronger offline policy than TD3+BC, yet its online fine-tuning score drops below its offline endpoint, while TD3+BC improves substantially.

This result argues against a simple "more conservatism is always better" story. Instead, it suggests a timed or state-dependent view: conservative or trusted-action constraints may be useful during offline pretraining, but online fine-tuning may require decaying, gating, or releasing the constraint. LJY's Gate-aware release idea can be positioned here as a possible follow-up mechanism, not as a replacement for the current project framing.

## 7. Limitations

The current draft has several limitations.

First, many C-line results are exploration-stage smoke experiments rather than full multi-seed benchmarks. The goal was to identify high-value mechanisms under budget constraints, not to produce a final benchmark suite.

Second, SSAR and ATLAS depend on an expensive IQL-qv teacher cache. ATLAS should therefore be described as post-cache or amortized cheaper, not as cheaper from scratch.

Third, the minimal online fine-tuning slice is intentionally short. It demonstrates that online fine-tuning is wired up and exposes an ATLAS limitation, but it is not a final O2O benchmark.

Fourth, A-line and B-line results are not yet merged. The final project claim should wait until those tracks provide comparable results.

## 8. Figure Plan

Figure 1: Project schematic. Three families under low-quality replay data: value conservatism, non-conservative contrast, and policy regularization / trusted-action selection.

Figure 2: C-line offline result bar chart on `hopper-medium-replay-v2`: TD3+BC, ReBRAC-lite, SSAR, CQL, ATLAS, shuffled-label ATLAS.

Figure 3: Offline-to-online curve: TD3+BC decay/fixed vs ATLAS decay/fixed.

Figure 4, optional: SSAR vs ATLAS on `walker2d-medium-replay-v2`.

## 9. What Remains Before a Full Draft

P0:

- Fill A-line result row.
- Fill B-line result row.
- Generate the three core figures.
- Verify all citations and create a real bibliography.
- Convert this Markdown draft to LaTeX if needed.

P1:

- Add one ATLAS label-usage ablation only if the final story needs more mechanism evidence.
- Add cost/time table: SSAR full IQL-qv preselection vs ATLAS post-cache selector training / reuse.
- Add appendix with exact command lines and result folders.

## Appendix A. Current Artifact Index

Primary evidence files:

- `refine-logs/C_LINE_MERGE_PACKAGE_20260513.md`
- `refine-logs/C_LINE_RESULTS_TABLE_20260513.md`
- `refine-logs/remote-results/o2o_minimal_20260513/README.md`
- `refine-logs/NOVELTY_CHECK_20260513.md`
- `refine-logs/EXPERIMENT_TRACKER.md`

Key executable entrypoints:

- `algorithms/td3_bc_o2o.py`
- `scripts/run_o2o_minimal.sh`
- `scripts/export_iql_trust_labels.py`
- `scripts/train_atlas_selector.py`
- `scripts/make_label_control.py`
