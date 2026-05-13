# C-Line Merge Package

Date: 2026-05-13  
Purpose: merge-ready summary for the final group report.

## What C-Line Has Completed

C-line focuses on policy / behavior regularization under low-quality replay data. The current work is enough for a course-project C-line contribution and should be merged with A/B-line results rather than expanded into a broad standalone sweep.

Completed evidence:

- C-line baseline screen on `hopper-medium-replay-v2`;
- official-source smoke for PRDC / A2PR / SSAR;
- SSAR mechanism check showing IQL-qv trusted action selection is critical;
- public IQL / CQL value baselines;
- ATLAS teacher-label selector;
- shuffled-label ablation showing teacher-label alignment matters;
- second replay environment check on `walker2d-medium-replay-v2`;
- minimal offline-to-online fine-tuning check.

## Main C-Line Claim

Recommended wording:

> Under low-quality replay data, action selection for behavior regularization matters. SSAR shows that IQL-qv trusted action selection is a powerful but expensive mechanism. ATLAS partially distills this trusted-action signal and improves offline performance, but the first online fine-tuning slice suggests that teacher-label regularization can over-constrain online adaptation.

Avoid stronger wording:

> ATLAS is a new SOTA offline RL algorithm.

> ATLAS improves offline-to-online fine-tuning.

The current evidence does not support those.

## Core Results To Merge

### Offline / Mechanism

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
| IQL compact | 0 | 100k | 5 | 45.27 | 81.28 | strong value baseline |
| CQL compact | 0 | 50k | 5 | 39.81 | 39.81 | conservative value anchor |
| ATLAS | 0 | 100k | 5 | 69.97 | 69.97 | positive offline signal |
| ATLAS | 1 | 100k | 20 | 68.11 | 68.11 | seed stability check |
| ATLAS shuffled labels | 0 | 50k | 5 | 18.78 | 19.35 | label alignment matters |

### Second Environment

Environment: `walker2d-medium-replay-v2`.

| Method | Seed | Steps | Eval eps | Final | Best | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| SSAR full IQL-qv | 0 | 100k | 10 | 94.28 | 94.60 | strong teacher anchor |
| ATLAS | 0 | 100k | 10 | 71.26 | 77.86 | signal survives, but gap remains |

### Offline-to-Online Minimal Slice

Environment: `hopper-medium-replay-v2`, seed0, 50k offline + 10k online.

| Method | Offline final | Online final | Online best | Interpretation |
|---|---:|---:|---:|---|
| TD3+BC decay | 22.43 | 39.64 | 39.64 | online fine-tuning works; decay slightly better than fixed |
| TD3+BC fixed | 22.43 | 36.78 | 36.78 | online improves but less than decay |
| ATLAS decay | 45.29 | 37.97 | 38.26 | offline advantage does not persist |
| ATLAS fixed | 45.29 | 29.22 | 32.27 | fixed teacher regularization over-constrains online adaptation |
| ATLAS decay, online fraction 0.25 | 45.29 | 31.46 | 43.73 | lower online replay fraction does not fix final degradation |

## Files And Artifacts

Primary docs:

- `refine-logs/EXPERIMENT_RESULTS.md`
- `refine-logs/C_LINE_RESULTS_TABLE_20260513.md`
- `refine-logs/NOVELTY_CHECK_20260513.md`
- `refine-logs/ONLINE_FINETUNE_MINIMAL_PLAN_20260513.md`

Key result folders:

- `refine-logs/remote-results/atlas_labels_20260512_2205/`
- `refine-logs/remote-results/atlas_seed1_eval20_20260513_004052/`
- `refine-logs/remote-results/atlas_label_ablation_shuffle_20260513_003101/`
- `refine-logs/remote-results/atlas_walker_seed0_eval10_20260513_023101/`
- `refine-logs/remote-results/o2o_minimal_20260513/`

Executable entrypoints:

- `algorithms/td3_bc_o2o.py`
- `scripts/run_o2o_minimal.sh`

## What A/B Lines Should Provide For Merge

This is not a detailed task assignment. It is the minimum comparable format needed for the final group report.

Working merge proposal after the latest discussion: B-line should not duplicate the original IQL / implicit-conservatism line. Instead, it is cleaner if B-line serves as a normal / non-conservative contrast line. This makes the final project cleaner:

- A-line: conservative value methods;
- B-line: normal / non-conservative learning or weak-regularization contrast;
- C-line: policy / behavior regularization and trusted-action selection.

For each method/run:

- method name;
- method family: value conservatism / non-conservative contrast / policy regularization;
- environment;
- seed;
- offline training steps;
- online fine-tuning steps if any;
- eval episodes;
- final normalized score;
- best normalized score;
- curve file or log path;
- one-sentence interpretation.

Recommended minimal environments:

- `hopper-medium-replay-v2`;
- `walker2d-medium-replay-v2` if time permits.

Recommended minimal methods:

- A-line: CQL or CQL-family value conservatism;
- B-line: normal / non-conservative contrast, such as PPO/SAC-style online baseline, vanilla TD3-style online fine-tuning, or another agreed weak-regularization baseline;
- C-line: TD3+BC / ReBRAC / SSAR / ATLAS evidence from this package.

For B-line, the key question is not "does implicit conservatism work?" anymore. The key question is:

> If we remove explicit conservatism / trusted-action regularization, how does a normal learner behave on the same low-quality-data-to-online-finetuning pipeline?

This gives the report a clear contrast against A-line and C-line.

## Current Stop Rule

Do not expand C-line into broad 6-env x 3-seed sweeps now. The next useful C-line work, if needed, should be targeted:

- ATLAS offline label-usage ablation, if the report needs one more mechanism check;
- one stronger teacher-side O2O curve only if the group report needs it;
- plotting and merge formatting.
