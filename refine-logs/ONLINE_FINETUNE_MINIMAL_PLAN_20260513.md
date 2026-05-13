# Minimal Offline-to-Online Fine-Tuning Plan

Date: 2026-05-13  
Status: executable scaffold added; remote run not launched yet.

## Why This Is P0

The original project is explicitly about offline-to-online RL. Our current C-line evidence is mostly offline / mechanism evidence. To keep the project aligned with the original topic, we need at least one short online fine-tuning slice.

This should be small:

- one environment first;
- one seed first;
- two or three representative methods;
- short online horizon;
- no broad 6-env x 3-seed sweep.

## New Executable Entry

Added:

- `algorithms/td3_bc_o2o.py`
- `scripts/run_o2o_minimal.sh`

The runner does:

1. offline TD3-style pretraining on D4RL;
2. online interaction with the real environment;
3. mixed replay during online fine-tuning;
4. offline BC / ATLAS regularization only on offline samples;
5. optional linear decay of the offline regularization coefficient during online fine-tuning.

Result phase labels:

- `offline`
- `online_finetune`

Result directory default:

- `results/o2o_minimal/`

## First Remote Runs

Environment:

- `hopper-medium-replay-v2`

Seed:

- `0`

Budget:

- Start with `offline_steps=50k`, `online_steps=10k`.
- This is a smoke-style online curve, not a final benchmark.

### Run 1: TD3+BC O2O Decay

Purpose: anchor online fine-tuning curve.

```bash
cd /root/autodl-tmp/taoyao-rl/project
ENV=hopper-medium-replay-v2 \
SEED=0 \
OFFLINE_STEPS=50000 \
ONLINE_STEPS=10000 \
EVAL_EPISODES=5 \
RUN=td3_bc_o2o \
RESULT_DIR=results/o2o_minimal \
bash scripts/run_o2o_minimal.sh
```

This uses `online_bc_coef_start=1.0`, `online_bc_coef_end=0.0`.

### Run 2: TD3+BC O2O Fixed

Purpose: simple fixed-regularization comparison.

```bash
cd /root/autodl-tmp/taoyao-rl/project
ENV=hopper-medium-replay-v2 \
SEED=0 \
OFFLINE_STEPS=50000 \
ONLINE_STEPS=10000 \
EVAL_EPISODES=5 \
RUN=td3_bc_o2o_fixed \
RESULT_DIR=results/o2o_minimal \
bash scripts/run_o2o_minimal.sh
```

This uses `online_bc_coef_start=1.0`, `online_bc_coef_end=1.0`.

### Run 3: ATLAS O2O Decay

Purpose: test whether the ATLAS trusted-label signal helps the online fine-tuning slice.

Remote label file already exists:

```text
/root/autodl-tmp/taoyao-rl/project/results/atlas_labels/atlas_selector_hopper-medium-replay-v2_seed0.npz
```

Command:

```bash
cd /root/autodl-tmp/taoyao-rl/project
ENV=hopper-medium-replay-v2 \
SEED=0 \
OFFLINE_STEPS=50000 \
ONLINE_STEPS=10000 \
EVAL_EPISODES=5 \
RUN=atlas_o2o \
LABEL_PATH=/root/autodl-tmp/taoyao-rl/project/results/atlas_labels/atlas_selector_hopper-medium-replay-v2_seed0.npz \
LABEL_SCORE_KEY=atlas_score \
RESULT_DIR=results/o2o_minimal \
bash scripts/run_o2o_minimal.sh
```

## Go / Stop Rule

After these runs, inspect:

- whether online fine-tuning improves over the offline endpoint;
- whether decay beats fixed regularization;
- whether ATLAS improves the online curve or only the offline endpoint.

Stop if:

- all online curves are flat or unstable;
- ATLAS loses its offline advantage immediately;
- implementation bugs make the curve uninterpretable.

Continue only if:

- at least one method shows meaningful online improvement;
- decay vs fixed gives a clear difference;
- ATLAS improves sample efficiency or tail stability.

## Notes

- This scaffold intentionally reruns offline pretraining inside the same process before online fine-tuning. That avoids introducing checkpoint compatibility problems before the first O2O smoke.
- If the O2O smoke is useful, the next engineering improvement should be checkpoint save/load so we can reuse existing offline models.
- If runtime is too high, reduce `OFFLINE_STEPS=30000` and keep `ONLINE_STEPS=10000` for a cheaper first curve.

## Completed First Slice

Completed on AutoDL instance `pro-7785f027d673` with RTX 4090 48G. The instance was powered off after results were copied back.

| Run | Offline final | Online final | Online best | Interpretation |
|---|---:|---:|---:|---|
| TD3+BC O2O decay | 22.43 | 39.64 | 39.64 | online fine-tuning works; decay slightly beats fixed |
| TD3+BC O2O fixed | 22.43 | 36.78 | 36.78 | online fine-tuning works but below decay |
| ATLAS O2O decay | 45.29 | 37.97 | 38.26 | ATLAS offline advantage does not persist |
| ATLAS O2O fixed | 45.29 | 29.22 | 32.27 | fixed teacher-label regularization over-constrains online adaptation |

Local copy:

```text
refine-logs/remote-results/o2o_minimal_20260513/
```

Next O2O ablation, if needed, should not be another broad seed. Prefer changing online mixing/label strength:

- lower `ONLINE_BATCH_FRACTION`, e.g. `0.25`;
- lower ATLAS regularization strength;
- compare with IQL/SSAR O2O only if the group needs a stronger teacher-side curve.
