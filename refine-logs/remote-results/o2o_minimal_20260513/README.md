# Minimal Offline-to-Online Smoke

Date: 2026-05-13  
Instance: `pro-7785f027d673`  
GPU: RTX 4090 48G / vGPU-48GB  
Env: `hopper-medium-replay-v2`  
Seed: `0`  
Setting: 50k offline pretraining + 10k online fine-tuning, eval every 10k offline / 1k online, 5 eval episodes.

## Results

| Run | Offline final | Online final | Online best | Best online step | Interpretation |
|---|---:|---:|---:|---:|---|
| TD3+BC O2O decay | 22.43 | 39.64 | 39.64 | 60k | Online fine-tuning improves over offline; decay slightly beats fixed in this short slice |
| TD3+BC O2O fixed | 22.43 | 36.78 | 36.78 | 60k | Online fine-tuning improves but remains below decay |
| ATLAS O2O decay | 45.29 | 37.97 | 38.26 | 56k | ATLAS offline advantage does not persist under this first online setup |
| ATLAS O2O fixed | 45.29 | 29.22 | 32.27 | 58k | Fixed ATLAS regularization is worse than decay, suggesting over-constraint during online adaptation |
| ATLAS O2O decay, online fraction 0.25 | 45.29 | 31.46 | 43.73 | 54k | Lower online replay fraction produces a transient spike but worse final score |

## Takeaway

The missing offline-to-online project piece now has a working minimal curve. The result supports keeping online fine-tuning in the main project story, but it also shows that the current ATLAS weighting is not automatically beneficial once online data is mixed in.

Current interpretation:

- TD3+BC can improve from `22.43` offline to around `39.64` after 10k online steps.
- Regularization decay has a small positive signal over fixed regularization in this one-seed short run.
- ATLAS starts higher offline (`45.29`) but drops during online fine-tuning. Fixed ATLAS regularization drops more than decay, suggesting the current teacher-label regularizer can over-constrain online adaptation.
- Reducing online replay fraction from `0.50` to `0.25` does not fix ATLAS O2O. It gives a transient `43.73` spike but worse final `31.46`.

## Local Files

- Results: `results/*.jsonl`
- Logs:
  - `logs/o2o_td3_decay_20260513.log`
  - `logs/o2o_td3_fixed_20260513.log`
  - `logs/o2o_atlas_decay_20260513.log`
  - `logs/o2o_atlas_fixed_20260513.log`
  - `logs/o2o_atlas_decay_ofrac025_20260513.log`

## Next Decision

Do not expand to multi-seed yet. The next useful check is either:

1. avoid claiming ATLAS improves O2O under the current runner;
2. try lower ATLAS label strength / higher label floor only if we want to rescue the O2O angle;
3. compare against IQL/SSAR O2O only if the group needs a stronger teacher-side online curve.
