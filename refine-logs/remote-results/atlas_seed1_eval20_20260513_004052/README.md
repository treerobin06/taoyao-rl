# ATLAS Seed1 Eval20 Stability Check

Date: 2026-05-13

Purpose: test whether ATLAS survives a second seed using the preserved seed1 SSAR/IQL-qv cache.

Pipeline:

- exported seed1 teacher labels from `/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/hopper-medium-replay-v2/seed1/0.7_model.pth`
- trained seed1 ATLAS selector for 5 epochs
- ran `trusted_td3_bc_atlas` for 100k steps with 20 eval episodes

Selector metadata:

- teacher hard trust fraction: `0.3651`
- selector validation accuracy: `0.7207`
- predicted score mean/std: `0.3668 / 0.2107`
- weighted BC mean with min weight 0.05: `0.3985`

Result:

| Step | Normalized Score |
|------|-----------------:|
| 10k | 11.19 |
| 20k | 23.58 |
| 30k | 27.27 |
| 40k | 26.70 |
| 50k | 31.21 |
| 60k | 27.42 |
| 70k | 56.45 |
| 80k | 37.14 |
| 90k | 58.35 |
| 100k | 68.11 |

Summary:

- final: `68.11`
- best: `68.11 @100k`
- interpretation: ATLAS seed1 final is close to seed0 final `69.97`, and exceeds SSAR seed1 final `60.88` while avoiding the sharp SSAR tail drop. This supports ATLAS as a stable contribution candidate, though more env/seed evidence is still needed before a paper claim.
