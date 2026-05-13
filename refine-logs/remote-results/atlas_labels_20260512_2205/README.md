# ATLAS Label Export Smoke

Date: 2026-05-12

Remote paths:

- teacher label file: `/root/autodl-tmp/taoyao-rl/project/results/atlas_labels/iql_qv_hopper-medium-replay-v2_seed0.npz`
- ATLAS smoke prediction file: `/root/autodl-tmp/taoyao-rl/project/results/atlas_labels/atlas_selector_smoke_hopper-medium-replay-v2_seed0.npz`
- ATLAS smoke model: `/root/autodl-tmp/taoyao-rl/project/results/atlas_labels/atlas_selector_smoke_hopper-medium-replay-v2_seed0.pt`
- RL smoke result: `results/atlas_label_smoke/trusted_td3_bc_atlas_smoke_hopper-medium-replay-v2_seed0.jsonl`
- full ATLAS prediction file: `/root/autodl-tmp/taoyao-rl/project/results/atlas_labels/atlas_selector_hopper-medium-replay-v2_seed0.npz`
- full ATLAS model: `/root/autodl-tmp/taoyao-rl/project/results/atlas_labels/atlas_selector_hopper-medium-replay-v2_seed0.pt`
- ATLAS 50k result: `results/atlas_50k/trusted_td3_bc_atlas_hopper-medium-replay-v2_seed0.jsonl`
- ATLAS 100k result: `results/atlas_100k/trusted_td3_bc_atlas_hopper-medium-replay-v2_seed0.jsonl`

Teacher label export:

- source checkpoint: `/root/autodl-tmp/external_repos/SSAR/model/iql_qv/hopper-medium-replay-v2/0/0.7_model.pth`
- dataset size: `401598`
- hard trust fraction: `0.3696`
- advantage mean/std: `-1.4102 / 5.7651`
- trust score mean: `0.3885`

ATLAS selector smoke:

- target: `hard_trust`
- train subset: `20000`
- epochs: `1`
- validation accuracy: `0.6375`
- predicted atlas score mean/std: `0.3663 / 0.0670`
- predicted weight mean with min weight 0.05: `0.3980`

RL integration smoke:

- command path: `RUNS=trusted_td3_bc_label_file`
- label key: `atlas_score`
- steps: `2`
- result: compile, label load, weighted BC update, eval write all passed.

Full selector training:

- target: `hard_trust`
- train set: full `401598` transitions
- epochs: `5`
- validation accuracy: `0.7313`
- predicted atlas score mean/std: `0.3823 / 0.2212`
- predicted weight mean with min weight 0.05: `0.4132`

ATLAS method runs:

| Run | Final normalized score | Best normalized score | Best step | Notes |
|-----|------------------------|-----------------------|-----------|-------|
| 50k | 45.29 | 45.29 | 50k | passed first gate vs ReBRAC-lite 34.48 / CQL 39.81 |
| 100k | 69.97 | 69.97 | 100k | positive seed0 stability; below cached SSAR 92.44 final / 100.98 best |

Interpretation: ATLAS is now the strongest local contribution candidate. It needs one narrow stability check and a teacher-label ablation before becoming a stronger claim.
