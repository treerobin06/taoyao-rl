# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | loader sanity | `D4RLDataset` vs `d4rl.qlearning_dataset` | `hopper-medium-replay-v2`, `halfcheetah-medium-v2` | transition count | MUST | DONE | `401598`, `999000` verified on AutoDL |
| R002 | M0 | import/compile sanity | `bc`, `td3_bc`, `rebrac_lite` | local code | py_compile | MUST | TODO | Run after edits |
| R003 | M1 | anchor behavior cloning | `bc` | `hopper-medium-replay-v2`, seed0, 50k | normalized score | MUST | TODO | cheap anchor |
| R004 | M1 | fixed baseline | `td3_bc` | `hopper-medium-replay-v2`, seed0, 50k | normalized score, critic loss | MUST | TODO | shared loader fixed |
| R005 | M1 | stronger Q term variant | `td3_bc_alpha5` | `hopper-medium-replay-v2`, seed0, 50k | normalized score, actor loss | MUST | TODO | config-only TD3+BC ablation |
| R006 | M1 | ReBRAC-style action regularization | `rebrac_lite` | `hopper-medium-replay-v2`, seed0, 50k | normalized score, bc_mse_policy | MUST | TODO | compact PyTorch port for signal only |
| R007 | M2 | seed expansion if signal | winner + `td3_bc` | `hopper-medium-replay-v2`, seed1/2, 100k | mean/std normalized score | NICE | BLOCKED | only if M1 has +5-10 signal |
| R008 | M3 | second replay env if signal | winner + `td3_bc` | replay env TBD, seed0, 100k | normalized score | NICE | BLOCKED | choose after M1 |
