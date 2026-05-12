# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | loader sanity | `D4RLDataset` vs `d4rl.qlearning_dataset` | `hopper-medium-replay-v2`, `halfcheetah-medium-v2` | transition count | MUST | DONE | `401598`, `999000` verified on AutoDL |
| R002 | M0 | import/compile sanity | `bc`, `td3_bc`, `rebrac_lite` | local + AutoDL code | py_compile + 2-step smoke | MUST | DONE | remote four-system `STEPS=2` smoke passed |
| R003 | M1 | anchor behavior cloning | `bc` | `hopper-medium-replay-v2`, seed0, 50k | normalized score | MUST | DONE | final 17.86, best 32.26 @30k |
| R004 | M1 | fixed baseline | `td3_bc` | `hopper-medium-replay-v2`, seed0, 50k | normalized score, critic loss | MUST | DONE | final/best 22.43 |
| R005 | M1 | stronger Q term variant | `td3_bc_alpha5` | `hopper-medium-replay-v2`, seed0, 50k | normalized score, actor loss | MUST | DONE | final/best 21.95; no uplift |
| R006 | M1 | ReBRAC-style action regularization | `rebrac_lite` | `hopper-medium-replay-v2`, seed0, 50k | normalized score, bc_mse_policy | MUST | DONE | final/best 34.48; +12.05 over TD3+BC |
| R007 | S0 | official-source compatibility | PRDC / A2PR / SSAR | `hopper-medium-replay-v2`, seed0, 100-step preflight | import, dataset load, eval | MUST | DONE | official repos run outside public repo; SSAR needed compatibility-only import patch |
| R008 | S1 | fast official-source screen | PRDC | `hopper-medium-replay-v2`, seed0, 50k | normalized score | MUST | DONE | final/best 23.54; source repo `LAMDA-RL/PRDC`; no local algorithm changes |
| R009 | S1 | fast official-source screen | A2PR | `hopper-medium-replay-v2`, seed0, 50k | normalized score | MUST | DONE | final 22.31, best 22.81 @40k; source repo `ltlhuuu/A2PR`; no local algorithm changes |
| R010 | S2 | full-logic modern source screen | SSAR TD3+BC backbone | `hopper-medium-replay-v2`, seed0, full IQL-qv + 50k offline | normalized score, IQL-qv cost | MUST | DONE | final 38.56, best 43.97 @40k; IQL-qv full preselection + 50k took 4003s |
| R011 | M2 | seed expansion if signal | SSAR + `rebrac_lite` + `td3_bc` | `hopper-medium-replay-v2`, seed1/2, 50k or 100k | mean/std normalized score | NICE | READY | next if we want reliability beyond seed0 |
| R012 | M3 | second replay env if signal | SSAR + `rebrac_lite` + `td3_bc` | replay env TBD, seed0, 50k or 100k | normalized score | NICE | READY | choose after deciding whether to spend on another source run |
| R013 | A1 | SSAR stability after 50k | SSAR cached IQL-qv | `hopper-medium-replay-v2`, seed0, 100k | normalized score curve | MUST | DONE | final 92.44, best 100.98 @90k; cache reuse works and SSAR remains strong |
| R014 | A1 | isolate IQL-qv action selection | cheap SSAR no IQL selection | `hopper-medium-replay-v2`, seed0, 100k | normalized score curve | MUST | DONE | final 25.48, best 30.34 @90k; removing trusted selection collapses gains |
| R015 | A1 | same-budget simple baseline | `rebrac_lite` | `hopper-medium-replay-v2`, seed0, 100k | normalized score curve | MUST | DONE | final 36.54, best 54.36 @90k; spikes but far below cached SSAR |
