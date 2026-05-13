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
| R016 | C1 | cheap local trusted selector | `trusted_td3_bc_top20` | `hopper-medium-replay-v2`, seed0, 50k/100k | normalized score curve | MUST | DONE | 50k final/best 45.13; 100k final 28.76, best 45.13 @50k; simple return-ranked mask is promising but not stable |
| R017 | C2 | implement cheap Q-gap trusted selector | `trusted_td3_bc_qgap_soft` | local + AutoDL 2-step smoke | py_compile, smoke result write | MUST | DONE | smoke passed with real Q-gap path; trust_w approx 0.525, qgap nonzero |
| R018 | C2 | Q-gap selector go/no-go | `trusted_td3_bc_qgap_soft` | `hopper-medium-replay-v2`, seed0, 50k | normalized score curve, Q-gap stats | MUST | DONE | final 19.94, best 22.05; below ReBRAC-lite and not worth 100k |
| R019 | C2 | Q-gap selector stability | `trusted_td3_bc_qgap_soft` | `hopper-medium-replay-v2`, seed0, 100k | normalized score curve | MUST | CUT | skipped because R018 failed 50k gate |
| R020 | C2 | conservative fallback selector | `trusted_td3_bc_consistency` | `hopper-medium-replay-v2`, seed0, 50k | normalized score, trusted fraction | NICE | DONE | final 19.77, best 28.00 @40k; below ReBRAC-lite, no 100k |
| R021 | C3 | minimal stability gate | winning cheap selector | seed1 or second replay env, 100k | normalized score mean/std or cross-env direction | NICE | BLOCKED | no cheap selector passed the 50k gate |

## Next Contribution TODO

Current decision: baseline/reproduction is sufficient for exploration. Do not expand PRDC/A2PR or run 6-env multi-seed sweeps yet. The next work should target a cheaper replacement or amortization strategy for SSAR's expensive IQL-qv trusted action selection.

| TODO ID | Task | Why | First Run / Output | Status | Notes |
|---------|------|-----|--------------------|--------|-------|
| T001 | Design cheap trusted-action selector candidates | This is the likely contribution: approximate SSAR's IQL-qv selection without 1M-step pretraining | short design note with 2-3 selectors and expected behavior | DONE | see `refine-logs/TRUSTED_SELECTOR_PLAN.md`; first candidate is return-ranked trajectory trust |
| T002 | Implement one minimal selector in local code | Need an editable/local mechanism, not just external SSAR reproduction | config/script that produces trusted mask or beta weights | DONE | `algorithms/trusted_td3_bc.py`; `python3 -m py_compile` passed |
| T003 | Run seed0 `hopper-medium-replay-v2` 50k/100k for the selector | Fast go/no-go for contribution signal | compare against TD3+BC, ReBRAC-lite, cheap SSAR no-IQL, SSAR cached | DONE | final 28.76, best 45.13 @50k; do not scale this exact selector yet |
| T004 | Add cache discipline for SSAR/IQL-qv | Avoid paying the 1-hour IQL-qv cost repeatedly | documented cache path + checksum + reuse script | PARTIAL | seed0 cache exists and is backed up; next env/seed needs same convention |
| T005 | Only after a positive local selector signal, run seed1 or second replay env | Stability validation should follow contribution signal, not replace it | seed1 on `hopper-medium-replay-v2` or one second replay env | BLOCKED | unblock only if T003 beats ReBRAC-lite or clearly narrows gap to SSAR |
| T006 | Replace return-only trust with a cheap critic/Q-gap selector | Return-ranked mask produced one good 50k point but did not stay stable to 100k | short critic warmup or Q(policy) vs Q(dataset action) trust score | TODO | next local contribution candidate; closer to SSAR mechanism than trajectory-return filtering |
| T007 | Do not scale `trusted_td3_bc_top20` | Its 50k peak did not persist to 100k | keep as negative/partial probe in results table | DONE | no seed expansion for this exact variant |
| T008 | Stop cheap online-selector scaling | Q-gap and consistency both failed 50k gate | record as negative evidence, redesign selector before more GPU runs | DONE | likely need offline value labels, critic pretraining, or amortized SSAR cache rather than online TD3 critic alone |

## Explicitly Deferred

| Deferred Item | Reason |
|---------------|--------|
| PRDC multi-seed / multi-env expansion | 50k source result only reached 23.54, not worth mainline compute now |
| A2PR multi-seed / multi-env expansion | 50k source result final 22.31, no clear uplift over TD3+BC |
| 6 D4RL env x 3 seed baseline table | Too expensive for exploration; does not create contribution by itself |
| More TD3+BC alpha sweeps | `td3_bc_alpha5` did not improve over TD3+BC |
