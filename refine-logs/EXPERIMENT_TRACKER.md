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
| R013 | A1 | SSAR stability after 50k | SSAR cached IQL-qv | `hopper-medium-replay-v2`, seed0, 100k | normalized score curve | MUST | DONE | final 92.44, best 100.98 @90k; later seed1 eval20 reframes this as high-variance upper anchor |
| R014 | A1 | isolate IQL-qv action selection | cheap SSAR no IQL selection | `hopper-medium-replay-v2`, seed0, 100k | normalized score curve | MUST | DONE | final 25.48, best 30.34 @90k; removing trusted selection collapses gains |
| R015 | A1 | same-budget simple baseline | `rebrac_lite` | `hopper-medium-replay-v2`, seed0, 100k | normalized score curve | MUST | DONE | final 36.54, best 54.36 @90k; spikes but far below cached SSAR |
| R016 | C1 | cheap local trusted selector | `trusted_td3_bc_top20` | `hopper-medium-replay-v2`, seed0, 50k/100k | normalized score curve | MUST | DONE | 50k final/best 45.13; 100k final 28.76, best 45.13 @50k; simple return-ranked mask is promising but not stable |
| R017 | C2 | implement cheap Q-gap trusted selector | `trusted_td3_bc_qgap_soft` | local + AutoDL 2-step smoke | py_compile, smoke result write | MUST | DONE | smoke passed with real Q-gap path; trust_w approx 0.525, qgap nonzero |
| R018 | C2 | Q-gap selector go/no-go | `trusted_td3_bc_qgap_soft` | `hopper-medium-replay-v2`, seed0, 50k | normalized score curve, Q-gap stats | MUST | DONE | final 19.94, best 22.05; below ReBRAC-lite and not worth 100k |
| R019 | C2 | Q-gap selector stability | `trusted_td3_bc_qgap_soft` | `hopper-medium-replay-v2`, seed0, 100k | normalized score curve | MUST | CUT | skipped because R018 failed 50k gate |
| R020 | C2 | conservative fallback selector | `trusted_td3_bc_consistency` | `hopper-medium-replay-v2`, seed0, 50k | normalized score, trusted fraction | NICE | DONE | final 19.77, best 28.00 @40k; below ReBRAC-lite, no 100k |
| R021 | C3 | minimal stability gate | ATLAS | `hopper-medium-replay-v2`, seed1, 100k, eval20 | normalized score final/best | NICE | DONE | final/best 68.11 @100k, close to seed0 final 69.97 |
| R022 | V0 | public value baseline smoke | `iql`, `cql` | local + AutoDL 2-step smoke | py_compile, smoke result write | MUST | DONE | local compile/bash check passed; AutoDL 2-step smoke passed for both with finite losses and result writes |
| R023 | V1 | IQL baseline check | `iql` | `hopper-medium-replay-v2`, seed0, 100k | normalized score curve | MUST | DONE | compact shared-pipeline IQL final 45.27, best 81.28 @80k; strong value baseline but still below SSAR cached best 100.98 |
| R024 | V1 | CQL baseline check | `cql` | `hopper-medium-replay-v2`, seed0, 50k | normalized score curve | SHOULD | DONE | compact shared-pipeline CQL final/best 39.81; useful conservative value anchor, not a C-line expansion target |
| R025 | D1 | teacher-label export path | SSAR/IQL-qv labels + ATLAS smoke selector | `hopper-medium-replay-v2`, seed0 | label export, selector train smoke, 2-step RL smoke | MUST | DONE | exported Q-V labels from SSAR cache; hard trust fraction 0.3696; 1-epoch ATLAS smoke val acc 0.6375; label-file TD3+BC 2-step smoke passed |
| R026 | D2 | first ATLAS gate | `trusted_td3_bc_atlas` | `hopper-medium-replay-v2`, seed0, 50k | normalized score curve, label agreement, runtime | MUST | DONE | final/best 45.29; beats ReBRAC-lite 34.48 and CQL 39.81; extend to 100k for stability |
| R027 | D2 | ATLAS stability gate | `trusted_td3_bc_atlas` | `hopper-medium-replay-v2`, seed0, 100k | normalized score curve, tail stability | MUST | DONE | final/best 69.97 @100k; passes seed0 gate and should be compared to SSAR high-variance anchor, not a stable SSAR mean |
| R028 | D3 | ATLAS label-quality ablation | teacher labels vs shuffled labels | `hopper-medium-replay-v2`, seed0, 50k | normalized score, label control gap | SHOULD | DONE | shuffled score distribution final 18.78 / best 19.35 vs aligned ATLAS 45.29; teacher alignment matters |
| R029 | A2 | SSAR anchor reliability | SSAR full IQL-qv | `hopper-medium-replay-v2`, seed1, 100k, eval20 | normalized score curve, cache checksum | SHOULD | DONE | final 60.88, best 99.22 @90k; seed1 confirms high-score spikes but not stable 90+ final |
| R030 | D4 | ATLAS seed1 stability | `trusted_td3_bc_atlas` | `hopper-medium-replay-v2`, seed1, 100k, eval20 | normalized score final/best | SHOULD | DONE | final/best 68.11 @100k; seed0/seed1 finals are both near 68-70 |
| R031 | D5 | ATLAS second replay env | SSAR teacher cache + `trusted_td3_bc_atlas` | `walker2d-medium-replay-v2`, seed0, 100k, eval10 | SSAR anchor, cache, ATLAS final/best | SHOULD | DONE | SSAR final 94.28 / best 94.60; ATLAS final 71.26 / best 77.86; cross-env signal survives but leaves clear optimization gap |
| R032 | O1 | required P0 O2O eval20 panel | TD3+BC, ATLAS, random trust subset, SSAR/IQL-qv labels | `hopper-medium-replay-v2`, seed0, 50k offline + 10k online, eval20 | offline final/best, online final/best, final delta | MUST | DONE | TD3+BC decay best final online 40.06; ATLAS/SSAR stronger offline but worse final online; SSAR fixed transient best 96.22 but final 38.61 |
| R033 | O2 | external review of paper/value | Gemini reviewer | draft + P0 table | score, weaknesses, next experiments | SHOULD | DONE | score 6/10 borderline; recommended constraint-transfer gap framing and Q-filtered trust |
| R034 | O3 | Q-filtered trust diagnostic | ATLAS qgate fixed + SSAR/IQL qgate fixed | `hopper-medium-replay-v2`, seed0, 50k offline + 10k online, eval20 | online final/best | SHOULD | DONE | ATLAS qgate final/best 48.41; SSAR qgate final 38.88, best 65.73 |
| R035 | O3 | qgate seed check | TD3+BC release + ATLAS qgate fixed | `hopper-medium-replay-v2`, seed1, 50k offline + 10k online, eval20 | online final/best | SHOULD | DONE | TD3+BC release final 98.86; ATLAS qgate fixed final 39.91, best 70.03; no robust superiority claim |
| R036 | O3 | ATLAS seed1 release/fixed control | ATLAS release/fixed | `hopper-medium-replay-v2`, seed1, 50k offline + 10k online, eval20 | online final/best | SHOULD | DONE | ATLAS release final 84.35, fixed final 23.88; release is better than qgate on seed1 |

## Next Contribution TODO

Current decision: baseline/reproduction is sufficient for exploration. Do not expand PRDC/A2PR or run 6-env multi-seed sweeps yet. The next work should target a cheaper replacement or amortization strategy for SSAR's expensive IQL-qv trusted action selection.

| TODO ID | Task | Why | First Run / Output | Status | Notes |
|---------|------|-----|--------------------|--------|-------|
| T001 | Design cheap trusted-action selector candidates | This is the likely contribution: approximate SSAR's IQL-qv selection without 1M-step pretraining | short design note with 2-3 selectors and expected behavior | DONE | see `refine-logs/TRUSTED_SELECTOR_PLAN.md`; first candidate is return-ranked trajectory trust |
| T002 | Implement one minimal selector in local code | Need an editable/local mechanism, not just external SSAR reproduction | config/script that produces trusted mask or beta weights | DONE | `algorithms/trusted_td3_bc.py`; `python3 -m py_compile` passed |
| T003 | Run seed0 `hopper-medium-replay-v2` 50k/100k for the selector | Fast go/no-go for contribution signal | compare against TD3+BC, ReBRAC-lite, cheap SSAR no-IQL, SSAR cached | DONE | final 28.76, best 45.13 @50k; do not scale this exact selector yet |
| T004 | Add cache discipline for SSAR/IQL-qv | Avoid paying the 1-hour IQL-qv cost repeatedly | documented cache path + checksum + reuse script | PARTIAL | seed0 cache exists and is backed up; next env/seed needs same convention |
| T005 | Run one narrow ATLAS stability check | Stability validation should follow contribution signal, not replace it | seed1 on `hopper-medium-replay-v2` | DONE | final/best 68.11 @100k eval20, close to seed0 final 69.97 |
| T006 | Replace return-only trust with a cheap critic/Q-gap selector | Return-ranked mask produced one good 50k point but did not stay stable to 100k | short critic warmup or Q(policy) vs Q(dataset action) trust score | DONE | tried as Q-gap/consistency; failed 50k and is superseded by ATLAS teacher-label direction |
| T007 | Do not scale `trusted_td3_bc_top20` | Its 50k peak did not persist to 100k | keep as negative/partial probe in results table | DONE | no seed expansion for this exact variant |
| T008 | Stop cheap online-selector scaling | Q-gap and consistency both failed 50k gate | record as negative evidence, redesign selector before more GPU runs | DONE | likely need offline value labels, critic pretraining, or amortized SSAR cache rather than online TD3 critic alone |
| T009 | Add public IQL/CQL baselines | Without IQL, SSAR gains may be misattributed to SSAR instead of IQL signal | run IQL 100k and CQL 50k on `hopper-medium-replay-v2` seed0 | DONE | IQL final 45.27, best 81.28 @80k; CQL final/best 39.81 |
| T010 | Build ATLAS label-export infrastructure | Need reusable teacher labels before amortized selector experiments | export SSAR/IQL Q-V labels and verify label-file TD3+BC smoke | DONE | remote label export + selector 1-epoch smoke + 2-step RL smoke passed |
| T011 | Run ATLAS 50k gate | First actual method test for amortized trusted-action labels | train selector on full/large labels, then `trusted_td3_bc_label_file` 50k | DONE | final/best 45.29; passed first gate |
| T012 | Run ATLAS 100k stability gate | 50k passed, but return-ranked selector also had a non-persistent 50k peak | same label-file variant at 100k | DONE | final/best 69.97 @100k; positive seed0 signal, now needs one stability check and label-quality ablation |
| T013 | Add ATLAS label-quality control | A paper-facing claim needs to show the teacher signal matters, not just any weighting | shuffled label file at 50k | DONE | shuffled labels collapse to 18.78 final, supporting teacher-signal claim |
| T014 | Reframe SSAR anchor as high variance | seed1 eval20 does not support treating seed0 92/101 as stable final score | update reports and avoid stable-score language | DONE | SSAR remains a strong upper anchor, not a reliable final-score estimate |
| T015 | Test ATLAS on one second replay env | Need to know whether ATLAS is hopper-specific before discussing a course-project claim | `walker2d-medium-replay-v2`, seed0, SSAR cache + ATLAS 100k | DONE | second env passed directionally: ATLAS 71.26 final / 77.86 best vs SSAR 94.28 final / 94.60 best |
| T016 | ATLAS optimization ablations | If second env is positive or ambiguous, tune label usage rather than adding baselines | soft vs hard labels, label min weight, selector capacity, continuous advantage target | READY | run only 1-2 cheap ablations first; prioritize those that explain the walker gap |
| T017 | Run required P0 O2O eval20 panel | Need a comparable O2O table before writing the course-project claim | TD3+BC, ATLAS, random-subset control, SSAR/IQL-qv teacher labels on one env/seed | DONE | local copy in `refine-logs/remote-results/o2o_p0_eval20_20260514/`; current claim should emphasize offline label utility plus online adaptation failure |
| T018 | Test online Q-filtered trust | External review suggested a gate rather than time-only release | ATLAS/SSAR qgate fixed on seed0, plus seed1 ATLAS/TD3 check | DONE | seed0 positive for ATLAS, but seed1 shows high O2O variance and no stable dominance |

## Explicitly Deferred

| Deferred Item | Reason |
|---------------|--------|
| PRDC multi-seed / multi-env expansion | 50k source result only reached 23.54, not worth mainline compute now |
| A2PR multi-seed / multi-env expansion | 50k source result final 22.31, no clear uplift over TD3+BC |
| 6 D4RL env x 3 seed baseline table | Too expensive for exploration; does not create contribution by itself |
| More TD3+BC alpha sweeps | `td3_bc_alpha5` did not improve over TD3+BC |
