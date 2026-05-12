# Experiment Plan

**Problem**: TD3+BC 在 D4RL replay 数据集，尤其 `hopper-medium-replay-v2` 上早期分数偏低，需要快速判断 C 方向的 policy regularization 变体是否有真实信号。
**Method Thesis**: 用一个 replay 子数据集、一个 seed、四个模型/变体做探索期筛选，先找到能改善 replay 低分的机制，再扩大到多 seed / 多环境。
**Date**: 2026-05-12

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1: replay 低分主要需要 policy/action regularization，而不是继续铺 baseline sweep | 这是 C 线是否值得推进的核心判断 | 在 `hopper-medium-replay-v2` seed0 的 50k/100k smoke 中，ReBRAC-style 或 TD3+BC variant 比修后 TD3+BC 高 5-10 normalized score，或曲线明显更早上升 | B1, B2 |
| C2: 当前阶段只需要发现方向性信号，不需要 paper 级稳定结论 | 避免把算力花在 18-run baseline 上 | 单 env / 单 seed / 4 系统能给出明确 go/no-go；正信号后再补 3 seeds 和第二个 env | B1, B3 |

## Paper Storyline

- Main paper must prove: C 方向的核心机制能改善 replay/offline-to-online 主线中 TD3+BC 的弱点。
- Appendix can support: 多 seed、多环境、多步数曲线、完整 baseline family。
- Experiments intentionally cut: 现阶段不跑 6 env x 3 seeds 的全量 baseline，不跑没有机制差异的重复 TD3+BC。

## Experiment Blocks

### Block 1: One-Env Four-System Screen

- Claim tested: C1, C2。
- Why this block exists: 用最小成本判断有没有值得继续追的 policy regularization 信号。
- Dataset / split / task: D4RL `hopper-medium-replay-v2`，seed 0，offline training。
- Compared systems: `bc`, `td3_bc`, `td3_bc_alpha5`, `rebrac_lite`。
- Metrics: normalized score first；raw return、episode length、critic loss、actor loss second。
- Setup details: 先 50k steps，eval every 10k，5 eval episodes；结果写 `results/c_track_smoke/`，Aim 默认开启，W&B 可选。
- Success criterion: 任一 C 方向 variant 在 50k/100k 比 `td3_bc` 高 5-10 normalized score，或曲线明显更稳定/更早上升。
- Failure interpretation: 如果全部接近 BC/TD3+BC 或更差，说明当前 variant 不足以解释 replay 低分，需要改机制而不是扩大 seeds。
- Table / figure target: 探索期内部表；若有正信号，后续扩成 paper main/appendix 曲线。
- Priority: MUST-RUN。

### Block 2: Loader-Fix Regression Check

- Claim tested: C2。
- Why this block exists: 确保后续所有算法共享 timeout-safe D4RL 口径。
- Dataset / split / task: `hopper-medium-replay-v2`, `halfcheetah-medium-v2` 真实 D4RL dataset。
- Compared systems: `D4RLDataset` vs `d4rl.qlearning_dataset` transition count。
- Metrics: transition count 是否一致；训练脚本是否能采样 `next_act`。
- Setup details: 已核对 `hopper-medium-replay-v2=401598`，`halfcheetah-medium-v2=999000`。
- Success criterion: shared loader 与官方 transition count 对齐，四路 smoke 都能端到端启动。
- Failure interpretation: 若 count 或 shape 不一致，先修 loader，不跑模型。
- Table / figure target: README / engineering note。
- Priority: MUST-RUN。

### Block 3: Scale-Up Gate

- Claim tested: C1。
- Why this block exists: 防止单 seed 偶然性，同时只在有价值信号后花钱。
- Dataset / split / task: 先补 `hopper-medium-replay-v2` seeds 1/2，再加 `halfcheetah-medium-replay-v2` 或 `walker2d-medium-replay-v2`。
- Compared systems: 只保留 Block 1 中胜出的 1-2 个 variant，加 `td3_bc` anchor。
- Metrics: normalized score mean/std，best@step，曲线稳定性。
- Setup details: 100k steps 起步；若继续有信号再到 1M。
- Success criterion: 多 seed 平均仍超过 TD3+BC 5+ 分，或 replay 曲线有一致改善。
- Failure interpretation: 单 seed 信号不可复现，回到机制和超参设计。
- Table / figure target: 可能成为 paper main table 的 seed-expanded 前身。
- Priority: NICE-TO-HAVE after Block 1 passes。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | 代码/loader sanity | py_compile；真实 D4RL count；每个模型 1-2 step smoke | 全部 import、采样、写结果正常 | 本地/AutoDL 几分钟 | ReBRAC next_action shape 或 LayerNorm 实现问题 |
| M1 | 四路筛选 | `bc`, `td3_bc`, `td3_bc_alpha5`, `rebrac_lite` on `hopper-medium-replay-v2`, seed0, 50k | 有 variant 比 TD3+BC 高 5-10 分则进入 M2 | 约 1-2 GPU 小时，4090 约 ¥3-6 | 小网络 CPU/env eval 瓶颈，可能慢于预估 |
| M2 | 胜出者复核 | 胜出者 + TD3+BC，seed1/2，100k | 信号跨 seed 保持 | 约 2-4 GPU 小时 | 单 seed 偶然性 |
| M3 | 第二 replay env | 加 `halfcheetah-medium-replay-v2` 或 `walker2d-medium-replay-v2` | 至少一个第二环境有同向改善 | 约 2-4 GPU 小时 | Hopper-specific trick |

## Compute and Data Budget

- Total estimated GPU-hours for current must-run: 1-2 hours on retained AutoDL 4090。
- Data preparation needs: 已缓存 D4RL；不新增数据集。
- Human evaluation needs: 无。
- Biggest bottleneck: MuJoCo eval 和小网络训练 CPU-bound，GPU 利用率可能不高。

## Risks and Mitigations

- Risk: ReBRAC-lite 是 PyTorch compact port，不等于官方 JAX ReBRAC。
- Mitigation: 只作为探索期 variant；若有信号，再做官方/更严格复现或完整 port。
- Risk: 50k 太短，TD3+BC replay 可能慢热。
- Mitigation: 先看 50k 曲线；若 ReBRAC/variant 早期更好，再补 100k。
- Risk: 四路顺序跑时间过长。
- Mitigation: 任一 run 明显异常或低于 BC 很多时可停掉该 variant，不扩大 sweep。

## Final Checklist

- [x] Main paper tables are not claimed yet; current run is exploration only
- [x] Novelty is isolated as policy/action regularization candidate
- [x] Simplicity is defended by one env / one seed / four systems first
- [x] Frontier contribution is not claimed
- [x] Nice-to-have runs are separated from must-run runs
