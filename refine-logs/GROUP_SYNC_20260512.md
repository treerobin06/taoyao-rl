# RL 项目阶段同步：实验结论与下一步策略

**日期**：2026-05-12  
**当前阶段**：探索阶段，不是最终 paper-ready 全量复现阶段

## 一句话结论

我们已经完成了足够的基础 baseline / 官方源码筛选。接下来不建议继续做 6 个数据集、多 seed、大规模 baseline sweep；更应该围绕 SSAR 的核心机制做贡献探索：**能否用更便宜、更稳定、更容易复现的方法替代或摊销 SSAR 中昂贵的 IQL-qv trusted action selection**。

## 为什么先选 `hopper-medium-replay-v2`

这轮实验统一选 `hopper-medium-replay-v2`、seed 0，是因为它适合做探索阶段的“压力测试”：

- TD3+BC 在 medium 类环境上大致正常，但在 replay 数据上明显偏低，更容易暴露方法差异。
- replay 数据更贴近 C 方向关心的问题：policy regularization / behavior regularization 对低质量、混合质量数据是否真的有帮助。
- 单环境单 seed 能快速筛掉没信号的方法，避免一开始就把算力花在 6 env x 3 seeds 上。

所以这个选择不是为了最终结论，而是为了快速判断“哪条机制值得继续”。

## 已完成实验

### 1. 统一数据和评估口径

我们先修正并核对了 D4RL loader，统一走 `d4rl.qlearning_dataset`，避免 timeout 边界处理不一致导致结果不可比。

已核对：

| 数据集 | transition 数 |
|--------|---------------:|
| `hopper-medium-replay-v2` | 401,598 |
| `halfcheetah-medium-v2` | 999,000 |

### 2. 基础本地 baseline / variant

设置：`hopper-medium-replay-v2`，seed 0，50k steps，eval every 10k，5 eval episodes。

| 方法 | Final normalized score | Best | 结论 |
|------|------------------------:|-----:|------|
| BC | 17.86 | 32.26 | 便宜 anchor，但波动大 |
| TD3+BC | 22.43 | 22.43 | replay 上偏低，是弱 anchor |
| TD3+BC alpha5 | 21.95 | 21.95 | 简单增强 Q 项没有帮助 |
| ReBRAC-lite | 34.48 | 34.48 | 有正信号，作为简单强 baseline 保留 |

### 3. 官方源码方法筛选

同样先只跑 `hopper-medium-replay-v2`、seed 0、50k。

| 方法 | Final | Best | 结论 |
|------|------:|-----:|------|
| PRDC official source | 23.54 | 23.54 | 基本只略高于 TD3+BC，暂不扩 |
| A2PR official source | 22.31 | 22.81 | 没有明显超过 TD3+BC，暂不扩 |
| SSAR official source-localized | 38.56 | 43.97 | 当前最强现代 baseline，但 full IQL-qv 预筛较贵 |

### 4. 机制 ablation

为了判断 SSAR 的收益来自哪里，我们做了 100k 的机制对照。

| 变体 | Final | Best | 结论 |
|------|------:|-----:|------|
| SSAR cached IQL-qv | 92.44 | 100.98 | 复用完整 IQL-qv cache 后非常强 |
| cheap SSAR without IQL selection | 25.48 | 30.34 | 去掉 IQL-qv trusted action selection 后基本掉回弱 baseline |
| ReBRAC-lite 100k | 36.54 | 54.36 | 简单 regularization 有信号，但不接近完整 SSAR |

这个结果说明：SSAR 的高分很大程度来自 IQL-qv trusted action selection。这个环节不是可有可无的工程细节，而是当前最重要的机制线索。

## 为什么现在不建议多 seed / 多数据集

目前多 seed / 多数据集不是完全没意义，而是**时机还没到**。

原因：

- PRDC / A2PR 在 seed0 50k 下没有明显信号，扩多 seed 大概率只是确认它们暂时不强。
- 6 env x 3 seeds 成本高，而且主要产出是 baseline 表格，不直接形成 contribution。
- 当前我们真正有价值的问题已经浮现：SSAR 为什么强、IQL-qv selection 是否能被更便宜地替代。
- 在还没有自己的机制前，继续扩大 baseline 会消耗时间和预算，但不一定推进论文贡献。

所以现阶段策略应该是：

> 单数据集、单 seed、短训练先探索机制；只有新机制出现明显信号后，再补 seed 和第二个 replay 数据集。

## 当前建议保留 / 暂停的方法

### 保留

- **TD3+BC**：弱 anchor，必须保留作为基础对照。
- **ReBRAC-lite**：简单强 baseline，用来判断 cheap 方法是否有意义。
- **SSAR**：当前最强现代 baseline，也是机制探索对象。

### 暂停扩展

- **PRDC**：先不做多 seed / 多数据集。
- **A2PR**：先不做多 seed / 多数据集。
- **TD3+BC alpha sweep**：alpha5 没有帮助，暂不继续扫。

## 下一步指导

接下来应该从 baseline reproduction 转向 contribution exploration。

优先做：

1. **设计 cheap trusted-action selector**
   - 目标是近似 SSAR 的 IQL-qv selection，但避免每个 env/seed 都跑 1M-step IQL-qv。
   - 可尝试方向：短 critic warmup、return-ranked trajectory filter、behavior-consistency filter、Q-gap proxy。

2. **实现一个最小本地版本**
   - 输出 trusted mask 或 beta weight。
   - 接入现有 `common.data.D4RLDataset` 和统一 evaluator。
   - 不要一开始做复杂 framework，先做能跑的最小变体。

3. **只在 `hopper-medium-replay-v2`, seed 0 跑 50k/100k**
   - 对照：TD3+BC、ReBRAC-lite、cheap SSAR no-IQL、SSAR cached。
   - 如果明显超过 ReBRAC-lite，或者显著缩小到 SSAR 的差距，再补 seed。

4. **保留并复用 SSAR cache**
   - 已保存 `hopper-medium-replay-v2`, seed0 的 IQL-qv cache。
   - 后续不要重复跑 clean full-IQL，除非专门研究 IQL-qv 方差。

暂时不要做：

- PRDC/A2PR 多 seed。
- 6 个 D4RL 环境全量表。
- 没有机制目的的 baseline sweep。

## 成本和资源策略

- AutoDL 实例用于持续项目时：**关机保留，不释放**，避免每次从零配置环境。
- 预算应该优先花在机制验证，而不是扩没信号的 baseline。
- 每个新想法先做 1 env / 1 seed / 50k 或 100k。
- 只有满足以下条件之一，再扩实验：
  - 超过 ReBRAC-lite；
  - 明显缩小和 SSAR 的差距；
  - 曲线/机制现象能支持论文故事。

## 当前项目问题

现在 C 方向真正的问题可以表述为：

> SSAR 很强，但它依赖昂贵的 IQL-qv trusted action selection。我们能否提出一个更便宜、更稳定、更容易复现的 trusted-action / data-filtering 替代机制，在 replay 数据上接近 SSAR 的收益？

这比继续补 baseline 更可能形成论文贡献。

