# Tree 个人项目状态记录

**日期**：2026-05-12  
**定位**：个人决策文档；用于回看当前 RL 项目做了什么、为什么这样做、下一步该怎么判断。  
**当前阶段**：探索期，不是最终论文实验阶段。

## 1. 当前总判断

这个项目目前已经从“环境和 baseline 能不能跑”推进到“哪些机制值得研究”。

现在不应该直接做大规模实验表，例如 6 个 D4RL 环境、3 个 seed、很多 baseline 全铺。原因是：我们还没有确定自己的有效机制，过早扩实验主要会花钱和时间，但不一定产生贡献。

当前更合理的策略是：

> 先用一个关键 replay 数据集、一个 seed、较短训练步数做机制筛选；只有方法出现明确正信号后，再补 seed、补环境、补 paper-ready 表格。

## 2. 为什么先选 `hopper-medium-replay-v2`

我们选择 `hopper-medium-replay-v2`、seed0 作为第一道 gate，不是因为它能代表最终结论，而是因为它适合快速暴露方法差异。

理由：

- TD3+BC 在 medium 类数据上大致正常，但在 replay 类数据上偏低。
- replay 数据质量混杂，更容易看出 policy/value regularization 是否真的有用。
- 单 env / 单 seed 成本低，适合快速筛掉没信号的方法。
- 如果一个方法在这个 gate 下完全没起色，通常不值得立刻扩到多 seed / 多环境。

这个选择是探索期策略，不是最终论文评估协议。

## 3. 已完成的公共基础

项目底座已经基本搭好：

- 统一 D4RL loader：`common.data.D4RLDataset`
- 统一 evaluator：`common.eval.eval_episodes`
- 统一结果写入：`common.eval.write_result`
- 统一 normalized score 口径
- AutoDL 4090 环境已搭好，可以关机保留
- Aim 本地追踪可用，W&B 可选

重要修复：

- loader 已统一走 `d4rl.qlearning_dataset`
- 避免 raw D4RL dataset 的 timeout boundary 导致 transition 口径不一致

已核对 transition 数：

| 数据集 | transition 数 |
|--------|---------------:|
| `hopper-medium-replay-v2` | 401,598 |
| `halfcheetah-medium-v2` | 999,000 |

这个修复对整个项目很关键，因为后续所有方向都应该共用同一套 data / eval 口径。

## 4. 第一轮基础实验

设置：`hopper-medium-replay-v2`，seed0，50k steps，eval every 10k，5 eval episodes。

| 方法 | Final | Best | 结果判断 |
|------|------:|-----:|----------|
| BC | 17.86 | 32.26 | 便宜 anchor，但波动较大 |
| TD3+BC | 22.43 | 22.43 | replay 上偏低，是弱 anchor |
| TD3+BC alpha5 | 21.95 | 21.95 | 简单加大 Q 项没有帮助 |
| ReBRAC-lite | 34.48 | 34.48 | 有正信号，作为简单强 baseline 保留 |

决策：

- TD3+BC 保留为弱 baseline。
- ReBRAC-lite 保留为简单强 baseline。
- TD3+BC alpha sweep 暂停，至少 alpha5 没有给出信号。
- 不因为 BC 某个中间 step 高就过度解读，因为它波动大。

## 5. 官方源码筛选

同样先只跑 `hopper-medium-replay-v2`、seed0、50k。

| 方法 | Final | Best | 结果判断 |
|------|------:|-----:|----------|
| PRDC official source | 23.54 | 23.54 | 基本接近 TD3+BC，没有明显信号 |
| A2PR official source | 22.31 | 22.81 | 没有明显超过 TD3+BC |
| SSAR official source-localized | 38.56 | 43.97 | 当前最强现代 baseline，但 full IQL-qv 预筛较贵 |

决策：

- PRDC / A2PR 暂时不扩多 seed、多数据集。
- SSAR 保留为当前强 reference。
- 官方源码可用于复现/参考，但无 license 的源码不要直接 vendor 到公开仓库。

为什么这样决策：

- PRDC / A2PR 当前没有表现出值得扩实验的信号。
- SSAR 虽然强，但成本主要在 IQL-qv trusted action selection。
- 与其扩 PRDC/A2PR，不如先解释 SSAR 为什么强。

## 6. SSAR 机制 ablation

设置：`hopper-medium-replay-v2`，seed0，100k。

| 变体 | Final | Best | 结果判断 |
|------|------:|-----:|----------|
| SSAR cached IQL-qv | 92.44 | 100.98 @90k | 非常强，说明 50k 高点不是偶然 |
| cheap SSAR without IQL selection | 25.48 | 30.34 @90k | 去掉 IQL-qv selection 后明显掉分 |
| ReBRAC-lite 100k | 36.54 | 54.36 @90k | 有一定峰值，但离完整 SSAR 很远 |

关键结论：

> SSAR 的收益很大程度来自 IQL-qv trusted action selection。这个环节不是工程细节，而是当前最重要的机制线索。

决策：

- 不继续把精力放在“我们也跑了 SSAR”上。
- 主线应转为：能否更便宜地获得类似 IQL-qv selection 的高质量 value / action-selection 信号。
- SSAR seed0 IQL-qv cache 要保留并复用，不要重复跑昂贵预筛。

## 7. 本地 cheap selector 尝试

我们尝试了几个更便宜的替代机制，但目前都不能作为最终方法。

| 变体 | 设置 | Final | Best | 结果判断 |
|------|------|------:|-----:|----------|
| return-ranked selector | 100k | 28.76 | 45.13 @50k | 有局部峰值，但不稳定 |
| Q-gap online selector | 50k | 19.94 | 22.05 @10k | 失败，不跑 100k |
| behavior consistency selector | 50k | 19.77 | 28.00 @40k | 失败，不跑 100k |

解释：

- return-ranked selector 说明 cheap trust weighting 可能会动曲线，但不稳定。
- Q-gap online selector 依赖当前 TD3 critic，critic 太弱/太噪，不足以做 trusted-action selector。
- behavior consistency 太保守，不能替代 value-quality 信号。

决策：

- 不扩这些 cheap selector 的 seed。
- 不跑它们的 100k。
- 不做大规模超参搜索。
- 下一步如果继续 selector 方向，需要更强的 offline value source。

## 8. 补充的 IQL / CQL 公共 baseline

为了判断 SSAR 的结果到底是 SSAR 机制强，还是 IQL/value 底座本身强，我们补了两个公共 value baseline。

设置仍然是 `hopper-medium-replay-v2`、seed0、5 eval episodes。

| 方法 | Steps | Final | Best | 结果判断 |
|------|------:|------:|-----:|----------|
| IQL compact baseline | 100k | 45.27 | 81.28 @80k | IQL 本身很强，能解释 SSAR 的一部分收益 |
| CQL compact baseline | 50k | 39.81 | 39.81 @50k | 可作为保守 value baseline anchor，但不是当前主线 |

新的判断：

- 不能再把 SSAR 只和 TD3+BC 这种弱 baseline 比，然后说 SSAR 有很大提升。
- IQL 是必须纳入讨论的强 baseline。
- 但 SSAR cached 仍然达到 92.44 final / 100.98 best，而且去掉 IQL selection 后只有 25.48 final。
- 所以真正的问题不是“SSAR 是否比弱 baseline 强”，而是：**能否低成本复现或摊销 IQL 派生的 trusted-action signal**。

## 9. 当前不该做什么

目前明确不建议：

- 直接跑 6 env x 3 seeds 全量 baseline 表。
- 扩 PRDC / A2PR 多 seed、多环境。
- 继续扫 TD3+BC alpha。
- 给失败的 Q-gap / consistency selector 做大量超参搜索。
- 把所有时间花在复现更多 baseline 上。

这些事情不是永远不做，而是现在做性价比低。等有明确方法信号后再补。

## 10. 当前应该做什么

### 项目层面

1. 统一实验协议：
   - 新方法先跑 `hopper-medium-replay-v2`、seed0、50k。
   - 过 gate 再跑 100k。
   - 100k 稳定后再补 seed1 或第二个 replay env。

2. 统一 anchor：
   - TD3+BC：弱 baseline。
   - ReBRAC-lite：简单强 baseline。
   - IQL：强 value baseline。
   - CQL：保守 value anchor。
   - SSAR：当前强 reference，但必须和 IQL 一起解释。

3. 统一代码接口：
   - A/B/C 各方向都接 `common.data` / `common.eval` / JSONL schema。

### 个人技术主线

更值得继续想的是：

> 如何低成本获得类似 SSAR IQL-qv selection 的高质量 value / action-selection 信号？

可能路线：

- light critic / value pretraining
- 用 SSAR/IQL 生成 trusted labels，再训练便宜 selector
- amortize IQL-qv labels，让一次昂贵预筛变成可复用模型
- 把 value-quality 信号接入多个算法方向，而不是只服务某个 C 方向变体

## 11. 当前会议要推动的决策

今晚和组员讨论时，重点不是展示所有细节，而是推动这些项目规则：

1. 当前阶段不做 6 env x 3 seeds。
2. 所有新方法先过 `hopper-medium-replay-v2` seed0 50k gate。
3. IQL 必须作为强 baseline 纳入解释，不能只和 TD3+BC 比。
4. 统一使用同一套 loader / evaluator / result schema。
5. TD3+BC / ReBRAC-lite / IQL / CQL / SSAR 作为公共 anchor。
6. 明确谁维护 benchmark 协议和 result tracker。
7. 下一步围绕 value / trusted-action signal 找项目级贡献。

## 12. 当前一句话方向

如果要把现在的项目状态压成一句话：

> 我们已经搭好统一实验底座，并通过一轮 pilot 发现：IQL 是强 baseline，SSAR 的优势不能只和 TD3+BC 比；真正值得追的是如何低成本复现或摊销 IQL/SSAR 暴露出的高质量 trusted-action signal，而不是现在盲目扩 baseline 表。

## 13. 2026-05-14 更新：C 线已经补上 offline-to-online

现在 C 线已经不只是 offline 分数了，已完成 `hopper-medium-replay-v2` 的 P0 offline-to-online eval20 panel：

| 方法 | Seed | Offline Final | Online Final | Online Best | 判断 |
|------|-----:|--------------:|-------------:|------------:|------|
| TD3+BC release | 0 | 22.20 | 40.06 | 40.06 | offline 弱，但在线能提升 |
| ATLAS release | 0 | 46.70 | 37.50 | 38.79 | offline 强，但 online final 掉下去 |
| SSAR/IQL-qv release | 0 | 50.71 | 28.87 | 31.85 | teacher 最强 offline，但 online final 最差 |
| SSAR/IQL-qv fixed | 0 | 50.71 | 38.61 | 96.22 | 有 near-100 spike，但 tail 不稳 |
| ATLAS Q-gate fixed | 0 | 46.70 | 48.41 | 48.41 | seed0 修复 fixed failure |
| TD3+BC release | 1 | 20.19 | 98.86 | 98.86 | 暴露 Hopper O2O 高方差 |
| ATLAS release | 1 | 31.21 | 84.35 | 84.35 | seed1 也能强，但仍低于 TD3+BC |
| ATLAS Q-gate fixed | 1 | 31.21 | 39.91 | 70.03 | 比 fixed 好，但不如 release |

新的结论：

> ATLAS/SSAR 的 trusted-action labels 对 offline initialization 有价值，但固定带到 online fine-tuning 会出现 constraint-transfer gap。Q-filtered trust 是一个有希望的诊断方向，但目前不是稳健新 SOTA。

下一步对 Tree 自己最重要的是：

1. 不再花主力做 C 线 baseline sweep。
2. 等 A/B 线补可比 O2O anchor，或者把论文改成更聚焦的 C-line mechanism paper。
3. 如果还跑 C 线，只跑能解释 online constraint transfer 的小实验，不跑 PRDC/A2PR 扩表。
