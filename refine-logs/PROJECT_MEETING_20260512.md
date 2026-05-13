# RL 项目组会 Brief

**日期**：2026-05-12  
**定位**：项目级同步，不是 C 方向单独汇报  
**当前阶段**：探索期；先找有效机制，不急着做 paper-ready 全量表

## 1. 核心结论

我们已经把统一实验底座跑通，也完成了一轮小规模 pilot。当前最重要的判断是：

> 不建议现在直接做 6 env x 3 seeds 的大 baseline 表。下一步应先统一实验协议，用单环境、单 seed、短训练快速筛出真正有机制信号的方法；有信号后再补 seed 和更多数据集。

## 2. 项目已完成的公共基础

- 统一 D4RL loader：`common.data.D4RLDataset`
- 统一 evaluator：`common.eval.eval_episodes`
- 统一结果 JSONL：`common.eval.write_result`
- 统一 normalized score / seed / eval episode 口径
- AutoDL 环境已搭好，可关机保留、下次继续用
- Aim 可本地看曲线，W&B 可选

这意味着 A/B/C 各方向后续应该接同一个 pipeline，不要各写各的 loader 和 evaluator。

## 3. 当前最关键的实验事实

第一轮 pilot 统一用 `hopper-medium-replay-v2`、seed0，目的是快速筛方向，不是最终论文结论。

### 基础 anchor

| 方法 | Final | Best | 判断 |
|------|------:|-----:|------|
| TD3+BC | 22.43 | 22.43 | replay 上偏低，作为弱 anchor |
| ReBRAC-lite | 34.48 | 34.48 | 简单强 baseline，应保留 |

### 官方源码筛选

| 方法 | Final | Best | 判断 |
|------|------:|-----:|------|
| PRDC | 23.54 | 23.54 | 暂无明显信号 |
| A2PR | 22.31 | 22.81 | 暂无明显信号 |
| SSAR | 38.56 | 43.97 | 当前最强现代 baseline，但预筛成本高 |

### SSAR 机制结果

| 变体 | Final | Best | 判断 |
|------|------:|-----:|------|
| SSAR cached IQL-qv | 92.44 | 100.98 | 非常强 |
| SSAR 去掉 IQL selection | 25.48 | 30.34 | 明显掉分 |
| ReBRAC-lite 100k | 36.54 | 54.36 | 有信号，但离 SSAR 远 |

项目级含义：**高质量 value / trusted-action selection 信号可能是最值得研究的机制**，比单纯扩大 baseline 表更重要。

### 补充公共 value baseline

| 方法 | Final | Best | 判断 |
|------|------:|-----:|------|
| IQL compact 100k | 45.27 | 81.28 | IQL 本身很强，必须纳入公平比较 |
| CQL compact 50k | 39.81 | 39.81 | 可作为 conservative value anchor |

补充后的判断：SSAR 的提升不能只和 TD3+BC 比；IQL/value signal 解释了一部分收益。但 SSAR cached 仍显著高于普通 IQL final，并且去掉 IQL selection 后会掉到 25.48，所以下一步应聚焦如何低成本复现或摊销 trusted-action signal。

## 4. 已经排除的方向

我们试了几个便宜 selector，但目前不能作为方法：

| 变体 | Final | Best | 判断 |
|------|------:|-----:|------|
| return-ranked selector | 28.76 | 45.13 @50k | 有峰值但不稳定 |
| Q-gap online selector | 19.94 | 22.05 | 失败 |
| behavior consistency selector | 19.77 | 28.00 | 失败 |

结论：简单在线 TD3 critic 或行为一致性信号不够。后续如果做 selector，应该考虑更强的 offline value source，例如 light critic pretraining、IQL/SSAR label amortization。

## 5. 建议的项目策略

### 现在先不要做

- PRDC / A2PR 多 seed、多数据集扩展
- 6 env x 3 seeds 全量 baseline 表
- TD3+BC alpha sweep
- 已失败 cheap selector 的超参搜索

### 现在应该做

1. **统一 benchmark 协议**
   - 新方法先跑 `hopper-medium-replay-v2`, seed0, 50k
   - 过 gate 再跑 100k
   - 100k 稳定后再补 seed1 或第二个 replay env

2. **保留公共 anchor**
   - TD3+BC：弱 baseline
   - ReBRAC-lite：简单强 baseline
   - SSAR：当前强 reference，但要复用 cache，避免重复 IQL-qv 预筛

3. **围绕机制设计下一步**
   - 重点不是“再跑哪个 baseline”
   - 重点是能否更便宜地获得类似 SSAR IQL-qv 的 value / action-selection 信号

## 6. 组会需要拍板的问题

1. 是否同意当前阶段不做 6 env x 3 seeds？
2. 是否统一把 `hopper-medium-replay-v2 seed0 50k` 作为新方法第一道 gate？
3. 是否把 TD3+BC / ReBRAC-lite / IQL / CQL / SSAR 作为公共 anchor？
4. 谁负责维护 benchmark 协议和 result tracker？
5. A/B/C 各方向谁负责接入统一接口？
6. 下一步是否围绕 value / trusted-action signal 做项目级贡献探索？

## 7. 建议分工

| 工作 | 目标 | 优先级 |
|------|------|--------|
| 统一实验协议 | 防止结果不可比 | 高 |
| 维护 SSAR cache / source 记录 | 避免重复昂贵预筛 | 高 |
| A/B/C 接入统一接口 | 让各方向能公平比较 | 高 |
| 设计 offline value / label 机制 | 形成潜在贡献 | 高 |
| 多 seed / 多 env | 有信号后再补 | 中 |
