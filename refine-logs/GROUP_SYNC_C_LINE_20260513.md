# C 线进展与路线同步

日期：2026-05-13  
用途：发给组员同步当前状态、实验结果和后续方向  
状态：路线同步，不是正式论文稿

## 一句话结论

我这边建议项目主线先保持原题：**低质量离线数据下，不同 conservatism / regularization 设计如何影响 offline-to-online RL 的迁移效率**。  

C 线目前聚焦 **policy / behavior regularization**，同时把 SSAR 的机制诊断和 ATLAS 轻量 trusted-action selector 作为扩展方向。现在还不急着写正式论文稿，先把方向、结果表和后续实验计划固定下来，等 A/B/C 三条线结果合并后再统一写 LaTeX。

## 项目主线理解

原始项目不是单纯追一个新算法，也不是只做 baseline 复现，而是：

1. 先用低质量 D4RL MuJoCo v2 离线数据做 offline pretraining；
2. 再做 online fine-tuning；
3. 比较不同保守性 / 正则化设计对迁移效率、最终性能和稳定性的影响；
4. 最后用少量消融解释为什么某类方法更适合低质量数据。

因此，后续报告最好围绕这个问题组织：

> 低质量离线数据会放大保守性设计的影响：过弱约束可能不稳，过强约束可能限制在线微调；不同算法的 regularization 机制会导致不同的 offline-to-online transfer behavior。

## C 线定位

C 线不建议和 A 线重复做 CQL/value conservatism，也不建议把 IQL 当作自己的主创新。

我这边建议 C 线定位为：

> policy / behavior regularization family under low-quality replay data.

当前涉及的方法：

| 方法 | 角色 | 当前判断 |
|---|---|---|
| TD3+BC | anchor baseline | 必须保留，但不是创新点 |
| ReBRAC-lite | 强 regularization baseline | 比 TD3+BC 明显强，是稳定参考 |
| PRDC / A2PR | 近年 policy regularization 方法 | 已跑 smoke，但当前 hopper-replay 信号不强 |
| IQL / CQL | public value baselines | 用来判断 SSAR/ATLAS 是否只是 value baseline 效果 |
| SSAR | 强 teacher / mechanism anchor | 强但成本高，依赖 IQL-qv trusted action selection |
| ATLAS | 我们的轻量机制尝试 | 目前最像 C 线的局部 contribution，但不能 claim SOTA |

## 已完成实验概况

当前主要实验集中在 `hopper-medium-replay-v2` 和 `walker2d-medium-replay-v2`，都是探索阶段 smoke / mechanism check，不是最终多 seed 完整表。

### 1. 基础 C 线方法筛选

在 `hopper-medium-replay-v2`, seed0, 50k steps 下：

| 方法 | Final normalized score | Best normalized score | 结论 |
|---|---:|---:|---|
| TD3+BC | 22.43 | 22.43 | 基础 anchor，效果偏低 |
| ReBRAC-lite | 34.48 | 34.48 | 当前较稳的 regularization baseline |
| PRDC official source | 23.54 | 23.54 | 能跑通，但没有明显优于 TD3+BC |
| A2PR official source | 22.31 | 22.81 | 能跑通，但当前没有明显正信号 |
| SSAR official source-localized | 38.56 | 43.97 | 50k 下最强，但 full IQL-qv 预筛成本高 |

这说明：继续盲目扩大 PRDC/A2PR 的多 seed 暂时不划算，更应该先理解 SSAR 为什么强。

### 2. SSAR / IQL-qv 机制诊断

关键发现：SSAR 的强信号很大程度来自 **IQL-qv trusted action selection**。

| 实验 | Final | Best | 结论 |
|---|---:|---:|---|
| SSAR cached IQL-qv, hopper seed0, 100k | 92.44 | 100.98 | 很强，但 seed0/eval5 有高方差 |
| SSAR full IQL-qv, hopper seed1, eval20, 100k | 60.88 | 99.22 | 仍能冲到接近 100，但 final/tail 不稳定 |
| cheap SSAR without IQL action selection | 25.48 | 30.34 | 去掉 IQL-qv 后明显塌掉 |
| IQL compact baseline | 45.27 | 81.28 | IQL 本身也很强，必须作为 fair baseline |
| CQL compact baseline | 39.81 | 39.81 | 有必要作为 value-conservatism 参考 |

当前判断：SSAR 不能只和弱 TD3+BC 比。更准确的问题是：**能不能用更低成本复现或近似 SSAR/IQL-qv 的 trusted-action signal？**

### 3. ATLAS 当前结果

ATLAS 的思路是：把 SSAR/IQL-qv 产生的 trusted labels 导出，用一个轻量 `(state, action) -> trust score` selector 学出来，再把这个 score 用作 TD3+BC-style behavior regularization 的权重。

当前结果：

| 实验 | Final | Best | 结论 |
|---|---:|---:|---|
| ATLAS hopper seed0, 100k | 69.97 | 69.97 | seed0 正信号，不像 top-return selector 那样 100k 崩掉 |
| ATLAS hopper seed1, eval20, 100k | 68.11 | 68.11 | 和 seed0 final 接近，说明不是单 seed 偶然 |
| ATLAS shuffled-label ablation, 50k | 18.78 | 19.35 | 打乱 label 后明显下降，说明不是普通 weighted BC 分布起作用 |
| ATLAS walker2d seed0, 100k | 71.26 | 77.86 | 第二个环境也有正信号，但仍落后 SSAR |
| SSAR walker2d seed0, 100k | 94.28 | 94.60 | SSAR 仍是强 teacher，ATLAS 尚未替代它 |

结论：ATLAS 有机制价值，但现在不能说是 SOTA。更合理的说法是：

> ATLAS shows that SSAR/IQL trusted-action information can be partially distilled into a lightweight selector, but it remains a lower-cost approximation rather than a full replacement for SSAR.

## 当前不建议做的事

1. 不建议现在跑 6 个数据集 x 3 seed 的大 sweep。  
   当前还在探索阶段，大 sweep 成本高，而且不会自动产生贡献。

2. 不建议把 ATLAS 包装成最终主算法。  
   它现在是机制扩展，不是已经全面打败 SSAR 的新 SOTA。

3. 不建议 C 线继续深挖 CQL-family。  
   CQL/value conservatism 更适合 A 线负责，C 线应该保留 policy regularization 边界。

4. 不建议现在写完整 LaTeX 正文。  
   其他线结果还没合并，现在先写 Markdown 路线、结果表和实验计划更合适。

## 接下来 P0

1. 补一个 online fine-tuning 最小闭环。  
   原项目核心是 offline-to-online，目前我们 offline / mechanism 结果多，online fine-tuning 曲线还不够。建议先在 `hopper-medium-replay-v2` 上选 2-3 个代表方法短跑。

2. 固定 C 线结果总表。  
   把 TD3+BC / ReBRAC-lite / IQL / CQL / SSAR / ATLAS 放到同一张表里，列 env、seed、steps、eval episodes、final、best、interpretation。

3. 只补一个 ATLAS 高价值 ablation。  
   优先尝试 soft vs hard labels、`label_min_weight` 或 selector capacity，用来解释 ATLAS 和 SSAR 的差距。

4. 和组员确认分工边界。  
   A 线：CQL / value conservatism；B 线：normal / non-conservative contrast，例如 PPO/SAC-style online baseline 或 vanilla TD3-style online fine-tuning；C 线：policy regularization + SSAR/ATLAS mechanism；公共部分：统一 evaluator、统一画图、统一 offline-to-online 记录格式。

## 接下来 P1

1. 做 cost/time 对比。  
   比较 SSAR full IQL-qv preselection 和 ATLAS selector 的时间/成本，这是 ATLAS 最容易站住的点之一。

2. 在 walker2d 上补一个 ATLAS ablation。  
   walker 上 SSAR 94、ATLAS 71，差距明显，适合定位 teacher-label 使用方式的问题。

3. 只有在关键结果明确后再补 seed。  
   seed 应该服务于关键 claim，而不是替代实验设计。

4. 准备两张图。  
   一张 C 线结果图；一张机制图：SSAR/IQL teacher -> trust labels -> ATLAS selector -> weighted TD3+BC。

## 可以发群的短版

我这边先同步一下 C 线状态：我建议项目主线还是保持原题，也就是低质量离线数据下，不同 conservatism / regularization 设计对 offline-to-online RL 迁移效率的影响。C 线这边我主要聚焦 policy / behavior regularization，不和 CQL/value conservatism 那条线重复。

目前已经跑了一批 hopper-medium-replay-v2 和 walker2d-medium-replay-v2 的探索实验。基础方法里 ReBRAC-lite 比 TD3+BC 明显强，PRDC/A2PR 目前 smoke 信号不明显；SSAR 很强，但主要依赖 IQL-qv trusted action selection，预筛成本比较高。基于这个发现，我做了一个轻量化尝试 ATLAS，把 SSAR/IQL 的 trusted labels 蒸馏成 `(state, action) -> trust score` selector，再用于 weighted TD3+BC。ATLAS 在 hopper seed0/seed1 上 final 都在 68-70 左右，walker 上也有 71 左右；打乱 labels 后会掉到 18 左右，说明它确实依赖 teacher-label alignment，不只是普通 weighted BC。

所以我现在的判断是：ATLAS 可以作为 C 线的机制扩展，但还不能 claim 成新 SOTA。接下来我建议先不要做 6 环境 x 3 seed 的大 sweep，而是优先补一个最小 online fine-tuning 闭环，再做 1-2 个关键 ATLAS ablation，同时把 C 线结果表和路线文档整理好。等 A/B/C 三条线结果合并之后，再统一写正式 LaTeX 报告。
