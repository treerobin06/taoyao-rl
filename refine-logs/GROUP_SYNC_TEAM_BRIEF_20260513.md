# 小组同步：C 线当前状态与后续协作建议

日期：2026-05-13  
用途：发群同步我这边做了什么、当前对项目主线的判断、以及其他线大概还需要补什么  
状态：讨论用，不是正式论文稿

## 1. 我这边目前做了什么

我这边主要做了 C 线，也就是 **policy / behavior regularization** 相关方法的探索。现在还不是最终完整实验表，主要目标是先判断哪些方向值得继续做，避免太早跑很贵的大规模多 seed。

已经做的事情大概有：

- 整理了 C 线相关文献和候选方法，重点看 TD3+BC、ReBRAC、PRDC、A2PR、SSAR 这类 policy / behavior regularization 方法。
- 跑了 `hopper-medium-replay-v2` 上的一批 smoke / mechanism 实验：
  - TD3+BC 作为基础 anchor；
  - ReBRAC-lite 比 TD3+BC 明显更好；
  - PRDC / A2PR 能跑通，但目前 smoke 信号不强；
  - SSAR 效果最好，但它依赖 IQL-qv trusted action selection，预筛成本比较高。
- 额外补了 IQL / CQL 这类 value baseline，用来判断 SSAR 的提升是不是只是因为用了更强的 value backbone。
- 基于 SSAR 的机制诊断，做了一个轻量尝试 ATLAS：把 SSAR/IQL-qv 的 trusted labels 蒸馏成 `(state, action) -> trust score` selector，再用于 weighted TD3+BC。

ATLAS 现在的初步结果是：

- hopper seed0 / seed1 的 100k final 都在 68-70 左右；
- walker2d seed0 也有 71 左右；
- shuffled-label ablation 会掉到 18 左右，说明不是普通 weighted BC 起作用，而是 teacher-label alignment 真的重要；
- 但 ATLAS 还没有全面追上 SSAR，所以不能 claim 成 SOTA。

## 2. 我对项目主线的判断

我建议项目主线还是保留原来的方向：

> 低质量离线数据下，不同 conservatism / regularization 设计如何影响 offline-to-online RL 的迁移效率。

也就是说，最终报告不应该只写“我们提出 ATLAS”，也不应该只写“跑了一堆 baseline”。结合组内最新讨论，更合理的结构是：

- A 线代表 conservative value methods；
- B 线改成 normal / non-conservative contrast；
- C 线代表 policy / behavior regularization 和 trusted-action selection；
- 先比较它们在低质量数据下的 offline 表现；
- 再尽量补一个 offline-to-online fine-tuning 的最小闭环；
- 最后用少量机制实验解释为什么某些方法更强或更贵。

ATLAS 可以作为 C 线的一个机制扩展：说明 SSAR 的 trusted-action selection 很关键，并尝试用更低成本的方式近似它。但它现在不应该替代整个项目主线。

## 3. 其他线大概还需要补什么

这里不是给大家排很细的任务，只是为了后面能合并成一个统一报告，建议每条线至少补到同一种可比格式。

### A 线：value conservatism / CQL-family

建议主要回答：

- CQL 或类似显式保守方法在低质量 replay 数据上是否稳定；
- 它和 policy regularization 方法相比，是更稳还是更容易过保守；
- 如果能做 online fine-tuning，最好补一条短曲线，用来对齐项目主线。

最少需要交付：

- 1-2 个代表环境的 final / best normalized score；
- seed、训练步数、eval episodes；
- 简单说明：效果稳定、偏保守、还是训练不理想；
- 如果跑失败，也最好记录失败原因。

### B 线：normal / non-conservative contrast

建议主要回答：

- 如果不使用显式 conservatism / trusted-action regularization，正常学习器在同样任务上表现如何；
- normal baseline 是不是在线适应更快，但 offline 初始化更弱；
- 它和 A 线的 conservative value methods、C 线的 policy regularization 相比，差异在哪里。

最少需要交付：

- PPO/SAC/vanilla TD3-style online fine-tuning，或组内确认的其他非保守/弱正则 baseline；
- final / best normalized score 和曲线；
- 如果可能，补一个 online fine-tuning 小闭环；
- 简单说明：它是 offline 弱但 online 适应快，还是整体都不稳定。

### 公共部分：大家最好统一

为了最后能合并，建议大家尽量统一：

- 优先环境：`hopper-medium-replay-v2`、`walker2d-medium-replay-v2`；有余力再扩展；
- 先做 1 seed smoke，不要一开始就 6 环境 x 3 seed；
- 记录格式至少包含：method、env、seed、steps、eval episodes、final score、best score、log path；
- 重要曲线最好都能导出成 CSV / JSON / W&B 链接；
- online fine-tuning 至少要有一个最小闭环，否则原题的 offline-to-online 部分会比较弱。

## 4. 我这边接下来会做什么

我这边后续不会先展开大规模 sweep。C 线的最小 O2O 闭环已经补完，接下来主要是整理和合并：

- 把 C 线结果表、O2O 曲线和机制结论整理成可合并格式；
- 等 A 线 conservative value 结果和 B 线 non-conservative contrast 结果；
- 如果报告还缺机制解释，再考虑一个很小的 ATLAS offline label-usage ablation。

更细的 ATLAS 调参和多 seed 可以等大家结果合并后再决定是否值得继续烧。

## 5. 可以直接发群的版本

我这边先同步一下 C 线进展。C 线我主要在看 policy / behavior regularization 这类方法，和 A 线的 CQL/value conservatism 尽量错开。到目前为止，我整理并跑了一批 `hopper-medium-replay-v2` 和 `walker2d-medium-replay-v2` 的探索实验：TD3+BC 作为 anchor，ReBRAC-lite 比 TD3+BC 明显更好；PRDC/A2PR 能跑通但目前 smoke 信号不强；SSAR 效果最好，不过它主要依赖 IQL-qv trusted action selection，预筛成本比较高。

基于这个机制，我又做了一个轻量尝试 ATLAS：把 SSAR/IQL-qv 的 trusted labels 蒸馏成 `(state, action) -> trust score` selector，再用于 weighted TD3+BC。目前 ATLAS 在 hopper seed0/seed1 的 100k final 都在 68-70 左右，walker2d seed0 也有 71 左右；打乱 labels 后会掉到 18 左右，说明它不是普通 weighted BC，而是确实依赖 teacher-label alignment。不过 ATLAS 现在还不能 claim 成 SOTA，只能作为 C 线的机制扩展。

我对整体项目的建议是：主线还是保持原题，也就是低质量离线数据下，不同 conservatism / regularization 设计如何影响 offline-to-online RL 的迁移效率。结合刚才讨论，后续 A 线可以主要补 CQL/value conservatism 的可比结果，B 线改成 normal / non-conservative contrast，C 线保留 policy regularization + SSAR/ATLAS 机制分析；大家最好统一记录 method、env、seed、steps、eval episodes、final/best score 和曲线。现在先不建议直接做 6 环境 x 3 seed 的大 sweep，先各自把 1-2 个代表环境的结果和最小 online fine-tuning 闭环补齐，后面再合并写正式报告。
