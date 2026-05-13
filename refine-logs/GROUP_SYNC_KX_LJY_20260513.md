# Group Sync Draft For KX / LJY

Date: 2026-05-13  
Purpose: brief team status sync and next-step guidance based on the real WeChat discussion with KX / LJY.

## 这条消息是干什么

这条同步不是要大家现在开始大规模补实验，而是解决群里暴露出的两个问题：

1. 大家对主线有点混乱：到底是做原来的 offline-to-online empirical study，还是改成 C 线 / Gate-aware / ATLAS 这类新机制。
2. 分工还没说清：KX 问自己是不是 A 线，LJY 在跑 PPO/IQL，也提出了 Gate-aware release 规则的想法。

建议先把项目收回到原始主线：

> 低质量离线数据下，不同 conservatism / regularization 设计如何影响 offline-to-online RL 的迁移效率。

在这个主线下，比较稳的分工是：

- A 线：value conservatism，例如 CQL / Cal-QL；
- B 线：normal / non-conservative contrast，例如 PPO/SAC-style online baseline 或 vanilla TD3-style online fine-tuning；
- C 线：policy / behavior regularization 和 trusted-action selection，例如 TD3+BC / ReBRAC / SSAR / ATLAS。

## 可以发群里的短版

我刚看了一下我们现在的讨论，感觉需要先把主线和分工收一下，不然容易各跑各的。我的理解是，课程项目主线还是先保持原来的问题：低质量离线数据下，不同 conservatism / regularization 设计如何影响 offline-to-online RL 的迁移效率。ATLAS、Gate-aware release 这些可以作为后面的机制发现或扩展，但先不要替代整个项目主线。

分工上我建议先这样对齐：KX 这边可以偏 A 线，主要补 CQL/Cal-QL 这类 value conservatism 的可比结果；LJY 这边可以偏 B 线，继续把 PPO/SAC/vanilla TD3-style online fine-tuning 这类 normal / non-conservative contrast 跑清楚；我这边 C 线继续整理 TD3+BC、ReBRAC、SSAR、ATLAS 这类 policy regularization / trusted-action selection 的结果。

为了最后能合表，大家先别做 6 环境 x 多 seed 的大 sweep。先统一 `hopper-medium-replay-v2`、seed0、小步数 smoke，把 method、env、seed、offline steps、online steps、eval episodes、final/best normalized score、log/curve 路径和一句话解释记下来。等这个最小表能解释清楚，再决定要不要补 seed 和更多环境。

## 给 KX / LJY 的具体指导

KX：

1. 先按 A 线理解：value conservatism / CQL-family。
2. 当前优先级不是创新算法，而是把环境装好后跑出可合并的 CQL/Cal-QL 类结果。
3. 最小实验：`hopper-medium-replay-v2`、seed0、统一 evaluator，记录 final/best normalized score 和曲线。
4. 如果 CQL 跑不理想，也要记录失败原因；这可以支撑“保守 value 方法在低质量数据下是否过保守/不稳定”的分析。

LJY：

1. 你现在跑 PPO/IQL 的方向有用，但要把定位说清楚。
2. PPO 从零开始 online 学，50k 短步数差是正常风险，不能直接和看过 offline 数据的方法公平比较。
3. 更适合把 PPO/SAC/vanilla TD3-style online fine-tuning 当 B 线：normal / non-conservative contrast。
4. Gate-aware release 规则可以先作为想法保留，但建议等 A/B/C 最小可比表出来后再决定是否升级成主 contribution。

## 我们这边 C 线当前可同步结论

- TD3+BC 在 `hopper-medium-replay-v2` 上较弱，ReBRAC-lite 和 SSAR 说明 action / behavior regularization 是关键变量。
- SSAR 的 IQL-qv trusted action selection 很强，但 full preselection 成本高。
- ATLAS 能部分蒸馏 trusted-action signal，offline 结果有正信号，但当前 offline-to-online 结果显示 teacher-label regularization 可能会限制 online adaptation。
- 因此 C 线现在的 claim 应该保守写成机制发现，而不是 claim “ATLAS SOTA”。

## 会议上需要定的事

1. 主线是否继续采用原始问题：low-quality offline data 下，不同 conservatism / regularization 对 offline-to-online 迁移的影响。
2. KX 是否正式负责 A 线，LJY 是否正式负责 B 线。
3. Gate-aware release / ATLAS 是否只作为机制扩展，而不是替代主线。
4. 统一最小实验协议：env、seed、steps、eval episodes、score 口径和日志格式。
