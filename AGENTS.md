# Taoyao RL · 项目本地 Agent 指令

给 Claude/Codex/Cursor 等 AI 助手用。**只写本项目特化内容**，不重复全局规则。

## 项目身份

- 类型：高校小组 offline RL / offline-to-online RL 项目
- 算力：Linux GPU 服务器（AutoDL / Vast / 自建，Python 3.10）+ 本地 Mac 写代码
- 目标：多人协作下保证 cross-algorithm / cross-line 结果可比

## 强制规则

1. **不要写跨平台兼容代码**——只针对 Linux + CUDA + Python 3.10，macOS / Python 3.11 跑不通是预期
2. **算法实现必须用 `common.data.D4RLDataset` 和 `common.eval.eval_episodes`**——不要自己写 loader/eval
3. **不要硬编码 wandb entity、API key 或个人路径**——W&B 凭据统一走用户级 `wandb login`，项目只放 `WANDB_PROJECT` / `WANDB_ENTITY`
4. **不要新增数据集环境名**——只用 `envs.txt` 里列出的，要加先和组员讨论
5. **离线主训练 1M steps + online fine-tune 100k steps** 是默认值，不要随便改

## 算法引入流程

新增算法（如组员想加 IQL / CQL / Cal-QL）：
1. 阅读 `algorithms/README.md`
2. 单文件 CORL-style 放进 `algorithms/<name>.py`
3. CLI 接口对齐 `algorithms/bc.py`
4. 在 `hopper-medium-v2` 上跑 100k step smoke 后再扩展

## 失败排查顺序

```
环境问题  → 跑 smoke_test.py 看哪一级挂
数据问题  → 检查 D4RLDataset.size 和 shape 是否符合预期
训练不收敛 → 检查 set_seed 是否被调用，是否在 dataset 之前
分数偏低  → 对比 configs/shared.yaml > expected_scores
```

## 不要做

- ❌ 在本目录引入 d3rlpy / Tianshou / stable-baselines3（除非作为 online baseline 单独工具）
- ❌ 把 CORL 整个 framework 拷进来（只拷单个算法文件）
- ❌ 给每个算法搞独立 conda env（全组共用一个 venv）
- ❌ 自动 commit / push 代码（小组共享 repo，必须 review）
- ❌ 把数据 / checkpoint 提交进 git（`.gitignore` 已配好）

## 与全局规则的关系

本项目运行在通用 Linux GPU 服务器上，**不依赖任何个人凭据 / proxy / token**。
组员各自的 wandb account、SSH 配置、AutoDL token 等都在各人本地 `.env` 或用户级登录缓存里处理，不进本仓库。W&B 用 `wandb login` 做用户级配置，登录态由 W&B CLI 存在用户目录，所有项目可复用；不要把 W&B API key/token 明文写进 `AGENTS.md`、README、`.env.local` 或任何会提交的文件。
