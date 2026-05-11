# Taoyao Offline RL

> 组内统一的 Offline RL / Offline-to-Online RL 项目代码库。  
> English version: [README_EN.md](README_EN.md)

这个仓库不是一开始就把所有算法都写好，而是先把**全组共用的实验底座**搭起来：

- 统一 Linux + CUDA + MuJoCo + D4RL 环境
- 统一 D4RL 数据加载方式
- 统一评估函数和 normalized score 计算
- 统一结果 JSONL schema
- 统一 seed / eval episodes / 训练步数约定
- 默认本地 Aim 实验面板，可选 W&B 云端同步
- 提供一个已经真实跑通的 BC smoke test

这样 A/B/C 各条线后面接算法时，最后的结果可以直接放在一起比较。

## 当前状态

这个仓库已经在 AutoDL RTX 4090 48G 上真实 smoke test 通过，环境为：

- Python 3.10
- CUDA 11.8
- PyTorch 2.1.2
- MuJoCo 2.1
- D4RL 1.1

`hopper-medium-v2` 实测结果：

```text
torch 2.1.2+cu118 | CUDA: True
gym.make(hopper-medium-v2) OK
D4RL dataset: 1,000,000 transitions
BC 2k steps: loss 0.444 -> 0.119
normalized_score = 38.4
ALL PASSED (6/6)
```

如果你在服务器上能跑通 `python smoke_test.py`，说明本机环境、MuJoCo、D4RL 数据、训练和评估链路都基本 OK。

## 项目分工

| 方向 | 主要算法 | 说明 |
|---|---|---|
| A · Value Conservatism | CQL, IQL, Cal-QL | 偏 value pessimism / conservative Q；输出仍需走统一 evaluator。 |
| B · New SOTA Extensions | DMG, SCQ | 较新的高风险方向；重点是把结果接到统一数据和评估协议。 |
| C · Policy Regularization / O2O | TD3+BC, ReBRAC, PRDC, A2PR | 偏 policy regularization 和 offline-to-online fine-tuning，是 D4RL MuJoCo 主线之一。 |

无论谁负责哪个算法，都尽量遵守同一套接口：

- `common.data.D4RLDataset`
- `common.eval.eval_episodes`
- `common.eval.write_result`
- `common.seed.set_seed`

## 为什么要统一代码库

Offline RL 很容易因为一些细节不同导致结果不能比，例如：

- D4RL timeout / terminal 处理不一致
- normalized score 计算口径不一致
- eval episode 数不同
- seed 没统一
- 每个人自己写一套 data loader
- 每个人输出的结果格式不同

本仓库的核心作用就是把这些公共部分先固定住。后面各自接算法时，不要重复造 data / eval / seed 轮子。

## 环境要求

请使用 Linux GPU 服务器运行。本地 macOS 适合写代码和看结果，不建议训练。

推荐环境：

- Ubuntu / Debian-like Linux
- Python 3.10
- CUDA 11.8+
- NVIDIA GPU，建议 24GB 显存以上
- AutoDL / Vast.ai / 自建 Linux GPU 服务器均可

注意：D4RL 仍然依赖 legacy `mujoco-py`，所以这里统一使用 Python 3.10，不支持 Python 3.11。

## 快速开始

在 Linux GPU 服务器上运行：

```bash
git clone https://github.com/treerobin06/taoyao-rl.git
cd taoyao-rl

bash setup_env.sh
source .venv/bin/activate

python download_d4rl.py hopper-medium-v2
python smoke_test.py
```

如果 `python smoke_test.py` 最后显示：

```text
ALL PASSED (6/6)
```

就说明这台机器可以开始接算法和跑实验。

如果想一次性下载全部配置好的 MuJoCo 数据集：

```bash
python download_d4rl.py
```

如果想跑一个 3 seeds 的 BC baseline 验证完整 pipeline：

```bash
bash scripts/run_bc.sh
```

## Smoke Test 会检查什么

`smoke_test.py` 会做一个小但完整的端到端检查：

1. PyTorch + CUDA 是否可用
2. `gym.make("hopper-medium-v2")` 是否成功
3. MuJoCo env reset / step 是否正常
4. D4RL dataset 是否能下载和读取
5. BC 是否能训练 2,000 step
6. 是否能用统一 evaluator 得到 normalized score

长实验前必须先跑通这个脚本。

## 目录结构

```text
.
├── README.md
├── README_EN.md
├── AGENTS.md
├── requirements.txt
├── setup_env.sh
├── download_d4rl.py
├── envs.txt
├── smoke_test.py
├── configs/
│   └── shared.yaml
├── common/
│   ├── data.py
│   ├── eval.py
│   ├── seed.py
│   └── tracking.py
├── algorithms/
│   ├── README.md
│   ├── bc.py
│   └── td3_bc.py
├── scripts/
│   ├── aim_ui.sh
│   ├── export_wandb.py
│   ├── run_bc.sh
│   └── run_td3_bc_pilot.sh
├── results/
│   └── README.md
└── notebooks/
    └── README.md
```

## 全组约定

### 数据

- 所有算法必须用 `common.data.D4RLDataset` 加载数据。
- 数据集只用 `envs.txt` 里列出的环境；新增环境前先组内讨论。
- 不要提交 D4RL 数据、checkpoint、`.aim/`、wandb 目录或实验结果 JSONL。

### 评估

- 所有算法必须用 `common.eval.eval_episodes` 做评估。
- 所有评估记录必须用 `common.eval.write_result` 写入 `results/`。
- normalized score 统一用 `env.get_normalized_score(raw_return) * 100`。

### Seeds

- 默认 seeds: `0, 1, 2`。
- 关键 ablation 有时间则扩到 5 seeds。
- 每个训练脚本必须调用 `common.seed.set_seed(seed)`。

### 训练协议

- Offline training 默认 1M gradient steps。
- Online fine-tuning 默认 100k environment steps。
- Offline eval 默认每 5k steps 一次。
- Online eval 默认每 1k steps 一次。

### 实验追踪：Aim + W&B

本仓库默认使用：

| 用途 | 工具 | 是否需要账号 | 默认状态 |
|---|---|---:|---:|
| 本地曲线和 run 对比 | Aim | 不需要 | 开启 |
| 云端同步和远程查看 | W&B | 需要 | 关闭 |
| 最终聚合和复现备份 | JSONL | 不需要 | 开启 |

Aim 是默认本地面板。直接跑脚本会在项目目录生成 `.aim/`：

```bash
bash scripts/run_td3_bc_pilot.sh
```

打开本地 Aim UI：

```bash
bash scripts/aim_ui.sh
```

浏览器访问输出的地址，默认是：

```text
http://127.0.0.1:43800
```

如果只想写 JSONL、不想写 Aim：

```bash
USE_AIM=0 bash scripts/run_td3_bc_pilot.sh
```

W&B 作为可选云端同步，适合需要远程看 AutoDL 进度时使用。

- wandb project 名统一为 `taoyao-rl`。
- run 名格式：`<algo>_<env>_s<seed>`，例如 `td3_bc_hopper-medium-v2_s0`。
- 每个人用自己的 wandb 账号登录，不要提交 token。
- `WANDB_ENTITY` 可选：个人实验可以不设；如果组里建了 team，就设成 team 名。

首次使用：

```bash
cp .env.example .env.local
bash scripts/setup_wandb.sh
```

`setup_wandb.sh` 做的是用户级 W&B 登录；API key 由 W&B CLI 保存到用户目录，之后本机其他项目也能复用。不要把真实 key 写进 `.env.local` 或仓库文件。

打开 wandb 记录：

```bash
USE_WANDB=1 bash scripts/run_td3_bc_pilot.sh

# 如果要明确写入某个 team / workspace
WANDB_ENTITY=<your-team-or-username> USE_WANDB=1 bash scripts/run_td3_bc_pilot.sh
```

把 wandb runs 导回本地：

```bash
python scripts/export_wandb.py --entity <your-team-or-username> --project taoyao-rl
```

导出文件默认在 `results/wandb_export/`，包括 `runs_summary.csv`、每个 run 的 config/summary/history。该目录被 git ignore，不要提交 token、wandb 目录或导出的私有实验数据。

## 如何接入新算法

现有 `algorithms/bc.py` 是接口模板：

```bash
python -m algorithms.bc --env hopper-medium-v2 --seed 0 --steps 50000
```

新算法建议：

- 放在 `algorithms/<algo>.py`
- 尽量保持单文件 CORL-style，方便对照和修改
- CLI 参数对齐 `algorithms/bc.py`
- 使用 `D4RLDataset`, `eval_episodes`, `write_result`, `set_seed`
- 输出结果到 `results/`
- 如需实验曲线，使用 `ExperimentLogger`，不要在算法里重复写 Aim/W&B 逻辑

更多计划中的算法迁移说明见 [algorithms/README.md](algorithms/README.md)。

## 当前配置的数据集

主数据集是 D4RL MuJoCo v2：

- `hopper-medium-v2`
- `hopper-medium-replay-v2`
- `halfcheetah-medium-v2`
- `halfcheetah-medium-replay-v2`
- `walker2d-medium-v2`
- `walker2d-medium-replay-v2`

AntMaze 暂时在 `envs.txt` 里注释掉，等组内确认要扩展再启用。

## 算力估算

RTX 4090 单 seed 粗略估计：

| 算法 | Offline 1M steps | Online 100k steps | 单 seed 总时长 |
|---|---:|---:|---:|
| BC | 20 min | - | 20 min |
| TD3+BC | 40 min | 20 min | ~1 h |
| ReBRAC | 50 min | 25 min | ~1.25 h |
| PRDC | 45 min | 25 min | ~1.2 h |
| A2PR | 60 min | 30 min | ~1.5 h |
| DMG | 50 min | 25 min | ~1.25 h |
| SCQ | 60 min | 30 min | ~1.5 h |

如果这是持续项目，AutoDL 上更推荐 setup 成功后关机保留实例，而不是立刻 release。这样下次可以复用环境、数据集缓存和 `mujoco-py` 编译结果。

## 常见问题

### MuJoCo 下载慢或失败

`setup_env.sh` 会尝试多个 MuJoCo 下载地址，也支持手动指定本地 tarball：

```bash
MUJOCO_TARBALL=/path/to/mujoco210-linux-x86_64.tar.gz bash setup_env.sh
```

AutoDL 上推荐缓存路径：

```text
/root/autodl-tmp/mujoco210-linux-x86_64.tar.gz
```

### `GLIBCXX_3.4.30 not found`

AutoDL conda 镜像可能会优先加载旧版 `libstdc++.so.6`。脚本里已经用 `LD_PRELOAD` 优先加载系统 libstdc++，一般不需要手动处理。

### `hopper-medium-v2` 不存在

D4RL 的 MuJoCo 环境注册依赖 `mjrl`。`setup_env.sh` 会先安装 `D4RL==1.1`，再安装 `mjrl`。如果 GitHub 访问失败，需要挂代理或手动安装 `mjrl`。

### macOS 能跑吗

不建议。macOS 上 MuJoCo + D4RL legacy 依赖很容易出问题。本项目默认只支持 Linux GPU 服务器。

## 公开仓库注意事项

本仓库不会提交：

- D4RL HDF5 数据集
- checkpoint
- `.aim/`
- wandb 目录
- result JSONL
- 本地 token、SSH 配置或个人凭据

仓库里只保留源码、配置和轻量文档。
