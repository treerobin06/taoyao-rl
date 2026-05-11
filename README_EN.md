# Taoyao Offline RL

Unified codebase for a course project on **offline reinforcement learning** and
**offline-to-online fine-tuning**.

The goal of this repository is not to provide every algorithm implementation on
day one. It provides the shared infrastructure that every group member should use:
same environment, same D4RL datasets, same evaluation protocol, same result schema,
and a smoke-tested Behavior Cloning baseline.

## Current Status

This repository has been smoke-tested on an AutoDL RTX 4090 48G instance with
Python 3.10, CUDA 11.8, PyTorch 2.1.2, MuJoCo 2.1, and D4RL.

Verified smoke result on `hopper-medium-v2`:

```text
torch 2.1.2+cu118 | CUDA: True
gym.make(hopper-medium-v2) OK
D4RL dataset: 1,000,000 transitions
BC 2k steps: loss 0.444 -> 0.119
normalized_score = 38.4
ALL PASSED (6/6)
```

## Project Directions

| Track | Main algorithms | Notes |
|---|---|---|
| A · Value Conservatism | CQL, IQL, Cal-QL | Use the shared evaluator; avoid duplicating C-track policy-regularization work. |
| B · New SOTA Extensions | DMG, SCQ | Higher-risk recent methods; keep outputs compatible with `common.eval`. |
| C · Policy Regularization / O2O | TD3+BC, ReBRAC, PRDC, A2PR | Main policy-regularization family for D4RL MuJoCo and offline-to-online comparison. |

Every track should use `common.data.D4RLDataset`, `common.eval.eval_episodes`,
and `common.eval.write_result`, so final plots compare like with like.

## Why This Repo Exists

Offline RL experiments often fail because different people quietly use different:

- D4RL preprocessing rules
- timeout / terminal handling
- normalized score calculation
- seeds and evaluation episodes
- result JSON formats
- MuJoCo / D4RL install recipes

This repo makes those choices explicit and shared.

## Requirements

Use a Linux GPU server. macOS is fine for editing code, but this project does not
support local macOS training.

Recommended environment:

- Ubuntu / Debian-like Linux
- Python 3.10
- CUDA 11.8+
- NVIDIA GPU, 24GB+ VRAM preferred
- AutoDL, Vast.ai, or a self-managed Linux GPU server

Important: D4RL still depends on the legacy `mujoco-py` stack. Python 3.11 is not
supported here.

## Quick Start

```bash
git clone https://github.com/treerobin06/taoyao-rl.git
cd taoyao-rl

bash setup_env.sh
source .venv/bin/activate

python download_d4rl.py hopper-medium-v2
python smoke_test.py
```

If `python smoke_test.py` passes, the machine is ready for algorithm development.

To pre-download all configured MuJoCo datasets:

```bash
python download_d4rl.py
```

To run the BC baseline for 3 seeds:

```bash
bash scripts/run_bc.sh
```

## What `smoke_test.py` Checks

The smoke test is intentionally small but end-to-end:

1. PyTorch and CUDA are available.
2. `gym.make("hopper-medium-v2")` works.
3. MuJoCo reset/step works.
4. D4RL dataset loads and has expected shape.
5. Behavior Cloning trains for 2,000 updates.
6. Evaluation writes normalized score through the shared evaluator.

Do not start long experiments before this passes.

## Repository Layout

```text
.
├── README.md
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
│   └── seed.py
├── algorithms/
│   ├── README.md
│   └── bc.py
├── scripts/
│   └── run_bc.sh
├── results/
│   └── README.md
└── notebooks/
    └── README.md
```

## Shared Rules

Data:

- All algorithms must load datasets through `common.data.D4RLDataset`.
- Use only environments listed in `envs.txt`, unless the group agrees to extend the list.
- Do not commit D4RL data, checkpoints, wandb logs, or result JSONL files.

Evaluation:

- All algorithms must use `common.eval.eval_episodes`.
- All evaluation records must be written through `common.eval.write_result`.
- Normalized score must be `env.get_normalized_score(raw_return) * 100`.

Seeds:

- Default seeds are `0, 1, 2`.
- Important ablations should use 5 seeds if time allows.
- Every training script must call `common.seed.set_seed(seed)`.

Training protocol:

- Offline training: 1M gradient steps by default.
- Online fine-tuning: 100k environment steps by default.
- Offline eval frequency: every 5k steps.
- Online eval frequency: every 1k steps.

wandb:

- Project name: `taoyao-rl`.
- Run name: `<algo>_<env>_s<seed>`, for example `td3_bc_hopper-medium-v2_s0`.
- Everyone logs in with their own wandb account. Do not commit tokens.

## Adding an Algorithm

Use the existing BC implementation as the interface template:

```bash
python -m algorithms.bc --env hopper-medium-v2 --seed 0 --steps 50000
```

New algorithm files should:

- live under `algorithms/<algo>.py`
- follow a single-file CORL-style structure where practical
- expose CLI arguments compatible with `algorithms/bc.py`
- use `D4RLDataset`, `eval_episodes`, `write_result`, and `set_seed`
- write JSONL results into `results/`

See `algorithms/README.md` for the planned algorithm list and migration notes.

## Configured Datasets

Primary D4RL MuJoCo v2 datasets:

- `hopper-medium-v2`
- `hopper-medium-replay-v2`
- `halfcheetah-medium-v2`
- `halfcheetah-medium-replay-v2`
- `walker2d-medium-v2`
- `walker2d-medium-replay-v2`

AntMaze is intentionally commented out in `envs.txt` until the group decides to
expand the scope.

## Expected Compute

Approximate single-seed runtime on RTX 4090:

| Algorithm | Offline 1M steps | Online 100k steps | Total |
|---|---:|---:|---:|
| BC | 20 min | - | 20 min |
| TD3+BC | 40 min | 20 min | ~1 h |
| ReBRAC | 50 min | 25 min | ~1.25 h |
| PRDC | 45 min | 25 min | ~1.2 h |
| A2PR | 60 min | 30 min | ~1.5 h |
| DMG | 50 min | 25 min | ~1.25 h |
| SCQ | 60 min | 30 min | ~1.5 h |

For a continued project on AutoDL, it is usually better to power off the instance
after setup and reuse it later, instead of releasing it and rebuilding the full
environment from scratch.

## Troubleshooting

### MuJoCo download is slow or blocked

`setup_env.sh` tries multiple URLs and supports a local tarball cache:

```bash
MUJOCO_TARBALL=/path/to/mujoco210-linux-x86_64.tar.gz bash setup_env.sh
```

On AutoDL, a convenient cache path is:

```text
/root/autodl-tmp/mujoco210-linux-x86_64.tar.gz
```

### `GLIBCXX_3.4.30 not found`

AutoDL conda images may load an older `libstdc++.so.6` before the system version.
`setup_env.sh` and `scripts/run_bc.sh` set `LD_PRELOAD` to the system libstdc++
when available.

### `hopper-medium-v2` does not exist

D4RL MuJoCo registration requires `mjrl`. `setup_env.sh` installs D4RL separately
and then installs `mjrl` from GitHub. If GitHub is unreachable, rerun with a working
proxy or install `mjrl` manually before importing D4RL.

### macOS fails

Expected. Use macOS for editing and a Linux GPU server for running.

## Public Sharing Notes

This repository intentionally excludes:

- D4RL HDF5 datasets
- checkpoints
- wandb directories
- result JSONL files
- local credentials, tokens, or SSH configuration

Only source code, configuration, and lightweight documentation should be pushed.
