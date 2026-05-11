"""Pre-download D4RL datasets to local cache (~/.d4rl/datasets/).

避免每个组员第一次跑训练时各自下载浪费时间。
下载完后所有人共用本地缓存。

Usage:
    python download_d4rl.py             # 下载 envs.txt 里所有
    python download_d4rl.py hopper-medium-v2  # 指定环境

预计耗时 5-10 min（取决于网速），磁盘 ~1 GB。
"""

import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
mujoco_bin = os.path.expanduser("~/.mujoco/mujoco210/bin")
if os.path.isdir(mujoco_bin):
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if mujoco_bin not in cur_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{mujoco_bin}:{cur_ld}" if cur_ld else mujoco_bin

import gym
import d4rl  # noqa: F401  # required to register d4rl envs

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENVS_FILE = os.path.join(PROJECT_ROOT, "envs.txt")


def load_env_list():
    with open(ENVS_FILE) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def download_one(env_name: str):
    t0 = time.time()
    print(f"\n→ {env_name}")
    env = gym.make(env_name)
    dataset = env.get_dataset()
    n = dataset["observations"].shape[0]
    obs_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]
    elapsed = time.time() - t0
    print(f"  ✓ {n:>10,} transitions  |  obs={obs_dim}  act={act_dim}  |  {elapsed:.1f}s")
    return n


def main():
    envs = sys.argv[1:] if len(sys.argv) > 1 else load_env_list()
    print(f"Downloading {len(envs)} D4RL datasets → ~/.d4rl/datasets/")

    total = 0
    for env_name in envs:
        try:
            total += download_one(env_name)
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")

    print(f"\nDone. Total {total:,} transitions cached.")


if __name__ == "__main__":
    main()
