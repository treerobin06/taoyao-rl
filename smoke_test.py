"""Smoke test — 验证环境完整可用（约 3 分钟）。

每位组员搭好环境后，必须先跑通此脚本，再开始任何训练。

四级检查：
  L1: torch + cuda
  L2: MuJoCo env step
  L3: D4RL dataset 加载 + shape
  L4: BC 训练 2k step + 1 episode eval

通过 = 你的 pipeline 完全 OK，可以开始正式训练。
失败 = 在群里贴报错，不要硬上手训练。

Usage:
    python smoke_test.py
"""
import os
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
mujoco_bin = os.path.expanduser("~/.mujoco/mujoco210/bin")
if os.path.isdir(mujoco_bin):
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if mujoco_bin not in cur_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{mujoco_bin}:{cur_ld}" if cur_ld else mujoco_bin

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

SMOKE_ENV = "hopper-medium-v2"
PASSED = []
FAILED = []


def check(label, fn):
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        msg = f"  ✓ {label}  ({elapsed:.1f}s)"
        if result:
            msg += f"  →  {result}"
        print(msg)
        PASSED.append(label)
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ✗ {label}  ({elapsed:.1f}s)")
        print(f"    {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        FAILED.append(label)
        return False


# ============== L1 ==============
print("\n[L1] Python · torch · CUDA")

def _torch_check():
    import torch
    cuda = torch.cuda.is_available()
    return f"torch {torch.__version__} | CUDA: {cuda}"
check("torch + cuda", _torch_check)

# ============== L2 ==============
print("\n[L2] MuJoCo env")

env = None
def _make_env():
    global env
    import gym
    import d4rl  # noqa
    env = gym.make(SMOKE_ENV)
    return f"{SMOKE_ENV} | obs={env.observation_space.shape} act={env.action_space.shape}"
check(f"gym.make({SMOKE_ENV})", _make_env)

if env is not None:
    def _step():
        obs = env.reset()
        for _ in range(10):
            obs, r, done, _ = env.step(env.action_space.sample())
            if done:
                obs = env.reset()
        return f"step × 10 ok"
    check("env.reset + step × 10", _step)

# ============== L3 ==============
print("\n[L3] D4RL dataset")

dataset_raw = None
def _load_dataset():
    global dataset_raw
    dataset_raw = env.get_dataset()
    n = dataset_raw["observations"].shape[0]
    return f"{n:,} transitions"
check("env.get_dataset()", _load_dataset)

if dataset_raw is not None:
    def _shape():
        assert dataset_raw["observations"].shape[1] == 11, "hopper obs_dim should be 11"
        assert dataset_raw["actions"].shape[1] == 3, "hopper act_dim should be 3"
        assert dataset_raw["observations"].shape[0] > 900_000, \
            f"hopper-medium-v2 should have >900k transitions, got {dataset_raw['observations'].shape[0]}"
        return "shape OK"
    check("hopper-medium-v2 shape & size", _shape)

# ============== L4 ==============
print("\n[L4] BC 训练 smoke (2000 steps, ~30s on GPU)")

def _bc_smoke():
    from common import D4RLDataset, make_env, get_obs_act_dims, set_seed, eval_episodes
    from algorithms.bc import BCAgent
    import torch

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    e = make_env(SMOKE_ENV, seed=0)
    obs_dim, act_dim = get_obs_act_dims(e)
    ds = D4RLDataset(e, raw_dataset=dataset_raw, device=device, normalize_obs=False)
    agent = BCAgent(obs_dim, act_dim, hidden=128, device=device)

    losses = []
    for _ in range(2000):
        info = agent.update(ds.sample(256))
        losses.append(info["bc_loss"])

    eval_env = make_env(SMOKE_ENV, seed=100)
    m = eval_episodes(agent, eval_env, n_episodes=3)
    loss_drop = (losses[0] - losses[-1]) / max(losses[0], 1e-6) * 100
    return (f"loss {losses[0]:.3f}→{losses[-1]:.3f} ({loss_drop:.0f}% ↓)  "
            f"| norm_score={m['normalized_score']:.1f}")

check("BC train 2k + eval 3 ep", _bc_smoke)

# ============== summary ==============
print("\n" + "=" * 60)
total = len(PASSED) + len(FAILED)
if not FAILED:
    print(f"  ALL PASSED ({len(PASSED)}/{total})")
    print(f"  环境完全 OK，可以开始正式训练")
    sys.exit(0)
else:
    print(f"  FAILED ({len(FAILED)}/{total})")
    for label in FAILED:
        print(f"  ✗ {label}")
    print(f"\n  在群里贴报错，不要硬上手训练")
    sys.exit(1)
