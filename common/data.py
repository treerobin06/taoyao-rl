"""Unified D4RL dataset wrapper — 全组共用接口。

所有算法实现都用这个 loader，确保数据切分、normalization、
next-state 计算等细节完全一致；保证 cross-algorithm 比较公平。
"""
import os
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")


def _prepend_mujoco_path():
    """Best-effort MuJoCo path for subprocess-free imports.

    setup_env.sh also writes this into .venv/bin/activate. Keeping it here makes
    ad-hoc Python entrypoints a little less fragile.
    """
    mujoco_bin = os.path.expanduser("~/.mujoco/mujoco210/bin")
    if os.path.isdir(mujoco_bin):
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if mujoco_bin not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{mujoco_bin}:{cur}" if cur else mujoco_bin


_prepend_mujoco_path()

import gym
import d4rl  # noqa: F401  # register d4rl envs


def make_env(env_name: str, seed: int = 0):
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def get_obs_act_dims(env):
    return env.observation_space.shape[0], env.action_space.shape[0]


class D4RLDataset:
    """In-memory D4RL replay buffer.

    Done flag follows D4RL convention: terminals[i]=1 表示 i 步后 episode 真的结束
    （不是 time-limit truncation）。

    重要：不要直接用 observations[1:] 自己拼 next_obs。D4RL raw dataset 有
    timeouts 边界，直接 shift 会把一个 episode 的最后一步接到下一个 episode 的
    reset observation。这里优先使用 d4rl.qlearning_dataset；如果传入的数据已经
    含 next_observations，则直接信任它。
    """

    def __init__(self, env, raw_dataset=None, device: str = "cpu", normalize_obs: bool = False):
        raw = raw_dataset if raw_dataset is not None else env.get_dataset()
        ds = self._as_qlearning_dataset(env, raw)

        self.observations = ds["observations"].astype(np.float32)
        self.actions      = ds["actions"].astype(np.float32)
        self.next_obs     = ds["next_observations"].astype(np.float32)
        self.rewards      = ds["rewards"].astype(np.float32)
        self.dones        = ds["terminals"].astype(np.float32)

        self.size = len(self.observations)
        self.device = device

        # 可选 observation normalization（TD3+BC / ReBRAC 原论文都用）
        if normalize_obs:
            self.obs_mean = self.observations.mean(0, keepdims=True)
            self.obs_std  = self.observations.std(0, keepdims=True) + 1e-3
            self.observations = (self.observations - self.obs_mean) / self.obs_std
            self.next_obs     = (self.next_obs - self.obs_mean) / self.obs_std
        else:
            self.obs_mean = np.zeros((1, self.observations.shape[1]), dtype=np.float32)
            self.obs_std  = np.ones((1, self.observations.shape[1]), dtype=np.float32)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean.squeeze(0)) / self.obs_std.squeeze(0)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.from_numpy(self.observations[idx]).to(self.device),
            "act":      torch.from_numpy(self.actions[idx]).to(self.device),
            "rew":      torch.from_numpy(self.rewards[idx]).to(self.device).unsqueeze(-1),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(self.device),
            "done":     torch.from_numpy(self.dones[idx]).to(self.device).unsqueeze(-1),
        }

    def __len__(self):
        return self.size

    @staticmethod
    def _as_qlearning_dataset(env, ds: dict) -> dict:
        """Return obs/action/next_obs/reward/done with timeout boundaries handled."""
        if "next_observations" in ds:
            return ds

        if env is not None and hasattr(d4rl, "qlearning_dataset"):
            return d4rl.qlearning_dataset(env, dataset=ds)

        # Fallback for tests/mocks without a real D4RL env. Mirrors D4RL's default
        # qlearning_dataset behavior: skip time-limit final transitions when the
        # raw dataset exposes a timeouts field.
        n = ds["rewards"].shape[0]
        use_timeouts = "timeouts" in ds
        obs, next_obs, actions, rewards, terminals = [], [], [], [], []
        for i in range(n - 1):
            if use_timeouts and bool(ds["timeouts"][i]):
                continue
            obs.append(ds["observations"][i])
            next_obs.append(ds["observations"][i + 1])
            actions.append(ds["actions"][i])
            rewards.append(ds["rewards"][i])
            terminals.append(ds["terminals"][i])

        return {
            "observations": np.asarray(obs),
            "actions": np.asarray(actions),
            "next_observations": np.asarray(next_obs),
            "rewards": np.asarray(rewards),
            "terminals": np.asarray(terminals),
        }
