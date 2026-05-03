import numpy as np
import gymnasium as gym
import minigrid
from envs.base_env import EnvAdapter


class MiniGridEnv(EnvAdapter):
    """MiniGrid environment adapter.

    Two observation modes:
      - compact (default): uses raw grid image (7x7 object-type channel) -> 49 dims.
        Fast enough for CPU training.
      - flat: uses FlatObsWrapper ~2835 dims. Accurate but very slow on CPU.
    """

    def __init__(self, env_name: str = "MiniGrid-FourRooms-v0", max_steps: int = 200,
                 fully_obs: bool = False, compact_obs: bool = True):
        self.compact_obs = compact_obs
        self.env = gym.make(env_name)
        if max_steps is not None:
            self.env = gym.wrappers.TimeLimit(self.env.unwrapped, max_episode_steps=max_steps)
        if fully_obs:
            self.env = minigrid.wrappers.FullyObsWrapper(self.env)

        if not compact_obs:
            self.env = minigrid.wrappers.FlatObsWrapper(self.env)
            sample_obs, _ = self.env.reset()
            self._obs_dim = int(np.prod(sample_obs.shape)) if isinstance(sample_obs, np.ndarray) else len(sample_obs)
        else:
            sample_obs, _ = self.env.reset()
            self._obs_dim = int(np.prod(sample_obs["image"].shape))

        self._action_num = self.env.action_space.n
        self._legal_action = [1] * self._action_num

    def reset(self) -> dict:
        obs, info = self.env.reset()
        feature = self._extract_feature(obs)
        return {
            "obs": feature,
            "obs_raw": obs,
            "legal_action": self._legal_action,
            "info": info,
        }

    def step(self, action: int) -> dict:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        feature = self._extract_feature(obs)
        return {
            "obs": feature,
            "obs_raw": obs,
            "legal_action": self._legal_action,
            "reward": float(reward),
            "done": done,
            "info": info,
        }

    def _extract_feature(self, obs) -> np.ndarray:
        if self.compact_obs:
            img = obs["image"].astype(np.float32)
            img = img / max(img.max(), 1.0)
            return img.flatten()
        else:
            if isinstance(obs, np.ndarray):
                flat = obs.flatten().astype(np.float32)
                if flat.max() > 1.0:
                    flat = flat / max(flat.max(), 1.0)
                return flat
            return np.array(obs, dtype=np.float32).flatten()

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_num(self) -> int:
        return self._action_num

    def get_legal_action(self) -> list:
        return self._legal_action

    def close(self):
        self.env.close()
