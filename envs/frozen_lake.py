import numpy as np
import gymnasium as gym
from envs.base_env import EnvAdapter


class FrozenLakeEnv(EnvAdapter):
    def __init__(self, map_size: int = 8, is_slippery: bool = True):
        if map_size == 4:
            env_name = "FrozenLake-v1"
            kwargs = {"map_name": "4x4", "is_slippery": is_slippery}
        else:
            env_name = "FrozenLake-v1"
            kwargs = {"map_name": "8x8", "is_slippery": is_slippery}

        self.env = gym.make(env_name, **kwargs)
        self._map_size = map_size
        self._n_states = map_size * map_size
        self._action_num = 4
        self._obs = None
        self._legal_action = [1, 1, 1, 1]

    def reset(self) -> dict:
        obs, info = self.env.reset()
        self._obs = obs
        return {
            "obs": self._extract_feature(obs),
            "obs_raw": obs,
            "legal_action": self._legal_action,
            "info": info,
        }

    def step(self, action: int) -> dict:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._obs = obs
        return {
            "obs": self._extract_feature(obs),
            "obs_raw": obs,
            "legal_action": self._legal_action,
            "reward": float(reward),
            "done": done,
            "info": info,
        }

    def _extract_feature(self, obs: int) -> np.ndarray:
        feature = np.zeros(self._n_states, dtype=np.float32)
        feature[obs] = 1.0
        return feature

    @property
    def obs_dim(self) -> int:
        return self._n_states

    @property
    def action_num(self) -> int:
        return self._action_num

    def get_legal_action(self) -> list:
        return self._legal_action

    def close(self):
        self.env.close()
