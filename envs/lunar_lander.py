import numpy as np
import gymnasium as gym
from envs.base_env import EnvAdapter


class LunarLanderEnv(EnvAdapter):
    def __init__(self):
        self.env = gym.make("LunarLander-v3")
        self._obs_dim = 8
        self._action_num = 4
        self._legal_action = [1, 1, 1, 1]
        self._obs_low = np.array([-1.5, -1.5, -5.0, -5.0, -3.14, -5.0, -1.0, -1.0], dtype=np.float32)
        self._obs_high = np.array([1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0], dtype=np.float32)

    def reset(self) -> dict:
        obs, info = self.env.reset()
        return {
            "obs": self._normalize(obs),
            "obs_raw": obs,
            "legal_action": self._legal_action,
            "info": info,
        }

    def step(self, action: int) -> dict:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return {
            "obs": self._normalize(obs),
            "obs_raw": obs,
            "legal_action": self._legal_action,
            "reward": float(reward),
            "done": done,
            "info": info,
        }

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self._obs_low) / (self._obs_high - self._obs_low + 1e-8)

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
