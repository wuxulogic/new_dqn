import numpy as np
import gymnasium as gym
from envs.base_env import EnvAdapter


class CartPoleEnv(EnvAdapter):
    """CartPole-v1 adapter — fast environment for few-shot validation."""

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self._obs_dim = 4
        self._action_num = 2
        self._legal_action = [1, 1]

    def reset(self) -> dict:
        obs, info = self.env.reset()
        return {
            "obs": obs.astype(np.float32),
            "obs_raw": obs,
            "legal_action": self._legal_action,
            "info": info,
        }

    def step(self, action: int) -> dict:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return {
            "obs": obs.astype(np.float32),
            "obs_raw": obs,
            "legal_action": self._legal_action,
            "reward": float(reward),
            "done": done,
            "info": info,
        }

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
