import numpy as np
import gymnasium as gym
from envs.base_env import EnvAdapter, Discretizer


def mountain_car_height(position: float) -> float:
    """Height of the terrain at a given position (matches gymnasium's internal function)."""
    return float(np.sin(3.0 * position) * 0.45 + 0.55)


class MountainCarEnv(EnvAdapter):
    def __init__(self, n_bins: int = 20, height_scale: float = 200.0):
        self.env = gym.make("MountainCar-v0")
        self._obs_low = self.env.observation_space.low
        self._obs_high = self.env.observation_space.high
        self._obs_dim = 2
        self._action_num = 3
        self._legal_action = [1, 1, 1]
        self._n_bins = n_bins
        self._height_scale = height_scale
        self.discretizer = Discretizer(n_bins, self._obs_low, self._obs_high)
        self._last_height = 0.0

    def reset(self) -> dict:
        obs, info = self.env.reset()
        self._last_height = mountain_car_height(obs[0])
        return {
            "obs": self._normalize(obs),
            "obs_raw": obs,
            "state_id": self.discretizer.discretize(obs),
            "legal_action": self._legal_action,
            "info": info,
        }

    def step(self, action: int) -> dict:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Height-based potential reward shaping: encourages climbing either hill.
        # Going up left hill builds momentum, which the car then uses to reach
        # the right-side goal. Height potential correctly rewards both directions.
        new_height = mountain_car_height(obs[0])
        height_bonus = self._height_scale * (new_height - self._last_height)
        self._last_height = new_height
        shaped_reward = float(reward) + height_bonus

        return {
            "obs": self._normalize(obs),
            "obs_raw": obs,
            "state_id": self.discretizer.discretize(obs),
            "legal_action": self._legal_action,
            "reward": shaped_reward,
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
