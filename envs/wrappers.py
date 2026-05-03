import numpy as np
from envs.base_env import EnvAdapter, SequenceWrapper


class SequenceMaintainedEnv:
    def __init__(self, env: EnvAdapter, seq_length: int = 16):
        self.env = env
        self.seq_wrapper = SequenceWrapper(seq_length)

    def reset(self) -> dict:
        result = self.env.reset()
        self.seq_wrapper.reset()
        self.seq_wrapper.update(result["obs"])
        result["sequence"] = self.seq_wrapper.get_sequence(self.env.obs_dim)
        return result

    def step(self, action: int) -> dict:
        result = self.env.step(action)
        next_feature = result["obs"]
        result["next_sequence"] = self.seq_wrapper.get_next_sequence(next_feature, self.env.obs_dim)
        self.seq_wrapper.update(next_feature)
        result["sequence"] = self.seq_wrapper.get_sequence(self.env.obs_dim)
        return result

    @property
    def obs_dim(self) -> int:
        return self.env.obs_dim

    @property
    def action_num(self) -> int:
        return self.env.action_num

    def get_legal_action(self) -> list:
        return self.env.get_legal_action()

    def close(self):
        self.env.close()


class RewardShapingWrapper:
    def __init__(self, env: EnvAdapter, shaping_fn=None):
        self.env = env
        self.shaping_fn = shaping_fn

    def reset(self) -> dict:
        return self.env.reset()

    def step(self, action: int) -> dict:
        result = self.env.step(action)
        if self.shaping_fn is not None:
            result["reward"] = self.shaping_fn(result)
        return result

    @property
    def obs_dim(self) -> int:
        return self.env.obs_dim

    @property
    def action_num(self) -> int:
        return self.env.action_num

    def get_legal_action(self) -> list:
        return self.env.get_legal_action()

    def close(self):
        self.env.close()


def make_env(env_name: str, **kwargs) -> EnvAdapter:
    if env_name == "FrozenLake-v1" or env_name == "frozen_lake":
        from envs.frozen_lake import FrozenLakeEnv
        return FrozenLakeEnv(**kwargs)
    elif env_name == "CartPole-v1" or env_name == "cartpole":
        from envs.cartpole import CartPoleEnv
        return CartPoleEnv()
    elif env_name == "MountainCar-v0" or env_name == "mountain_car":
        from envs.mountain_car import MountainCarEnv
        return MountainCarEnv(n_bins=kwargs.pop("n_bins", 20),
                             height_scale=kwargs.pop("height_scale", 50.0))
    elif env_name in ("LunarLander-v2", "LunarLander-v3", "lunar_lander"):
        from envs.lunar_lander import LunarLanderEnv
        return LunarLanderEnv()
    elif env_name.startswith("MiniGrid") or env_name == "minigrid":
        from envs.minigrid_env import MiniGridEnv
        env_id = kwargs.pop("env_id", "MiniGrid-FourRooms-v0")
        fully_obs = kwargs.pop("fully_obs", False)
        return MiniGridEnv(env_id, max_steps=kwargs.get("max_steps", 200),
                           fully_obs=fully_obs, compact_obs=kwargs.pop("compact_obs", True))
    else:
        raise ValueError(f"Unknown environment: {env_name}")


class ObservationDropWrapper:
    """POMDP wrapper: randomly drops observation dimensions.

    At each step, each observation dimension is independently zeroed out with
    probability `p_drop`. This introduces partial observability, forcing the
    agent to integrate information across timesteps — precisely where temporal
    attention provides value.
    """

    def __init__(self, env, p_drop: float = 0.3):
        self.env = env
        self.p_drop = p_drop

    def reset(self) -> dict:
        result = self.env.reset()
        result["obs"] = self._drop(result["obs"])
        return result

    def step(self, action: int) -> dict:
        result = self.env.step(action)
        result["obs"] = self._drop(result["obs"])
        return result

    def _drop(self, obs: np.ndarray) -> np.ndarray:
        mask = (np.random.random(obs.shape) > self.p_drop).astype(np.float32)
        return obs * mask

    @property
    def obs_dim(self) -> int:
        return self.env.obs_dim

    @property
    def action_num(self) -> int:
        return self.env.action_num

    def get_legal_action(self) -> list:
        return self.env.get_legal_action()

    def close(self):
        self.env.close()


class DelayRewardWrapper:
    """Delays reward by N steps to test temporal credit assignment.

    The reward observed at step t is the ground-truth reward from step t-N.
    Early rewards are zero. This forces the agent to link current observations
    to rewards that arrive several steps later — a hard credit assignment
    problem that temporal attention should help solve.
    """

    def __init__(self, env, delay: int = 5):
        self.env = env
        self.delay = delay
        self._buffer = [0.0] * delay
        self._buf_idx = 0

    def reset(self) -> dict:
        self._buffer = [0.0] * self.delay
        self._buf_idx = 0
        return self.env.reset()

    def step(self, action: int) -> dict:
        result = self.env.step(action)
        true_reward = result["reward"]

        delayed_reward = self._buffer[self._buf_idx]
        self._buffer[self._buf_idx] = true_reward
        self._buf_idx = (self._buf_idx + 1) % self.delay

        result["reward"] = delayed_reward
        return result

    @property
    def obs_dim(self) -> int:
        return self.env.obs_dim

    @property
    def action_num(self) -> int:
        return self.env.action_num

    def get_legal_action(self) -> list:
        return self.env.get_legal_action()

    def close(self):
        self.env.close()
