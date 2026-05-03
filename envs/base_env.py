import abc
import numpy as np


class EnvAdapter(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: int) -> dict:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def obs_dim(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_num(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_legal_action(self) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError


class SequenceWrapper:
    def __init__(self, seq_length=16):
        self.seq_length = seq_length
        self.feature_history = []

    def update(self, feature: np.ndarray):
        self.feature_history.append(feature.copy())
        if len(self.feature_history) > self.seq_length:
            self.feature_history = self.feature_history[-self.seq_length:]

    def get_sequence(self, feat_dim: int) -> np.ndarray:
        seq = np.zeros((self.seq_length, feat_dim), dtype=np.float32)
        start = max(0, self.seq_length - len(self.feature_history))
        for i, feat in enumerate(self.feature_history):
            seq[start + i] = feat
        return seq

    def get_next_sequence(self, next_feature: np.ndarray, feat_dim: int) -> np.ndarray:
        temp_history = self.feature_history + [next_feature.copy()]
        if len(temp_history) > self.seq_length:
            temp_history = temp_history[-self.seq_length:]
        seq = np.zeros((self.seq_length, feat_dim), dtype=np.float32)
        start = max(0, self.seq_length - len(temp_history))
        for i, feat in enumerate(temp_history):
            seq[start + i] = feat
        return seq

    def reset(self):
        self.feature_history = []


class Discretizer:
    def __init__(self, n_bins_per_dim: int, obs_low: np.ndarray, obs_high: np.ndarray):
        self.n_bins_per_dim = n_bins_per_dim
        self.bins = []
        for low, high in zip(obs_low, obs_high):
            self.bins.append(np.linspace(low, high, n_bins_per_dim + 1)[1:-1])

    def discretize(self, obs: np.ndarray) -> tuple:
        return tuple(
            int(np.digitize(o, bins))
            for o, bins in zip(obs, self.bins)
        )

    @property
    def n_states(self) -> int:
        return self.n_bins_per_dim ** len(self.bins)
