import random
from collections import deque, namedtuple
import numpy as np

SeqTransition = namedtuple("SeqTransition", (
    "obs", "action", "reward", "next_obs", "done", "legal_action", "sequence", "next_sequence"
))


class SequenceReplayBuffer:
    """Replay buffer with uniform sampling, supporting observation sequences.

    Used by HCRA-DQN instead of prioritized replay. Same O(1) sampling as
    standard DQN but stores the 16-step sequence alongside each transition.
    """

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(SeqTransition(*args))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        batch = SeqTransition(*zip(*transitions))
        obs = np.array(batch.obs, dtype=np.float32)
        action = np.array(batch.action, dtype=np.int64).reshape(-1, 1)
        reward = np.array(batch.reward, dtype=np.float32).reshape(-1, 1)
        next_obs = np.array(batch.next_obs, dtype=np.float32)
        done = np.array(batch.done, dtype=np.float32).reshape(-1, 1)
        legal_action = np.array(batch.legal_action, dtype=np.float32)
        sequence = np.array(batch.sequence, dtype=np.float32)
        next_sequence = np.array(batch.next_sequence, dtype=np.float32)
        return obs, action, reward, next_obs, done, legal_action, sequence, next_sequence

    def __len__(self):
        return len(self.buffer)
