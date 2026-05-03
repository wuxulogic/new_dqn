import numpy as np
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.qlearning.q_table import QTable


class QLearningAlgorithm(BaseAlgorithm):
    def __init__(self, action_num: int, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.9995):
        self.action_num = action_num
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = QTable(action_num)
        self.train_step = 0

    def select_action(self, obs, legal_action, **kwargs) -> int:
        state_id = kwargs.get("state_id", obs)
        self.train_step += 1
        if np.random.random() < self.epsilon:
            legal_indices = [i for i, a in enumerate(legal_action) if a > 0]
            if legal_indices:
                return int(np.random.choice(legal_indices))
            return int(np.random.randint(0, self.action_num))

        return self.q_table.get_best_action(state_id, legal_action)

    def select_greedy_action(self, obs, legal_action, **kwargs) -> int:
        state_id = kwargs.get("state_id", obs)
        return self.q_table.get_best_action(state_id, legal_action)

    def store_transition(self, **kwargs):
        self._last_transition = kwargs

    def learn(self):
        if not hasattr(self, "_last_transition") or self._last_transition is None:
            return None

        t = self._last_transition
        state_id = t.get("state_id", t["obs"])
        action = t["action"]
        reward = t["reward"]
        next_state_id = t.get("next_state_id", t["next_obs"])
        done = t["done"]
        legal_action = t.get("legal_action", [1] * self.action_num)

        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table.get_max_q(next_state_id)

        self.q_table.update(state_id, action, td_target, self.alpha)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._last_transition = None

        return td_target

    def save(self, path: str):
        self.q_table.save(path)

    def load(self, path: str):
        self.q_table.load(path)
