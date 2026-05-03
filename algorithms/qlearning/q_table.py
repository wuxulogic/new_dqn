import pickle
from collections import defaultdict
import numpy as np


class QTable:
    def __init__(self, action_num: int):
        self.action_num = action_num
        self.q_table = defaultdict(lambda: np.zeros(action_num, dtype=np.float64))
        self.visit_count = defaultdict(lambda: np.zeros(action_num, dtype=np.int32))

    def get_q_values(self, state_id) -> np.ndarray:
        return self.q_table[state_id].copy()

    def get_max_q(self, state_id) -> float:
        return float(np.max(self.q_table[state_id]))

    def get_best_action(self, state_id, legal_action=None) -> int:
        q_values = self.q_table[state_id]
        if legal_action is not None:
            mask = np.array(legal_action, dtype=np.float64)
            q_values = q_values * mask
            if np.sum(mask) == 0:
                return int(np.argmax(self.q_table[state_id]))
        return int(np.argmax(q_values))

    def update(self, state_id, action: int, td_target: float, alpha: float = 0.1):
        current_q = self.q_table[state_id][action]
        self.q_table[state_id][action] = current_q + alpha * (td_target - current_q)
        self.visit_count[state_id][action] += 1

    def state_count(self, state_id) -> int:
        return int(np.sum(self.visit_count[state_id]))

    def total_states(self) -> int:
        return len(self.q_table)

    def save(self, path: str):
        data = {
            "q_table": dict(self.q_table),
            "visit_count": dict(self.visit_count),
            "action_num": self.action_num,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.action_num = data.get("action_num", self.action_num)
        loaded_q = data.get("q_table", {})
        loaded_visits = data.get("visit_count", {})
        self.q_table = defaultdict(lambda: np.zeros(self.action_num, dtype=np.float64))
        self.visit_count = defaultdict(lambda: np.zeros(self.action_num, dtype=np.int32))
        for k, v in loaded_q.items():
            self.q_table[k] = np.array(v, dtype=np.float64)
        for k, v in loaded_visits.items():
            self.visit_count[k] = np.array(v, dtype=np.int32)
