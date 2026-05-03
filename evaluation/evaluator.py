import numpy as np
from evaluation.metrics import compute_metrics


class Evaluator:
    def __init__(self, algorithm, env, seq_length: int = 16):
        self.algorithm = algorithm
        self.env = env
        self.seq_length = seq_length

    def evaluate(self, num_episodes: int = 20) -> dict:
        rewards = []
        lengths = []
        successes = []

        for _ in range(num_episodes):
            result = self.env.reset()
            obs = result["obs"]
            legal_action = result["legal_action"]

            has_sequence = "sequence" in result
            sequence = result.get("sequence", np.zeros((self.seq_length, self.env.obs_dim), dtype=np.float32))

            total_reward = 0.0
            step = 0
            done = False

            while not done:
                if has_sequence:
                    action = self.algorithm.select_greedy_action(obs, legal_action, sequence=sequence)
                else:
                    action = self.algorithm.select_greedy_action(obs, legal_action)

                step_result = self.env.step(action)
                obs = step_result["obs"]
                legal_action = step_result["legal_action"]
                reward = step_result["reward"]
                done = step_result["done"]

                if has_sequence:
                    sequence = step_result.get(
                        "sequence",
                        np.zeros((self.seq_length, self.env.obs_dim), dtype=np.float32)
                    )

                total_reward += reward
                step += 1

            rewards.append(total_reward)
            lengths.append(step)
            successes.append(1.0 if total_reward > 0 else 0.0)

        result = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
            "num_episodes": num_episodes,
        }

        if hasattr(self.algorithm, "get_fusion_lambda"):
            result["fusion_lambda"] = self.algorithm.get_fusion_lambda()

        return result
