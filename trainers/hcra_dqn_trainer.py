import numpy as np
from trainers.base_trainer import BaseTrainer


class HCRADQNTrainer(BaseTrainer):
    def __init__(self, algorithm, env, config: dict, result_dir: str = "results",
                 seq_length: int = 16):
        super().__init__(algorithm, env, config, result_dir)
        self.seq_length = seq_length

    def train_episode(self) -> tuple:
        result = self.env.reset()
        obs = result["obs"]
        legal_action = result["legal_action"]
        sequence = result.get("sequence", np.zeros((self.seq_length, self.env.obs_dim), dtype=np.float32))

        total_reward = 0.0
        step = 0
        done = False
        last_loss = None

        while not done:
            action = self.algorithm.select_action(obs, legal_action, sequence=sequence)

            step_result = self.env.step(action)
            next_obs = step_result["obs"]
            reward = step_result["reward"]
            done = step_result["done"]
            next_legal_action = step_result["legal_action"]
            next_sequence = step_result.get(
                "next_sequence",
                np.zeros((self.seq_length, self.env.obs_dim), dtype=np.float32)
            )

            self.algorithm.store_transition(
                obs=obs, action=action, reward=reward,
                next_obs=next_obs, done=done,
                legal_action=next_legal_action,
                sequence=sequence, next_sequence=next_sequence,
            )
            loss = self.algorithm.learn()
            if loss is not None:
                last_loss = loss

            obs = next_obs
            legal_action = next_legal_action
            sequence = step_result.get(
                "sequence",
                np.zeros((self.seq_length, self.env.obs_dim), dtype=np.float32)
            )
            total_reward += reward
            step += 1

        return total_reward, step, last_loss

    def evaluate(self, num_episodes: int = None) -> dict:
        if num_episodes is None:
            num_episodes = self.eval_episodes

        rewards = []
        successes = []
        saved_epsilon = self.algorithm.epsilon

        for _ in range(num_episodes):
            self.algorithm.epsilon = self.eval_epsilon
            result = self.env.reset()
            obs = result["obs"]
            legal_action = result["legal_action"]
            sequence = result.get("sequence", np.zeros((self.seq_length, self.env.obs_dim), dtype=np.float32))

            total_reward = 0.0
            done = False

            while not done:
                action = self.algorithm.select_action(obs, legal_action, sequence=sequence)
                step_result = self.env.step(action)
                obs = step_result["obs"]
                legal_action = step_result["legal_action"]
                reward = step_result["reward"]
                done = step_result["done"]
                sequence = step_result.get(
                    "sequence",
                    np.zeros((self.seq_length, self.env.obs_dim), dtype=np.float32)
                )
                total_reward += reward

            rewards.append(total_reward)
            successes.append(1.0 if total_reward > 0 else 0.0)

        self.algorithm.epsilon = saved_epsilon
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "success_rate": np.mean(successes),
            "fusion_lambda": self.algorithm.get_fusion_lambda(),
        }
