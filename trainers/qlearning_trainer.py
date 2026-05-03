import numpy as np
from trainers.base_trainer import BaseTrainer


class QLearningTrainer(BaseTrainer):
    def __init__(self, algorithm, env, config: dict, result_dir: str = "results"):
        super().__init__(algorithm, env, config, result_dir)

    def train_episode(self) -> tuple:
        result = self.env.reset()
        obs = result["obs"]
        state_id = result.get("state_id", result.get("obs_raw", obs))
        legal_action = result["legal_action"]

        total_reward = 0.0
        step = 0
        done = False

        while not done:
            action = self.algorithm.select_action(obs, legal_action, state_id=state_id)

            step_result = self.env.step(action)
            next_obs = step_result["obs"]
            next_state_id = step_result.get("state_id", step_result.get("obs_raw", next_obs))
            reward = step_result["reward"]
            done = step_result["done"]
            next_legal_action = step_result["legal_action"]

            self.algorithm.store_transition(
                obs=obs, action=action, reward=reward,
                next_obs=next_obs, done=done,
                legal_action=legal_action,
                state_id=state_id, next_state_id=next_state_id,
            )
            loss = self.algorithm.learn()

            obs = next_obs
            state_id = next_state_id
            legal_action = next_legal_action
            total_reward += reward
            step += 1

        return total_reward, step, loss

    def evaluate(self, num_episodes: int = None) -> dict:
        if num_episodes is None:
            num_episodes = self.eval_episodes

        rewards = []
        successes = []

        for _ in range(num_episodes):
            result = self.env.reset()
            obs = result["obs"]
            state_id = result.get("state_id", result.get("obs_raw", obs))
            legal_action = result["legal_action"]

            total_reward = 0.0
            done = False

            while not done:
                action = self.algorithm.select_greedy_action(obs, legal_action, state_id=state_id)
                step_result = self.env.step(action)
                obs = step_result["obs"]
                state_id = step_result.get("state_id", step_result.get("obs_raw", obs))
                legal_action = step_result["legal_action"]
                reward = step_result["reward"]
                done = step_result["done"]
                total_reward += reward

            rewards.append(total_reward)
            successes.append(1.0 if total_reward > 0 else 0.0)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "success_rate": np.mean(successes),
        }
