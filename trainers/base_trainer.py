import os
import time
import logging
import numpy as np
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, algorithm, env, config: dict, result_dir: str = "results"):
        self.algorithm = algorithm
        self.env = env
        self.config = config
        self.result_dir = result_dir

        self.max_episodes = config.get("max_episodes", 5000)
        self.eval_interval = config.get("eval_interval", 100)
        self.eval_episodes = config.get("eval_episodes", 20)
        self.eval_epsilon = config.get("eval_epsilon", 0.01)
        self.save_interval = config.get("save_interval", 500)
        self.log_interval = config.get("log_interval", 10)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_episodes_list = []
        self.losses = []

    @abstractmethod
    def train_episode(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, num_episodes: int = None) -> dict:
        raise NotImplementedError

    def train(self):
        self.logger.info(f"Starting training for {self.max_episodes} episodes")

        for episode in range(1, self.max_episodes + 1):
            episode_reward, episode_length, loss = self.train_episode()

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if loss is not None:
                self.losses.append(loss)

            if episode % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.log_interval:])
                self.logger.info(
                    f"Episode {episode}/{self.max_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | "
                    f"Epsilon: {getattr(self.algorithm, 'epsilon', 'N/A')}"
                )

            if episode % self.eval_interval == 0:
                eval_result = self.evaluate()
                self.eval_rewards.append(eval_result["mean_reward"])
                self.eval_episodes_list.append(episode)
                self.logger.info(
                    f"[EVAL] Episode {episode} | "
                    f"Mean Reward: {eval_result['mean_reward']:.2f} | "
                    f"Std: {eval_result['std_reward']:.2f} | "
                    f"Success Rate: {eval_result.get('success_rate', 'N/A')}"
                )

            if episode % self.save_interval == 0:
                self._save_model(episode)

        self._save_model(self.max_episodes)
        self._save_results()

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_rewards": self.eval_rewards,
            "eval_episodes": self.eval_episodes_list,
            "losses": self.losses,
        }

    def _save_model(self, episode: int):
        model_dir = os.path.join(self.result_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_ep{episode}.pth")
        try:
            self.algorithm.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save model: {e}")

    def _save_results(self):
        import json
        result_dir = os.path.join(self.result_dir, "logs")
        os.makedirs(result_dir, exist_ok=True)

        results = {
            "episode_rewards": [float(r) for r in self.episode_rewards],
            "episode_lengths": [int(l) for l in self.episode_lengths],
            "eval_rewards": [float(r) for r in self.eval_rewards],
            "eval_episodes": [int(e) for e in self.eval_episodes_list],
        }

        result_path = os.path.join(result_dir, "training_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {result_path}")
