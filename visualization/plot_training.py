import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(results: dict, title: str = "Training Curves",
                         save_path: str = None, window: int = 50):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    rewards = results.get("episode_rewards", [])
    if rewards:
        ax = axes[0, 0]
        ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) >= window:
            running_mean = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(rewards)), running_mean, color='blue',
                    label=f'Moving Avg ({window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

    lengths = results.get("episode_lengths", [])
    if lengths:
        ax = axes[0, 1]
        ax.plot(lengths, alpha=0.3, color='green')
        if len(lengths) >= window:
            running_mean = np.convolve(lengths, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(lengths)), running_mean, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)

    losses = results.get("losses", [])
    if losses:
        ax = axes[1, 0]
        ax.plot(losses, alpha=0.3, color='red')
        if len(losses) >= window:
            running_mean = np.convolve(losses, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(losses)), running_mean, color='red')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)

    eval_rewards = results.get("eval_rewards", [])
    eval_episodes = results.get("eval_episodes", [])
    if eval_rewards and eval_episodes:
        ax = axes[1, 1]
        ax.plot(eval_episodes, eval_rewards, 'o-', color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Eval Reward')
        ax.set_title('Evaluation Reward')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
