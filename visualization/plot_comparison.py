import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(all_results: dict, env_name: str = "",
                    save_path: str = None, window: int = 50):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Algorithm Comparison - {env_name}", fontsize=16)

    colors = {'qlearning': 'blue', 'dqn': 'green', 'hcra_dqn': 'red', 'tara_dqn': 'red'}
    labels = {'qlearning': 'Q-Learning', 'dqn': 'DQN', 'hcra_dqn': 'HCRA-DQN', 'tara_dqn': 'HCRA-DQN'}

    ax = axes[0]
    for algo_name, results in all_results.items():
        rewards = results.get("episode_rewards", [])
        if rewards:
            color = colors.get(algo_name, 'gray')
            label = labels.get(algo_name, algo_name)
            if len(rewards) >= window:
                running_mean = np.convolve(rewards, np.ones(window) / window, mode='valid')
                ax.plot(range(window - 1, len(rewards)), running_mean,
                        color=color, label=label, linewidth=2)
            else:
                ax.plot(rewards, color=color, label=label, alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Reward (Moving Avg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for algo_name, results in all_results.items():
        eval_rewards = results.get("eval_rewards", [])
        eval_episodes = results.get("eval_episodes", [])
        if eval_rewards and eval_episodes:
            color = colors.get(algo_name, 'gray')
            label = labels.get(algo_name, algo_name)
            ax.plot(eval_episodes, eval_rewards, 'o-', color=color, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Evaluation Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    bar_data = {}
    for algo_name, results in all_results.items():
        rewards = results.get("episode_rewards", [])
        if rewards:
            last_n = min(100, len(rewards))
            bar_data[labels.get(algo_name, algo_name)] = np.mean(rewards[-last_n:])

    if bar_data:
        bars = ax.bar(bar_data.keys(), bar_data.values(),
                       color=[colors.get(k, 'gray') for k in all_results.keys() if k in bar_data])
        ax.set_ylabel('Mean Reward (Last 100)')
        ax.set_title('Final Performance')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
