import numpy as np


def compute_metrics(rewards: list, window: int = 100) -> dict:
    rewards = np.array(rewards, dtype=np.float64)

    metrics = {
        "total_episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "median_reward": float(np.median(rewards)),
    }

    if len(rewards) >= window:
        metrics["mean_reward_last_{}".format(window)] = float(np.mean(rewards[-window:]))
        metrics["std_reward_last_{}".format(window)] = float(np.std(rewards[-window:]))

    metrics["convergence_episode"] = _find_convergence(rewards, window)

    threshold = np.mean(rewards) + np.std(rewards) * 0.5
    metrics["steps_to_threshold"] = _find_threshold(rewards, threshold)

    return metrics


def _find_convergence(rewards: np.ndarray, window: int = 100) -> int:
    if len(rewards) < window:
        return -1

    running_mean = np.convolve(rewards, np.ones(window) / window, mode='valid')
    if len(running_mean) < 2:
        return -1

    target = running_mean[-1]
    tolerance = 0.05 * abs(target) if abs(target) > 1e-6 else 1.0

    for i in range(len(running_mean)):
        if abs(running_mean[i] - target) < tolerance:
            return i + window

    return -1


def _find_threshold(rewards: np.ndarray, threshold: float) -> int:
    running_max = np.maximum.accumulate(rewards)
    for i, r in enumerate(running_max):
        if r >= threshold:
            return i + 1
    return -1


def compare_algorithms(algorithm_results: dict) -> dict:
    comparison = {}
    for algo_name, results in algorithm_results.items():
        rewards = results.get("episode_rewards", [])
        if rewards:
            comparison[algo_name] = compute_metrics(rewards)
    return comparison
