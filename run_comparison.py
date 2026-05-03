import os
import sys
import argparse
import logging
import yaml
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import load_config, create_algorithm, create_env, create_trainer, RobustStreamHandler


def main():
    parser = argparse.ArgumentParser(description="Run comparison experiment")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--algorithms", type=str, default="qlearning,dqn,hcra_dqn")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--result_dir", type=str, default=None)
    args = parser.parse_args()

    algo_list = []
    for a in args.algorithms.split(","):
        a = a.strip()
        if a == "tara_dqn":
            a = "hcra_dqn"
        algo_list.append(a)

    device = torch.device(args.device)

    config_name_map = {"frozen_lake": "frozen_lake", "FrozenLake-v1": "frozen_lake",
                       "mountain_car": "mountain_car", "MountainCar-v0": "mountain_car",
                       "cartpole": "frozen_lake", "CartPole-v1": "frozen_lake"}
    config_name = config_name_map.get(args.env, "frozen_lake")

    if args.config:
        config = load_config(args.config)
    else:
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        config_path = os.path.join(config_dir, f"{config_name}.yaml")
        config = load_config(config_path)

    if args.episodes:
        config.setdefault("training", {})["max_episodes"] = args.episodes

    base_dir = args.result_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", args.env, "comparison")
    os.makedirs(base_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[RobustStreamHandler(sys.stdout), logging.FileHandler(os.path.join(base_dir, "comparison.log"), encoding="utf-8")])

    all_results = {}

    for algo_name in algo_list:
        logging.info(f"\n{'='*60}\nTraining {algo_name} on {args.env}\n{'='*60}")
        seed_results = []

        for seed in range(args.seeds):
            np.random.seed(seed)
            torch.manual_seed(seed)

            env = create_env(args.env, config)
            if algo_name in ("hcra_dqn", "tara_dqn"):
                from envs.wrappers import SequenceMaintainedEnv
                cfg = config.get("hcra_dqn", config.get("tara_dqn", {}))
                env = SequenceMaintainedEnv(env, cfg.get("sequence_length", 16))

            algorithm = create_algorithm(algo_name, env.obs_dim, env.action_num, config, device)
            result_dir = os.path.join(base_dir, algo_name, f"seed{seed}")
            os.makedirs(result_dir, exist_ok=True)

            trainer = create_trainer(algo_name, algorithm, env, config, result_dir)
            results = trainer.train()
            seed_results.append(results)
            env.close()

        merged = {}
        for key in ["episode_rewards", "episode_lengths", "eval_rewards", "eval_episodes"]:
            all_data = [sr[key] for sr in seed_results if key in sr and sr[key]]
            if all_data:
                merged[key] = np.mean(all_data, axis=0).tolist()
        all_results[algo_name] = merged

    from visualization.plot_comparison import plot_comparison
    plot_comparison(all_results, env_name=args.env, save_path=os.path.join(base_dir, "comparison.png"))

    from evaluation.metrics import compare_algorithms
    comparison = compare_algorithms(all_results)
    with open(os.path.join(base_dir, "comparison_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    logging.info(f"\nComparison complete. Results saved to {base_dir}")
    logging.info(f"Metrics:\n{json.dumps(comparison, indent=2)}")


if __name__ == "__main__":
    main()
