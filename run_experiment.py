import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class RobustStreamHandler(logging.StreamHandler):
    def flush(self):
        try:
            super().flush()
        except OSError:
            pass


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_algorithm(algo_name, obs_dim, action_num, config, device):
    if algo_name == "qlearning":
        from algorithms.qlearning.algorithm import QLearningAlgorithm
        cfg = config.get("qlearning", {})
        return QLearningAlgorithm(
            action_num=action_num, alpha=cfg.get("alpha", 0.1), gamma=cfg.get("gamma", 0.95),
            epsilon_start=cfg.get("epsilon_start", 1.0), epsilon_end=cfg.get("epsilon_end", 0.05),
            epsilon_decay=cfg.get("epsilon_decay", 0.9995))
    elif algo_name == "dqn":
        from algorithms.dqn.algorithm import DQNAlgorithm
        cfg = config.get("dqn", {})
        return DQNAlgorithm(
            obs_dim=obs_dim, action_num=action_num, gamma=cfg.get("gamma", 0.99),
            learning_rate=cfg.get("learning_rate", 1e-4), epsilon_start=cfg.get("epsilon_start", 1.0),
            epsilon_end=cfg.get("epsilon_end", 0.05), epsilon_decay_steps=cfg.get("epsilon_decay_steps", 100000),
            batch_size=cfg.get("batch_size", 32), buffer_size=cfg.get("buffer_size", 50000),
            learn_start=cfg.get("learn_start", 5000), target_update_freq=cfg.get("target_update_freq", 500),
            tau=cfg.get("tau", 0.005), grad_clip=cfg.get("grad_clip", 10.0),
            hidden_dim=cfg.get("hidden_dim", 128), mid_dim=cfg.get("mid_dim", 64), device=device)
    elif algo_name in ("hcra_dqn", "tara_dqn"):
        from algorithms.hcra_dqn.algorithm import HCRADQNAlgorithm
        cfg = config.get("hcra_dqn", config.get("tara_dqn", {}))
        return HCRADQNAlgorithm(
            obs_dim=obs_dim, action_num=action_num, gamma=cfg.get("gamma", 0.99),
            learning_rate=cfg.get("learning_rate", 1e-4), epsilon_start=cfg.get("epsilon_start", 1.0),
            epsilon_end=cfg.get("epsilon_end", 0.05), epsilon_decay_steps=cfg.get("epsilon_decay_steps", 100000),
            batch_size=cfg.get("batch_size", 32), buffer_size=cfg.get("buffer_size", 50000),
            learn_start=cfg.get("learn_start", 5000), target_update_freq=cfg.get("target_update_freq", 500),
            tau=cfg.get("tau", 0.005), grad_clip=cfg.get("grad_clip", 10.0),
            hidden_dim=cfg.get("hidden_dim", 128), mid_dim=cfg.get("mid_dim", 64),
            sequence_length=cfg.get("sequence_length", 16),
            attention_heads=cfg.get("attention_heads", 4),
            attention_dim=cfg.get("attention_dim", 64),
            num_segments=cfg.get("num_segments", 4),
            num_refinements=cfg.get("num_refinements", 2),
            fusion_lambda=cfg.get("fusion_lambda", 0.0),
            contract_reg_coeff=cfg.get("contract_reg_coeff", 0.01),
            device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def create_env(env_name, config):
    env_cfg = config.get("env", {})
    from envs.wrappers import make_env
    if env_name in ("frozen_lake", "FrozenLake-v1"):
        return make_env("frozen_lake", map_size=env_cfg.get("map_size", 8), is_slippery=env_cfg.get("is_slippery", True))
    elif env_name in ("mountain_car", "MountainCar-v0"):
        return make_env("mountain_car", n_bins=config.get("qlearning", {}).get("n_bins", 20),
                        height_scale=env_cfg.get("height_scale", 50.0))
    elif env_name in ("cartpole", "CartPole-v1"):
        return make_env("cartpole")
    elif env_name.startswith("MiniGrid") or env_name == "minigrid":
        return make_env("minigrid", env_id=env_cfg.get("name", "MiniGrid-FourRooms-v0"), max_steps=env_cfg.get("max_steps", 200))
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def create_trainer(algo_name, algorithm, env, config, result_dir):
    training_cfg = config.get("training", {})
    if algo_name == "qlearning":
        from trainers.qlearning_trainer import QLearningTrainer
        return QLearningTrainer(algorithm, env, training_cfg, result_dir)
    elif algo_name == "dqn":
        from trainers.dqn_trainer import DQNTrainer
        return DQNTrainer(algorithm, env, training_cfg, result_dir)
    elif algo_name in ("hcra_dqn", "tara_dqn"):
        from trainers.hcra_dqn_trainer import HCRADQNTrainer
        cfg = config.get("hcra_dqn", config.get("tara_dqn", {}))
        seq_length = cfg.get("sequence_length", 16)
        return HCRADQNTrainer(algorithm, env, training_cfg, result_dir, seq_length=seq_length)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def main():
    parser = argparse.ArgumentParser(description="Run RL experiment")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["qlearning", "dqn", "hcra_dqn", "tara_dqn"])
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--result_dir", type=str, default=None)
    args = parser.parse_args()

    # Normalize legacy algo name
    if args.algorithm == "tara_dqn":
        args.algorithm = "hcra_dqn"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    env = create_env(args.env, config)
    if args.algorithm in ("hcra_dqn", "tara_dqn"):
        from envs.wrappers import SequenceMaintainedEnv
        cfg = config.get("hcra_dqn", config.get("tara_dqn", {}))
        env = SequenceMaintainedEnv(env, cfg.get("sequence_length", 16))

    algorithm = create_algorithm(args.algorithm, env.obs_dim, env.action_num, config, device)
    result_dir = args.result_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", args.env, args.algorithm, f"seed{args.seed}")
    os.makedirs(result_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[RobustStreamHandler(sys.stdout), logging.FileHandler(os.path.join(result_dir, "training.log"), encoding="utf-8")])

    trainer = create_trainer(args.algorithm, algorithm, env, config, result_dir)
    logging.info(f"Algorithm: {args.algorithm} | Env: {args.env} | Device: {device}")
    logging.info(f"Obs dim: {env.obs_dim} | Action num: {env.action_num}")

    results = trainer.train()

    from visualization.plot_training import plot_training_curves
    plot_training_curves(results, title=f"{args.algorithm} on {args.env}", save_path=os.path.join(result_dir, "training_curves.png"))
    logging.info(f"Training complete. Results saved to {result_dir}")
    env.close()


if __name__ == "__main__":
    main()
