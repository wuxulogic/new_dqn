"""HCRA-DQN Few-Shot Offline Validation.

Two-round data collection with iterative training:
  Round 1: random policy collects diverse data → train briefly
  Round 2: epsilon-greedy (eps=0.3) with partially-trained model → collect better data
  Then train to convergence on the combined dataset.

This isolates algorithm quality: same data for all variants, but the data
includes non-random trajectories that give the algorithms something to learn.

Usage:
    python run_fewshot.py                          # default ablation
    python run_fewshot.py --config hcra_ablation   # explicit config
    python run_fewshot.py --env CartPole-v1 --p-drop 0.3 --seeds 20
"""

import argparse
import json
import os
import random
import sys
import time
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.wrappers import make_env, SequenceMaintainedEnv, ObservationDropWrapper, DelayRewardWrapper
from algorithms.dqn.algorithm import DQNAlgorithm
from algorithms.hcra_dqn.algorithm import HCRADQNAlgorithm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(env_name: str, p_drop: float = 0.0, reward_delay: int = 0, seq_length: int = 16):
    base = make_env(env_name)
    if p_drop > 0:
        base = ObservationDropWrapper(base, p_drop)
    if reward_delay > 0:
        base = DelayRewardWrapper(base, reward_delay)
    return SequenceMaintainedEnv(base, seq_length)


def build_algorithm(variant_name: str, variant_cfg: dict, shared_cfg: dict,
                    obs_dim: int, action_num: int, device, seq_length: int = 16):
    use_attention = variant_cfg.get("use_attention", False)

    if not use_attention:
        return DQNAlgorithm(
            obs_dim=obs_dim, action_num=action_num,
            gamma=shared_cfg["gamma"],
            learning_rate=shared_cfg["learning_rate"],
            epsilon_start=shared_cfg["epsilon_start"],
            epsilon_end=shared_cfg["epsilon_end"],
            epsilon_decay_steps=shared_cfg["epsilon_decay_steps"],
            batch_size=shared_cfg["batch_size"],
            buffer_size=shared_cfg["buffer_size"],
            learn_start=shared_cfg["learn_start"],
            target_update_freq=shared_cfg["target_update_freq"],
            tau=shared_cfg["tau"],
            grad_clip=shared_cfg["grad_clip"],
            hidden_dim=shared_cfg["hidden_dim"],
            mid_dim=shared_cfg["mid_dim"],
            device=device,
        )
    else:
        return HCRADQNAlgorithm(
            obs_dim=obs_dim, action_num=action_num,
            gamma=shared_cfg["gamma"],
            learning_rate=shared_cfg["learning_rate"],
            epsilon_start=shared_cfg["epsilon_start"],
            epsilon_end=shared_cfg["epsilon_end"],
            epsilon_decay_steps=shared_cfg["epsilon_decay_steps"],
            batch_size=shared_cfg["batch_size"],
            buffer_size=shared_cfg["buffer_size"],
            learn_start=shared_cfg["learn_start"],
            target_update_freq=shared_cfg["target_update_freq"],
            tau=shared_cfg["tau"],
            grad_clip=shared_cfg["grad_clip"],
            hidden_dim=shared_cfg["hidden_dim"],
            mid_dim=shared_cfg["mid_dim"],
            sequence_length=seq_length,
            attention_heads=4,
            attention_dim=64,
            num_segments=variant_cfg.get("num_segments", 4),
            num_refinements=variant_cfg.get("num_refinements", 1),
            fusion_lambda=variant_cfg.get("fusion_lambda", 0.0),
            contract_reg_coeff=variant_cfg.get("contract_reg_coeff", 0.0),
            device=device,
        )


def collect_dataset(env, num_episodes: int, seq_length: int, obs_dim: int,
                    algorithm=None, epsilon: float = 1.0) -> list:
    """Collect transitions. If algorithm is None, uses random policy.
    Otherwise uses epsilon-greedy with the given algorithm.
    """
    transitions = []
    zero_seq = np.zeros((seq_length, obs_dim), dtype=np.float32)

    for _ in range(num_episodes):
        result = env.reset()
        obs = result["obs"]
        seq = result.get("sequence", zero_seq.copy())
        done = False

        while not done:
            if algorithm is None or np.random.random() < epsilon:
                action = random.randint(0, env.action_num - 1)
            else:
                action = algorithm.select_action(obs, env.get_legal_action(), sequence=seq)

            step_result = env.step(action)
            next_obs = step_result["obs"]
            reward = step_result["reward"]
            done = step_result["done"]
            next_seq = step_result.get("next_sequence", zero_seq.copy())

            transitions.append({
                "obs": obs.copy(),
                "action": action,
                "reward": reward,
                "next_obs": next_obs.copy(),
                "done": done,
                "legal_action": env.get_legal_action(),
                "sequence": seq.copy(),
                "next_sequence": next_seq.copy(),
            })
            obs = next_obs
            seq = step_result.get("sequence", zero_seq.copy())

    return transitions


def load_dataset(algorithm, dataset: list):
    for t in dataset:
        algorithm.store_transition(**t)


def train_offline(algorithm, num_batches: int):
    losses = []
    lambdas = []
    for _ in range(num_batches):
        loss = algorithm.learn()
        if loss is not None:
            losses.append(loss)
        if hasattr(algorithm, "get_fusion_lambda"):
            lambdas.append(algorithm.get_fusion_lambda())
    return losses, lambdas


def evaluate_online(algorithm, env, num_episodes: int, seq_length: int, obs_dim: int):
    zero_seq = np.zeros((seq_length, obs_dim), dtype=np.float32)
    saved_epsilon = algorithm.epsilon
    algorithm.epsilon = 0.01

    rewards = []
    for _ in range(num_episodes):
        result = env.reset()
        obs = result["obs"]
        seq = result.get("sequence", zero_seq.copy())
        done = False
        total = 0.0

        while not done:
            action = algorithm.select_action(obs, env.get_legal_action(), sequence=seq)
            step_result = env.step(action)
            obs = step_result["obs"]
            reward = step_result["reward"]
            done = step_result["done"]
            seq = step_result.get("sequence", zero_seq.copy())
            total += reward

        rewards.append(total)

    algorithm.epsilon = saved_epsilon
    return np.mean(rewards), np.std(rewards)


def run_experiment(variant_name: str, variant_cfg: dict, shared_cfg: dict,
                   env_name: str, env_cfg: dict, device, seed: int,
                   offline_cfg: dict, seq_length: int = 16):
    """Two-round data collection + training for one variant/seed."""
    set_seed(seed)

    p_drop = env_cfg.get("p_drop", 0.0)
    reward_delay = env_cfg.get("reward_delay", 0)
    obs_dim = env_cfg.get("obs_dim", 4)
    action_num = env_cfg.get("action_num", 2)

    half_eps = offline_cfg.get("collect_episodes", 500) // 2
    round1_batches = offline_cfg.get("round1_batches", 200)
    round2_batches = offline_cfg.get("train_batches", 500) - round1_batches

    # ---- Round 1: Random policy data collection ----
    env_r1 = build_env(env_name, p_drop, reward_delay, seq_length)
    dataset_r1 = collect_dataset(env_r1, half_eps, seq_length, obs_dim, algorithm=None)
    env_r1.close()

    # Build algorithm and train briefly on random data
    algorithm = build_algorithm(variant_name, variant_cfg, shared_cfg,
                                obs_dim, action_num, device, seq_length)
    load_dataset(algorithm, dataset_r1)
    train_offline(algorithm, round1_batches)

    # ---- Round 2: Epsilon-greedy collection with partially trained model ----
    env_r2 = build_env(env_name, p_drop, reward_delay, seq_length)
    dataset_r2 = collect_dataset(env_r2, half_eps, seq_length, obs_dim,
                                 algorithm=algorithm, epsilon=0.3)
    env_r2.close()

    # Combine datasets and train to convergence
    full_dataset = dataset_r1 + dataset_r2
    algorithm = build_algorithm(variant_name, variant_cfg, shared_cfg,
                                obs_dim, action_num, device, seq_length)
    load_dataset(algorithm, full_dataset)

    t0 = time.time()
    losses, lambdas = train_offline(algorithm, round2_batches)
    train_time = time.time() - t0

    # Evaluate
    eval_env = build_env(env_name, p_drop, reward_delay, seq_length)
    mean_r, std_r = evaluate_online(algorithm, eval_env, offline_cfg["eval_episodes"],
                                     seq_length, obs_dim)
    eval_env.close()

    result = {
        "variant": variant_name,
        "env": env_name,
        "p_drop": p_drop,
        "seed": seed,
        "eval_mean_reward": float(mean_r),
        "eval_std_reward": float(std_r),
        "final_loss": float(losses[-1]) if losses else None,
        "train_time_s": float(train_time),
        "num_transitions": len(full_dataset),
    }
    if lambdas:
        result["final_fusion_lambda"] = float(lambdas[-1]) if lambdas else None
        result["lambda_trajectory"] = [float(l) for l in lambdas]

    return result


def main():
    parser = argparse.ArgumentParser(description="HCRA-DQN Few-Shot Ablation")
    parser.add_argument("--config", default="hcra_ablation",
                        help="Config file name (without .yaml)")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--p-drop", type=float, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/fewshot")
    args = parser.parse_args()

    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    config_path = os.path.join(config_dir, f"{args.config}.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    shared_cfg = cfg["defaults"]["shared"]
    offline_cfg = cfg["offline"]
    envs_cfg = cfg["environments"]
    variants_cfg = cfg["variants"]
    num_seeds = args.seeds or cfg.get("num_seeds", 20)
    seq_length = 16

    device = torch.device(args.device)

    if args.env:
        envs_cfg = [e for e in envs_cfg if e["name"] == args.env]
    if args.p_drop is not None:
        for e in envs_cfg:
            e["p_drop"] = args.p_drop

    os.makedirs(args.output, exist_ok=True)

    all_results = []

    for env_cfg in envs_cfg:
        env_name = env_cfg["name"]
        print(f"\n{'='*60}")
        print(f"Environment: {env_name} (p_drop={env_cfg.get('p_drop', 0)})")
        print(f"{'='*60}")

        for variant_name, variant_cfg in variants_cfg.items():
            print(f"\n  Variant: {variant_name} — {variant_cfg['description']}")

            for seed in range(num_seeds):
                try:
                    result = run_experiment(
                        variant_name, variant_cfg, shared_cfg,
                        env_name, env_cfg, device, seed,
                        offline_cfg, seq_length,
                    )
                    all_results.append(result)
                    lambda_str = ""
                    if result.get("final_fusion_lambda") is not None:
                        lambda_str = f"  lambda={result['final_fusion_lambda']:.4f}"
                    print(f"    seed={seed:2d}  eval_reward={result['eval_mean_reward']:7.2f} +/- {result['eval_std_reward']:6.2f}  "
                          f"time={result['train_time_s']:.1f}s{lambda_str}")
                except Exception as e:
                    print(f"    seed={seed:2d}  FAILED: {e}")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Group by env + p_drop
    seen = set()
    for env_cfg in envs_cfg:
        env_name = env_cfg["name"]
        p_drop = env_cfg.get("p_drop", 0)
        key = (env_name, p_drop)
        if key in seen:
            continue
        seen.add(key)

        print(f"\n  {env_name} (p_drop={p_drop}):")
        print(f"  {'Variant':<15s} {'Mean':>8s} {'Std':>8s} {'95% CI':>20s} {'lambda':>8s} {'Seeds':>6s}")
        print(f"  {'-'*68}")

        for variant_name in variants_cfg:
            vr = [r for r in all_results
                  if r["variant"] == variant_name and r["env"] == env_name
                  and r["p_drop"] == p_drop]
            if vr:
                rewards = [r["eval_mean_reward"] for r in vr]
                mean_r = np.mean(rewards)
                std_r = np.std(rewards)
                n = len(rewards)
                # Bootstrap 95% CI
                bs_means = [np.mean(np.random.choice(rewards, n, replace=True)) for _ in range(10000)]
                bs_means.sort()
                ci_lo, ci_hi = bs_means[250], bs_means[9750]
                lambdas = [r.get("final_fusion_lambda", float("nan")) for r in vr]
                lambda_str = f"{np.nanmean(lambdas):.4f}" if lambdas and not np.all(np.isnan(lambdas)) else "N/A"
                print(f"  {variant_name:<15s} {mean_r:>8.2f} {std_r:>8.2f} [{ci_lo:>8.2f}, {ci_hi:>8.2f}] {lambda_str:>8s} {n:>6d}")

    output_path = os.path.join(args.output, "fewshot_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
