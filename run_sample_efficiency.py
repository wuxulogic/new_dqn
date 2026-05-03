"""Online Sample Efficiency Comparison.

Compares DQN vs dqn_hier (hierarchical causal attention) on POMDP environments.
Focuses on MiniGrid where partial observability is inherent.

Usage:
    python run_sample_efficiency.py --env MiniGrid-FourRooms-v0 --episodes 500 --seeds 5
"""

import argparse
import json
import os
import random
import sys
import time
import numpy as np
import torch
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.wrappers import make_env, SequenceMaintainedEnv, ObservationDropWrapper
from algorithms.dqn.algorithm import DQNAlgorithm
from algorithms.hcra_dqn.algorithm import HCRADQNAlgorithm
from trainers.dqn_trainer import DQNTrainer
from trainers.hcra_dqn_trainer import HCRADQNTrainer


SHARED_CFG = dict(
    gamma=0.99, learning_rate=0.0003,
    epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=20000,
    batch_size=64, buffer_size=50000, learn_start=2000,
    target_update_freq=500, tau=0.005, grad_clip=10.0,
    hidden_dim=128, mid_dim=64,
)

ENVIRONMENTS = [
    dict(name="MiniGrid-FourRooms-v0", obs_dim=None, action_num=7),
    dict(name="MiniGrid-DoorKey-5x5-v0", obs_dim=None, action_num=7),
    dict(name="MiniGrid-KeyCorridorS3R1-v0", obs_dim=None, action_num=7),
]

VARIANTS = {
    "dqn": {
        "description": "Standard DQN (baseline)",
        "use_attention": False,
    },
    "dqn_hier": {
        "description": "DQN + hierarchical causal attention",
        "use_attention": True,
        "num_segments": 4, "num_refinements": 1,
        "fusion_lambda": 0.0,
    },
}

SEQ_LENGTH = 16


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(env_name: str, p_drop: float = 0.0, seq_length: int = 16):
    base = make_env(env_name)
    if p_drop > 0:
        base = ObservationDropWrapper(base, p_drop)
    return SequenceMaintainedEnv(base, seq_length)


def build_algorithm(variant_name: str, variant_cfg: dict,
                    obs_dim: int, action_num: int, device):
    use_attention = variant_cfg.get("use_attention", False)
    if not use_attention:
        return DQNAlgorithm(
            obs_dim=obs_dim, action_num=action_num, device=device,
            **{k: SHARED_CFG[k] for k in SHARED_CFG},
        )
    else:
        return HCRADQNAlgorithm(
            obs_dim=obs_dim, action_num=action_num, device=device,
            sequence_length=SEQ_LENGTH,
            attention_heads=4, attention_dim=64,
            num_segments=variant_cfg.get("num_segments", 4),
            num_refinements=variant_cfg.get("num_refinements", 1),
            fusion_lambda=variant_cfg.get("fusion_lambda", 0.0),
            contract_reg_coeff=variant_cfg.get("contract_reg_coeff", 0.0),
            **{k: SHARED_CFG[k] for k in SHARED_CFG},
        )


def run_experiment(variant_name: str, variant_cfg: dict, env_cfg: dict,
                   device, seed: int, max_episodes: int):
    set_seed(seed)

    env_name = env_cfg["name"]
    p_drop = env_cfg.get("p_drop", 0.0)

    env = build_env(env_name, p_drop, SEQ_LENGTH)
    obs_dim = env.obs_dim
    action_num = env.action_num

    algorithm = build_algorithm(variant_name, variant_cfg, obs_dim, action_num, device)

    trainer_cfg = dict(
        max_episodes=max_episodes,
        eval_interval=max(10, max_episodes // 10),
        eval_episodes=20,
        eval_epsilon=0.01,
        save_interval=max_episodes + 1,
        log_interval=max(1, max_episodes // 20),
    )

    tmpdir = tempfile.mkdtemp(prefix=f"hcra_{variant_name}_s{seed}_")
    if variant_cfg.get("use_attention", False):
        trainer = HCRADQNTrainer(algorithm, env, trainer_cfg,
                                 result_dir=tmpdir, seq_length=SEQ_LENGTH)
    else:
        trainer = DQNTrainer(algorithm, env, trainer_cfg, result_dir=tmpdir)

    t0 = time.time()
    results = trainer.train()
    elapsed = time.time() - t0

    env.close()

    rewards = results["episode_rewards"]
    eval_rewards = results["eval_rewards"]

    auc = float(np.trapezoid(rewards)) / len(rewards) if len(rewards) > 1 else float(rewards[0]) if rewards else 0.0
    final_n = max(1, len(rewards) // 5)
    final_perf = float(np.mean(rewards[-final_n:])) if rewards else 0.0
    peak_perf = float(np.max(rewards)) if rewards else 0.0

    fusion_lambda = algorithm.get_fusion_lambda() if hasattr(algorithm, "get_fusion_lambda") else None

    return {
        "variant": variant_name, "env": env_name, "p_drop": p_drop, "seed": seed,
        "auc": auc, "final_perf": final_perf, "peak_perf": peak_perf,
        "eval_rewards": [float(r) for r in eval_rewards],
        "episode_rewards": [float(r) for r in rewards],
        "fusion_lambda": float(fusion_lambda) if fusion_lambda else None,
        "time_s": float(elapsed),
    }


def bootstrap_ci(samples, n_bootstrap=10000):
    n = len(samples)
    means = sorted([np.mean(np.random.choice(samples, n, replace=True)) for _ in range(n_bootstrap)])
    return np.mean(samples), means[250], means[9750], np.std(samples)


def main():
    parser = argparse.ArgumentParser(description="Online Comparison: DQN vs dqn_hier")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = os.path.join("results", "minigrid_compare")
    os.makedirs(output_dir, exist_ok=True)

    envs_to_run = ENVIRONMENTS
    if args.env:
        envs_to_run = [e for e in ENVIRONMENTS if e["name"] == args.env]

    all_results = []

    for env_cfg in envs_to_run:
        env_name = env_cfg["name"]
        print(f"\n{'='*60}")
        print(f"Env: {env_name} | {args.episodes} episodes | {args.seeds} seeds")
        print(f"{'='*60}")

        for vname, vcfg in VARIANTS.items():
            print(f"\n  {vname} — {vcfg['description']}")

            for seed in range(args.seeds):
                try:
                    r = run_experiment(vname, vcfg, env_cfg, device, seed, args.episodes)
                    all_results.append(r)
                    lam_str = f"  lam={r['fusion_lambda']:.4f}" if r.get("fusion_lambda") else ""
                    print(f"    seed={seed:2d}  AUC={r['auc']:7.2f}  final={r['final_perf']:7.2f}  "
                          f"peak={r['peak_perf']:7.2f}  time={r['time_s']:.0f}s{lam_str}")
                except Exception as e:
                    import traceback
                    print(f"    seed={seed:2d}  FAILED: {e}")
                    traceback.print_exc()

    # ---- Summary (per environment) ----
    print(f"\n{'='*80}")
    print(f"CROSS-ENVIRONMENT SUMMARY")
    print(f"{'='*80}")

    all_paired = []
    for env_cfg in envs_to_run:
        env_name = env_cfg["name"]
        print(f"\n  {env_name}:")
        print(f"  {'Variant':<15s} {'AUC':>8s} {'95% CI':>22s} {'Final':>8s} {'Peak':>8s} {'lambda':>8s}")
        print(f"  {'-'*72}")

        for vname in VARIANTS:
            vr = [r for r in all_results if r["variant"] == vname and r["env"] == env_name]
            if not vr:
                continue
            aucs = [r["auc"] for r in vr]
            finals = [r["final_perf"] for r in vr]
            peaks = [r["peak_perf"] for r in vr]
            lambdas = [r["fusion_lambda"] for r in vr if r.get("fusion_lambda")]
            mean_auc, lo, hi, _ = bootstrap_ci(aucs)
            lam_str = f"{np.mean(lambdas):.4f}" if lambdas else "N/A"
            print(f"  {vname:<15s} {mean_auc:>8.4f} [{lo:>8.4f}, {hi:>8.4f}] "
                  f"{np.mean(finals):>8.4f} {np.mean(peaks):>8.1f} {lam_str:>8s}")

        by_seed = {}
        for r in all_results:
            if r["env"] == env_name:
                by_seed.setdefault(r["seed"], {})[r["variant"]] = r["auc"]
        common = [(s, d["dqn_hier"], d["dqn"]) for s, d in sorted(by_seed.items())
                  if "dqn_hier" in d and "dqn" in d]
        if common:
            diffs = [d1 - d2 for _, d1, d2 in common]
            mean_d, lo_d, hi_d, _ = bootstrap_ci(diffs)
            n_bs = 10000
            bs_means = [np.mean(np.random.choice(diffs, len(diffs), replace=True)) for _ in range(n_bs)]
            p_val = 2 * min(np.mean(np.array(bs_means) <= 0), np.mean(np.array(bs_means) >= 0))
            sig = " ***" if p_val < 0.001 else " **" if p_val < 0.01 else " *" if p_val < 0.05 else ""
            print(f"    dqn_hier - dqn: {mean_d:+.4f} [{lo_d:+.4f}, {hi_d:+.4f}] p={p_val:.4f}{sig} (n={len(common)})")
            all_paired.append((env_name, mean_d, lo_d, hi_d, p_val))

    # Overall meta-summary
    if len(all_paired) > 1:
        print(f"\n  OVERALL: {len(all_paired)} environments")
        wins = sum(1 for _, d, _, _, _ in all_paired if d > 0)
        sigs = sum(1 for _, _, _, _, p in all_paired if p < 0.05)
        print(f"    dqn_hier wins: {wins}/{len(all_paired)} environments")
        print(f"    Significant (p<0.05): {sigs}/{len(all_paired)} environments")

    # Save
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
