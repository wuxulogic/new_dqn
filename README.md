# HCRA-DQN: Hierarchical Causal Attention DQN

Hierarchical causal attention mechanism for temporal credit assignment in reinforcement learning.

**实验状态**：MiniGrid FourRooms 上统计显著优于标准 DQN（p=0.0034）。完整实验报告见 [docs/experiment_report.md](docs/experiment_report.md)。

## 快速开始

```bash
# 验证：MiniGrid 在线训练对比 dqn vs dqn_hier
python run_sample_efficiency.py --env MiniGrid-FourRooms-v0 --episodes 500 --seeds 5

# 单算法在线训练
python run_experiment.py --algorithm hcra_dqn --env FrozenLake-v1 --seed 42

# 多算法对比
python run_comparison.py --env FrozenLake-v1 --algorithms "dqn,hcra_dqn" --seeds 5
```

## 已验证的创新

| 组件 | 状态 | 说明 |
|------|------|------|
| **层次因果注意力** | ✅ 已验证 | 段级+步级两层因果注意力，MiniGrid 上 p=0.0034 |
| 可学习融合系数 λ | ✅ 有效 | 从 0.5 自适应收敛到 0.23，找到最优平衡点 |
| 递归精炼 | ❌ 无效 | 增加方差，无一致收益 |
| 收缩正则化 | ❌ 有害 | 任意非零系数均降低性能 |
| PER（优先经验回放） | — | 非本文贡献，已从架构中移除 |

## 项目结构

```
czh_dqn_benchmark/
  algorithms/
    dqn/                  # 标准 DQN (Double DQN + soft target)
    qlearning/            # 表格 Q-Learning
    hcra_dqn/             # HCRA-DQN（核心贡献）
  trainers/               # 训练器（dqn, qlearning, hcra_dqn）
  envs/                   # 环境适配器 + POMDP wrapper
  configs/                # 训练配置
  docs/                   # 文档
    dqn_vs_qlearning.md     # DQN vs Q-Learning 创新性分析（中文）
    experiment_report.md    # 完整实验报告（论文格式）
  run_experiment.py       # 单算法训练入口
  run_comparison.py       # 多算法对比入口
  run_sample_efficiency.py # 在线样本效率对比入口
```

## 文档

- [实验报告（论文格式）](docs/experiment_report.md) — 完整的方法、实验、讨论、结论
- [DQN vs Q-Learning 创新性分析](docs/dqn_vs_qlearning.md) — 中文分析文档

## 架构

```
obs (s_t)                           sequence (s_t-15,...,s_t)
  |                                       |
  +--> [base_net: 128→64]                 +--> [seq_encoder]
  |                                       |
  +--> base_q_head --> Q_base              +--> [HierarchicalCausalAttention]
                                                  |
                                           Level 1: 段级因果注意力
                                           Level 2: 步级因果注意力
                                                  |
                                           +--> attention_q_head --> Q_attn

Q_total = Q_base + sigmoid(lambda) * Q_attn
```

## 最终推荐算法

**dqn_hier（= HCRA-DQN）**：
- 4 段 × 4 步层次因果注意力
- 1 轮（无递归精炼）
- λ 初始化为 0.0（sigmoid=0.5）
- 无收缩正则化
- 标准均匀采样 replay buffer
