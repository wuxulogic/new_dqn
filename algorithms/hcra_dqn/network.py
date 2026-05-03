import math
import torch
import torch.nn as nn
import numpy as np


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class CausalTemporalAttention(nn.Module):
    """Multi-head causal self-attention.

    Each time step can only attend to itself and previous steps (not future).
    """

    def __init__(self, input_dim: int, attention_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        assert attention_dim % num_heads == 0

        self.key_proj = make_fc_layer(input_dim, attention_dim)
        self.query_proj = make_fc_layer(input_dim, attention_dim)
        self.value_proj = make_fc_layer(input_dim, attention_dim)
        self.out_proj = make_fc_layer(attention_dim, attention_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, sequence):
        batch_size, seq_len, _ = sequence.shape

        keys = self.key_proj(sequence).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.query_proj(sequence).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(sequence).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=sequence.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attention_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class HierarchicalCausalAttention(nn.Module):
    """Two-level hierarchical causal attention with recursive refinement.

    Level 1 (segment): groups timesteps into segments, pools within each,
        applies causal attention over segments to find important time periods.
    Level 2 (step): applies causal attention over individual steps, biased by
        segment-level importance weights.
    Recursive refinement: re-weights features using step-level attention and
        re-runs the two-level process, producing increasingly focused attention.
    """

    def __init__(self, input_dim: int, attention_dim: int, num_heads: int = 4,
                 num_segments: int = 4, segment_size: int = 4, num_refinements: int = 2):
        super().__init__()
        self.num_segments = num_segments
        self.segment_size = segment_size
        self.seq_len = num_segments * segment_size
        self.num_refinements = num_refinements

        self.seg_attention = CausalTemporalAttention(input_dim, attention_dim, num_heads)
        self.step_attention = CausalTemporalAttention(input_dim, attention_dim, num_heads)
        self.context_proj = make_fc_layer(attention_dim, attention_dim)

    def forward(self, sequence):
        batch_size = sequence.shape[0]
        flat_mode = self.num_segments == 1

        weighted_seq = sequence
        prev_step_attn = None
        contract_kl = 0.0

        for refine_idx in range(self.num_refinements):
            if flat_mode:
                # Flat causal attention — no segment-level gating
                step_out, step_attn = self.step_attention(weighted_seq)
                seg_attn = None
            else:
                # ---- Level 1: Segment-level attention ----
                segs = weighted_seq.view(batch_size, self.num_segments, self.segment_size, -1)
                seg_pooled = segs.mean(dim=2)  # [B, N_seg, D]

                seg_out, seg_attn = self.seg_attention(seg_pooled)
                seg_importance = seg_attn.mean(dim=1)[:, -1, :]  # [B, N_seg]

                # ---- Level 2: Step-level attention ----
                seg_weight_per_step = seg_importance.unsqueeze(-1).repeat(1, 1, self.segment_size)
                seg_weight_per_step = seg_weight_per_step.view(batch_size, self.seq_len, 1)
                biased_seq = weighted_seq * (1.0 + seg_weight_per_step)
                step_out, step_attn = self.step_attention(biased_seq)
            step_weights = step_attn.mean(dim=1)[:, -1, :]  # [B, seq_len]

            # Contraction: KL divergence from previous attention to current
            if prev_step_attn is not None and refine_idx > 0:
                prev_mean = prev_step_attn.mean(dim=1)[:, -1, :]  # [B, seq_len]
                curr_mean = step_attn.mean(dim=1)[:, -1, :]
                contract_kl = contract_kl + _kl_divergence(prev_mean, curr_mean)

            prev_step_attn = step_attn.detach()

            # Re-weight features by step attention for next refinement
            weighted_seq = sequence * (1.0 + step_weights.unsqueeze(-1))

        context = self.context_proj(step_out[:, -1, :])
        return context, seg_attn, step_attn, contract_kl


def _kl_divergence(p, q):
    """KL(p || q) for batched probability distributions."""
    p = p.clamp(1e-9, 1.0)
    q = q.clamp(1e-9, 1.0)
    return (p * (p.log() - q.log())).sum(-1).mean()


class HCRAQNetwork(nn.Module):
    """HCRA-DQN: Hierarchical Causal Recursive Attention DQN.

    Q_total = Q_base(s) + sigmoid(lambda) * Q_attention(sequence)

    Architecture:
      base_net: MLP(s) -> base_q
      seq_encoder: encodes each frame -> hierarchical causal attention -> attention_q
      fusion_lambda: learned scalar controlling attention contribution
    """

    def __init__(self, input_dim: int, action_num: int, attention_dim: int = 64,
                 num_heads: int = 4, hidden_dim: int = 128, mid_dim: int = 64,
                 num_segments: int = 4, segment_size: int = 4,
                 num_refinements: int = 2, fusion_lambda: float = 0.0):
        super().__init__()

        self.base_net = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )
        self.base_q_head = make_fc_layer(mid_dim, action_num)

        self.seq_encoder = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attn_seq_dim = hidden_dim

        self.hierarchy_attention = HierarchicalCausalAttention(
            hidden_dim, attention_dim, num_heads,
            num_segments, segment_size, num_refinements,
        )

        self.attention_q_head = nn.Sequential(
            make_fc_layer(attention_dim, mid_dim),
            nn.ReLU(),
            make_fc_layer(mid_dim, action_num),
        )

        self.fusion_lambda = nn.Parameter(torch.tensor(fusion_lambda))

    def forward(self, obs, sequence=None):
        base_hidden = self.base_net(obs)
        base_q = self.base_q_head(base_hidden)

        if sequence is not None:
            batch_size, seq_len, _ = sequence.shape
            flat_seq = sequence.view(batch_size * seq_len, -1)
            encoded_seq = self.seq_encoder(flat_seq)
            encoded_seq = encoded_seq.view(batch_size, seq_len, self.attn_seq_dim)

            context, seg_attn, step_attn, contract_kl = self.hierarchy_attention(encoded_seq)
            attention_q = self.attention_q_head(context)

            lam = torch.sigmoid(self.fusion_lambda)
            q_values = base_q + lam * attention_q
            return q_values, seg_attn, step_attn, contract_kl
        else:
            return base_q, None, None, 0.0


class HCRAQNModel:
    """Model manager for HCRA-DQN (online + target network, soft updates)."""

    def __init__(self, input_dim: int, action_num: int, attention_dim: int = 64,
                 num_heads: int = 4, hidden_dim: int = 128, mid_dim: int = 64,
                 num_segments: int = 4, segment_size: int = 4,
                 num_refinements: int = 2, fusion_lambda: float = 0.0, device=None):
        self.device = device
        self.online_net = HCRAQNetwork(
            input_dim, action_num, attention_dim, num_heads,
            hidden_dim, mid_dim, num_segments, segment_size,
            num_refinements, fusion_lambda,
        ).to(device)
        self.target_net = HCRAQNetwork(
            input_dim, action_num, attention_dim, num_heads,
            hidden_dim, mid_dim, num_segments, segment_size,
            num_refinements, fusion_lambda,
        ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def get_q_values(self, obs_tensor, seq_tensor=None):
        self.online_net.eval()
        with torch.no_grad():
            q_values, _, _, _ = self.online_net(obs_tensor, seq_tensor)
        return q_values

    def get_target_q_values(self, obs_tensor, seq_tensor=None):
        with torch.no_grad():
            q_values, _, _, _ = self.target_net(obs_tensor, seq_tensor)
        return q_values

    def update_target_hard(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update_target_soft(self, tau: float = 0.005):
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def set_train_mode(self):
        self.online_net.train()

    def set_eval_mode(self):
        self.online_net.eval()

    def save(self, path: str):
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.online_net.state_dict().items()}
        torch.save(state_dict_cpu, path)

    def load(self, path: str):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.online_net.state_dict())
