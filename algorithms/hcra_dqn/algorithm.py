import torch
import numpy as np
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.hcra_dqn.network import HCRAQNModel
from algorithms.hcra_dqn.replay_buffer import SequenceReplayBuffer


class HCRADQNAlgorithm(BaseAlgorithm):
    def __init__(self, obs_dim: int, action_num: int, gamma: float = 0.99,
                 learning_rate: float = 1e-4, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay_steps: int = 100000,
                 batch_size: int = 64, buffer_size: int = 50000,
                 learn_start: int = 5000, target_update_freq: int = 500,
                 tau: float = 0.005, grad_clip: float = 10.0,
                 hidden_dim: int = 128, mid_dim: int = 64,
                 sequence_length: int = 16, attention_heads: int = 4,
                 attention_dim: int = 64, num_segments: int = 4,
                 num_refinements: int = 2, fusion_lambda: float = 0.0,
                 contract_reg_coeff: float = 0.01,
                 device=None, logger=None):
        self.device = device
        self.logger = logger
        self.action_num = action_num
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self.sequence_length = sequence_length
        self.contract_reg_coeff = contract_reg_coeff

        segment_size = sequence_length // num_segments
        assert sequence_length % num_segments == 0, (
            f"sequence_length ({sequence_length}) must be divisible by num_segments ({num_segments})"
        )

        self.model = HCRAQNModel(
            obs_dim, action_num, attention_dim, attention_heads,
            hidden_dim, mid_dim, num_segments, segment_size,
            num_refinements, fusion_lambda, device,
        )
        self.optimizer = torch.optim.Adam(
            params=self.model.online_net.parameters(),
            lr=learning_rate,
        )
        self.replay_buffer = SequenceReplayBuffer(buffer_size)
        self.train_step = 0
        self.total_step = 0

    def select_action(self, obs, legal_action, **kwargs) -> int:
        self.total_step += 1
        decay_frac = min(1.0, self.total_step / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_frac

        if np.random.random() < self.epsilon:
            legal_indices = [i for i, a in enumerate(legal_action) if a > 0]
            if legal_indices:
                return int(np.random.choice(legal_indices))
            return int(np.random.randint(0, self.action_num))

        obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32).to(self.device)
        seq_tensor = None
        sequence = kwargs.get("sequence", None)
        if sequence is not None:
            seq_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32).to(self.device)

        q_values = self.model.get_q_values(obs_tensor, seq_tensor).cpu().numpy()[0]

        legal_mask = np.array(legal_action, dtype=np.float32)
        q_values = q_values * legal_mask
        valid_indices = [i for i, v in enumerate(legal_action) if v > 0]
        if valid_indices:
            return int(np.argmax(q_values))
        return int(np.random.randint(0, q_values.shape[0]))

    def select_greedy_action(self, obs, legal_action, **kwargs) -> int:
        obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32).to(self.device)
        seq_tensor = None
        sequence = kwargs.get("sequence", None)
        if sequence is not None:
            seq_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32).to(self.device)

        q_values = self.model.get_q_values(obs_tensor, seq_tensor).cpu().numpy()[0]

        legal_mask = np.array(legal_action, dtype=np.float32)
        q_values = q_values * legal_mask
        valid_indices = [i for i, v in enumerate(legal_action) if v > 0]
        if valid_indices:
            return int(np.argmax(q_values))
        return int(np.random.randint(0, q_values.shape[0]))

    def store_transition(self, **kwargs):
        seq = kwargs.get("sequence", None)
        next_seq = kwargs.get("next_sequence", None)
        if seq is None:
            seq = np.zeros((self.sequence_length, self.obs_dim), dtype=np.float32)
        if next_seq is None:
            next_seq = np.zeros((self.sequence_length, self.obs_dim), dtype=np.float32)
        self.replay_buffer.push(
            kwargs["obs"], kwargs["action"], kwargs["reward"],
            kwargs["next_obs"], kwargs["done"], kwargs["legal_action"],
            seq, next_seq,
        )

    def learn(self):
        if len(self.replay_buffer) < self.learn_start:
            return None

        obs, action, reward, next_obs, done, legal_action, sequence, next_sequence = self.replay_buffer.sample(self.batch_size)

        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action_t = torch.tensor(action, dtype=torch.long).to(self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        done_t = torch.tensor(done, dtype=torch.float32).to(self.device)
        legal_action_t = torch.tensor(legal_action, dtype=torch.float32).to(self.device)
        seq_t = torch.tensor(sequence, dtype=torch.float32).to(self.device)
        next_seq_t = torch.tensor(next_sequence, dtype=torch.float32).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        q_values, seg_attn, step_attn, contract_kl = self.model.online_net(obs_t, seq_t)
        q_value = q_values.gather(1, action_t)

        with torch.no_grad():
            next_q_online, _, _, _ = self.model.online_net(next_obs_t, next_seq_t)
            next_q_target, _, _, _ = self.model.target_net(next_obs_t, next_seq_t)

            next_legal_mask = next_q_online * legal_action_t
            best_actions = next_legal_mask.argmax(1, keepdim=True)

            next_q_max = next_q_target.gather(1, best_actions)
            td_target = reward_t + self.gamma * next_q_max * (1.0 - done_t)

        loss = torch.nn.functional.mse_loss(q_value, td_target)

        contract_reg = self.contract_reg_coeff * contract_kl
        total_loss = loss + contract_reg

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_step += 1

        if self.train_step % self.target_update_freq == 0:
            self.model.update_target_soft()

        return total_loss.item()

    def get_fusion_lambda(self) -> float:
        return torch.sigmoid(self.model.online_net.fusion_lambda).item()

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model.load(path)
