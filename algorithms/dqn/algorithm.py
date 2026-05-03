import torch
import numpy as np
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.dqn.network import DQNModel
from algorithms.dqn.replay_buffer import ReplayBuffer


class DQNAlgorithm(BaseAlgorithm):
    def __init__(self, obs_dim: int, action_num: int, gamma: float = 0.99,
                 learning_rate: float = 1e-4, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay_steps: int = 100000,
                 batch_size: int = 64, buffer_size: int = 50000,
                 learn_start: int = 5000, target_update_freq: int = 500,
                 tau: float = 0.005, grad_clip: float = 10.0,
                 hidden_dim: int = 128, mid_dim: int = 64,
                 device=None, logger=None):
        self.device = device
        self.logger = logger
        self.action_num = action_num
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        self.model = DQNModel(obs_dim, action_num, hidden_dim, mid_dim, device)
        self.optimizer = torch.optim.Adam(
            params=self.model.online_net.parameters(),
            lr=learning_rate,
        )
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.train_step = 0
        self.total_step = 0

    def select_action(self, obs, legal_action, **kwargs) -> int:
        self.total_step += 1
        # Linear epsilon decay based on environment steps
        decay_frac = min(1.0, self.total_step / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_frac

        if np.random.random() < self.epsilon:
            legal_indices = [i for i, a in enumerate(legal_action) if a > 0]
            if legal_indices:
                return int(np.random.choice(legal_indices))
            return int(np.random.randint(0, self.action_num))

        obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32).to(self.device)
        q_values = self.model.get_q_values(obs_tensor).cpu().numpy()[0]

        legal_mask = np.array(legal_action, dtype=np.float32)
        q_values = q_values * legal_mask
        return int(np.argmax(q_values))

    def select_greedy_action(self, obs, legal_action, **kwargs) -> int:
        obs_tensor = torch.tensor(np.array([obs]), dtype=torch.float32).to(self.device)
        q_values = self.model.get_q_values(obs_tensor).cpu().numpy()[0]

        legal_mask = np.array(legal_action, dtype=np.float32)
        q_values = q_values * legal_mask
        return int(np.argmax(q_values))

    def store_transition(self, **kwargs):
        self.replay_buffer.push(
            kwargs["obs"], kwargs["action"], kwargs["reward"],
            kwargs["next_obs"], kwargs["done"], kwargs["legal_action"]
        )

    def learn(self):
        if len(self.replay_buffer) < self.learn_start:
            return None

        obs, action, reward, next_obs, done, legal_action = self.replay_buffer.sample(self.batch_size)

        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action_t = torch.tensor(action, dtype=torch.long).to(self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        done_t = torch.tensor(done, dtype=torch.float32).to(self.device)
        legal_action_t = torch.tensor(legal_action, dtype=torch.float32).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        q_values = self.model.online_net(obs_t)
        q_value = q_values.gather(1, action_t)

        with torch.no_grad():
            next_q_online = self.model.online_net(next_obs_t)
            next_q_target = self.model.target_net(next_obs_t)

            next_legal_mask = next_q_online * legal_action_t
            best_actions = next_legal_mask.argmax(1, keepdim=True)

            next_q_max = next_q_target.gather(1, best_actions)
            td_target = reward_t + self.gamma * next_q_max * (1.0 - done_t)

        loss = torch.nn.functional.mse_loss(q_value, td_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_step += 1

        if self.train_step % self.target_update_freq == 0:
            self.model.update_target_soft()

        return loss.item()

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model.load(path)
