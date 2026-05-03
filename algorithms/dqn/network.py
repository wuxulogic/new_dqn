import torch
import torch.nn as nn
import numpy as np


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, action_num: int, hidden_dim: int = 128, mid_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
            make_fc_layer(mid_dim, action_num),
        )

    def forward(self, x):
        return self.net(x)


class DQNModel:
    def __init__(self, input_dim: int, action_num: int, hidden_dim: int = 128,
                 mid_dim: int = 64, device=None):
        self.device = device
        self.online_net = DQNNetwork(input_dim, action_num, hidden_dim, mid_dim).to(device)
        self.target_net = DQNNetwork(input_dim, action_num, hidden_dim, mid_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def get_q_values(self, obs_tensor):
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(obs_tensor)
        return q_values

    def get_target_q_values(self, obs_tensor):
        with torch.no_grad():
            q_values = self.target_net(obs_tensor)
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
