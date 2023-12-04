import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, env, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.env = env
        self.actor_logstd = 1
        self.variance_multiplier = 0.15
       
        self.fc1_a = torch.nn.Linear(state_space, hidden_size)
        self.fc2_a = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_a = torch.nn.Linear(hidden_size, action_space)

        self.fc1_c = torch.nn.Linear(state_space, hidden_size)
        self.fc2_c = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_c = torch.nn.Linear(hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)
                
    def set_logstd_ratio(self, ratio):
        self.actor_logstd = ratio
    
    def set_variance_multiplier(self, multiplier):
        self.variance_multiplier = multiplier

    def forward(self, x):
        x_a = self.fc1_a(x)
        x_a = F.relu(x_a)
        x_a = self.fc2_a(x_a)
        x_a = F.relu(x_a)
        x_a = self.fc3_a(x_a)

        x_c = self.fc1_c(x)
        x_c = F.relu(x_c)
        x_c = self.fc2_c(x_c)
        x_c = F.relu(x_c)
        x_c = self.fc3_c(x_c)

        action_mean = torch.tanh(x_a)
        action_dist = MultivariateNormal(action_mean, scale_tril=torch.diag(self.variance_multiplier * np.exp(self.actor_logstd) * torch.ones(self.action_space)))

        return action_dist, x_c