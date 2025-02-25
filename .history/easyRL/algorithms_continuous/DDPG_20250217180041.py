import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random

class ActorNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64, action_dim)
        
        # 参数初始化
        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        
    def forward(self, x):
        x = self.shared_net(x)
        mu = torch.tanh(self.mu(x)) + torch.normal(mean=0.0, std=0.1, size=self.mu(x).shape).to(self.device)
        return mu

class CriticNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)
    
class DDPG:
    def __init__(self,
                 feature_dim,
                 action_dim,
                 learning_rate=0.001,
                 gamma=0.99,
                 memory_size=10000,
                 batch_size=32,
                 replace_target_iter=100):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.target_policy_net = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.value_net = CriticNetwork(feature_dim, action_dim).to(self.device)
        self.target_value_net = CriticNetwork(feature_dim, action_dim).to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        self.memory = deque(maxlen=memory_size)

    def _replace_target_params(self):
        tau = 0.005
        # 对Critic网络进行软更新
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
        # 对Actor网络进行软更新
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
    
    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, sigma = self.policy_net(state_tensor)
        dist = Normal(mu, sigma)
        raw_action = dist.rsample()  
        action = torch.tanh(raw_action)

        return action.detach().squeeze().cpu().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, done, step_counter):
        if len(self.memory) < self.batch_size:
            return
        
        if step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        batch = random.sample(self.memory, self.batch_size)
        states, actions,rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        log_probs_t = torch.stack(stored_log_probs).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_actions = self.target_policy_net(next_states_t)
            next_values = self.target_value_net(next_states_t, next_actions).squeeze()
        targets = (rewards_t + (1 - dones_t) * self.gamma * next_values).squeeze()

        # 更新 Critic
        values = self.value_net(states_t).squeeze()
        value_loss = nn.MSELoss()(values, targets)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        # 更新 Actor
        policy_loss = -self.value_net(states_t, self.policy_net(states_t)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()