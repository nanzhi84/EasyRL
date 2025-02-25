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
        self.sigma = nn.Linear(64, action_dim)
        
        # 初始化参数
        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.sigma.weight, -1e-3, 1e-3)
        
    def forward(self, x):
        x = self.shared_net(x)
        mu = torch.tanh(self.mu(x))  
        sigma = torch.nn.functional.softplus(self.sigma(x))
        return mu, sigma
    
class CriticNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
class ActorCritic:
    def __init__(self,
                 feature_dim,
                 action_dim,
                 learning_rate=0.001,
                 gamma=0.99,
                 memory_size=10000,
                 batch_size=32):
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.value_net = CriticNetwork(feature_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)


    @torch.no_grad()
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, sigma = self.policy_net(state)
        dist = Normal(mu, sigma)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)

        # 添加雅可比修正项
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        
        return action.cpu().numpy()[0], log_prob.item

    def store_transition(self, state, action, reward, log_prob, next_state, done):
        transition = np.hstack((state, [action, log_prob, reward], next_state, [done]))
        self.memory.append(transition)

    def learn(self, done):
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样
        batch_memory = np.array(random.sample(self.memory, self.batch_size))
        
        batch_state = torch.FloatTensor(batch_memory[:, :self.feature_dim])
        batch_action = torch.FloatTensor(batch_memory[:, self.feature_dim].astype(float))
        batch_log_prob = torch.FloatTensor(batch_memory[:, self.feature_dim+1])
        batch_reward = torch.FloatTensor(batch_memory[:, self.feature_dim+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, self.feature_dim+3:-1])
        batch_done = torch.BoolTensor(batch_memory[:, -1].astype(bool))

        with torch.no_grad():
            next_value = torch.zeros_like(batch_reward) 
            mask = ~batch_done  
            next_value[mask] = self.value_net(batch_next_state[mask]).squeeze()

        batch_TD_error = batch_reward + self.gamma * next_value - self.value_net(batch_state).squeeze()

        value_loss = (batch_TD_error ** 2).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_value_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()

        policy_loss = - (batch_log_prob * batch_TD_error.detach()).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()