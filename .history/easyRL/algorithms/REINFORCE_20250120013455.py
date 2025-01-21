import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class Network(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 action_dim):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=action_dim)
        )

    def forward(self, s):
        q = self.net(s)
        return q

class REINFORCE:
    def __init__(self, 
                 feature_dim, 
                 action_dim,
                 learning_rate=0.001,
                 gamma=0.99):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.policy_net = PolicyNetwork(feature_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.episode_rewards = []
        self.episode_log_probs = []
        
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        self.episode_log_probs.append(m.log_prob(action))
        return action.item()
    
    def store_transition(self, state, action, reward, next_state):
        self.episode_rewards.append(reward)
    
    def learn(self, done):
        if not done:
            return
            
        # 计算回报
        returns = []
        R = 0
        for r in self.episode_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # 标准化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算策略损失
        policy_loss = []
        for log_prob, R in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()
        
        # 优化策略网络
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # 清空episode数据
        self.episode_rewards = []
        self.episode_log_probs = []