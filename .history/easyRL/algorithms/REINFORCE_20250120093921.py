import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.policy(x)
    

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
    
    def store_transition(self, reward):
        self.episode_rewards.append(reward)
    
    def learn(self, done):
        if not done:
            return
            
        # Calculate returns
        returns = []
        R = 0
        for r in self.episode_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()
        
        # Optimize policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.episode_rewards = []
        self.episode_log_probs = []