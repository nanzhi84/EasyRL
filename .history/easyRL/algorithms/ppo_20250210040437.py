import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 可学习log标准差

    def forward(self, x):
        x = self.common(x)
        mean = torch.tanh(self.mean(x))  # 输出范围[-1,1]
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

class PPO:
    def __init__(self, feature_dim, action_dim, 
                 lr=3e-4, gamma=0.99, clip_eps=0.2,
                 entropy_coef=0.01, ppo_epochs=10, 
                 batch_size=64, gae_lambda=0.95):
        
        # 网络结构
        self.actor = PolicyNetwork(feature_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=lr)
        
        # 超参数配置
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        
        # 经验缓冲区
        self.memory = []

    def store_transition(self, transition):
        self.memory.append(transition)

    def choose_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            value = self.critic(state).squeeze()
            
        return action.numpy(), log_prob.item(), value.item()

    def update(self):
        # 数据转换
        states, actions, old_log_probs, values, rewards, dones = map(np.array, zip(*self.memory))
        
        # 计算GAE和returns
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮PPO更新
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start+self.batch_size]
                
                # 新策略计算
                means, stds = self.actor(states[batch_idx])
                dist = Normal(means, stds.expand_as(means))
                new_log_probs = dist.log_prob(actions[batch_idx]).sum(-1)
                entropy = dist.entropy().mean()
                
                # 策略损失
                ratio = (new_log_probs - old_log_probs[batch_idx]).exp()
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                values_pred = self.critic(states[batch_idx]).squeeze()
                value_loss = 0.5 * (returns[batch_idx] - values_pred).pow(2).mean()
                
                # 总损失
                loss = policy_loss + value_loss - self.entropy_coef * entropy
                
                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()
        
        self.memory = []

    def _compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t+1] if t+1 < len(rewards) else 0.0
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return advantages, returns
