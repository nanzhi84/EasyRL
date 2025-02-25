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
        
        # 更合理的参数初始化
        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.constant_(self.sigma.bias, -1)  # 初始探索性更小
        
    def forward(self, x):
        x = self.shared_net(x)
        mu = torch.tanh(self.mu(x))  # 输出范围[-1,1]
        sigma = torch.nn.functional.softplus(self.sigma(x)) + 1e-4  # 保证正值
        return mu, sigma
    
class CriticNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),  # 统一网络结构
            nn.ReLU(),
            nn.Linear(64, 64),
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
        
        # 网络初始化
        self.policy_net = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.value_net = CriticNetwork(feature_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)


    @torch.no_grad()
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, sigma = self.policy_net(state)
        dist = Normal(mu, sigma)
        action = torch.tanh(dist.sample())
        
        return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        transition = np.hstack([
            state,
            action, 
            reward,
            next_state,
            done
        ])
        self.memory.append(transition)

    def learn(self, done):
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样
        batch_memory = np.array(random.sample(self.memory, self.batch_size))
        
        # 动态计算索引
        state_end = self.feature_dim
        action_end = state_end + self.action_dim
        reward_idx = action_end
        next_state_start = reward_idx + 1
        next_state_end = next_state_start + self.feature_dim
        done_idx = next_state_end

        # 转换为Tensor并转移到设备
        batch_state = torch.FloatTensor(batch_memory[:, :state_end]).to(self.device)
        batch_action = torch.FloatTensor(batch_memory[:, state_end:action_end]).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, reward_idx]).to(self.device)
        batch_next_state = torch.FloatTensor(batch_memory[:, next_state_start:next_state_end]).to(self.device)
        batch_done = torch.BoolTensor(batch_memory[:, done_idx].astype(bool)).to(self.device)

        # 计算TD目标
        with torch.no_grad():
            next_value = torch.zeros_like(batch_reward)
            mask = ~batch_done
            next_value[mask] = self.value_net(batch_next_state[mask]).squeeze()
            target_value = batch_reward + self.gamma * next_value

        current_value = self.value_net(batch_state).squeeze()
        value_loss = nn.MSELoss()(current_value, target_value)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_value_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()

        batch_TD_error = (target_value - current_value).detach()

        eps = 1e-6
        batch_action_clamped = batch_action.clamp(-1 + eps, 1 - eps)
        raw_action = 0.5 * torch.log((1 + batch_action_clamped) / (1 - batch_action_clamped))  # atanh

        mu, sigma = self.policy_net(batch_state)
        dist = Normal(mu, sigma)
        batch_log_prob = dist.log_prob(raw_action)
        batch_log_prob -= torch.log(1 - batch_action.pow(2) + 1e-6)
        batch_log_prob = batch_log_prob.sum(dim=-1)

        policy_loss = - (batch_log_prob * batch_TD_error.detach()).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()