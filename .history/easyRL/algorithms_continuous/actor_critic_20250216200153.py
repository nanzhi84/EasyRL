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
        
        # 优化器配置
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)

    @torch.no_grad()
    def choose_action(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, sigma = self.policy_net(state_tensor)
        dist = Normal(mu, sigma)
        action = dist.sample()
        return torch.tanh(action).cpu().numpy()[0]  # 确保动作在[-1,1]之间

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append( (state, action, reward, next_state, done) )

    def learn(self, done):
        if len(self.memory) < self.batch_size:
            return
        
        # 更高效的数据采样方式
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为Tensor
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)

        # 价值网络更新
        with torch.no_grad():
            next_values = self.value_net(next_states_t).squeeze()
            target_values = rewards_t + self.gamma * next_values * (~dones_t)
            
        current_values = self.value_net(states_t).squeeze()
        value_loss = nn.MSELoss()(current_values, target_values)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)  # 改用梯度裁剪
        self.value_optimizer.step()

        # 策略网络更新
        advantages = (target_values - current_values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 标准化
        
        # 更稳定的动作转换
        clamped_actions = actions_t.clamp(-0.999, 0.999)
        raw_actions = 0.5 * torch.log( (1 + clamped_actions) / (1 - clamped_actions) )
        
        mu, sigma = self.policy_net(states_t)
        dist = Normal(mu, sigma)
        log_probs = dist.log_prob(raw_actions).sum(dim=-1)
        log_probs -= torch.log(1 - actions_t.pow(2) + 1e-6).sum(dim=-1)  # 修正概率计算
        
        policy_loss = -(log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()