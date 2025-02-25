import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

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
        
        # 参数初始化
        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.sigma.weight, -1e-3, 1e-3)
        
    def forward(self, x):
        x = self.shared_net(x)
        mu = torch.tanh(self.mu(x))
        sigma = torch.nn.functional.softplus(self.sigma(x)) + 1e-4
        return mu, sigma

class CriticNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
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
                 gamma=0.99):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.policy_net = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.value_net = CriticNetwork(feature_dim).to(self.device)
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # 用于保存采样到的 transition，格式为：
        # (state, action, log_prob, reward, next_state, done)
        self.memory = []

    def choose_action(self, state):
        """
        采样动作时使用 rsample 保留 reparameterization 的梯度，
        同时计算经过 tanh 变换后的 log 概率（包含 Jacobian 修正项）。
        返回时 action 已 detach 给环境使用，而 log_prob 保留计算图供后续更新。
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, sigma = self.policy_net(state_tensor)
        dist = Normal(mu, sigma)
        raw_action = dist.rsample()  
        action = torch.tanh(raw_action)
        log_prob = (dist.log_prob(raw_action).sum(dim=-1) -
                    torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1))
        return action.detach(), log_prob  

    def store_transition(self, state, action, log_prob, reward, next_state, done):
        self.memory.append((state, action, log_prob, reward, next_state, done))

    def learn(self, done):
        if len(self.memory) == 0:
            return

        states, actions, stored_log_probs, rewards, next_states, dones = zip(*self.memory)
        
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.stack(actions).to(self.device)
        log_probs_t = torch.stack(stored_log_probs).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # 计算 TD target：如果 done 则将下个状态的值视为0
        with torch.no_grad():
            next_values = self.value_net(next_states_t).squeeze()
        targets = rewards_t + (1 - dones_t) * self.gamma * next_values

        # 更新 Critic（评论员）
        values = self.value_net(states_t).squeeze()
        value_loss = nn.MSELoss()(values, targets)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        # 更新 Actor（演员）
        # advantage = TD_target - V(state)
        advantages = targets - values.detach()
        # 可选：标准化 advantage 提高训练稳定性
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 策略梯度：使得优势较大时增大对应动作的概率
        policy_loss = -(log_probs_t * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # 清空内存
        self.memory = []