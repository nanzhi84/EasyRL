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
                 gamma=0.99,
                 entropy_coef=0.01):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.policy_net = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.value_net = CriticNetwork(feature_dim).to(self.device)
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # 用于保存当前轨迹
        self.memory = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, sigma = self.policy_net(state_tensor)
        dist = Normal(mu, sigma)
        raw_action = dist.rsample()  
        # 进行 tanh 变换，并计算相应的 log 概率（包含 Jacobian 修正项）
        log_prob = (dist.log_prob(raw_action).sum(dim=-1) -
                    torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6).sum(dim=-1))
        action = torch.tanh(raw_action)
        # 与环境交互时通常只需传 action 的数值结果，可 detach 后返回
        return action.detach(), log_prob  # 注意这里保持 log_prob 的计算图

    def store_transition(self, state, action, log_prob, reward, next_state, done):
        self.memory.append((state, action, log_prob, reward, next_state, done))

    def learn(self):
        if len(self.memory) == 0:
            return
        
        # 解包轨迹
        states, actions, stored_log_probs, rewards, next_states, dones = zip(*self.memory)
        
        # 计算折扣回报（returns）
        returns = []
        R = 0
        # 从后向前累积，当遇到 done 时重置累计回报
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = r
            else:
                R = r + self.gamma * R
            returns.insert(0, R)
        
        # 转换为 Tensor
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        # actions 在 choose_action 中已为 Tensor，这里用 torch.stack
        actions_t = torch.stack(actions).to(self.device)
        # 存储时的 log_prob 已是 Tensor（且保留了计算图）
        stored_log_probs_t = torch.stack(stored_log_probs).to(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # ------------------ 更新 Critic 网络 ------------------
        values = self.value_net(states_t).squeeze()  # 形状 (N,)
        value_loss = nn.MSELoss()(values, returns_t)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        # ------------------ 更新 Actor 网络 ------------------
        advantages = returns_t - values.detach()
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 此处直接使用 choose_action 阶段保存的 log_prob，无需再做反 tanh 操作
        policy_loss = -(stored_log_probs_t * advantages).mean()
        
        # 可选：添加熵正则项鼓励探索，注意这里重新计算当前策略下的熵
        mu, sigma = self.policy_net(states_t)
        dist = Normal(mu, sigma)
        entropy = dist.entropy().sum(dim=-1)
        policy_loss -= self.entropy_coef * entropy.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # 清空轨迹
        self.memory = []