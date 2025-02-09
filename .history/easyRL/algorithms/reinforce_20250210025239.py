import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # 策略网络结构
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # 输出动作概率分布
        )
        
    def forward(self, x):
        return self.policy(x)

class ValueNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(ValueNetwork, self).__init__()
        # 价值网络结构
        self.value = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出状态价值
        )
    
    def forward(self, x):
        return self.value(x)

class REINFORCE:
    def __init__(self, 
                 feature_dim, 
                 action_dim,
                 learning_rate=0.001,
                 gamma=0.99):
        # 初始化参数
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子

        # 初始化策略网络和价值网络
        self.policy_net = PolicyNetwork(feature_dim, action_dim)
        self.value_net = ValueNetwork(feature_dim)

        # 初始化优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # 存储episode数据
        self.episode_rewards = []  # 奖励
        self.episode_log_probs = []  # 动作对数概率
        self.episode_states = []  # 状态
        self.dones = [] # 终止标志
        
    def choose_action(self, state, greedy=False):
        # 选择动作
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)  # 获取动作概率
        m = Categorical(probs)  # 创建分类分布
        action = m.sample()  # 采样动作
        # 存储数据
        self.episode_log_probs.append(m.log_prob(action))
        self.episode_states.append(state)
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        # 存储奖励和终止标志
        self.episode_rewards.append(reward)
        self.dones.append(done)
    
    def learn(self, done):
        if not done:
            return
            
        # 计算回报
        returns = self._compute_returns()

        # 计算基线值
        states = torch.cat(self.episode_states)
        baseline = self.value_net(states).squeeze()

        # 计算优势函数
        advantages = returns - baseline.detach()
        
        # 计算策略损失
        policy_loss = []
        for log_prob, advantage in zip(self.episode_log_probs, advantages):
            policy_loss.append(-log_prob * advantage)  # 策略梯度
        policy_loss = torch.stack(policy_loss).sum()
        
        # 计算价值损失
        value_loss = nn.MSELoss()(baseline, returns)

        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 更新价值网络
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # 清空episode数据
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_states = []

    def _compute_returns(self):
        # 计算折扣回报
        returns = []
        R = 0
        for r, done in zip(reversed(self.episode_rewards), reversed(self.dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns