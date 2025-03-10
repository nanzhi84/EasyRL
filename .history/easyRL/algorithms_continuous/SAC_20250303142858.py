import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.net
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std
        
    def sample(self, x):
        mean, std = self.forward(x)
        normal = Normal(mean, std)
        
        # 使用重参数化技巧采样
        x_t = normal.rsample()
        
        # 利用tanh将动作限制在[-1, 1]范围内
        action = torch.tanh(x_t)
        
        # 计算log概率，考虑tanh变换的效果
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value

class SAC:
    def __init__(self,
                 feature_dim,
                 action_dim,
                 learning_rate=0.0003,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 memory_size=1000000,
                 batch_size=256,
                 auto_entropy_tuning=True):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 策略网络
        self.policy_net = PolicyNetwork(feature_dim, action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Q网络，使用两个Q网络减少过度估计
        self.q_net1 = QNetwork(feature_dim, action_dim).to(self.device)
        self.q_net2 = QNetwork(feature_dim, action_dim).to(self.device)
        
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=learning_rate)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=learning_rate)
        
        # 目标Q网络
        self.target_q_net1 = QNetwork(feature_dim, action_dim).to(self.device)
        self.target_q_net2 = QNetwork(feature_dim, action_dim).to(self.device)
        
        # 硬拷贝参数到目标网络
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        
        # 自动调整熵权重
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(memory_size)
    
    def choose_action(self, state, evaluate=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            # 评估模式下，直接使用均值作为动作
            with torch.no_grad():
                mean, _ = self.policy_net(state)
                return torch.tanh(mean).cpu().numpy()[0]
        else:
            # 训练模式下，使用策略网络采样动作
            with torch.no_grad():
                action, _ = self.policy_net.sample(state)
                return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)
    
    def soft_update(self, target, source):
        """软更新目标网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def learn(self, done):
        """训练网络"""
        if len(self.memory) < self.batch_size:
            return
        
        # 从经验回放缓冲区中采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 更新Q网络
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_net.sample(next_states)
            next_q_value1 = self.target_q_net1(next_states, next_actions)
            next_q_value2 = self.target_q_net2(next_states, next_actions)
            next_q_value = torch.min(next_q_value1, next_q_value2) - self.alpha * next_log_probs
            expected_q_value = rewards + self.gamma * (1 - dones) * next_q_value
        
        # 计算Q网络的损失
        q_value1 = self.q_net1(states, actions)
        q_value2 = self.q_net2(states, actions)
        q_value_loss1 = F.mse_loss(q_value1, expected_q_value)
        q_value_loss2 = F.mse_loss(q_value2, expected_q_value)
        
        # 更新Q网络
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.q_optimizer1.step()
        
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.q_optimizer2.step()
        
        # 更新策略网络
        new_actions, log_probs = self.policy_net.sample(states)
        q_value1 = self.q_net1(states, new_actions)
        q_value2 = self.q_net2(states, new_actions)
        min_q_value = torch.min(q_value1, q_value2)
        
        policy_loss = (self.alpha * log_probs - min_q_value).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 自动调整熵权重
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # 软更新目标网络
        self.soft_update(self.target_q_net1, self.q_net1)
        self.soft_update(self.target_q_net2, self.q_net2)