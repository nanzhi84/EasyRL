import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=action_dim)
        )

    def forward(self, s):
        q = self.net(s)
        return q

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=action_dim)
        )
        
    def forward(self, s):
        logits = self.net(s)
        return logits
        
    def sample_action(self, s, noise=None):
        logits = self.forward(s)
        if noise is not None:
            # 添加噪声到策略输出
            logits = logits + noise
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

class SQL(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 action_dim, 
                 learning_rate=0.001, 
                 gamma=0.99,
                 temperature=1.0,
                 temperature_decay=0.999,
                 temperature_min=0.1, 
                 memory_size=10000, 
                 batch_size=32, 
                 replace_target_iter=100):
        super().__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.learn_step_counter = 0
        
        self.memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q网络及其目标网络 (参数 θ 和 θ̄)
        self.q_net = QNetwork(self.feature_dim, self.action_dim).to(self.device)
        self.target_q_net = QNetwork(self.feature_dim, self.action_dim).to(self.device)
        
        # 策略网络 (参数 φ)
        self.policy_net = PolicyNetwork(self.feature_dim, self.action_dim).to(self.device)
        
        # 优化器
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def _replace_target_params(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    @torch.no_grad()
    def choose_action(self, state, greedy=False):
        """使用策略网络选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if greedy:
            # 贪婪模式直接选择Q值最高的动作
            q_values = self.q_net(state)
            return torch.argmax(q_values, dim=1).item()
        else:
            # 使用策略网络采样动作
            # 添加高斯噪声, 符合算法伪代码中 ξ ~ N(0, I)
            noise = torch.randn(1, self.action_dim).to(self.device) 
            return self.policy_net.sample_action(state, noise)
        
    def decay_temperature(self):
        """衰减温度参数"""
        self.temperature = max(self.temperature * self.temperature_decay, self.temperature_min)

    def store_transition(self, state, action, reward, next_state, done):
        """存储转移样本到经验池"""
        transition = np.hstack((state, [action, reward], next_state, [done]))
        self.memory.append(transition)

    def learn(self, done):
        """训练网络"""
        if len(self.memory) < self.batch_size:
            return
        
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
        
        # 随机采样
        batch_memory = np.array(random.sample(self.memory, self.batch_size))
        
        batch_state = torch.FloatTensor(batch_memory[:, :self.feature_dim]).to(self.device)
        batch_action = torch.LongTensor(batch_memory[:, self.feature_dim].astype(int)).unsqueeze(1).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, self.feature_dim+1]).to(self.device)
        batch_next_state = torch.FloatTensor(batch_memory[:, self.feature_dim+2:-1]).to(self.device)
        batch_done = torch.BoolTensor(batch_memory[:, -1].astype(bool)).to(self.device)

        # 1. 更新Q网络参数 (参数 θ)
        q_values = self.q_net(batch_state).gather(1, batch_action).squeeze()
        
        with torch.no_grad():
            # 计算目标soft值 - 使用LogSumExp技巧实现数值稳定性
            q_next = self.target_q_net(batch_next_state)
            v_next = self.temperature * torch.log(torch.sum(
                torch.exp(q_next / self.temperature), dim=1))
            q_target = batch_reward + (~batch_done) * self.gamma * v_next
        
        # Q网络损失和优化
        q_loss = F.mse_loss(q_values, q_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 0.5)
        self.q_optimizer.step()
        
        # 2. 更新策略网络参数 (参数 φ)
        # 在算法伪代码中，对每个状态采样多个动作，这里采样M个动作
        M = 10
        policy_loss = 0
        
        for i in range(self.batch_size):
            state = batch_state[i].unsqueeze(0)
            next_state = batch_next_state[i].unsqueeze(0)
            
            # 从策略网络采样多个动作
            sampled_actions = []
            action_logits = []
            
            for _ in range(M):
                noise = torch.randn(1, self.action_dim).to(self.device)
                logits = self.policy_net(state)
                action_logits.append(logits)
                
                # 添加噪声进行探索
                noisy_logits = logits + noise
                probs = F.softmax(noisy_logits, dim=1)
                action = torch.multinomial(probs, 1).item()
                sampled_actions.append(action)
            
            # 计算Q值
            with torch.no_grad():
                q_values = self.q_net(state)
            
            # 计算策略梯度
            for j in range(M):
                action = sampled_actions[j]
                log_prob = F.log_softmax(action_logits[j], dim=1)[0, action]
                # 策略梯度公式：最大化 E[log π(a|s) * (Q(s,a) - baseline)]
                advantage = q_values[0, action] - torch.mean(q_values)
                policy_loss -= log_prob * advantage
        
        policy_loss /= (self.batch_size * M)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()

        self.learn_step_counter += 1
        self.decay_temperature()