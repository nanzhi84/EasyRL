import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 action_dim):
        super(Network, self).__init__()
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
        self.eval_net = Network(self.feature_dim, self.action_dim).to(self.device)
        self.target_net = Network(self.feature_dim, self.action_dim).to(self.device)
        
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)

    def _replace_target_params(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
    
    @torch.no_grad()
    def choose_action(self, state, greedy=False):
        """基于softmax的动作选择"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.eval_net(state)
        
        if greedy:
            action = torch.argmax(q_values, dim=1).item()
        else:
            # 使用带温度参数的softmax计算动作概率
            probs = F.softmax(q_values / self.temperature, dim=1).cpu().numpy()[0]
            action = np.random.choice(self.action_dim, p=probs)
            
        return action
        
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

        q_eval = self.eval_net(batch_state).gather(1, batch_action).squeeze()
        
        # Soft Q-learning 目标值计算
        with torch.no_grad():
            q_next = self.target_net(batch_next_state)
            # 使用LogSumExp技巧保证数值稳定性
            v_next = self.temperature * torch.log(torch.sum(
                torch.exp(q_next / self.temperature), dim=1))
            q_target = batch_reward + (~batch_done) * self.gamma * v_next

        # 训练评估网络
        loss = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_value_(self.eval_net.parameters(), 0.5)
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_temperature()