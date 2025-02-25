import torch
import torch.nn as nn
import torch.optim as optim
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
        
        # 参数初始化
        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        
    def forward(self, x):
        x = self.shared_net(x)
        mu = torch.tanh(self.mu(x))
        return mu

class CriticNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.net2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2
    
class TD3:
    def __init__(self,
                 feature_dim,
                 action_dim,
                 learning_rate=0.001,
                 gamma=0.99,
                 memory_size=10000,
                 batch_size=32,
                 replace_target_iter=100,
                 noise_std_init=0.3,    
                 noise_decay=0.998,     
                 noise_std_min=0.01,
                 policy_noise=0.2,          
                 noise_clip=0.5):  
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.target_policy_net = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.value_net = CriticNetwork(feature_dim, action_dim).to(self.device)
        self.target_value_net = CriticNetwork(feature_dim, action_dim).to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        self.memory = deque(maxlen=memory_size)
        
        # 添加噪声参数
        self.noise_std = noise_std_init
        self.noise_decay = noise_decay
        self.noise_std_min = noise_std_min

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def _replace_target_params(self):
        tau = 0.005
        # 对Critic网络进行软更新
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
        # 对Actor网络进行软更新
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
    
    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu = self.policy_net(state_tensor)
        noise = torch.normal(mean=0.0, std=self.noise_std, size=mu.shape).to(self.device)
        action = torch.clamp(mu + noise, -1.0, 1.0)
        return action.detach().squeeze().cpu().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, done, step_counter):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_actions = self.target_policy_net(next_states_t)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_policy_net(next_states_t) + noise).clamp(-1.0, 1.0)
            
            q1, q2 = self.target_value_net(next_states_t, next_actions)
            next_values = torch.min(q1, q2).squeeze()

        targets = (rewards_t + (1 - dones_t) * self.gamma * next_values).squeeze()

        # 更新 Critic
        current_q1, current_q2 = self.value_net(states_t, actions_t)
        value_loss = nn.MSELoss()(current_q1, targets) + nn.MSELoss()(current_q2, targets)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        if step_counter % self.replace_target_iter == 0:
            # 更新 Actor
            policy_actions = self.policy_net(states_t)
            policy_loss = -self.value_net(states_t, policy_actions)[0].mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()
        
            self._replace_target_params()

        # 衰减噪声
        self.noise_std = max(self.noise_std * self.noise_decay, self.noise_std_min)