import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class CriticNetwork(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class PPO:
    def __init__(self, 
                 feature_dim,
                 action_dim,
                 learning_rate=0.001,
                 gamma=0.99,
                 clip_epsilon=0.2,
                 entropy_coef=0.01,
                 ppo_epochs=4,
                 batch_size=64,
                 memory_size=1000):
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # 网络初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(feature_dim).to(self.device)
        
        # 经验缓冲区
        self.memory = deque(maxlen=memory_size)
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=learning_rate)

    def store_transition(self, state, action, log_prob, value, reward, done):
        transition = (state, action, log_prob, value, reward, done)
        self.memory.append(transition)

    def choose_action(self, state, greedy=False):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
            value = self.critic(state)
        
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 转换经验数据
        states, actions, old_log_probs, values, rewards, dones = zip(*self.memory)
        
        # 计算GAE和回报
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        # 转换为张量
        states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # 多轮PPO更新
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = np.arange(len(self.memory))
            np.random.shuffle(indices)
            
            # 小批量更新
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                # 计算新策略
                new_probs = self.actor(batch_states)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 策略损失
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                values = self.critic(batch_states).squeeze()
                value_loss = 0.5 * (values - batch_returns).pow(2).mean()
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()
        
        self.memory.clear()

    def _compute_gae(self, rewards, values, dones):
        returns = []
        advantages = []
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[t])
            next_value = values[t]
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
        
        # 归一化优势
        advantages = (np.array(advantages) - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return returns, advantages