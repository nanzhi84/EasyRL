import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # 演员网络（策略）
        self.actor = nn.Linear(64, action_dim)
        # 网络
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

class PPO:
    def __init__(self, 
                 feature_dim, 
                 action_dim,
                 learning_rate=0.001,
                 gamma=0.99,
                 clip_epsilon=0.2,
                 entropy_coef=0.01,
                 ppo_epochs=4,
                 batch_size=64):
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # 初始化网络
        self.actor_critic = ActorCritic(feature_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def choose_action(self, state, greedy=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.actor_critic(state)
        probs = nn.Softmax(dim=-1)(logits)
        m = Categorical(probs)
        action = m.sample()
        
        # 存储数据
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(m.log_prob(action))
        self.values.append(value.squeeze())
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        # 存储奖励和终止标志
        self.rewards.append(reward)
        self.dones.append(done)
    
    def learn(self, done):
        if not done:
            return
        
        # 计算GAE和回报
        returns = self._compute_returns()
        advantages = self._compute_advantages(returns)
        
        # 转换数据为张量
        states = torch.cat(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 多次PPO更新
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            # 小批量更新
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                # 获取小批量数据
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # 计算新策略的概率
                logits, values = self.actor_critic(batch_states)
                probs = nn.Softmax(dim=-1)(logits)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算比率和裁剪损失
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                value_loss = 0.5 * (values.squeeze() - batch_returns).pow(2).mean()
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # 清空缓冲区
        self._clear_buffer()
    
    def _compute_returns(self):
        # 计算折扣回报
        returns = []
        R = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns
    
    def _compute_advantages(self, returns):
        # 计算GAE优势函数
        values = torch.FloatTensor([v.item() for v in self.values])
        deltas = torch.FloatTensor(returns) - values
        advantages = []
        advantage = 0
        for delta in reversed(deltas):
            advantage = delta + self.gamma * advantage
            advantages.insert(0, advantage)
        return (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    def _clear_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
