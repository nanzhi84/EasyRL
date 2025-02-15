import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.distributions import Categorical

class ActorNetwork(nn.Module):
    """策略网络，输出动作概率分布"""
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
    """价值网络，评估状态价值"""
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
                 lr=3e-4,
                 gamma=0.99,
                 clip_epsilon=0.2,
                 ppo_epochs=4,
                 batch_size=64,
                 memory_size=10000):
        
        # 超参数设置
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # 网络初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(feature_dim).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=lr)
        
        # 经验池配置
        self.memory = deque(maxlen=memory_size)
        self.trajectory = []
    
    def choose_action(self, state):
        """选择动作并返回动作、对数概率和状态价值"""
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor)
            
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, value, reward, done):
        """存储单步转移数据"""
        self.trajectory.append((
            nstate, dtype=np.float32),
            action,
            log_prob,
            value,
            reward,
            done
        ))
        
        if done:
            self._process_trajectory()

    def _process_trajectory(self):
        """处理完整轨迹并计算GAE和returns"""
        states, actions, log_probs, values, rewards, dones = zip(*self.trajectory)
        
        # 转换为numpy数组
        states = np.array(states)
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones, dtype=np.float32)
        
        # 计算GAE和returns
        returns, advantages = self._compute_gae(values, rewards, dones)
        
        # 标准化优势
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # 存入经验池
        for s, a, lp, ret, adv in zip(states, actions, log_probs, returns, advantages):
            self.memory.append( (s, a, lp, ret, adv) )
        
        self.trajectory.clear()

    def _compute_gae(self, values, rewards, dones):
        """计算广义优势估计(GAE)和returns"""
        gae = 0
        returns = []
        advantages = []
        
        next_value = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[t])
            next_value = values[t]
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
            
        return np.array(returns), np.array(advantages)

    def learn(self):
        """执行PPO更新"""
        if len(self.memory) < self.batch_size:
            return
        
        # 转换经验数据为张量
        states, actions, old_log_probs, returns, advantages = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        
        # 多轮PPO更新
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = np.arange(len(self.memory))
            np.random.shuffle(indices)
            
            # 小批量训练
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 计算新策略的对数概率
                action_probs = self.actor(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算价值函数
                values = self.critic(batch_states).squeeze()
                
                # 计算策略损失
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = 0.5 * (values - batch_returns).pow(2).mean()
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

    def save_model(self, path):
        """保存模型参数"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load_model(self, path):
        """加载模型参数"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])