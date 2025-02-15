import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class ActorNetwork(nn.Module):
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
                 learning_rate=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 ppo_epochs=4,
                 batch_size=64,
                 is_continuous=False):
        
        self.is_continuous = is_continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(feature_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(feature_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=learning_rate)
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def choose_action(self, state, greedy=False):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            action_probs = dist
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()

            value = self.critic(state_tensor)
        return action, log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def learn(self, done):
        if not done:
            return
        
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)

        # 计算GAE和returns
        returns, advantages = self._compute_gae(values, rewards, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # 多轮PPO更新
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 计算新策略的log prob和熵
                action_probs = self.actor(batch_states)
                if self.is_continuous:
                    dist = action_probs
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                else:
                    dist = Categorical(action_probs)
                    new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # 策略损失
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                values_pred = self.critic(batch_states).squeeze()
                value_loss = 0.5 * (values_pred - batch_returns).pow(2).mean()

                # 总损失
                loss = policy_loss + value_loss - 0.01 * entropy

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

        # 清空当前episode数据
        self._clear_memory()

    def _compute_gae(self, values, rewards, dones):
        """计算GAE，基于当前critic网络估计价值"""
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        with torch.no_grad():
            new_values = self.critic(states_tensor).squeeze().cpu().numpy()
        
        gae = 0
        returns = []
        advantages = []
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0  # 终止状态的后续价值为0
            else:
                next_value = new_values[t+1] * (1 - dones[t])
            
            delta = rewards[t] + self.gamma * next_value - new_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + new_values[t])
            
        return np.array(returns), np.array(advantages)

    def _clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []