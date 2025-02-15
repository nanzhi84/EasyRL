import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
import time
import gymnasium as gym
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class DiscreteActorNetwork(nn.Module):
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

class ContinuousActorNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim, log_std_init=-0.5):
        super().__init__()
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        self.mean_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        action_mean = self.mean_net(x)
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

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
        self.actor = (ContinuousActorNetwork if self.is_continuous else DiscreteActorNetwork)(feature_dim, action_dim).to(self.device)
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
            dist = self.actor(state_tensor)

            if self.is_continuous:
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
                action = action.squeeze(0).cpu().numpy()
            else:
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
        gae = 0
        returns = []
        advantages = []
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0  # 终止状态的后续价值为0
            else:
                next_value = values[t+1] * (1 - dones[t])
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        return np.array(returns), np.array(advantages)

    def _clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

@dataclass
class Config:
    algorithm: str = 'ppo'
    learning_rate: float = 1e-3
    gamma: float = 0.99
    train_eps: int = 800               # 总训练回合数
    max_steps: int = 1000              # 单回合最大步数
    clip_epsilon: float = 0.1          # clip范围
    ppo_epochs: int = 10               # PPO更新轮次
    batch_size: int = 32               # batch size
    gae_lambda: float = 0.95

class AgentWrapper:
    def __init__(self, config, feature_dim, action_dim):
        self.config = config
        if config.algorithm.lower() == 'ppo':
            self.agent = PPO(
                feature_dim=feature_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                batch_size=config.batch_size,
                clip_epsilon=config.clip_epsilon,
                ppo_epochs=config.ppo_epochs,
                gae_lambda=config.gae_lambda,
                is_continuous=True    
            )
        else:
            raise ValueError("Only PPO supported for continuous control")

    def choose_action(self, state, greedy=False):
        return self.agent.choose_action(state)
    
    def store_transition(self, *args):
        self.agent.store_transition(*args)

    def learn(self, done):
        self.agent.learn(done)

def train_agent(config):
    env = gym.make('MountainCarContinuous-v0')
    agent = AgentWrapper(
        config=config,
        feature_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )

    rewards = []
    ma_rewards = []
    
    for i_episode in range(config.train_eps):
        state, _ = env.reset()
        ep_reward = 0
        steps = 0
        
        for i_step in range(config.max_steps):
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, log_prob, value, reward, done)
            
            ep_reward += reward
            steps += 1
            
            if done:
                agent.learn(done)
                break
        
        # 记录奖励
        ma_rewards.append(ma_rewards[-1]*0.9 + ep_reward*0.1 if ma_rewards else ep_reward)
        rewards.append(ep_reward)
        
        # 输出训练信息
        if (i_episode+1) % 10 == 0:
            print(f"Episode:{i_episode + 1:03d}/{config.train_eps}, "
                  f"Reward:{ep_reward:.1f}, "
                  f"MA Reward:{ma_rewards[-1]:.1f}, "
                  f"Steps:{steps}")

    env.close()
    return agent, rewards, ma_rewards

def plot_rewards(rewards, ma_rewards, config):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode', color='blue', alpha=0.6)
    plt.plot(ma_rewards, label='Moving Average (0.1)', color='red', alpha=0.9)
    plt.title(f"{config.algorithm} on MountainCarContinuous-v0")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_agent(agent, env_name='MountainCarContinuous-v0', episodes=10, render=True):
    env = gym.make(env_name)
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            if render:
                env.render()
                time.sleep(0.01)
            
            with torch.no_grad():
                action = agent.choose_action(state, greedy=True)
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                print(f'Episode {episode + 1}: Reward = {episode_reward}')
                total_rewards.append(episode_reward)
                break
                
    env.close()
    
    print(f'Average Reward: {sum(total_rewards)/len(total_rewards):.2f}')
    print(f'Max Reward: {max(total_rewards)}')
    print(f'Min Reward: {min(total_rewards)}')

if __name__ == "__main__":
    cfg = Config()
    agent, rewards, ma_rewards = train_agent(cfg)
    plot_rewards(rewards, ma_rewards, cfg)
    test_agent(agent, env_name='MountainCarContinuous-v0', episodes=1, render=True)