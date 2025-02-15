import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


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

class PPOAgent:
    def __init__(self, 
                 feature_dim,
                 action_dim,
                 lr=3e-4,
                 gamma=0.99,
                 clip_epsilon=0.2,
                 ppo_epochs=4,
                 batch_size=64,
                 memory_size=10000):
        
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
        
        # 经验池
        self.memory = deque(maxlen=memory_size)
        self.trajectory = []

    def store_transition(self, state, action, log_prob, value, reward, done):
        """存储单步转移数据"""
        self.trajectory.append((
            state,
            action,
            log_prob,
            value,
            reward,
            done
        ))
        if done:
            self._process_trajectory()
    
    def _process_trajectory(self):
        """处理完整轨迹并存入经验池"""
        states, actions, log_probs, values, rewards, dones = zip(*self.trajectory)
        
        # 计算GAE和returns
        returns, advantages = self._compute_gae(values, rewards, dones)
        
        # 存入经验池
        for s, a, lp, ret, adv in zip(states, actions, log_probs, returns, advantages):
            self.memory.append( (s, a, lp, ret, adv) )
        
        self.trajectory.clear()

    def _compute_gae(self, values, rewards, dones):
        """计算广义优势估计"""
        # ... 原有GAE计算逻辑保持不变 ...
        return returns, advantages

    def learn(self):
        """训练入口，类似DQN的learn方法"""
        if len(self.memory) < self.batch_size:
            return
        
        # 转换经验数据
        states, actions, old_log_probs, returns, advantages = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)

        # PPO多轮更新
        for _ in range(self.ppo_epochs):
            # ... 原有PPO更新逻辑保持不变 ...
            
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state_val = self.critic(state)
        
        self.trajectory.append((
            state,
            action,
            action_logprob,
            state_val,
            None,
            False
        ))
        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.state_values, dim=0)).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.ppo_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))