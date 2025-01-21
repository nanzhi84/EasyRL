import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import math

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

class DQN(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 action_dim, 
                 learning_rate=1e-3, 
                 gamma=0.99, 
                 epsilon_start=1.0, 
                 epsilon_end=0.01, 
                 epsilon_decay=0.995, 
                 memory_size=10000, 
                 batch_size=64, 
                 replace_target_iter=1000):
        super(DQN, self).__init__()

        # Hyperparameters
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.learn_step_counter = 0
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.eval_net = Network(feature_dim, action_dim).to(self.device)
        self.target_net = Network(feature_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()  # Target network in eval mode by default
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()  

    def _replace_target_params(self):
        """Soft update of the target network parameters"""
        self.target_net.load_state_dict(self.eval_net.state_dict())

    @torch.no_grad()
    def choose_action(self, state, greedy=False):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon and not greedy:
            return np.random.choice(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.eval_net(state)
        return torch.argmax(q_values).item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def store_transition(self, state, action, reward, next_state):
        """Store experience in replay memory"""
        transition = (state, action, reward, next_state)
        self.memory.append(transition)

    def learn(self, done):
        """Update the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*transitions)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor([1 - done] * self.batch_size).to(self.device)

        # Compute Q values
        q_eval = self.eval_net(states).gather(1, actions).squeeze()
        q_next = self.target_net(next_states).max(1)[0].detach()
        q_target = rewards + dones * self.gamma * q_next

        # Compute loss and update
        loss = self.loss_fn(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1.0)
        
        self.optimizer.step()

        # Update target network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        self.learn_step_counter += 1
        self.decay_epsilon()