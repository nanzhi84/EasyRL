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
                 learning_rate=0.01, 
                 gamma=0.9, 
                 epsilon_start=0.9, 
                 epsilon_end=0.1, 
                 epsilon_decay=0.995, 
                 memory_size=500, 
                 batch_size=32, 
                 replace_target_iter=300):
        super(DQN, self).__init__()
        
        # Hyperparameters
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training counters
        self.learn_step_counter = 0
        self.memory_counter = 0
        
        # Experience replay buffer
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self._init_networks()
        
        # Loss and optimizer
        self.loss_function = nn.SmoothL1Loss()  # Changed to Huber loss for better stability
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

    def _init_networks(self):
        """Initialize evaluation and target networks"""
        self.eval_net = Network(self.feature_dim, self.action_dim).to(self.device)
        self.target_net = Network(self.feature_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()  # Set target network to eval mode

    def _replace_target_params(self):
        """Soft update target network parameters"""
        # Soft update instead of hard replacement
        tau = 0.01
        for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)
    
    @torch.no_grad()
    def choose_action(self, state, greedy=False):
        """Epsilon-greedy action selection"""
        if np.random.uniform(0, 1) < self.epsilon and not greedy:
            return np.random.choice(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actions_value = self.eval_net(state)
        return torch.argmax(actions_value).item()
        
    def decay_epsilon(self):
        """Decay epsilon using exponential decay"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action, reward, next_state):
        """Store transition in replay buffer"""
        transition = (state, action, reward, next_state)
        self.memory.append(transition)
        self.memory_counter += 1

    def learn(self, done):
        """Perform one learning step"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget params replaced\n')
        
        # sample batch from memory
        batch_memory = np.array(random.sample(self.memory, self.batch_size))
        
        batch_state = torch.FloatTensor(batch_memory[:, :self.feature_dim])
        batch_action = torch.LongTensor(batch_memory[:, self.feature_dim].astype(int)).unsqueeze(1)
        batch_reward = torch.FloatTensor(batch_memory[:, self.feature_dim+1])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.feature_dim:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action).squeeze()
        
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + (1 - done) * self.gamma * q_next.max(dim=1)[0]

        # train eval network
        loss = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()