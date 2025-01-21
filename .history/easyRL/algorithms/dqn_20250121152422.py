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
        super().__init__()
        
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
        
        # Training state
        self.learn_step_counter = 0
        self.memory_counter = 0
        
        # Experience replay
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.eval_net = Network(feature_dim, action_dim).to(self.device)
        self.target_net = Network(feature_dim, action_dim).to(self.device)
        self._replace_target_params()  # Initialize target net with eval net weights
        
        # Optimization
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)

    def _replace_target_params(self):
        """Synchronize target network parameters with evaluation network"""
        self.target_net.load_state_dict(self.eval_net.state_dict())

    @torch.no_grad()
    def choose_action(self, state, greedy=False):
        """Epsilon-greedy action selection"""
        if np.random.uniform() < self.epsilon and not greedy:
            return np.random.choice(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.eval_net(state_tensor)
        return q_values.argmax().item()

    def decay_epsilon(self):
        """Exponentially decay exploration rate"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def store_transition(self, state, action, reward, next_state):
        """Store experience in replay memory"""
        transition = (state, action, reward, next_state)
        self.memory.append(transition)
        self.memory_counter += 1

    def learn(self, done):
        """Perform one learning step"""
        if len(self.memory) < self.batch_size:
            return
            
        # Update target network periodically
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        # Compute Q values
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