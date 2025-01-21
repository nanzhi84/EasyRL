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
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for more stable training

    def _replace_target_params(self):
        self.memory_counter = 0
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_net = Network(feature_dim=self.feature_dim,
                                action_dim=self.action_dim).to(self.device)
        self.target_net = Network(feature_dim=self.feature_dim, 
                                  action_dim=self.action_dim).to(self.device)
        
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), 
                                          lr=self.lr)

    def _replace_target_params(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
    
    @torch.no_grad()
    def choose_action(self, state, greedy=False):
        # epsilon_greedy
        if np.random.uniform(0, 1) < self.epsilon and not greedy:
            return np.random.choice(self.action_dim)
        else:
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            actions_value = self.eval_net(state)
            action = torch.max(actions_value, 1)[1].detach().cpu().numpy()
            return action[0]
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        # Use exponential decay
        # self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1. * self.memory_counter / self.epsilon_decay)

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        self.memory.append(transition)
        self.memory_counter += 1

    def learn(self, done):
        if self.memory_counter < self.batch_size:
            return
        
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