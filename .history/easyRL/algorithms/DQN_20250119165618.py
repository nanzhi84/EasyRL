import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class Network(nn.Module):
    def __init__(self, 
                 feature_dim, action_dim, n_neuron=10):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=feature_dim, 
                      out_features=n_neuron, 
                      bias=True),
            nn.Linear(in_features=n_neuron, 
                      out_features=action_dim, 
                      bias=True),
            nn.ReLU()
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

        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        
        self.memory_counter = 0
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, self.feature_dim * 2 + 2))

        self.eval_net = Network(feature_dim=self.feature_dim,
                                action_dim=self.action_dim)
        self.target_net = Network(feature_dim=self.feature_dim, 
                                  action_dim=self.action_dim)
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            actions_value = self.eval_net.forward(state)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
            return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.feature_dim])
        batch_action = torch.LongTensor(batch_memory[:, self.feature_dim:self.feature_dim+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.feature_dim+1:self.feature_dim+2])
        batch_next_state