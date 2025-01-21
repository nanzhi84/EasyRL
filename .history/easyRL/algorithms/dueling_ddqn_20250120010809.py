import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from easyRL.algorithms.ddqn import DDQN

class DuelingNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(DuelingNetwork, self).__init__()
        
        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q

class DuelingDDQN(DD):
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
        super(DuelingDDQN, self).__init__()

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
        
        self.memory_counter = 0
        self.memory_size = memory_size
        self.memory = pd.DataFrame(np.zeros((self.memory_size, 
                                           self.feature_dim * 2 + 2)))

        self.eval_net = DuelingNetwork(feature_dim=self.feature_dim,
                                     action_dim=self.action_dim)
        self.target_net = DuelingNetwork(feature_dim=self.feature_dim,
                                       action_dim=self.action_dim)
        
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                        lr=self.lr)

    def _replace_target_params(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
    
    def choose_action(self, state):
        # epsilon_greedy
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            actions_value = self.eval_net(state)
            action = torch.max(actions_value, 1)[1].detach().cpu().numpy()
            return action[0]
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition
        self.memory_counter += 1

    def learn(self, done):
        if self.memory_counter < self.batch_size:
            return
        
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget params replaced\n')
        
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            batch_memory = self.memory.sample(n=self.batch_size)
        else:
            batch_memory = self.memory.iloc[:self.memory_counter].sample(n=self.batch_size, replace=True)

        batch_state = torch.FloatTensor(batch_memory.iloc[:, :self.feature_dim].values)
        batch_action = torch.LongTensor(batch_memory.iloc[:, self.feature_dim].values.astype(int)).unsqueeze(1)
        batch_reward = torch.FloatTensor(batch_memory.iloc[:, self.feature_dim+1].values)
        batch_next_state = torch.FloatTensor(batch_memory.iloc[:, -self.feature_dim:].values)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        
        q_eval_next = self.eval_net(batch_next_state).detach()
        eval_act_next = q_eval_next.max(dim=1)[1].unsqueeze(1)
        
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + (1 - done) * self.gamma * q_next.gather(1, eval_act_next).squeeze()

        # train eval network
        loss = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()