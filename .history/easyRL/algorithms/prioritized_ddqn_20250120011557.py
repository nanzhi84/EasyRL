import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from easyRL.algorithms.ddqn import DDQN

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
        
    def total(self):
        return self.tree[0]
    
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
        
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedDDQN(DDQN):
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
                 replace_target_iter=300,
                 alpha=0.6,  # priority exponent
                 beta=0.4,   # importance sampling exponent
                 beta_increment=0.001,
                 epsilon_prio=0.01  # small constant to avoid zero priority
                ):
        super(PrioritizedDDQN, self).__init__(feature_dim, action_dim, learning_rate, gamma,
                                             epsilon_start, epsilon_end, epsilon_decay,
                                             memory_size, batch_size, replace_target_iter)
        
        # Priority replay related
        self.memory = SumTree(memory_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon_prio = epsilon_prio
        
    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        # Set maximum priority for new samples
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = 1
        self.memory.add(max_priority, transition)
        self.memory_counter += 1
        
    def learn(self, done):
        if self.memory.n_entries < self.batch_size:
            return
        
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget params replaced\n')
            
        # Beta annealing
        beta = min(1., self.beta + self.beta_increment)
        
        # Sample batch
        batch = []
        idxs = []
        segment = self.memory.total() / self.batch_size
        priorities = []
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.memory.get(s)
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)
            
        batch = np.array(batch)
        
        # Importance sampling weights
        sampling_probabilities = priorities / self.memory.total()
        is_weights = np.power(self.memory.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()
        is_weights = torch.FloatTensor(is_weights)
        
        batch_state = torch.FloatTensor(batch[:, :self.feature_dim])
        batch_action = torch.LongTensor(batch[:, self.feature_dim].astype(int)).unsqueeze(1)
        batch_reward = torch.FloatTensor(batch[:, self.feature_dim+1])
        batch_next_state = torch.FloatTensor(batch[:, -self.feature_dim:])
        
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        
        q_eval_next = self.eval_net(batch_next_state).detach()
        eval_act_next = q_eval_next.max(dim=1)[1].unsqueeze(1)
        
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + (1 - done) * self.gamma * q_next.gather(1, eval_act_next).squeeze()
        
        # Calculate TD-error for priority update
        errors = torch.abs(q_target - q_eval.squeeze()).detach().cpu().numpy()
        
        # Update priorities
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, (errors[i] + self.epsilon_prio) ** self.alpha)
        
        # Calculate weighted loss
        loss = (is_weights * (q_target - q_eval.squeeze()) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learn_step_counter += 1