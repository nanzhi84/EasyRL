import numpy as np
import torch
import random
from .ddqn import DDQN

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
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])

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

class PER_DDQN(DDQN):
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
                 alpha=0.6,
                 beta=0.4,
                 beta_increment=0.001,
                 epsilon=1e-5):
        
        super(PER_DDQN, self).__init__(
            feature_dim, action_dim, learning_rate, gamma, 
            epsilon_start, epsilon_end, epsilon_decay, 
            memory_size, batch_size, replace_target_iter
        )
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.tree = SumTree(memory_size)
        self.max_priority = 1.0  # 初始优先级

    def store_transition(self, s, a, r, s_):
        transition = (s, a, r, s_)
        self.tree.add(self.max_priority, transition)

    def _sample(self):
        batch_idx = []
        batch_weights = []
        batch_transitions = []
        segment = self.tree.total() / self.batch_size

        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
