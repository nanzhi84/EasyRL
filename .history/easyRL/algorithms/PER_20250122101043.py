import numpy as np
import torch
import random
from .ddqn import DDQN

class SumTree:
    """SumTree数据结构实现优先级采样"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.entries = 0

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
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, p)

        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0
        if self.entries < self.capacity:
            self.entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PER_DDQN(DDQN):
    """实现带优先级经验回放的Double DQN"""
    def __init__(self, 
                 feature_dim, 
                 action_dim,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=0.9,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 memory_size=10000,
                 batch_size=64,
                 replace_target_iter=300,
                 alpha=0.6,
                 beta=0.4,
                 beta_increment=0.001):
