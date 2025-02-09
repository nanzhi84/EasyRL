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
        super(PER_DDQN, self).__init__(
            feature_dim, action_dim, learning_rate, gamma, 
            epsilon_start, epsilon_end, epsilon_decay, 
            memory_size, batch_size, replace_target_iter)
        
        # PER参数
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.absolute_error_upper = 1.0  # 限制优先级
        
        # 用SumTree代替普通记忆库
        self.memory = SumTree(memory_size)
        self.max_p = 1.0  # 初始最大优先级
        
    def store_transition(self, s, a, r, s_):
        transition = (s, a, r, s_)
        self.memory.add(self.max_p, transition)  # 添加新经验并设置最大优先级
        
    def _sample(self):
        batch_idx = []
        batch_memory = []
        ISWeights = np.empty((self.batch_size, 1))
        
        pri_segment = self.memory.total() / self.batch_size
        
        self.beta = np.min([1., self.beta + self.beta_increment])
        
        min_p = np.min(self.memory.tree[-self.memory.capacity:]) / self.memory.total()
        max_weight = (min_p * self.batch_size) ** (-self.beta)
        
        for i in range(self.batch_size):
            a, b = pri_segment * i, pri_segment * (i + 1)
            value = np.random.uniform(a, b)
            idx, p, data = self.memory.get(value)
            
            prob = p / self.memory.total()
            ISWeights[i, 0] = (self.batch_size * prob) ** (-self.beta) / max_weight
            batch_idx.append(idx)
            batch_memory.append(data)
            
        return batch_idx, batch_memory, ISWeights
    
    def learn(self, done):
        if self.memory.entries < self.batch_size:
            return
            
        # 更新目标网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            
        # 采样并计算重要性权重
        tree_idx, batch_memory, ISWeights = self._sample()
        ISWeights = torch.FloatTensor(ISWeights)
        
        # 转换为张量
        batch_state = torch.FloatTensor(np.array([m[0] for m in batch_memory]))
        batch_action = torch.LongTensor(np.array([m[1] for m in batch_memory])).unsqueeze(1)
