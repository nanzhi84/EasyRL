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
