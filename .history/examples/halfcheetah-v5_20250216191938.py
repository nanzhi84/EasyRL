import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
from easyRL.algorithms_continuous import ActorCritic

@dataclass
class Config:
    algorithm: str = 'ppo'              # 选择使用的算法
    learning_rate: float = 0.001        # 学习率
    gamma: float = 0.99                 # 折扣因子
    train_eps: int = 300                # 训练回合数
    max_steps: int = 500                # 每个回合最大步数
    memory_size: int = 10000            # 经验回放缓冲区大小
    batch_size: int = 32                # 训练批大小
    clip_epsilon: float = 0.2           # PPO算法中用于限制策略更新幅度的裁剪参数
    ppo_epochs: int = 4                 # PPO算法中每次更新时进行梯度更新的轮数
    gae_lambda: float = 0.95            # 广义优势估计(GAE)中的λ参数

class AgentWrapper:
    def __init__(self, config, feature_dim, action_dim):
        self.config = config
        if config.alg