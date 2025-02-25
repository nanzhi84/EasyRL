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
    epsilon_start: float = 1.0          # ε-greedy初始探索率
    epsilon_end: float = 0.01           # ε-greedy最小探索率
    epsilon_decay: float = 0.998        # ε衰减率
    memory_size: int = 10000            # 经验回放缓冲区大小
    batch_size: int = 32                # 训练批大小
    replace_target_iter: int = 100      # 目标网络更新间隔（步数）
    alpha: float = 0.6                  # PER优先级系数（0-1，0表示均匀采样）
    beta: float = 0.4                   # PER重要性采样权重系数
    beta_increment: float = 0.001       # PER beta的增量系数
    clip_epsilon: float = 0.2           # PPO算法中用于限制策略更新幅度的裁剪参数
    ppo_epochs: int = 4                 # PPO算法中每次更新时进行梯度更新的轮数
    gae_lambda: float = 0.95            # 广义优势估计(GAE)中的λ参数