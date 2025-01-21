import gym
import numpy as np
import matplotlib.pyplot as plt
from easyRL.algorithms.Sarsa import Sarsa
from easyRL.algorithms.QLearning import QLearning

env = gym.make('CartPole-v0') 
env.seed(1) # 设置env随机种子
n_states = env.observation_space.shape[0] # 获取总的状态数
n_actions = env.action_space.n # 获取总的动作数
