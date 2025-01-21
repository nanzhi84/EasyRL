import sys
import os

# 添加 easyRL 的父目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import gym
import matplotlib.pyplot as plt
from easyRL.algorithms.dqn import DQN

# 配置参数
class Config:
    algorithm = 'dqn'
    learning_rate = 0.01
    gamma = 0.99
    train_eps = 2000
    epsilon_start = 0.9, 
    epsilon_end = 0.1, 
    epsilon_decay = 0.995,
    memory_size = 1000, 
    batch_size = 64, 
    replace_target_iter = 300

# 代理封装类
class AgentWrapper:
    def __init__(self, config, state_dim, action_dim):
        self.config = config
        if config.algorithm.lower() == 'dqn':
            self.agent = DQN(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                epsilon_start=config.epsilon_start,
                epsilon_end=config.epsilon_end,
                epsilon_decay=config.epsilon_decay,
                memory_size=config.memory_size,
                batch_size=config.batch_size,
                replace_target_iter=config.replace_target_iter
            )
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    def choose_action(self, state):
        return self.agent.choose_action(state)

    def update(self, *args, **kwargs):
        self.agent.update(*args, **kwargs)


# 训练函数
def train_agent(config):
    # 初始化环境
    env = gym.make('CartPole-v0') 

    # 初始化代理
    agent_wrapper = AgentWrapper(
        config=config,
        feature_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    rewards = []  # 记录总的rewards
    ma_rewards = []  # 记录总的经滑动平均处理后的rewards

    for i_ep in range(config.train_eps):
        ep_reward = 0
        state = env.reset()
        action = agent_wrapper.choose_action(state)

        while True:
            next_state, reward, done, _ = env.step(action)
            next_action = agent_wrapper.choose_action(next_state)
            
            if config.algorithm.lower() == 'sarsa':
                agent_wrapper.update(state, action, reward, next_state, next_action, done)
            elif config.algorithm.lower() == 'qlearning':
                agent_wrapper.update(state, action, reward, next_state, done)
            
            state = next_state
            action = next_action
            ep_reward += reward
            if done:
                break

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"Episode:{i_ep + 1}/{config.train_eps}: reward:{ep_reward:.1f}")

    env.close()
    return rewards, ma_rewards




