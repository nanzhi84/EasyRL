import sys
import os

# 添加 easyRL 的父目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import gymnasium as gym
import matplotlib.pyplot as plt
from easyRL.algorithms import Sarsa
from easyRL.algorithms import QLearning

# 配置参数
class Config:
    algorithm = 'sarsa'
    learning_rate = 0.1
    gamma = 0.99
    train_eps = 500
    max_steps = 500
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995

# 代理封装类
class AgentWrapper:
    def __init__(self, config, state_dim, action_dim):
        self.config = config
        if config.algorithm.lower() == 'sarsa':
            self.agent = Sarsa(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                epsilon_start=config.epsilon_start,
                epsilon_end=config.epsilon_end,
                epsilon_decay=config.epsilon_decay,
            )
        elif config.algorithm.lower() == 'qlearning':
            self.agent = QLearning(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                epsilon_start=config.epsilon_start,
                epsilon_end=config.epsilon_end,
                epsilon_decay=config.epsilon_decay,
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
    env = gym.make("CliffWalking-v0")

    # 初始化代理
    agent = AgentWrapper(
        config=config,
        state_dim=env.observation_space.n,
        action_dim=env.action_space.n
    )

    rewards = []  # 记录总的rewards
    ma_rewards = []  # 记录总的经滑动平均处理后的rewards

    for i_episode in range(config.train_eps):
        ep_reward = 0
        state = env.reset(seed=1)
        action = agent.choose_action(state)

        for i_step in range(config.max_steps):
            next_state, reward, done, _, _ = env.step(action)
            print(next_state)
            next_action = agent.choose_action(next_state)
            
            if config.algorithm.lower() == 'sarsa':
                agent.update(state, action, reward, next_state, next_action, done)
            elif config.algorithm.lower() == 'qlearning':
                agent.update(state, action, reward, next_state, done)
            
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
        print(f"Episode:{i_episode + 1}/{config.train_eps}: reward:{ep_reward:.1f}")

    env.close()
    return rewards, ma_rewards

# 绘图函数
def plot_rewards(rewards, ma_rewards, config):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode', color='blue', alpha=0.6)
    plt.plot(ma_rewards, label='Moving Average (0.1)', color='red', alpha=0.9)
    plt.title(f"{config.algorithm} on CliffWalking-v0")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    cfg = Config()
    rewards, ma_rewards = train_agent(cfg)
    plot_rewards(rewards, ma_rewards, cfg)