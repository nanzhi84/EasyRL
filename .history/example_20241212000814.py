import gym
import numpy as np
import matplotlib.pyplot as plt
from easyRL.algorithms.Sarsa import Sarsa
from easyRL.algorithms.QLearning import QLearning

# 配置参数
class Config:
    algorithm = 'QLe'
    policy_lr = 0.1
    gamma = 0.99
    train_eps = 500
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
                learning_rate=config.policy_lr,
                gamma=config.gamma,
                epsilon_start=config.epsilon_start,
                epsilon_end=config.epsilon_end,
                epsilon_decay=config.epsilon_decay,
            )
        elif config.algorithm.lower() == 'qlearning':
            self.agent = QLearning(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=config.policy_lr,
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
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

    # 初始化代理
    agent_wrapper = AgentWrapper(
        config=config,
        state_dim=env.observation_space.n,
        action_dim=env.action_space.n
    )

    rewards = []
    ma_rewards = []  # 移动平均奖励

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