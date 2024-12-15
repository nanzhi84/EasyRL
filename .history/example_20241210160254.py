import gym
import numpy as np
from easyRL.algorithms import QLearningSarsa

# 配置参数
class Config:
    policy_lr = 0.1
    gamma = 0.99
    train_eps = 500
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995

cfg = Config()

# 初始化环境和代理
env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
# agent = QLearning(
#     state_dim=env.observation_space.n,
#     action_dim=env.action_space.n,
#     learning_rate=cfg.policy_lr,
#     gamma=cfg.gamma,
# )
agent = Sarsa(
    state_dim=env.observation_space.n,
    action_dim=env.action_space.n,
    learning_rate=cfg.policy_lr,
    gamma=cfg.gamma,
    epsilon_start=cfg.epsilon_start,
    epsilon_end=cfg.epsilon_end,
    epsilon_decay=cfg.epsilon_decay,
)

rewards = []
ma_rewards = []  # moving average reward

# for i_ep in range(cfg.train_eps):  # train_eps: 训练的最大episodes数
#     ep_reward = 0  # 记录每个episode的reward
#     state = env.reset()  # 重置环境，开始新的一个episode
#     while True:
#         action = agent.choose_action(state)  # 根据算法选择一个动作
#         next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
#         agent.update(state, action, reward, next_state, done)  # Q-learning算法更新
#         state = next_state  # 存储上一个观察值
#         ep_reward += reward
#         if done:
#             break
#     rewards.append(ep_reward)
#     if ma_rewards:
#         ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
#     else:
#         ma_rewards.append(ep_reward)
#     print(f"Episode:{i_ep + 1}/{cfg.train_eps}: reward:{ep_reward:.1f}")
    

for i_ep in range(cfg.train_eps):
    ep_reward = 0
    state = env.reset()
    action = agent.choose_action(state)

    while True:
        next_state, reward, done, _ = env.step(action)
        next_action = agent.choose_action(next_state)
        agent.update(state, action, reward, next_state, next_action, done)
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
    print(f"Episode:{i_ep + 1}/{cfg.train_eps}: reward:{ep_reward:.1f}")