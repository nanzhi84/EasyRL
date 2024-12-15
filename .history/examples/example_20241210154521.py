import gym
import numpy as np
from easyRL.algorithms.q_learning import QLearning

# 假设您有一个配置对象 cfg，定义了训练参数
class Config:
    policy_lr = 0.1
    gamma = 0.99
    train_eps = 500  # 例如训练500个episodes

cfg = Config()

# 初始化环境和代理
env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
# env = CliffWalkingWapper(env)  # 如果您有自定义的环境包装器
agent = QLearning(
    state_dim=env.observation_space.n,
    action_dim=env.action_space.n,
    learning_rate=cfg.policy_lr,
    gamma=cfg.gamma,
)

rewards = []
ma_rewards = []  # moving average reward

for i_ep in range(cfg.train_eps):  # train_eps: 训练的最大episodes数
    ep_reward = 0  # 记录每个episode的reward
    state = env.reset()  # 重置环境，开始新的一个episode
    while True:
        action = agent.choose_action(state)  # 根据算法选择一个动作
        next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
        agent.update(state, action, reward, next_state, done)  # Q-learning算法更新
        state = next_state  # 存储上一个观察值
        ep_reward += reward
        if done:
            break
    rewards.append(ep_reward)
    if ma_rewards:
        ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
    else:
        ma_rewards.append(ep_reward)
    print(f"Episode:{i_ep + 1}/{cfg.train_eps}: reward:{ep_reward:.1f}")

# 训练完成后，您可以保存 Q 表
agent.save_Q_table("q_table.npy")
