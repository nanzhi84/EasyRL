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
    train_eps = 200
    max_steps = 200
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
    agent = AgentWrapper(
        config=config,
        feature_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    rewards = []  # 记录总的rewards
    ma_rewards = []  # 记录总的经滑动平均处理后的rewards

    ep_steps = []
    for i_episode in range(config.train_eps): 
        state = env.reset()
        ep_reward = 0
        for i_step in range(config.max_steps):
            action = agent.select_action(state) 
            next_state, reward, done, _ = env.step(action) # 更新环境参数
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state # 跳转到下一个状态
            agent.update() # 每步更新网络
            if done:
                break
        # 更新target network，复制DQN中的所有weights and biases
        if i_episode % cfg.target_update == 0: #  cfg.target_update为target_net的更新频率
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:', i_episode, ' Reward: %i' %
            int(ep_reward), 'n_steps:', i_step, 'done: ', done,' Explore: %.2f' % agent.epsilon)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
    return rewards, ma_rewards




