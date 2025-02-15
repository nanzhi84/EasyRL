import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import gymnasium as gym
from easyRL.algorithms import PPO
import torch

# ... existing imports ...

# 初始化环境（训练时不需要render_mode）
env = gym.make('BipedalWalker-v3')
feature_dim = env.observation_space.shape[0]
action_dim = env.observation_space.shape[0]  # 实际应为env.action_space.shape[0]，但原代码有误需要修正

# 创建PPO代理（注意设置is_continuous=True）
agent = PPO(
    feature_dim=feature_dim,
    action_dim=4,  # BipedalWalker的动作空间是4维连续
    learning_rate=3e-4,
    gamma=0.99,
    clip_epsilon=0.2,
    is_continuous=True  # 关键参数，启用连续动作网络
)

# 训练参数
total_episodes = 2000
print_interval = 20
save_interval = 100

for ep in range(total_episodes):
    state, _ = env.reset()
    ep_reward = 0
    steps = 0
    
    while True:
        # 选择动作
        action, log_prob, value = agent.choose_action(state)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 存储转移
        agent.store_transition(state, action, log_prob, value, reward, done)
        
        state = next_state
        ep_reward += reward
        steps += 1
        
        # 学习时机
        if done:
            agent.learn(done)
            break
    
    # 输出训练信息
    if (ep+1) % print_interval == 0:
        print(f"Episode {ep+1}, Reward: {ep_reward:.1f}, Steps: {steps}")
    
    # 定期保存模型
    if (ep+1) % save_interval == 0:
        torch.save(agent.actor.state_dict(), f"bipedalwalker_actor_{ep+1}.pth")
        torch.save(agent.critic.state_dict(), f"bipedalwalker_critic_{ep+1}.pth")

env.close()