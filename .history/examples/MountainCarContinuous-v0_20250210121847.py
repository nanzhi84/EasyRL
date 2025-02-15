import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import gymnasium as gym
from easyRL.algorithms import PPO
import torch

# 初始化环境（修改环境名称）
env = gym.make('MountainCarContinuous-v3')  # 注意实际版本号可能为v0
feature_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 正确获取动作维度（此处应为1）

# 创建PPO代理（调整超参数）
agent = PPO(
    feature_dim=feature_dim,
    action_dim=action_dim,  # MountainCarContinuous的动作空间是1维连续
    learning_rate=1e-3,     # 增大学习率
    gamma=0.99,
    clip_epsilon=0.1,       # 减小clip范围
    ppo_epochs=10,          # 增加PPO更新轮次
    batch_size=256,         # 增大batch size
    is_continuous=True
)

# 调整训练参数（减少总回合数）
total_episodes = 500
print_interval = 10
save_interval = 50

for ep in range(total_episodes):
    state, _ = env.reset()
    ep_reward = 0
    steps = 0
    
    while True:
        # 选择动作（自动处理连续动作）
        action, log_prob, value = agent.choose_action(state)
        
        # 执行动作（动作范围自动裁剪）
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 存储转移（自动处理连续动作的log_prob）
        agent.store_transition(state, action, log_prob, value, reward, done)
        
        state = next_state
        ep_reward += reward
        steps += 1
        
        if done:
            agent.learn(done)
            break
    
    # 输出训练信息
    if (ep+1) % print_interval == 0:
        print(f"Episode {ep+1}, Reward: {ep_reward:.1f}, Steps: {steps}")
    
    # 修改保存文件名
    if (ep+1) % save_interval == 0:
        torch.save(agent.actor.state_dict(), f"mountaincar_actor_{ep+1}.pth")

env.close()