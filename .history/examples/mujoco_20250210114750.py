import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import gymnasium as gym
from easyRL.algorithms import PPO

def train_halfcheetah():
    # 环境参数
    env = gym.make('HalfCheetah-v')
    feature_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # PPO配置（针对连续控制优化）
    config = {
        'feature_dim': feature_dim,
        'action_dim': action_dim,
        'is_continuous': True,
        'lr': 3e-4,
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'ppo_epochs': 10,          # 增加更新次数
        'batch_size': 256,         # 增大批大小
        'memory_size': 100000,     # 更大的经验池
        'log_std_init': -0.5,      # 初始探索幅度
        'max_episodes': 5000,
        'eval_interval': 50,
        'target_reward': 3000      # 目标平均回报
    }
    
    agent = PPO(**config)
    total_steps = 0
    best_eval = -np.inf
    
    for episode in range(config['max_episodes']):
        state = env.reset()
        ep_reward = 0
        ep_length = 0
        
        while True:
            # 收集轨迹数据
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 存储转换数据
            agent.store_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=done
            )
            
            state = next_state
            ep_reward += reward
            ep_length += 1
            total_steps += 1
            
            # 定期更新
            if total_steps % 2048 == 0:  # 每收集2048步更新
                agent.learn()
            
            if done:
                break
        
        # 定期评估
        if episode % config['eval_interval'] == 0:
            eval_reward = evaluate(agent, env)
            if eval_reward > best_eval:
                best_eval = eval_reward
                agent.save_model('halfcheetah_ppo_best.pth')
            
            print(f"Ep {episode:4d} | "
                  f"Steps {total_steps:6d} | "
                  f"Train Reward {ep_reward:7.1f} | "
                  f"Eval Reward {eval_reward:7.1f} | "
                  f"Best {best_eval:7.1f}")
            
            if eval_reward >= config['target_reward']:
                print(f"Solved at episode {episode}!")
                break

def evaluate(agent, env, n_episodes=3):
    total_reward = 0
    for _ in range(n_episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            action, _, _ = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        total_reward += ep_reward
    return total_reward / n_episodes

if __name__ == "__main__":
    train_halfcheetah()