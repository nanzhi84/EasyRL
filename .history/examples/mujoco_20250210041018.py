import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
from easyRL.algorithms import PPO

@dataclass
class Config:
    algorithm: str = 'ppo'             # 使用PPO算法
    env_name: str = 'HalfCheetah-v4'   # MuJoCo环境名称
    learning_rate: float = 3e-4        # 调整学习率
    gamma: float = 0.99                # 折扣因子
    train_eps: int = 1000              # 增加训练回合数
    max_steps: int = 1000              # 增加最大步数
    clip_epsilon: float = 0.2          # PPO裁剪系数
    entropy_coef: float = 0.01         # 熵系数
    ppo_epochs: int = 10               # PPO更新轮数
    batch_size: int = 64               # 增大批大小
    gae_lambda: float = 0.95           # GAE系数
    memory_size: int = 1000            # 经验缓冲区大小

class AgentWrapper:
    def __init__(self, config, feature_dim, action_dim):
        self.config = config
        if  config.algorithm.lower() == 'ppo':
            self.agent = PPO(
                feature_dim=feature_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                clip_epsilon=config.clip_epsilon,
                entropy_coef=config.entropy_coef,
                ppo_epochs=config.ppo_epochs,
                batch_size=config.batch_size
            )
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    def choose_action(self, state, greedy=False):
        return self.agent.choose_action(state, greedy)
    
    def store_transition(self, *args, **kwargs):
        self.agent.store_transition(*args, **kwargs)

    def learn(self, done):
        self.agent.learn(done)

def train_agent(config):
    env = gym.make(config.env_name)
    
    agent = AgentWrapper(
        config=config,
        feature_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]  # 连续动作空间维度
    )

    rewards = []  
    ma_rewards = [] 

    for i_episode in range(config.train_eps): 
        state = env.reset()
        episode_reward = 0
        episode_transitions = []

        for _ in range(config.max_steps):
            action, log_prob, value = agent.agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 存储transition（保持接口一致）
            agent.store_transition(state, action, reward, next_state, done)
            
            # 添加PPO专用数据
            episode_transitions.append((
                state, action, log_prob, value, reward, done
            ))
            
            state = next_state
            episode_reward += reward

            if done:
                break

        # 替换为PPO专用学习逻辑
        agent.agent.memory = episode_transitions
        agent.agent.learn()
        
        # 保持原有进度记录方式
        ma_reward = ma_rewards[-1]*0.9 + episode_reward*0.1 if ma_rewards else episode_reward
        ma_rewards.append(ma_reward)
        rewards.append(episode_reward)
        
        if (i_episode + 1) % 10 == 0:
            print(f"Episode:{i_episode + 1:03d}/{config.train_eps}, "
                  f"Reward:{episode_reward:.1f}, "
                  f"MA Reward:{ma_rewards[-1]:.1f}")

    return agent, rewards, ma_rewards

# 修改测试函数支持连续动作
def test_agent(agent, env_name='HalfCheetah-v4', episodes=1, render=True):
    env = gym.make(env_name)
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            if render:
                env.render()
                time.sleep(0.01)
            
            with torch.no_grad():
                action, _, _ = agent.agent.choose_action(state)  # 适配PPO选择方式
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                print(f'Episode {episode + 1}: Reward = {episode_reward}')
                total_rewards.append(episode_reward)
                break
                
    env.close()
    print(f'Test Average Reward: {sum(total_rewards)/len(total_rewards):.1f}')

if __name__ == "__main__":
    cfg = Config()
    agent, rewards, ma_rewards = train_agent(cfg)
    plot_rewards(rewards, ma_rewards, cfg)
    test_agent(agent, episodes=1, render=True)
