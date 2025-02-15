import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
import time
import gymnasium as gym
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
from easyRL.algorithms import PPO

@dataclass
class Config:
    algorithm: str = 'ppo'
    learning_rate: float = 1e-3        # 增大学习率
    gamma: float = 0.99
    train_eps: int = 500               # 总训练回合数
    max_steps: int = 200               # 单回合最大步数
    clip_epsilon: float = 0.1          # 减小clip范围
    ppo_epochs: int = 10               # 增加PPO更新轮次
    batch_size: int = 256               # 增大batch size
    gae_lambda: float = 0.95
    save_interval: int = 50            # 新增保存间隔

class AgentWrapper:
    def __init__(self, config, feature_dim, action_dim):
        self.config = config
        if config.algorithm.lower() == 'ppo':
            self.agent = PPO(
                feature_dim=feature_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                batch_size=config.batch_size,
                clip_epsilon=config.clip_epsilon,
                ppo_epochs=config.ppo_epochs,
                gae_lambda=config.gae_lambda,
                is_continuous=True     # 新增连续动作标志
            )
        else:
            raise ValueError("Only PPO supported for continuous control")

    def choose_action(self, state):
        return self.agent.choose_action(state)
    
    def store_transition(self, *args):
        self.agent.store_transition(*args)

    def learn(self, done):
        self.agent.learn(done)

def train_agent(config):
    env = gym.make('MountainCarContinuous-v0')
    agent = AgentWrapper(
        config=config,
        feature_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )

    rewards = []
    ma_rewards = []
    
    for i_episode in range(config.train_eps):
        state, _ = env.reset()
        ep_reward = 0
        steps = 0
        
        while True:
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, log_prob, value, reward, done)
            
            ep_reward += reward
            steps += 1
            
            if done:
                agent.learn(done)
                break
        
        # 记录奖励
        ma_rewards.append(ma_rewards[-1]*0.9 + ep_reward*0.1 if ma_rewards else ep_reward)
        rewards.append(ep_reward)
        
        # 输出训练信息
        if (i_episode+1) % 10 == 0:
            print(f"Episode:{i_episode + 1:03d}/{config.train_eps}, "
                  f"Reward:{ep_reward:.1f}, "
                  f"MA Reward:{ma_rewards[-1]:.1f}, "
                  f"Steps:{steps}")
        
        # 定期保存模型
        if (i_episode+1) % config.save_interval == 0:
            torch.save(agent.agent.actor.state_dict(), 
                      f"mountaincar_actor_{i_episode+1}.pth")
            torch.save(agent.agent.critic.state_dict(),
                      f"mountaincar_critic_{i_episode+1}.pth")

    env.close()
    return agent, rewards, ma_rewards

def plot_rewards(rewards, ma_rewards, config):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode', color='blue', alpha=0.6)
    plt.plot(ma_rewards, label='Moving Average (0.1)', color='red', alpha=0.9)
    plt.title(f"{config.algorithm} on CartPole-v1")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_agent(agent, env_name='MountainCarContinuous-v0', episodes=10, render=True):
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
                action = agent.choose_action(state, greedy=True)
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                print(f'Episode {episode + 1}: Reward = {episode_reward}')
                total_rewards.append(episode_reward)
                break
                
    env.close()
    
    print(f'Average Reward: {sum(total_rewards)/len(total_rewards):.2f}')
    print(f'Max Reward: {max(total_rewards)}')
    print(f'Min Reward: {min(total_rewards)}')

if __name__ == "__main__":
    cfg = Config()
    agent, rewards, ma_rewards = train_agent(cfg)
    plot_rewards(rewards, ma_rewards, cfg)
    test_agent(agent, env_name='MountainCarContinuous-v0', episodes=, render=True)