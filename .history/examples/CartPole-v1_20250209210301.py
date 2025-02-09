import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
from easyRL.algorithms import DQN, DDQN, DUELING_DDQN, REINFORCE, PER_DDQN

@dataclass
class Config:
    algorithm: str = 'reinforce' 
    learning_rate: float = 0.001
    gamma: float = 0.99
    train_eps: int = 300
    max_steps: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01  
    epsilon_decay: float = 0.998
    memory_size: int = 10000
    batch_size: int = 32
    replace_target_iter: int = 100
    alpha: float = 0.6
    beta: float = 0.4
    beta_increment: float = 0.001
    clip_epsilon: float = 0.2      # PPO裁剪系数
    entropy_coef: float = 0.01     # 熵系数
    ppo_epochs: int = 4            # PPO更新轮数
    batch_size: int = 64        

class AgentWrapper:
    def __init__(self, config, feature_dim, action_dim):
        self.config = config
        if config.algorithm.lower() == 'reinforce':
            self.agent = REINFORCE(
                feature_dim=feature_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma
            )
        elif config.algorithm.lower() == 'dqn':
            self.agent = DQN(
                feature_dim=feature_dim,
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
        elif config.algorithm.lower() == 'ddqn':
            self.agent = DDQN(
                feature_dim=feature_dim,
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
        elif config.algorithm.lower() == 'dueling_ddqn':
            self.agent = DUELING_DDQN(
                feature_dim=feature_dim,
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
        elif config.algorithm.lower() == 'per_ddqn':
            self.agent = PER_DDQN(
                feature_dim=feature_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                epsilon_start=config.epsilon_start,
                epsilon_end=config.epsilon_end,
                epsilon_decay=config.epsilon_decay,
                memory_size=config.memory_size,
                batch_size=config.batch_size,
                replace_target_iter=config.replace_target_iter,
                alpha=config.alpha,
                beta=config.beta,
                beta_increment=config.beta_increment
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
    env = gym.make('CartPole-v1') 

    agent = AgentWrapper(
        config=config,
        feature_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    rewards = []  # 记录每个episode的总奖励
    ma_rewards = []  # 记录经过移动平均处理后的总奖励

    for i_episode in range(config.train_eps): 
        state = env.reset(seed=1)
        ep_reward = 0
        step_counter = 0
        for i_step in range(config.max_steps):
            action = agent.choose_action(state) 
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            ep_reward += reward

            if step_counter % 1 == 0:
                agent.learn(done)

            state = next_state
            step_counter += 1

            if done:
                break

        ma_rewards.append(ma_rewards[-1]*0.9 + ep_reward*0.1 if ma_rewards else ep_reward)
        rewards.append(ep_reward)
        
        if (i_episode + 1) % 10 == 0:
            print(f"Episode:{i_episode + 1:03d}/{config.train_eps}, "
                  f"Reward:{ep_reward:.1f}, "
                  f"MA Reward:{ma_rewards[-1]:.1f}")

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

def test_agent(agent, env_name='CartPole-v1', episodes=10, render=True):
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
    # test_agent(agent, episodes=1, render=True)