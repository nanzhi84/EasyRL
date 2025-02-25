import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
from easyRL.algorithms_continuous import ActorCritic, PPO, DDPG

@dataclass
class Config:
    algorithm: str = 'ppo'              
    learning_rate: float = 0.001        # 学习率
    gamma: float = 0.99                 # 折扣因子
    train_eps: int = 500                # 训练回合数
    max_steps: int = 200                # 每个回合最大步数
    memory_size: int = 10000
    replace_target_iter: int = 100
    batch_size: int = 32                # 训练批大小
    clip_epsilon: float = 0.2           # PPO算法中用于限制策略更新幅度的裁剪参数
    ppo_epochs: int = 4               # PPO算法中每次更新时进行梯度更新的轮数
    gae_lambda: float = 0.95            # 广义优势估计(GAE)中的λ参数

class AgentWrapper:
    def __init__(self, config, feature_dim, action_dim):
        self.config = config
        if config.algorithm.lower() == 'actor-critic':
            self.agent = ActorCritic(
                feature_dim=feature_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
            )
        elif config.algorithm.lower() == 'ppo':
            self.agent = PPO(
                feature_dim=feature_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_epsilon=config.clip_epsilon,
                ppo_epochs=config.ppo_epochs,
                batch_size=config.batch_size
            )
        elif config.algorithm.lower() == 'ddpg':
            self.agent = DDPG(
                feature_dim=feature_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                memory_size=config.memory_size,
                batch_size=config.batch_size,
                replace_target_iter=config.replace_target_iter
            )
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    
    def choose_action(self, state):
        return self.agent.choose_action(state)
    
    def store_transition(self, *args):
        self.agent.store_transition(*args)

    def learn(self, done, step_counter):
        self.agent.learn(done, step_counter)
    
    def save(self, path):
        torch.save({
            'actor': self.agent.policy_net.state_dict(),
            'critic': self.agent.value_net.state_dict() 
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.policy_net.load_state_dict(checkpoint['actor'])
        self.agent.value_net.load_state_dict(checkpoint['critic'])


def train_agent(config):
    env = gym.make('HalfCheetah-v5')
    best_reward = -float('inf')
    save_path = 'best_model.pth'
    
    agent = AgentWrapper(
        config=config,
        feature_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )

    rewards = []  # 记录每个episode的总奖励
    ma_rewards = []  # 记录经过移动平均处理后的总奖励

    for i_episode in range(config.train_eps): 
        state, _ = env.reset(seed=1)
        ep_reward = 0
        step_counter = 0
        for i_step in range(config.max_steps):
            if config.algorithm.lower() == 'ppo':
                action, raw_action, log_prob, value = agent.choose_action(state)
                next_state, reward, done, truncated, info = env.step(action)

                agent.store_transition(state, action, raw_action, log_prob, value, reward, done)
            else:
                action, log_prob = agent.choose_action(state)
                next_state, reward, done, truncated, info = env.step(action)

                agent.store_transition(state, action, log_prob, reward, next_state, done)
            
            ep_reward += reward

            agent.learn(done, step_counter)

            state = next_state
            step_counter += 1

            if done:
                break

        ma_rewards.append(ma_rewards[-1]*0.9 + ep_reward*0.1 if ma_rewards else ep_reward)
        rewards.append(ep_reward)

        # 更新最佳模型
        if rewards[-1] > best_reward:
            best_reward = rewards[-1]
            agent.save(save_path)
            print(f"New best model saved with reward: {best_reward:.1f}")
        
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

def test_agent(config, model_path, test_eps=1):
    env = gym.make('HalfCheetah-v5', render_mode='human', disable_env_checker=True,)
    agent = AgentWrapper(
        config=config,
        feature_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    agent.load(model_path)
    
    total_rewards = []
    for ep in range(test_eps):
        state, _ = env.reset()
        ep_reward = 0
        while True:
            with torch.no_grad():
                if config.algorithm.lower() == 'ppo':
                    action, _, _, _ = agent.choose_action(state)
                else:
                    action, _ = agent.choose_action(state)
            
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            ep_reward += reward
            env.render()
            
            if done or truncated:
                break
                
        total_rewards.append(ep_reward)
        print(f"Test Episode:{ep+1}, Reward:{ep_reward:.1f}")
    
    env.close()
    print(f"Average test reward over {test_eps} episodes: {sum(total_rewards)/len(total_rewards):.1f}")

if __name__ == "__main__":
    cfg = Config()
    # agent, rewards, ma_rewards = train_agent(cfg)
    # plot_rewards(rewards, ma_rewards, cfg)
    test_agent(cfg, '../best_model.pth')