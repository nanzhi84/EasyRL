import sys
import os

# Add the parent directory of easyRL to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import gym
import matplotlib.pyplot as plt
from easyRL.algorithms import DQN, DDQN, DuelingDDQN, REINFORCE

# Configuration parameters
class Config:
    algorithm = 'ddqn'
    learning_rate = 0.0001
    gamma = 0.99
    train_eps = 300
    max_steps = 1000
    epsilon_start = 0.9
    epsilon_end = 0.1
    epsilon_decay = 0.99
    memory_size = 10000
    batch_size = 32
    replace_target_iter = 200

# Agent wrapper class
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
            self.agent = DuelingDDQN(
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
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    def choose_action(self, state):
        return self.agent.choose_action(state)
    
    def store_transition(self, *args, **kwargs):
        self.agent.store_transition(*args, **kwargs)

    def learn(self, done):
        self.agent.learn(done)

# Training function
def train_agent(config):
    # Initialize environment
    env = gym.make('CartPole-v0') 

    # Initialize agent
    agent = AgentWrapper(
        config=config,
        feature_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    rewards = []  # Record total rewards
    ma_rewards = []  # Record total rewards after moving average

    for i_episode in range(config.train_eps): 
        state = env.reset()
        ep_reward = 0
        for i_step in range(config.max_steps):
            action = agent.choose_action(state) 
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.store_transition(state, action, reward, next_state)
            state = next_state
            agent.learn(done)
            if done:
                break
        
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        
        print(f"Episode:{i_episode + 1}/{config.train_eps}: reward:{ep_reward:.1f}")

    return rewards, ma_rewards

# Plotting function
def plot_rewards(rewards, ma_rewards, config):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode', color='blue', alpha=0.6)
    plt.plot(ma_rewards, label='Moving Average (0.1)', color='red', alpha=0.9)
    plt.title(f"{config.algorithm} on CartPole-v0")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    cfg = Config()
    rewards, ma_rewards = train_agent(cfg)
    plot_rewards(rewards, ma_rewards, cfg)