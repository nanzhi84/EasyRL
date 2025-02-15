import gym
import numpy as np
from easyRL.algorithms.ppo import PPO

class BipedalWalkerPPOConfig:
    def __init__(self):
        # 环境参数
        self.env_name = "BipedalWalker-v3"
        self.feature_dim = 24  # 状态空间维度
        self.action_dim = 4    # 动作空间维度
        
        # PPO超参数（针对BipedalWalker优化）
        self.lr = 3e-4
        self.gamma = 0.99
        self.clip_epsilon = 0.2
        self.ppo_epochs = 8
        self.batch_size = 128
        self.memory_size = 50000
        self.log_std_init = -0.7  # 更大初始探索
        
        # 训练参数
        self.max_episodes = 3000
        self.eval_interval = 20
        self.target_reward = 300  # 环境最大理论奖励约300
        self.max_steps = 1600     # 每个episode最大步数

def train_bipedalwalker():
    config = BipedalWalkerPPOConfig()
    env = gym.make(config.env_name)
    
    # 初始化智能体
    agent = PPOAgent(
        feature_dim=config.feature_dim,
        action_dim=config.action_dim,
        is_continuous=True,
        lr=config.lr,
        gamma=config.gamma,
        clip_epsilon=config.clip_epsilon,
        ppo_epochs=config.ppo_epochs,
        batch_size=config.batch_size,
        memory_size=config.memory_size,
        log_std_init=config.log_std_init
    )
    
    # 训练循环
    best_mean_reward = -float('inf')
    reward_history = []
    
    for episode in range(config.max_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < config.max_steps:
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 奖励调整（关键修改点）
            adjusted_reward = reward * 0.1  # 缩小奖励范围
            
            agent.store_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=adjusted_reward,
                done=done
            )
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # 定期更新策略
            if steps % 512 == 0:
                agent.learn()
            
            if done:
                break
        
        # 记录并评估
        reward_history.append(episode_reward)
        if len(reward_history) > 100:
            reward_history.pop(0)
        mean_reward = np.mean(reward_history[-20:])  # 计算最近20个episode平均
        
        # 定期保存最佳模型
        if episode % config.eval_interval == 0:
            eval_reward = evaluate(agent, env)
            if eval_reward > best_mean_reward:
                best_mean_reward = eval_reward
                agent.save_model("bipedalwalker_best.pth")
            
            print(f"Ep {episode:4d} | "
                  f"Steps {steps:4d} | "
                  f"Reward {episode_reward:6.1f} | "
                  f"Avg20 {mean_reward:6.1f} | "
                  f"Eval {eval_reward:6.1f} | "
                  f"Best {best_mean_reward:6.1f}")
            
            if mean_reward >= config.target_reward:
                print(f"Solved at episode {episode}!")
                break

def evaluate(agent, env, n_episodes=3, render=False):
    total_reward = 0
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            action, _, _ = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if render:
                env.render()
            if done:
                break
        total_reward += episode_reward
    return total_reward / n_episodes

if __name__ == "__main__":
    train_bipedalwalker()
