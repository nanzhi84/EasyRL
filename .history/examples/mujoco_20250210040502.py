@dataclass
class MuJoCoConfig:
    env_name: str = 'HalfCheetah-v4'
    algorithm: str = 'ppo'
    learning_rate: float = 3e-4
    gamma: float = 0.99
    train_eps: int = 1000
    max_steps: int = 1000
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    ppo_epochs: int = 10
    batch_size: int = 64
    gae_lambda: float = 0.95

def train_mujoco(config=MuJoCoConfig()):
    env = gym.make(config.env_name)
    agent = PPO(
        feature_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr=config.learning_rate,
        gamma=config.gamma,
        clip_eps=config.clip_epsilon,
        entropy_coef=config.entropy_coef,
        ppo_epochs=config.ppo_epochs,
        batch_size=config.batch_size,
        gae_lambda=config.gae_lambda
    )
    
    rewards = []
    ma_rewards = []
    
    for ep in range(config.train_eps):
        state = env.reset()
        ep_reward = 0
        episode_data = []
        
        for _ in range(config.max_steps):
            # 选择动作并执行
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            episode_data.append((state, action, log_prob, value, reward, done))
            
            state = next_state
            ep_reward += reward
            
            if done:
                break
        
        # 后处理经验数据
        states, actions, log_probs, values, rewards, dones = zip(*episode_data)
        agent.memory = episode_data  # 存储完整episode数据
        
        # 更新策略
        agent.update()
        
        # 记录训练进度
        ma_rewards.append(ma_rewards[-1]*0.9 + ep_reward*0.1 if ma_rewards else ep_reward)
        rewards.append(ep_reward)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{config.train_eps} | "
                  f"Return: {ep_reward:.1f} | "
                  f"MA Return: {ma_rewards[-1]:.1f}")
    
    return rewards, ma_rewards

# 使用示例
if __name__ == "__main__":
    rewards, ma_rewards = train_mujoco()
    plt.plot(rewards, alpha=0.6)
    plt.plot(ma_rewards, linewidth=2)
    plt.title('PPO on HalfCheetah-v4')
    plt.show()
