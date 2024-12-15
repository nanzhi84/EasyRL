import matplotlib.pyplot as plt

def plot_rewards(rewards, ma_rewards, algorithm, env_name):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode', color='blue', alpha=0.6)
    plt.plot(ma_rewards, label='Moving Average (0.1)', color='red', alpha=0.9)
    plt.title(f"{algorithm} on {env_name}")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()