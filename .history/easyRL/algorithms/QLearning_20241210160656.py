import numpy as np

class QLearning:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        """
        Q-Learning 算法的实现。

        参数:
        - state_dim (int): 状态空间的维度（离散状态的数量）。
        - action_dim (int): 动作空间的维度（可选动作的数量）。
        - learning_rate (float): 学习率 (α)。
        - gamma (float): 折扣因子 (γ)。
        - epsilon_start (float): 初始探索率 (ε)。
        - epsilon_end (float): 最小探索率 (ε)。
        - epsilon_decay (float): 探索率衰减因子。
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma

        # 初始化 Q 表为全零
        self.Q_table = np.zeros((state_dim, action_dim))

        # ε-贪婪策略参数
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        """
        根据当前状态选择一个动作，使用 ε-贪婪策略。

        参数:
        - state (int): 当前状态。

        返回:
        - action (int): 选择的动作。
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # 探索：随机选择动作
            action = np.random.choice(self.action_dim)
        else:
            # 利用：选择 Q 值最大的动作
            action = np.argmax(self.Q_table[state, :])
        return action

    def update(self, state, action, reward, next_state, done):
        # 当前 Q 值
        current_Q = self.Q_table[state, action]

        if done:
            target = reward
        else:
            # 估计未来奖励的最大 Q 值
            target = reward + self.gamma * np.max(self.Q_table[next_state, :])

        # Q-Learning 更新规则
        self.Q_table[state, action] += self.lr * (target - current_Q)

        # 衰减 ε，确保 ε 不低于最小值
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_Q_table(self, file_path):
        """
        保存 Q 表到文件。

        参数:
        - file_path (str): 文件路径。
        """
        np.save(file_path, self.Q_table)
        print(f"Q表已保存到 {file_path}")

    def load_Q_table(self, file_path):
        """
        从文件加载 Q 表。

        参数:
        - file_path (str): 文件路径。
        """
        self.Q_table = np.load(file_path)
        print(f"Q表已从 {file_path} 加载")
