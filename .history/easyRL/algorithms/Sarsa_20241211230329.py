import numpy as np

class Sarsa:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.Q_table = np.zeros((state_dim, action_dim))
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):

    def update(self, state, action, reward, next_state, next_action, done):

    def save_Q_table(self, file_path):
        np.save(file_path, self.Q_table)
        print(f"Q表已保存到 {file_path}")

    def load_Q_table(self, file_path):
        self.Q_table = np.load(file_path)
        print(f"Q表已从 {file_path} 加载")