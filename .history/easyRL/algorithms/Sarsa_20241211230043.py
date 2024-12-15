import numpy as np

class Sarsa:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):

    def choose_action(self, state):

    def update(self, state, action, reward, next_state, next_action, done):

    def save_Q_table(self, file_path):
        np.save(file_path, self.Q_table)
        print(f"Q表已保存到 {file_path}")

    def load_Q_table(self, file_path):
        self.Q_table = np.load(file_path)
        print(f"Q表已从 {file_path} 加载")