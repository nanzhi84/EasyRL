import numpy as np

class Sarsa:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 learning_rate=0.1, 
                 gamma=0.99,
                 epsilon_start=1.0, 
                 epsilon_end=0.1, 
                 epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.Q_table = self.initialize_Q_table()
    
    def choose_action(self, state):
        # epsilon_greedy
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            max_actions = np.argwhere(self.Q_table[state, :] == np.max(self.Q_table[state, :])).flatten()
            return np.random.choice(max_actions)

    def update(self, state, action, reward, next_state, next_action, done):
        print(type(state),typr)
        current_Q = self.Q_table[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q_table[next_state, next_action]
        self.Q_table[state, action] += self.lr * (target - current_Q)
        if done:
            self.decay_epsilon()

    def initialize_Q_table(self):
        return np.zeros((self.state_dim, self.action_dim))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_Q_table(self, file_path):
        np.save(file_path, self.Q_table)
        print(f"Q-table saved to {file_path}")

    def load_Q_table(self, file_path):
        self.Q_table = np.load(file_path)
        print(f"Q-table loaded from {file_path}")