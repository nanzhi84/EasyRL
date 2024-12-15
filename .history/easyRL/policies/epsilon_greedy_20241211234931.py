import numpy as np

class EpsilonGreedy:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

    def choose_action(self, Q_values, action_dim):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(action_dim)
        else:
            max_actions = np.argwhere(Q_values == np.max(Q_values)).flatten()
            return np.random.choice(max_actions)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)