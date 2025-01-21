import numpy as np
from .base import BaseAlgorithm

class QLearning(BaseAlgorithm):
    
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            max_actions = np.argwhere(self.Q_table[state, :] == np.max(self.Q_table[state, :])).flatten()
            return np.random.choice(max_actions)

    def update(self, state, action, reward, next_state, done):
        current_Q = self.Q_table[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q_table[next_state, :])
        self.Q_table[state, action] += self.lr * (target - current_Q)
        if done:
            self.decay_epsilon()