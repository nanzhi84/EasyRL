# rl_library/algorithms/q_learning.py
import numpy as np

class QLearning:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        q_values = self.get_q(state)
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state):
        q_values = self.get_q(state)
        next_q_values = self.get_q(next_state)
        q_values[action] += self.alpha * (reward + self.gamma * np.max(next_q_values) - q_values[action])
