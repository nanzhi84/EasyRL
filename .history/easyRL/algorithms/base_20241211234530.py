from abc import ABC, abstractmethod
import np.

class BaseAlgorithm(ABC):
    def __init__(self, state_dim, action_dim, learning_rate=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.Q_table = self.initialize_Q_table()

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    def initialize_Q_table(self):
        return np.zeros((self.state_dim, self.action_dim))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_Q_table(self, file_path):
        np.save(file_path, self.Q_table)
        print(f"Q表已保存到 {file_path}")

    def load_Q_table(self, file_path):
        self.Q_table = np.load(file_path)
        print(f"Q表已从 {file_path} 加载")