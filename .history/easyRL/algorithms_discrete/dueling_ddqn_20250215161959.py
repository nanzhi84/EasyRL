import torch
import torch.nn as nn
from easyRL.algorithms.ddqn import DDQN

class DuelingNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(DuelingNetwork, self).__init__()
        
        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q

class DUELING_DDQN(DDQN):
    def __init__(self, 
                 feature_dim, 
                 action_dim, 
                 learning_rate=0.001, 
                 gamma=0.99, 
                 epsilon_start=1.0, 
                 epsilon_end=0.01, 
                 epsilon_decay=0.998, 
                 memory_size=10000, 
                 batch_size=32, 
                 replace_target_iter=100):
        super().__init__(
            feature_dim, action_dim, learning_rate, gamma, 
            epsilon_start, epsilon_end, epsilon_decay, 
            memory_size, batch_size, replace_target_iter
        )
        
        # Override the networks with Dueling networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_net = DuelingNetwork(self.feature_dim, self.action_dim).to(self.device)
        self.target_net = DuelingNetwork(self.feature_dim, self.action_dim).to(self.device)
        
        # Reset optimizer with new eval_net
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)