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
                 learning_rate=0.01,
                 gamma=0.9,
                 epsilon_start=0.9,
                 epsilon_end=0.1,
                 epsilon_decay=0.995,
                 memory_size=500,
                 batch_size=32,
                 replace_target_iter=300):
        super(DUELING_DDQN, self).__init__(
            feature_dim, action_dim, learning_rate, gamma, 
            epsilon_start, epsilon_end, epsilon_decay, 
            memory_size, batch_size, replace_target_iter)
        
        # Override the networks with Dueling networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_net = DuelingNetwork(feature_dim=self.feature_dim,
                                     action_dim=self.action_dim).to(self.device)
        self.target_net = DuelingNetwork(feature_dim=self.feature_dim,
                                       action_dim=self.action_dim).to(self.device)
        
        # Reset optimizer with new eval_net
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)