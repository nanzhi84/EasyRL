import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class Network(nn.Module):
    def __init__(self, feature_dim, action_dim, n_neuron=10):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=n_neuron, bias=True),
            nn.Linear(in_features=n_neuron, out_features=action_dim, bias=True),
            nn.ReLU()
        )

    def forward(self, s):
        q = self.net(s)
        return q

class DQN(nn.Module):
    def __init__(self, feature_dim, action_dim,)