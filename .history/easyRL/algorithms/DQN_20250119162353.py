import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class Network(nn.Module):
    """
    Network Structure
    """
    def __init__(self,
                 n_features,
                 n_actions,
                 n_neuron=10
                 ):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_neuron, bias=True),
            nn.Linear(in_features=n_neuron, out_features=n_actions, bias=True),
            nn.ReLU()
        )

    def forward(self, s):
        """
        :param s: s
        :return: q
        """
        q = self.net(s)
        return q