import pfrl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QFunction(torch.nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=100):
        super().__init__()
        self.l1 = nn.Linear(obs_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        h = x
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        return pfrl.action_value.DiscreteActionValue(h)