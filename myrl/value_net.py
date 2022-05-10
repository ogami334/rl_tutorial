import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, obs_size, hidden_size=100):
        super().__init__()
        self.l1 = nn.Linear(obs_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
