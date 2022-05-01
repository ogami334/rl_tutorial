from replay_buffer import ReplayBuffer
from q_func import QFunction
import numpy as np
import copy
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


class DQNAgent:
    def __init__(self, obs_size, action_size, gamma=0.98, lr=0.0005, epsilon=0.1, buffer_size=10000, batch_size=32):
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs_size = obs_size
        self.action_size = action_size

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QFunction(obs_size, action_size)
        self.qnet_target = QFunction(obs_size, action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            tmp_state = torch.tensor(state)
            tmp_state = tmp_state.unsqueeze(0)
            qs = self.qnet(tmp_state)
            return int(qs.data.argmax())

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        self.optimizer.zero_grad()
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[torch.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = torch.max(next_qs, axis=1).values
        target = reward + (1 - done) * self.gamma * next_q

        loss = self.criterion(q, target)
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def save(self, path):
        model_path = os.path.join(path, 'model.pth')
        target_path = os.path.join(path, 'target_model.pth')
        torch.save(self.qnet.state_dict(), model_path)
        torch.save(self.qnet_target.state_dict(), target_path)

    def load(self, path):
        model_path = os.path.join(path, 'model.pth')
        target_path = os.path.join(path, 'target_model.pth')
        self.qnet.load_state_dict(torch.load(model_path))
        self.qnet_target.load_state_dict(torch.load(target_path))
