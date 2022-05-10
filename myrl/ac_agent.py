from http.client import CannotSendRequest
from secrets import choice
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
import os

from policy_net import PolicyNet
from value_net import ValueNet


class ACAgent:
    def __init__(self, obs_size, action_size, gamma, lr_pi, lr_v):
        self.gamma = gamma
        self.lr_pi = lr_pi
        self.lr_v = lr_v

        self.pi = PolicyNet(action_size=action_size, obs_size=obs_size)
        self.v = ValueNet(obs_size=obs_size)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)
        self.criterion_v = nn.MSELoss()

    def act(self, state):
        state = torch.tensor(state)
        state = torch.unsqueeze(state, 0)
        probs = self.pi(state)
        probs = probs[0]
        sampler = Categorical(probs)
        action = sampler.sample()
        return action

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state)
        state = torch.unsqueeze(state, 0)
        next_state = torch.tensor(next_state)
        next_state = torch.unsqueeze(next_state, 0)

        # ========== (1) Update V network ===========
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        v = self.v(state)
        loss_v = self.criterion_v(v, target)

        # ========== (2) Update pi network ===========
        delta = target - v

        probs = self.pi(state)
        sampler = Categorical(probs)
        log_probs = -sampler.log_prob(action)
        # print(log_probs)
        pseudo_loss = torch.sum(log_probs * delta)
        # print(pseudo_loss)

        self.optimizer_pi.zero_grad()
        self.optimizer_v.zero_grad()
        loss_v.backward(retain_graph=True)  # いいのか？
        pseudo_loss.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()

    def save(self, path):
        policy_path = os.path.join(path, 'policy_model.pth')
        value_path = os.path.join(path, 'value_model.pth')
        torch.save(self.pi.state_dict(), policy_path)
        torch.save(self.v.state_dict(), value_path)

    def load(self, path):
        policy_path = os.path.join(path, 'policy_model.pth')
        value_path = os.path.join(path, 'value_model.pth')
        self.pi.load_state_dict(torch.load(policy_path))
        self.v.load_state_dict(torch.load(value_path))
