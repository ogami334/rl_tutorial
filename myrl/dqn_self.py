import copy 
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs

from utils import cur_time

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (list(state), action, reward, list(next_state), done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)#データは消える？

        # state: array with length 4
        # action: int
        # reward: float
        # next_state: array with length 4
        # done: 0 or 1
        state = torch.tensor([x[0] for x in data])
        action = torch.tensor([x[1] for x in data], dtype=torch.int64)
        reward = torch.tensor([x[2] for x in data])
        next_state = torch.tensor([x[3] for x in data])
        done = torch.tensor([x[4] for x in data], dtype=torch.int32)
        return state, action, reward, next_state, done

class ReplayBuffer_Dict:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = {"state": list(state), "action": action, "reward": reward, "next_state": list(next_state), "done": done}
        # arrayになってしまう なぜ？
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)#データは消える？

        # state: array with length 4
        # action: int
        # reward: float
        # next_state: array with length 4
        # done: 0 or 1
        state = torch.tensor([x["state"] for x in data])
        action = torch.tensor([x["action"] for x in data], dtype=torch.int64)
        reward = torch.tensor([x["reward"] for x in data])
        next_state = torch.tensor([x["next_state"] for x in data])
        done = torch.tensor([x["done"] for x in data], dtype=torch.int32)
        return state, action, reward, next_state, done


class QFunction(nn.Module):
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
    
        return h

    
class DQNAgent:
    def __init__(self, obs_size, action_size):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.obs_size = obs_size
        self.action_size = action_size

        self.replay_buffer = ReplayBuffer_Dict(self.buffer_size, self.batch_size)
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




env = gym.make('CartPole-v1')
obs_size = env.observation_space.low.size
action_size = env.action_space.n
agent = DQNAgent(obs_size=obs_size, action_size=action_size)

episodes = 10000
sync_interval = 20
reward_history = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, int(done))
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode :{}, total reward : {}".format(episode, total_reward))



