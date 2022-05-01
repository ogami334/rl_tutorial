import sys
import os
# for importing the parent dirs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import copy
from q_func import QFunction
from replay_buffer import ReplayBuffer
from utils import cur_time



class DQNAgent:
    def __init__(self, obs_size, action_size):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
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


env = gym.make('CartPole-v1')
obs_size = env.observation_space.low.size
action_size = env.action_space.n
agent = DQNAgent(obs_size=obs_size, action_size=action_size)
dir_name = cur_time()

train_episodes = 10000
eval_interval = 100
eval_episodes = 10
sync_interval = 20
reward_history = []
current_best = -1e20
os.makedirs(f'results/{dir_name}/best/')


for episode in range(train_episodes):
    if episode % eval_interval == 0:  # テストを行う。
        reward_list = []
        for eval_episode in range(eval_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                # updateを行わない
                state = next_state
                total_reward += reward
            reward_list.append(total_reward)
        average_reward = sum(reward_list) / eval_episodes
        if average_reward > current_best:
            current_best = average_reward
            agent.save(f'results/{dir_name}/best')
    # trainをやる
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
