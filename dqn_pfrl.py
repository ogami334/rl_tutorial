import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import pfrl
from tqdm import tqdm
env = gym.make('CartPole-v1')


class QFunction(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = nn.Linear(obs_size, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, n_actions)

    def forward(self, x):
        h = x
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        return pfrl.action_value.DiscreteActionValue(h)
    

obs_size = env.observation_space.low.size
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)

optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
gamma = 0.99
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample
)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity = 10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)
gpu = -1

agent = pfrl.agents.DoubleDQN(
    q_function=q_func,
    optimizer=optimizer,
    replay_buffer=replay_buffer,
    gamma=gamma,
    explorer=explorer,
    gpu=gpu,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    phi=phi,
)

n_episodes = 10000
max_episode_len = 200
for i in tqdm(range(1, n_episodes + 1)):
    obs = env.reset()
    R = 0
    t = 0
    while True:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 100 == 0:
        print('episode:', i, 'R:', R)
        print('statistics:', agent.get_statistics())
print('Finished.')

with agent.eval_mode():
    for i in range(10):
        obs = env.reset()
        R = 0
        t = 0
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
            reset = t == 200
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
        print('evaluation episode:', i, 'R:', R)
