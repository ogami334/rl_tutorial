import pfrl
from q_func import QFunction
import torch
import numpy as np
import gym
import logging
import sys
import os
# for importing the parent dirs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import cur_time

env = gym.make('CartPole-v1')
env.seed(0)
env.reset()

obs_size = env.observation_space.low.size
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)

optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
gamma = 0.99
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample
)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
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
# Set up the logger to print info messages for understandability.

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
dir_name = cur_time()

pfrl.experiments.train_agent_with_evaluation(
    agent=agent,
    env=env,
    steps=20000,           # Train the agent for 2000 steps
    outdir=f'pfrl/results/{dir_name}',
    eval_max_episode_len=500,  # Maximum length of each episode
    eval_n_steps=None,
    eval_n_episodes=100,
    eval_interval=1000,
)

# モデルの情報は保存したものをロードして使いたい気持ち。
