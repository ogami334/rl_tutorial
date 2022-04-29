import pfrl
from q_func import QFunction
import torch
import numpy as np
import gym

env = gym.make('CartPole-v1')
env.reset()

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
# Set up the logger to print info messages for understandability.
import logging
import sys
logger =logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

pfrl.experiments.train_agent_with_evaluation(
    agent=agent,
    env=env,
    steps=1000000,           # Train the agent for 2000 steps
    outdir='result',      # Save everything to 'result' directory
    eval_max_episode_len=500,  # Maximum length of each episode
    eval_n_steps=None,
    eval_n_episodes=100,
    eval_interval=10000,
    logger=logger
)

# モデルの情報は保存したものをロードして使いたい気持ち。