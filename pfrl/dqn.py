import pfrl
from q_func import QFunction
import torch
import numpy as np
import gym
import logging
import sys
import os
import click
import json
# for importing the parent dirs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import cur_time


@click.command()
@click.argument("config-path", type=click.Path(exists=True))
def train(config_path: str):
    config = json.load(open(config_path))
    agent_params = config["agent_params"]
    train_params = config["train_params"]
    env = gym.make('CartPole-v1')
    # env = gym.make('SpaceInvaders-v0')
    env.seed(0)
    env.reset()

    obs_size = env.observation_space.low.size
    n_actions = env.action_space.n
    gamma = agent_params["gamma"]
    epsilon = agent_params["epsilon"]
    lr = agent_params["lr"]
    buffer_size = agent_params["buffer_size"]
    sync_int_steps = train_params["sync_int_steps"]
    eval_itr_episodes = train_params["eval_itr_episodes"]
    eval_int_steps = train_params["eval_int_episodes"] * 100
    q_func = QFunction(obs_size, n_actions)

    optimizer = torch.optim.Adam(q_func.parameters(), lr=lr)
    explorer = pfrl.explorers.ConstantEpsilonGreedy(
        epsilon=epsilon, random_action_func=env.action_space.sample
    )
    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=buffer_size)
    phi = lambda x: x.astype(np.float32, copy=False)
    gpu = 1

    agent = pfrl.agents.DoubleDQN(
        q_function=q_func,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=gamma,
        explorer=explorer,
        gpu=gpu,
        replay_start_size=1000,
        update_interval=1,
        target_update_interval=sync_int_steps,
        phi=phi,
    )
    # Set up the logger to print info messages for understandability.

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
    dir_name = cur_time()

    pfrl.experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        steps=2000000,
        outdir=f'pfrl/results/{dir_name}',
        eval_max_episode_len=500,  # Maximum length of each episode
        eval_n_steps=None,
        eval_n_episodes=eval_itr_episodes,
        eval_interval=eval_int_steps
    )


if __name__ == "__main__":
    train()
