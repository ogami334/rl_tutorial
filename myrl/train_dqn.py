from comet_ml import Experiment
import sys
import os
# # for importing the parent dirs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gym
import numpy as np
import click
import json
from utils import cur_time
from dqn_agent import DQNAgent


# configにenvのaction_size, obs_sizeを記入してあげるのもあり。
@click.command()
@click.argument("config-path", type=click.Path(exists=True))
def train(config_path: str):
    alg_name = (config_path.split('/')[1]).split('.')[0]
    config = json.load(open(config_path))
    env = gym.make('CartPole-v1')
    env.seed(0)
    obs_size = env.observation_space.low.size
    action_size = env.action_space.n
    agent_params = config["agent_params"]
    dir_name = alg_name + cur_time()
    experiment = Experiment(
        api_key="PT4wmzuGGtRqDkX3LbOSuXrS3",
        project_name="rl_tutorial",
        workspace="ogami")
    experiment.set_name(dir_name)
    experiment.log_parameters(agent_params)
    agent = DQNAgent(obs_size=obs_size, action_size=action_size, **agent_params)
    # DQN-specific
    train_params = config["train_params"]

    train_itr_steps = train_params["train_itr_steps"]
    eval_int_steps = train_params["eval_int_steps"]
    eval_itr_episodes = train_params['eval_itr_episodes']
    sync_int_steps = train_params['sync_int_steps']

    result_path = f'myrl/results/{dir_name}/'
    best_path = f'myrl/results/{dir_name}/best/'
    os.makedirs(best_path)
    # 実験の設定をディレクトリに保存する
    with open(f'{result_path}config.json', mode="w") as f:
        json.dump(config, f, indent=4)
    f.close()

    count_episodes = 0
    cur_best_score = -1e20
    next_log_step = eval_int_steps
    train_step = 0

    with experiment.train():
        while train_step < train_itr_steps:
            if train_step >= next_log_step:
                reward_list = []
                for eval_episode in range(eval_itr_episodes):
                    state = env.reset()
                    done = False
                    total_reward = 0
                    while not done:
                        action = agent.act(state)
                        next_state, reward, done, info = env.step(action)
                        state = next_state
                        total_reward += reward
                    reward_list.append(total_reward)
                reward_array = np.array(reward_list)
                _mean = np.mean(reward_array)
                experiment.log_metric("mean_reward", _mean, step=train_step)
                next_log_step = (1 + (train_step // eval_int_steps)) * eval_int_steps
                print(train_step, _mean, count_episodes)
                if _mean > cur_best_score:
                    cur_best_score = _mean
                    agent.save(best_path)
            else:
                state = env.reset()
                done = False
                total_reward = 0
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    loss = agent.update(state, action, reward, next_state, int(done))
                    train_step += 1
                    if loss > 1e-15:
                        experiment.log_metric("loss", loss, step=train_step)
                    if train_step % sync_int_steps == 0:
                        agent.sync_qnet()
                    # DQN-specific
                    state = next_state
                    total_reward += reward
                count_episodes += 1
                experiment.log_metric("total_episodes", count_episodes, step=train_step)

    with experiment.test():
        last_eval_rewards = []
        # エージェントの評価と描画を行う；
        agent.load(best_path)
        for eval_episode in range(100):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                # updateを行わない
                state = next_state
                total_reward += reward
            last_eval_rewards.append(total_reward)
        last_eval_rewards_array = np.array(last_eval_rewards)

        last_mean = np.mean(last_eval_rewards_array)
        last_min = np.min(last_eval_rewards_array)
        last_max = np.max(last_eval_rewards_array)
        experiment.log_metric("mean_reward", last_mean, step=0)
        experiment.log_metric("min_reward", last_min, step=0)
        experiment.log_metric("max_reward", last_max, step=0)

    return


if __name__ == "__main__":
    train()
