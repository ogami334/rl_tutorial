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
    config = json.load(open(config_path))
    env = gym.make('CartPole-v1')
    env.seed(0)
    obs_size = env.observation_space.low.size
    action_size = env.action_space.n

    agent_params = config["agent_params"]
    agent = DQNAgent(obs_size=obs_size, action_size=action_size, **agent_params)

    train_params = config["train_params"]
    train_itr_episodes = train_params['train_itr_episodes']
    eval_int_episodes = train_params['eval_int_episodes']
    eval_itr_episodes = train_params['eval_itr_episodes']
    sync_int_episodes = train_params['sync_int_episodes']
    display_int = train_params['display_int']

    dir_name = cur_time()
    result_path = f'myrl/results/{dir_name}/'
    best_path = f'myrl/results/{dir_name}/best/'
    os.makedirs(best_path)
    # 実験の設定をディレクトリに保存する
    with open(f'{result_path}config.json', mode="w") as f:
        json.dump(config, f, indent=4)
    f.close()
    # scores.txtを作成する
    f = open(f'{result_path}scores.txt', mode="w")
    out_category = ["episode", "mean", "max", "min", "stdev"]
    f.write("\t".join(out_category) + "\n")

    cur_best_score = -1e20

    for episode in range(train_itr_episodes):
        if episode % eval_int_episodes == 0:  # テストを行う。
            reward_list = []
            for eval_episode in range(eval_itr_episodes):
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
            reward_array = np.array(reward_list)
            _episode = episode
            _mean, _max, _min, _stdev = np.mean(reward_array), np.max(reward_array), np.min(reward_array), np.std(reward_array)
            out_str = "\t".join(list(map(str, [_episode, _mean, _max, _min, _stdev])))
            f.write(out_str + "\n")
            average_reward = sum(reward_list) / eval_itr_episodes
            if average_reward > cur_best_score:
                cur_best_score = average_reward
                agent.save(best_path)

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

        if episode % sync_int_episodes == 0:
            agent.sync_qnet()

        if episode % display_int == 0:
            print("episode :{}, total reward : {}".format(episode, total_reward))


    # エージェントの評価と描画を行う；
    agent.load(best_path)
    for eval_episode in range(eval_itr_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            # env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            # updateを行わない
            state = next_state
            total_reward += reward
        print(total_reward)
    return

if __name__ == "__main__":
    train()
