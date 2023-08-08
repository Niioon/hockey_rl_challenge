
from soft_actor_critic import SacAgent
import copy
from train_sac import save_stats
import numpy as np
import torch
import matplotlib.pyplot as plt
import laserhockey.hockey_env as h_env
import os
import argparse
from eval_model import eval_agent


def main(args):
    # trains agent against itself

    automatic_entropy_tuning = False
    env = h_env.HockeyEnv()

    ac_space = env.action_space
    o_space = env.observation_space

    episodes_per_update = 3000
    agent_resets = 3
    path = args.model_path
    stats_dict_all = {'losses': [], 'stats': [], 'winners': [], 'winners_weak': [], 'winners_strong': []}

    for i in range(agent_resets):

        print(f'Training agent against it self for {episodes_per_update} episodes')
        agent = SacAgent(observation_space=o_space, action_space=ac_space,
                         automatic_entropy_tuning=automatic_entropy_tuning)
        agent.load_checkpoint(path, load_buffer=False)

        evaluation_agent = SacAgent(observation_space=o_space, action_space=ac_space,
                                    automatic_entropy_tuning=automatic_entropy_tuning)
        evaluation_agent.load_checkpoint(path, load_buffer=False)
        print(agent.train_log)

        # use eval agent as opponent, non changing version of the agent
        save_name, stats_dict = self_play(env, agent, evaluation_agent, evaluation_agent, episodes=episodes_per_update)
        for key in stats_dict.keys():
            stats_dict_all[key].extend(stats_dict[key])

        path = 'checkpoints/' + save_name

    print('saving stats')
    save_stats(stats_dict_all, save_name)


def self_play(env, agent, opponent, evaluation_agent, episodes=1000, eval=True, eval_episodes=500, episodes_per_step=1, update_steps=32):

    """
    trains agent against itself
    :param env:
    :param agent:
    :param episodes:
    :return:
    """
    i = 0
    losses = []
    winners = []
    winners_weak = []
    winners_strong = []
    stats = []


    # additional opoponents for evaluation needed for plots
    basic_weak_opponent = h_env.BasicOpponent(weak=True)
    basic_strong_opponent = h_env.BasicOpponent(weak=False)

    # copy_interval = episodes_per_step*20
    while i < episodes:

        if i % 500 == 0 and eval:
            print(f'Evaluation at episode {i}')
            n_wins = eval_agent(agent, evaluation_agent, env, episodes=eval_episodes, render=False)
            print(f'Evaluation against weak basic opponent')
            n_wins_weak = eval_agent(agent, basic_weak_opponent, env, episodes=eval_episodes, render=False)
            print(f'Evaluation against strong basic opponent')
            n_wins_strong = eval_agent(agent, basic_strong_opponent, env, episodes=eval_episodes, render=False)

            winners.append(n_wins)
            winners_weak.append(n_wins_weak)
            winners_strong.append(n_wins_strong)

        stats_temp, winner = duel_models(env, agent, opponent, episodes=episodes_per_step, render=False, store_transitions_1=True)

        for j in range(update_steps):
            losses.append(list(agent.update()))

        stats.extend(stats_temp)

        if i % 20 == 0:
            print(f'episode {i}: winner {winner}')

        i += episodes_per_step

    print(f'Evaluation at episode {i}')
    n_wins = eval_agent(agent, evaluation_agent, env, episodes=eval_episodes, render=False)
    print(f'Evaluation against weak basic opponent')
    n_wins_weak = eval_agent(agent, basic_weak_opponent, env, episodes=eval_episodes, render=False)
    print(f'Evaluation against strong basic opponent')
    n_wins_strong = eval_agent(agent, basic_strong_opponent, env, episodes=eval_episodes, render=False)

    winners.append(n_wins)
    winners_weak.append(n_wins_weak)
    winners_strong.append(n_wins_strong)

    stats_dict = {'losses': losses, 'stats': stats, 'winners': winners,
                  'winners_weak': winners_weak, 'winners_strong': winners_strong}

    agent.update_train_log(f'Trained against self for {episodes} episodes')
    save_name = f'sac_checkpoint_hockey_selfplay_e={episodes}_a={round(agent.alpha.item(), 4)}'
    agent.save_checkpoint(save_name=save_name)
    return save_name, stats_dict

def duel_models(env, agent_1, agent_2, episodes=1000, render=False, store_transitions_1=False):

    stats = []
    winner = []
    max_steps = 250

    for i in range(episodes):
        total_reward = 0
        obs_agent_1, _info = env.reset()
        obs_agent_2 = env.obs_agent_two()
        for t in range(max_steps):
            if render:
                env.render()
            done = False
            a1 = agent_1.act(obs_agent_1)
            a2 = agent_2.act(obs_agent_2)

            (obs_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))
            total_reward += reward

            if store_transitions_1:
                agent_1.store_transition((obs_agent_1, a1, reward, obs_new, done))
            obs_agent_1 = obs_new
            obs_agent_2 = env.obs_agent_two()
            if done:
                break
        winner.append(_info['winner'])
        stats.append([i, total_reward, t + 1])

    winner = np.asarray(winner).flatten()
    n_win = np.count_nonzero(winner == 1)
    n_l = np.count_nonzero(winner == -1)
    n_draw = np.count_nonzero(winner == 0)
    # print(f'Agent 1 won {n_win} games, Agent 2 won {n_l} games, draw {n_draw} games')
    return stats, [n_win, n_l, n_draw]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a SoftActorCritic RL-Agent in the hockey environment')
    parser.add_argument('model_path',
                        help='path to the checkpoint file of model_1')
    parser.add_argument('--render',  action="store_true", help='if true render env')
    parser.add_argument('-e', '--episodes', type=int, default=1000,
                        help='number of training episodes')
    args = parser.parse_args()
    main(args)
