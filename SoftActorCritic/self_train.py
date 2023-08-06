
from soft_actor_critic import SacAgent
import copy
from train_sac import save_stats
import numpy as np
import torch
import matplotlib.pyplot as plt
import laserhockey.hockey_env as h_env
import os
import argparse
from duel_models import duel_models
from eval_model import eval_agent


def main(args):
    # trains agent against itself
    env = h_env.HockeyEnv()

    ac_space = env.action_space
    o_space = env.observation_space
    basic_opponent = h_env.BasicOpponent(weak=False)

    episodes_per_update = 2500
    agent_resets = 4
    path = args.model_path
    stats_dict_all = {'losses': [], 'stats': [], 'winners': [], 'winners_basic': []}

    for i in range(agent_resets):

        print(f'Training agent against it self for {episodes_per_update} episodes')
        agent = SacAgent(observation_space=o_space, action_space=ac_space)
        agent.load_checkpoint(path, load_buffer=False)

        evaluation_agent = SacAgent(observation_space=o_space, action_space=ac_space)
        evaluation_agent.load_checkpoint(path, load_buffer=False)
        print(agent.train_log)

        print('Starting with evaluation against strong basic opponent')
        n_wins_basic = eval_agent(agent, basic_opponent, env, episodes=500, render=False)
        stats_dict_all['winners_basic'].extend(n_wins_basic)

        # use eval agent as opponent, non changing version of the agent
        save_name, stats_dict = self_play(env, agent, evaluation_agent, evaluation_agent, episodes=episodes_per_update)
        stats_dict_all['losses'].extend(stats_dict['losses'])
        stats_dict_all['stats'].extend(stats_dict['stats'])
        stats_dict_all['winners'].extend(stats_dict['winners'])

        path = 'checkpoints/' + save_name

    print('Final evaluation aginst basic opponent')
    n_wins_basic = eval_agent(agent, basic_opponent, env, episodes=500, render=False)
    stats_dict_all['winners_basic'].extend(n_wins_basic)
    print('Final evaluation against self')
    n_wins = eval_agent(agent, evaluation_agent, env, episodes=500, render=False)
    stats_dict_all['winners'].extend(n_wins)

    print('saving stats')
    save_stats(stats_dict_all, f'sac_checkpoint_hockey_selfplay_e={episodes_per_update}_{agent_resets}_resets'
)


def self_play(env, agent, opponent, evaluation_agent, episodes=1000, eval=True, episodes_per_step=1, update_steps=32):

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
    stats = []
    # copy_interval = episodes_per_step*20
    while i < episodes:

        if i % 500 == 0 and eval:
            print(f'Evaluation at episode {i}')
            n_wins = eval_agent(agent, evaluation_agent, env, episodes=500, render=False)
            winners.append(n_wins)

        stats_temp, winner = duel_models(env, agent, opponent, episodes=episodes_per_step, render=False, store_transitions_1=True)

        for j in range(update_steps):
            losses.append(list(agent.update()))

        stats.extend(stats_temp)

        if i % 20 == 0:
            print(f'episode {i}: winner {winner}')

        i += episodes_per_step

    stats_dict = {'losses': losses, 'stats': stats, 'winners': winners}

    agent.update_train_log(f'Trained against self for {episodes} episodes')
    save_name = f'sac_checkpoint_hockey_selfplay_e={episodes}'
    # save_stats(stats_dict, save_name)
    # plot_loss_rewards(stats, losses, title='Losses and Rewards for Defense Training')
    agent.save_checkpoint(save_name=save_name)
    return save_name, stats_dict




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a SoftActorCritic RL-Agent in the hockey environment')
    parser.add_argument('model_path',
                        help='path to the checkpoint file of model_1')
    parser.add_argument('--render',  action="store_true", help='if true render env')
    parser.add_argument('-e', '--episodes', type=int, default=1000,
                        help='number of training episodes')
    args = parser.parse_args()
    main(args)
