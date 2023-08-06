import numpy as np
import torch
import matplotlib.pyplot as plt
import laserhockey.hockey_env as h_env
import os
import argparse
import time
import sys
from soft_actor_critic import SacAgent
import copy
from train_sac import save_stats

# from TD3.TD3 import TD3Agent


def main(args):

    # evaluates two trained models against each other

    env = h_env.HockeyEnv()

    ac_space = env.action_space
    o_space = env.observation_space

    if args.model_1 == 'sac':
        agent_1 = SacAgent(observation_space=o_space, action_space=ac_space)
    elif args.model_1 == 'td3':
        agent_1 = TD3Agent(env.observation_space, env.action_space)
    else:
        raise ValueError('Unknown model type for player 1')

    if args.model_2 == 'sac':
        agent_2 = SacAgent(observation_space=o_space, action_space=ac_space)
    elif args.model_2 == 'td3':
        agent_2 = TD3Agent(env.observation_space, env.action_space)
    else:
        raise ValueError('Unknown model type for player 2')

    agent_1.load_checkpoint(args.model_path_1, load_buffer=False)
    agent_2.load_checkpoint(args.model_path_2, load_buffer=False)

    stats, n_wins = duel_models(env, agent_1, agent_2, episodes=args.episodes, render=args.render)
    print(n_wins)


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
    parser.add_argument('model_1', choices=['sac', 'td3'],
                        help='type of model 1')
    parser.add_argument('model_2', choices=['sac', 'td3'],
                        help='type of model 2')
    parser.add_argument('model_path_1',
                        help='path to the checkpoint file of model_1')
    parser.add_argument('model_path_2',
                        help='path to the checkpoint file of model_2')
    parser.add_argument('--render',  action="store_true", help='if true render env')
    parser.add_argument('-e', '--episodes', type=int, default=1000,
                        help='number of training episodes')
    args = parser.parse_args()
    main(args)
