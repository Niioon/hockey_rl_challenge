import numpy as np
import torch
import matplotlib.pyplot as plt
from soft_actor_critic import SacAgent
import laserhockey.hockey_env as h_env
import argparse
import os
import time

parser = argparse.ArgumentParser(description='Trains a SoftActorCritic RL-Agent in the hockey environment')
parser.add_argument('--model_path',
                    help='loads an already existing model from the specified path instead of initializing a new one')
parser.add_argument('--mode', choices=['defense', 'shooting', 'normal'], default='normal',
                    help='game mode for evaluation')
args = parser.parse_args()


def main():
    env = h_env.HockeyEnv()
    ac_space = env.action_space
    o_space = env.observation_space
    agent = SacAgent(observation_space=o_space, action_space=ac_space)

    # load model parameters
    agent.load_checkpoint(args.model_path)

    print(args.mode)
    # check loading
    print(agent.train_iter)
    print(agent.log_alpha)
    print(agent.alpha)

    if args.mode == 'defense':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
        opponent = None
    elif args.mode == 'shooting':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        opponent = None
    elif args.mode == 'normal':
        env = h_env.HockeyEnv()
        opponent = h_env.BasicOpponent(weak=True)
    else:
        raise ValueError('Unknown mode, chose one of [defense, shooting, normal')

    render = False
    max_steps = 250
    mode = args.mode
    stats = []
    eval_episodes = 100
    winner = []
    for i in range(eval_episodes):
        total_reward = 0
        obs, _info = env.reset()
        obs_agent2 = env.obs_agent_two()
        for t in range(max_steps):
            if render:
                env.render()

            done = False
            # get action of agent to be trained
            a1 = agent.select_action(obs)
            # get action of opponent
            if opponent is None:
                # second agent is static in defense training
                a2 = [0, 0, 0, 0]
            else:
                a2 = opponent.act(obs_agent2)

            (obs_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))
            total_reward += reward
            obs = obs_new
            obs_agent2 = env.obs_agent_two()
            if done:
                break
        print(_info)
        winner.append(_info['winner'])
        stats.append([i, total_reward, t + 1])
    rewards = np.asarray(stats)[:, 1]
    mean_reward = np.mean(rewards)
    winner = np.asarray(winner)
    n_win = np.count_nonzero(winner == 1)
    n_l = np.count_nonzero(winner == -1)
    n_draw = np.count_nonzero(winner == 0)
    print(f'Agent won {n_win} games, lost {n_l} games, draw {n_draw} games')
    print(f'Average reward over {eval_episodes} episodes: {mean_reward}')


if __name__ == '__main__':
    main()
