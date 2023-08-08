import numpy as np
import torch
import matplotlib.pyplot as plt
from TD3 import TD3Agent
import laserhockey.hockey_env as h_env
import argparse
import os
import time


def main(opts):
    env = h_env.HockeyEnv()
    td3_agent = TD3Agent(env.observation_space, env.action_space)
    
    # load model parameters
    td3_agent.load_checkpoint(opts.model_path, load_buffer=False)

    print('mode ', opts.mode)
    # check loading
    print('training_iterations: ', td3_agent.train_iter)
    #print('training_log', td3_agent.train_log)

    if args.mode == 'defense':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
        opponent = None
    elif args.mode == 'shooting':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        opponent = None
    elif args.mode == 'normal':
        env = h_env.HockeyEnv()
        opponent = h_env.BasicOpponent(weak=args.weak)
        print('weak opponent: ', args.weak)
    else:
        raise ValueError('Unknown mode, chose one of [defense, shooting, normal')

    render = False
    max_steps = 250
    eval_episodes = 500

    winner = eval_agent(td3_agent, opponent, env, episodes=eval_episodes, render=render)

    # agent.set_eval()


def eval_agent(td3_agent, opponent, env, episodes=250, render=False, max_steps=250):
    stats = []
    winner = []
    for i in range(episodes):
        total_reward = 0
        obs, _info = env.reset()
        obs_agent2 = env.obs_agent_two()
        for t in range(max_steps):
            if render:
                env.render()

            done = False
            # get action of agent       to be trained
            a1 = td3_agent.act(obs)
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
        # print(_info)
        winner.append(_info['winner'])
        stats.append([i, total_reward, t + 1])
    rewards = np.asarray(stats)[:, 1]
    #plot_rewards(rewards)
    mean_reward = np.mean(rewards)
    winner = np.asarray(winner)
    n_win = np.count_nonzero(winner == 1)
    n_l = np.count_nonzero(winner == -1)
    n_draw = np.count_nonzero(winner == 0)
    print(f'Agent won {n_win} games, lost {n_l} games, draw {n_draw} games')
    print(f'Win/Loss+Win Ratio {n_win/(n_l+n_win)}')
    print(f'Average reward over {episodes} episodes: {mean_reward}')
    return [n_win, n_l, n_draw]

def plot_rewards(rewards):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('rewards in evaluation')
    ax.plot(rewards)
    ax.set_ylabel('total_reward')
    ax.set_xlabel('num_episodes')
    plt.savefig(f'stats/figure')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a SoftActorCritic RL-Agent in the hockey environment')
    parser.add_argument('--model_path',
                        help='loads an already existing model from the specified path instead of initializing a new one')
    parser.add_argument('--mode', choices=['defense', 'shooting', 'normal'], default='normal',
                        help='game mode for evaluation')
    parser.add_argument('--weak', action="store_true",
                        help='difficulty of the opponent in the normal mode, no influence in other modes')

    args = parser.parse_args()
    main(args)
