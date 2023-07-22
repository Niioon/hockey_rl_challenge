import numpy as np
import torch
import matplotlib.pyplot as plt
from soft_actor_critic import SacAgent
import laserhockey.hockey_env as h_env
import os
import argparse


parser = argparse.ArgumentParser(description='Trains a SoftActorCritic RL-Agent in the hockey environment')
parser.add_argument('-e', '--episodes', action='store_const', type=int, default=1000,
                    help='number of training episodes')
parser.add_argument('--model_path', type=str,
                    help='loads an already existing model from the specified path instead of initializing a new one')
parser.add_argument('--train_mode', choices=['defense', 'shooting', 'normal'], type=str, default='normal',
                    help='game mode to train in')
parser.add_argument('-l', 'learning_rate',  help='learning rate', default='0.0002')

args = parser.parse_args()


def main():
    mode = args.train_mode
    model_path = args.model_path
    episodes = args.episodes
    env = h_env.HockeyEnv()
    ac_space = env.action_space
    o_space = env.observation_space
    # print(ac_space, o_space)
    # print(ac_space.shape[0])
    sac_agent = SacAgent(observation_space=o_space, action_space=ac_space)

    # TRAIN DEFENSE
    mode = 'train_defense'
    episodes = 3000
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    losses, stats = train_agent(sac_agent, env, mode=mode, max_episodes=episodes)
    env.close()
    save_stats(np.asarray(stats), np.asarray(losses), mode + str(episodes))
    # plot_loss_rewards(stats, losses, title='Losses and Rewards for Defense Training')
    #
    # sac_agent.save_checkpoint('hockey', 'defense')

    # Evaluate Defense
    # env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    # stats = eval_agent(sac_agent, env, render=False, mode='train_defense')
    # env.close()
    # plot_rewards(stats)

    # env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
#
    # losses, stats = train_agent(sac_agent, env, mode='train_shooting', max_episodes=1000)
    # plot_loss_rewards(stats, losses, title='Losses and Rewards for Normal Training')
    # env.close()

    # env = h_env.HockeyEnv()
#
    # losses, stats = train_agent(sac_agent, env, mode='normal', max_episodes=1000)
    # plot_loss_rewards(stats, losses, title='Losses and Rewards for Normal Training')
    # stats = eval_agent(sac_agent, env, render=True, mode='normal')
#
    # env.close()
#
    # sac_agent.save_checkpoint('hockey')


def train_agent(agent, env, mode='normal', max_episodes=1000):
    stats = []
    losses = []
    max_steps = 500
    update_steps = 64

    if mode == ('train_defense' or 'train_shooting'):
        opponent = None
    else:
        opponent = h_env.BasicOpponent(weak=True)

    for i in range(max_episodes):
        # print("Starting a new episode")
        agent.set_train()
        total_reward = 0
        obs, _info = env.reset()
        obs_agent2 = env.obs_agent_two()
        for t in range(max_steps):
            done = False
            # get action of agent to be trained
            a1 = agent.select_action(obs)
            # get action of opponent
            if mode == ('train_defense' or 'train_shooting'):
                # second agent is static in defense training
                a2 = [0, 0, 0, 0]
            else:
                a2 = opponent.act(obs_agent2)

            (obs_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))
            total_reward += reward
            agent.store_transition((obs, a1, reward, obs_new, done))
            obs = obs_new
            obs_agent2 = env.obs_agent_two()
            if done:
                break
        for j in range(update_steps):
            losses.append(list(agent.update()))
        stats.append([i, total_reward, t + 1])

        if i % 20 == 0:
            print("{}: Done after {} steps. Reward: {}".format(i, t + 1, total_reward))
        if i % 100 == 0:
            eval_episodes = 20
            eval_agent(agent, env, mode=mode, render=False, episodes=eval_episodes)
            rewards = np.asarray(stats)[:, 1]
            mean_reward = np.mean(rewards)
            print(f'Evaluation at step {i}: Average reward over {eval_episodes} episodes: {mean_reward}')

    return losses, stats


def eval_agent(agent, env, mode='normal', episodes=100, render=False):
    stats = []
    max_steps = 500
    # set agent to evaluate mode
    agent.set_eval()

    if mode == ('train_defense' or 'train_shooting'):
        opponent = None
    else:
        opponent = h_env.BasicOpponent(weak=True)

    for i in range(episodes):

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
            if mode == ('train_defense' or 'train_shooting'):
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
        stats.append([i, total_reward, t + 1])
    return stats


def save_stats(stats, losses, path):
    if not os.path.exists('stats/'):
        os.makedirs('stats/')
    np.savetxt('stats/' + path + 'stats', stats)
    np.savetxt('stats/' + path + 'losses', losses)



def plot_rewards(stats):
    stats_np = np.asarray(stats)
    rewards = stats_np[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('rewards in evaluation')
    ax.plot(rewards)
    ax.set_ylabel('total_reward')
    ax.set_xlabel('num_episodes')
    plt.show()




def plot_loss_rewards(stats, losses, title=' ', kernel_size=25):
    stats_np = np.asarray(stats)
    rewards = stats_np[:, 1]
    kernel = np.ones(kernel_size) / kernel_size
    kernel_rewards = np.ones(100) / 100
    # smooth rewards
    rewards_smooth = np.convolve(rewards, kernel_rewards, mode='same')
    losses_q1 = np.asarray(losses)[:, 0]
    losses_q2 = np.asarray(losses)[:, 1]
    losses_actor = np.asarray(losses)[:, 2]
    # smooth losses
    losses_smooth_q1 = np.convolve(losses_q1, kernel, mode='same')
    losses_smooth_q2 = np.convolve(losses_q2, kernel, mode='same')

    losses_smooth_actor = np.convolve(losses_actor, kernel, mode='same')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title)

    axes[0, 0].plot(rewards)
    axes[0, 0].set_ylabel('total_reward')
    axes[0, 0].set_xlabel('num_episodes')

    axes[0, 1].plot(losses_actor)
    axes[0, 1].set_xlabel('num_train_steps')
    axes[0, 1].set_ylabel('actor loss')

    axes[1, 0].plot(losses_q1)
    axes[1, 0].set_xlabel('num_train_steps')
    axes[1, 0].set_ylabel('q1 loss')

    axes[1, 1].plot(losses_q2)
    axes[1, 1].set_xlabel('num_train_steps')
    axes[1, 1].set_ylabel('q2 loss')

    plt.show()


if __name__ == '__main__':
    main()
