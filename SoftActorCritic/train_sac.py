import numpy as np
import torch
import matplotlib.pyplot as plt
from soft_actor_critic import SacAgent
import laserhockey.hockey_env as h_env
import os
import argparse
import time
from eval_model import eval_agent


def main(args):
    mode = args.mode
    model_path = args.model_path
    episodes = args.episodes
    env = h_env.HockeyEnv()
    ac_space = env.action_space
    o_space = env.observation_space
    # print(ac_space, o_space)
    # print(ac_space.shape[0])
    sac_agent = SacAgent(observation_space=o_space, action_space=ac_space)
    # load model parameters if path is specified
    if model_path is not None:
        sac_agent.load_checkpoint(model_path)

    if mode == 'defense':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
        opponent = None
    elif mode == 'shooting':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        opponent = None
    elif mode == 'normal':
        env = h_env.HockeyEnv()
        opponent = h_env.BasicOpponent(weak=args.weak)
        print(args.weak)


    losses, stats = train_agent(sac_agent, env, opponent, mode=mode, max_episodes=episodes, eval=True)
    rewards = np.asarray(stats)[:, 1]
    mean_reward = np.mean(rewards)
    print(f'average reward {mean_reward}')
    env.close()
    # store training specifications to keep track of total training time over different modes
    sac_agent.update_train_log(f'Trained in mode {mode} with weak={args.weak} opponent for {episodes} episodes, mean reward: {mean_reward}')
    save_stats(np.asarray(stats), np.asarray(losses), mode + str(episodes))
    # plot_loss_rewards(stats, losses, title='Losses and Rewards for Defense Training')
    #
    sac_agent.save_checkpoint('hockey', f'{mode}_weak={args.weak}_e={episodes}_r={round(mean_reward, 4)}')
    eval_agent(sac_agent, opponent, env, render=False, episodes=250)

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


def train_agent(agent, env, opponent, mode='normal', max_episodes=1000, eval=False):
    stats = []
    losses = []
    max_steps = 250
    update_steps = 32

    print(f'Simulating {max_episodes} episodes')
    start_time = time.time()
    for i in range(max_episodes):
        agent.set_train()
        total_reward = 0
        obs, _info = env.reset()
        obs_agent2 = env.obs_agent_two()
        for t in range(max_steps):
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
            agent.store_transition((obs, a1, reward, obs_new, done))
            obs = obs_new
            obs_agent2 = env.obs_agent_two()
            if done:
                break
        # print(f'time needed : {time.time() - start_time}')

        start_time = time.time()
        # print(f'training {update_steps} batches')
        for j in range(update_steps):
            losses.append(list(agent.update()))
        stats.append([i, total_reward, t + 1])
        # print(f'time needed : {time.time() - start_time}')

        if i % 20 == 0:
            print("{}: Done after {} steps. Reward: {}".format(i, t + 1, total_reward))
            print('buffer size', agent.buffer.size)

        if i % 200 == 0 and eval:
            print(f'Evaluation at episode {i}')
            stats, winner = eval_agent(agent, opponent, env, episodes=250, render=False)

    return losses, stats


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
    parser = argparse.ArgumentParser(description='Trains a SoftActorCritic RL-Agent in the hockey environment')

    parser.add_argument('-e', '--episodes', type=int, default=1000,
                        help='number of training episodes')
    parser.add_argument('--model_path',
                        help='loads an already existing model from the specified path instead of initializing a new one')
    parser.add_argument('--mode', choices=['defense', 'shooting', 'normal'], default='normal',
                        help='game mode to train in')
    parser.add_argument('--eval',  action="store_true", help='if true evaluates agent in regular intervals to keep track of performance')
    parser.add_argument('--weak', action="store_true",
                        help='difficulty of the opponent in the normal mode, no influence in other modes')

    args = parser.parse_args()
    print(args)
    main(args)
