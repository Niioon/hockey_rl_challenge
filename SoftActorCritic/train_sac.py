import numpy as np
import matplotlib.pyplot as plt
from soft_actor_critic import SacAgent
import laserhockey.hockey_env as h_env
import os
import argparse
import pickle
from eval_model import eval_agent


def main(args):
    mode = args.mode
    model_path = args.model_path
    episodes = args.episodes
    env = h_env.HockeyEnv()
    ac_space = env.action_space
    o_space = env.observation_space

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
        print('Weak Opponent: ', args.weak)

    stats_dict = train_agent(sac_agent, env, opponent, max_episodes=episodes, eval=True)
    rewards = np.asarray(stats_dict['stats'])[:, 1]
    mean_reward = np.mean(rewards)
    env.close()
    # store training specifications to keep track of total training time over different modes
    sac_agent.update_train_log(f'Trained in mode {mode} with weak={args.weak} opponent for {episodes} episodes, mean reward: {mean_reward}')
    save_name = f'sac_checkpoint_hockey_{mode}_et={sac_agent.automatic_entropy_tuning}_a={round(sac_agent.alpha.item(), 4)}_weak={args.weak}_e={episodes}_r={round(mean_reward, 4)}'
    save_stats(stats_dict, save_name)
    sac_agent.save_checkpoint(save_name=save_name)
    eval_agent(sac_agent, opponent, env, render=False, episodes=250)


def train_agent(agent, env, opponent, max_episodes=1000, eval=False):
    stats = []
    losses = []
    winners = []
    max_steps = 250
    update_steps = 32

    print(f'Simulating {max_episodes} episodes')
    for i in range(max_episodes):
        agent.set_train()
        total_reward = 0
        obs, _info = env.reset()
        obs_agent2 = env.obs_agent_two()
        for t in range(max_steps):
            done = False
            # get action of agent to be trained
            a1 = agent.act(obs)
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
        for j in range(update_steps):
            losses.append(list(agent.update()))
        stats.append([i, total_reward, t + 1])

        if i % 20 == 0:
            print("{}: Done after {} steps. Reward: {}".format(i, t + 1, total_reward))

        if i % 500 == 0 and eval:
            print(f'Evaluation at episode {i}')
            n_wins = eval_agent(agent, opponent, env, episodes=500, render=False)
            print('buffer size', agent.buffer.size)
            winners.append(n_wins)

    stats_dict = {'losses': losses, 'stats': stats, 'winners': winners}
    return stats_dict


def save_stats(stats_dict, save_name):
    if not os.path.exists('stats/'):
        os.makedirs('stats/')
    with open('stats/' + save_name + '.pkl', 'wb') as f:
        pickle.dump(stats_dict, f)


def plot_rewards(stats):
    stats_np = np.asarray(stats)
    rewards = stats_np[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('rewards in evaluation')
    ax.plot(rewards)
    ax.set_ylabel('total_reward')
    ax.set_xlabel('num_episodes')
    plt.show()


def plot_loss_rewards(stats, losses, title=' '):
    stats_np = np.asarray(stats)
    rewards = stats_np[:, 1]
    # smooth rewards
    losses_q1 = np.asarray(losses)[:, 0]
    losses_q2 = np.asarray(losses)[:, 1]
    losses_actor = np.asarray(losses)[:, 2]

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
    main(args)
