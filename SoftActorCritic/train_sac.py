import numpy as np
import torch
import matplotlib.pyplot as plt
from soft_actor_critic import SacAgent
import laserhockey.hockey_env as h_env


def main():
    env = h_env.HockeyEnv()
    ac_space = env.action_space
    o_space = env.observation_space
    # print(ac_space, o_space)
    # print(ac_space.shape[0])
    sac_agent = SacAgent(observation_space=o_space, action_space=ac_space)

    # TRAIN DEFENSE
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    losses, stats = train_agent(sac_agent, env, mode='train_defense', max_episodes=500)
    env.close()

    plot_loss_rewards(stats, losses, title='Losses and Rewards for Defense Training')

    sac_agent.save_checkpoint('hockey', 'defense')


def train_agent(agent, env, mode='normal', max_episodes=1000):
    stats = []
    losses = []
    max_steps = 500
    update_steps = 64

    if mode == ('train_defense' or 'train_shooting'):
        opponent = None
    else:
        opponent = env.BasicOpponent(weak=True)

    for i in range(max_episodes):
        # print("Starting a new episode")
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
            losses.extend(agent.update())
        stats.append([i, total_reward, t + 1])

        if (i - 1) % 20 == 0:
            print("{}: Done after {} steps. Reward: {}".format(i, t + 1, total_reward))
    return losses, stats


def plot_loss_rewards(stats, losses, title=' ', kernel_size=25):
    stats_np = np.asarray(stats)
    rewards = stats_np[:, 1]
    kernel = np.ones(kernel_size) / kernel_size
    kernel_rewards = np.ones(100)/100
    # smooth rewards
    rewards_smooth = np.convolve(rewards, kernel_rewards, mode='same')
    losses_np = np.asarray(losses)
    # smooth losses
    losses_smooth = np.convolve(losses_np, kernel, mode='same')

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title)
    axes[0].plot(rewards_smooth)
    axes[0].set_ylabel('total_reward')
    axes[0].set_xlabel('num_episodes')

    axes[1].plot(losses_smooth)
    axes[0].set_xlabel('num_episodes')
    axes[1].set_ylabel('l1 loss')

    plt.show()


if __name__ == '__main__':
    main()