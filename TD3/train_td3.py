import numpy as np
import matplotlib.pyplot as plt
from TD3 import TD3Agent
import laserhockey.hockey_env as h_env
import os
import argparse
import pickle
from eval_model import eval_agent


def main(opts):
    mode = opts.mode
    print(mode)
    model_path = opts.model_path
    episodes = opts.episodes
    env = h_env.HockeyEnv()
    td3_agent = TD3Agent(env.observation_space, env.action_space)
    
    # load model parameters if path is specified
    if model_path is not None:
        td3_agent.load_checkpoint(model_path)

    if mode == 'defense':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
        opponent = None
    elif mode == 'shooting':
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        opponent = None
    elif mode == 'normal':
        env = h_env.HockeyEnv()
        opponent = h_env.BasicOpponent(weak=opts.weak)
        print('Weak Opponent: ', opts.weak)
    
    stats_dict = train_agent(td3_agent, env, opponent, max_episodes=episodes, eval=True)
    rewards = np.asarray(stats_dict['stats'])[:, 1]
    mean_reward = np.mean(rewards)
    print(f'average reward {mean_reward}')
    env.close()
   
    # store training specifications to keep track of total training time over different modes
    td3_agent.update_train_log(f'Trained in mode {mode} with weak={opts.weak} opponent for {episodes} episodes, mean reward: {mean_reward}')

    save_name = f'td3_checkpoint_hockey_2x256_{mode}_weak={opts.weak}_e={episodes}_r={round(mean_reward, 4)}'
    save_stats(stats_dict, save_name)
    td3_agent.save_checkpoint(save_name=save_name)
    eval_agent(td3_agent, opponent, env, render=False, episodes=250)


def train_agent(agent, env, opponent, max_episodes=1000, eval=False):
    stats = []
    losses = []
    winners = []
    max_steps = 250
    max_episodes = opts.episodes
    update_steps = 32
    episode_counter = 1

    print(f'Simulating {max_episodes} episodes')
    #for i in range(max_episodes):
    while episode_counter <= max_episodes:
        agent.set_train
        total_reward = 0
    
        obs, _info = env.reset()
        obs_agent2 = env.obs_agent_two()
        
        for t in range(max_steps):
            #env.render()
            done = False
            # get action of agent to be trained
            a1 = agent.act(obs)
            # get action of opponent
            if opponent is None:
                a2 = [0, 0, 0, 0]
            else:
                a2 = opponent.act(obs_agent2)

            (obs_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))

            # Adapted reward 
            #adapted_reward = 0.9 * (reward - _info['reward_closeness_to_puck']) + 1.1 * _info['reward_closeness_to_puck']
            #total_reward += adapted_reward
            #agent.store_transition((obs, a1, adapted_reward, obs_new, done))

            # Normal reward 
            total_reward += reward
            agent.store_transition((obs, a1, reward, obs_new, done))
            obs = obs_new
            obs_agent2 = env.obs_agent_two()

         
            if done:
               break
            
        for j in range(update_steps):
            losses.append(agent.train())
        stats.append([episode_counter, total_reward, t + 1])

        if episode_counter % 20 == 0:
            print("{}: Done after {} steps. Reward: {}".format(episode_counter, t + 1, total_reward))
        
        if episode_counter % 500 == 0 and eval:
            print(f'Evaluation at episode {episode_counter}')
            n_wins = eval_agent(agent, opponent, env, episodes=500, render=False)
            print('buffer size', agent.buffer.size)
            winners.append(n_wins)

        episode_counter +=1
    
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

def plot_loss_rewards(stats, losses, title=' ', kernel_size=25):
    stats_np = np.asarray(stats)
    rewards = stats_np[:, 1]
    kernel = np.ones(kernel_size) / kernel_size
    kernel_rewards = np.ones(100) / 100
    # smooth rewards
    rewards_smooth = np.convolve(rewards, kernel_rewards, mode='same')
    losses_q1 = np.asarray(losses)
    # smooth losses
    losses_smooth_q1 = np.convolve(losses_q1, kernel, mode='same')

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(title)

    axes[0, 0].plot(rewards_smooth)
    axes[0, 0].set_ylabel('total_reward')
    axes[0, 0].set_xlabel('num_episodes')

    axes[1, 0].plot(losses_smooth_q1)
    axes[1, 0].set_xlabel('num_train_steps')
    axes[1, 0].set_ylabel('q1 loss')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trains a TD3 RL-Agent in the hockey environment')
    parser.add_argument('--mode', choices=['defense', 'shooting', 'normal'], type=str, default='normal',
                    help='game mode to train in')
    parser.add_argument('--model_path',
                    help='loads an already existing model from the specified path instead of initializing a new one')
    parser.add_argument('-e', '--episodes', type = int, default=1000, help='number of training episodes')
    parser.add_argument('--eval',  action="store_true", help='if true evaluates agent in regular intervals to keep track of performance')
    parser.add_argument('--weak', action="store_true",
                        help='difficulty of the opponent in the normal mode, no influence in other modes')
    opts = parser.parse_args()

    main(opts)