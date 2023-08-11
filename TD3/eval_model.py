import numpy as np
from TD3 import TD3Agent
import laserhockey.hockey_env as h_env
import argparse

# script for evaluation
# For evaluating the models that participated in the tournament use the following path:
# trained_model\td3_checkpoint_hockey_tau=0.0025_normal_2x256_weak=False_e=25000_r=7.9188

def main(args):
    env = h_env.HockeyEnv()
    td3_agent = TD3Agent(env.observation_space, env.action_space)
    
    # load model parameters
    td3_agent.load_checkpoint(args.model_path, load_buffer=False)

    print('mode ', args.mode)
    # check loading
    print('training_iterations: ', td3_agent.train_iter)
    print('training_log', td3_agent.train_log)

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

    _ = eval_agent(td3_agent, opponent, env, episodes=args.episodes, render=args.render)



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
            # get action of agent to be trained
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
        winner.append(_info['winner'])
        stats.append([i, total_reward, t + 1])
    rewards = np.asarray(stats)[:, 1]
    mean_reward = np.mean(rewards)
    winner = np.asarray(winner)
    n_win = np.count_nonzero(winner == 1)
    n_l = np.count_nonzero(winner == -1)
    n_draw = np.count_nonzero(winner == 0)
    print(f'Agent won {n_win} games, lost {n_l} games, draw {n_draw} games')
    print(f'Win/Loss+Win Ratio {n_win/(n_l+n_win)}')
    print(f'Average reward over {episodes} episodes: {mean_reward}')
    return [n_win, n_l, n_draw]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a SoftActorCritic RL-Agent in the hockey environment')
    parser.add_argument('--model_path',
                        help='loads an already existing model from the specified path instead of initializing a new one')
    parser.add_argument('--mode', choices=['defense', 'shooting', 'normal'], default='normal',
                        help='game mode for evaluation')
    parser.add_argument('--weak', action="store_true",
                        help='difficulty of the opponent in the normal mode, no influence in other modes')
    parser.add_argument('--render',  action="store_true", help='if true render env')
    parser.add_argument('-e', '--episodes', type=int, default=1000,
                        help='number of training episodes')
    args = parser.parse_args()
    main(args)
