import numpy as np
import laserhockey.hockey_env as h_env
import gymnasium as gym
from importlib import reload
import time

np.set_printoptions(suppress=True)



def main():
    env = h_env.HockeyEnv()


    print('!')
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    _ = env.render()

    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()

    # for _ in range(600):
    #     env.render(mode="human")
    #     a1 = np.random.uniform(-1, 1, 4)
    #     a2 = np.random.uniform(-1, 1, 4)
    #     obs, r, d, t, info = env.step(np.hstack([a1, a2]))
    #     obs_agent2 = env.obs_agent_two()
    #     if d: break

    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)

    o, info = env.reset()
    print(env.action_space)
    _ = env.render()

    env = h_env.HockeyEnv()

    o, info = env.reset()
    _ = env.render()
    player1 = h_env.BasicOpponent(weak=False)
    player2 = h_env.BasicOpponent()

    obs_buffer = []
    reward_buffer = []
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    for _ in range(250):
        env.render()
        a1 = player1.act(obs)
        a2 = player2.act(obs_agent2)
        obs, r, d, _, info = env.step(np.hstack([a1, a2]))
        obs_buffer.append(obs)
        reward_buffer.append(r)
        obs_agent2 = env.obs_agent_two()
        if d: break
    obs_buffer = np.asarray(obs_buffer)
    reward_buffer = np.asarray(reward_buffer)


if __name__ == '__main__':
    main()
