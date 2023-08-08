import numpy as np
import sys
sys.path.append('.\SAC')
sys.path.append('.\TD3')
# print(sys.path)
from td3.TD3 import TD3Agent
from SAC.soft_actor_critic import SacAgent
import laserhockey.hockey_env as h_env
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client
import argparse

class RemoteAgent(TD3Agent, RemoteControllerInterface):

    def __init__(self, agent, identifier):
        
         self.agent = agent
         RemoteControllerInterface.__init__(self, identifier)

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        return self.agent.act(obs)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote Agent for tournament')
    parser.add_argument('identifier', choices=['td3', 'sac'])
    parser.add_argument('path',
                        help='loads an already existing model from the path')
    args = parser.parse_args()
    print(args.identifier)
    env = h_env.HockeyEnv()
    if args.identifier == 'td3':
        agent = TD3Agent(env.observation_space, env.action_space)
        agent.load_checkpoint(args.path, load_buffer=False)
    elif args.identifier == 'sac':
        agent = SacAgent(env.observation_space, env.action_space)
        agent.load_checkpoint(args.path, load_buffer=False)
    else:
        raise ValueError('Unknown agent type, choose one of [td3, sac]')
    controller = RemoteAgent(agent, args.identifier)

    # Play n (None for an infinite amount) games and quit
    client = Client(username='LaserLearningLunatics', # Testuser
                    password='Aeriefe8da',
                    controller=controller, 
                    output_path='/tmp/ALRL2020/client/LaserLearningLunatics', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(username='user0', 
    #                 password='1234',
    #                 controller=controller, 
    #                 output_path='/tmp/ALRL2020/client/user0',
    #                )
