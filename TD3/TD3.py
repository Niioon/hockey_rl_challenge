import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
import memory as mem
import os

from ActorCritic import Actor, CriticTwin
from pink import ColoredNoiseProcess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)


class TD3Agent(object):
    """
    Implements a TD3 agent with NN function approximation.

    trick 1: Clipped Double-Q Learning: learn two Q-functions instead of one
    trick 2: "delayed" policy updates
    trick 3: target policy smoothing, add noise to target action
    """
    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self._observation_space = observation_space
        self._obs_dim=self._observation_space.shape[0]
        self._action_space = action_space
        # action space is divided by 2 because environment takes actions for both players 
        self._action_n = action_space.shape[0] / 2
        self._config = {
            #"eps": 0.05,            # Epsilon: noise strength to add to policy
            "discount": 0.99,
            "buffer_size": int(1e5),
            "batch_size": 128, #Paper: 100
            "learning_rate_actor": 0.001,
            "learning_rate_critic": 0.001,
            "hidden_sizes_actor": [256,256],      #[128,128],
            "hidden_sizes_critic": [256, 256],  #[128,128,64], oder 100,100
            "update_target_every": 2, # in DDPG: 100
            "tau": 0.005
        }
        self._config.update(userconfig)
        # self._eps = self._config['eps']
        self._tau = self._config["tau"]
        self.train_iter = 0
        
        # Pink Noise
        self.colored_noise = ColoredNoiseProcess(beta=1, 
                                                 scale=1, 
                                                 size=(int(self._action_n), 32))

        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        
        self.critic = CriticTwin(observation_dim=self._obs_dim,
                           action_dim=self._action_n,
                           output_dim=1,
                           hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_critic"],
                           activation_fun = torch.nn.ReLU(),
                            output_activation = None)
        self.critic_target = CriticTwin(observation_dim=self._obs_dim,
                                  action_dim=self._action_n,
                                  output_dim=1,
                                  hidden_sizes= self._config["hidden_sizes_critic"],
                                  learning_rate = self._config["learning_rate_critic"],
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = None)
        
        self.actor = Actor(input_dim=self._obs_dim,
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_dim=self._action_n,
                                  learning_rate=self._config["learning_rate_actor"],
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = torch.nn.Tanh())
        self.actor_target = Actor(input_dim=self._obs_dim,
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_dim=self._action_n,
                                         learning_rate=self._config["learning_rate_actor"],
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = torch.nn.Tanh())

        self._copy_nets()

    def _copy_nets(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def act(self, observation, eps=None):
        #if eps is None:
        #    eps = self._eps

        action = self.actor.predict(observation) + self.colored_noise.sample()  # action in -1 to 1 (+ noise)
        return action.clip(-1,1)

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.critic.state_dict(), self.actor.state_dict())

    def restore_state(self, state):
        self.critic.load_state_dict(state[0])
        self.actor.load_state_dict(state[1])
        self._copy_nets()

    def reset(self):
        self.colored_noise.reset()

    def soft_update_target_net(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update_target_net(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


    def train(self):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        self.train_iter+=1

        # sample from the replay buffer
        data=self.buffer.sample(batch=self._config['batch_size'])
        s = to_torch(np.stack(data[:,0])) # s_t
        a = to_torch(np.stack(data[:,1])) # a_t
        rew = to_torch(np.stack(data[:,2])[:,None]) # rew  (batchsize,1)
        s_prime = to_torch(np.stack(data[:,3])) # s_t+1
        done = to_torch(np.stack(data[:,4])[:,None]) # done signal  (batchsize,1)
        
        a_prime = self.actor_target.forward(s_prime)
        q1_prime, q2_prime = self.critic_target(torch.hstack([s_prime, a_prime]))  
        # trick 1
        target_q = torch.min(q1_prime, q2_prime)

        # target
        gamma=self._config['discount']
        td_target = rew + gamma * (1.0-done) * target_q

        # optimize the critic network
        q1_current, q2_current = self.critic(torch.hstack([s,a]))
        # loss
        critic_loss = F.mse_loss(q1_current, td_target) + F.mse_loss(q2_current, td_target)

        # gradient descent for critic network
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
            
        # trick 2: delayed update
        if self.train_iter % self._config['update_target_every'] == 0:

            # optimize actor network
            q = self.critic.Q1(torch.hstack([s, self.actor(s)]))
            actor_loss = -torch.mean(q)

            # gradient descent for actor network
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()    
                    
            # update target networks
            # self._copy_nets()
            self.soft_update_target_net(self.critic_target, self.critic, self._tau)
            self.soft_update_target_net(self.actor_target, self.actor, self._tau)
            #self.hard_update_target_net(self.critic_target, self.critic)
            #self.hard_update_target_net(self.actor_target, self.actor)
        
        return critic_loss.item() # actor_loss.item()


    def save_checkpoint(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if save_path is None:
            save_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(save_path))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'actor_target_state_dict': self.actor_target.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic.optimizer.state_dict(),
                    'actor_optimizer_state_dict': self.actor.optimizer.state_dict()}, save_path)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])    
