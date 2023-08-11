import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
import memory as mem
import os
import warnings

from ActorCritic import Actor, CriticTwin
from cnrl import ColoredNoiseProcess


class TD3Agent(object):
    """
    Implements a TD3 agent with NN function approximation.

    trick 1: Clipped Double-Q Learning: learn two Q-functions instead of one
    trick 2: "delayed" policy updates
    trick 3: target policy smoothing, add noise to target action
    """
    def __init__(self, observation_space, action_space, **userconfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        self._observation_space = observation_space
        self._obs_dim=self._observation_space.shape[0]
        self._action_space = action_space
        # action space is divided by 2 because environment takes actions for both players 
        self._action_n = action_space.shape[0] / 2
        
        self.eval = False

        self._config = {
            "discount": 0.95,
            "buffer_size": int(1e5)*3,
            "batch_size": 128, 
            "learning_rate_actor": 0.0002,
            "learning_rate_critic": 0.0002,
            "hidden_sizes_actor": [256, 256],     
            "hidden_sizes_critic": [256, 256],  
            "update_target_every": 1, 
            "tau": 0.0025,
            "noise": 0.2,
            "noise_clip": 0.5
        }
        self._config.update(userconfig)
        self._tau = self._config["tau"]
        self.train_iter = 0
        self.train_log = []
        
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
                            device = self.device)
        self.critic_target = CriticTwin(observation_dim=self._obs_dim,
                                  action_dim=self._action_n,
                                  output_dim=1,
                                  hidden_sizes= self._config["hidden_sizes_critic"],
                                  learning_rate = self._config["learning_rate_critic"],
                                  activation_fun = torch.nn.ReLU(),
                                  device = self.device)
        
        self.actor = Actor(input_dim=self._obs_dim,
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_dim=self._action_n,
                                  learning_rate=self._config["learning_rate_actor"],
                                  activation_fun = torch.nn.ReLU(),
                                  device = self.device)
        self.actor_target = Actor(input_dim=self._obs_dim,
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_dim=self._action_n,
                                         learning_rate=self._config["learning_rate_actor"],
                                         activation_fun = torch.nn.ReLU(),
                                         device = self.device)

        
    
    def set_eval(self):
        self.eval = True
    
    def set_train(self):
        self.eval = False

    def act(self, observation):
    
        observation = torch.FloatTensor(observation).to(self.device)
        action = self.actor.forward(observation)
        action = action.detach().cpu().numpy()[0]

        if not eval:
            action = (action + self.colored_noise.sample())  # action in -1 to 1 (+ noise))
        return action.clip(-1,1)

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def update_train_log(self, entry):
        self.train_log.append(entry)

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
        self.train_iter+=1

        # sample from the replay buffer
        data=self.buffer.sample(batch=self._config['batch_size'])
        s = torch.FloatTensor(np.stack(data[:, 0])).to(self.device) # s_t
        a = torch.FloatTensor(np.stack(data[:, 1])).to(self.device) # a_t
        rew = torch.FloatTensor(np.stack(data[:, 2])[:, None]).to(self.device) # rew  (batchsize,1)
        s_prime = torch.FloatTensor(np.stack(data[:, 3])).to(self.device) # s_t+1
        done = torch.FloatTensor((np.stack(data[:, 4]).reshape((-1, 1)))).to(
            self.device) # done signal  (batchsize,1)
        
        noise = torch.FloatTensor(a.cpu()).data.normal_(0, self._config['noise']).to(self.device)
        noise = noise.clamp(-self._config['noise_clip'], self._config['noise_clip'])
        # trick 3: target policy smoothing
        a_prime = (self.actor_target.forward(s_prime).to(self.device) + noise).clamp(-1,1)
        q1_prime, q2_prime = self.critic_target(torch.hstack([s_prime, a_prime]))  
        # trick 1: clipped double Q-learning
        target_q = torch.min(q1_prime, q2_prime).to(self.device)

        # target
        gamma=self._config['discount']
        td_target = rew + gamma * (1.0-done) * target_q
        td_target = td_target.to(self.device) 

        # optimize the critic network
        q1_current, q2_current = self.critic(torch.hstack([s,a]))
        q1_current = q1_current.to(self.device)
        q2_current = q2_current.to(self.device)
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
            self.soft_update_target_net(self.critic_target, self.critic, self._tau)
            self.soft_update_target_net(self.actor_target, self.actor, self._tau)
            #self.hard_update_target_net(self.critic_target, self.critic)
            #self.hard_update_target_net(self.actor_target, self.actor)
        
        return critic_loss.item() 


    def save_checkpoint(self, save_name=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if save_name is None:
            save_path = "checkpoints/td3_checkpoint"
        else:
            save_path = "checkpoints/" + save_name
        print('Saving models to {}'.format(save_path))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'actor_target_state_dict': self.actor_target.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic.optimizer.state_dict(),
                    'actor_optimizer_state_dict': self.actor.optimizer.state_dict(),
                    'train_iter': self.train_iter, 
                    'config': self._config, 
                    'train_log': self.train_log,
                    }, save_path)

        buffer_path = save_path + '_buffer'
        print('Saving buffer to {}'.format(buffer_path))
        torch.save({'buffer_transitions': self.buffer.get_all_transitions()}, buffer_path)

    def load_checkpoint(self, ckpt_path, load_buffer=True):
        print('Loading models from {}'.format(ckpt_path))
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device))
            #checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic.optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])    
            self.train_iter = checkpoint['train_iter']
            self._config = checkpoint['config']
            self.train_log = checkpoint['train_log']

            if load_buffer:
                buffer_path = ckpt_path + '_buffer'
                if os.path.isfile(buffer_path):
                    buffer_dict = torch.load(buffer_path, map_location=torch.device(self.device))
                    self.buffer.clone_old_transitions(buffer_dict['buffer_transitions'])
                else:
                    warnings.warn('no stored buffer found for the given checkpoint path, training will resume with empty buffer')

        else:
            raise FileNotFoundError('No checkpoint file under the given path')