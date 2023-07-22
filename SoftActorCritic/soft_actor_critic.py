import memory as mem
import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from modules import ActorNetwork, CriticNetwork
import os


class SacAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(self, observation_space, action_space, **userconfig):
        # if not isinstance(observation_space, spaces.box.Box):
        #    raise UnsupportedSpace('Observation space {} incompatible ' \
        #                            'with {}. (Require: Box)'.format(observation_space, self))
        # if not isinstance(action_space, spaces.discrete.Discrete):
        #     raise UnsupportedSpace('Action space {} incompatible with {}.' \
        #                            ' (Reqire Discrete.)'.format(action_space, self))

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'

        self._observation_space = observation_space
        self._observation_dim = observation_space.shape[0]
        self._action_space = action_space
        # divide n_action by 2 because env expects actions for 2 players
        self._action_n = int(action_space.shape[0]/2)

        self.eval = False

        self._config = {
            "eps": 0.05,
            "discount": 0.95,
            "alpha": 0.2,
            "buffer_size": int(1e5),
            "batch_size": 256,
            "learning_rate": 0.0002,
            "target_update_interval": 20,
            "tau": 0.005
        }
        self._config.update(userconfig)

        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # used to keep track of when to update the target network
        self.train_iter = 0

        # factor for balancing influence of entropy term in loss function, can be learned by gradient descent
        self.alpha = self._config['alpha']
        # self.log_alpha =

        self.actor = ActorNetwork(self._observation_dim, self._action_n)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self._config['learning_rate'],
                                                eps=0.000001)

        self.q = CriticNetwork(self._observation_dim, self._action_n)
        self.target_q = CriticNetwork(self._observation_dim, self._action_n)

        self.q_optimizer = torch.optim.Adam(self.q.parameters(),
                                            lr=self._config['learning_rate'],
                                            eps=0.000001)
        self.q_loss_function = nn.MSELoss()
        # update target net once at the beginning
        self._hard_update_target_net()

    def _hard_update_target_net(self):
        """
        Copies the parameters from the trained Critic Network to the Target Critic network
        :return: None
        """
        self.target_q.load_state_dict(self.q.state_dict())

    def _soft_update_target_net(self):
        """
        Updates the target network parameters by weighted averaging with the trained Critic Network Parameters
        :return: None
        """
        tau = self._config['tau']
        for target_param, param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def set_eval(self):
        self.eval = True

    def set_train(self):
        self.eval = False

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
            # action, _, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def update(self):
        self.train_iter += 1
        # sample batch from replay buffer
        # print(self.train_iter)
        data = self.buffer.sample(self._config['batch_size'])

        state = torch.FloatTensor(np.stack(data[:, 0])).to(self.device)  # (batchzize, ob_dim)
        action = torch.FloatTensor(np.stack(data[:, 1])).to(self.device)
        next_state = torch.FloatTensor(np.stack(data[:, 3])).to(self.device)  # (batchzize, ob_dim)
        reward = torch.FloatTensor(np.stack(data[:, 2])[:, None]).to(self.device)  # (batchsize,1)
        term_s = torch.FloatTensor((np.stack(data[:, 4]).reshape((-1, 1)))).to(
            self.device)  # indicates if state is terminal

        # no gradients needed for target computation
        with torch.no_grad():
            # sample actions for next states, needed for computing the entropy and q values
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            # get q value predictions of next state from target network
            q1_next_target, q2_next_target = self.target_q(next_state, next_state_action)
            # use double q trick and take min and subtract scaled log_probs (entropy=-1*log_prob)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            # compute targets for q-functions, reward + not_terminal * gamma * min_q
            q_target = (reward + (1 - term_s) * self._config['discount'] * min_q_next_target).squeeze()

        # Optimize Critic Networks
        # compute current q values
        q1, q2 = self.q(state, action)

        # loss
        q1_loss = self.q_loss_function(q1.squeeze(), q_target)
        q2_loss = self.q_loss_function(q2.squeeze(), q_target)
        q_loss = q1_loss + q2_loss

        # perform gradient descent for q networks
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Optimize Actor Network
        # get suggested actions and log_probs for states
        pi, log_pi, _ = self.actor.sample(state)
        # get q-values for suggested actions
        q1_pi, q2_pi = self.q(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        # policy_loss = (min_q_pi - self.alpha * log_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Optimize alpha
        # TODO

        if self.train_iter % self._config['target_update_interval'] == 0:
            self._soft_update_target_net()

        return q1_loss.item(), q2_loss.item(), policy_loss.item()

    def save_checkpoint(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if save_path is None:
            save_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(save_path))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.q.state_dict(),
                    'critic_target_state_dict': self.target_q.state_dict(),
                    'critic_optimizer_state_dict': self.q_optimizer.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict()}, save_path)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.q.load_state_dict(checkpoint['critic_state_dict'])
            self.target_q.load_state_dict(checkpoint['critic_target_state_dict'])
            self.q_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

