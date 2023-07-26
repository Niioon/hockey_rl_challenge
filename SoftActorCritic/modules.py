import torch
from torch import nn
import numpy as np


class CriticNetwork(nn.Module):
    """
    Implements a critic network for the soft actor critic algorithm
    The critic network takes a state action pair as input and computes an estimate of the  corresponding q-value.
    Since Sac uses the minimum of two q value estimates to reduce the overestimation bias of q-learning, this class
    implements two parallel running q-networks
    """

    def __init__(self, observation_dim, action_dim, device,
                 hidden_sizes=[128, 128]):
        super(CriticNetwork, self).__init__()

        self.input_size = observation_dim + action_dim
        self.hidden_sizes = hidden_sizes
        self.output_size = 1

        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.q_1 = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.q_2 = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        # output layer does not have an activation function
        self.activations = [torch.nn.ReLU() for _ in self.q_1[:-1]]

        if device.type == 'cuda':
            self.cuda()

    def forward(self, state, action):

        x_1 = torch.cat([state, action], 1)
        x_2 = x_1
        for layer, activation_fun in zip(self.q_1[:-1], self.activations):
            x_1 = activation_fun(layer(x_1))
        x_1 = self.q_1[-1](x_1)

        for layer, activation_fun in zip(self.q_2[:-1], self.activations):
            x_2 = activation_fun(layer(x_2))
        x_2 = self.q_2[-1](x_2)

        return x_1, x_2


class ActorNetwork(nn.Module):
    """
    Implements an actor network for the soft actor critic algorithm
    The actor network represents the policy and computes a probability distribution over actions for a given state.
    """

    def __init__(self, observation_dim, action_dim, device,
                 hidden_sizes=[128, 128], action_space=None, learning_rate=0.0002):

        super(ActorNetwork, self).__init__()

        self.input_size = observation_dim
        self.hidden_sizes = hidden_sizes
        self.output_size = action_dim

        self.device = device

        self.epsilon = 1e-6

        layer_sizes = [self.input_size] + self.hidden_sizes
        # output size of previous layer is input size of next layer, achieved by zipping the lists
        self.layers = torch.nn.ModuleList([nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [torch.nn.ReLU() for _ in self.layers]

        # 2 output layers, one produces the mean for each action and one the log of standard deviation
        # these are used to sample an action in a continuous space
        self.mean = nn.Linear(hidden_sizes[-1], self.output_size)
        self.log_std = nn.Linear(hidden_sizes[-1], self.output_size)

        if device.type == 'cuda':
            self.cuda()

        self.loss = torch.nn.SmoothL1Loss()

        # action rescaling, needed if action space is not [-1, 1], scales action space using linear transformation
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(self.device)
            self.action_bias = torch.tensor(0.).to(self.device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(self.device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=10)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # use re-parameterization trick to make computation differentiable with regard to mean and std
        # (mean + std * N(0,1)) instead of sampling with mean and std
        x = normal.rsample()
        # squash action with tanh function to [-1, 1]
        y = torch.tanh(x)
        # rescale action in case the bounds are not [-1, 1]
        action = y * self.action_scale + self.action_bias

        # get log probs for entropy computation
        log_prob = normal.log_prob(x)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + self.epsilon)
        # sum up log_probs, log_prob = -1*Entropy
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
