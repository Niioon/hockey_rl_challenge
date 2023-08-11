import numpy as np
import torch 

class Actor(torch.nn.Module):

    """ 
    Implements an actor network for the TD3 algorithm.
    The actor network network represents the policy.
    The TD3 algorithm uses an actor and an actor target network. 
    """

    def __init__(self, input_dim, output_dim, hidden_sizes, learning_rate, activation_fun, device):
        super(Actor, self).__init__()
        
        self.device = device
        
        self.input_dim = input_dim
        self.output_dim = int(output_dim)
        self.hidden_sizes  = hidden_sizes
        self.learning_rate = learning_rate
        
        layer_sizes = [self.input_dim] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ activation_fun for l in  self.layers ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_dim)
        # activation function of the output layer
        self.output_activation = torch.nn.Tanh()
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        if device.type == 'cuda':
            self.cuda()
    

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = x.to(self.device)
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        # tanh output activation since the action space lies between [-1, 1]
        return self.output_activation(self.readout(x))

            

class CriticTwin(torch.nn.Module):

    """ 
    Implements a critic network for the TD3 algorithm.
    The critic network takes a state action pair as input and computes an estimate of the
    corresponding Q-value.
    Since TD3 uses the minimum of two Q-value estimates to reduce the overestimation bias 
    of Q-learning, this class implements two parallel running Q-networks.
    There are two separate target target critic networks.
    """

    def __init__(self, observation_dim, action_dim, output_dim, hidden_sizes, learning_rate, activation_fun, device):
        super(CriticTwin, self).__init__()

        self.device = device
        
        self.input_dim = int(observation_dim + action_dim)
        self.hidden_sizes  = hidden_sizes
        self.output_dim  = output_dim
        self.learning_rate = learning_rate
    
        layer_sizes = [self.input_dim] + self.hidden_sizes
        
        self.layers1 = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.layers2 = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        # no output activation function in the critic network
        self.activations1 = [ activation_fun for l in  self.layers1 ]
        self.activations2 = [ activation_fun for l in  self.layers2 ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_dim)

        self.optimizer=torch.optim.Adam(self.parameters(),
                                       lr=learning_rate,
                                       eps=0.000001)
        self.loss = torch.nn.MSELoss()
        if device.type == 'cuda':
            self.cuda()


    def forward(self, x):
        x1 = self.Q1(x)
        x = x.to(self.device)
        
        for layer,activation_fun in zip(self.layers2, self.activations2):
            x = activation_fun(layer(x))
        
        x = self.readout(x)
        
        return x1, x
                         
        
    def Q1(self, x1):
    
        x1 = x1.to(self.device)
        for layer,activation_fun in zip(self.layers1, self.activations1):
            x1 = activation_fun(layer(x1))
        
        return self.readout(x1)
