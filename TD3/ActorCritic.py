import numpy as np
import torch 

class Actor(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, learning_rate, activation_fun, output_activation):
        super(Actor, self).__init__()

        self.input_dim = input_dim
        #self.action_dim = action_dim / 2 #?
        self.output_dim = int(output_dim)
        self.hidden_sizes  = hidden_sizes
        self.learning_rate = learning_rate
        
        layer_sizes = [self.input_dim] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ activation_fun for l in  self.layers ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_dim)

        self.output_activation = torch.nn.Tanh()
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        
        
    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        return self.output_activation(self.readout(x))
                
    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
            

class CriticTwin(torch.nn.Module):
    def __init__(self, observation_dim, action_dim, output_dim, hidden_sizes, learning_rate, activation_fun, output_activation):

        super(CriticTwin, self).__init__()
    
        self.input_dim = int(observation_dim + action_dim)
        self.hidden_sizes  = hidden_sizes
        self.output_dim  = output_dim
        self.learning_rate = learning_rate
    
        layer_sizes = [self.input_dim] + self.hidden_sizes
        # Meike
        self.layers1 = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.layers2 = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations1 = [ activation_fun for l in  self.layers1 ]
        self.activations2 = [ activation_fun for l in  self.layers2 ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_dim)

        self.optimizer=torch.optim.Adam(self.parameters(),
                                       lr=learning_rate,
                                       eps=0.000001)
        self.loss = torch.nn.MSELoss()


    def forward(self, x):
        # Meike
        x1 = self.Q1(x)
        
        for layer,activation_fun in zip(self.layers2, self.activations2):
            x = activation_fun(layer(x))
        
        x = self.readout(x)
        
        return x1, x
                         
        
    # Meike
    def Q1(self, x1):
        for layer,activation_fun in zip(self.layers1, self.activations1):
            x1 = activation_fun(layer(x1))
        
        return self.readout(x1)


    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
