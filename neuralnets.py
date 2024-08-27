import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Solution at t=0
class Ynet(nn.Module): #input [M,D+1]   #output [M,1]
    def __init__(self,pde,sim):
        super(Ynet, self).__init__()
        dim = pde['dim']
        num_neurons = sim['num_neurons']
        self.linear_stack = nn.Sequential(
            nn.Linear(dim, num_neurons),
            # nn.BatchNorm1d(num_features=8),# We should never use Batch mormalization in these type of problems when the input and scale back to a smaller region. The input is normalized with a different scale than the training data and out functions are going to be screwed.
            nn.Tanh(),
            nn.Linear(num_neurons, num_neurons),
            # nn.BatchNorm1d(num_features=8),
            nn.Tanh(),
            nn.Linear(num_neurons,1),
        )
    def forward(self, x):
        logits = self.linear_stack(x)
        return logits  
    
    
# derivative of the solution at all times
class Znet(nn.Module): #input [M,D+1]   #output [M,1]
    def __init__(self,pde,sim):
        dim = pde['dim']
        num_neurons = sim['num_neurons']
        super(Znet, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(dim+1, num_neurons),
            nn.Tanh(),
            nn.Linear(num_neurons, num_neurons),
            # nn.BatchNorm1d(num_features=20),
            nn.Tanh(),
            nn.Linear(num_neurons,dim),
        )
    def forward(self, x):
        logits = self.linear_stack(x)
        return logits#.reshape([dim,dim])  
    
    
# Value of the solution at all times
class Ytnet(nn.Module): #input [M,D+1]   #output [M,1]
    def __init__(self,pde,sim):
        dim = pde['dim']
        num_neurons = sim['num_neurons']
        super(Ytnet, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(dim+1, num_neurons),
            # nn.BatchNorm1d(num_features=8),
            nn.Tanh(),
            nn.Linear(num_neurons, num_neurons),
            # nn.BatchNorm1d(num_features=8),
            nn.Tanh(),
            nn.Linear(num_neurons,1),
        )
    def forward(self, x):
        logits = self.linear_stack(x)
        return logits    
           
           
# Value of updated sigma
class sigmanet(nn.Module): #input [M,D+1]   #output [M,1]
    def __init__(self,pde,sim):
        dim = pde['dim']
        num_neurons = sim['num_neurons']
        super(Ytnet, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(dim+1, num_neurons),
            # nn.BatchNorm1d(num_features=8),
            nn.Tanh(),
            nn.Linear(num_neurons, num_neurons),
            # nn.BatchNorm1d(num_features=8),
            nn.Tanh(),
            nn.Linear(num_neurons,1),
        )
    def forward(self, x):
        logits = self.linear_stack(x)
        return logits               