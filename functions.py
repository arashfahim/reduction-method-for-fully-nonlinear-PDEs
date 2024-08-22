import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''Class to create functions which take parameters from a dictionary'''
class solution(object):
    def __init__(self,params):
        self.dim=params['dim']#2
        self.nu = params['nu']
        self.kappa = params['kappa'][0:self.dim]
        self.theta = params['theta'][0:self.dim]
        self.eta = params['eta']
        self.lb = params['lb'][0:self.dim]
        self.T = params['T']
        
        
class exp_solution(solution):
    def __init__(self,params,alpha):
        self.alpha = alpha
        super(exp_solution, self).__init__(params)
    def __call__(self,x):
        # print(self.alpha)
        return torch.tensor([1.])-torch.exp(-self.eta*x[:,1]+self.alpha*(self.T-x[:,0])).to(device)     

class zero_solution(solution):
    def __init__(self,params):
        super(zero_solution, self).__init__(params)
    def __call__(self,x):
        # print(self.alpha)
        return torch.zeros([x.shape[0],1]).to(device) 
    
class time_solution(solution):
    def __init__(self,params,constant):
        self.constant = constant
        super(time_solution, self).__init__(params)
    def __call__(self,x):
        return (self.T-x[:,0].unsqueeze(-1))*self.constant*torch.ones([x.shape[0],1]).to(device)    

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