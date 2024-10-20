import torch
from torch import nn # type: ignore
import torch.optim as optim
import numpy as np
from derivation import Grad, Grad_Hess
# from neuralnets import sigmanet
from time import time
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Frobenius(A,B):
    return torch.vmap(torch.trace)(torch.bmm(A,torch.transpose(B,1,2)))


'''Implements the coefficeints based on the parameters '''
class coefficient(object):
    def __init__(self,params): #### out_shape =([M]) |  input:  x_shape=[M,D,1],  z shape = [M,D,1],a_shape= [M,D,D]  #This is for rho=0
        self.dim = torch.tensor(params['dim']).clone().detach()#.to(device)#2
        self.nu = torch.tensor(params['nu'][0:self.dim]).clone().detach()#.to(device)
        self.kappa = torch.tensor(params['kappa'][0:self.dim]).clone().detach()#.to(device)
        self.theta = torch.tensor(params['theta'][0:self.dim]).clone().detach()#.to(device)
        self.eta = torch.tensor(params['eta']).clone().detach()#.to(device)
        self.lb = torch.tensor(params['lb'][0:self.dim]).clone().detach()#.to(device)
        self.lb_norm = torch.sqrt(torch.pow(self.lb,2).sum())
        self.params = params
    def __call__(self,x):
        pass
    def __add__(self, other):
        tmp = coefficient(self.params)
        tmp.__call__ = lambda x : self(x) + other(x)
        return tmp
        
    
''' diffusion'''
class custom_diff(coefficient):
    def __init__(self, params,s):
        super(custom_diff, self).__init__(params)
        if torch.is_tensor(s):
            self.val = lambda x:s
        else:
            self.val = s
    def __call__(self, x):
        dim = self.params['dim']
        num_samples = x.shape[0]
        A=torch.zeros(num_samples,dim,dim)
        A[:,1:,1:] = torch.diag(self.nu[0:dim-1])
        A[:,0,0] = self.val(x)
        return A
    def __add__(self, other):
        tmp = lambda x : self.val(x) + other.val(x)
        return custom_diff(self.params,tmp)
    
    
'''Drift of semilinear eqn: first component drift=0, others are OU'''
class OU_drift_semi(coefficient):
    def __init__(self,params):
        super(OU_drift_semi, self).__init__(params)
    def __call__(self,x):
        num_samples = x.shape[0]
        output = torch.zeros(num_samples,self.dim)
        for i in range(0,self.dim):
            output[:,i] = self.kappa[i]*(self.theta[i] - x[:,i+1])
        return output  
    
'''Source'''      
class zero_source(coefficient):
    def __init__(self,params):
        super(zero_source, self).__init__(params)
    def __call__(self,x):
        return torch.zeros(x.shape[0],1)      
    
'''Zero discount coefficient for all components'''  
class zero_discount(coefficient):
    def __init__(self,params):
        super(zero_discount, self).__init__(params)
    def __call__(self,x):
        return torch.zeros(x.shape[0],1)          
    
class direction(coefficient):
    def __init__(self,params,v,sigma,**kwargs):
        self.p = v# we need to evaluate gradient and hessian of v
        self.sigma = sigma
    
        if 'magnitude' in kwargs:
            magnitude = kwargs['magnitude']
            if torch.is_tensor(magnitude)|isinstance(magnitude, float):
                if isinstance(magnitude, float):
                    magnitude = torch.tensor(magnitude)
                self.magnitude = lambda x:magnitude.repeat(x.shape[0],1)
            else:
                self.magnitude = magnitude
        else:
            self.magnitude = lambda x:torch.ones(x.shape[0],1)
        if 'bound' in kwargs:
            bound = kwargs['bound']
            if torch.is_tensor(bound)|isinstance(bound, float):
                if isinstance(bound, float):
                    bound = torch.tensor(bound)
                self.bound = lambda x:bound.repeat(x.shape[0],1)
            else:
                self.bound = bound   
        else:
            self.bound = lambda x:torch.ones(x.shape[0],1)    
        self.Lb = lambda x:  self.lb_norm.repeat(x.shape[0])              
        super(direction, self).__init__(params)
    def val(self,x):
        D2 = Grad_Hess(x,self.p)[1][:,1,1]
        D1 = Grad_Hess(x,self.p)[0][:,1]
        return torch.maximum(torch.minimum(self.magnitude(x).squeeze(-1)*(self.Lb(x)*torch.abs(D1.squeeze(-1))+self.sigma(x)[:,0,0]*D2),self.bound(x).squeeze(-1)), -self.bound(x).squeeze(-1))
    # torch.maximum(torch.minimum(self.magnitude(x).squeeze(-1)*(self.lb_norm*torch.abs(D1.squeeze(-1))+self.sigma(x)[:,0,0]*D2),self.bound(x).squeeze(-1)), torch.zeros(x.shape[0])) 

    def __call__(self,x):
        A = torch.zeros(x.shape[0],self.dim,self.dim)
        # print(self.magnitude(x).shape,(self.lb_norm*torch.abs(D1.squeeze(-1))+self.sigma(x)[:,0,0]*D2).shape)
        A[:,0,0] = self.val(x)
        return A   
    def __mul__(self, magnitude):
        return direction(self.params,self.p,self.sigma,magnitude=magnitude,bound=self.bound)
    def __rmul__(self, magnitude):
        return direction(self.params,self.p,self.sigma,magnitude=magnitude,bound=self.bound)
        
'''Driver for a semilinear'''               
class f_driver(coefficient):
    def __init__(self,params): #### out_shape =([M]) |  input:  x_shape=[M,D,1],  z shape = [M,D,1],a_shape= [M,D,D]  #This is for rho=0
        super(f_driver, self).__init__(params)
    def __call__(self,z,a):
        return -torch.sqrt(torch.sum(torch.square(self.lb)))*torch.abs(z[:,0,0])*torch.abs(a)# + output
                

class exponential_terminal(coefficient):
    def __init__(self,params):
        self.eta = params['eta']
    def __call__(self,x):
        return (torch.tensor([1.])-torch.exp(-self.eta*x[:,0])).unsqueeze(-1)            
        