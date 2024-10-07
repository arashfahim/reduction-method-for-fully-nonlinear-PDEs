import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''Class to create functions which take parameters from a dictionary'''
class solution(object):
    def __init__(self,params):
        self.dim = params['dim']#2
        self.nu = params['nu'][0:self.dim]
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
    
    
class ChesneyScott(solution):
    def __init__(self, params):
        super(ChesneyScott,self).__init__(params)   
        self.hk = torch.sqrt(torch.pow(torch.tensor(self.kappa),2)+ torch.pow(torch.tensor(self.nu)*torch.tensor(self.lb),2))

    def auxillary(self,i,t):
        khk = self.kappa[i]/self.hk[i]
        sinh = torch.sinh(self.hk[i]*t)
        cosh = torch.cosh(self.hk[i]*t)
        term0 = cosh + khk*sinh
        # print(term0)
        denom = self.hk[i]*term0
        term1 = (cosh-1)/denom
        term2 = sinh/denom
        phi = torch.pow(torch.tensor(self.lb[i]),2)*term2
        psi = torch.pow(torch.tensor(self.lb[i]),2)*self.theta[i]*khk*term1
        chi = .5*torch.log(term0) - .5*self.kappa[i]*t -  torch.pow(self.lb[i]*khk*self.theta[i],2)*(0.5*(term2 -t) + khk*term1)
        # print(.5*torch.log(term0) - .5*self.kappa[i]*t)
        return phi,  psi, chi
    def wtv(self,x):
        # print(x[:,2])
        tmp = torch.zeros(x.shape[0])
        for i in range(1,self.dim):
            phi, psi, chi = self.auxillary(i,self.T-x[:,0])
            tmp = tmp -0.5*phi*torch.pow(x[:,i+1],2) - psi* x[:,i+1] - chi
            # print(tmp,phi,psi,chi,-0.5*torch.pow(self.lb[i],2)*(self.T-x[:,0]))
        return tmp
    def __call__(self,x):
        return torch.tensor([1.])-torch.exp(-self.eta*x[:,1]+self.wtv(x)).to(device)     
