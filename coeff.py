import torch
from derivation import Grad, Grad_Hess
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Frobenius(A,B):
    return torch.vmap(torch.trace)(torch.bmm(A,torch.transpose(B,1,2)))



'''Implements the coefficients based on the parameters '''
class coefficient(object):
    def __init__(self,params): #### out_shape =([M]) |  input:  x_shape=[M,D,1],  z shape = [M,D,1],a_shape= [M,D,D]  #This is for rho=0
        self.dim = torch.tensor(params['dim']).clone().detach().to(device)#2
        self.nu = torch.tensor(params['nu'][0:self.dim]).clone().detach().to(device)
        self.kappa = torch.tensor(params['kappa'][0:self.dim]).clone().detach().to(device)
        self.theta = torch.tensor(params['theta'][0:self.dim]).clone().detach().to(device)
        self.eta = torch.tensor(params['eta']).clone().detach().to(device)
        self.lb = torch.tensor(params['lb'][0:self.dim]).clone().detach().to(device)
        self.lb_norm = torch.sqrt(torch.pow(self.lb,2).sum())
        self.params = params
    
# '''constant diffusion coefficient'''  
# class constant_diff(coefficient):
#     '''This class is a constant diffusion coefficient which is the optimal diffusion'''
#     def __init__(self,params,**kwargs):
#         super(constant_diff, self).__init__(params)
#         if kwargs:
#             if 'constant_diff' in kwargs.keys():
#                 self.diff = kwargs['constant_diff']
#             else:
#                 self.diff = torch.sqrt(torch.pow(self.lb,2).sum())/self.eta
#         else:
#             self.diff = torch.sqrt(torch.pow(self.lb,2).sum())/self.eta
#     def __call__(self,x):
#         tmp = x.shape[0]
#         return torch.diag(torch.cat((self.diff,self.nu[1:]),axis=0)).repeat(tmp,1,1)

# '''Random diffusion coefficient for wealth process with diffusion of volatility processes all constant'''   
# class random_diff(coefficient):
#     def __init__(self,params):
#         super(random_diff, self).__init__(params)
#     def __call__(self,x):
#         tmp = x.shape[0]
#         return torch.diag(torch.cat((torch.rand(tmp,1),self.nu[1:].repeat(tmp,1)),axis=0))
    

    

''' diffusion'''
class custom_diff(coefficient):
    def __init__(self, params,s):
        super(custom_diff, self).__init__(params)
        if torch.is_tensor(s):
            self.val = lambda x:s
        else:
            self.val = s
    def __call__(self, x):
        dim = self.dim
        num_samples = x.shape[0]
        A=torch.zeros(num_samples,dim,dim)
        A[:,1:,1:] = torch.diag(self.nu[1:])
        A[:,0,0] = self.val(x)
        return A
    def __add__(self, other):
        tmp = lambda x : self.val(x) + other.val(x)
        return custom_diff(self.params,tmp)
    
''' diffusion'''
# class custom_diff(coefficient):
#     def __init__(self, params,s, **kwargs):
#         super(custom_diff, self).__init__(params)
#         if torch.is_tensor(s):
#             self.val = lambda x: s.repeat(x.shape[0])# Shit!
#         else:
#             self.val = s
        
#     def __call__(self, x):
        
#         return self.val(x)
#     def __add__(self, other):
#         tmp = lambda x : self.val(x) + other.val(x)
#         return custom_diff(self.params,tmp)
    
    
    
'''Zero drift coefficient for all components'''  
class zero_drift(coefficient):
    def __init__(self,params):
        super(zero_drift, self).__init__(params)
    def __call__(self,x):
        return torch.zeros(x.shape[0],x.shape[1]-1)
    
    
'''Drift of linear eqn: first component drift =|lb|*sgn(v_e)sigma_{00}(x), others are OU'''
class OU_drift_lin(coefficient):
    def __init__(self,params,v,sigma):
        self.p = v
        self.sigma = sigma
        super(OU_drift_lin, self).__init__(params)
    def __call__(self,x):
        num_samples = x.shape[0]
        q = self.p(x)[:,0]
        output = torch.zeros(num_samples,self.dim)
        output[:,0] = torch.sqrt(torch.pow(self.lb,2).sum())*torch.sgn(q)*self.sigma(x)[:,0,0]
        for i in range(1,self.dim):
            output[:,i] = self.kappa[i]*(self.theta[i] - x[:,i+1])
        return output 
    
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
    
    
'''Drift of gradient linear eqn in Chesney-Scott model: first component drift=0, others are OU'''
class OU_drift_lin_CS(coefficient):
    def __init__(self,params,v,sigma):
        self.p = v
        self.sigma = sigma
        super(OU_drift_lin, self).__init__(params)
    def __call__(self,x):
        num_samples = x.shape[0]
        q = self.p(x)[:,0]
        output = torch.zeros(num_samples,self.dim)
        output[:,0] = torch.sqrt(torch.pow(self.lb*x[:,1:],2).sum(axis=1))*torch.sgn(q)*self.sigma(x)[:,0,0]
        for i in range(1,self.dim):
            output[:,i] = self.kappa[i]*(self.theta[i] - x[:,i+1])
        return output 

    
'''Drift of linear eqn: first component drift =|lb|*sgn(v_e)sigma_{00}(x), others are OU'''
    
class custom_drift(coefficient):
    def __init__(self, params,s,sigma):
        super(custom_drift, self).__init__(params)
        if torch.is_tensor(s):
            self.s = lambda x:s
        else:
            self.s = s
            if torch.is_tensor(sigma):
                self.sigma = lambda x:sigma
            else:
                self.sigma = sigma
    def __call__(self, x):
        num_samples = x.shape[0]
        output = torch.zeros(num_samples,self.dim)
        for i in range(1,self.dim):
            output[:,i] = self.kappa[i]*(self.theta[i] - x[:,i+1])
        output[:,0] = self.lb_norm*torch.sgn(Grad(x,self.s)[:,1].squeeze(-1))
        return output    
    
class zero_source(coefficient):
    def __init__(self,params):
        super(zero_source, self).__init__(params)
    def __call__(self,x):
        return torch.zeros(x.shape[0],1)   
    
class custom_source(coefficient):
    def __init__(self,params,source):
        if torch.is_tensor(source):
            self.val = lambda x:source 
        else:
            self.val = source
        super(custom_source, self).__init__(params)
    def __call__(self,x):
        return self.val(x) 
    
'''Direction'''
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
            self.bound = lambda x:2*torch.ones(x.shape[0],1)
        if 'ChesneyScott' in kwargs:
            if kwargs['ChesneyScott']:
                self.CS = True
                self.Lb = lambda x: torch.pow(self.lb*x[:,1:],2).sum(axis=1)
        else:
            self.CS = False
            self.Lb = lambda x:  self.lb_norm.repeat(x.shape[0])
                        
        super(direction, self).__init__(params)
    def val(self,x):
        D2 = Grad_Hess(x,self.p)[1][:,1,1]
        D1 = Grad_Hess(x,self.p)[0][:,1]
        return torch.maximum(torch.minimum(self.magnitude(x).squeeze(-1)*(self.Lb(x)*torch.abs(D1.squeeze(-1))+self.sigma(x)[:,0,0]*D2),self.bound(x).squeeze(-1)), -self.bound(x).squeeze(-1)) 
    def __call__(self,x):
        A = torch.zeros(x.shape[0],self.dim,self.dim)
        A[:,0,0] = self.val(x)
        return A        
    def __mul__(self, magnitude):
        return direction(self.params,self.p,self.sigma,magnitude=magnitude,bound=self.bound,ChesneyScott=self.CS)
    def __rmul__(self, magnitude):
        return direction(self.params,self.p,self.sigma,magnitude=magnitude,bound=self.bound,ChesneyScott=self.CS)
        

'''Source'''      
class source_from_direction(coefficient): # source term for the linear equations
    def __init__(self,params,direction):
        # self.alpha = direction.magnitude
        self.direction = direction
        super(source_from_direction, self).__init__(params)
    def __call__(self,x):
        A = self.direction(x)
        return -torch.sqrt(Frobenius(A,A)).unsqueeze(-1)
        
'''Zero discount coefficient for all components'''  
class zero_discount(coefficient):
    def __init__(self,params):
        super(zero_discount, self).__init__(params)
    def __call__(self,x):
        return torch.zeros(x.shape[0],1)     
    
        
'''Driver with linear Chesney-Scott dependence on vol for a semilinear'''               
class f_driver(coefficient):
    def __init__(self,params,**kwargs): 
        if 'ChesneyScott' in kwargs:
            tmp = float(kwargs['ChesneyScott'])
            self.lbv_norm = lambda x:tmp*torch.sqrt(torch.square(self.lb*x[:,1:]).sum(axis=1))+(1-tmp)*self.lb_norm
        else:
            self.lbv_norm = lambda x: self.lb_norm
        super(f_driver, self).__init__(params)
    def __call__(self,x,z,a):
        return -self.lbv_norm(x)*torch.abs(z[:,0,0])*torch.abs(a[:,0,0])
    
'''Driver for viscose Burger'''               
class burger(coefficient):
    def __init__(self,params,**kwargs): 
        super(f_driver, self).__init__(params)
    def __call__(self,x,z,y):
        return y*(z.sum(axis=1))
        
        
'''Adding two coefficients together'''
class add_coeff(coefficient):
    def __init__(self,s1,s2):
        self.s1 = s1
        self.s2 = s2
        super(add_coeff, self).__init__(s1.params)
    def __call__(self,x):
        return self.s1(x)+self.s2(x)
        
'''Terminal functions'''        
class exponential_terminal(coefficient):
    def __init__(self,params):
        super(exponential_terminal,self).__init__(params)
    def __call__(self,x):
        return (torch.tensor([1.])-torch.exp(-self.eta*x[:,0])).unsqueeze(-1)            
    
class zero_terminal(coefficient):
    def __init__(self,params):
        super(zero_terminal,self).__init__(params)
    def __call__(self,x):
        num_samples = x.shape[0]
        return torch.zeros(num_samples,1)
    
    
# class solution(object):
#     def __init__(self,params):
#         self.dim = torch.tensor(params['dim']).to(device)#2
#         self.nu = torch.tensor(params['nu'][0:self.dim]).to(device)
#         self.kappa = torch.tensor(params['kappa'][0:self.dim]).to(device)
#         self.theta = torch.tensor(params['theta'][0:self.dim]).to(device)
#         self.eta = torch.tensor(params['eta']).to(device)
#         self.lb = torch.tensor(params['lb'][0:self.dim]).to(device)
#         self.lb_norm = torch.sqrt(torch.pow(self.lb,2).sum())
#         self.T = params['T']
        
        
# class exp_solution(solution):
#     def __init__(self,params,alpha):
#         self.alpha = alpha
#         super(exp_solution, self).__init__(params)
#     def __call__(self,x):
#         # print(self.alpha)
#         return torch.tensor([1.])-torch.exp(-self.eta*x[:,1]+self.alpha*(self.T-x[:,0])).to(device)     

# class zero_solution(solution):
#     def __init__(self,params):
#         super(zero_solution, self).__init__(params)
#     def __call__(self,x):
#         # print(self.alpha)
#         return torch.zeros([x.shape[0],1]).to(device) 
    
# class time_solution(solution):
#     def __init__(self,params,constant):
#         self.constant = constant
#         super(time_solution, self).__init__(params)
#     def __call__(self,x):
#         return (self.T-x[:,0].unsqueeze(-1))*self.constant*torch.ones([x.shape[0],1]).to(device)        
    
    
    