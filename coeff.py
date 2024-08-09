import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Frobenius(A,B):
    return torch.vmap(torch.trace)(torch.bmm(A,torch.transpose(B,1,2)))

def Grad(x,v): #output= [M,D,D], #input: x=[M,D], t=[M,1], xt= [M,D+1]
    d = x.shape[1]
    Du=torch.zeros(x.shape[0],d).to(device)
    xin=x.clone().detach()
    xin.requires_grad=True
    u=v(xin)
    Du=torch.autograd.grad(outputs=[u],inputs=[xin],grad_outputs=torch.ones_like(u),
                           allow_unused=True,retain_graph=True,create_graph=True)[0].unsqueeze(2)
    Du = torch.reshape(Du,(Du.shape[0],d,1))
    return Du

def Grad_Hess(x,v): #output= [M,D,D], #input: x=[M,D], t=[M,1], xt= [M,D+1]
    d = x.shape[1]
    hess_temp=torch.zeros(x.shape[0],d,d).to(device)
    Du=torch.zeros(x.shape[0],d).to(device)
    xin=x.clone().detach()
    xin.requires_grad=True
    u=v(xin)
    Du=torch.autograd.grad(outputs=[u],inputs=[xin],grad_outputs=torch.ones_like(u),
                           allow_unused=True,retain_graph=True,create_graph=True)[0].unsqueeze(2)
    hess_temp= torch.cat([torch.autograd.grad(outputs=[Du[:,i,:]],inputs=[xin],grad_outputs=torch.ones_like(Du[:,i,:]),
                           allow_unused=True,retain_graph=True,create_graph=True)[0] for i in range(d)],1)
    Du = torch.reshape(Du,(Du.shape[0],d,1))
    hess_temp=torch.reshape(hess_temp,(hess_temp.shape[0],d,d))
    return Du, hess_temp



'''Implements the coefficeints based on the parameters '''
class coefficient(object):
    def __init__(self,params): #### out_shape =([M]) |  input:  x_shape=[M,D,1],  z shape = [M,D,1],a_shape= [M,D,D]  #This is for rho=0
        self.dim=params['dim']#2
        self.nu = params['nu']
        self.kappa = params['kappa'][0:self.dim]
        self.theta = params['theta'][0:self.dim]
        self.eta = params['eta']
        self.lb = params['lb'][0:self.dim]
        self.lb_norm = torch.sqrt(torch.pow(params['lb'][0:self.dim],2).sum())
        self.params = params
    
'''constant diffution coefficient'''  
class constant_diff(coefficient):
    '''This class is a constant diffusion coefficient which is the optimal diffusion'''
    def __init__(self,params,**kwargs):
        super(constant_diff, self).__init__(params)
        if kwargs:
            if 'constant_diff' in kwargs.keys():
                self.diff = kwargs['constant_diff']
            else:
                self.diff = torch.sqrt(torch.pow(self.lb,2).sum())/self.eta
        else:
            self.diff = torch.sqrt(torch.pow(self.lb,2).sum())/self.eta
    def __call__(self,x):
        tmp = x.shape[0]
        return torch.diag(torch.cat((self.diff,self.nu[0:self.dim-1]),axis=0)).repeat(tmp,1,1)

'''Random diffusion coefficient for wealth process with diffusion of volatility processes all constant'''   
class random_diff(coefficient):
    def __init__(self,params):
        super(random_diff, self).__init__(params)
    def __call__(self,x):
        tmp = x.shape[0]
        return torch.diag(torch.cat((torch.rand(tmp,1),self.nu[0:self.dim-1].repeat(tmp,1)),axis=0))
    

    

''' diffusion'''
class custom_diff(coefficient):
    def __init__(self, params,s):
        super(custom_diff, self).__init__(params)
        if torch.is_tensor(s):
            self.s = lambda x:s
        else:
            self.s = s
    def __call__(self, x):
        dim = self.params['dim']
        num_samples = x.shape[0]
        A=torch.zeros(num_samples,dim,dim)
        A[:,1:,1:] = torch.diag(self.params['nu'][0:dim-1])
        A[:,0,0] = self.s(x)
        return A
    
    
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
            output[:,i] = self.kappa[i]*(self.theta[i] - x[:,i])
        return output 
    
'''Drift of semilinear eqn: first component drift=0, others are OU'''
class OU_drift_semi(coefficient):
    def __init__(self,params):
        super(OU_drift_semi, self).__init__(params)
    def __call__(self,x):
        num_samples = x.shape[0]
        output = torch.zeros(num_samples,self.dim)
        for i in range(0,self.dim):
            output[:,i] = self.kappa[i]*(self.theta[i] - x[:,i])
        return output
 
     
    
    
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
            self.lb_norm = torch.sqrt(torch.pow(params['lb'][0:params['dim']],2).sum())
    def __call__(self, x):
        num_samples = x.shape[0]
        output = torch.zeros(num_samples,self.dim)
        for i in range(1,self.dim):
            output[:,i] = self.kappa[i]*(self.theta[i] - x[:,i])
        output[:,0] = self.lb_norm*torch.sgn(Grad(x,self.s)[:,1].squeeze(-1))
        return output    
    
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
            output[:,i] = self.kappa[i]*(self.theta[i] - x[:,i])
        return output 
    
    
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
            self.lb_norm = torch.sqrt(torch.pow(params['lb'][0:params['dim']],2).sum())
    def __call__(self, x):
        num_samples = x.shape[0]
        output = torch.zeros(num_samples,self.dim)
        for i in range(1,self.dim):
            output[:,i] = self.kappa[i]*(self.theta[i] - x[:,i])
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
    
class direction(coefficient):
    def __init__(self,params,v,sigma,magnitude):
        self.p = v# we need to evaluate gradient and hessian of v
        self.sigma = sigma
        if torch.is_tensor(magnitude):
            self.magnitude = lambda x:magnitude.repeat(x.shape[0],1)
        else:
            self.magnitude = magnitude
        super(direction, self).__init__(params)
    def __call__(self,x):
        D2 = Grad_Hess(x,self.p)[1][:,1:,1:]
        D1 = Grad_Hess(x,self.p)[0][:,1]
        A = torch.zeros(x.shape[0],self.dim,self.dim)
        A[:,0,0] = -self.lb_norm*torch.abs(D1.squeeze(-1))#
        A = A-torch.bmm(self.sigma(x),D2)
        # print(self.magnitude(x).shape)
        return torch.bmm(torch.diag_embed(self.magnitude(x).repeat(1,self.params['dim'])),A)            
    
class source_from_direction(coefficient): # source term for the linear equations
    def __init__(self,params,direction):
        # self.alpha = direction.magnitude
        self.direction = direction
        super(source_from_direction, self).__init__(params)
    def __call__(self,x):
        A = self.direction(x)
        return torch.sqrt(Frobenius(A,A)).unsqueeze(-1)
        
'''Zero discount coefficient for all components'''  
class zero_discount(coefficient):
    def __init__(self,params):
        super(zero_discount, self).__init__(params)
    def __call__(self,x):
        return torch.zeros(x.shape[0],1)                    
    
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
        self.eta = params['eta']
    def __call__(self,x):
        return (torch.tensor([1.])-torch.exp(-self.eta*x[:,0])).unsqueeze(-1)            
    
class zero_terminal(coefficient):
    def __init__(self,params):
        self.eta = params['eta']
    def __call__(self,x):
        num_samples = x.shape[0]
        return torch.zeros(num_samples,1)