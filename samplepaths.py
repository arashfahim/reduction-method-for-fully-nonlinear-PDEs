import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class data_gen(object):
    def __init__(self,sigma,mu,pde,sim):
        self.mu = mu
        self.sigma = sigma
        a = torch.tensor(sim['start']).to(device)
        b = torch.tensor(sim['end']).to(device)
        iid = sim['iid']
        self.num_samples = sim['num_samples']
        self.n = sim['num_time_intervals']
        self.dim = pde['dim']
        self.dt = torch.tensor([pde['T']/self.n]).to(device)
        self.dw = iid[0:self.num_samples*self.dim*n].reshape([self.num_samples,self.dim,self.n]).to(device)* torch.sqrt(self.dt)#you can make randomness a universal variable if 
        self.x = torch.zeros((self.num_samples,self.dim+1,self.n+1))
        
        
        self.x[:,1:,0]= a+(b-a)*torch.rand(self.num_samples,self.dim).to(device)  
        self.sigmadw = torch.zeros((self.num_samples,self.dim,self.n)).to(device)
        for i in range(self.n):
            self.sigmadw[:,:,i] = torch.bmm(self.sigma(self.x[:,:,i]).reshape((self.num_samples,self.dim,self.dim)),self.dw[:, :, i].unsqueeze(2)).squeeze(2)
            self.x[:,1:,i+1] = self.x[:,1:,i] + self.mu(self.x[:,:,i])*self.dt + self.sigmadw[:,:,i]
            self.x[:,0,i+1] = self.x[:,0,i]+self.dt