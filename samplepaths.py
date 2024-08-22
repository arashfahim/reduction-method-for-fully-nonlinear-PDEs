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
        num_samples = sim['num_samples']
        n = sim['num_time_intervals']
        dim = pde['dim']
        self.dt = torch.tensor([pde['T']/n]).to(device)
        self.dw = iid[0:num_samples*dim*n].reshape([num_samples,dim,n]).to(device)* torch.sqrt(self.dt)#you can make randomness a universal variable if 
        self.x = torch.zeros((num_samples,dim+1,n+1))
        
        self.x[:,1:,0]= a+(b-a)*torch.rand(num_samples,dim).to(device)  
        self.sigmadw = torch.zeros((num_samples,dim,n)).to(device)
        for i in range(n):
            self.sigmadw[:,:,i] = torch.bmm(self.sigma(self.x[:,:,i]).reshape((num_samples,dim,dim)),self.dw[:, :, i].unsqueeze(2)).squeeze(2)
            self.x[:,1:,i+1] = self.x[:,1:,i] + self.mu(self.x[:,:,i])*self.dt + self.sigmadw[:,:,i]
            self.x[:,0,i+1] = self.x[:,0,i]+self.dt