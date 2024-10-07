import torch
# import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# from IPython.display import display, Markdown


# import coeff
# import functions
from neuralnets import Ynet, Ytnet
from samplepaths import data_gen
import copy
from derivation import Grad
 

    
class parabolic(object):
    def __init__(self,sigma,mu,kappa,terminal,pde,sim):        
        self.Y0 = Ynet(pde,sim) # NN for value at t=0 
        # self.Z = Znet(pde,sim) # NN for gradient at all times
        self.Yt = Ytnet(pde,sim) # NN for value function at all times, required to update sigma
        self.terminal = terminal # terminal condition
        self.loss_epoch = [] # list to keep the loss at each training epoch
        self.epoch=0 # initializing epoch to zero
        # self.num_epochs = 5000 # total number of epochs
        self.mu = mu # drift of the SDE
        self.sigma = sigma # diffusion coef. for SDE
        self.kappa = kappa # discount factor
        self.n = sim['num_time_intervals'] # number of time intervals
        self.dim = pde['dim']
        self.num_samples = sim['num_samples']
        data = data_gen(sigma,mu,pde,sim)
        self.dt = data.dt.to(device)
        self.x = data.x.to(device).clone().detach()
        self.sigmadw = data.sigmadw.to(device).clone().detach()
        self.r = torch.ones((self.x.shape[0],1,self.n+1)).to(device)
        for i in range(self.n):
            self.r[:,:,i+1] = self.r[:,:,i]* torch.exp(-self.kappa(self.x[:,:,i])*self.dt)
        self.r = self.r.clone().detach()
        self.trained = False
        self.params = {**copy.deepcopy(pde),**copy.deepcopy(sim)} 
        
    def train(self,lr,delta_loss,max_num_epochs):
        if max_num_epochs <= self.epoch:
            print("The maximum number of epochs, {}, is reached. Increase max_num_epochs and run again.".format(max_num_epochs))
        else:
            t_0 = time.time()
            self.lr = lr
            parameters = list(self.Y0.parameters()) + list(self.Yt.parameters())
            optimizer = optim.Adam(parameters, self.lr)
            L_ = -2.0
            L = 2.0
            while (np.abs(L_-L)>delta_loss) & (self.epoch < max_num_epochs):# epoch in range(num_epochs):
                t_1 = time.time()
                optimizer.zero_grad()
                if self.epoch>0:
                    L_ = self.loss_epoch[self.epoch-1]
                loss= self.loss()#self.cost(self.X,self.modelu(X))+ torch.mean(self.terminal(update(self.X,self.modelu(X))))#
                loss.backward()
                optimizer.step()
                L = loss.clone().detach().numpy()
                self.loss_epoch.append(L)
                if (self.epoch % int(max_num_epochs/3)== int(max_num_epochs/3)-1) | (self.epoch == 0):
                    print('At epoch {}, mean loss is {:.2E}.'.format(self.epoch+1,L))
                    self.time_display(t_0, t_1)
                self.epoch += 1
            t_delta = time.time()-t_0
            self.params['training_time'] = t_delta
            print(r'Training took {} epochs and {:,} ms and the final loss is {:.2E}.'.format(self.epoch,round(1000*(t_delta),2),loss))
        self.trained = True
        # self.value_fnc(lr=1e-2,delta_loss=delta_loss,max_num_epochs=1000,num_batches=10)
        self.params['Y0'] = self.Y0
        # self.params['Z'] = self.Z
        self.params['value'] = self.Yt
        self.params['loss'] = self.loss_epoch
        self.params['max_epochs'] = self.epoch
        

    def time_display(self, t_0, t_1):
        print(r'Training this epoch takes {:,} ms. So far: {:,} ms in training.'.format(round(1000*(time.time()-t_1),2),round(1000*(time.time()-t_0),2)))
        

class linear(parabolic):
    def __init__(self, sigma, mu, source, kappa, terminal, pde, sim):
        self.source = source # source term for the PDE
        super().__init__(sigma, mu, kappa, terminal, pde, sim)  
        self.c = torch.ones((self.x.shape[0],1,self.n+1)).to(device)
        for i in range(self.n):
            if i == self.n -1 :
                self.c[:,:,i+1] = self.terminal(self.x[:,1:,i+1])
            self.c[:,:,i] = self.source(self.x[:,:,i])
        self.c = self.c.clone().detach()  
        
    def loss(self):
        for i in range(self.n):   
            if i == 0:
                Y =  self.Y0(self.x[:,1:,0])
            else:
                Z = Grad(self.x[:,:,i-1],self.Yt)[:,1:,:].view(-1,1,self.dim)
                Y = Y*self.r[:,:,i] + self.c[:,:,i-1]*self.dt + torch.bmm(Z,self.sigmadw[:,:,i-1].unsqueeze(2)).squeeze(2)
                if i == self.n-1:
                    Z = Grad(self.x[:,:,i],self.Yt)[:,1:,:].view(-1,1,self.dim)
                    Y = Y*self.r[:,:,i] + self.c[:,:,i]*self.dt + torch.bmm(Z,self.sigmadw[:,:,i].unsqueeze(2)).squeeze(2)
        L1 = torch.pow(self.c[:,:,-1]-Y,2)
        L2 = torch.pow(self.Y0(self.x[:,1:,0])-self.Yt(self.x[:,:,0]),2)# Match with Y0
        L3 = torch.pow(self.c[:,:,-1]-self.Yt(self.x[:,:,-1]),2) # match with terminal
        L = L1 + L2 + L3
        return L.mean()        
    
class semilinear(parabolic):
    def __init__(self,sigma, mu, driver, kappa, terminal, pde, sim):
        self.F = driver    
        super(semilinear,self).__init__(sigma, mu, kappa, terminal, pde, sim)   
        self.sigmax = torch.ones((self.x.shape[0],self.dim,self.dim,self.n)).to(device)
        for i in range(self.n):
            #evaluate and reuse self.sigma(self.x[:,:,i]).reshape((self.num_samples,self.dim,self.dim))[:,0,0] 
            self.sigmax[:,:,:,i] = self.sigma(self.x[:,:,i]).reshape((self.num_samples,self.dim,self.dim))
        self.sigmax = self.sigmax.clone().detach()
        
    def loss(self):
        c = torch.zeros((self.num_samples,1,self.n+1)).to(device)
        for i in range(self.n):   
            if i == 0:
                Y =  self.Y0(self.x[:,1:,0])
            else:
                Z = Grad(self.x[:,:,i-1],self.Yt)[:,1:,:].view(-1,1,self.dim)                    
                c[:,:,i-1] = self.F(self.x[:,:,i-1],Z,self.sigmax[:,:,:,i-1]).unsqueeze(-1)
                Y = Y*self.r[:,:,i] + c[:,:,i-1]*self.dt + torch.bmm(Z,self.sigmadw[:,:,i-1].unsqueeze(2)).squeeze(2)
                if i == self.n - 1:
                    Z = Grad(self.x[:,:,i],self.Yt)[:,1:,:].view(-1,1,self.dim)                    
                    c[:,:,i] = self.F(self.x[:,:,i],Z,self.sigmax[:,:,:,i]).unsqueeze(-1)
                    Y = Y*self.r[:,:,i] + c[:,:,i]*self.dt + torch.bmm(Z,self.sigmadw[:,:,i].unsqueeze(2)).squeeze(2)
                    c[:,:,i+1] = self.terminal(self.x[:,1:,i+1])
        L1 = torch.pow(c[:,:,-1]-Y,2)
        L2 = torch.pow(self.Y0(self.x[:,1:,0])-self.Yt(self.x[:,:,0]),2)# Match with Y0
        L3 = torch.pow(c[:,:,-1]-self.Yt(self.x[:,:,-1]),2) # match with terminal
        L = L1 + L2 + L3
        return L.mean()            