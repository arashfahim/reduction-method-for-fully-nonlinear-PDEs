import torch # type: ignore
from torch import nn # type: ignore
import numpy as np # type: ignore
import torch.optim as optim # type: ignore
import time
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from neuralnets import sigmanet

# the sigma update helps memory useage stay constant at the cost of time

class sigma_update(object):
    def __init__(self, pde, sim, data, sigma_dir) -> None:
        # self.sigma = sigma_dir
        self.dim = pde['dim']
        for i in range(sim['num_time_intervals']):
            if i==0:
                tmp = data[:,:,i]
            else:
                tmp = torch.cat((tmp,data[:,:,i]),axis=0)
        self.x = tmp
        self.new_sigma = sigmanet(pde,sim)
        self.y_true = sigma_dir(self.x)
    
    def train(self,lr=1e-2,delta_loss=1e-9,max_num_epochs=2000):
        t_0 = time.time()
        self.lr = lr
        parameters = list(self.new_sigma.parameters())
        optimizer = optim.Adam(parameters, self.lr)
        L_ = -2.0
        L = 2.0
        loss_fnc = nn.MSELoss()
        while (np.abs(L_-L)>delta_loss) & (self.epoch < max_num_epochs):# epoch in range(num_epochs):
            t_1 = time.time()
            optimizer.zero_grad()
            if self.epoch>0:
                L_ = self.loss_epoch[self.epoch-1]
            loss = loss_fnc(self.new_sigma(self.x),self.y_true)#self.cost(self.X,self.modelu(X))+ torch.mean(self.terminal(update(self.X,self.modelu(X))))#
            loss.backward()
            optimizer.step()
            L = loss.clone().detach().numpy()
            self.loss_epoch.append(L)
            if (self.epoch % int(max_num_epochs/2)== int(max_num_epochs/2)-1) | (self.epoch == 0):
                print('At epoch {}, mean loss is {:.2E}.'.format(self.epoch+1,L))
                self.time_display(t_0, t_1)
            self.epoch += 1
        t_delta = time.time()-t_0
        print(r'Training took {} epochs and {:,} ms and the final loss is {:.2E}.'.format(self.epoch,round(1000*(t_delta),2),loss))
        
        
        