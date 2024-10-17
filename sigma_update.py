import torch # type: ignore
from torch import nn # type: ignore
import numpy as np # type: ignore
import torch.optim as optim # type: ignore
import time
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from neuralnets import sigmanet

# the sigma update helps memory useage stay constant at the cost of time

class fussion(object):
    def __init__(self, pde, sim, data, sigma_dir) -> None:
        # self.sigma = sigma_dir
        self.dim = pde['dim']
        for i in range(sim['num_time_intervals']):
            if i==0:
                tmp = data[:,:,i]
            else:
                tmp = torch.cat((tmp,data[:,:,i]),axis=0)
        self.x = tmp
        self.val = sigmanet(pde,sim)
        self.y_true = sigma_dir(self.x)[:,0,0].unsqueeze(-1).clone().detach()
        self.loss_epoch = []
    
    def train(self,lr=1e-2,delta_loss=1e-9,max_num_epochs=2000):
        print("training sigma started ...")
        epoch = 0
        t_0 = time.time()
        self.lr = lr
        parameters = list(self.val.parameters())
        optimizer = optim.Adam(parameters, self.lr)
        L_ = -2.0
        L = 2.0
        loss_fnc = nn.MSELoss()
        while (np.abs(L_-L)>delta_loss) & (epoch < max_num_epochs):# epoch in range(num_epochs):
            t_1 = time.time()
            optimizer.zero_grad()
            if epoch>0:
                L_ = self.loss_epoch[epoch-1]
            loss = loss_fnc(self.val(self.x),self.y_true)#self.cost(self.X,self.modelu(X))+ torch.mean(self.terminal(update(self.X,self.modelu(X))))#
            loss.backward()
            optimizer.step()
            L = loss.clone().detach().numpy()
            self.loss_epoch.append(L)
            if (epoch % int(max_num_epochs/2)== int(max_num_epochs/2)-1) | (epoch == 0):
                print('At epoch {}, mean loss is {:.2E}.'.format(epoch+1,L))
                self.time_display(t_0, t_1)
            epoch += 1
        t_delta = time.time()-t_0
        print(r'Training took {} epochs and {:,} ms and the final loss is {:.2E}.'.format(epoch,round(1000*(t_delta),2),loss))
        
    def time_display(self, t_0, t_1):
        print(r'Training this epoch takes {:,} ms. So far: {:,} ms in training.'.format(round(1000*(time.time()-t_1),2),round(1000*(time.time()-t_0),2)))
    
    def __call__(self, x):
        # print(self.val(x).shape) 
        return self.val(x).squeeze(-1)
                
        