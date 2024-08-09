import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import time
from IPython.display import display, Markdown

import neuralnets.Ynet as Ynet
import neuralnets.Znet as Znet
import neuralnets.Ytnet as Ytnet
import samplepaths.data_gen as data_gen
import coeff
import functions

# from mpl_toolkits.mplot3d import Axes3D

class linear(object):
    def __init__(self,sigma,mu,source,kappa,terminal,pde,sim):        
        self.Y0 = Ynet() # NN for value at t=0 
        self.Z = Znet() # NN for gradient at all times
        self.Yt = Ytnet() # NN for value function at all times, required to update sigma
        self.terminal = terminal # terminal condition
        self.loss_epoch = [] # list to keep the loss at each training epoch
        self.epoch=0 # initializing epoch to zero
        # self.num_epochs = 5000 # total number of epochs
        self.mu = mu # drift of the SDE
        self.sigma = sigma # diffusion coef. for SDE
        self.kappa = kappa # discount factor
        self.source = source # source term for the PDE
        self.n = sim['num_time_intervals'] # number of time intervals
        self.dim = pde['dim']
        data = data_gen(sigma,mu,pde,sim)
        self.dt = data.dt.to(device)
        self.x = data.x.to(device).clone().detach()
        self.sigmadw = data.sigmadw.to(device).clone().detach()
        self.r = torch.ones((self.x.shape[0],1,self.n+1)).to(device)
        self.c = torch.ones((self.x.shape[0],1,self.n+1)).to(device)
        for i in range(self.n):
            self.r[:,:,i+1] = self.r[:,:,i]* torch.exp(-self.kappa(self.x[:,:,i])*self.dt)
            if i == self.n -1 :
                self.c[:,:,i+1] = self.terminal(self.x[:,1:,i+1])
            self.c[:,:,i] = self.source(self.x[:,:,i])
        self.r = self.r.clone().detach()
        self.c = self.c.clone().detach()
        self.trained = False
        
    def loss(self):
        # self.Zsigmadw = torch.zeros((num_samples,1,n)).to(device)
        for i in range(self.n):   
            if i == 0:
                Y =  self.Y0(self.x[:,1:,0])
            else:
                Y = Y*self.r[:,:,i] + self.c[:,:,i]*self.dt + torch.bmm(self.Z(self.x[:,:,i]).unsqueeze(1),self.sigmadw[:,:,i].unsqueeze(2)).squeeze(2)
        return torch.pow(self.c[:,:,-1]-Y,2).mean()
        
    def train(self,lr,delta_loss,max_num_epochs):
        t_0 = time.time()
        self.lr = lr
        parameters = list(self.Y0.parameters()) + list(self.Z.parameters())
        optimizer = optim.Adam(parameters, self.lr)
        L_ = torch.Tensor([-2.0])
        loss = torch.Tensor([2.0])
        while (torch.abs(L_-loss)>delta_loss) & (self.epoch < max_num_epochs):# epoch in range(num_epochs):
            t_1 = time.time()
            optimizer.zero_grad()
            if self.epoch>0:
                L_ = self.loss_epoch[self.epoch-1]
            loss= self.loss()#self.cost(self.X,self.modelu(X))+ torch.mean(self.terminal(update(self.X,self.modelu(X))))#
            loss.backward()
            optimizer.step()
            self.loss_epoch.append(loss)
            if (self.epoch % int(max_num_epochs/10)== int(max_num_epochs/10)-1) | (self.epoch == 0):
                print("At epoch {}, mean loss is {:.2E}.".format(self.epoch+1,loss.detach()))
                self.time_display(t_0, t_1)
            self.epoch += 1
        print("Training took {} epochs and {:,} ms and the final loss is {:.2E}.".format(self.epoch,round(1000*(time.time()-t_0),2),loss))
        self.trained = True
        self.value_fnc(lr=1e-2,delta_loss=delta_loss,max_num_epochs=1000,num_batches=10)

    def time_display(self, t_0, t_1):
        print("Training this epoch takes {:,} ms. So far: {:,} ms in training.".format(round(1000*(time.time()-t_1),2),round(1000*(time.time()-t_0),2)))
        
        
    def value_fnc(self,lr,delta_loss,max_num_epochs,num_batches):
        t_0 = time.time()
        if self.trained == False:
            print("The neural nets are not trained yet. Train the neural nets by running self.train(lr,delta_loss,max_num_epochs).")
        else:
            for i in range(self.n):   
                if i == 0:
                    Y =  self.Y0(self.x[:,1:,0])
                    x_data = self.x[:,:,i]
                    y_data = Y
                else:
                    if i == self.n - 1:
                        Y = self.terminal(self.x[:,1:,i+1])
                        x_data = torch.cat((x_data,self.x[:,:,i+1]),axis=0)
                        y_data = torch.cat((y_data,Y),axis=0)
                        #evaluate and reuse self.sigma(self.x[:,:,i]).reshape((self.num_samples,dim,dim))[:,0,0] 
                    Y = Y*self.r[:,:,i] + self.c[:,:,i]*self.dt + torch.bmm(self.Z(self.x[:,:,i]).unsqueeze(1),self.sigmadw[:,:,i].unsqueeze(2)).squeeze(2)        
                    x_data = torch.cat((x_data,self.x[:,:,i]),axis=0)
                    y_data = torch.cat((y_data,Y),axis=0)
            print("Data for value function is gathered in {:,} ms.".format(round(1000*(time.time()-t_0),2)))
            perm = torch.randperm(x_data.shape[0])
            y_data = y_data[perm,:].clone().detach()
            x_data = x_data[perm,:].clone().detach()
            parameters = self.Yt.parameters()
            optimizer = optim.Adam(parameters, lr)
            L_ = torch.Tensor([-2.0])
            loss = torch.Tensor([2.0])
            initiation = True # to print the first epoch or last epoch in multiple rounds of training
            epoch=0
            mse = nn.MSELoss()
            batch_size = int(x_data.shape[0]/num_batches)
            batch_epochs = int(max_num_epochs/num_batches)
            b=0
            # loss_epoch = []
            # max_num_epochs = 500
            t_0 = time.time()
            while (torch.abs(L_-loss)>delta_loss) & (epoch < max_num_epochs):# epoch in range(num_epochs):
                optimizer.zero_grad()
                if epoch>0:
                    L_ = loss
                index = [b*batch_size, x_data.shape[0]] if (b+1==num_batches) else [b*batch_size, (b+1)*batch_size]    
                loss= mse(self.Yt(x_data[index[0]:index[1],:]),y_data[index[0]:index[1],:])
                loss.backward()
                optimizer.step()
                if epoch == (b+1)*batch_epochs:
                    print("At epoch {:,} batch {} is used.".format(epoch+1,b+2))
                if epoch >= (b+1)*batch_epochs:
                    b = b + 1
                
                if initiation:
                    initiation = False
                    print("\nFitting a neural net to the data initiated.")
                    print("The L^2 error of the fitted value function at epoch {} is {:.2E}.".format(epoch+1,torch.sqrt(loss.detach())))
                if (epoch % int(max_num_epochs/5)== int(max_num_epochs/5)-1):    
                    print("The $L^2$-norm of the error of the fitted value function at epoch {} is {:.2E}.".format(epoch+1,torch.sqrt(loss.detach())))
                epoch += 1
            print("Value function is evaluated in {:,} ms.".format(round(1000*(time.time()-t_0),2)))