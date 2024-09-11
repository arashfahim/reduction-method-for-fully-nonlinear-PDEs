import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
# from pylab import *



plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": ["Helvetica"],
    'text.latex.preamble' : r'\usepackage{amssymb} \usepackage{amsmath}' #for \text command
    })
import torch
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class distance_to_optimal(object):
    def __init__(self,sigma,optimal_sigma,pde, sim):
        self.sim = sim
        self.approx = sigma
        self.target = optimal_sigma
        step = pde['T']/sim['num_time_intervals']
        dim = pde['dim']
        sample_size = 2**10
        for t in np.arange(stop=pde['T']+step,step=step):
            t_ = torch.tensor(t).repeat(sample_size,1)
            if t == 0:
                data = torch.cat((t_,torch.rand(sample_size,dim)),axis=-1)
            else:
                tmp_ = torch.cat((t_,torch.rand(sample_size,pde['dim'])),axis=-1)
                data = torch.cat((data,tmp_),axis=0)
        self.err = torch.sqrt(torch.pow(sigma(data) - optimal_sigma(data),2)).mean()
        print('The L^2 norm of difference between current sigma and optimal sigma is {:.2E}.'.format(self.err.item()))

        
    def __call__(self,t,**kwargs):
        steps =100
        x = torch.linspace(self.sim['start'].item(),self.sim['end'].item(),steps=steps)
        y = torch.linspace(self.sim['start'].item(),self.sim['end'].item(),steps=steps)
        xy = torch.cartesian_prod(x,y)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        txy = torch.cat((t*torch.ones(xy.shape[0],1),xy),axis=1)
        self.s_opt = self.target(txy)[:,0,0].detach().reshape(steps,steps)
        self.s_new = self.approx(txy)[:,0,0].detach().reshape(steps,steps)
        f = plt.figure(figsize=(6,6),dpi=300);
        ax = f.add_subplot(111, projection='3d');
        ax.plot_surface(self.X.numpy(),self.Y.numpy(),self.s_opt.numpy(),alpha=0.5,label='optimal',color='b');
        ax.plot_surface(self.X.numpy(),self.Y.numpy(),self.s_new.numpy(),alpha=0.5,label='new',color='r');
        legend_loc=(0.0,0.8)
        if kwargs:
            if 'zlim' in kwargs.keys():
                ax.set_zlim(kwargs['zlim'])

            if 'legend_loc' in kwargs.keys():
                legend_loc = kwargs['legend_loc']

        ax.legend(loc=legend_loc)  
        f.suptitle(r'Difference between optimal $\sigma$ vs approximation at time t={:.2f}'.format(t))
        
        


class loss_plot(object):
    def __init__(self,eqn,**kwargs):
        f_log= plt.figure(dpi=300)
        f_log.suptitle(r'Loss function vs epochs')
        ax_log = f_log.add_subplot(111)
        ax_log.plot(np.log(np.arange(len(eqn.loss_epoch))+1),[np.log(l.detach().numpy()) for l in eqn.loss_epoch]);
        ax_log.set_xlabel('$\ln($epoch$)$')
        ax_log.set_ylabel('$\ln($loss$)$');
        if kwargs:
            path = os.path.join(kwargs['path'],"log_loss.png")
            if 'path' in kwargs:
                plt.savefig(path,bbox_inches='tight',dpi=300)
                



class display_it(object):
    def __init__(self,eqn,**kwargs):
        t = 0# default time. Change it if kwargs kick in later.
                
        if eqn.dim == 2:
            f = plt.figure(figsize=(6,6),dpi=300);
            ax = f.add_subplot(111, projection='3d')
            ax.set_xlabel('wealth')
            ax.set_ylabel('volatility');
            
            
            steps =100
            x = torch.linspace(eqn.params['start'],eqn.params['end'],steps=steps)
            y = torch.linspace(eqn.params['start'],eqn.params['end'],steps=steps)
            xy = torch.cartesian_prod(x,y)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            v_T = eqn.terminal(xy).detach().reshape(steps,steps).squeeze(-1)
            ax.plot_surface(X.numpy(),Y.numpy(),v_T.numpy(),alpha=0.5,label='Terminal',color='gray');
            txy = torch.cat((torch.zeros(xy.shape[0],1),xy),axis=1)
            if kwargs:
                if 't' in kwargs.keys():
                    t = kwargs['t']
                    txy = torch.cat((t*torch.ones(xy.shape[0],1),xy),axis=1)
                else:
                    v = eqn.Y0(xy).detach().reshape(steps,steps).squeeze(-1)
                    ax.plot_surface(X.numpy(),Y.numpy(),v.numpy(),alpha=0.5,label="approximation",color='b');
            else:
                v = eqn.Y0(xy).detach().reshape(steps,steps).squeeze(-1)
                ax.plot_surface(X.numpy(),Y.numpy(),v.numpy(),alpha=0.5,label="approximation",color='b');

            
            vt = eqn.Yt(txy).detach().reshape(steps,steps).squeeze(-1)
            ax.plot_surface(X.numpy(),Y.numpy(),vt.numpy(),alpha=0.5,label="fitted",color='r');
            
            if kwargs:
                if 'closed_form' in kwargs.keys():
                    sol = kwargs['closed_form']
                    v_c = sol(txy).detach().reshape(steps,steps).squeeze(-1)                
                    ax.plot_surface(X.numpy(),Y.numpy(),v_c.numpy(),alpha=0.5,label='closed-form',color='g');
 
            
            legend_loc=(0.0,0.8)
            if kwargs:
                if 'zlim' in kwargs.keys():
                    ax.set_zlim(kwargs['zlim'])
                if 'legend_loc' in kwargs.keys():
                    legend_loc = kwargs['legend_loc']


            ax.legend(loc=legend_loc)  
            f.suptitle(r'At time t={:.2f}'.format(t))

            # path = os.path.join(kwargs['path'],"figure_"+str(t)+"_.png")
            # if 'path' in kwargs:
            #     plt.savefig(path)
            # plt.show();
                
        if kwargs:
            data = eqn.params['start']+(eqn.params['end']-eqn.params['start'])*torch.rand(2**14,eqn.dim)
            if 't' in kwargs.keys():
                t = kwargs['t']
                tdata = torch.cat((t*torch.ones(data.shape[0],1),data),axis=1)
                v_t = eqn.Yt(tdata).detach().squeeze(-1)
                if 'closed_form' in kwargs.keys():
                    sol = kwargs['closed_form']
                    v_c = sol(tdata).detach().squeeze(-1)
                    print(r'The $L^2$ distance between the fitted value function at time {:.2f} and closed-form is  {:.3E}.'.format(t,pow(v_c-v_t,2).mean()))
                    # eqn.params['L2_fit_cf'] = [t,pow(v_c-v_t,2).mean()]
            else:
                v_0 = eqn.Y0(data).detach().squeeze(-1)        
                tdata = torch.cat((torch.zeros(data.shape[0],1),data),axis=1)
                v_t = eqn.Yt(tdata).detach().squeeze(-1)     
                if 'closed_form' in kwargs.keys():
                    sol = kwargs['closed_form']
                    v_c = sol(tdata).detach().squeeze(-1)              
                    print(r'The $L^2$ distance between the fitted value function at time 0 and closed-form is  {:.3E}.'.format(pow(v_c-v_t,2).mean()))
                    print(r'The $L^2$ distance between the approximated value function at time 0 and closed-form for optimal $\sigma$  is  {:.3E}.'.format(pow(v_c-v_0,2).mean()))
                # eqn.params['L2_fit_cf'] = [0.0,pow(v_c-v_t,2).mean()]
                # eqn.params['L2_v0_cf'] = [0.0,pow(v_c-v_0,2).mean()]
        
                    
               

        
