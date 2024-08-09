import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 10
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class display_it(object):
    def __init__(self,eqn,**kwargs):
        if eqn.dim == 2:
            f,ax = plt.subplots(1,1,figsize=(6,6),dpi=300);
            ax = plt.axes(projection='3d')
            ax.set_xlabel('wealth')
            ax.set_ylabel('volatility');
            
            fake2Dline1 = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o',alpha=0.5)#fitted
            fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle="none", c='gray', marker = 'o',alpha=0.5)#terminal

            fake_leg = [fake2Dline1, fake2Dline2]
            leg_labels = ["Fitted","Terminal"]
            
            steps =100
            x = torch.linspace(0.,1.,steps=steps)
            y = torch.linspace(0.,1.,steps=steps)
            xy = torch.cartesian_prod(x,y)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            v_T = eqn.terminal(xy).detach().reshape(steps,steps).squeeze(-1)
            surf = ax.plot_surface(X.numpy(),Y.numpy(),v_T.numpy(),alpha=0.5,label='Terminal',color='gray');

            if kwargs:
                if 't' in kwargs.keys():
                    t = kwargs['t']
                    txy = torch.cat((t*torch.ones(xy.shape[0],1),xy),axis=1)
                else:
                    txy = torch.cat((torch.zeros(xy.shape[0],1),xy),axis=1)
                    v = eqn.Y0(xy).detach().reshape(steps,steps).squeeze(-1)
                    surf = ax.plot_surface(X.numpy(),Y.numpy(),v.numpy(),alpha=0.5,label="approximation",color='b');
                    fake2Dline3 = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o',alpha=0.5)#approximation
                    fake_leg.append(fake2Dline3)
                    leg_labels.append("Approximation at 0")
                    

            
            vt = eqn.Yt(txy).detach().reshape(steps,steps).squeeze(-1)
            surf = ax.plot_surface(X.numpy(),Y.numpy(),vt.numpy(),alpha=0.5,label="fitted",color='r');
            
            if kwargs:
                if 'closed_form' in kwargs.keys():
                    sol = kwargs['closed_form']
                    v_c = sol(txy).detach().reshape(steps,steps).squeeze(-1)                
                    surf = ax.plot_surface(X.numpy(),Y.numpy(),v_c.numpy(),alpha=0.5,label='closed-form',color='g');
                    leg_labels.append("Closer-from")
                    fake2Dline4 = mpl.lines.Line2D([0],[0], linestyle="none", c='g', marker = 'o',alpha=0.5)                
                    fake_leg.append(fake2Dline4)
            
            
            if kwargs:
                if 'zlim' in kwargs.keys():
                    ax.set_zlim(kwargs['zlim'])
                if 'legend_loc' in kwargs.keys():
                    legend_loc = kwargs['legend_loc']
                else:
                    legend_loc=(0.0,0.8)

            
            ax.legend(fake_leg, leg_labels, numpoints = 1, loc=legend_loc)
            plt.tight_layout(rect=[0,0,1,1]);      
        if kwargs:
            data = torch.rand(2**14,eqn.dim)
            if 't' in kwargs.keys():
                t = kwargs['t']
                tdata = torch.cat((t*torch.ones(data.shape[0],1),data),axis=1)
                v_t = eqn.Yt(tdata).detach().squeeze(-1)
                if 'closed_form' in kwargs.keys():
                    sol = kwargs['closed_form']
                    v_c = sol(tdata).detach().squeeze(-1)
                    print("The $L^2$ distance between the fitted value function at time {:.2f} and closed-form is  {:.3E}.".format(t,pow(v_c-v_t,2).mean()))
            else:
                v_0 = eqn.Y0(data).detach().squeeze(-1)        
                tdata = torch.cat((torch.zeros(data.shape[0],1),data),axis=1)
                v_t = eqn.Yt(tdata).detach().squeeze(-1)     
                if 'closed_form' in kwargs.keys():
                    sol = kwargs['closed_form']
                    v_c = sol(tdata).detach().squeeze(-1)              
                    print("The $L^2$ distance between the fitted value function at time 0 and closed-form is  {:.3E}.".format(pow(v_c-v_t,2).mean()))
                    print("The $L^2$ distance between the approximated value function at time 0 and closed-form for optimal $\sigma$  is  {:.3E}.".format(pow(v_c-v_0,2).mean()))
               

        
