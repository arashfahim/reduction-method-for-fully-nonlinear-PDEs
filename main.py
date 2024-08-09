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
import coeff as cf
import functions
import equation as eqn

from absl import app

def main(argv):
    del argv
    pde_params={'dim':2,
                'kappa':torch.tensor([0.,1.,0.8,0.6,0.4,0.5,0.3,0.2,0.1,0.7]).to(device), # The first kappa=0 because the drift of wealth process is zero
                'theta':torch.tensor([0.,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1]).to(device),
                'nu':torch.tensor([0.02,0.015,0.11,0.12,0.01,0.013,0.14,0.14,0.01]).to(device),
                # 'lb':torch.tensor([0.,0.15,0.11,0.12,0.13,0.15,0.11,0.12,0.13,0.15]).to(device),   Hung's params
                'lb':torch.tensor([0.,1.15,1.11,0.12,0.13,0.15,0.11,0.12,0.13,0.15]).to(device), # New params Make closed form solution more sensitive to time
                'rho':torch.tensor([0.,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).to(device),
                'eta':torch.tensor([1.]).to(device),
                'T': torch.tensor([1.]).to(device),
        }
    t0 = time.time()
    num_samples = 2**16
    num_time_intervals = 30
    max_dim = 10
    size = num_samples* max_dim * num_time_intervals
    iid = torch.randn(size=[size]).to(device)
    print("It takes {:.0f} ms to generate {:,} iid samples.".format(round(1000*(time.time()-t0),6),size))
    sim_params={'num_samples':2**11,
          'num_time_intervals': 20,
          'iid':iid,
          'start' : torch.tensor([0.0]),  
          'end' : torch.tensor([1.0]),
          'num_neurons':8        
          }
    
    # v = sample_Znet()
    # s = sigma1D(pde_params)
    # sigma = NN_diff_1D(pde_params,diff = s) 
    mu = cf.OU_drift_semi(pde_params)
    off_diff = torch.tensor([1.0]) #choose larger diffusion coef to make the difference between the terminal and time-zero solution larger
    sigma = cf.constant_diff(pde_params,constant_diff = off_diff)# With large constant_diff the trainning sucks!
    # mu = OU_drift_lin(pde_params,v,sigma)

    k = cf.zero_discount(pde_params)
    f = cf.zero_source(pde_params)
    # f = custom_source(pde_params,off_diff)
    g = cf.exponential_terminal(pde_params)
    # g = zero_terminal(pde_params)
    
    heat = eqn.linear(sigma,mu,f,k,g,pde_params,sim_params)
    
    heat.train(lr=1e-2,delta_loss=1e-10,max_num_epochs=5000)

if __name__ == '__main__':
    app.run(main)    