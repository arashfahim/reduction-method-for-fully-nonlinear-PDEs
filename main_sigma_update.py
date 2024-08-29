import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import time
# from IPython.display import display, Markdown

import equation as eqn
# from neuralnets import Ynet
# from neuralnets import Znet
# from neuralnets import Ytnet
from samplepaths import data_gen
import sigma_update as su # type: ignore
import coeff as cf
# import functions
import equation as eqn
# import visuals
import time

from absl import app # type: ignore
# import pickle as pk
import os
import json




           

def main(argv):
    del argv
    pde_params={'dim':2,
                'kappa':[0.,1.,0.8,0.6,0.4,0.5,0.3,0.2,0.1,0.7,1.,0.8,0.6,0.4,0.5,0.3,0.2,0.1,0.7,1.,0.8,0.6,0.4,0.5,0.3,0.2,0.1,0.7], # The first kappa=0 because the drift of wealth process is zero
                'theta':[0.,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1],
                # 'nu':[0.02,0.015,0.11,0.12,0.01,0.013,0.14,0.14,0.01,], #Hung's params
                'nu':[0.0, 0.2,0.15,0.11,0.12,0.1,0.13,0.14,0.14,0.1,0.2,0.15,0.11,0.12,0.1,0.13,0.14,0.14,0.1,0.2,0.15,0.11,0.12,0.1,0.13,0.14,0.14,0.1],# we do not like vanishing diffusion coefficient
                # 'lb':[0.,0.15,0.11,0.12,0.13,0.15,0.11,0.12,0.13,0.15],  # Hung's params
                'lb':[0.,1.15,1.11,0.12,0.13,0.15,0.11,0.12,0.13,0.15,1.15,1.11,0.12,0.13,0.15,0.11,0.12,0.13,0.15,1.15,1.11,0.12,0.13,0.15,0.11,0.12,0.13,0.15], # New params Make closed form solution more sensitive to time
                'rho':[0.,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                'eta':1.,
                'T': 1.,#torch.tensor([1.]).to(device),
        }
    t0 = time.time()
    num_samples = 2**14
    num_time_intervals = 20
    max_dim = 10
    size = num_samples* max_dim * num_time_intervals
    iid = torch.randn(size=[size]).to(device)
    print("It takes {:.0f} ms to generate {:,} iid samples.".format(round(1000*(time.time()-t0),6),size))
    sim_params={'num_samples':2**9,
            'num_time_intervals': 20,
            'iid':iid,
            'start' : 0.0,  
            'end' : 1.0,
            'num_neurons':4
            }
    
    
    num_ite = 10
    bounds = [.5 for n in range(num_ite+1)]# bounds
    
    path = os.path.dirname(__file__)
    
    timestr = time.strftime("%Y%m%d-%H%M")
    file = os.path.join(path,"ite_"+str(pde_params['dim'])+"_"+str(bounds[0])+"_"+timestr)
    output_dict = {}
    
    output_dict['pde'] = pde_params
    
    output_dict['simulation'] = {'num_samples':sim_params['num_samples'],
                                'num_time_intervals': sim_params['num_time_intervals'],
                                # 'iid':iid, everything but iid
                                'start' : sim_params['start'],  
                                'end' : sim_params['end'],
                                'num_neurons': sim_params['num_neurons']
                                    }

    m = cf.OU_drift_semi(pde_params)
    rand_diff = torch.tensor([2.])
    semi_diff = cf.custom_diff(pde_params,rand_diff)
    k = cf.zero_discount(pde_params)
    g = cf.exponential_terminal(pde_params)
    F = cf.f_driver(pde_params, ChesneyScott = False)
    

    output_dict['optimal'] = (m.lb_norm/m.eta).item()
    
    output_dict['bounds'] = bounds
    
    
    semi = eqn.semilinear(semi_diff,m,F,k,g,pde_params,sim_params)

    
    sigma = semi_diff
    for i in range(sim_params['num_time_intervals']):
        if i == 0:
            t = sigma(semi.x[:,:,i]).squeeze(-1)
        else:
            t = torch.cat((t,sigma(semi.x[:,:,i]).squeeze(-1)),axis=0)
    output_dict[0] = {}
    output_dict[0]['ell'] = {'min':0.,
                            'mean':0.,
                            'median':0.,
                            'max':0.,
                            'std':0.
                            }
    output_dict[0]['sigma'] = {'min':t[:,0,0].min().clone().detach().numpy().item(),
                            'mean':t[:,0,0].mean().clone().detach().numpy().item(),
                            'median':t[:,0,0].median().clone().detach().numpy().item(),
                            'max':t[:,0,0].max().clone().detach().numpy().item(),
                            'std':t[:,0,0].std().clone().detach().numpy().item()
                            }
    
    print("semi 1")
    semi.train(lr=1e-2,delta_loss=1e-10,max_num_epochs=2500)
    

    with open(file+".json", "w") as outfile: 
        json.dump(output_dict, outfile) 


    for j in range(1,num_ite+1):
        ell = cf.direction(pde_params,semi.Yt,semi_diff, bound = bounds[j-1])
    
        for i in range(sim_params['num_time_intervals']):
            if i == 0:
                t = torch.abs(ell(semi.x[:,:,i])).squeeze(-1)
            else:
                t = torch.cat((t,torch.abs(ell(semi.x[:,:,i])).squeeze(-1)),axis=0)
        output_dict[j] = {}
        output_dict[j]['ell'] = {'min':t[:,0,0].min().clone().detach().numpy().item(),
                                'mean':t[:,0,0].mean().clone().detach().numpy().item(),
                                'median':t[:,0,0].median().clone().detach().numpy().item(),
                                'max':t[:,0,0].max().clone().detach().numpy().item(),
                                'std':t[:,0,0].std().clone().detach().numpy().item()
                                }
        print(output_dict[j])
        
        with open(file+".json", "w") as outfile: 
            json.dump(output_dict, outfile) 
            
            
        diff_tmp = semi_diff + ell
        semi = eqn.semilinear(diff_tmp,m,F,k,g,pde_params,sim_params)
        
        new_sigma = su.sigma_update(pde_params,sim_params,semi.x,diff_tmp)
        new_sigma.train()
        
        semi_diff = cf.custom_diff(pde_params,new_sigma)
        
        sigma = semi_diff
        for i in range(sim_params['num_time_intervals']):
            if i == 0:
                t = sigma(semi.x[:,:,i]).squeeze(-1)
            else:
                t = torch.cat((t,sigma(semi.x[:,:,i]).squeeze(-1)),axis=0)
        output_dict[j]['sigma'] = {'min':t[:,0,0].min().clone().detach().numpy().item(),
                                'mean':t[:,0,0].mean().clone().detach().numpy().item(),
                                'median':t[:,0,0].median().clone().detach().numpy().item(),
                                'max':t[:,0,0].max().clone().detach().numpy().item(),
                                'std':t[:,0,0].std().clone().detach().numpy().item()
                                }
        print(output_dict[j])
        print("semi "+str(j+1))
        
        
        
        if j < num_ite:
            semi = eqn.semilinear(semi_diff,m,F,k,g,pde_params,sim_params)
            semi.train(lr=1e-2,delta_loss=1e-10,max_num_epochs=2500)
            
            ell = cf.direction(pde_params,semi.Yt,semi_diff, bound = bounds[j-1])
    
        print("Written on {}.json".format(file))
        with open(file+".json", "w") as outfile: 
            json.dump(output_dict, outfile) 
        
          

if __name__ == '__main__':
    app.run(main)            