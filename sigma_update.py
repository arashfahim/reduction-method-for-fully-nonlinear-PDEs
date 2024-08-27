import torch # type: ignore
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from neuralnets import sigmanet

# the sigma update helps memory useage stay constant at the cost of time

class sigma_update(object):
    def __init__(self, pde, sim, data , sigma, dir) -> None:
        self.sigma = sigma
        self.dir = dir
        self.dim = pde['dim']
        self.data = data
        self.new_sigma = sigmanet(pde,sim)
        
        