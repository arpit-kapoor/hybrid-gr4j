import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_gui
import torch.distributions as dist


class ProductionStorage(nn.Module):
    
    def __init__(self, s_init: float=0.0, x1: torch.Tensor=None,
                mu: torch.Tensor=None, sigma: torch.Tensor=None) -> None:
        super(ProductionStorage, self).__init__()
        self.s_init = s_init
        if x1 is None:
            x1 = dist.uniform.Uniform(100, 1000).sample()
        self.x1 = x1
        self.mu = mu
        self.sigma = sigma

    def set_x1(self, value: float) -> None:
        self.x1 = torch.tensor(value, dtype=torch.float)
    
    def get_x1(self):
        return self.x1.detach().numpy()

    def forward(self, x):
        # Precip and Evaporanspiration
        P = x[:, 0]
        E = x[:, 1]
        
        # Number of simulation timesteps
        num_timesteps = len(P)
        
        # Unpack the model parameters
        x1 = self.x1

        # Production Storage
        p_n = torch.relu(P - E)
        e_n = torch.relu(E - P)

        p_s_list = []
        e_s_list = []
        perc_list = []
        s_store_list = []

        s_store = self.s_init * x1

        for t in range(num_timesteps):
            # calculate fraction of netto precipitation that fills
            #  production store (eq. 3)
            p_s = x1 * (1 - (s_store/ x1)**2) * torch.tanh(p_n[t]/x1) / (1 + (s_store / x1) * torch.tanh(p_n[t] / x1))

            # from the production store (eq. 4)
            e_s = s_store * (2 - s_store/x1) * torch.tanh(e_n[t]/x1) / (1 + (1 - s_store/x1) * torch.tanh(e_n[t] / x1))

            s_store = s_store + p_s - e_s

            # calculate percolation from actual storage level
            perc = s_store * (1 - (1 + (4/9 * s_store / x1)**4)**(-0.25))
            
            # final update of the production store for this timestep
            s_store = s_store - perc

            # Append updated values
            p_s_list.append(p_s)
            e_s_list.append(e_s)
            perc_list.append(perc)
            s_store_list.append(s_store)
        

        # Expand dim
        p_n = p_n[:, None]
        e_n = e_n[:, None]
        p_s = torch.stack(p_s_list)[:, None] 
        perc = torch.stack(perc_list)[:, None]
        
        # Concatenate
        out = torch.concat([x, p_n, e_n, p_s, perc], dim=1)

        # Scale
        if self.mu is None and self.sigma is None:
            self.mu = out.mean(dim=0)
            self.sigma = out.std(dim=0)

        out = (out - self.mu)/self.sigma

        return out
