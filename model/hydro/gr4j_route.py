import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_gui
import torch.distributions as dist


class RoutingStorage(nn.Module):
    
    def __init__(self, r_init: float=0.0, x2: torch.Tensor=None,
                x3: torch.Tensor=None, x4: torch.Tensor=None,
                mu: torch.Tensor=None, sigma: torch.Tensor=None) -> None:
        
        super(RoutingStorage, self).__init__()

        # Default value of routing storage
        self.r_init = r_init
        
        # Sample parameters if not provided
        if x2 is None:
            x2 = dist.uniform.Uniform(-3, 5).sample()
        if x3 is None:
            x3 = dist.uniform.Uniform(20, 300).sample()
        if x4 is None:
            x4 = dist.uniform.Uniform(1, 3).sample()
        
        # Set param values
        self.set_params(x2, x3, x4)
        
        # Scaling params
        self.mu = mu
        self.sigma = sigma

    def set_params(self, x2: float, x3: float, x4: float) -> None:
        self.x2 = torch.tensor(x2, dtype=torch.float)
        self.x3 = torch.tensor(x3, dtype=torch.float)
        self.x4 = torch.tensor(x4, dtype=torch.float)
    
    def get_params(self):
        x2 = self.x2.detach().numpy()
        x3 = self.x3.detach().numpy()
        x4 = self.x4.detach().numpy()
        return x2, x3, x4

    def _s_curve1(self, t, x4):
        """Calculate the s-curve of the unit-hydrograph 1.
        
        Args:
            t: timestep
            x4: model parameter x4 of the gr4j model.
            
        """
        if t <= 0:
            return 0.
        elif t < x4:
            return (t / x4)**2.5
        else:
            return 1.


    def _s_curve2(self, t, x4): 
        """Calculate the s-curve of the unit-hydrograph 2.
        
        Args:
            t: timestep
            x4: model parameter x4 of the gr4j model.
            
        """
        if t <= 0:
            return 0.
        elif t <= x4:
            return 0.5 * ((t / x4) ** 2.5)
        elif t < 2*x4:
            return 1 - 0.5 * ((2 - t / x4) ** 2.5)
        else:
            return 1.

    def forward(self, x):
        
        # Create separate tensors
        p_n = x[:, 0]
        e_n = x[:, 1]
        p_s = x[:, 2]
        perc = x[:, 3]

        # Number of timesteps in the data
        num_timesteps = x.shape[0]

        # Initialize routing storage
        r_store = torch.zeros(num_timesteps+1)
        r_store[0] = self.r_init * self.x3

        # Tensor to store streamflow
        qsim = torch.zeros(num_timesteps+1)
        
        # calculate number of unit hydrograph ordinates
        num_uh1 = int(torch.ceil(self.x4))
        num_uh2 = int(torch.ceil(2*self.x4 + 1))
        
        # calculate the ordinates of both unit-hydrographs (eq. 16 & 17)
        uh1_ordinates = torch.zeros(num_uh1)
        uh2_ordinates = torch.zeros(num_uh2)
        
        for j in range(1, num_uh1 + 1):
            uh1_ordinates[j - 1] = self._s_curve1(j, self.x4) - self._s_curve1(j - 1, self.x4)
            
        for j in range(1, num_uh2 + 1):
            uh2_ordinates[j - 1] = self._s_curve2(j, self.x4) - self._s_curve2(j - 1, self.x4)
        
        # arrys to store the rain distributed through the unit hydrographs
        uh1 = torch.zeros(num_uh1)
        uh2 = torch.zeros(num_uh2)

        # total quantity of water that reaches the routing
        p_r = perc + (p_n - p_s)
            
        # split this water quantity by .9/.1 for diff. routing (UH1 & UH2)
        p_r_uh1 = 0.9 * p_r 
        p_r_uh2 = 0.1 * p_r

        for t in range(1, num_timesteps+1):
            
            # update state of rain, distributed through the unit hydrographs
            for j in range(0, num_uh1 - 1):
                uh1[j] = uh1[j + 1] + uh1_ordinates[j] * p_r_uh1[t - 1]
            uh1[-1] = uh1_ordinates[-1] * p_r_uh1[t - 1]
            
            for j in range(0, num_uh2 - 1):
                uh2[j] = uh2[j + 1] + uh2_ordinates[j] * p_r_uh2[t - 1]
            uh2[-1] = uh2_ordinates[-1] * p_r_uh2[t - 1]
            
            # calculate the groundwater exchange F (eq. 18)
            gw_exchange = self.x2 * (r_store[t - 1] / self.x3) ** 3.5
            
            # update routing store
            r_store[t] = max(0, r_store[t - 1] + uh1[0] + gw_exchange)
            
            # outflow of routing store
            q_r = r_store[t] * (1 - (1 + (r_store[t] / self.x3)**4)**(-0.25))
            
            # subtract outflow from routing store level
            r_store[t] = r_store[t] - q_r
            
            # calculate flow component of unit hydrograph 2
            q_d = max(0, uh2[0] + gw_exchange)
            
            # total discharge of this timestep
            qsim[t] = q_r + q_d

        return qsim[1:].reshape(-1, 1), r_store[1:]

