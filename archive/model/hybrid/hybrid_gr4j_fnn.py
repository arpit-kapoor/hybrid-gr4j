import torch
import torch.nn as nn

import torch.distributions as dist


class HybridGR4J(nn.Module):
    """Hybridization class for GR4J using a feed-forward 
    neural network to replace the routing storage
    """
    
    def __init__(self, s_init):
        
        super(HybridGR4J, self).__init__()

        # GR4J param
        self.s_init = torch.tensor(s_init)
        self.x1 = nn.Parameter(dist.uniform.Uniform(100, 1200).sample())
        self.register_parameter(name='x1', param=self.x1)

        # NN
        self.h1 = nn.Linear(4, 32)
        self.h2 = nn.Linear(32, 32)
        self.h3 = nn.Linear(32, 4)
        self.o = nn.Linear(4, 1)


    def forward(self, x):
        """Forward pass through the hybrid model"""

        # Extract precipitation and evapotranspiration
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
        inp = torch.concat([p_n, e_n, p_s, perc], dim=1)

        # ANN
        out = torch.relu(self.h1(inp.float()))
        out = torch.relu(self.h3(self.h2(out)))
        out = self.o(out)

        return out