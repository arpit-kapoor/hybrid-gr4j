from __future__ import absolute_import

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.linen.initializers import constant

class ProductionStorage(nn.Module):

    s_init:float=0.5
    x1_init:float=800.0
    scale: float=1000.0
    
    def setup(self):
        # self.s = self.param('s', constant(self.s_init), (1,))
        self.x1 = self.param('x1', constant(self.x1_init), (1,))

    def calculate_precip_store(self, s_store, p_n, x1):
        """Calculates the amount of rainfall which enters the storage reservoir."""
        n = x1*(1 - (s_store / x1)**2) * nn.tanh(p_n/x1)
        d = 1 + (s_store / x1) * nn.tanh(p_n / x1)
        return n/d

    # Determines the evaporation loss from the production store
    def calculate_evap_store(self, s_store, e_n, x1):
        """Calculates the amount of evaporation out of the storage reservoir."""
        n = s_store * (2 - s_store / x1) * nn.tanh(e_n/x1)
        d = 1 + (1- s_store/x1) * nn.tanh(e_n / x1)
        return n/d

    def time_update(self, carry, t_input):

        precip_difference = t_input[0] - t_input[1]
        p_n  = nn.leaky_relu(precip_difference)
        e_n  = nn.leaky_relu(-precip_difference)

        p_s = self.calculate_precip_store(carry['s_store'], p_n, carry['x1'])
        e_s = self.calculate_evap_store(carry['s_store'], e_n, carry['x1'])

        # update the s store
        tmp_s_store = carry['s_store'] + p_s - e_s

        # calculate percolation from actual storage level
        perc = tmp_s_store * (1 - (1 + (4/9 * tmp_s_store / carry['x1'])**4)**(-0.25))

        # final update of the production store for this timestep
        carry['s_store'] = tmp_s_store - perc


        out = jnp.concatenate([t_input, p_n.reshape((1,)), e_n.reshape((1,)), p_s, perc, carry['s_store']], axis=0)

        return carry, out 
    
    def __call__(self, x):
        
        carry = {
            's_store':  self.s_init * self.x1 * self.scale,
            'x1': self.x1 * self.scale
        }
        carry_out, outputs = jax.lax.scan(self.time_update, carry, x)
        return outputs[:, :-1], outputs[:, -1]