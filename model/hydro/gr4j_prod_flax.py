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

    def time_update(self, carry, t_input):
        # first calculate the net precipitation effect on stores
        is_gain = t_input[0] >= t_input[1]
        p_n_gain = t_input[0] - t_input[1]
        p_n_loss = jnp.float32(0.0)
        pe_n_loss = t_input[1] - t_input[0]
        pe_n_gain = jnp.float32(0.0)

        # calculate the evaporation effect on production store
        e_s_gain = jnp.float32(0.0)
        e_s_loss = jnp.divide(
            carry['s_store'] * (2 - carry['s_store']/carry['x1']) * jnp.tanh(pe_n_loss / carry['x1']),
            1 + (1 - carry['s_store'] / carry['x1']) * jnp.tanh(pe_n_loss / carry['x1'])
        ) 

        # calculate the precipitation effect on production store
        p_s_gain = jnp.divide(
            carry['x1'] * (1 - (carry['s_store']  / carry['x1'])**2) * jnp.tanh(p_n_gain/carry['x1']),
            1 + (carry['s_store']  / carry['x1']) * jnp.tanh(p_n_gain / carry['x1'])
        ) 
        p_s_loss = jnp.float32(0.0) 

        # avoiding if statements
        e_n = pe_n_gain * is_gain + pe_n_loss * (1 - is_gain)
        p_n = p_n_gain * is_gain + p_n_loss * (1 - is_gain)
        p_s = p_s_gain * is_gain + p_s_loss * (1 - is_gain)
        e_s = e_s_gain * is_gain + e_s_loss * (1 - is_gain)

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