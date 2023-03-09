from __future__ import absolute_import
from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp

from flax import linen as nn


class ConvNet(nn.Module):

    out_dim: int=1
    n_filters: Tuple[int, int, int]=(8, 8, 8)
    dropout_p: float=0.2
    training: bool=True

    
    def setup(self):
        # Define Layers
        self.conv_1 = nn.Conv(features=self.n_filters[0], 
                              kernel_size=(2, 1),
                              use_bias=True, 
                              padding='valid')
        
        self.conv_2 = nn.Conv(features=self.n_filters[1], 
                              kernel_size=(3, 3),
                              use_bias=True, 
                              padding='valid')

        self.conv_3 = nn.Conv(features=self.n_filters[2], 
                              kernel_size=(2, 2),
                              use_bias=True, 
                              padding='valid')

        self.dropout = nn.Dropout(rate=self.dropout_p, 
                                 deterministic=not self.training)

        self.dense = nn.Dense(features=self.out_dim)

    def __call__(self, x):
        out = nn.relu(self.conv_1(x))
        out = nn.relu(self.conv_2(out))
        out = nn.max_pool(out, window_shape=(2, 1), strides=(2, 1))
        out = nn.relu(self.conv_3(out))
        out = self.dropout(out.reshape(out.shape[0], -1))
        out = self.dense(out)
        return out


if __name__=='__main__':
    net = ConvNet(dropout_p=0.5)
    init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
    print(net.tabulate(init_rngs, jnp.ones((5, 7, 20, 1))))