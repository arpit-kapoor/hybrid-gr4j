from __future__ import absolute_import
from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp

from flax import linen as nn




class LSTM(nn.Module):

    # input_dim: int
    lstm_dim: int
    hidden_dim: int
    output_dim: int
    dropout_p: float=0.5
    max_len: int=7
    training: bool=True
    
    def setup(self):

        # RNN layer
        self.lstm_layer = nn.scan(nn.OptimizedLSTMCell,
                                  variable_broadcast="params",
                                  split_rngs={"params": False},
                                  in_axes=1, 
                                  out_axes=1,
                                  length=self.max_len,
                                  reverse=False)

        self.lstm1 = self.lstm_layer()
        self.lstm2 = self.lstm_layer()
        
        # Fully-connected output layer
        self.fc1 = nn.Dense(self.hidden_dim)

        self.fc2 = nn.Dense(self.output_dim)

        self.do = nn.Dropout(rate=self.dropout_p, 
                             deterministic=not self.training)

    @nn.remat
    def __call__(self, x):

        batch_size, seq_size, input_size = x.shape

        carry, hidden = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), batch_dims=(len(x),), size=self.lstm_dim)
        (carry, hidden), x = self.lstm1((carry, hidden), x)

        carry, hidden = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), batch_dims=(len(x),), size=self.lstm_dim)
        (carry, hidden), x = self.lstm2((carry, hidden), x)

        out = nn.tanh(x[:, -2:, :].reshape(batch_size, -1))
        out = self.do(out)
        out = nn.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


if __name__=='__main__':

    net = LSTM(lstm_dim=64,
               hidden_dim=32,
               output_dim=1,
               dropout_p=0.2)

    init_rngs = {'params': jax.random.PRNGKey(0), 
                'dropout': jax.random.PRNGKey(1)}

    print(net.tabulate(init_rngs, jnp.ones((5, 7, 10))))