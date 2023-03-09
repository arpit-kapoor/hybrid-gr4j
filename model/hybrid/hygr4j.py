from __future__ import absolute_import
import sys

import numpy as np
import jax.numpy as jnp

from typing import Tuple
from flax import linen as nn

sys.path.append("~/project")

from model.hydro.flax import ProductionStorage
from model.ml.flax import ConvNet, LSTM

class HyGR4J(nn.Module):
    # Attributes
    dl_model: str
    x1_init: float
    s_init: float
    x_mu: np.ndarray
    x_sigma: np.ndarray
    n_filters: tuple=(8, 8, 6)
    hidden_dim: int = 16
    lstm_dim: int = 32
    output_dim: int = 1
    dropout_p: float=0.2
    window_size: int=7
    training: bool=True
    scale: float=1000.0

    def setup(self):
        """Function to setup class attributes
        """
        # GR4J production storage instance
        self._prod_store = ProductionStorage(x1_init=self.x1_init,
                                             s_init=self.s_init,
                                             scale=self.scale)
        
        # Deep Learning model
        if self.dl_model == 'CNN':
            self._model = ConvNet(n_filters=self.n_filters,
                                  dropout_p=self.dropout_p,
                                  training=self.training)
        elif self.dl_model == 'LSTM':
            self._model = LSTM(hidden_dim=self.hidden_dim,
                               lstm_dim=self.lstm_dim,
                               output_dim=self.output_dim,
                               dropout_p=self.dropout_p,
                               training=self.training)
        else:
            raise(ValueError("dl_model must be either `CNN` or `LSTM`!"))

    
    def create_sequence(self, X:jnp.ndarray, q:jnp.ndarray):
        """Create Sequences from 2D Inputs and Labels

        Args
        ----
        X: 2D input matrix (`batch_size`x `n_features`)
        q: Historical Streamflow (`batch_size`-1 x 1)

        Returns
        -------
        jnp.ndarray: Sequenced input 
        [(`batch_size` - `window_size`), `window_size`, `n_features` + 1]
        """

        # Create empyty sequences
        Xs = []

        # Add sequences to Xs and ys
        for i in range(1, len(X) - self.window_size):
            Xs.append(jnp.concatenate([
                                    X[i: (i + self.window_size)],
                                    q[i - 1:(i + self.window_size) - 1]
                                ], axis=1))

        Xs = jnp.stack(Xs)
        if self.dl_model == 'CNN':
            Xs = jnp.expand_dims(Xs, axis=3)

        return Xs

    
    def __call__(self, x, q):
        """Simulate streamflow using hybrid model

        Args
        ----
        X: 2D input matrix (`batch_size`x `n_features`)
        q: Historical Streamflow (`batch_size`-1 x 1)
        """
        # GR4J production storage simulation
        prod_out, s_store = self._prod_store(x)

        # Scale DL model inputs
        prod_out = (prod_out - self.x_mu)/self.x_sigma

        # Create sequences
        seq = self.create_sequence(prod_out, q)

        # DL model forward pass
        q_out = self._model(seq)

        return q_out, s_store[-self.window_size-2]
         
