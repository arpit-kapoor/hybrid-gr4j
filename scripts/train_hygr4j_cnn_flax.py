import argparse
import datetime as dt
import json
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optax
import torch
import torch.utils.data as torchdata

from flax import linen as nn
from tqdm import tqdm
from clu import metrics
from flax.training import train_state
from flax import struct

sys.path.append('../')
from model.hybrid import HyGR4J
from model.hydro.gr4j_prod_flax import ProductionStorage
from model.utils.training import EarlyStopper
from model.utils.evaluation import nse, normalize
from data.utils import read_dataset_from_file, get_station_list
from data.camels_sampler import CamelsBatchSampler



# Create parser
parser = argparse.ArgumentParser(description="Train Hybrid GR4J model on CAMELS dataset")

parser.add_argument('--data-dir', type=str, default='/data/camels/aus/')
parser.add_argument('--sub-dir', type=str, required=True)
parser.add_argument('--station-id', type=str, default=None)
parser.add_argument('--run-dir', type=str, default='/project/results/hygr4j')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--n-epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=0.04)
parser.add_argument('--n-filters', nargs='+', type=int)
parser.add_argument('--n-features', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--window-size', type=int, default=7)
# parser.add_argument('--q-in', action='store_true')
parser.add_argument('--x1-init', type=float, default=0.6)
parser.add_argument('--s-init', type=float, default=0.0)
parser.add_argument('--scale', type=float, default=1000.0)



@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
    metrics: Metrics
    key: jax.random.KeyArray

def create_train_state(module, params_key, dropout_key,
                       lr, batch_size, n_features, weight_decay):
    """Creates an initial `TrainState`."""
    
    params = module.init({'params': params_key, 'dropout': dropout_key}, 
                     jax.random.normal(main_key, (batch_size, n_features)), 
                     jax.random.normal(main_key, (batch_size, 1)))['params']
    
    tx = optax.adamw(lr, b1=0.89, b2=0.97, weight_decay=weight_decay)

    return TrainState.create(apply_fn=module.apply, 
                             params=params, tx=tx,
                             key=dropout_key,
                             metrics=Metrics.empty())

@jax.jit
def train_step(state, dropout_key, batch, targets):
    """Train for a single step."""
    
    def loss_fn(params):
        preds, s_store = state.apply_fn({'params': params}, batch, targets,
                                rngs={'dropout': dropout_key})
        loss = optax.l2_loss(preds, targets[args.window_size+1:]).mean()
        return loss
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state

@jax.jit
def compute_metrics(*, state, dropout_key, batch, targets):
    
    preds, s_store = state.apply_fn({'params': state.params}, batch, targets,
                            rngs={'dropout': dropout_key})
    
    loss = optax.l2_loss(preds, targets[args.window_size+1:]).mean()
    
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    
    return state, s_store


def evaluate(module, y_mu, y_sigma, dl, s_init, window_size,
             state=None, params=None):

    if params is None:
        if state is None:
            raise("No params provided!")
        params = state.params

    # Empty list to store batch-wise tensors
    P = []
    ET = []
    Q = []
    Q_hat = []

    module.training = False
    module.s_init = s_init

    for i, (X, y) in enumerate(dl, start=1):

        X = X.detach().numpy()
        y = y.detach().numpy()
        
        y_hat, s_store = module.apply({'params': params}, X, y)
        module.s_init = s_store/(params['_prod_store']['x1']*module.scale)

        P.append(X[window_size+1:, 0])
        ET.append(X[window_size+1:, 1])

        Q.append((y[window_size+1:]*y_sigma+y_mu))
        Q_hat.append((y_hat*y_sigma+y_mu))

    Q = np.concatenate(Q, axis=0)
    P = np.concatenate(P, axis=0)
    ET = np.concatenate(ET, axis=0)
    Q_hat = jnp.clip(np.concatenate(Q_hat, axis=0), 0)

    nse_score = nse(Q, Q_hat)
    nnse_score = normalize(nse_score)

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(P, 'g--', label='precip', alpha=0.40)
    ax.plot(ET, 'y--', label='etp', alpha=0.30)
    ax.plot(Q, color='black', label='obs', alpha=1.0)
    ax.plot(Q_hat, color='red', label='pred', alpha=0.75)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Flow (mm/day)')

    ax.annotate(f'NSE: {nse_score:.4f}',
            xy=(0.7, 0.88), xycoords='figure fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=12)
    ax.set_title('Streamflow prediction')

    plt.legend()

    return nse_score, nnse_score, fig


def plot_metric_trace(metrics_history, plot_dir):

    train_loss = jnp.array(metrics_history['train_loss'])
    test_loss = jnp.array(metrics_history['test_loss'])

    fig, ax = plt.subplots()
    ax.plot(train_loss, color='black', label='train')
    ax.plot(test_loss, color='red', label='test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(os.path.join(plot_dir, 'loss.png'))

    x1 = jnp.array(metrics_history['x1'])

    fig, ax = plt.subplots()
    ax.plot(x1, color='black', label='x1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    plt.savefig(os.path.join(plot_dir, 'x1.png'))



def get_scaling_params(X, y, s_init, x1_init, scale, params_key):

    # Initialize production storage
    prod_store = ProductionStorage(s_init=s_init, 
                                   x1_init=x1_init, 
                                   scale=scale)

    params = prod_store.init(rngs={'params': params_key}, 
                             x=jnp.ones((1, X.shape[1])))
    
    x_out, s_store = prod_store.apply(params, X)

    x_mu = x_out.mean(axis=0)
    x_sigma = x_out.std(axis=0)

    y_mu = y.mean(axis=0)
    y_sigma = y.std(axis=0)

    return x_mu, x_sigma, y_mu, y_sigma



def train_and_evaluate(train_ds, val_ds, 
                       params_key, dropout_key, 
                       station_id, n_epoch=100, 
                       batch_size=256, lr=0.001,
                       plot_dir='',
                       run_dir='/project/results/hygr4j',
                       **kwargs):
    
    # Get Tensors
    t_train, X_train, y_train = train_ds.tensors
    t_val, X_val, y_val = val_ds.tensors

    # Handle nan values
    X_train = torch.nan_to_num(X_train)
    X_val = torch.nan_to_num(X_val)

    # Concatenate train and val to compute scaling params
    X = torch.concat([X_train, X_val])
    y = torch.concat([y_train, y_val])

    # Get scaling params
    x_mu, x_sigma, y_mu, y_sigma = get_scaling_params(X.detach().numpy(), 
                                                      y.detach().numpy(),
                                                      s_init=kwargs['s_init'],
                                                      x1_init=kwargs['x1_init'],
                                                      scale=kwargs['scale'],
                                                      params_key=params_key)

    # Scale y values
    y_train = (y_train - y_mu)/y_sigma
    y_val = (y_val - y_mu)/y_sigma

    # Create model instance
    hygr4j = HyGR4J(x1_init=kwargs['x1_init'], s_init=kwargs['s_init'], 
                n_filters=kwargs['n_filters'], dropout_p=kwargs['dropout'],
                x_mu=x_mu, x_sigma=x_sigma, window_size=kwargs['window_size'],
                scale=kwargs['scale'])
    
    # Create dataset and Dataloader
    train_ds = torchdata.TensorDataset(X_train, y_train)
    train_batch_sampler = CamelsBatchSampler(train_ds, batch_size=batch_size, 
                                            window_size=kwargs['window_size'],
                                            drop_last=False)
    train_dl = torchdata.DataLoader(train_ds, batch_sampler=train_batch_sampler, 
                                    num_workers=2, prefetch_factor=2)

    val_ds = torchdata.TensorDataset(X_val, y_val)
    val_batch_sampler = CamelsBatchSampler(val_ds, batch_size=batch_size, 
                                        window_size=kwargs['window_size'],
                                        drop_last=False)
    val_dl = torchdata.DataLoader(val_ds, batch_sampler=val_batch_sampler,
                                num_workers=2, prefetch_factor=2)

    # Create train state for optax
    state = create_train_state(hygr4j, params_key=params_key,
                               dropout_key=dropout_key, lr=lr,
                               batch_size=batch_size, 
                               n_features=kwargs['n_features'], 
                               weight_decay=kwargs['weight_decay'])
    
    metrics_history = {'train_loss': [],
                        'test_loss': [],
                        'x1': []}
    
    pbar = tqdm(range(1, n_epoch+1))
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)

    for epoch in pbar:

        hygr4j.s_init = kwargs['s_init']
        hygr4j.training = True

        # Train step
        for step, (batch, targets) in enumerate(train_dl):

            batch = batch.detach().numpy()
            targets = targets.detach().numpy()

            state = train_step(state, dropout_key, batch, targets)
            state, s_store_train = compute_metrics(state=state, 
                                                dropout_key=dropout_key,
                                                batch=batch, 
                                                targets=targets)
            hygr4j.s_init = s_store_train/(state.params['_prod_store']['x1']*hygr4j.scale)
        
        for metric, value in state.metrics.compute().items(): # compute metrics
            metrics_history[f'train_{metric}'].append(value) # record metrics
        state = state.replace(metrics=state.metrics.empty())

        # Validation step
        test_state = state
        hygr4j.s_init = kwargs['s_init']
        hygr4j.training = False
        for step, (batch, targets) in enumerate(val_dl):
            batch = batch.detach().numpy()
            targets = targets.detach().numpy()
            test_state, s_store_test = compute_metrics(state=test_state, 
                                                       dropout_key=dropout_key,
                                                       batch=batch, 
                                                       targets=targets)
            hygr4j.s_init = s_store_test/(test_state.params['_prod_store']['x1']*hygr4j.scale)

        for metric, value in test_state.metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)
        
        metrics_history['x1'].append(state.params['_prod_store']['x1'])
        
        pbar.set_description(f"""Epoch {epoch}/{n_epoch} loss: {metrics_history['train_loss'][-1]:.4f} val_loss: {metrics_history['test_loss'][-1]:.4f}""")

        if early_stopper.early_stop(metrics_history['test_loss'][-1]):
            break

    # Generate traceplots for loss
    plot_metric_trace(metrics_history, plot_dir=plot_dir)

    nse_train, nnse_train, fig_train = evaluate(hygr4j, y_mu, y_sigma,
                                                train_dl, kwargs['s_init'],
                                                kwargs['window_size'], 
                                                state)
    
    print(f"Train NSE: {nse_train:.3f}")
    print(f"Train Normalized NSE: {nnse_train:.3f}")

    fig_train.savefig(os.path.join(plot_dir, f"{station_id}_train.png"))
    
    nse_val, nnse_val, fig_val = evaluate(hygr4j, y_mu, y_sigma,
                                            val_dl, kwargs['s_init'],
                                            kwargs['window_size'], 
                                            state)
    
    print(f"Validation NSE: {nse_val:.3f}")
    print(f"Validation Normalized NSE: {nnse_val:.3f}")

    fig_val.savefig(os.path.join(plot_dir, f"{station_id}_val.png"))

    # Write results to file
    dikt = {
        'station_id': station_id,
        'x1': metrics_history['x1'][-1],
        'nse_train': nse_train,
        'nnse_train': nnse_train,
        'nse_val': nse_val,
        'nnse_val': nnse_val,
        'run_ts': dt.datetime.now()
    }
    df = pd.DataFrame(dikt, index=[0])

    csv_path = os.path.join(run_dir, 'result.csv')
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    return nse_train, nse_val


if __name__=='__main__':
    # Parse command line arguments
    args = parser.parse_args()

    # Create Directories
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
    
    with open(os.path.join(args.run_dir, 'run_params.json'), 'w') as f_args:
        json.dump(vars(args), f_args, indent=2)

    print(args)

    # Key Generators
    root_key = jax.random.PRNGKey(seed=1)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

    if args.station_id is None:

        station_ids = get_station_list(args.data_dir, args.sub_dir)[:10]
        
        for ind, station_id in enumerate(station_ids):

            plot_dir = os.path.join(args.run_dir, 'plots', station_id)
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            print(f"\n{ind+1}/{len(station_ids)}: Reading data for station_id: {station_id}\n")
            args.station_id = station_id
            train_ds, val_ds = read_dataset_from_file(args.data_dir, 
                                                      args.sub_dir, 
                                                      station_id=station_id)
            args.station_id = station_id
            
            print("Training the neural network model..")
            nse_train, nse_val = train_and_evaluate(train_ds, val_ds,
                                                    params_key=params_key,
                                                    dropout_key=dropout_key,
                                                    plot_dir=plot_dir,
                                                    **vars(args))

    else:
        plot_dir = os.path.join(args.run_dir, 'plots', args.station_id)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        print(f"Reading data for station_id: {args.station_id}")
        train_ds, val_ds = read_dataset_from_file(args.data_dir, 
                                                  args.sub_dir, 
                                                  station_id=args.station_id)
        
        print("Training the neural network model..")
        nse_train, nse_val = train_and_evaluate(train_ds, val_ds,
                                                params_key=params_key,
                                                dropout_key=dropout_key,
                                                plot_dir=plot_dir,
                                                **vars(args))
