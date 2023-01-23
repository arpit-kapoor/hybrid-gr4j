import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

from tqdm import tqdm

sys.path.append("..")

from model.hybrid.hgr4j_ann import HyGR4JNN
from model.utils.training import EarlyStopper
from model.utils.evaluation import evaluate
from model.optim.pso_np import PSO
from data.utils import read_dataset_from_file, get_station_list




parser = argparse.ArgumentParser(description="Train hybrid GR4J ANN model")

parser.add_argument('--data-dir', type=str, default='/data/camels/aus/')
parser.add_argument('--sub-dir', type=str, default='no-scale')
parser.add_argument('--station-id', type=str, default=None)
parser.add_argument('--run-dir', type=str, default='/project/results/hygr4j_ann')
parser.add_argument('--n-epoch', type=int, default=200)
parser.add_argument('--lr', type=int, default=0.01)
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--pop-size', type=int, default=20)



def val_step(model, X, y, loss_fn):
    model.eval()
    y_hat = model(X)
    loss = loss_fn(y, y_hat)
    return loss.detach()


def train_step(model, pso, X, y, loss_fn, opt, n_samples):
    swarm, best_pos, best_fit = pso.swarm, pso.best_swarm_pos, pso.best_swarm_err

    for i in range(n_samples):
        swarm, best_pos, best_fit = pso.evolve(swarm, best_pos, best_fit)
        
    model.set_x1(best_pos[0])

    model.train()
    opt.zero_grad()
    
    y_hat = model(X)
    loss = loss_fn(y, y_hat)
    loss.backward()
    
    opt.step()

    return loss.detach(), best_pos

def evaluate_preds(model, X, y):
    # Evaluate on train data
    model.eval()

    # Prediction using model
    y_hat = model(X)

    # Tensors for evaluation
    Q = y.numpy()
    Q_hat = y_hat.detach().numpy()
    P = X[:, 0].detach().numpy()
    ET = X[:, 1].detach().numpy()

    return evaluate(P, ET, Q, Q_hat)

    
def train_and_evaluate(train_ds, val_ds,
                        station_id, n_epoch=100, lr=0.001,
                        run_dir='/project/results/lstm',
                        pop_size=20, 
                        n_samples=1,
                        **kwargs):
    
    # Hyper-params
    min_pos = 100.0
    max_pos = 1200.0
    min_vel = -5.0
    max_vel = 5.0


    # Create run dir
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Create direcrtory to save plots and model
    plot_dir = os.path.join(run_dir, 'plots')
    model_dir = os.path.join(run_dir, 'models')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    

    # Get tensors from the dataset
    t_train, X_train, y_train = train_ds.tensors
    t_val, X_val, y_val = val_ds.tensors

    # Create model instance
    model = HyGR4JNN(0.0)

    # Create optimizer and loss instance
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Early stopper to avoid overfitting
    early_stopper = EarlyStopper(patience=5, min_delta=0.05)

    # Fittness functoin for swarm optimization of x1 param
    def fit_fn(params):
        model.set_x1(params[0])
        y_hat = model(X_train)
        return loss_fn(y_hat, y_train).detach().numpy()

    print(f"Initialize PSO.. time: {dt.datetime.now().time()}")
    
    dims = 1
    pso = PSO(
        pop_size=pop_size,
        fitness_function=fit_fn,
        num_params=dims,
        max_limits=max_pos*np.ones(dims),
        min_limits=min_pos*np.ones(dims),
        max_limits_vel=max_vel*np.ones(dims),
        min_limits_vel=min_vel*np.ones(dims)
    )
    print(f"Train Model.. time: {dt.datetime.now().time()}")

    # Handle nan values in the data
    X_train = torch.nan_to_num(X_train)
    X_val = torch.nan_to_num(X_val)

    # Best validation loss
    best_val = torch.inf

    # Early stopping
    early_stopper = EarlyStopper(patience=10, min_delta=0.01)

    pbar = tqdm(range(1, n_epoch+1))

    for epoch in pbar:

        # Train step
        train_loss, best_pos = train_step(model, pso, X_train, y_train, 
                                          loss_fn, opt, n_samples)

        # Validation step
        val_loss = val_step(model, X_val, y_val, loss_fn)
        
        print(f"Epoch: {epoch}: train loss: {train_loss.numpy():.4f} val loss: {val_loss.numpy():.4f} best_pos: {best_pos[0]:.2f} timestamp: {dt.datetime.now().time()}")

        # Save the model with best validation loss
        if val_loss < best_val:
            torch.save(model, os.path.join(model_dir, f"{station_id}_best_model.pt"))

        # Stop early if validation loss hasn't improved in a while
        if early_stopper.early_stop(val_loss):
            break

    # Load the best model
    model = torch.load(os.path.join(model_dir, f"{station_id}_best_model.pt"))
    model.eval()

    # Evaluate on train data
    nse_train, nnse_train, fig_train = evaluate_preds(model, X_train, y_train)
    fig_train.savefig(os.path.join(plot_dir, f"{station_id}_train.png"))
    
    # Evaluate on val data
    nse_val, nnse_val, fig_val = evaluate_preds(model, X_val, y_val)
    fig_val.savefig(os.path.join(plot_dir, f"{station_id}_val.png"))

    # Write results to file
    dikt = {
        'station_id': station_id,
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





if __name__ == '__main__':

    # Parse command line arguments
    args = parser.parse_args()

    if args.station_id is None:

        station_ids = get_station_list(args.data_dir, args.sub_dir)[:10]
        
        for ind, station_id in enumerate(station_ids):
            print(f"\n{ind+1}/{len(station_ids)}: Reading data for station_id: {station_id}\n")
            args.station_id = station_id
            train_ds, val_ds = read_dataset_from_file(
                                                        args.data_dir, 
                                                        args.sub_dir, 
                                                        station_id=station_id
                                                    )
            print("Training the neural network model..")
            nse_train, nse_val = train_and_evaluate(
                                    train_ds, val_ds, 
                                    **vars(args)
                                )

    else:
        print(f"Reading data for station_id: {args.station_id}")
        train_ds, val_ds = read_dataset_from_file(
                                                    args.data_dir, 
                                                    args.sub_dir, 
                                                    station_id=args.station_id
                                                )
        print("Training the neural network model..")
        nse_train, nse_val = train_and_evaluate(
                                train_ds, val_ds,
                                **vars(args)
                            )