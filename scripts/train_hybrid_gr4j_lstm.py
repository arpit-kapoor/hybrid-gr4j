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

sys.path.append("../")

from model.ml.lstm import LSTM
from model.hydro.gr4j_prod import ProductionStorage
from model.utils.training import EarlyStopper
from model.utils.evaluation import evaluate
from model.optim.pso_np import PSO
from data.utils import read_dataset_from_file, get_station_list


# Create parser
parser = argparse.ArgumentParser(description="Train Hybrid GR4J-LSTM model on CAMELS dataset")

parser.add_argument('--data-dir', type=str, default='/data/camels/aus/')
parser.add_argument('--sub-dir', type=str, required=True)
parser.add_argument('--station-id', type=str, default=None)
parser.add_argument('--run-dir', type=str, default='/project/results/lstm')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--n-epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--input-dim', type=int, default=9)
parser.add_argument('--hidden-dim', type=int, default=32)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--gr4j-run-dir', type=str, default='/project/results/gr4j')
parser.add_argument('--window-size', type=int, default=4)


# ----
def val_step(model, dl, loss_fn):
    loss_val = 0.
    model.eval()
    for j, (X, y) in enumerate(dl, start=1):
        y_hat = model(X)
        loss_val += loss_fn(y, y_hat)
    return (loss_val/j).detach()


def train_step(model, dl, loss_fn, opt):
    total_loss = 0.
    model.train()
    for i, (X, y) in enumerate(dl, start=1):
        opt.zero_grad()
        y_hat = model(X)
        loss_val = loss_fn(y, y_hat)
        total_loss += loss_val
        loss_val.backward()
        opt.step()
    return (total_loss/i).detach()

def evaluate_preds(model, prod_store, ds, batch_size, y_mu, y_sigma):
    # Evaluate on train data
    model.eval()
    dl = torchdata.DataLoader(ds, 
                              batch_size=batch_size,
                              shuffle=False)

    # Empty list to store batch-wise tensors
    P = []
    ET = []
    Q = []
    Q_hat = []

    for i, (X, y) in enumerate(dl, start=1):
        
        y_hat = model(X)

        Q.append((y*y_sigma+y_mu).detach().numpy())
        Q_hat.append((y_hat*y_sigma+y_mu).detach().numpy())
        
        X_inv = X[:, -1]*prod_store.sigma+prod_store.mu
        
        P.append((X_inv[:, 0]).detach().numpy())
        ET.append((X_inv[:, 1]).detach().numpy())
    
    P = np.concatenate(P, axis=0)
    ET = np.concatenate(ET, axis=0)
    Q = np.concatenate(Q, axis=0).flatten()
    Q_hat = np.clip(np.concatenate(Q_hat, axis=0).flatten(), 0, None)


    return evaluate(P, ET, Q, Q_hat)


def create_sequence(X, y, window_size):

        assert window_size is not None, "Window size cannot be NoneType."

        # Create empyty sequences
        Xs, ys = [], []

        # Add sequences to Xs and ys
        for i in range(len(X)-window_size):
            Xs.append(X[i: (i + window_size)])
            ys.append(y[i + window_size-1])

        Xs, ys = torch.stack(Xs), torch.stack(ys)

        return Xs, ys


def train_and_evaluate(train_ds, val_ds,
                        station_id, n_epoch=100, 
                        batch_size=256, lr=0.001,
                        run_dir='/project/results/lstm',
                        gr4j_run_dir='/project/results/gr4j',
                        **kwargs):
    
    # Create Directories
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    plot_dir = os.path.join(run_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Get tensors from dataset
    t_train, X_train, y_train = train_ds.tensors
    t_val, X_val, y_val = val_ds.tensors

    # Handle nan values
    X_train = torch.nan_to_num(X_train)
    X_val = torch.nan_to_num(X_val)

    # Mean and std 
    y_mu = y_train.mean(dim=0)
    y_sigma = y_train.std(dim=0)

    # Scale labels
    y_train = (y_train - y_mu)/y_sigma
    y_val = (y_val - y_mu)/y_sigma

    # Read GR4J results
    gr4j_results_df = pd.read_csv(os.path.join(gr4j_run_dir, 'result.csv')).reset_index()
    gr4j_results_df.station_id = gr4j_results_df.station_id.astype(str)
    x1 = gr4j_results_df.loc[gr4j_results_df.station_id==station_id, 'x1'].values[0]

    # Create production storage instance
    prod_store = ProductionStorage(x1=x1)
    inp_train = prod_store(X_train)
    inp_val = prod_store(X_val)

    # Create Input sequence
    X_train, y_train = create_sequence(inp_train, y_train, window_size=kwargs['window_size'])
    X_val, y_val = create_sequence(inp_val, y_val, window_size=kwargs['window_size'])

    # Create Sequence Datasets and DataLoaders
    train_ds = torchdata.TensorDataset(X_train, y_train)
    train_dl = torchdata.DataLoader(train_ds, 
                                    batch_size=batch_size, 
                                    shuffle=True)

    val_ds = torchdata.TensorDataset(X_val, y_val)
    val_dl = torchdata.DataLoader(val_ds, 
                                  batch_size=batch_size,
                                  shuffle=True)

    # Create lstm model
    model = LSTM(input_dim=kwargs['input_dim'],
                 hidden_dim=kwargs['hidden_dim'],
                 output_dim=1,
                 n_layers=kwargs['n_layers'],
                 dropout=kwargs['dropout'])
    
    # Create optimizer and loss instance
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Early stopping
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)

    pbar = tqdm(range(1, n_epoch+1))

    for epoch in pbar:

        # Train step
        train_loss = train_step(model, train_dl, loss_fn, opt)

        # Validation step
        val_loss = val_step(model, val_dl, loss_fn)
        
        pbar.set_description(f"""Epoch {epoch} loss: {train_loss.numpy():.4f} val_loss: {val_loss.numpy():.4f}""")

        if early_stopper.early_stop(val_loss):
            break

    # Evaluate on train data
    nse_train, nnse_train, fig_train = evaluate_preds(model, prod_store,
                                                      train_ds, batch_size,
                                                      y_mu, y_sigma)
   
    fig_train.savefig(os.path.join(plot_dir, f"{station_id}_train.png"))
    
    # Evaluate on val data
    nse_val, nnse_val, fig_val = evaluate_preds(model, prod_store,
                                                val_ds, batch_size,
                                                y_mu, y_sigma)
   
    fig_val.savefig(os.path.join(plot_dir, f"{station_id}_val.png"))

    # Write results to file
    dikt = {
        'station_id': station_id,
        'x1': x1,
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



# ------------
if __name__ == '__main__':
    # Parse command line arguments
    args = parser.parse_args()

    print(args)

    if args.station_id is None:

        station_ids = get_station_list(args.data_dir, args.sub_dir)
        
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