# %%
import argparse
import os
import sys

import numpy as np
import pandas as pd
import datetime as dt

import torch
import torch.nn as nn
import torch.utils.data as torchdata

from tqdm import tqdm

sys.path.append('/project')
from model.ml import LSTM
from data.camels_dataset import CamelsAusDataset
from data.utils import  read_dataset_from_file, get_station_list
from model.utils.evaluation import evaluate
from model.utils.training import EarlyStopper


# Create parser
parser = argparse.ArgumentParser(description="Train LSTM model on CAMELS dataset")

parser.add_argument('--data-dir', type=str, default='/data/camels/aus/')
parser.add_argument('--sub-dir', type=str, required=True)
parser.add_argument('--station-id', type=str, default=None)
parser.add_argument('--run-dir', type=str, default='/project/results/lstm')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--n-epoch', type=int, default=200)
parser.add_argument('--lr', type=int, default=0.001)



# %%
def val_step(model, dl, loss_fn):
    loss_val = 0.
    model.eval()
    for j, (t, X, y) in enumerate(dl, start=1):
        y_hat = model(X)
        loss_val += loss_fn(y, y_hat)
    return (loss_val/j).detach()


def train_step(model, dl, loss_fn, opt):
    total_loss = 0.
    model.train()
    for i, (t, X, y) in enumerate(dl, start=1):
        opt.zero_grad()
        y_hat = model(X)
        loss_val = loss_fn(y, y_hat)
        total_loss += loss_val
        loss_val.backward()
        opt.step()
    return (total_loss/i).detach()

def evaluate_preds(model, ds, batch_size, x_scaler=None, y_scaler=None):
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

    for i, (t, X, y) in enumerate(dl, start=1):
        
        y_hat = model(X)

        Q.append(y_scaler.inverse_transform(y))
        Q_hat.append(y_scaler.inverse_transform(y_hat.detach()))
        
        X_inv = x_scaler.inverse_transform(X[:, -1])
        
        P.append(X_inv[:, 0])
        ET.append(X_inv[:, 1])
    
    P_train = np.concatenate(P, axis=0)
    ET_train = np.concatenate(ET, axis=0)
    Q_train = np.concatenate(Q, axis=0).flatten()
    Q_hat = np.concatenate(Q_hat, axis=0).flatten()

    return evaluate(P_train, ET_train, Q_train, Q_hat)

    
def train_and_evaluate_lstm(train_ds, val_ds,
                            station_id, n_epoch=100, 
                            batch_size=256, lr=0.001,
                            run_dir='/project/results/lstm'):
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    plot_dir = os.path.join(run_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Create data-loaders
    train_loader = torchdata.DataLoader(train_ds, 
                                        batch_size=batch_size,
                                        shuffle=True)

    val_loader = torchdata.DataLoader(val_ds, 
                                      batch_size=batch_size,
                                      shuffle=False)

    # Create model instance
    model = LSTM(input_dim=5,
                 hidden_dim=64,
                 output_dim=1,
                 n_layers=2)

    # Create optimizer and loss instance
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Early stopping
    early_stopper = EarlyStopper(patience=10, min_delta=0.1)

    pbar = tqdm(range(1, n_epoch+1))

    for epoch in pbar:

        # Train step
        train_loss = train_step(model, train_loader, loss_fn, opt)

        # Validation step
        val_loss = val_step(model, val_loader, loss_fn)
        
        pbar.set_description(f"""Epoch {epoch} loss: {train_loss.numpy():.4f} val_loss: {val_loss.numpy():.4f}""")

        if early_stopper.early_stop(val_loss):
            break

    # Evaluate on train data
    nse_train, nnse_train, fig_train = evaluate_preds(model, train_ds,
                                                      batch_size=batch_size, 
                                                      x_scaler=x_scaler, 
                                                      y_scaler=y_scaler)
   
    fig_train.savefig(os.path.join(plot_dir, f"{station_id}_train.png"))
    
    # Evaluate on val data
    nse_val, nnse_val, fig_val = evaluate_preds(model, val_ds,
                                                      batch_size=batch_size, 
                                                      x_scaler=x_scaler, 
                                                      y_scaler=y_scaler)
   
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


    # Parse command line arguments
    args = parser.parse_args()

    if args.station_id is None:

        station_ids = get_station_list(args.data_dir, args.sub_dir)
        
        for ind, station_id in enumerate(station_ids):
            print(f"\n{ind+1}/{len(station_ids)}: Reading data for station_id: {station_id}\n")
            train_ds, val_ds, x_scaler, y_scaler  = read_dataset_from_file(
                                                        args.data_dir, 
                                                        args.sub_dir, 
                                                        station_id=station_id
                                                    )
            
            print("Training the neural network model..")
            nse_train, nse_val = train_and_evaluate_lstm(
                                    train_ds, val_ds, station_id, 
                                    n_epoch=args.n_epoch, 
                                    batch_size=args.batch_size, 
                                    lr=args.lr, run_dir=args.run_dir
                                )

    else:
        print(f"Reading data for station_id: {args.station_id}")
        train_ds, val_ds, x_scaler, y_scaler  = read_dataset_from_file(
                                                    args.data_dir, 
                                                    args.sub_dir, 
                                                    station_id=args.station_id
                                                )
        
        print("Training the neural network model..")
        nse_train, nse_val = train_and_evaluate_lstm(
                                train_ds, val_ds, args.station_id, 
                                n_epoch=args.n_epoch, 
                                batch_size=args.batch_size, 
                                lr=args.lr, run_dir=args.run_dir
                            )
