import time
import random
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import datetime as dt

import os
import sys

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from optuna.trial import TrialState

sys.path.append("/project")

from model.hydro.gr4j_prod import ProductionStorage
from model.utils.training import EarlyStopper
from model.utils.evaluation import evaluate
from data.utils import read_dataset_from_file, get_station_list
from model.ml.lstm import LSTM
from model.utils.evaluation import evaluate



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


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_dim = trial.suggest_int("hidden_dim", 16, 64, step=16)
    p = trial.suggest_float("dropout", 0.2, 0.5)
    
    model = LSTM(input_dim=INPUT_DIM,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers,
            dropout=p)

    return model


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

def evaluate_preds(model, ds, batch_size, x_scaler=None, y_scaler=None, plot=True):
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


    return evaluate(P, ET, Q, Q_hat, plot=plot)


def objective(trial, return_model=False):
    # Generate the model.
    model = define_model(trial)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    opt = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Loss function
    loss_fn = nn.MSELoss()

    # Early stopping
    early_stopper = EarlyStopper(patience=10, min_delta=0.01)

    pbar = range(1, N_EPOCHS+1)

    for epoch in pbar:

        # Train step
        train_loss = train_step(model, train_dl, loss_fn, opt)

        # Validation step
        val_nse, _ = evaluate_preds(model, val_dl.dataset, 
                                    batch_size=BATCH_SIZE, plot=False)

        trial.report(val_nse, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    if return_model:
        return model
    return val_nse



def train_model(trial):
    
    # Generate the model.
    model = define_model(trial)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    opt = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Loss function
    loss_fn = nn.MSELoss()

    # Early stopping
    early_stopper = EarlyStopper(patience=10, min_delta=0.01)

    pbar = tqdm(range(1, N_EPOCHS+1))

    for epoch in pbar:

        # Train step
        train_loss = train_step(model, train_dl, loss_fn, opt)

        # Train step
        val_loss = val_step(model, val_dl, loss_fn)

        pbar.set_description(f"""Epoch {epoch} loss: {train_loss.numpy():.4f} val_loss: {val_loss.numpy():.4f}""")

        if early_stopper.early_stop(val_loss):
            break
    
    return model



if __name__ == '__main__':

    data_dir = '/data/camels/aus/'
    sub_dir = 'no-scale'
    run_dir = '/project/results/hygr4j_lstm_hyperopt'
    gr4j_run_dir = '/project/results/gr4j_params'

    N_EPOCHS = 50
    BATCH_SIZE = 256
    N_STATIONS = 2

    INPUT_DIM = 9


    # Station list
    np.random.seed(int(time.time()))
    station_list = get_station_list(data_dir, sub_dir)


    for i in range(N_STATIONS):

        station_id = random.choice(station_list)
        station_list.remove(station_id)

        # Read GR4J results
        gr4j_results_df = pd.read_csv(os.path.join(gr4j_run_dir, 'result.csv')).reset_index()
        gr4j_results_df.station_id = gr4j_results_df.station_id.astype(str)
        x1 = gr4j_results_df.loc[gr4j_results_df.station_id==station_id, 'x1'].values[0]


        print(f"Reading data for station_id: {station_id}")
        train_ds, test_ds = read_dataset_from_file(data_dir, 
                                                sub_dir, 
                                                station_id=station_id)

        # Get tensors from dataset
        t_train, X_train, y_train = train_ds.tensors
        t_test, X_test, y_test = test_ds.tensors

        # Scale input tensor
        X_train = torch.nan_to_num(X_train)
        X_test = torch.nan_to_num(X_test)

        # Scale labels
        y_mu = y_train.mean(dim=0)
        y_sigma = y_train.std(dim=0)
        y_train = (y_train - y_mu)/y_sigma
        y_test = (y_test - y_mu)/y_sigma

        # Production storage
        prod_store = ProductionStorage(x1=x1)

        # Prod storage output
        prod_out_train = prod_store(X_train, include_x=True)[0]
        X_train, y_train = create_sequence(prod_out_train, y_train, window_size=4)

        prod_out_test = prod_store(X_test, include_x=True)[0]
        X_test, y_test = create_sequence(prod_out_test, y_test, window_size=4)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                        test_size=0.25, shuffle=False)

        # Create train dataset and loader
        train_ds = torchdata.TensorDataset(X_train, y_train)
        train_dl = torchdata.DataLoader(train_ds,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)

        # Create validation dataset and loader
        val_ds = torchdata.TensorDataset(X_val, y_val)
        val_dl = torchdata.DataLoader(val_ds,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True)

        test_ds = torchdata.TensorDataset(X_test, y_test)
        test_dl = torchdata.DataLoader(test_ds,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)


        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        plot_dir = os.path.join(run_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)


        # Optimize hyperparams
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))




        # Train and evaluate model
        N_EPOCHS = 200
        model = train_model(trial)

        nse_train, nnse_train, fig_train = evaluate_preds(model, train_ds, 
                                                  batch_size=BATCH_SIZE)
        fig_train.savefig(os.path.join(plot_dir, f"{station_id}_train.png"))

        nse_test, nnse_test, fig_test = evaluate_preds(model, test_ds, 
                                                batch_size=BATCH_SIZE)
        fig_test.savefig(os.path.join(plot_dir, f"{station_id}_val.png"))



        # Write results to file
        dikt = {
            'station_id': station_id,
            'x1': x1,
            'nse_train': nse_train,
            'nnse_train': nnse_train,
            'nse_val': nse_test,
            'nnse_val': nnse_test,
            'run_ts': dt.datetime.now()
        }
        df = pd.DataFrame(dikt, index=[0])

        csv_path = os.path.join(run_dir, 'result.csv')
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)





























