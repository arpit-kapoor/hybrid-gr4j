import os
import sys
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data as data
from rrmpg.models import GR4J

sys.path.append("..")
from utils.data import read_dataset_from_file, get_station_list
from utils.evaluation import evaluate


# Create parser
parser = argparse.ArgumentParser(description="Calibrate GR4J model")

parser.add_argument('--data-dir', type=str, default='/data/camels/aus/')
parser.add_argument('--sub-dir', type=str, required=True)
parser.add_argument('--station-id', type=str, default=None)


def calibrate_gr4j(train_ds, val_ds):
    # Create model instance
    model = GR4J()

    t_train, X_train, y_train = train_ds.tensors
    t_val, X_val, y_val = val_ds.tensors

    # Training data tensors
    P_train = X_train[:, 0].detach().numpy()
    ET_train = X_train[:, 1].detach().numpy()
    Q_train = y_train.flatten().detach().numpy()

    # Fit the model
    result = model.fit(Q_train, P_train, ET_train)

    # Update GR4J parameters
    params = {}
    param_names = model.get_parameter_names()
    for i, param in enumerate(param_names):
        params[param] = result.x[i]

    # This line set the model parameters to the ones specified in the dict
    model.set_params(params)

    # Simulate train data
    print("Evaluating on training data...")
    Q_hat = model.simulate(P_train, ET_train).flatten()
    nse_train = evaluate(P_train, ET_train, Q_train, Q_hat)

    # Validation data tensors
    P_val = X_val[:, 0].detach().numpy()
    ET_val = X_val[:, 1].detach().numpy()
    Q_val = y_val.flatten().detach().numpy()

    # Simulate val data
    print("Evaluating on validation data...")
    Q_hat = model.simulate(P_val, ET_val).flatten()
    nse_val = evaluate(P_val, ET_val, Q_val, Q_hat)

    return nse_train, nse_val





if __name__ == '__main__':

    # Parse command line arguments
    args = parser.parse_args()

    if args.station_id is None:

        results_df = pd.DataFrame(columns=['station_id', 'nse_train', 
                                           'nse_val'])

        station_ids = get_station_list(args.data_dir, args.sub_dir)
        for ind, station_id in enumerate(station_ids):
            print(f"{ind+1}/{len(station_ids)}: Reading data for station_id: {station_id}\n")
            train_ds, val_ds = read_dataset_from_file(args.data_dir, 
                                                      args.sub_dir, 
                                                      station_id=station_id)
            print(f"Calibrating GR4J model..")
            nse_train, nse_val = calibrate_gr4j(train_ds, val_ds)

            dikt = {
                'station_id': station_id,
                'nse_train': nse_train,
                'nse_val': nse_val
            }

            results_df = pd.concat([results_df, pd.DataFrame(dikt, index=[0])]).reset_index(drop=True)
            print(" ")
        
        results_df.to_csv('/project/results/gr4j.csv')

    else:
        print(f"Reading data for station_id: {args.station_id}")
        train_ds, val_ds = read_dataset_from_file(args.data_dir, 
                                                  args.sub_dir, 
                                                  station_id=args.station_id)
        print(f"Calibrating GR4J model..")
        nse_train, nse_val = calibrate_gr4j(train_ds, val_ds)



