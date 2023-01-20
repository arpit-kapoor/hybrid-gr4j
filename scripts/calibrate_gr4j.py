import os
import sys
import argparse
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data as data
from rrmpg.models import GR4J

sys.path.append("..")
from data.utils import read_dataset_from_file, get_station_list
from model.utils.evaluation import evaluate


# Create parser
parser = argparse.ArgumentParser(description="Calibrate GR4J model")

parser.add_argument('--data-dir', type=str, default='/data/camels/aus/')
parser.add_argument('--sub-dir', type=str, required=True)
parser.add_argument('--station-id', type=str, default=None)
parser.add_argument('--run-dir', type=str, default='/project/results/gr4j')


def calibrate_gr4j(train_ds, val_ds, station_id, run_dir='/project/results'):

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    plot_dir = os.path.join(run_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

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
    nse_train, nnse_train, fig_train = evaluate(P_train, ET_train, Q_train, Q_hat)
    fig_train.savefig(os.path.join(plot_dir, f"{station_id}_train.png"))

    # Validation data tensors
    P_val = X_val[:, 0].detach().numpy()
    ET_val = X_val[:, 1].detach().numpy()
    Q_val = y_val.flatten().detach().numpy()

    # Simulate val data
    print("Evaluating on validation data...")
    Q_hat = model.simulate(P_val, ET_val).flatten()
    nse_val, nnse_val, fig_val = evaluate(P_val, ET_val, Q_val, Q_hat)

    fig_val.savefig(os.path.join(plot_dir, f"{station_id}_val.png"))


    # Write results to file
    dikt = {
        'station_id': station_id,
        'nse_train': nse_train,
        'nnse_train': nnse_train,
        'nse_val': nse_val,
        'nnse_val': nnse_val,
        'run_ts': dt.datetime.now(),
        'x1': params['x1'],
        'x2': params['x2'],
        'x3': params['x3'],
        'x4': params['x4']
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

        station_ids = get_station_list(args.data_dir, args.sub_dir)
        
        for ind, station_id in enumerate(station_ids):
            print(f"\n{ind+1}/{len(station_ids)}: Reading data for station_id: {station_id}\n")
            train_ds, val_ds = read_dataset_from_file(args.data_dir, 
                                                      args.sub_dir, 
                                                      station_id=station_id)
            print("Calibrating GR4J model..")
            nse_train, nse_val = calibrate_gr4j(train_ds, val_ds,
                                                run_dir=args.run_dir, 
                                                station_id=station_id)

    else:
        print(f"Reading data for station_id: {args.station_id}")
        train_ds, val_ds = read_dataset_from_file(args.data_dir, 
                                                  args.sub_dir, 
                                                  station_id=args.station_id)
        print("Calibrating GR4J model..")
        nse_train, nse_val = calibrate_gr4j(train_ds, val_ds,
                                            run_dir=args.run_dir, 
                                            station_id=args.station_id)



