from __future__ import absolute_import

import os
import sys
import argparse

import joblib
import numpy as np
import torch

from tqdm import tqdm

sys.path.append('..')
from dataset.camels_dataset import CamelsAusDataset

# ------------------------------------------------

# Define command line arguments
parser = argparse.ArgumentParser(description="""Slice CAMELS Australia
                                streamflow data into train and validation for 
                                each station""")

parser.add_argument('--data-dir', type=str, default='/data/camels/aus/')
parser.add_argument('--sub-dir', type=str, required=True)
parser.add_argument('--scale', type=bool, default=True)
parser.add_argument('--create-seq', type=bool, default=False)
parser.add_argument('--window-size', type=int, default=7)

# ------------------------------------------------

def create_dir(path):
    """Function to create the directory path if it does not exists
    """
    if not os.path.exists(path):
        os.makedirs(path)


def create_datasets(data_dir, sub_dir, scale,
                    create_seq, window_size):
    """Function to write input and out arrays to file
    """
    # Create dataset object
    print("Reading and processing the data...")
    camels_ds = CamelsAusDataset(
        data_dir=data_dir,
        scale=scale, 
        create_seq=create_seq,
        window_size=window_size
    )

    # Create directories
    print("Now creating directories...")
    sub_dir_path = os.path.join(data_dir, '../processed', sub_dir)
    write_dir_train = os.path.join(sub_dir_path, 'datasets/train')
    write_dir_val = os.path.join(sub_dir_path, 'datasets/val')

    create_dir(write_dir_train)
    create_dir(write_dir_val)


    if scale:
        scaler_path = os.path.join(sub_dir_path, 'scalers')
        create_dir(scaler_path)

        # Save scalers
        print("Saving scalers to file...")
        joblib.dump(camels_ds.x_scaler, os.path.join(scaler_path, 'x_scaler.save'))
        joblib.dump(camels_ds.y_scaler, os.path.join(scaler_path, 'y_scaler.save'))
    
    for station_id in tqdm(camels_ds.stations):

        # print(f"Saving data for station Id: {station_id}")
        
        train_path = os.path.join(write_dir_train, station_id)
        val_path = os.path.join(write_dir_val, station_id)

        create_dir(train_path)
        create_dir(val_path)

        train_ds = camels_ds.ds_store[station_id]['train']
        val_ds = camels_ds.ds_store[station_id]['val']

        t_train, X_train, y_train = train_ds.tensors
        t_val, X_val, y_val = val_ds.tensors

        torch.save(t_train, os.path.join(train_path, 't_train.pt'))
        torch.save(X_train, os.path.join(train_path, 'X_train.pt'))
        torch.save(y_train, os.path.join(train_path, 'y_train.pt'))
        
        torch.save(t_val, os.path.join(val_path, 't_val.pt'))
        torch.save(X_val, os.path.join(val_path, 'X_val.pt'))
        torch.save(y_val, os.path.join(val_path, 'y_val.pt'))

    print("Done!")



# ------------------------------------------------
if __name__ == '__main__':

    # Parse command line arguments
    args = parser.parse_args()
    
    create_datasets(data_dir=args.data_dir,
                    sub_dir=args.sub_dir,
                    scale=args.scale,
                    create_seq=args.create_seq,
                    window_size=args.window_size)
