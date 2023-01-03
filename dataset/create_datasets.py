from __future__ import absolute_import

import os
import joblib
import torch

import numpy as np

from camels_dataset import CamelsAusDataset


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def create_datasets(data_dir:str='/data/camels/aus/'):

    # Create dataset object
    print("Reading and processing the data...")
    camels_ds = CamelsAusDataset(
        data_dir=data_dir,
        scale=True, 
        create_seq=True
    )

    # Create directories
    print("Now creating directories...")
    write_dir_train = os.path.join(data_dir, '../datasets/train')
    write_dir_val = os.path.join(data_dir, '../datasets/val')

    create_dir(write_dir_train)
    create_dir(write_dir_val)

    scaler_path = os.path.join(data_dir, '../scalers')
    create_dir(scaler_path)

    # Save scalers
    print("Saving scalers to file...")
    joblib.dump(camels_ds.x_scaler, os.path.join(scaler_path, 'x_scaler.save'))
    joblib.dump(camels_ds.y_scaler, os.path.join(scaler_path, 'y_scaler.save'))
    
    for station_id in camels_ds.stations:

        print(f"Saving data for station Id: {station_id}")
        
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




if __name__ == '__main__':

    create_datasets()
