from __future__ import absolute_import

import datetime as dt
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data as data
from camels_aus.repository import CamelsAus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 7


class CamelsAusDataset(object):
    """Class to read Camels dataset from file
    """

    x_col = ['precipitation_AWAP', 'et_morton_actual_SILO',
            'tmax_awap', 'tmin_awap', 'vprp_awap']
    y_col = ['streamflow_mmd']
    coord_col = ['station_id', 'time']

    def __init__(self, data_dir, x_col=None, y_col=None,
                scale:bool=True, create_seq:bool=True, 
                keep_z:bool=True, window_size:int=WINDOW_SIZE):

        # Path to Camels data
        self.data_dir = data_dir

        # Create data repository
        self.repo = CamelsAus()
        self.repo.load_from_text_files(self.data_dir)
        
        # Xarray dataset object
        self.ds = self.repo.daily_data.sel(time=slice(dt.datetime(1980, 1, 1), dt.datetime(2015, 1, 1)))

        # Define x and y columns
        if x_col is not None:
            self.x_col = x_col
        if y_col is not None:
            self.y_col = y_col

        # DS list
        self.ds_store = self.create_datasets(scale, create_seq, window_size=window_size)
    
    def create_sequence(self, t,  X, y, window_size):

        assert window_size is not None, "Window size cannot be NoneType."

        # Create empyty sequences
        ts, Xs, ys = [], [], []

        # Add sequences to Xs and ys
        for i in range(len(X)-window_size):
            Xs.append(X[i: (i + window_size)])
            ys.append(y[i + window_size-1])
            ts.append(t[i + window_size-1])

        ts, Xs, ys = torch.stack(ts), torch.stack(Xs), torch.stack(ys)

        return ts, Xs, ys

    
    def create_datasets(self, scale, create_seq, window_size=None):

        # Store station ids
        self.stations = self.ds.station_id.to_numpy()

        X_list, y_list, coord_list = [], [], []

        for station_id in self.stations:
            station_ds = self.ds.sel(station_id=station_id)
            station_ds = station_ds[self.x_col + self.y_col].where(
                                    lambda x: x[self.y_col[0]].notnull(), 
                                    drop=True
                                )
            for x_col in self.x_col:
                station_ds = station_ds[self.x_col + self.y_col].where(
                                    lambda x: x[x_col].notnull(), 
                                    drop=True
                                )
            station_df = station_ds.to_pandas().reset_index()

            station_df.time = station_df.time.apply(lambda x: time.mktime(x.timetuple()))

            X_list.append(station_df[self.x_col])
            y_list.append(station_df[self.y_col])
            coord_list.append(station_df[self.coord_col])

        
        X = pd.concat(X_list, axis=0).reset_index(drop=True)
        y = pd.concat(y_list, axis=0).reset_index(drop=True)
        coord = pd.concat(coord_list, axis=0).reset_index(drop=True)
        

        # Scaling preference
        self.scale = scale
        if scale:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
        
            # Scale
            X = self.x_scaler.fit_transform(X)
            y = self.y_scaler.fit_transform(y)
        
        else:
            X = X.values
            y = y.values

        ds_store = {}


        for station_id in self.stations:

            indices = coord.index[coord.station_id==station_id]

            indices_train, indices_val = train_test_split(indices, test_size=0.3, shuffle=False)

            X_train, X_val = torch.from_numpy(X[indices_train]), torch.from_numpy(X[indices_val])
            y_train, y_val = torch.from_numpy(y[indices_train]), torch.from_numpy(y[indices_val])
            time_train, time_val = (torch.from_numpy(coord.values[indices_train, 1].astype('float')), 
                                   torch.from_numpy(coord.values[indices_val, 1].astype('float')))

            # Create Sequences
            if create_seq:
                    time_train, X_train, y_train = self.create_sequence(
                        time_train, X_train, y_train, 
                        window_size=window_size
                    )

                    time_val, X_val, y_val = self.create_sequence(
                        time_val, X_val, y_val, 
                        window_size=window_size
                    )

            ds_store[station_id] = {
                'train': data.TensorDataset(time_train, X_train, y_train),
                'val': data.TensorDataset(time_val, X_val, y_val)
            }

        return ds_store


    def get_dataloader(self, station_id, train=True, batch_size=64, shuffle=False):
        
        if train: 
            return data.DataLoader(
                self.ds_store[station_id]['train'], shuffle=shuffle, batch_size=batch_size
            )
        else:
            return data.DataLoader(
                self.ds_store[station_id]['val'], shuffle=shuffle, batch_size=batch_size
            )