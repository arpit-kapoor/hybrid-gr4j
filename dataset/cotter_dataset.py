import os
import yaml
import pandas as pd
import numpy as np
import datetime as dt

import torch
import torch.nn as nn
import torch.utils.data as data

from sklearn.preprocessing import StandardScaler


WINDOW_SIZE = 7

class CotterData(object):
    """Class to read cotter river precipitation, evapotranspiration and 
        stream flow data
    """

    def __init__(self, config_file:str, train_dates:tuple, val_dates:tuple, key:str='cotter', scale:bool=True, create_seq:bool=True, keep_z:bool=True) -> None:

        # Read data config
        with open(config_file, 'r', encoding='UTF-8') as stream:
            config = yaml.safe_load(stream)[key]

        # Parse yaml
        self.data_location = config['location']
        self.silos_file = config['silos']
        self.gauge_file = config['gauge']

        # Read DF
        self._df = self.get_complete_data()

        # Cols
        X_col = ['daily_rain', 'et_tall_crop', 'flow_ml']
        y_col = ['flow_ml']

        # Input data
        dates = self._df[['date']]
        X = self._df[X_col]
        y = self._df[y_col]

        self.trainset, self.valset = self.process_data(
            X, y, dates, 
            scale, 
            train_dates, 
            val_dates, 
            create_seq=create_seq, 
            keep_z=keep_z
        )

    def get_dataloader(self, train=True, batch_size=64):
        if train:
            return data.DataLoader(
                self.trainset, shuffle=True, batch_size=batch_size
            )
        else:
            return data.DataLoader(
                self.valset, shuffle=False, batch_size=batch_size
            )


    def create_sequence(self, X, y, keep_z=False, window_size=5):
        # Create empyty sequences
        Xs, ys = [], []

        if keep_z:
            zs = []

        # Add sequences to Xs and ys
        for i in range(len(X)-window_size):
            Xs.append(X[i: (i + window_size)])
            ys.append(y[i + window_size])

            if keep_z:
                zs.append(X[i + window_size, :2])


        Xs, ys = torch.stack(Xs), torch.stack(ys)

        if keep_z:
            zs = torch.stack(zs)
            return Xs, ys, zs
        else:
            return Xs, ys


    def process_data(self, X, y, dates, scale, train_dates, val_dates,create_seq=False, window_size=WINDOW_SIZE, keep_z=False):

        # Scaling preference
        self.scale = scale
        if scale:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
        
            # Scale
            X = self.x_scaler.fit_transform(X)
            y = self.y_scaler.fit_transform(y)

        # Train data
        X_train = X[(dates.date>=train_dates[0])&(dates.date<=train_dates[1])]
        y_train = y[(dates.date>=train_dates[0])&(dates.date<=train_dates[1])]

        # Val data
        X_val = X[(dates.date>=val_dates[0])&(dates.date<=val_dates[1])]
        y_val = y[(dates.date>=val_dates[0])&(dates.date<=val_dates[1])]

        # Convert to Tensor
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)

        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)


        # Create Sequences
        if create_seq:
            if keep_z:
                X_train, y_train, z_train = self.create_sequence(X_train, y_train, keep_z)
                X_val, y_val, z_val = self.create_sequence(X_val, y_val, keep_z)

                trainset = data.TensorDataset(X_train, z_train, y_train)
                valset = data.TensorDataset(X_val, z_val, y_val)
            else:
                X_train, y_train = self.create_sequence(X_train, y_train, keep_z)
                X_val, y_val = self.create_sequence(X_val, y_val, keep_z)

                trainset = data.TensorDataset(X_train, y_train)
                valset = data.TensorDataset(X_val, y_val)
        else:
            trainset = data.TensorDataset(X_train, y_train)
            valset = data.TensorDataset(X_val, y_val)

        return trainset, valset

    @staticmethod
    def read_csv(data_location:str, file_name:str, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(
                data_location,
                file_name
            ),
            **kwargs
        )
        return df


    def get_silos_data(self, data_location:str, file_name:str) -> pd.DataFrame:
        
        # Read the csv
        cotter_silos_data = self.read_csv(
            data_location=data_location, 
            file_name=file_name
        )

        # Remove unwanted columns
        columns_to_keep = [col for col in cotter_silos_data.columns if 'source' not in col and col != 'metadata']
        cotter_silos_data = cotter_silos_data.loc[:, columns_to_keep]

        # Fix Date
        cotter_silos_data.rename(columns={'YYYY-MM-DD': 'date'}, inplace=True)
        cotter_silos_data['date'] = pd.to_datetime(cotter_silos_data['date'])

        return cotter_silos_data


    def get_gauge_data(self, data_location:str, file_name:str) -> pd.DataFrame:
        """"""

        # Read CSV
        cotter_gauge_data = self.read_csv(
            data_location=data_location,
            file_name=file_name,
            skiprows=26
        )

        # Handle columns
        cotter_gauge_data.rename(columns={'Date':'date', 'Flow (ML)':'flow_ml', 'Bureau QCode':'qcode'}, inplace=True)
        # cotter_gauge_data.drop(columns=['Bureau QCode'], inplace=True)

        # Handle Date
        cotter_gauge_data['date'] = pd.to_datetime(cotter_gauge_data['date'])

        return cotter_gauge_data


    def get_complete_data(self) -> pd.DataFrame:

        cotter_silos_data = self.get_silos_data(
            self.data_location, 
            self.silos_file
        )

        cotter_gauge_data = self.get_gauge_data(
            self.data_location,
            self.gauge_file
        )

        # Merge datasets
        merged_data = pd.merge(
            cotter_silos_data,
            cotter_gauge_data,
            on='date'
        )

        return merged_data

















