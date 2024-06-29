from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
import numpy as np
import lightning as L
from sklearn.base import TransformerMixin
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import timeseries_dataset_for_encoder as ts_ds_encoder

class TimeSeriesDataModuleForEncoder(L.LightningDataModule):
    def __init__(
        self, 
        data:pd.DataFrame, 
        sequences:list[tuple[int, list[str]]],
        pred_columns:list[str],
        pred_distance:int,
        user_tensor_dataset:bool, 
        batch_size):
        
        super(TimeSeriesDataModuleForEncoder, self).__init__()
        
        self.data = data
        self.sequences = sequences
        self.pred_columns = pred_columns
        self.pred_distance = pred_distance
        self.user_tensor_dataset = user_tensor_dataset
        self.batch_size = batch_size
        self.scaler = StandardScaler()

    def setup(self, stage=None):
        train_border = int(0.8*len(self.data))
        
        train_data:pd.DataFrame = self.data[:train_border]
        val_data:pd.DataFrame = self.data[train_border:]
        
        # Fit the scaler on the training dataset
        self.scaler.fit(train_data)
        
        scalled_train_data = self.scaler.transform(train_data)
        if isinstance(scalled_train_data, np.ndarray):
            train_data = pd.DataFrame(scalled_train_data, columns=self.data.columns)
        
        scalled_val_data = self.scaler.transform(val_data)
        if isinstance(scalled_val_data, np.ndarray):
            val_data = pd.DataFrame(scalled_val_data, columns=self.data.columns)
        
        if self.user_tensor_dataset:
            train_src, train_y = ts_ds_encoder.to_sequences(train_data, self.sequences, self.pred_columns, self.pred_distance)
            self.train_dataset = TensorDataset(train_src, train_y)
            
            val_scr, val_y = ts_ds_encoder.to_sequences(val_data, self.sequences, self.pred_columns, self.pred_distance)
            self.val_dataset = TensorDataset(val_scr, val_y)
        else:
            self.train_dataset = ts_ds_encoder.TimeSeriesDatasetForEncoder(train_data, self.sequences, self.pred_columns, self.pred_distance)
            self.val_dataset = ts_ds_encoder.TimeSeriesDatasetForEncoder(val_data, self.sequences, self.pred_columns, self.pred_distance)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def get_scaler(self) -> StandardScaler:
        return self.scaler