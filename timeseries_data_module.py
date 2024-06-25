from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
import numpy as np
import lightning as L
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

import timeseries_dataset as ts_ds

class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data:pd.DataFrame, 
        sequences:list[tuple[int, list[str]]],
        pred_columns:list[str], 
        batch_size=32, 
        pred_len=8, 
        step=2):
        
        super().__init__()
        self.data = data
        self.sequences = sequences
        self.pred_columns = pred_columns
        self.batch_size = batch_size
        self.pred_len = pred_len
        self.step = step
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
        
        self.train_dataset = ts_ds.TimeSeriesDataset(train_data, self.sequences, self.pred_columns, self.pred_len, self.step)
        self.val_dataset = ts_ds.TimeSeriesDataset(val_data, self.sequences, self.pred_columns, self.pred_len, self.step)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)