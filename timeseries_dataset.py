import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        data:pd.DataFrame, 
        sequences:list[tuple[int, list[str]]],
        pred_columns:list[str], 
        pred_len:int=8, 
        step:int=2):
        """
        data: DataFrame containing all the data
        sequences: List of tuples [(history_len, [column names]), ...]
        pred_columns: List of column names for the predicted values
        pred_len: Number of future steps to predict
        step: Step size for selecting future values
        """
        self.data = data
        self.sequences = sequences
        self.pred_columns = pred_columns
        self.pred_len = pred_len
        self.step = step
        self.max_history_len = max(seq[0] for seq in sequences)

    def __len__(self):
        return len(self.data) - self.max_history_len - self.pred_len

    def __getitem__(self, idx):
        src_sequences = []
        tgt_sequences = []
        
        for history_len, columns in self.sequences:
            # Prepare source sequences
            src = self.data.iloc[idx:idx+history_len][columns].values
            src_sequences.append(src)
            
            # Prepare target sequences
            tgt = self.data.iloc[idx+1:idx+history_len+1][columns].values
            tgt_sequences.append(tgt)
        
        # Concatenate all sequences
        src = np.concatenate(src_sequences, axis=1)
        tgt = np.concatenate(tgt_sequences, axis=1)
        
        # Select future values with a step for prediction
        y_indices = list(range(idx+self.max_history_len, idx+self.max_history_len+self.pred_len, self.step))
        y = self.data.iloc[y_indices][self.pred_columns].values  # Use pred_columns for y
        
        return src, tgt, y