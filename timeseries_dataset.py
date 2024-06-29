import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        data:pd.DataFrame, 
        sequences:list[tuple[int, list[str]]],
        pred_columns:list[str]):
        """
        data: DataFrame containing all the data
        sequences: List of tuples [(history_len, [column names]), ...]
        pred_columns: List of column names for the predicted values
        """
        self.data = data
        self.sequences = sequences
        self.pred_columns = pred_columns
        self.max_history_len = max(seq[0] for seq in sequences)

    def __len__(self):
        result = len(self.data) - self.max_history_len - 1 
        return result

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
        
        y = self.data.iloc[idx+1][self.pred_columns].values  # Use pred_columns for y
        
        return src, tgt, y
    
class HistoricalPredictionDataset(Dataset):
    def __init__(self, data: pd.DataFrame, sequences: list[tuple[int, list[str]]]) -> None:
        self.data = data
        self.sequences = sequences
        self.max_history_len = max(seq[0] for seq in sequences)

    def __len__(self):
        result = len(self.data) - self.max_history_len - 1  # Calculate the number of samples
        return result

    def __getitem__(self, idx: int):
        sequence_data = []
        for history_len, columns in self.sequences:
            src = self.data.iloc[idx:idx+history_len][columns].values
            sequence_data.append(src)
            
        sequence_data = np.concatenate(sequence_data, axis=1)
        return sequence_data