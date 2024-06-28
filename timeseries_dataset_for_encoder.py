import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDatasetForEncoder(Dataset):
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
        self.result_seq_len = sum((seq[0] * len(seq[1])) for seq in sequences)
        self.history_len = max(seq[0] for seq in sequences)
        self.y_len = len(pred_columns)

    def __len__(self):
        result = len(self.data) - self.history_len
        return result

    def __getitem__(self, idx):
        src_sequences = []
        
        j = idx
        for history_len, columns in self.sequences:
            src = self.data.iloc[j:j+history_len][columns].values
            src_sequences.append(src)
            j += history_len
        
        y = self.data.iloc[idx+1][self.pred_columns].values  
        
        src_sequence_tensor = torch.tensor(src_sequences, dtype=torch.float32).view(self.result_seq_len, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(self.y_len)
        
        return src_sequence_tensor, y_tensor
    
def to_sequences(
        data:pd.DataFrame, 
        sequences:list[tuple[int, list[str]]],
        pred_columns:list[str]):
    
    result_seq_len = sum((seq[0] * len(seq[1])) for seq in sequences)
    max_hystory_len = max(seq[0] for seq in sequences)
    
    y_len = len(pred_columns)
    
    result_rows = len(data) - max_hystory_len
    
    src_result = []
    y_result = []
    for i in range(result_rows):
        j = i
        for history_len, columns in sequences:
            src = data.iloc[j:j+history_len][columns].values
            src_result.append(src)
            j += history_len
        
        y = data[i + max_hystory_len][pred_columns].values
        y_result.append(y)
    return torch.tensor(src_result, dtype=torch.float32).view(-1, result_seq_len, 1), torch.tensor(y, dtype=torch.float32).view(-1, y_len)