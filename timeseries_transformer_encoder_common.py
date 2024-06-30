import timeseries_with_transformer_encoder_model as ts_tr_enc_model

"""
sequences:list[tuple[int,list[str]]] = [(128, ['value1'])]
pred_columns:list[str] = ['value1']
scaling_column_groups:dict[str, list[str]] = {'value1': []}
prediction_distance:int = 8
"""

"""
sequences:list[tuple[int,list[str]]] = [(128, ['value1_sum_value2'])]
pred_columns:list[str] = ['value1_sum_value2']
scaling_column_groups:dict[str, list[str]] = {'value1_sum_value2': []}
prediction_distance:int = 8
"""

sequences:list[tuple[int,list[str]]] = [(512, ['result'])]
pred_columns:list[str] = ['result']
scaling_column_groups:dict[str, list[str]] = {'result': []}
prediction_distance:int = 16

def create_timeseries_transformer_encoder_model() -> ts_tr_enc_model.TransformerEncoderModel:
    
    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns)
    
    model = ts_tr_enc_model.TransformerEncoderModel(
        input_dim,  # Number of input features
        d_model=16,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=8,  # Number of heads in the multiheadattention models
        num_layers=3,
        dropout=0.1
    )
    
    return model    

def load_timeseries_transformer_encoder_model(path:str) -> ts_tr_enc_model.TransformerEncoderModel:

    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns)

    model = ts_tr_enc_model.TransformerEncoderModel.load_from_checkpoint(
        path,
        input_dim=input_dim,  # Number of input features
        d_model=16,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=8,  # Number of heads in the multiheadattention models
        num_layers=3,
        dropout=0.1
    )
    
    return model