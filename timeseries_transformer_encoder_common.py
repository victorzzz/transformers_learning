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

sequences:list[tuple[int,list[str]]] = [(512, ['result1', 'result2'])]
pred_columns:list[str] = ['result1', 'result2']
scaling_column_groups:dict[str, list[str]] = {'result1': ['result2']}
prediction_distance:int = 8
d_model_param:int = 8
nhead_param:int = 4
num_layers_param:int = 3
dropout_param:float = 0.1

def create_timeseries_transformer_encoder_model() -> ts_tr_enc_model.TransformerEncoderModel:
    
    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns)
    
    model = ts_tr_enc_model.TransformerEncoderModel(
        input_dim,  # Number of input features
        d_model=d_model_param,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=nhead_param,  # Number of heads in the multiheadattention models
        num_layers=num_layers_param,
        dropout=dropout_param
    )
    
    return model    

def load_timeseries_transformer_encoder_model(path:str) -> ts_tr_enc_model.TransformerEncoderModel:

    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns)

    model = ts_tr_enc_model.TransformerEncoderModel.load_from_checkpoint(
        path,
        input_dim=input_dim,  # Number of input features
        d_model=d_model_param,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=nhead_param,  # Number of heads in the multiheadattention models
        num_layers=num_layers_param,
        dropout=dropout_param
    )
    
    return model