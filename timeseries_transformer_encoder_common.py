import timeseries_with_transformer_encoder_model as ts_tr_enc_model

sequences:list[tuple[int,list[str]]] = [(128, ['value1'])]
pred_columns:list[str] = ['value1']

def create_timeseries_transformer_encoder_model() -> ts_tr_enc_model.TransformerEncoderModel:
    
    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns)
    
    model = ts_tr_enc_model.TransformerEncoderModel(
        input_dim,  # Number of input features
        d_model=32,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=8,  # Number of heads in the multiheadattention models
        num_layers=3
    )
    
    return model    

def load_timeseries_transformer_encoder_model(path:str) -> ts_tr_enc_model.TransformerEncoderModel:

    input_dim = sum(len(columns) for _, columns in sequences)
    max_seq_len = max(seq[0] for seq in sequences)
    out_dim = len(pred_columns)

    model = ts_tr_enc_model.TransformerEncoderModel.load_from_checkpoint(
        path,
        input_dim=input_dim,  # Number of input features
        d_model=32,  # Embedding dimension
        out_dim=out_dim,  # Number of output features
        max_pos_encoder_length=max_seq_len,
        nhead=8,  # Number of heads in the multiheadattention models
        num_layers=3
    )
    
    return model