import timeseries_with_transformer_model as ts_transformer_model

sequences:list[tuple[int,list[str]]] = [(128, ['value1'])]
pred_columns:list[str] = ['value1']

def create_timeseries_transformer_model() -> ts_transformer_model.TimeSeriesTransformer:
    
    # Calculate input_dim based on concatenated sequences
    input_dim = sum(len(columns) for _, columns in sequences)
    
    model = ts_transformer_model.TimeSeriesTransformer(
        input_dim=input_dim,  # Number of input features
        d_model=128,  # Embedding dimension
        nhead=8,  # Number of heads in the multiheadattention models
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        sequences=sequences,  # Pass sequences for total_seq_len calculation
        number_pred_features=len(pred_columns)  # Number of predicted features
    )
    
    return model    

def load_timeseries_transformer_model(path:str) -> ts_transformer_model.TimeSeriesTransformer:

    # Calculate input_dim based on concatenated sequences
    input_dim = sum(len(columns) for _, columns in sequences)

    model = ts_transformer_model.TimeSeriesTransformer.load_from_checkpoint(
        path,
        input_dim=input_dim,  # Number of input features
        d_model=128,  # Embedding dimension
        nhead=8,  # Number of heads in the multiheadattention models
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        sequences=sequences,  # Pass sequences for total_seq_len calculation
        number_pred_features=len(pred_columns)  # Number of predicted features
    )
    
    return model