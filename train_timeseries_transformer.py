import pandas as pd
import numpy as np
import lightning as L
import torch
import matplotlib.pyplot as plt

import timeseries_data_module as ts_dm
import test_timeseries_generator as test_ts_gen
import timeseries_with_transformer_model as ts_transformer_model

torch.set_float32_matmul_precision('medium')

data:pd.DataFrame = test_ts_gen.generate_test_data(50000)
sequences:list[tuple[int,list[str]]] = [(50, ['value1'])]
pred_columns:list[str] = ['value1']

# Create data module
data_module = ts_dm.TimeSeriesDataModule(
    data,
    batch_size=1024,
    sequences=sequences,
    pred_columns=pred_columns,
    pred_len=8,
    step=2
)

# Calculate input_dim based on concatenated sequences
input_dim = sum(len(columns) for _, columns in sequences)

# Create model
model = ts_transformer_model.TimeSeriesTransformer(
    input_dim=input_dim,  # Number of input features
    d_model=64,  # Embedding dimension
    nhead=8,  # Number of heads in the multiheadattention models
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=256,
    dropout=0.1,
    sequences=sequences,  # Pass sequences for total_seq_len calculation
    number_pred_features=len(pred_columns),  # Number of predicted features
    pred_len=8,  # Number of future steps to predict
    step=2  # Step size for future predictions
)

# Train the model
trainer = L.Trainer(max_epochs=5)
trainer.fit(model, data_module)

# Validate the model
trainer.validate(model, data_module)

model.eval()

model_path = "final_model.ckpt"
torch.save(model.state_dict(), model_path)

prediction = trainer.predict(model, datamodule=data_module)

# Plot the prediction
if isinstance(prediction, torch.Tensor):
    prediction = prediction.detach().numpy()    
    plt.plot(prediction[0], label='Prediction')    

