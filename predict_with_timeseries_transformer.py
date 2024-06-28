import lightning as L

import torch
from torch.utils.data import DataLoader
import timeseries_dataset as ts_ds
import pandas as pd
import numpy as np
import timeseries_with_transformer_model as ts_transformer_model
import timeseries_transformer_common as ts_transformer_common
import test_timeseries_generator as test_ts_gen

data:pd.DataFrame = test_ts_gen.generate_test_data(1000)

# Create the historical prediction dataset and dataloader

pred_dataset = ts_ds.HistoricalPredictionDataset(data, ts_transformer_common.sequences)
pred_dataloader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

model = ts_transformer_common.load_timeseries_transformer_model("models/final_model2024-06-27-00-04.ckpt")

# Create a Trainer instance
trainer = L.Trainer()

# Use the trainer to predict historical data
historical_predictions = trainer.predict(model, dataloaders=pred_dataloader)

if historical_predictions is None:
    raise ValueError("No predictions made")

# Flatten predictions to match the historical data
flat_predictions = [pred.detach().numpy() for preds in historical_predictions for pred in preds]
flat_predictions = np.vstack(flat_predictions)

# Convert to DataFrame for visualization
prediction_df = pd.DataFrame(flat_predictions, columns=['value1'])

# Convert to DataFrame for visualization
prediction_df = pd.DataFrame(flat_predictions, columns=['value1'])

# Plot actual vs predicted values
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(data['value1'], label='Actual value1')
plt.plot(range(window_size, window_size + len(prediction_df)), prediction_df['value1'], label='Predicted value1')
plt.legend()
plt.show()