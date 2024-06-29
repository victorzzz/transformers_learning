import lightning as L

import torch
from torch.utils.data import DataLoader
import timeseries_dataset as ts_ds
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

import matplotlib.pyplot as plt

import timeseries_with_transformer_encoder_model as ts_tr_enc_model
import timeseries_transformer_encoder_common as ts_tr_enc_common
import timeseries_datamodule_for_encoder as ts_dm_encoder
import test_timeseries_generator as test_ts_gen

def transform_predictions_to_numpy(predictions):
    """
    Transforms the list of tensors (predictions) into a single numpy array.
    
    Parameters:
    predictions (list of torch.Tensor): List of prediction tensors where each tensor has shape [batch_size, 1].
    
    Returns:
    np.ndarray: Numpy array containing all predictions in the same order.
    """
    # Concatenate the list of tensors along the first dimension (batch dimension)
    concatenated_tensor = torch.cat(predictions, dim=0)
    
    # Convert the concatenated tensor to a numpy array
    numpy_array = concatenated_tensor.numpy()
    
    # Return the flattened numpy array
    return numpy_array.flatten()

if __name__ == "__main__":

    data:pd.DataFrame = test_ts_gen.generate_test_data(4096)

    model = ts_tr_enc_common.load_timeseries_transformer_encoder_model("models/final_tr_enc_2024-06-29-01-43.ckpt")

    data_module = ts_dm_encoder.TimeSeriesDataModuleForEncoder (
            data,
            sequences=ts_tr_enc_common.sequences,
            pred_columns=ts_tr_enc_common.pred_columns,
            pred_distance=4,
            user_tensor_dataset=True,
            batch_size=32
        )

    # Create a Trainer instance
    trainer = L.Trainer()

    # Use the trainer to predict historical data
    historical_predictions = trainer.predict(model, datamodule=data_module)

    if historical_predictions is None:
        raise ValueError("No predictions made")

    numpy_predictions = transform_predictions_to_numpy(historical_predictions)
    scaler = data_module.get_scaler()
    
    z = np.zeros_like(numpy_predictions)
    
    fff = np.column_stack((numpy_predictions, z, z, z, z, z))
    
    fff = scaler.inverse_transform(fff) 
    
    if not isinstance(fff, np.ndarray):
        raise ValueError("No predictions made")
    
    numpy_predictions_inverse_scaled = fff[:,0]
    
    plt.plot(numpy_predictions, label='Predicted value1')
    plt.plot(numpy_predictions_inverse_scaled, label='Predicted value1 inverse scaled')
    plt.plot(data['value1'], label='Actual value1')
    plt.legend()
    plt.show()

    """
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
    """