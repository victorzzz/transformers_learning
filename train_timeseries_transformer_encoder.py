import pandas as pd
import datetime as dt
import numpy as np
import lightning as L
import torch
import matplotlib.pyplot as plt

import timeseries_datamodule_for_encoder as ts_dm_encoder
import test_timeseries_generator as test_ts_gen

import timeseries_transformer_encoder_common as ts_tr_enc_common

if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')

    data:pd.DataFrame = test_ts_gen.generate_test_data(8192 * 16)

    plt.plot(data['result'], label='value1')
    plt.show()

    # Create data module
    data_module = ts_dm_encoder.TimeSeriesDataModuleForEncoder (
        data,
        sequences=ts_tr_enc_common.sequences,
        pred_columns=ts_tr_enc_common.pred_columns,
        scaling_column_groups=ts_tr_enc_common.scaling_column_groups,
        pred_distance=ts_tr_enc_common.prediction_distance,
        user_tensor_dataset=True,
        batch_size=128
    )

    model = ts_tr_enc_common.create_timeseries_transformer_encoder_model()

    # Train the model
    trainer = L.Trainer(
        # overfit_batches=5,
        # fast_dev_run=5,
        max_epochs=20, 
        log_every_n_steps=5)
    
    trainer.fit(model, data_module)

    # Validate the model
    trainer.validate(model, data_module)

    model.eval()

    date_str = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_path = f"models/final_tr_enc_{date_str}.ckpt"

    trainer.save_checkpoint(model_path)

    print(f"Checkpoint save manually at {model_path}")