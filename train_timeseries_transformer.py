import pandas as pd
import datetime as dt
import numpy as np
import lightning as L
import torch
import matplotlib.pyplot as plt

import timeseries_data_module as ts_dm
import test_timeseries_generator as test_ts_gen

import timeseries_transformer_common as ts_transformer_common

if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')

    data:pd.DataFrame = test_ts_gen.generate_test_data(8192)

    plt.plot(data['value1'], label='value1')
    plt.show()

    # Create data module
    data_module = ts_dm.TimeSeriesDataModule(
        data,
        batch_size=32,
        sequences=ts_transformer_common.sequences,
        pred_columns=ts_transformer_common.pred_columns
    )

    model = ts_transformer_common.create_timeseries_transformer_model()

    # Train the model
    trainer = L.Trainer(max_epochs=30, log_every_n_steps=3)
    trainer.fit(model, data_module)

    # Validate the model
    trainer.validate(model, data_module)

    model.eval()

    date_str = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_path = f"models/final_model{date_str}.ckpt"

    trainer.save_checkpoint(model_path)

    print(f"Checkpoint save manually at {model_path}")