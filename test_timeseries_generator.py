import pandas as pd
import numpy as np

def generate_test_data(num_rows:int = 50000) -> pd.DataFrame:
    
    df:pd.DataFrame = generate_ts_two_values(num_rows)
    df = add_mult_values(df, 'value1', 'value2')
    df = add_sum_values(df, 'value1', 'value2')
    df = add_sum_values(df, 'value1_mult_value2', 'value1_sum_value2')
    
    return df

def generate_timeseries(num_rows:int, params:list[tuple[float, float, float]]) -> pd.DataFrame:
    # Generate timestamp values
    timestamp = np.arange(1, num_rows + 1)
    
    # Create an empty DataFrame with timestamp column
    df = pd.DataFrame({'timestamp': timestamp})
    
    # Iterate over each tuple in params to create additional sine wave columns with random noise
    for idx, (period, amplitude, phase) in enumerate(params, start=1):
        column_name = f'value{idx}'
        # Generate sine values with added random noise
        noise = np.random.uniform(-1.0, 1.0, num_rows)
        df[column_name] = amplitude * np.sin(2 * np.pi * (timestamp + phase) / period) + noise
    
    return df.astype('float32', copy=True)    

def generate_ts_two_values(num_rows:int) -> pd.DataFrame:
    return generate_timeseries(num_rows, [(10, 23.0, 5.0), (30, 71.0, -3.0)])

def add_mult_values(df:pd.DataFrame, c1:str, c2:str) -> pd.DataFrame:
    df[f'{c1}_mult_{c2}'] = df[c1] * df[c2]
    return df.copy()

def add_div_values(df:pd.DataFrame, c1:str, c2:str) -> pd.DataFrame:
    df[f'{c1}_div_{c2}'] = df[c1] / df[c2]
    return df.copy()

def add_sum_values(df:pd.DataFrame, c1:str, c2:str) -> pd.DataFrame:
    df[f'{c1}_sum_{c2}'] = df[c1] + df[c2]
    return df.copy()

def add_diff_values(df:pd.DataFrame, c1:str, c2:str) -> pd.DataFrame:
    df[f'{c1}_diff_{c2}'] = df[c1] - df[c2]
    return df.copy()

