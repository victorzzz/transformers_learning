import pandas as pd
import numpy as np

"""
def generate_test_data(num_rows:int = 50000) -> pd.DataFrame:
    
    df:pd.DataFrame = generate_ts_two_values(num_rows)
    df = add_mult_values(df, 'value1', 'value2')
    df = add_sum_values(df, 'value1', 'value2')
    df = add_sum_values(df, 'value1_mult_value2', 'value1_sum_value2')
    
    return df    
"""

def generate_test_data(num_rows:int) -> pd.DataFrame:
    
    df:pd.DataFrame = generate_timeseries(
        num_rows, 
        [   (47, 17.0, 7.0, 3.0), 
            (77, 15.0, -31.0, 4.0), 
            (54, 7.0, -10.0, 1.0), 
            (63, 9.0, 7.0, 1.0),
            (32, 11.0, -10.0, 3.0), 
            (90, 5.0, 7.0, 2.0),            
            (4377, 433.0, 33.0, 6.0),
            (5732, 511.0, 1033.0, 8.0)])
    
    value1 = df['value1']
    value2 = df['value2']
    value3 = df['value3']
    value4 = df['value4']
    value5 = df['value5']
    value6 = df['value6']
    value7 = df['value7']
    value8 = df['value8']
    
    df["result1"] = value1 * value2 + value3 * value4 + value5 * value6 + value7
    df["result2"] = value1 * value3 - value2 * value5 + value4 * value6 + value8   

    return df

def generate_timeseries(num_rows:int, params:list[tuple[float, float, float, float]]) -> pd.DataFrame:
    # Generate timestamp values
    timestamp = np.arange(1, num_rows + 1)
    
    # Create an empty DataFrame with timestamp column
    df = pd.DataFrame({'timestamp': timestamp})
    
    # Iterate over each tuple in params to create additional sine wave columns with random noise
    for idx, (period, amplitude, phase, randomness) in enumerate(params, start=1):
        column_name = f'value{idx}'
        # Generate sine values with added random noise
        noise = np.random.uniform(-randomness, randomness, num_rows)
        df[column_name] = amplitude * np.sin(2 * np.pi * ((timestamp + phase) / period)) + noise
    
    return df.astype('float32', copy=True)    

def generate_ts_two_values(num_rows:int) -> pd.DataFrame:
    return generate_timeseries(num_rows, [(15, 23.0, 7.0, 5.0), (51, 87.0, -31.0, 15.0)])

def add_mult_values(df:pd.DataFrame, c1:str, c2:str, new_col:str | None = None) -> pd.DataFrame:
    if new_col is None:
        new_col = f'{c1}_mult_{c2}'
    df[new_col] = df[c1] * df[c2]
    return df.copy()

def add_div_values(df:pd.DataFrame, c1:str, c2:str, new_col:str | None = None) -> pd.DataFrame:
    if new_col is None:
        new_col = f'{c1}_div_{c2}'
    df[new_col] = df[c1] / df[c2]
    return df.copy()

def add_sum_values(df:pd.DataFrame, c1:str, c2:str, new_col:str | None = None) -> pd.DataFrame:
    if new_col is None:
        new_col = f'{c1}_sum_{c2}'
    df[new_col] = df[c1] + df[c2]
    return df.copy()

def add_diff_values(df:pd.DataFrame, c1:str, c2:str, new_col:str | None = None) -> pd.DataFrame:
    if new_col is None:
        new_col = f'{c1}_diff_{c2}'
    df[new_col] = df[c1] - df[c2]
    return df.copy()

