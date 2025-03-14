import numpy as np

def sliding_windows(data, lookback_len, predict_len):
    x = []
    y = []

    for i in range(lookback_len, len(data)):
        _x = data[i-lookback_len:i]
        _y = data[i:i+predict_len]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

def sliding_windows_multivariate(data, seq_length, pred_length):
    """
    Create sliding windows for multivariate time series data
    
    Args:
        data: 2D array of shape [time_steps, features]
        seq_length: length of input sequence
        pred_length: length of prediction sequence
    
    Returns:
        x: 3D array of input sequences [samples, seq_length, features]
        y: 3D array of target sequences [samples, pred_length, features]
    """
    x = []
    y = []
    
    for i in range(len(data) - seq_length - pred_length + 1):
        _x = data[i:(i + seq_length)]
        _y = data[(i + seq_length):(i + seq_length + pred_length)]
        x.append(_x)
        y.append(_y)
        
    return np.array(x), np.array(y)