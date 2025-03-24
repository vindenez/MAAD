import numpy as np
import pandas as pd
import os

# Data source path
data_source_path = "data/Austevoll_Autumn_2023_no_dcps_small.csv"

# Data source configuration
data_source = "LSTM-AE" 

# Configuration using column names
value_range_config = {
    "conductivity_conductivity": (25.0, 38.0),
    #"conductivity_temperature": (5.0, 17.0),
    #"conductivity_salinity": (18.0, 32.0),
    #"conductivity_density": (1008.0, 1030.0),
    #"conductivity_soundspeed": (1460.0, 1510.0),
    "pressure_pressure": (299.0, 321.0),
    "pressure_temperature": (5.0, 17.0)
}

weight_config = {
    "conductivity_conductivity": 1/16,
    "pressure_pressure": 5/16,
    "pressure_temperature": 5/8
}

# Extract the uncommented columns
feature_columns = list(value_range_config.keys())

# LSTM configuration
LSTM_size = 100             # Hidden units in LSTM
LSTM_size_layer = 3         # Number of LSTM layers for Encoder and Decoder

# Training parameters
epoch_train = 500           # Epochs for initial training
lr_train = 0.0005           # Learning rate for initial training

# Update MultivariateNormalDataPredictor parameters
epoch_update = 50           # Epochs for online updates
lr_update = 0.001           # Learning rate for online updates

# Update AnomalousThresholdGenerator parameters 
update_G_epoch = 50         # Epochs for threshold generator updates
update_G_lr = 0.001        # Learning rate for threshold generator updates

log_dir = f"results/{data_source}"

def init_config():
    """Initialize configuration for AdapAD"""
    global update_G_epoch 
    global update_G_lr

    # Predictor configuration
    predictor_config = dict()
    predictor_config['lookback_len'] = 3
    predictor_config['prediction_len'] = 1
    predictor_config['train_size'] = 5*predictor_config['lookback_len'] + \
                                     predictor_config['prediction_len'] 
    
    # Minimal threshold for anomaly detection
    minimal_threshold = 0.0025
    
    return predictor_config, value_range_config, minimal_threshold
