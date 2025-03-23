import numpy as np
import pandas as pd
import os

## This is a config file for AdapAD ##
# The hyper-parameters are set such that AdapAD achieves the highest precisions #

# Data source path
#data_source_path = "data/data_187_30min_without_dcps.csv"
data_source_path = "data/Austevoll_Autumn_2023_no_dcps.csv"

# Data source configuration
data_source = "LSTM-AE"  # Used for output file naming

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

# Extract the uncommented columns
feature_columns = list(value_range_config.keys())

# LSTM configuration
LSTM_size = 100  # Hidden units in LSTM
LSTM_size_layer = 3  # Number of LSTM layers
attention_size = 32  # Size of attention layer

# Training parameters
epoch_train = 500  # Epochs for initial training
lr_train = 0.0005  # Learning rate for initial training

# Update parameters
epoch_update = 50  # Epochs for online updates
lr_update = 0.001  # Learning rate for online updates

# Threshold generator parameters
update_G_epoch = 50  # Epochs for threshold generator updates
update_G_lr = 0.0005  # Learning rate for threshold generator updates

# Attention parameters
attention_dropout = 0.1  # Dropout rate for attention layer
use_multihead_attention = True  # Whether to use multi-head attention
num_attention_heads = 4  # Number of attention heads if using multi-head attention

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
    minimal_threshold = 0.001
    
    return predictor_config, value_range_config, minimal_threshold

def get_attention_config():
    """Get attention-specific configuration"""
    return {
        'attention_size': attention_size,
        'attention_dropout': attention_dropout,
        'use_multihead_attention': use_multihead_attention,
        'num_attention_heads': num_attention_heads
    }
