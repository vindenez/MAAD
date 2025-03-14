import numpy as np
import pandas as pd
import os

## This is a config file for AdapAD ##
# The hyper-parameters are set such that AdapAD achieves the highest precisions #

# Data source path
#data_source_path = "data/data_187_30min_without_dcps.csv"
data_source_path = "data/data_3716_20230829_0400_to_20231114_1300.csv"

# Data source configuration
data_source = "att-lstm"  # Used for output file naming

# Feature columns to use (multivariate)
feature_columns = ["conductivity_conductivity", "conductivity_temperature", "conductivity_salinity", 
                  "conductivity_density", "conductivity_soundspeed"]

# LSTM configuration
LSTM_size = 100  # Hidden units in LSTM
LSTM_size_layer = 3  # Number of LSTM layers
attention_size = 32  # Size of attention layer

# Training parameters
epoch_train = 10  # Epochs for initial training
lr_train = 0.001  # Learning rate for initial training

# Update parameters
epoch_update = 50  # Epochs for online updates
lr_update = 0.0002  # Learning rate for online updates

# Threshold generator parameters
update_G_epoch = 50  # Epochs for threshold generator updates
update_G_lr = 0.00005  # Learning rate for threshold generator updates

# Attention parameters
attention_dropout = 0.1  # Dropout rate for attention layer
use_multihead_attention = True  # Whether to use multi-head attention
num_attention_heads = 4  # Number of attention heads if using multi-head attention

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
    
    # Value range configuration for each feature
    # Format: {feature_index: (min_value, max_value)}
    value_range_config = {
        0: (25.0, 36.0),        # conductivity_conductivity
        1: (2.0, 20.0),         # conductivity_temperature
        2: (18.0, 32.0),        # conductivity_salinity
        3: (1008.0, 1030.0),    # conductivity_density
        4: (1460.0, 1510.0),    # conductivity_soundspeed
    }
    
    # Minimal threshold for anomaly detection
    minimal_threshold = 0.0070
    
    return predictor_config, value_range_config, minimal_threshold

def get_attention_config():
    """Get attention-specific configuration"""
    return {
        'attention_size': attention_size,
        'attention_dropout': attention_dropout,
        'use_multihead_attention': use_multihead_attention,
        'num_attention_heads': num_attention_heads
    }

# Ensure the results directory exists
os.makedirs(f"results/{data_source}", exist_ok=True)