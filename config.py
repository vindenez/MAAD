import numpy as np
import pandas as pd
import os

# Data source path
data_source_path = "data/SeaGuard_4500_2023-01-26_to_2023-04-29.csv"

# Data source configuration
data_source = "SeaGuard" 

parameter_config = {
    "Austevoll": {
        "conductivity_conductivity": {
            "value_range": (25.0, 38.0),
            "minimal_threshold": 0.02
        },
        "pressure_pressure": {
            "value_range": (299.0, 321.0),
            "minimal_threshold": 0.003
        },
        "pressure_temperature": {
            "value_range": (5.0, 17.0),
            "minimal_threshold": 0.0001
        }
    },
    "SeaGuard": {
        "SeaGuard_Nord_Conductivity_Sensor.Conductivity": {
            "value_range": (32.0, 38.0),
            "minimal_threshold": 0.02
        },
        "SeaGuard_Nord_Conductivity_Sensor.Temperature": {
            "value_range": (5.0, 12.0),
            "minimal_threshold": 0.0001
        },
        "SeaGuard_Nord_Pressure_Sensor.Pressure": {
            "value_range": (315.0, 355.0),
            "minimal_threshold": 0.003
        },
        "SeaGuard_Nord_Pressure_Sensor.Temperature": {
            "value_range": (5.0, 12.0),
            "minimal_threshold": 0.0001
        },
        "SeaGuard_Sor_Conductivity_Sensor.Conductivity": {
            "value_range": (32.0, 38.0), 
            "minimal_threshold": 0.02
        },
        "SeaGuard_Sor_Conductivity_Sensor.Temperature": {
            "value_range": (5.0, 12.0),
            "minimal_threshold": 0.0001
        },
        "SeaGuard_Sor_Pressure_Sensor.Pressure": {
            "value_range": (710.0, 760.0),
            "minimal_threshold": 0.003
        },
        "SeaGuard_Sor_Pressure_Sensor.Temperature": {
            "value_range": (5.0, 12.0),
        }
    }
}

# Extract feature columns based on the selected data source
feature_columns = list(parameter_config[data_source].keys())

# Create a flat value_range_config for easier access in the code
value_range_config = {}
for feature in feature_columns:
    if 'value_range' in parameter_config[data_source][feature]:
        value_range_config[feature] = parameter_config[data_source][feature]['value_range']

# LSTM configuration
LSTM_size = 100             # Increased from 100 to 128 for more representation capacity
LSTM_size_layer = 3         # Reduced from 3 to 2 to avoid overfitting

# Transformer configuration
transformer_dim = 96        # Increased from 64 to 96 for richer feature representation
transformer_heads = 4       # Reduced from 4 to 3 (better for small num_features)
transformer_layers = 2      # Keep at 2 layers
transformer_ff_dim = 384    # Increased from 256 to 384 for better expressivity
transformer_dropout = 0.1   # Explicit dropout rate

# Training parameters
epoch_train = 1000          # Increased from 1000 to 2000 for better convergence
lr_train = 0.0005           # Increased slightly from 0.00005 to 0.0001 for faster initial learning

# Update MultivariateNormalDataPredictor parameters
epoch_update = 100          # Reduced from 500 to 10 for frequent but lighter online updates
lr_update = 0.0005          # Increased from 0.0001 to 0.001 for better adaptation to new patterns

# Update AnomalousThresholdGenerator parameters 
update_G_epoch = 100         # Reduced from 500 to 20
update_G_lr = 0.0005        # Increased from 0.0001 to 0.001 for better threshold adaptation

log_dir = f"results/{data_source}"

def init_config():
    """Initialize configuration for AdapAD"""
    global update_G_epoch 
    global update_G_lr

    # Predictor configuration
    predictor_config = dict()
    predictor_config['lookback_len'] = 5                
    predictor_config['prediction_len'] = 1
    predictor_config['train_size'] = 5*predictor_config['lookback_len'] + \
                                     predictor_config['prediction_len'] 
    
    # Minimal threshold for anomaly detection
    system_threshold = 0.01
    
    return predictor_config, parameter_config, system_threshold
