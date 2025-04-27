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
            "minimal_threshold": 0.01
        },
        "SeaGuard_Nord_Conductivity_Sensor.Temperature": {
            "value_range": (5.0, 12.0),
            "minimal_threshold": 0.01
        },
        "SeaGuard_Nord_Pressure_Sensor.Pressure": {
            "value_range": (315.0, 355.0),
            "minimal_threshold": 0.0003
        },
        "SeaGuard_Nord_Pressure_Sensor.Temperature": {
            "value_range": (5.0, 12.0),
            "minimal_threshold": 0.01
        },
        "SeaGuard_Sor_Conductivity_Sensor.Conductivity": {
            "value_range": (32.0, 38.0), 
            "minimal_threshold": 0.01
        },
        "SeaGuard_Sor_Conductivity_Sensor.Temperature": {
            "value_range": (5.0, 12.0),
            "minimal_threshold": 0.01
        },
        "SeaGuard_Sor_Pressure_Sensor.Pressure": {
            "value_range": (710.0, 760.0),
            "minimal_threshold": 0.0003
        },
        "SeaGuard_Sor_Pressure_Sensor.Temperature": {
            "value_range": (5.0, 12.0),
            "minimal_threshold": 0.01
        }
    }
}

feature_columns = list(parameter_config[data_source].keys())

value_range_config = {}
for feature in feature_columns:
    if 'value_range' in parameter_config[data_source][feature]:
        value_range_config[feature] = parameter_config[data_source][feature]['value_range']

LSTM_size_layer = 3
LSTM_size = 100

# Transformer configuration
transformer_dim = 256       
transformer_heads = 16      
transformer_layers = 8      
transformer_ff_dim = 768    
transformer_dropout = 0.15  

mr_num_segments = 12           
mr_temporal_kernel_size = 5

# Training parameters
epoch_train = 1000          
lr_train = 0.0005          

# Update parameters
epoch_update = 100         
lr_update = 0.001          

# Update AnomalousThresholdGenerator parameters 
update_G_epoch = 100        
update_G_lr = 0.001        

log_dir = f"results/{data_source}"

def init_config():
    """Initialize configuration for AdapAD"""
    global update_G_epoch 
    global update_G_lr

    # Predictor configuration
    predictor_config = dict()
    predictor_config['lookback_len'] = 16               
    predictor_config['prediction_len'] = 1
    predictor_config['train_size'] = 4*predictor_config['lookback_len'] + \
                                     predictor_config['prediction_len'] 
    
    # Minimal threshold for anomaly detection
    system_threshold = 0.048
    
    return predictor_config, parameter_config, system_threshold
