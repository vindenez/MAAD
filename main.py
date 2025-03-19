import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import math
import json
from collections import defaultdict
from functools import reduce
import pickle
import sys

from utils import *
from learning_models import *
from supporting_components import *
import config as config

torch.manual_seed(0)
        
class AdapAD:
    def __init__(self, predictor_config, value_range_config, minimal_threshold, feature_columns):
        # operation range of the framework
        self.value_range = value_range_config
        self.operation_range = value_range_config
        
        self.sensor_range = NormalValueRangeDb()
        
        # configuration for predictor
        self.predictor_config = predictor_config
        self.feature_columns = feature_columns
        self.num_features = len(feature_columns)
        self.target_feature = 0  # Default target feature is the first one
        
        # init learning components
        self.data_predictor = NormalDataPredictor(
            config.LSTM_size_layer, 
            config.LSTM_size,
            self.predictor_config['lookback_len'],
            self.predictor_config['prediction_len'],
            self.num_features,
            self.target_feature  # Pass target feature index
        )
        
        # Use standard LSTM for the threshold generator
        self.generator = AnomalousThresholdGenerator(
            config.LSTM_size_layer,
            config.LSTM_size,
            self.predictor_config['lookback_len'],
            self.predictor_config['prediction_len']
        )
        
        # immediate databases
        self.predicted_vals = PredictedNormalDataDb()
        self.minimal_threshold = minimal_threshold
        print('Minimal threshold:', self.minimal_threshold)
        self.thresholds = AnomalousThresholdDb()
        self.thresholds.append(self.minimal_threshold)
        
        # detected anomalies
        self.anomalies = list()
        
        # for logging purpose
        self.feature_columns = feature_columns
        self.f_name = 'results/conductivity/' + config.data_source + '_' + str(minimal_threshold) + '.csv'
        print(self.f_name)
        os.makedirs(os.path.dirname(self.f_name), exist_ok=True)
        self.f_log = open(self.f_name, 'w')
        
        # Create header with simplified format focusing on target feature
        target_feature_name = feature_columns[self.target_feature]
        header = f'timestamp,{target_feature_name}_observed,{target_feature_name}_predicted,'
        header += f'{target_feature_name}_low,{target_feature_name}_high,{target_feature_name}_anomalous,'
        header += 'err,threshold,attention_weights\n'
        
        self.f_log.write(header)
        self.f_log.close()
        
    def set_training_data(self, data):
        # data is now a 2D array [time_steps, features]
        data = self.__normalize_data(data)
        self.observed_vals = DataSubject(data)
        
    def train(self, data):
        # Get training data once and store it in a variable
        train_data = self.observed_vals.get_training_data()
        
        # Train the normal data predictor
        self.data_predictor.train(config.epoch_train, 
                                 config.lr_train,
                                 train_data)  # Use the stored variable
        print('Trained NormalDataPredictor')
        
        # Get predictions for training data - no need to call get_training_data() again
        for i in range(len(train_data) - self.predictor_config['lookback_len']):
            # Prepare input for prediction
            input_data = train_data[i:i+self.predictor_config['lookback_len']]
            input_tensor = torch.from_numpy(np.array(input_data).reshape(1, -1)).float()
            
            # Get prediction
            train_predicted_val = self.data_predictor.predict(input_tensor)
            self.predicted_vals.append(train_predicted_val)

        # Get observed values (target feature only)
        observed_vals_ = [x[self.target_feature] for x in train_data[self.predictor_config['lookback_len']:]]
        
        # Calculate prediction errors
        predictive_errors = []
        for i in range(len(self.predicted_vals.predicted_vals)):
            prediction_error = NormalDataPredictionErrorCalculator.calc_error(
                self.predicted_vals.predicted_vals[i], 
                observed_vals_[i]
            )
            predictive_errors.append(prediction_error)
            
        # Train the threshold generator
        self.generator.train(config.epoch_train, 
                            config.lr_train, 
                            np.array(predictive_errors))
        print('Trained AnomalousThresholdGenerator')
        
        # Initialize prediction error database
        self.predictive_errors = PredictionErrorDb(predictive_errors)
        
    def __normalize_data(self, data):
        """Normalize data based on value range"""
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # For 2D array (multiple features)
            normalized_data = np.zeros_like(data, dtype=float)
            for i in range(data.shape[1]):
                # Convert data to float, handling any non-numeric values
                feature_data = np.array(data[:, i], dtype=float)
                min_val = self.sensor_range.lower(i)
                max_val = self.sensor_range.upper(i)
                normalized_data[:, i] = (feature_data - min_val) / (max_val - min_val)
            return normalized_data
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            # For 1D array (single timestep, multiple features)
            normalized_data = np.zeros_like(data, dtype=float)
            for i in range(data.shape[0]):
                # Convert data to float, handling any non-numeric values
                feature_data = float(data[i])
                min_val = self.sensor_range.lower(i)
                max_val = self.sensor_range.upper(i)
                normalized_data[i] = (feature_data - min_val) / (max_val - min_val)
            return normalized_data
        else:
            # For scalar value (single feature)
            min_val = self.sensor_range.lower(0)
            max_val = self.sensor_range.upper(0)
            return (float(data) - min_val) / (max_val - min_val)
            
    def __reverse_normalized_data(self, data, feature_idx=0):
        """Reverse normalization for a specific feature"""
        min_val = self.sensor_range.lower(feature_idx)
        max_val = self.sensor_range.upper(feature_idx)
        return data * (max_val - min_val) + min_val
        
    def __prepare_data_for_prediction(self, pos):
        """Prepare data for prediction"""
        past_observations = self.observed_vals.get_tail(self.predictor_config['lookback_len'])
        past_observations = np.array(past_observations).reshape(1, -1)
        return torch.from_numpy(past_observations).float()
        
    def is_inside_range(self, val, feature_idx):
        """Check if a value is inside the expected range for a feature"""
        # Get the range for this feature
        min_val, max_val = self.value_range.get(feature_idx, (-float('inf'), float('inf')))
        
        # Check if the value is inside the range
        denormalized_val = self.__reverse_normalized_data(val, feature_idx)
        is_inside = min_val <= denormalized_val <= max_val
        
        return is_inside
            
    def __is_default_normal(self):
        # Add debugging print
        result = self.observed_vals.get_length() <= self.predictor_config['train_size'] + self.predictor_config['lookback_len']
        return result
            
    def __logging(self, is_anomalous_ret):
        """Log results to file with attention weights for all features"""
        self.f_log = open(self.f_name, 'a')
        
        # Get the latest values
        observed = self.observed_vals.get_tail()
        predicted = self.predicted_vals.get_tail()
        threshold = self.thresholds.get_tail()
        
        # Get attention weights for logging
        attention_weights = self.data_predictor.get_attention_weights()
        
        # Start with timestamp
        timestamp = len(self.observed_vals.observed_vals)
        log_line = f"{timestamp},"
        
        # Log only the target feature prediction details
        target_observed = observed[self.target_feature]
        target_observed_denorm = self.__reverse_normalized_data(target_observed, self.target_feature)
        target_predicted_denorm = self.__reverse_normalized_data(predicted, self.target_feature)
        low = self.__reverse_normalized_data(predicted - threshold, self.target_feature)
        high = self.__reverse_normalized_data(predicted + threshold, self.target_feature)
        
        # Add target feature data to log
        log_line += f"{target_observed_denorm:.6f},{target_predicted_denorm:.6f},{low:.6f},{high:.6f},{is_anomalous_ret},"
        
        # Add error and threshold
        if hasattr(self, 'predictive_errors') and self.predictive_errors.get_length() > 0:
            err = self.predictive_errors.get_tail()
            log_line += f"{err:.6f},{threshold:.6f},"
        else:
            log_line += f"N/A,{threshold:.6f},"
        
        # Add attention weights information
        if attention_weights is not None:
            attention_weights = attention_weights.data.numpy()
            
            # For each feature, calculate average attention across time steps
            avg_attention = np.mean(attention_weights[0], axis=0)
            
            # Add attention weights for each feature
            attention_strs = []
            for i, feature_name in enumerate(self.feature_columns):
                attention_strs.append(f"{feature_name}:{avg_attention[i]:.4f}")
            
            log_line += "|".join(attention_strs)
        else:
            log_line += "No attention data"
        
        log_line += "\n"
        self.f_log.write(log_line)
        self.f_log.close()
            
    def is_anomalous(self, observed_val):
        is_anomalous_ret = False
        
        # Check for missing values before processing
        if hasattr(observed_val, '__iter__') and not isinstance(observed_val, str):
            if np.any(np.equal(observed_val, -999)) or np.any(np.isnan(observed_val)) or len(observed_val) == 0:
                is_anomalous_ret = True
                self.__logging(is_anomalous_ret)
                return is_anomalous_ret
        
        # save observed vals to the object
        observed_val = np.array(observed_val, dtype=float)
        observed_val = self.__normalize_data(observed_val)
        self.observed_vals.append(observed_val)
        supposed_anomalous_pos = self.observed_vals.get_length()
        
        # predict normal value
        past_observations = self.__prepare_data_for_prediction(supposed_anomalous_pos)
        predicted_val = self.data_predictor.predict(past_observations)
        self.predicted_vals.append(predicted_val)
        
        # perform range check for target feature
        if not self.is_inside_range(observed_val[self.target_feature], self.target_feature):
            self.anomalies.append(supposed_anomalous_pos)
            is_anomalous_ret = True
        else:
            self.generator.eval()
            past_predictive_errors = self.predictive_errors.get_tail(self.predictor_config['lookback_len'])
            past_predictive_errors = torch.from_numpy(np.array(past_predictive_errors).reshape(1, -1)).float()
            
            with torch.no_grad():       
                threshold = self.generator(past_predictive_errors)
                threshold = threshold.data.numpy()
                threshold = max(threshold[0,0], self.minimal_threshold)
            self.thresholds.append(threshold)
        
            prediction_error = NormalDataPredictionErrorCalculator.calc_error(
                predicted_val, 
                observed_val[self.target_feature]
            )
            self.predictive_errors.append(prediction_error)
            
            if prediction_error > threshold:
                if not self.__is_default_normal():
                    is_anomalous_ret = True
                    self.anomalies.append(supposed_anomalous_pos)
            
            self.data_predictor.update(
                config.epoch_update, 
                config.lr_update, 
                past_observations, 
                observed_val[self.target_feature]
            )
                
            if is_anomalous_ret or threshold > self.minimal_threshold:
                self.generator.update(
                    config.update_G_epoch,
                    config.update_G_lr,
                    past_predictive_errors, 
                    prediction_error
                )
            
        self.__logging(is_anomalous_ret)
            
        return is_anomalous_ret

    def clean(self):
        self.predicted_vals.clean(self.predictor_config['lookback_len'])
        self.predictive_errors.clean(self.predictor_config['lookback_len'])
        self.thresholds.clean(self.predictor_config['lookback_len'])
              

if __name__ == "__main__":
    predictor_config, value_range_config, minimal_threshold = config.init_config()
    if not minimal_threshold:
        raise Exception("It is mandatory to set a minimal threshold")
    
    data_source = pd.read_csv(config.data_source_path)
    
    # Clean column names by stripping whitespace
    data_source.columns = [col.strip() for col in data_source.columns]
    
    # Print available columns to help debug
    print("Available columns in dataset (after cleaning):", data_source.columns.tolist())
    
    # Select feature columns based on config
    feature_columns = config.feature_columns
    print(f"Configured features: {feature_columns}")
    
    # Check if all configured feature columns exist in the dataset
    missing_columns = [col for col in feature_columns if col not in data_source.columns]
    if missing_columns:
        print(f"Warning: The following configured columns are missing from the dataset: {missing_columns}")
        print("Using only available columns...")
        feature_columns = [col for col in feature_columns if col in data_source.columns]
        
    if not feature_columns:
        raise Exception("No valid feature columns found in the dataset. Please check your configuration.")
    
    print(f"Using features: {feature_columns}")
    
    # Extract data from dataframe
    # Convert all data to numeric values, coercing errors to NaN
    for col in feature_columns:
        data_source[col] = pd.to_numeric(data_source[col], errors='coerce')
    
    data_values = data_source[feature_columns].values
    len_data_subject = len(data_values)
    
    # AdapAD
    AdapAD_obj = AdapAD(predictor_config, value_range_config, minimal_threshold, feature_columns)
    print('GATHERING DATA FOR TRAINING...', predictor_config['train_size'])
    
    observed_data = []
    
    for data_idx in range(len_data_subject):
        measured_values = data_values[data_idx]
        timestamp = str(data_idx)
        
        observed_data.append(measured_values)
        observed_data_sz = len(observed_data)
        
        # perform warmup training or make a decision
        if observed_data_sz == predictor_config['train_size']:
            AdapAD_obj.set_training_data(np.array(observed_data))
            AdapAD_obj.train(measured_values)
            print('------------STARTING TO MAKE DECISION------------')
        elif observed_data_sz > predictor_config['train_size']:
            is_anomalous_ret = AdapAD_obj.is_anomalous(measured_values)
            AdapAD_obj.clean()
        else:
            print('{}/{} to warmup training'.format(len(observed_data), predictor_config['train_size']))
        
            
    print('Done! Check result at {}'.format(AdapAD_obj.f_name))