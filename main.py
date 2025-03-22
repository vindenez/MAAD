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
from config import value_range_config

torch.manual_seed(0)
        
# Extract the uncommented columns
feature_columns = list(value_range_config.keys())

# Initialize the value range database
value_range_db = NormalValueRangeDb()

class AdapAD:
    def __init__(self, predictor_config, minimal_threshold):
        self.predictor_config = predictor_config
        self.minimal_threshold = minimal_threshold
        self.num_features = len(feature_columns)
        
        # Initialize value range database
        self.sensor_range = NormalValueRangeDb()
        
        # Initialize learning components
        self.data_predictor = MultivariateNormalDataPredictor(
            lstm_layer=config.LSTM_size_layer,
            lstm_unit=config.LSTM_size,
            lookback_len=self.predictor_config['lookback_len'],
            prediction_len=self.predictor_config['prediction_len'],
            num_features=self.num_features
        )
        
        self.generator = AnomalousThresholdGenerator(
            lstm_layer=config.LSTM_size_layer,
            lstm_unit=config.LSTM_size,
            lookback_len=self.predictor_config['lookback_len'],
            prediction_len=self.predictor_config['prediction_len'],
            num_features=self.num_features
        )
        
        # Initialize databases
        self.predicted_vals = PredictedNormalDataDb()
        self.thresholds = AnomalousThresholdDb()
        self.thresholds.append(self.minimal_threshold)
        self.anomalies = []
        
        # Initialize predictive errors
        self.predictive_errors = None
        

    def set_training_data(self, training_data):
        """
        Set the training data for the model.
        Args:
            training_data: Numpy array of training data.
        """
        # Normalize the training data
        normalized_training_data = np.array([self.__normalize_data(row) for row in training_data])
        
        # Store the normalized training data
        self.observed_vals = DataSubject(normalized_training_data)
        
        # Store the training data for later use
        self.training_data = normalized_training_data

    def train(self):
        """
        Train the data predictor and threshold generator.
        """
        if not hasattr(self, 'training_data'):
            raise ValueError("Training data not set. Call set_training_data first.")
            
        # Train the data predictor
        trainX, trainY = self.data_predictor.train(
            config.epoch_train,
            config.lr_train,
            self.training_data
        )
        
        # Initialize predicted values and errors
        predicted_values = []
        for i in range(len(trainX)):
            train_predicted_val = self.data_predictor.predict(torch.reshape(trainX[i], (1, -1)))
            predicted_values.append(train_predicted_val[0])  # Take the first (and only) prediction
        
        # Convert to numpy arrays
        trainY = trainY.data.numpy()
        predicted_values = np.array(predicted_values)
        
        # Calculate mean error across features for each time step
        errors = np.mean((trainY - predicted_values)**2, axis=1)
        
        # Create a PredictionErrorDb with scalar errors
        self.predictive_errors = PredictionErrorDb(errors.tolist())
        
        # Get the errors for generator training
        original_errors = self.predictive_errors.get_tail(self.predictive_errors.get_length())
        
        # Apply log transformation to errors before training the generator
        log_transformed_errors = self.log_transform_errors(original_errors)
        print(f"Training generator with log-transformed errors")
        
        # Train the threshold generator with log-transformed errors
        self.generator.train(
            config.epoch_train,
            config.lr_train,
            log_transformed_errors  # Pass the log-transformed errors here
        )

    def __init_log_files(self):
        # Initialize log files for each feature
        log_files = {}
        for feature in feature_columns:
            log_files[feature] = open(f'results/LSTM-AE/{feature}_log.csv', 'w')
            log_files[feature].write('timestamp,observed,predicted,low,high,anomalous,err,threshold\n')
        return log_files

    def __normalize_data(self, data):
        # Normalize data based on the value range configuration
        normalized_data = np.zeros_like(data)
        for i, feature in enumerate(feature_columns):
            min_val, max_val = value_range_config[feature]
            normalized_data[i] = (data[i] - min_val) / (max_val - min_val)
        return normalized_data

    def __reverse_normalized_data(self, normalized_val, feature_idx):
        # Reverse normalization for a single value
        feature = feature_columns[feature_idx]
        min_val, max_val = value_range_config[feature]
        return normalized_val * (max_val - min_val) + min_val

    def __prepare_data_for_prediction(self, supposed_anomalous_pos):
        # Prepare input data for prediction
        # ... existing code ...
        pass

    def calculate_reconstruction_error(self, predicted, observed):
        """
        Calculate the reconstruction error between predicted and observed values.
        
        Args:
            predicted: Predicted values (normalized)
            observed: Observed values (normalized)
            
        Returns:
            float: Mean squared reconstruction error
        """
        return np.mean((predicted - observed)**2)

    def __logging(self, is_anomalous_ret):
        """
        Log the current state for debugging and visualization.
        """
        try:
            # Ensure log directory exists
            if not os.path.exists(config.log_dir):
                os.makedirs(config.log_dir)
            
            # Get the current observation index
            current_idx = self.observed_vals.get_length() - 1
            
            # Only log if we have enough data
            if current_idx >= self.predictor_config['lookback_len']:
                # Get the past observations for prediction
                past_observations = self.observed_vals.get_tail(self.predictor_config['lookback_len'])
                past_observations = np.array(past_observations)
                past_observations_tensor = torch.Tensor(past_observations.reshape(1, -1))
                
                # Get the predicted value
                predicted = self.data_predictor.predict(past_observations_tensor)
                
                # Get the current observation
                current_observation = self.observed_vals.get_tail(1)
                
                # Calculate the reconstruction error using the shared method
                reconstruction_error = self.calculate_reconstruction_error(
                    predicted[0, -1, :],
                    np.array(current_observation)
                )
                
                # Get the current threshold (if available)
                threshold = self.thresholds.get_tail(1) if self.thresholds.get_length() > 0 else self.minimal_threshold
                
                # Create log files for each feature if they don't exist
                if not hasattr(self, 'feature_log_files'):
                    self.feature_log_files = []
                    
                    # Get feature names from config.value_range_config
                    feature_names = list(config.value_range_config.keys())
                    print(f"Feature names: {feature_names}")
                    
                    for i in range(len(feature_names)):
                        feature_name = feature_names[i]
                        log_file_path = f"{config.log_dir}/{feature_name}.csv"
                        # Create header
                        with open(log_file_path, 'w') as f:
                            header = "idx,observed,predicted,lower_bound,upper_bound,is_anomalous,error,threshold\n"
                            f.write(header)
                        
                        self.feature_log_files.append(log_file_path)
                    
                    print(f"Created {len(self.feature_log_files)} feature log files")
                
                # Log data for each feature
                for i in range(len(self.feature_log_files)):
                    # Make sure we have enough data
                    if i < len(current_observation) and i < predicted[0, -1, :].shape[0]:
                        # Open log file for appending
                        with open(self.feature_log_files[i], 'a') as f:
                            # Get normalized values
                            observed_val_norm = current_observation[i]
                            predicted_val_norm = predicted[0, -1, i]
                            
                            # Denormalize values
                            observed_val = self.__reverse_normalized_data(observed_val_norm, i)
                            predicted_val = self.__reverse_normalized_data(predicted_val_norm, i)
                            
                            # Calculate denormalized bounds
                            # First get the feature's min/max values
                            feature = feature_columns[i]
                            min_val, max_val = value_range_config[feature]
                            
                            # Convert threshold to denormalized scale for this feature
                            denorm_threshold = threshold * (max_val - min_val)
                            
                            # Calculate bounds in denormalized space
                            lower_bound = max(min_val, predicted_val - denorm_threshold)
                            upper_bound = min(max_val, predicted_val + denorm_threshold)
                            
                            # Create log entry
                            text2write = f"{current_idx},{observed_val},{predicted_val},{lower_bound},{upper_bound},{reconstruction_error > threshold},{reconstruction_error},{threshold}\n"
                            f.write(text2write)
                            print(f"Logged data for feature {i}")
                
                # Also create a main log file for overall system metrics
                if not hasattr(self, 'main_log_file'):
                    self.main_log_file = f"{config.log_dir}/system.csv"
                    # Create header
                    with open(self.main_log_file, 'w') as f:
                        header = "idx,is_anomalous,error,threshold\n"
                        f.write(header)
                
                # Log overall system metrics
                with open(self.main_log_file, 'a') as f:
                    text2write = f"{current_idx},{reconstruction_error > threshold},{reconstruction_error},{threshold}\n"
                    f.write(text2write)
                
                # Print debug information
                print(f"Logged data for idx={current_idx}, anomaly={reconstruction_error > threshold}, error={reconstruction_error:.6f}, threshold={threshold:.6f}")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in logging: {e}")
            print(f"current_observation: {current_observation}")
            print(f"predicted: {predicted}")

    def is_anomalous(self, observed_val):
        is_anomalous_ret = False
        
        # Convert to numpy array and check for empty/NaN values
        observed_val = np.array([float(x) if x != '' else np.nan for x in observed_val])
        
        # Check for empty or NaN values
        if np.any(np.isnan(observed_val)):
            self.anomalies.append(self.observed_vals.get_length())
            self.__logging(is_anomalous_ret) 
            return True
            
        # Normalize data
        normalized_val = self.__normalize_data(observed_val)
        self.observed_vals.append(normalized_val)
        supposed_anomalous_pos = self.observed_vals.get_length()
        
        # prepare data for prediction
        past_observations = self.observed_vals.get_tail(self.predictor_config['lookback_len'])
        if len(past_observations) < self.predictor_config['lookback_len']:
            return False  # Not enough data for prediction
            
        past_observations = np.array(past_observations)
        past_observations_tensor = torch.Tensor(past_observations.reshape(1, -1))
        
        # predict normal value
        predicted_val = self.data_predictor.predict(past_observations_tensor)
        self.predicted_vals.append(predicted_val)
        
        # perform range check on individual features
        if not all(self.is_inside_range(x, i) for i, x in enumerate(normalized_val)):
            self.anomalies.append(supposed_anomalous_pos)
            is_anomalous_ret = True
        else:
            # calculate reconstruction error using the new function
            reconstruction_error = self.calculate_reconstruction_error(predicted_val, normalized_val)
            self.predictive_errors.append(reconstruction_error)
            
            # generate threshold only if we have enough historical errors
            if self.predictive_errors.get_length() >= self.predictor_config['lookback_len']:
                past_predictive_errors = self.predictive_errors.get_tail(self.predictor_config['lookback_len'])
                
                # Apply log transformation to handle very small error values
                log_transformed_errors = self.log_transform_errors(past_predictive_errors)
                
                # Create properly shaped input tensor for the generator
                log_errors_array = log_transformed_errors.reshape(1, -1)
                log_errors_tensor = torch.Tensor(log_errors_array)
                
                with torch.no_grad():
                    # Get threshold in log space
                    log_threshold_tensor = self.generator(log_errors_tensor)
                    log_threshold = log_threshold_tensor[0, 0].item()
                    
                    # The critical fix - convert the log threshold back to the original scale
                    threshold = 10 ** log_threshold
                    
                    print(f"Log threshold: {log_threshold:.4f}, Converted threshold: {threshold:.8f}")
                    
                    # Apply bounds
                    threshold = max(threshold, self.minimal_threshold)
                    threshold = min(threshold, reconstruction_error * 10.0)
                
                self.thresholds.append(threshold)
                
                # Check if error exceeds threshold
                if reconstruction_error > threshold:
                    is_anomalous_ret = True
                    self.anomalies.append(supposed_anomalous_pos)
                    
                # Update models
                self.data_predictor.update(
                    config.epoch_update,
                    config.lr_update,
                    past_observations_tensor,
                    normalized_val
                )
                
                if is_anomalous_ret or threshold > self.minimal_threshold:
                    # Transform the current error using the same log transform
                    log_current_error = self.log_transform_errors([reconstruction_error])[0]
                    
                    # Update the generator with log-transformed errors
                    self.generator.update(
                        config.update_G_epoch,
                        config.update_G_lr,
                        log_errors_tensor,
                        torch.tensor([[log_current_error]], dtype=torch.float32)
                    )
        
        self.__logging(is_anomalous_ret)
        return is_anomalous_ret

    def clean(self):
        # Clean prediction and error databases
        self.predicted_vals.clean(self.predictor_config['lookback_len'])
        self.predictive_errors.clean(self.predictor_config['lookback_len'])
        self.thresholds.clean(self.predictor_config['lookback_len'])
        
        # Don't close log files here - they need to stay open for the duration of the program
        # File closing should be handled in a separate method or at program exit

    def close_logs(self):
        """Close all log files"""
        for log_file in self.log_files.values():
            log_file.close()

    def is_inside_range(self, val, feature_idx=0):
        """
        Check if a value is within the normal operating range.
        
        Args:
            val: The value to check (normalized)
            feature_idx: Index of the feature to check (default: 0)
            
        Returns:
            bool: True if the value is within range, False otherwise
        """
        observed_val = self.__reverse_normalized_data(val, feature_idx)
        if observed_val >= self.sensor_range.lower(feature_idx) and observed_val <= self.sensor_range.upper(feature_idx):
            return True
        else:
            return False

    def log_transform_errors(self, errors, epsilon=1e-15):
        """
        Apply log transformation to error values to handle wide dynamic range.
        
        Args:
            errors: Array of error values
            epsilon: Small constant to avoid log(0)
            
        Returns:
            Log-transformed errors
        """
        errors_array = np.array(errors) + epsilon
        log_errors = np.log10(errors_array)
        # Debug output to verify transformation
        print(f"Error range: [{np.min(errors_array):.8f}, {np.max(errors_array):.8f}] → Log range: [{np.min(log_errors):.4f}, {np.max(log_errors):.4f}]")
        return log_errors

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
    AdapAD_obj = AdapAD(predictor_config, minimal_threshold)
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
            AdapAD_obj.train()
            print('------------STARTING TO MAKE DECISION------------')
        elif observed_data_sz > predictor_config['train_size']:
            is_anomalous_ret = AdapAD_obj.is_anomalous(measured_values)
            AdapAD_obj.clean()
        else:
            print('{}/{} to warmup training'.format(len(observed_data), predictor_config['train_size']))
        
            
    print('Done! Check results at results/LSTM-AE/')
    AdapAD_obj.close_logs()  # Close log files properly