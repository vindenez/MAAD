import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from utils import *
from learning_models import *
from supporting_components import *
import config as config
from config import value_range_config

torch.manual_seed(0)

class AdapAD:
    def __init__(self, predictor_config, system_threshold):
        self.predictor_config = predictor_config
        self.system_threshold = system_threshold  
        self.num_features = len(feature_columns)
        
        self.sensor_range = NormalValueRangeDb()
        
        # Initialize parameter-specific thresholds from config
        self.parameter_thresholds = {
            param: config.parameter_config[param]['minimal_threshold']
            for param in feature_columns
        }
        
        # Initialize main log file path
        self.main_log_file = f"{config.log_dir}/system_log.csv"
        
        # Initialize predictor
        self.data_predictor = MultivariateTimeSeriesPredictor(
            num_features=self.num_features,
            hidden_size=config.LSTM_size,
            num_layers=config.LSTM_size_layer,
            lookback_len=self.predictor_config['lookback_len']
        )
        
        # Initialize threshold generator
        self.generator = AnomalousThresholdGenerator(
            lstm_layer=config.LSTM_size_layer,
            lstm_unit=config.LSTM_size,
            lookback_len=self.predictor_config['lookback_len'],
            prediction_len=self.predictor_config['prediction_len']
        )
        
        self.predicted_vals = PredictedNormalDataDb()
        self.thresholds = AnomalousThresholdDb()
        self.thresholds.append(self.system_threshold)
        self.anomalies = []

        # Create log directory if it doesn't exist
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)

        # Initialize main system log file
        if not os.path.exists(self.main_log_file):
            with open(self.main_log_file, 'w') as f:
                f.write("idx,is_anomalous,error,threshold\n")

        # Initialize feature-specific log files
        for feature in feature_columns:
            log_file_path = f"{config.log_dir}/{feature}_log.csv"
            if not os.path.exists(log_file_path):
                with open(log_file_path, 'w') as f:
                    f.write("idx,observed,predicted,lower_bound,upper_bound,is_anomalous,error,threshold\n")

    def __normalize_data(self, data):
        normalized_data = np.zeros_like(data)
        for i, value in enumerate(data):
            feature = feature_columns[i]
            min_val, max_val = value_range_config[feature]
            normalized_data[i] = (value - min_val) / (max_val - min_val)
            normalized_data[i] = np.clip(normalized_data[i], 0, 1)
        return normalized_data

    def __reverse_normalized_data(self, normalized_val, feature_idx):
        feature = feature_columns[feature_idx]
        min_val, max_val = value_range_config[feature]
        return normalized_val * (max_val - min_val) + min_val

    def set_training_data(self, training_data):
        # Normalize the training data
        normalized_training_data = np.array([self.__normalize_data(row) for row in training_data])
        
        # Store the normalized training data
        self.observed_vals = DataSubject(normalized_training_data)
        
        # Store the training data for later use
        self.training_data = normalized_training_data

    def train(self):
        # Train Predictor with multivariate data
        trainX, trainY = self.data_predictor.train(
            config.epoch_train,
            config.lr_train,
            self.training_data
        )
        
        errors = []
        
        # Process model outputs
        trainY_np = trainY.detach().numpy() if torch.is_tensor(trainY) else trainY
        
        # Calculate prediction errors from training data
        for i in range(len(trainX)):
            input_tensor = torch.Tensor(trainX[i]).unsqueeze(0)  
            train_predicted_val = self.data_predictor.predict(input_tensor)
            
            # Use the first feature as the target for error calculation
            target_np = trainY_np[i]
            
            # Calculate MSE
            error = np.mean((train_predicted_val - target_np) ** 2)
            errors.append(error)
        
        self.predictive_errors = PredictionErrorDb(errors)
        original_errors = self.predictive_errors.get_tail(self.predictive_errors.get_length())
        
        # Train threshold generator
        self.generator.train(config.epoch_train, config.lr_train, original_errors)
        
    def is_anomalous(self, observed_val):
        # Convert input to numpy array and handle missing values
        observed_val = np.array([float(x) if x != '' else np.nan for x in observed_val])
        
        if np.any(np.isnan(observed_val)):
            self.anomalies.append(self.observed_vals.get_length())
            self.observed_vals.append(np.zeros_like(observed_val))  
            return True
        
        # Check if any value is outside its sensor range
        for i, val in enumerate(observed_val):
            if not self.is_inside_range(self.__normalize_data(observed_val)[i], i):
                self.anomalies.append(self.observed_vals.get_length())
                self.observed_vals.append(np.zeros_like(observed_val))
                return True
        
        # Normalize using config ranges
        normalized_val = self.__normalize_data(observed_val)
        
        # Get lookback window of past observations
        past_observations = self.observed_vals.get_tail(self.predictor_config['lookback_len'])
        
        if len(past_observations) < self.predictor_config['lookback_len']:
            self.observed_vals.append(normalized_val)
            return False
        
        past_observations = np.array(past_observations)
        for t in range(len(past_observations)):
            denorm_values = []
            for i in range(len(feature_columns)):
                denorm_val = self.__reverse_normalized_data(past_observations[t][i], i)
                denorm_values.append(denorm_val)
        
        # Prepare past observations for prediction
        past_observations_tensor = torch.Tensor(past_observations).unsqueeze(0)
        
        # Make prediction
        predicted_val = self.data_predictor.predict(past_observations_tensor)
        if isinstance(predicted_val, torch.Tensor):
            predicted_val = predicted_val.detach().numpy()
        if len(predicted_val.shape) == 2:
            predicted_val = predicted_val[0]
        
        # Calculate errors and check individual parameter thresholds
        errors = predicted_val - normalized_val
        self.current_errors = np.abs(errors)
        mean_squared_error = np.mean(errors ** 2)  
        
        # Check if any parameter exceeds its specific threshold
        is_anomalous_ret = False
        for i, feature in enumerate(feature_columns):
            if self.current_errors[i] > self.parameter_thresholds[feature]:
                is_anomalous_ret = True
                break
        
        # If no individual parameter is anomalous, check system-wide MSE
        if not is_anomalous_ret:
            is_anomalous_ret = mean_squared_error > self.system_threshold
        
        # Get threshold for logging 
        if self.predictive_errors and self.predictive_errors.get_length() >= self.predictor_config['lookback_len']:
            past_errors = np.array(self.predictive_errors.get_tail(self.predictor_config['lookback_len']))
            past_errors_tensor = torch.Tensor(past_errors).reshape(1, -1)
            threshold = self.generator.generate(past_errors_tensor, self.system_threshold)
            threshold = max(threshold, self.system_threshold)
            self.thresholds.append(threshold)
        else:
            threshold = self.system_threshold
        
        # Logging
        self.__logging(is_anomalous_ret, normalized_val, predicted_val, threshold, mean_squared_error)
        self.observed_vals.append(normalized_val)
        self.predicted_vals.append(predicted_val)
        self.predictive_errors.append(mean_squared_error)
        
        # Update models
        self.data_predictor.update(
            config.epoch_update,
            config.lr_update,
            past_observations_tensor,
            normalized_val
        )
        
        if threshold > self.system_threshold:
            self.generator.update(
                config.update_G_epoch,
                config.update_G_lr,
                past_errors_tensor,
                mean_squared_error
            )
        
        if is_anomalous_ret:
            self.anomalies.append(self.observed_vals.get_length())
        
        return is_anomalous_ret

    def __logging(self, is_anomalous_ret, normalized_val, predicted_val, threshold, mean_squared_error):
        try:
            current_idx = self.observed_vals.get_length() - 1
            
            # Log for each feature
            for i, feature in enumerate(feature_columns):
                log_file_path = f"{config.log_dir}/{feature}_log.csv"
                
                # Append data
                with open(log_file_path, 'a') as f:
                    # First denormalize the observed and predicted values
                    observed_val = self.__reverse_normalized_data(normalized_val[i], i)
                    predicted_val_denorm = self.__reverse_normalized_data(predicted_val[i], i)
                    
                    # Use parameter-specific threshold for individual features
                    parameter_threshold = self.parameter_thresholds[feature]
                    
                    # Calculate bounds using parameter-specific threshold
                    lower_bound_norm = predicted_val[i] - parameter_threshold
                    upper_bound_norm = predicted_val[i] + parameter_threshold
                    
                    # Then denormalize the bounds
                    lower_bound = self.__reverse_normalized_data(lower_bound_norm, i)
                    upper_bound = self.__reverse_normalized_data(upper_bound_norm, i)
                    
                    text2write = f"{current_idx},{observed_val},{predicted_val_denorm},{lower_bound},{upper_bound},"
                    text2write += f"{self.current_errors[i] > parameter_threshold},{self.current_errors[i]:.6f},{parameter_threshold:.6f}\n"
                    f.write(text2write)
            
            # Log system metrics (using system-wide threshold for MSE)
            if not os.path.exists(self.main_log_file):
                with open(self.main_log_file, 'w') as f:
                    f.write("idx,is_anomalous,error,threshold\n")
                
            with open(self.main_log_file, 'a') as f:
                f.write(f"{current_idx},{is_anomalous_ret},{mean_squared_error:.6f},{self.system_threshold:.6f}\n")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in logging: {e}")

    def clean(self):
        self.predicted_vals.clean(self.predictor_config['lookback_len'])
        self.predictive_errors.clean(self.predictor_config['lookback_len'])
        self.thresholds.clean(self.predictor_config['lookback_len'])
        
    def is_inside_range(self, val, feature_idx=0):
        observed_val = self.__reverse_normalized_data(val, feature_idx)
        if observed_val >= self.sensor_range.lower(feature_idx) and observed_val <= self.sensor_range.upper(feature_idx):
            return True
        else:
            return False

if __name__ == "__main__":

    feature_columns = list(value_range_config.keys())

    value_range_db = NormalValueRangeDb()

    predictor_config, value_range_config, minimal_threshold = config.init_config()
    if not minimal_threshold:
        raise Exception("It is mandatory to set a minimal threshold")
    
    data_source = pd.read_csv(config.data_source_path)
    data_source.columns = [col.strip() for col in data_source.columns]
    
    # Create log directory if it doesn't exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    
    feature_columns = config.feature_columns
    missing_columns = [col for col in feature_columns if col not in data_source.columns]
    
    if missing_columns:
        feature_columns = [col for col in feature_columns if col in data_source.columns]
    
    if not feature_columns:
        raise Exception("No valid feature columns found in the dataset. Please check your configuration.")
    
    # Extract data from dataframe
    for col in feature_columns:
        data_source[col] = pd.to_numeric(data_source[col], errors='coerce')
    
    data_values = data_source[feature_columns].values
    len_data_subject = len(data_values)
    
    # AdapAD
    AdapAD_obj = AdapAD(predictor_config, minimal_threshold)
    
    observed_data = []
    
    for data_idx in range(len_data_subject):
        measured_values = data_values[data_idx]
        observed_data.append(measured_values)
        observed_data_sz = len(observed_data)
        
        if observed_data_sz == predictor_config['train_size']:
            AdapAD_obj.set_training_data(np.array(observed_data))
            AdapAD_obj.train()
        elif observed_data_sz > predictor_config['train_size']:
            is_anomalous_ret = AdapAD_obj.is_anomalous(measured_values)
            AdapAD_obj.clean()
        
    AdapAD_obj.close_logs()