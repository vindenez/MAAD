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
        
# Extract the uncommented columns
feature_columns = list(value_range_config.keys())

# Initialize the value range database
value_range_db = NormalValueRangeDb()

class AdapAD:
    def __init__(self, predictor_config, minimal_threshold):
        self.predictor_config = predictor_config
        self.minimal_threshold = minimal_threshold
        self.num_features = len(feature_columns)
        
        self.sensor_range = NormalValueRangeDb()
        
        # Clear existing log files
        if os.path.exists(config.log_dir):
            # Remove all existing log files
            for feature in feature_columns:
                log_file_path = f"{config.log_dir}/{feature}_experiment_log.csv"
                if os.path.exists(log_file_path):
                    os.remove(log_file_path)
            # Remove system log file
            system_log_path = f"{config.log_dir}/system_experiment_log.csv"
            if os.path.exists(system_log_path):
                os.remove(system_log_path)
        else:
            os.makedirs(config.log_dir)
        
        # Initialize main log file path
        self.main_log_file = f"{config.log_dir}/system_experiment_log.csv"
        
        # Use the new MultivariateTimeSeriesPredictor with correct parameters
        self.data_predictor = MultivariateTimeSeriesPredictor(
            num_features=self.num_features,
            hidden_size=config.LSTM_size,
            num_layers=config.LSTM_size_layer,
            lookback_len=self.predictor_config['lookback_len']
            # Remove prediction_len as it's not needed - we only predict next timestep
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
        self.thresholds.append(self.minimal_threshold)
        self.anomalies = []
        
        self.predictive_errors = None
        self.window_size = predictor_config['lookback_len'] * 2  # Adjust window size as needed
        self.running_stats = {}  # Store running min/max for each feature

    def __initialize_running_stats(self, feature_idx):
        """Initialize running statistics for a feature"""
        self.running_stats[feature_idx] = {
            'min': float('inf'),
            'max': float('-inf'),
            'values': []
        }

    def __update_running_stats(self, value, feature_idx):
        """Update running min/max statistics for dynamic normalization"""
        if feature_idx not in self.running_stats:
            self.__initialize_running_stats(feature_idx)
            
        stats = self.running_stats[feature_idx]
        stats['values'].append(value)
        
        # Keep only recent values
        if len(stats['values']) > self.window_size:
            stats['values'].pop(0)
            
        # Update min/max based on recent window
        stats['min'] = min(stats['values'])
        stats['max'] = max(stats['values'])
        
        # Ensure we have some range to normalize
        if stats['min'] == stats['max']:
            stats['min'] -= 0.1
            stats['max'] += 0.1

    def __normalize_data(self, data):
        """Normalize data using static ranges from config"""
        normalized_data = np.zeros_like(data)
        for i, value in enumerate(data):
            feature = feature_columns[i]
            min_val, max_val = value_range_config[feature]
            normalized_data[i] = (value - min_val) / (max_val - min_val)
            normalized_data[i] = np.clip(normalized_data[i], 0, 1)
        return normalized_data

    def __reverse_normalized_data(self, normalized_val, feature_idx):
        """Reverse normalization using static ranges from config"""
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
        if not hasattr(self, 'training_data'):
            raise ValueError("Training data not set. Call set_training_data first.")
            
        # Train Predictor with multivariate data
        trainX, trainY = self.data_predictor.train(
            config.epoch_train,
            config.lr_train,
            self.training_data
        )
        
        # Calculate initial errors using MSE
        predicted_values = []
        errors = []
        
        # Process model outputs
        trainY_np = trainY.detach().numpy() if torch.is_tensor(trainY) else trainY
        
        # Calculate prediction errors from training data
        for i in range(len(trainX)):
            input_tensor = torch.Tensor(trainX[i]).unsqueeze(0)  # Add batch dimension
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

    def calculate_reconstruction_error(self, predicted, observed):
        # Ensure both arrays are proper numpy arrays
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.detach().numpy()
            
        observed_np = np.array(observed)
        
        # Calculate errors for all features
        errors = predicted - observed_np
        
        # Calculate MSE for each feature
        self.current_errors = np.abs(errors)  # Store raw errors for logging
        reconstruction_error = np.mean(errors ** 2)  # Overall reconstruction error
        
        return reconstruction_error

    def get_feature_weights(self):
        """Helper method to monitor how weights are being assigned to features"""
        if hasattr(self, 'current_weights'):
            return dict(zip(feature_columns, self.current_weights))
        return None

    def is_anomalous(self, observed_val):
        """
        Detect anomalies in the current observation using past observations for prediction.
        """
        print("\n=== Debug: is_anomalous() ===")
        print("Current observation (all features):", observed_val)
        
        # Convert input to numpy array and handle missing values
        observed_val = np.array([float(x) if x != '' else np.nan for x in observed_val])
        
        if np.any(np.isnan(observed_val)):
            print("Found NaN values, marking as anomaly")
            self.anomalies.append(self.observed_vals.get_length())
            self.observed_vals.append(np.zeros_like(observed_val))  
            return True
        
        # Normalize using config ranges
        normalized_val = self.__normalize_data(observed_val)
        
        print("\nNormalized values for each feature:")
        for i, feature in enumerate(feature_columns):
            print(f"{feature}: {normalized_val[i]:.4f} (original: {observed_val[i]:.4f})")
        
        # Get lookback window of past observations
        past_observations = self.observed_vals.get_tail(self.predictor_config['lookback_len'])
        
        if len(past_observations) < self.predictor_config['lookback_len']:
            print("\nNot enough history, adding current observation and continuing")
            self.observed_vals.append(normalized_val)
            return False
        
        print("\nPast observations window (showing all features):")
        past_observations = np.array(past_observations)
        for t in range(len(past_observations)):
            denorm_values = []
            for i in range(len(feature_columns)):
                denorm_val = self.__reverse_normalized_data(past_observations[t][i], i)
                denorm_values.append(denorm_val)
            print(f"t-{len(past_observations)-t}: {denorm_values}")
        
        # Prepare past observations for prediction
        past_observations_tensor = torch.Tensor(past_observations).unsqueeze(0)
        print("\nInput tensor shape:", past_observations_tensor.shape)
        
        # Make prediction
        predicted_val = self.data_predictor.predict(past_observations_tensor)
        if isinstance(predicted_val, torch.Tensor):
            predicted_val = predicted_val.detach().numpy()
        if len(predicted_val.shape) == 2:
            predicted_val = predicted_val[0]
        
        print("\nPredictions vs Actuals:")
        for i, feature in enumerate(feature_columns):
            pred_denorm = self.__reverse_normalized_data(predicted_val[i], i)
            obs_denorm = observed_val[i]  # Original value is already denormalized
            print(f"{feature}:")
            print(f"  Predicted: {pred_denorm:.4f}")
            print(f"  Actual: {obs_denorm:.4f}")
        
        # Calculate errors
        errors = predicted_val - normalized_val
        self.current_errors = np.abs(errors)
        reconstruction_error = np.mean(errors ** 2)
        
        # Get threshold
        if self.predictive_errors and self.predictive_errors.get_length() >= self.predictor_config['lookback_len']:
            past_errors = np.array(self.predictive_errors.get_tail(self.predictor_config['lookback_len']))
            past_errors_tensor = torch.Tensor(past_errors).reshape(1, -1)
            threshold = self.generator.generate(past_errors_tensor, self.minimal_threshold)
            threshold = max(threshold, self.minimal_threshold)
            self.thresholds.append(threshold)
        else:
            threshold = self.minimal_threshold
        
        print(f"\nReconstruction error: {reconstruction_error:.6f}")
        print(f"Threshold: {threshold:.6f}")
        
        # Determine if anomalous
        is_anomalous_ret = reconstruction_error > threshold
        
        # Log and update
        self.__logging(is_anomalous_ret, normalized_val, predicted_val, threshold, reconstruction_error)
        self.observed_vals.append(normalized_val)
        self.predicted_vals.append(predicted_val)
        self.predictive_errors.append(reconstruction_error)
        
        # Update model
        self.data_predictor.update(
            config.epoch_update,
            config.lr_update,
            past_observations_tensor,
            normalized_val
        )
        
        if threshold > self.minimal_threshold:
            self.generator.update(
                config.update_G_epoch,
                config.update_G_lr,
                past_errors_tensor,
                reconstruction_error
            )
        
        if is_anomalous_ret:
            self.anomalies.append(self.observed_vals.get_length())
        
        return is_anomalous_ret

    def __logging(self, is_anomalous_ret, normalized_val, predicted_val, threshold, reconstruction_error):
        try:
            current_idx = self.observed_vals.get_length() - 1
            
            # Log for each feature
            for i, feature in enumerate(feature_columns):
                log_file_path = f"{config.log_dir}/{feature}_experiment_log.csv"
                
                # Write header if file is empty/new
                if not os.path.exists(log_file_path):
                    with open(log_file_path, 'w') as f:
                        f.write("idx,observed,predicted,lower_bound,upper_bound,is_anomalous,error,threshold\n")
                
                # Append data
                with open(log_file_path, 'a') as f:
                    # First denormalize the observed and predicted values
                    observed_val = self.__reverse_normalized_data(normalized_val[i], i)
                    predicted_val_denorm = self.__reverse_normalized_data(predicted_val[i], i)
                    
                    # Calculate bounds by adding/subtracting threshold in normalized space
                    lower_bound_norm = predicted_val[i] - threshold
                    upper_bound_norm = predicted_val[i] + threshold
                    
                    # Then denormalize the bounds
                    lower_bound = self.__reverse_normalized_data(lower_bound_norm, i)
                    upper_bound = self.__reverse_normalized_data(upper_bound_norm, i)
                    
                    text2write = f"{current_idx},{observed_val},{predicted_val_denorm},{lower_bound},{upper_bound},"
                    text2write += f"{self.current_errors[i] > threshold},{self.current_errors[i]:.6f},{threshold:.6f}\n"
                    f.write(text2write)
            
            # Log system metrics
            if not os.path.exists(self.main_log_file):
                with open(self.main_log_file, 'w') as f:
                    f.write("idx,is_anomalous,error,threshold\n")
                
            with open(self.main_log_file, 'a') as f:
                f.write(f"{current_idx},{is_anomalous_ret},{reconstruction_error:.6f},{threshold:.6f}\n")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in logging: {e}")

    def clean(self):
        self.predicted_vals.clean(self.predictor_config['lookback_len'])
        self.predictive_errors.clean(self.predictor_config['lookback_len'])
        self.thresholds.clean(self.predictor_config['lookback_len'])
        
    def close_logs(self):
        pass

    def is_inside_range(self, val, feature_idx=0):
        observed_val = self.__reverse_normalized_data(val, feature_idx)
        if observed_val >= self.sensor_range.lower(feature_idx) and observed_val <= self.sensor_range.upper(feature_idx):
            return True
        else:
            return False

    def log_transform_errors(self, errors, epsilon=1e-15):
        errors_array = np.array(errors) + epsilon
        log_errors = np.log10(errors_array)
        return log_errors

if __name__ == "__main__":
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