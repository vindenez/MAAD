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
        self.num_features = len(feature_columns)
        
        # Initialize minimal thresholds for each parameter
        self.minimal_thresholds = [
            config.parameter_config[feature]["minimal_threshold"]
            for feature in feature_columns
        ]
        
        self.sensor_range = NormalValueRangeDb()
        
        # Add print statements to debug initialization
        print("\nInitializing predictor with parameters:")
        print(f"LSTM units: {[config.LSTM_l1_size, config.LSTM_l2_size, config.LSTM_bottleneck_size]}")
        print(f"Lookback length: {self.predictor_config['lookback_len']}")
        print(f"Prediction length: {self.predictor_config['prediction_len']}")
        print(f"Number of features: {self.num_features}")
        
        self.data_predictor = MultivariateNormalDataPredictor(
            lstm_unit=[config.LSTM_l1_size, config.LSTM_l2_size, config.LSTM_bottleneck_size],
            lookback_len=self.predictor_config['lookback_len'],
            prediction_len=self.predictor_config['prediction_len'],
            num_features=self.num_features
        )
        
        self.generator = LSTMPredictor(
            self.predictor_config['prediction_len'], 
            self.predictor_config['lookback_len'], 
            config.LSTM_size,
            config.LSTM_layer,
            self.predictor_config['lookback_len']
        ) 
        
        self.predicted_vals = PredictedNormalDataDb()
        self.thresholds = [AnomalousThresholdDb() for _ in range(self.num_features)]
        for i, threshold in enumerate(self.minimal_thresholds):
            self.thresholds[i].append(threshold)
        self.anomalies = []
        
        self.predictive_errors = None

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
            
        # Train Predictor
        trainX, trainY = self.data_predictor.train(
            config.epoch_train,
            config.lr_train,
            self.training_data
        )
        
        # Calculate initial errors using MSE
        predicted_values = []
        errors = []
        
        trainY_np = trainY.detach().numpy() if torch.is_tensor(trainY) else trainY
        
        for i in range(len(trainX)):
            train_predicted_val = self.data_predictor.predict(torch.reshape(trainX[i], (1, -1)))
            pred_np = train_predicted_val[0, -1, :] if isinstance(train_predicted_val, np.ndarray) else train_predicted_val[0, -1, :].detach().numpy()
            target_np = trainY_np[i, -1, :]
            
            # Use MSE instead of MAE
            error = np.mean((pred_np - target_np) ** 2)
            errors.append(error)
        
        self.predictive_errors = PredictionErrorDb(errors)
        original_errors = self.predictive_errors.get_tail(self.predictive_errors.get_length())
        
        # Train generator with MSE errors
        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=config.lr_train)
        
        # Slide data 
        x, y = sliding_windows(original_errors, self.predictor_config['lookback_len'], self.predictor_config['prediction_len'])
        x, y = x.astype(float), y.astype(float)
        y = np.reshape(y, (y.shape[0], y.shape[1]))
        
        # Train generator
        self.generator.train()
        for epoch in range(config.epoch_train):
            for i in range(len(x)):            
                optimizer.zero_grad()
                
                _x, _y = x[i], y[i]
                outputs = self.generator(torch.Tensor(_x).reshape((1,-1)))
                loss = criterion(outputs, torch.Tensor(_y).reshape((1,-1)))
                loss.backward(retain_graph=True)
                optimizer.step()

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
    
    def calculate_reconstruction_error(self, predicted, observed):
        if isinstance(predicted, np.ndarray) and len(predicted.shape) == 3:
            predicted_for_comparison = predicted[0, -1, :]
        else:
            predicted_for_comparison = predicted
        
        observed_np = np.array(observed)
        # Simple MSE calculation
        errors = predicted_for_comparison - observed_np
        reconstruction_error = np.mean(errors ** 2)
        
        # Store raw error for logging
        self.current_errors = np.abs(errors)
        
        return reconstruction_error

    def get_feature_weights(self):
        """Helper method to monitor how weights are being assigned to features"""
        if hasattr(self, 'current_weights'):
            return dict(zip(feature_columns, self.current_weights))
        return None

    def __logging(self, is_anomalous_ret):
        try:
            # Ensure log directory exists
            if not os.path.exists(config.log_dir):
                os.makedirs(config.log_dir)
            
            current_idx = self.observed_vals.get_length() - 1
            
            if current_idx >= self.predictor_config['lookback_len']:
                # Get the past observations for prediction
                past_observations = self.observed_vals.get_tail(self.predictor_config['lookback_len'])
                past_observations = np.array(past_observations)
                past_observations_tensor = torch.Tensor(past_observations.reshape(1, -1))
                
                # Get the predicted value
                predicted = self.data_predictor.predict(past_observations_tensor)
                
                # Get the current observation
                current_observation = self.observed_vals.get_tail(1)
                
                reconstruction_error = self.calculate_reconstruction_error(
                    predicted,
                    np.array(current_observation)
                )
                
                # Get the current thresholds (if available)
                thresholds = [
                    thresh.get_tail(1) if thresh.get_length() > 0 else self.minimal_thresholds[i]
                    for i, thresh in enumerate(self.thresholds)
                ]
                
                if not hasattr(self, 'feature_log_files'):
                    self.feature_log_files = []
                    
                    # Get feature names from config.value_range_config
                    feature_names = list(config.value_range_config.keys())
                    
                    for i in range(len(feature_names)):
                        feature_name = feature_names[i]
                        log_file_path = f"{config.log_dir}/{feature_name}_experiment_log.csv"
                        with open(log_file_path, 'w') as f:
                            header = "idx,observed,predicted,lower_bound,upper_bound,is_anomalous,error,threshold\n"
                            f.write(header)
                        
                        self.feature_log_files.append(log_file_path)
                    
                for i in range(len(self.feature_log_files)):
                    predicted_for_logging = predicted[0, -1, :] if len(predicted.shape) == 3 else predicted
                    if i < len(current_observation) and i < len(predicted_for_logging):
                        with open(self.feature_log_files[i], 'a') as f:
                            # Get normalized values
                            observed_val_norm = current_observation[i]
                            predicted_val_norm = predicted_for_logging[i]
                            
                            # Calculate current error
                            current_error = self.current_errors[i]
                            # Determine if this individual error exceeds threshold
                            is_anomalous = current_error > thresholds[i]
                            
                            # Calculate bounds in normalized space
                            lower_bound_norm = max(0, predicted_val_norm - thresholds[i])
                            upper_bound_norm = min(1, predicted_val_norm + thresholds[i])
                            
                            # Denormalize all values
                            observed_val = self.__reverse_normalized_data(observed_val_norm, i)
                            predicted_val = self.__reverse_normalized_data(predicted_val_norm, i)
                            lower_bound = self.__reverse_normalized_data(lower_bound_norm, i)
                            upper_bound = self.__reverse_normalized_data(upper_bound_norm, i)
                            
                            text2write = f"{current_idx},{observed_val},{predicted_val},{lower_bound},{upper_bound},"
                            text2write += f"{is_anomalous},{current_error:.6f},{thresholds[i]:.6f}\n"
                            f.write(text2write)
                
                if not hasattr(self, 'main_log_file'):
                    self.main_log_file = f"{config.log_dir}/system_experiment_log.csv"
                    with open(self.main_log_file, 'w') as f:
                        header = "idx,is_anomalous,error,threshold\n"
                        f.write(header)
                
                # Log overall system metrics
                text2write = f"{current_idx},{reconstruction_error > thresholds[0]},{reconstruction_error:.6f},{thresholds[0]:.6f}\n"
                with open(self.main_log_file, 'a') as f:
                    f.write(text2write)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in logging: {e}")

    def is_anomalous(self, observed_val):
        is_anomalous_ret = False
        
        # Convert to numpy array and check for empty/NaN values
        observed_val = np.array([float(x) if x != '' else np.nan for x in observed_val])
        
        # Check for empty or NaN values
        if np.any(np.isnan(observed_val)):
            print(f"NaN values detected in observation: {observed_val}")
            self.anomalies.append(self.observed_vals.get_length())
            self.__logging(is_anomalous_ret) 
            return True
            
        # Normalize data
        normalized_val = self.__normalize_data(observed_val)
        
        # Get past observations BEFORE adding current value (t-lookback to t-1)
        past_observations = self.observed_vals.get_tail(self.predictor_config['lookback_len'])
        if len(past_observations) < self.predictor_config['lookback_len']:
            print(f"Not enough past observations. Have {len(past_observations)}, need {self.predictor_config['lookback_len']}")
            self.observed_vals.append(normalized_val)  
            return False  
            
        # Make prediction for current timestep using past data
        past_observations = np.array(past_observations)
        past_observations_tensor = torch.Tensor(past_observations.reshape(1, -1))
        
        # Get prediction for current timestep t
        predicted_val = self.data_predictor.predict(past_observations_tensor)
        
        # Calculate reconstruction error for timestep t
        reconstruction_error = self.calculate_reconstruction_error(predicted_val, normalized_val)
        
        # Now add the current value and prediction to our history
        self.observed_vals.append(normalized_val)
        supposed_anomalous_pos = self.observed_vals.get_length()
        self.predicted_vals.append(predicted_val)
        self.predictive_errors.append(reconstruction_error)
        
        # Generate threshold using errors up to t-1 (don't include current error)
        if self.predictive_errors.get_length() > self.predictor_config['lookback_len']:
            self.generator.eval()
            # Get errors excluding the most recent one
            past_predictive_errors = self.predictive_errors.get_tail(self.predictor_config['lookback_len'] + 1)[:-1]
            
            past_errors_tensor = torch.from_numpy(np.array(past_predictive_errors).reshape(1, -1)).float()
            
            with torch.no_grad():
                thresholds = self.generator(past_errors_tensor)
                thresholds = thresholds.data.numpy()
                # Cap the thresholds to avoid extreme values, using per-parameter minimal thresholds
                thresholds = np.clip(
                    thresholds.flatten()[:self.num_features],
                    self.minimal_thresholds,
                    0.2
                )

            for i in range(self.num_features):
                self.thresholds[i].append(thresholds[i])
        else:
            thresholds = self.minimal_thresholds

        # Check for anomalies
        if not all(self.is_inside_range(val, i) for i, val in enumerate(normalized_val)):
            self.anomalies.append(supposed_anomalous_pos)
            is_anomalous_ret = True
        else:
            # Check if any error exceeds its corresponding threshold
            for i, error in enumerate(self.current_errors):
                if error > thresholds[i]:
                    is_anomalous_ret = True
                    self.anomalies.append(supposed_anomalous_pos)
                    break
        
        # Always update predictor
        self.data_predictor.update(
            config.epoch_update,
            config.lr_update,
            past_observations_tensor,
            normalized_val
        )
        
        # Update generator if anomaly detected or thresholds are above minimal
        if is_anomalous_ret or np.any(thresholds > self.minimal_thresholds):
            self.__update_generator(past_errors_tensor, reconstruction_error)
        
        self.__logging(is_anomalous_ret)
        return is_anomalous_ret

    def __update_generator(self, past_predictive_errors, prediction_error):
        self.generator.train()
        
        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=config.update_G_lr)
        
        loss_l = list()
        for epoch in range(config.update_G_epoch):
            predicted_val = self.generator(past_predictive_errors.float())
            optimizer.zero_grad()
            
            target = torch.tensor([[prediction_error]], dtype=torch.float32)
            
            loss = criterion(predicted_val, target)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # for early stopping
            if len(loss_l) > 1 and loss.item() > loss_l[-1]:
                break
            loss_l.append(loss.item())

    def clean(self):
        self.predicted_vals.clean(self.predictor_config['lookback_len'])
        self.predictive_errors.clean(self.predictor_config['lookback_len'])
        for threshold in self.thresholds:
            threshold.clean(self.predictor_config['lookback_len'])
        
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
    
    print("Available columns in dataset (after cleaning):", data_source.columns.tolist())
    
    feature_columns = config.feature_columns

    missing_columns = [col for col in feature_columns if col not in data_source.columns]
    if missing_columns:
        print(f"Warning: The following configured columns are missing from the dataset: {missing_columns}")
        print("Using only available columns...")
        feature_columns = [col for col in feature_columns if col in data_source.columns]
        
    if not feature_columns:
        raise Exception("No valid feature columns found in the dataset. Please check your configuration.")
    
    print(f"\nUsing features: {feature_columns}")
    print(f"Total data points: {len(data_source)}")
    
    # Extract data from dataframe
    # Convert all data to numeric values, coercing errors to NaN
    for col in feature_columns:
        data_source[col] = pd.to_numeric(data_source[col], errors='coerce')
    
    data_values = data_source[feature_columns].values
    len_data_subject = len(data_values)
    
    # AdapAD
    AdapAD_obj = AdapAD(predictor_config, minimal_threshold)
    print('\nGATHERING DATA FOR TRAINING...', predictor_config['train_size'])
    
    observed_data = []
    processed_count = 0
    
    for data_idx in range(len_data_subject):
        measured_values = data_values[data_idx]
        timestamp = str(data_idx)
        
        observed_data.append(measured_values)
        observed_data_sz = len(observed_data)
        
        # perform warmup training or make a decision
        if observed_data_sz == predictor_config['train_size']:
            print(f"{observed_data_sz}/{predictor_config['train_size']} - Starting training...")
            AdapAD_obj.set_training_data(np.array(observed_data))
            AdapAD_obj.train()
            print('\n------------STARTING TO MAKE DECISION------------')
        elif observed_data_sz > predictor_config['train_size']:
            is_anomalous_ret = AdapAD_obj.is_anomalous(measured_values)
            AdapAD_obj.clean()
            processed_count += 1
            
            # Print progress every 1000 samples
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count} samples...")
        else:
            # Print every warmup step
            print(f'{observed_data_sz}/{predictor_config["train_size"]} to warmup training')
    
    print(f'\nDone! Processed {processed_count} samples after training')
    print(f'Check results at {config.log_dir}/')
    AdapAD_obj.close_logs()  # Close log files properly