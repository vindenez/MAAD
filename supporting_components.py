import numpy as np
import config as config

class NormalDataPredictionErrorCalculator():
    @staticmethod
    def calc_error(ground_truth, predict):
        return (ground_truth - predict)**2
        
class NormalValueRangeDb():
    def __init__(self):
        _, self.__value_range, _ = config.init_config()
        
    def lower(self, feature_idx=0):
        if isinstance(self.__value_range, tuple):
            return self.__value_range[0]
        else:
            return self.__value_range[feature_idx][0]
        
    def upper(self, feature_idx=0):
        if isinstance(self.__value_range, tuple):
            return self.__value_range[1]
        else:
            return self.__value_range[feature_idx][1]
        
class PredictedNormalDataDb():
    def __init__(self):
        self.predicted_vals = list()
        self.attention_weights = list()  # Store attention weights
        
    def append(self, val, attention_weights=None):
        self.predicted_vals.append(val)
        if attention_weights is not None:
            self.attention_weights.append(attention_weights)
    
    def get_tail(self, length=1):
        if length == 1:
            return self.predicted_vals[-1]
        else:
            return self.predicted_vals[-length:]
            
    def get_attention_weights(self, length=1):
        if not self.attention_weights:
            return None
        if length == 1:
            return self.attention_weights[-1] if self.attention_weights else None
        else:
            return self.attention_weights[-length:] if len(self.attention_weights) >= length else None
            
    def get_length(self):
        return len(self.predicted_vals)
        
    def clean(self, len2keep):
        self.predicted_vals = self.predicted_vals[-len2keep:]
        if self.attention_weights:
            self.attention_weights = self.attention_weights[-len2keep:]
        
class AnomalousThresholdDb():
    def __init__(self):
        self.thresholds = list()
        self.attention_weights = list()  # Store attention weights
        
    def append(self, val, attention_weights=None):
        self.thresholds.append(val)
        if attention_weights is not None:
            self.attention_weights.append(attention_weights)
    
    def get_tail(self, length=1):
        if length == 1:
            return self.thresholds[-1]
        else:
            return self.thresholds[-length:]
            
    def get_attention_weights(self, length=1):
        if not self.attention_weights:
            return None
        if length == 1:
            return self.attention_weights[-1] if self.attention_weights else None
        else:
            return self.attention_weights[-length:] if len(self.attention_weights) >= length else None
            
    def get_length(self):
        return len(self.thresholds)
        
    def clean(self, len2keep):
        self.thresholds = self.thresholds[-len2keep:]
        if self.attention_weights:
            self.attention_weights = self.attention_weights[-len2keep:]
        
class PredictionErrorDb():
    def __init__(self, prediction_error_training):
        self.prediction_errors = prediction_error_training.copy()
        
    def append(self, val):
        self.prediction_errors.append(val)
    
    def get_tail(self, length=1):
        if length == 1:
            return self.prediction_errors[-1]
        else:
            return self.prediction_errors[-length:]
    
    def get_length(self):
        return len(self.prediction_errors)
        
    def clean(self, len2keep):
        self.prediction_errors = self.prediction_errors[-len2keep:]
        
class DataSubject():
    def __init__(self, normal_data):
        # Convert numpy array to list of arrays if needed
        if isinstance(normal_data, np.ndarray):
            self.observed_vals = [row for row in normal_data]
        else:
            self.observed_vals = normal_data.copy()
        self.is_retrieved_training_data = False
    
    def get_tail(self, length=1):
        if length == 1:
            return self.observed_vals[-1]
        else:
            return self.observed_vals[-length:]
        
    def append(self, val):
        self.observed_vals.append(val)
        
    def get_training_data(self):
        if self.is_retrieved_training_data:
            raise Exception("Already retrieved training data! Check the flow...")
        
        self.is_retrieved_training_data = True
        return self.observed_vals
        
    def get_length(self):
        return len(self.observed_vals)