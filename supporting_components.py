import numpy as np
import config as config
        
class NormalValueRangeDb():
    def __init__(self):
        self.__value_range = {}
        
    def lower(self, feature_idx=0):
        if isinstance(feature_idx, int) and feature_idx in self.__value_range:
            return self.__value_range[feature_idx][0]
        return 0.0  # Default value if not found
        
    def upper(self, feature_idx=0):
        if isinstance(feature_idx, int) and feature_idx in self.__value_range:
            return self.__value_range[feature_idx][1]
        return 1.0  # Default value if not found
        
    def set(self, feature_idx, min_val, max_val):
        self.__value_range[feature_idx] = (min_val, max_val)
        
class PredictedNormalDataDb():
    def __init__(self):
        self.predicted_vals = list()
        
    def append(self, val):
        self.predicted_vals.append(val)
    
    def get_tail(self, length=1):
        if length == 1:
            return self.predicted_vals[-1]
        else:
            return self.predicted_vals[-length:]
            
    def get_length(self):
        return len(self.predicted_vals)
        
    def clean(self, len2keep):
        self.predicted_vals = self.predicted_vals[-len2keep:]
        
class AnomalousThresholdDb():
    def __init__(self):
        self.thresholds = list()
        
    def append(self, val):
        self.thresholds.append(val)
    
    def get_tail(self, length=1):
        if length == 1:
            return self.thresholds[-1]
        else:
            return self.thresholds[-length:]
            
    def get_length(self):
        return len(self.thresholds)
        
    def clean(self, len2keep):
        self.thresholds = self.thresholds[-len2keep:]
        
class PredictionErrorDb():
    def __init__(self, prediction_error_training):
        self.prediction_errors = []
        for err in prediction_error_training:
            if isinstance(err, (list, np.ndarray)):
                self.prediction_errors.append(float(np.mean(err)))
            else:
                self.prediction_errors.append(float(err))
        
    def append(self, val):
        if isinstance(val, (list, np.ndarray)):
            self.prediction_errors.append(float(np.mean(val)))
        else:
            self.prediction_errors.append(float(val))
    
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