import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from utils import *

class LSTMPredictor(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, lookback_len):
        super(LSTMPredictor, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = lookback_len

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)

    def forward(self, x):     
        x, (h_out, _) = self.lstm(x)
        h_out = h_out.view(-1, self.hidden_size)   
        out = self.fc(h_out)
        return out


class AnomalousThresholdGenerator():
    def __init__(self, lstm_layer, lstm_unit, lookback_len, prediction_len):
        hidden_size = lstm_unit
        num_layers = lstm_layer
        self.lookback_len = lookback_len
        self.prediction_len = prediction_len
        self.generator = LSTMPredictor(prediction_len, 
                    lookback_len, 
                    hidden_size, 
                    num_layers, 
                    lookback_len)
                    
    def train(self, epoch, lr, data2learn):
        num_epochs = epoch
        learning_rate = lr

        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        loss_l = list()
        
        # slide data 
        x, y = sliding_windows(data2learn, self.lookback_len, self.prediction_len)
        x, y = x.astype(float), y.astype(float)
        y = np.reshape(y, (y.shape[0], y.shape[1]))
        
        # prepare data for training
        train_tensorX = torch.Tensor(np.array(x))
        train_tensorY = torch.Tensor(np.array(y))
        train_dataset = torch.utils.data.TensorDataset(train_tensorX, train_tensorY)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
        
        # training
        self.generator.train()
        for epoch in range(num_epochs):
            for i in range(len(x)):            
                optimizer.zero_grad()
                
                _x, _y = x[i], y[i]
                outputs = self.generator(torch.Tensor(_x).reshape((1,-1)))
                loss = criterion(outputs, torch.Tensor(_y).reshape((1,-1)))
                loss.backward(retain_graph=True)
                optimizer.step()
                
    def update(self, epoch_update, lr_update, past_errors, recent_error):
        num_epochs = epoch_update
        learning_rate = lr_update
        
        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        
        self.generator.train()
        loss_l = list()
        for epoch in range(num_epochs):
            predicted_val = self.generator(past_errors.float())
            optimizer.zero_grad()           
            loss = criterion(predicted_val, torch.from_numpy(np.array(recent_error).reshape(1, -1)).float())        
            loss.backward(retain_graph=True)
            optimizer.step()
            # for early stopping
            if len(loss_l) > 1 and loss.item() > loss_l[-1]:
              break     
            loss_l.append(loss.item())
            
    def generate(self, prediction_errors, minimal_threshold):
        self.generator.eval()
        with torch.no_grad():
            threshold = self.generator(prediction_errors)
            threshold = threshold.data.numpy()
            threshold = max(minimal_threshold, threshold[0,0])
        return threshold

class CNNLSTMPredictor(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers):
        super(CNNLSTMPredictor, self).__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        # CNN for temporal pattern extraction
        self.conv1 = nn.Conv1d(in_channels=num_features, 
                              out_channels=32, 
                              kernel_size=3,
                              padding=1)
        
        self.relu1 = nn.ReLU()
        
        # Unidirectional LSTM for sequence learning
        self.lstm = nn.LSTM(input_size=32, 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, 
                           batch_first=True)  
        
        self.fc = nn.Linear(hidden_size, num_features)
        
    def forward(self, x):
        # Reshape for CNN
        x = x.permute(0, 2, 1)  # (batch_size, num_features, sequence_length)
        x = self.relu1(self.conv1(x))
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, channels)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        out = self.fc(x)
        
        return out

class MultivariateTimeSeriesPredictor:
    def __init__(self, num_features, hidden_size=64, num_layers=2, lookback_len=5):
        self.num_features = num_features
        self.lookback_len = lookback_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.predictor = CNNLSTMPredictor(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
    def prepare_data(self, data):
        x, y = [], []
        
        for i in range(len(data) - self.lookback_len):
            x_i = data[i:i + self.lookback_len]
            y_i = data[i + self.lookback_len]
            
            x.append(x_i)
            y.append(y_i)
            
        return np.array(x), np.array(y)
        
    def train(self, epoch, lr, data):
        x, y = self.prepare_data(data)
        x, y = x.astype(float), y.astype(float)
        
        # Convert to PyTorch tensors
        train_tensorX = torch.Tensor(np.array(x))
        train_tensorY = torch.Tensor(np.array(y))
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        
        self.predictor.train()
        for epoch in range(epoch):
            total_loss = 0
            # Iterate through each sliding window
            for i in range(len(x)):
                optimizer.zero_grad()
                
                _x, _y = x[i], y[i]
                # Add batch dimension and make prediction
                outputs = self.predictor(torch.Tensor(_x).unsqueeze(0))
                loss = criterion(outputs, torch.Tensor(_y).unsqueeze(0))
                
                loss.backward(retain_graph=True)
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epoch}], Loss: {total_loss/len(x):.4f}')
        
        return train_tensorX, train_tensorY
    
    def predict(self, observed):
        self.predictor.eval()
        with torch.no_grad():
            predictions = self.predictor(observed)
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.numpy()
        return predictions
    
    def update(self, epoch_update, lr_update, past_observations, recent_observations):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr_update)
        
        self.predictor.train()
        
        for _ in range(epoch_update):
            optimizer.zero_grad()
            
            predicted_val = self.predictor(past_observations)
            target = torch.from_numpy(np.array(recent_observations)).float().unsqueeze(0)
            
            loss = criterion(predicted_val, target)
            loss.backward()
            optimizer.step()

