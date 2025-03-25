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

class ConvLSTMAutoencoder(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, lookback_len):
        super(ConvLSTMAutoencoder, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookback_len = lookback_len
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=16,  
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose1d(hidden_size, 32, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose1d(32, num_features, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape to [batch_size, lookback_len, num_features]
        x = x.view(batch_size, self.lookback_len, self.num_features)
        
        # Transform for Conv1d [batch_size, num_features, lookback_len]
        x = x.transpose(1, 2)
        
        # Convolutional encoding
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        
        # Prepare for LSTM [batch_size, lookback_len, features]
        x = x.transpose(1, 2)
        
        # LSTM encoding
        enc_output, (hidden, cell) = self.encoder(x)
        
        # Apply attention
        attn_output, _ = self.attention(enc_output, enc_output, enc_output)
        attn_output = self.norm(attn_output + enc_output)  # Skip connection
        
        # Decoding
        decoder_output, _ = self.decoder(attn_output, (hidden, cell))
        
        # Reshape for deconv [batch_size, channels, length]
        x = decoder_output.transpose(1, 2)
        
        # Deconvolutional decoding
        x = F.relu(self.deconv1(x))
        x = self.dropout(x)
        output = self.deconv2(x)
        
        # Final reshape [batch_size, lookback_len, num_features]
        output = output.transpose(1, 2)
        
        return output

class MultivariateNormalDataPredictor():
    def __init__(self, lstm_layer, lstm_unit, lookback_len, prediction_len, num_features):
        hidden_size = lstm_unit
        num_layers = lstm_layer
        self.lookback_len = lookback_len
        self.prediction_len = prediction_len
        self.num_features = num_features
        
        self.predictor = ConvLSTMAutoencoder(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            lookback_len=lookback_len
        )
        
    def train(self, epoch, lr, data2learn):
        num_epochs = epoch
        learning_rate = lr

        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        
        # Slide data
        x, y = sliding_windows_multivariate(data2learn, self.lookback_len, self.prediction_len)
        x = x.astype(float)
        y = y.astype(float)
        
        # Prepare data for training
        train_tensorX = torch.Tensor(np.array(x))
        train_tensorY = torch.Tensor(np.array(y))
        
        self.predictor.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.predictor(train_tensorX)
            
            # Calculate loss (reconstruction error)
            loss = criterion(outputs, train_tensorX)
            loss.backward()
            optimizer.step()
            
        return train_tensorX, train_tensorY
        
    def predict(self, x):
        self.predictor.eval()
        with torch.no_grad():
            predicted = self.predictor(x)
            predicted = predicted.data.numpy()
        return predicted
        
    def update(self, epoch_update, lr_update, past_observations, recent_observation):
        num_epochs = epoch_update
        learning_rate = lr_update
        
        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        
        self.predictor.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            # past_observations shape: [1, lookback_len * num_features]
            predicted_val = self.predictor(past_observations.float())
            
            # The autoencoder outputs [batch_size, lookback_len, num_features]
            batch_size = past_observations.size(0)
            reshaped_input = past_observations.view(batch_size, self.lookback_len, self.num_features)
            
            # Calculate loss (reconstruction error)
            loss = criterion(predicted_val, reshaped_input)
            loss.backward()
            optimizer.step()

class MultivariateLSTMPredictor(nn.Module):
    def __init__(self, num_features, input_size, hidden_size, num_layers, lookback_len):
        super(MultivariateLSTMPredictor, self).__init__()
        
        self.num_features = num_features
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = lookback_len

        self.lstm = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size,
                           num_layers=num_layers, 
                           batch_first=True)
        
        # Output layer produces threshold for each feature
        self.fc = nn.Linear(in_features=hidden_size, 
                           out_features=num_features)

    def forward(self, x):     
        x, (h_out, _) = self.lstm(x)
        h_out = h_out.view(-1, self.hidden_size)   
        # Generate threshold for each feature
        thresholds = self.fc(h_out)
        return thresholds

class MultivariateTresholdGenerator():
    def __init__(self, lstm_layer, lstm_unit, lookback_len, prediction_len, num_features):
        hidden_size = lstm_unit
        num_layers = lstm_layer
        self.lookback_len = lookback_len
        self.prediction_len = prediction_len
        self.num_features = num_features
        
        # Rename this to model to be more explicit
        self.model = MultivariateLSTMPredictor(
            num_features=num_features,
            input_size=lookback_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            lookback_len=lookback_len
        )

    def parameters(self):
        # Add this method to expose the model's parameters
        return self.model.parameters()

    def train(self):
        # Set model to training mode
        self.model.train()

    def eval(self):
        # Set model to evaluation mode
        self.model.eval()

    def generate(self, prediction_errors, minimal_threshold):
        self.model.eval()
        with torch.no_grad():
            thresholds = self.model(prediction_errors)
            thresholds = thresholds.data.numpy()
            # Apply minimal threshold to all features
            thresholds = np.maximum(minimal_threshold, thresholds)
        return thresholds