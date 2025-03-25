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

class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features, bottleneck_size, lookback_len, hidden_sizes=None):
        super(LSTMAutoencoder, self).__init__()
        self.num_features = num_features
        self.bottleneck_size = bottleneck_size
        self.lookback_len = lookback_len
        
        # Default hidden sizes if not provided
        if hidden_sizes is None:
            hidden_sizes = [100, 50]
        
        self.hidden_size_1 = hidden_sizes[0]
        self.hidden_size_2 = hidden_sizes[1]
        
        # Encoder - gradually reduce dimension
        self.encoder_lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_size_1,
            num_layers=1,
            batch_first=True
        )
        
        self.encoder_lstm2 = nn.LSTM(
            input_size=self.hidden_size_1,
            hidden_size=self.hidden_size_2,
            num_layers=1,
            batch_first=True
        )
        
        self.encoder_lstm3 = nn.LSTM(
            input_size=self.hidden_size_2,
            hidden_size=bottleneck_size,
            num_layers=1,
            batch_first=True
        )
        
        # Decoder - gradually expand dimension
        self.decoder_lstm1 = nn.LSTM(
            input_size=bottleneck_size,
            hidden_size=self.hidden_size_2,
            num_layers=1,
            batch_first=True
        )
        
        self.decoder_lstm2 = nn.LSTM(
            input_size=self.hidden_size_2,
            hidden_size=self.hidden_size_1,
            num_layers=1,
            batch_first=True
        )
        
        self.decoder_lstm3 = nn.LSTM(
            input_size=self.hidden_size_1,
            hidden_size=num_features,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # x shape: [batch_size, lookback_len * num_features]
        batch_size = x.size(0)
        
        # Reshape to [batch_size, lookback_len, num_features]
        x = x.view(batch_size, self.lookback_len, self.num_features)
        
        # Encoding - progressively reduce dimensions
        enc1_out, _ = self.encoder_lstm1(x)
        enc2_out, _ = self.encoder_lstm2(enc1_out)
        _, (bottleneck, _) = self.encoder_lstm3(enc2_out)
        
        # bottleneck shape: [1, batch_size, bottleneck_size]
        # Repeat the bottleneck for each time step
        bottleneck = bottleneck.permute(1, 0, 2)  # [batch_size, 1, bottleneck_size]
        bottleneck_repeated = bottleneck.repeat(1, self.lookback_len, 1)  # [batch_size, lookback_len, bottleneck_size]
        
        # Decoding - progressively increase dimensions
        dec1_out, _ = self.decoder_lstm1(bottleneck_repeated)
        dec2_out, _ = self.decoder_lstm2(dec1_out)
        output, _ = self.decoder_lstm3(dec2_out)
        
        return output

class MultivariateNormalDataPredictor():
    def __init__(self, lstm_unit, lookback_len, prediction_len, num_features):
        self.lookback_len = lookback_len
        self.prediction_len = prediction_len
        self.num_features = num_features
        
        # Unpack the LSTM sizes
        hidden_size_1, hidden_size_2, bottleneck_size = lstm_unit
        
        self.predictor = LSTMAutoencoder(
            num_features=num_features,
            bottleneck_size=bottleneck_size,
            lookback_len=lookback_len,
            hidden_sizes=[hidden_size_1, hidden_size_2]
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