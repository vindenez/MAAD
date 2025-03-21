import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from utils import *

class FeatureAttention(nn.Module):
    def __init__(self, num_features):
        super(FeatureAttention, self).__init__()
        self.num_features = num_features
        
    def forward(self, x, target_idx=0):
        """
        x: input tensor of shape [batch_size, seq_len, num_features]
        target_idx: index of the target feature (default: 0)
        
        Returns:
        - weighted_x: weighted features based on attention
        - attention_weights: weights assigned to each feature
        """
        batch_size, seq_len, num_features = x.size()
        
        # Reshape to calculate similarity
        # For each sequence position, calculate similarity between target and other features
        attention_weights = []
        
        for t in range(seq_len):
            # Extract features at time step t
            features_t = x[:, t, :]  # [batch_size, num_features]
            
            # Get target feature
            target_feature = features_t[:, target_idx:target_idx+1]  # [batch_size, 1]
            
            # Calculate similarity (dot product) between target and all features
            # sim(target, Ki) = target · Ki
            similarity = torch.bmm(
                target_feature.view(batch_size, 1, 1),
                features_t.view(batch_size, 1, num_features)
            ).squeeze(1)  # [batch_size, num_features]
            
            # Apply softmax to get attention weights
            # wi = softmax(simi) = e^(simi) / sum(e^(simj))
            weights = F.softmax(similarity, dim=1)  # [batch_size, num_features]
            attention_weights.append(weights)
        
        # Stack attention weights for all time steps
        attention_weights = torch.stack(attention_weights, dim=1)  # [batch_size, seq_len, num_features]
        
        # Apply attention weights to input
        weighted_x = x * attention_weights
        
        return weighted_x, attention_weights

class ATTLSTMPredictor(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, lookback_len, num_features, target_feature=0):
        super(ATTLSTMPredictor, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lookback_len = lookback_len
        self.num_features = num_features
        self.target_feature = target_feature

        # Feature attention layer
        self.feature_attention = FeatureAttention(num_features)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)

    def forward(self, x):
        # x shape: [batch_size, lookback_len * num_features]
        batch_size = x.size(0)
        
        # Reshape to [batch_size, lookback_len, num_features]
        x = x.view(batch_size, self.lookback_len, self.num_features)
        
        # Apply feature attention
        weighted_x, self.attention_weights = self.feature_attention(x, self.target_feature)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(weighted_x)
        
        # Use the final hidden state
        h_out = h_n[-1]  # Get the hidden state from the last layer
        
        # Final prediction
        out = self.fc(h_out)
        return out
    
    def get_attention_weights(self):
        if hasattr(self, 'attention_weights'):
            return self.attention_weights
        return None

class NormalDataPredictor():
    def __init__(self, lstm_layer, lstm_unit, lookback_len, prediction_len, num_features, target_feature=0):
        hidden_size = lstm_unit
        num_layers = lstm_layer
        self.lookback_len = lookback_len
        self.prediction_len = prediction_len
        self.num_features = num_features
        self.target_feature = target_feature
        
        self.predictor = ATTLSTMPredictor(
            num_classes=1,  # Predict one value (target feature)
            input_size=lookback_len, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            lookback_len=lookback_len,
            num_features=num_features,
            target_feature=target_feature
        )
        
    def train(self, epoch, lr, data2learn):
        num_epochs = epoch
        learning_rate = lr

        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        
        # slide data
        x, y = sliding_windows_multivariate(data2learn, self.lookback_len, self.prediction_len)
        x = x.astype(float)
        y = y.astype(float)
        
        # Extract target feature for training
        y_target = y[:, 0, self.target_feature]  # First prediction step, target feature
        
        # prepare data for training
        train_tensorX = torch.Tensor(np.array(x))
        train_tensorY = torch.Tensor(np.array(y_target))
        
        self.predictor.train()
        for epoch in range(num_epochs):
            for i in range(len(x)):            
                optimizer.zero_grad()
                
                _x, _y = x[i], y_target[i]
                # Reshape for the model
                _x = _x.reshape(1, self.lookback_len * self.num_features)
                outputs = self.predictor(torch.Tensor(_x))
                loss = criterion(outputs, torch.Tensor([_y]).reshape((1,-1)))
                loss.backward(retain_graph=True)
                optimizer.step()
                
    def predict(self, x):
        self.predictor.eval()
        with torch.no_grad():
            predicted = self.predictor(x)
            predicted = predicted.data.numpy()
            predicted = predicted[0,0]
        return predicted
        
    def update(self, epoch_update, lr_update, past_observations, recent_observation):
        num_epochs = epoch_update
        learning_rate = lr_update
        
        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        
        self.predictor.train()
        loss_l = list()
        for epoch in range(num_epochs):
            predicted_val = self.predictor(past_observations.float())
            optimizer.zero_grad()           
            loss = criterion(predicted_val, torch.from_numpy(np.array([recent_observation]).reshape(1, -1)).float())        
            loss.backward(retain_graph=True)
            optimizer.step()
            # for early stopping
            if len(loss_l) > 1 and loss.item() > loss_l[-1]:
              break     
            loss_l.append(loss.item())
            
    def get_attention_weights(self):
        return self.predictor.get_attention_weights()

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
        # Reshape input if needed
        batch_size = x.size(0)
        if x.dim() == 2:  # If input is [batch_size, seq_length*input_size]
            x = x.view(batch_size, self.seq_length, self.input_size)
            
        # Process through LSTM
        x, (h_out, _) = self.lstm(x)
        h_out = h_out.view(-1, self.hidden_size)   
        out = self.fc(h_out)
        return out

class AnomalousThresholdGenerator():
    def __init__(self, lstm_layer, lstm_unit, lookback_len, prediction_len, num_features):
        hidden_size = lstm_unit
        num_layers = lstm_layer
        self.lookback_len = lookback_len
        self.prediction_len = prediction_len
        self.num_features = num_features
        
        # Create a LSTM predictor for threshold generation
        # For scalar errors, input_size=1
        self.generator = LSTMPredictor(
            num_classes=prediction_len,
            input_size=1,  # For scalar prediction errors
            hidden_size=hidden_size,
            num_layers=num_layers,
            lookback_len=lookback_len
        )
                    
    def train(self, epoch, lr, data2learn):
        num_epochs = epoch
        learning_rate = lr

        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        loss_l = list()
        
        # Prepare data
        x, y = sliding_windows(data2learn, self.lookback_len, self.prediction_len)
        x, y = x.astype(float), y.astype(float)
        y = np.reshape(y, (y.shape[0], y.shape[1]))
        
        # Prepare data for training
        train_tensorX = torch.Tensor(np.array(x))
        train_tensorY = torch.Tensor(np.array(y))
        
        # Training loop
        self.generator.train()
        for epoch in range(num_epochs):
            for i in range(len(x)):            
                optimizer.zero_grad()
                
                _x, _y = x[i], y[i]
                # Reshape input for scalar errors
                _x_reshaped = torch.Tensor(_x).reshape((1, self.lookback_len, 1))
                outputs = self.generator(_x_reshaped)
                loss = criterion(outputs, torch.Tensor(_y).reshape((1,-1)))
                loss.backward(retain_graph=True)
                optimizer.step()
                
        return train_tensorX, train_tensorY
                
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
            loss = criterion(predicted_val, recent_error)
            loss.backward(retain_graph=True)
            optimizer.step()
            # for early stopping
            if len(loss_l) > 1 and loss.item() > loss_l[-1]:
              break     
            loss_l.append(loss.item())
            
    def __call__(self, x):
        """Allow the class to be called like a function, forwarding to the generator"""
        self.generator.eval()
        with torch.no_grad():
            return self.generator(x)

class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, lookback_len):
        super(LSTMAutoencoder, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookback_len = lookback_len

        # Encoder
        self.encoder = nn.LSTM(
            input_size=num_features,  # Each time step has num_features
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Final output layer
        self.fc = nn.Linear(hidden_size, num_features)

    def forward(self, x):
        # x shape: [batch_size, lookback_len * num_features]
        batch_size = x.size(0)
        
        # Reshape to [batch_size, lookback_len, num_features]
        x = x.view(batch_size, self.lookback_len, self.num_features)
        
        # Encoding
        _, (hidden, cell) = self.encoder(x)
        
        # Decoding
        decoder_input = torch.zeros(batch_size, self.lookback_len, self.hidden_size)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Final prediction
        output = self.fc(decoder_output)
        return output

class MultivariateNormalDataPredictor():
    def __init__(self, lstm_layer, lstm_unit, lookback_len, prediction_len, num_features):
        hidden_size = lstm_unit
        num_layers = lstm_layer
        self.lookback_len = lookback_len
        self.prediction_len = prediction_len
        self.num_features = num_features
        
        self.predictor = LSTMAutoencoder(
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
        
        # Prepare data
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
            
        # Return the training data tensors
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
            
            # Reshape past_observations to match the output shape of the autoencoder
            # The autoencoder outputs [batch_size, lookback_len, num_features]
            batch_size = past_observations.size(0)
            reshaped_input = past_observations.view(batch_size, self.lookback_len, self.num_features)
            
            # Calculate loss (reconstruction error)
            loss = criterion(predicted_val, reshaped_input)
            loss.backward()
            optimizer.step()