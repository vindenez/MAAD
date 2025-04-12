import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from utils import *
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

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

class MultivariateLSTMPredictor(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, lookback_len):
        super(MultivariateLSTMPredictor, self).__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.lookback_len = lookback_len
        
        # Feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, num_features),
            nn.Sigmoid()
        )
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=num_features,  
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layer for predicting next values of all features
        self.fc = nn.Linear(hidden_size, num_features)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply feature attention
        # Calculate attention scores for each feature
        feature_weights = self.feature_attention(x.mean(dim=1))
        # Apply feature attention weights
        x = x * feature_weights.unsqueeze(1)
        
        # Process through LSTM
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Apply temporal attention
        # lstm_out shape: [batch_size, lookback_len, hidden_size]
        temporal_scores = self.temporal_attention(lstm_out).squeeze(-1)  # [batch_size, lookback_len]
        temporal_weights = F.softmax(temporal_scores, dim=1).unsqueeze(2)  # [batch_size, lookback_len, 1]
        
        # Apply temporal weights to get context vector
        context = torch.sum(lstm_out * temporal_weights, dim=1)  # [batch_size, hidden_size]
        
        # Generate predictions
        out = self.fc(context)
        
        return out

class ParameterAwareTransformerLSTMPredictor(nn.Module):
    def __init__(self, num_nodes, params_per_node, hidden_size, num_layers, lookback_len, 
                 d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super(ParameterAwareTransformerLSTMPredictor, self).__init__()
        
        self.num_nodes = num_nodes  # Number of sensor nodes
        self.params_per_node = params_per_node  # Parameters per node
        self.num_features = num_nodes * params_per_node  # Total features
        self.hidden_size = hidden_size
        self.lookback_len = lookback_len
        self.d_model = d_model
        
        # Parameter type embeddings (to distinguish temperature, pressure, etc.)
        self.parameter_embeddings = nn.Parameter(torch.randn(params_per_node, d_model // 2))
        
        # Node embeddings (to identify which sensor node the data comes from)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, d_model // 2))
        
        # Input projection to transformer dimension
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding for temporal dimension
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder - separate self-attention for each time step
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Calculate the actual input size for LSTM
        lstm_input_size = self.num_nodes * self.params_per_node * d_model
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.output_projection = nn.Linear(hidden_size, self.num_features)
        
    def forward(self, x):
        # x shape: [batch_size, lookback_len, num_features]
        batch_size = x.size(0)
        
        # Reshape to separate node and parameter dimensions
        # [batch, time, node, param]
        x_reshaped = x.view(batch_size, self.lookback_len, self.num_nodes, self.params_per_node)
        
        # Process each time step through transformer to model parameter relationships
        transformed_sequence = []
        
        for t in range(self.lookback_len):
            # Get data for current time step: [batch, node, param]
            x_t = x_reshaped[:, t, :, :]
            
            # Prepare input for transformer with feature embeddings
            # Reshape to [batch * node * param, 1]
            x_flat = x_t.reshape(batch_size * self.num_nodes * self.params_per_node, 1)
            
            # Project to embedding space: [batch * node * param, d_model]
            x_proj = self.input_projection(x_flat)
            
            # Create indices for parameter and node types
            node_indices = torch.arange(self.num_nodes).repeat_interleave(self.params_per_node)
            node_indices = node_indices.repeat(batch_size).to(x.device)
            
            param_indices = torch.arange(self.params_per_node).repeat(self.num_nodes)
            param_indices = param_indices.repeat(batch_size).to(x.device)
            
            # Add parameter and node embeddings
            node_emb = self.node_embeddings[node_indices]  # [batch * node * param, d_model//2]
            param_emb = self.parameter_embeddings[param_indices]  # [batch * node * param, d_model//2]
            
            # Concatenate embeddings with projected values
            # Replace first half of x_proj with node_emb and second half with param_emb
            x_proj[:, :self.d_model//2] = node_emb
            x_proj[:, self.d_model//2:] = param_emb
            
            # Reshape for transformer: [batch, node*param, d_model]
            x_transformer = x_proj.view(batch_size, self.num_nodes * self.params_per_node, self.d_model)
            
            # Apply transformer to capture parameter relationships
            x_transformer = self.transformer_encoder(x_transformer)
            
            # Store the transformed representation
            transformed_sequence.append(x_transformer)
        
        # Stack along time dimension: [batch, time, node*param, d_model]
        x_transformed = torch.stack(transformed_sequence, dim=1)
        
        # Reshape for LSTM: [batch, time, node*param*d_model]
        x_for_lstm = x_transformed.reshape(batch_size, self.lookback_len, -1)
        
        # Process through LSTM for temporal dynamics
        _, (h_n, _) = self.lstm(x_for_lstm)
        h_n = h_n[-1]  # Take the last layer's hidden state
        
        # Generate predictions for all features
        out = self.output_projection(h_n)  # [batch_size, num_features]
        
        return out

class ParameterAwareTimeSeriesPredictor:
    def __init__(self, num_nodes, params_per_node, hidden_size=100, num_layers=3, lookback_len=3,
                 d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        
        self.num_nodes = num_nodes
        self.params_per_node = params_per_node
        self.num_features = num_nodes * params_per_node
        self.lookback_len = lookback_len
        
        self.predictor = ParameterAwareTransformerLSTMPredictor(
            num_nodes=num_nodes,
            params_per_node=params_per_node,
            hidden_size=hidden_size,
            num_layers=num_layers,
            lookback_len=lookback_len,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
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
                
                loss.backward()
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

