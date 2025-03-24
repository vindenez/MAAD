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
    def __init__(self, num_features, bottleneck_size, num_layers, lookback_len, hidden_sizes=None):
        super(LSTMAutoencoder, self).__init__()
        self.num_features = num_features
        self.bottleneck_size = bottleneck_size
        self.num_layers = num_layers
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
    def __init__(self, lstm_layer, lstm_unit, lookback_len, prediction_len, num_features, bottleneck_size=1):
        hidden_size = lstm_unit
        num_layers = lstm_layer
        self.lookback_len = lookback_len
        self.prediction_len = prediction_len
        self.num_features = num_features
        self.bottleneck_size = bottleneck_size
        
        # Use the AttentionLSTMAutoencoder
        self.predictor = AttentionLSTMAutoencoder(
            num_features=num_features,
            bottleneck_size=bottleneck_size,
            num_layers=num_layers,
            lookback_len=lookback_len
        )
        
        # Always use adaptive weights
        self.criterion = FeatureWeightedMSELoss(
            num_features=num_features,
            error_history_size=25,
            stability_factor=0.5
        )
        
        # Store the latest attention weights
        self.last_time_attention = None
        self.last_feature_attention = None

    def train(self, epoch, lr, data2learn):
        num_epochs = epoch
        learning_rate = lr

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
            
            # Forward pass - capture attention weights
            outputs, time_attn, feature_attn = self.predictor(train_tensorX)
            
            # Store the final attention weights
            self.last_time_attention = time_attn.detach()
            self.last_feature_attention = feature_attn.detach()
            
            # Calculate loss using weighted MSE
            batch_size = train_tensorX.size(0)
            reshaped_input = train_tensorX.view(batch_size, self.lookback_len, self.num_features)
            loss = self.criterion(outputs, reshaped_input)
            
            # Update weights based on current errors
            squared_diff = (outputs - reshaped_input) ** 2
            self.criterion.update_weights(squared_diff.detach())
            
            loss.backward()
            optimizer.step()
            
        return train_tensorX, train_tensorY
        
    def predict(self, x):
        self.predictor.eval()
        with torch.no_grad():
            predicted, time_attn, feature_attn = self.predictor(x)
            
            # Update the latest attention weights
            self.last_time_attention = time_attn
            self.last_feature_attention = feature_attn
            
            predicted = predicted.data.numpy()
        return predicted
        
    def update(self, epoch_update, lr_update, past_observations, recent_observation):
        num_epochs = epoch_update
        learning_rate = lr_update
        
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        
        self.predictor.train()
        
        # Calculate squared differences once, outside the epoch loop
        outputs, time_attn, feature_attn = self.predictor(past_observations.float())
        batch_size = past_observations.size(0)
        reshaped_input = past_observations.view(batch_size, self.lookback_len, self.num_features)
        squared_diff = (outputs - reshaped_input) ** 2
        
        # Update weights only once per timestep
        self.criterion.update_weights(squared_diff.detach())
        
        # Then do the training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            outputs, time_attn, feature_attn = self.predictor(past_observations.float())
            self.last_time_attention = time_attn
            self.last_feature_attention = feature_attn
            
            loss = self.criterion(outputs, reshaped_input)
            loss.backward()
            optimizer.step()

    def get_attention_weights(self):
        """Return the latest feature attention weights for use in error calculation"""
        if self.last_feature_attention is not None:
            # Convert to a normalized weight vector
            # Taking mean across batch and time dimensions if needed
            if len(self.last_feature_attention.shape) > 2:
                weights = torch.mean(self.last_feature_attention, dim=[0, 1])
            else:
                weights = self.last_feature_attention
                
            # Normalize to sum to 1
            weights = weights / torch.sum(weights)
            return weights
        
        # Fall back to None if no attention weights available yet
        return None

class FeatureWeightedMSELoss(nn.Module):
    def __init__(self, num_features, error_history_size=25, stability_factor=0.5):
        super(FeatureWeightedMSELoss, self).__init__()
        self.feature_weights = torch.ones(num_features, dtype=torch.float32) / num_features
        self.error_history_size = error_history_size
        self.error_history = []
        
    def forward(self, y_pred, y_true):
        if len(y_pred.shape) == 3:
            squared_diff = (y_pred - y_true) ** 2
            feature_errors = torch.mean(squared_diff, dim=[0, 1])
            
            weights = self.feature_weights.to(y_pred.device)
            weighted_errors = feature_errors * weights
            
            return torch.mean(weighted_errors)
        else:
            batch_size = y_pred.size(0)
            seq_len = y_true.size(1) if len(y_true.shape) > 2 else 1
            num_features = len(self.feature_weights)
            
            y_pred_reshaped = y_pred.view(batch_size, seq_len, num_features)
            y_true_reshaped = y_true.view(batch_size, seq_len, num_features)
            
            return self.forward(y_pred_reshaped, y_true_reshaped)
            
    def update_weights(self, errors):
        mean_errors = torch.mean(errors, dim=[0, 1])
        print(f"Current errors: {mean_errors}")
        
        # Store in history for smoothing
        self.error_history.append(mean_errors.detach())
        if len(self.error_history) > self.error_history_size:
            self.error_history.pop(0)
            
        if len(self.error_history) < self.error_history_size:
            return
            
        # Calculate average errors over the history window
        error_history_tensor = torch.stack(self.error_history)
        avg_errors = torch.mean(error_history_tensor, dim=0)
        print(f"Average errors: {avg_errors}")
        
        # Instead of using ratios, compare errors to their mean
        mean_error = torch.mean(avg_errors)
        error_scores = mean_error / (avg_errors + 1e-8)
        print(f"Error scores relative to mean: {error_scores}")
        
        # Normalize scores directly without softmax
        weights = error_scores / torch.sum(error_scores)
        
        # Apply softer bounds (0.1 to 0.7)
        weights = 0.6 * weights + 0.1
        
        print(f"New weights: {weights}")
        self.feature_weights = weights

class AttentionLSTMAutoencoder(nn.Module):
    def __init__(self, num_features, bottleneck_size, num_layers, lookback_len):
        super(AttentionLSTMAutoencoder, self).__init__()
        self.num_features = num_features
        self.bottleneck_size = bottleneck_size
        self.num_layers = num_layers
        self.lookback_len = lookback_len
        self.hidden_size = 100

        # Encoder
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Attention mechanism for time steps
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.Tanh(),
            nn.Linear(32, num_features)
        )
        
        # Bottleneck projections
        self.bottleneck_encode = nn.Linear(self.hidden_size, bottleneck_size)
        
        # Decoder initial state generator
        self.decoder_h0 = nn.Linear(bottleneck_size, self.hidden_size)
        self.decoder_c0 = nn.Linear(bottleneck_size, self.hidden_size)
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=bottleneck_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_layer = nn.Linear(self.hidden_size, num_features)
        
        # Update feature image processing layers
        self.feature_image_processor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Process 3x3 feature images
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: [batch_size, 32, 1, 1]
        )
        
        # Update correlation attention network
        self.correlation_attention = nn.Sequential(
            nn.Linear(num_features + 32, 64),  # Combine original features + CNN-processed correlations
            nn.Tanh(),
            nn.Linear(64, num_features)
        )

    def build_feature_image(self, x):
        """
        Implements vectorized computation of pairwise inner products across features.
        """
        batch_size, seq_len, num_features = x.shape
        
        # Vectorized computation using matrix multiplication
        x_transposed = x.transpose(1, 2)  # [batch_size, num_features, lookback_len]
        feature_images = torch.matmul(x, x_transposed) / seq_len  # [batch_size, num_features, num_features]
        
        return feature_images.unsqueeze(1)  # Add channel dim: [batch_size, 1, num_features, num_features]

    def apply_feature_attention(self, x):
        # x shape: [batch_size, lookback_len, num_features]
        batch_size = x.size(0)
        
        # 1. Build feature image using vectorized computation
        feature_image = self.build_feature_image(x)  # [batch_size, 1, num_features, num_features]
        
        # 2. Process through CNN
        processed_correlations = self.feature_image_processor(feature_image)  # [batch_size, 32, 1, 1]
        processed_correlations = processed_correlations.view(batch_size, -1)  # [batch_size, 32]
        
        # 3. Combine with original features
        expanded_correlations = processed_correlations.unsqueeze(1).expand(-1, self.lookback_len, -1)
        combined = torch.cat([x, expanded_correlations], dim=2)  # [batch_size, lookback_len, num_features+32]
        
        # 4. Compute correlation-aware attention weights
        attn_scores = self.correlation_attention(combined)  # [batch_size, lookback_len, num_features]
        feature_weights = F.softmax(attn_scores, dim=2)
        
        # 5. Apply weights to original features
        weighted_x = x * feature_weights
        
        return weighted_x, feature_weights
    
    def apply_attention(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores
        attn_scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights to lstm output
        context = torch.sum(attn_weights * lstm_output, dim=1)  # [batch_size, hidden_size]
        
        return context, attn_weights

    def forward(self, x):
        # x shape: [batch_size, lookback_len * num_features]
        batch_size = x.size(0)
        
        # Reshape to [batch_size, lookback_len, num_features]
        x = x.view(batch_size, self.lookback_len, self.num_features)
        
        # Apply feature attention to focus on important features at each time step
        weighted_x, feature_attn_weights = self.apply_feature_attention(x)
        
        # Pass through encoder
        encoder_output, (hidden, cell) = self.encoder(weighted_x)
        
        # Apply temporal attention to focus on important time steps
        context, time_attn_weights = self.apply_attention(encoder_output)
        
        # Create bottleneck representation
        bottleneck_vec = self.bottleneck_encode(context)
        
        # Generate initial states for decoder
        h0 = self.decoder_h0(bottleneck_vec).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.decoder_c0(bottleneck_vec).unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # Create decoder input sequence (repeat bottleneck for each time step)
        decoder_input = bottleneck_vec.unsqueeze(1).repeat(1, self.lookback_len, 1)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input, (h0, c0))
        
        # Project to output dimensions
        output = self.output_layer(decoder_output)
        
        # Return output along with both attention weights
        return output, time_attn_weights, feature_attn_weights

class AttentionAwareThresholdGenerator(nn.Module):
    def __init__(self, prediction_len, lookback_len, lstm_unit, lstm_layer):
        super().__init__()
        self.lookback_len = lookback_len
        self.prediction_len = prediction_len
        
        # The input_size should be 1 because each time step has a single scalar value
        # The sequence length is handled by the lookback_len parameter in forward()
        self.generator = LSTMPredictor(
            num_classes=prediction_len,  # Output size (prediction_len)
            input_size=1,                # Each time step has 1 feature (a scalar error value)
            hidden_size=lstm_unit,       # Hidden unit size
            num_layers=lstm_layer,       # Number of LSTM layers
            lookback_len=lookback_len    # Sequence length
        )
        self.attention_weights = None
    
    def set_attention_weights(self, weights):
        """Update the attention weights used for threshold generation"""
        self.attention_weights = weights
    
    def forward(self, x):
        # x is expected to be a tensor of shape [batch_size, lookback_len] 
        # We need to reshape it to [batch_size, lookback_len, 1] for LSTM input
        
        # Check dimensions
        if len(x.shape) == 1:
            # If input is a 1D tensor, add batch dimension and feature dimension
            x = x.view(1, -1, 1)
        elif len(x.shape) == 2:
            # If input is a 2D tensor (batch_size, sequence_length)
            # Add feature dimension
            x = x.unsqueeze(-1)
        
        # Ensure the sequence length matches lookback_len
        if x.shape[1] != self.lookback_len:
            print(f"Warning: Input sequence length {x.shape[1]} doesn't match lookback_len {self.lookback_len}")
            # Handle different sequence lengths if needed
            # ...
        
        return self.generator(x)
    
    def update(self, epoch_update, lr_update, past_errors, recent_error):
        num_epochs = epoch_update
        learning_rate = lr_update
        
        # Create a custom criterion that uses attention weights if available
        if self.attention_weights is not None:
            criterion = lambda pred, target: torch.mean(
                self.attention_weights * ((pred - target) ** 2)
            )
        else:
            criterion = torch.nn.MSELoss()
            
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.train()
        loss_l = list()
        for epoch in range(num_epochs):
            predicted_val = self(past_errors.float())
            optimizer.zero_grad()
            
            target = torch.from_numpy(np.array(recent_error).reshape(1, -1)).float()
            
            loss = criterion(predicted_val, target)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Early stopping
            if len(loss_l) > 1 and loss.item() > loss_l[-1]:
                break
            loss_l.append(loss.item())