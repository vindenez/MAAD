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
class LongShortTermAttention(nn.Module):
    def __init__(self, d_model, n_heads, num_segments=8, dropout=0.1):
        super(LongShortTermAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_segments = num_segments
        
        # Long-term attention (vanilla attention)
        self.long_term_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Short-term attention (segment-wise attention)
        self.short_term_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Prototypical hidden series for adaptive segmentation
        self.proto_segments = nn.Parameter(torch.randn(num_segments, d_model))
        
        # Fusion layer to combine long and short term representations
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def adaptive_segment(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Compute DTW alignment matrices between sequence and prototypes
        segments = []
        segment_indices = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
        
        # Simple implementation of soft-DTW segmentation
        # In practice, you would implement a differentiable DTW as in the paper
        for b in range(batch_size):
            # Compute similarities between sequence tokens and prototypes
            similarities = torch.matmul(x[b], self.proto_segments.transpose(0, 1))  # [seq_len, num_segments]
            
            # Assign each token to the most similar prototype
            segment_idx = torch.argmax(similarities, dim=1)  # [seq_len]
            segment_indices[b] = segment_idx
            
            # Group by segments
            batch_segments = []
            for i in range(self.num_segments):
                # Get indices for this segment
                mask = (segment_idx == i)
                if mask.sum() > 0:
                    # Extract tokens for this segment
                    segment_tokens = x[b, mask]
                    batch_segments.append((i, segment_tokens))
                    
            segments.append(batch_segments)
            
        return segments, segment_indices
            
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Long-term attention (global)
        long_term_out, _ = self.long_term_attn(x, x, x)
        
        # Adaptive segmentation for short-term attention
        segments, segment_indices = self.adaptive_segment(x)  # Get both segments and indices
        
        # Process each segment with short-term attention
        short_term_out = torch.zeros_like(x)
        
        for b in range(batch_size):
            for seg_idx, segment_tokens in segments[b]:
                # Apply self-attention within each segment
                if len(segment_tokens) > 1:  # Need at least 2 tokens for attention
                    seg_out, _ = self.short_term_attn(
                        segment_tokens.unsqueeze(0),
                        segment_tokens.unsqueeze(0),
                        segment_tokens.unsqueeze(0)
                    )
                    
                    mask = (segment_indices[b] == seg_idx)
                    positions = torch.where(mask)[0]
                    
                    for i, pos in enumerate(positions):
                        if i < len(seg_out[0]):  # Safety check
                            short_term_out[b, pos] = seg_out[0, i]
                else:
                    # For segments with only one token, just copy the token
                    positions = torch.where(segment_indices[b] == seg_idx)[0]
                    if len(positions) > 0:
                        pos = positions[0]
                        short_term_out[b, pos] = segment_tokens[0]
        
        # Combine long-term and short-term representations
        combined = torch.cat([long_term_out, short_term_out], dim=-1)
        output = self.fusion(combined)
        
        return output
    
class VariableSpecificConvolution(nn.Module):
    def __init__(self, num_nodes, params_per_node, d_model, kernel_size=3):
        super(VariableSpecificConvolution, self).__init__()
        self.num_nodes = num_nodes
        self.params_per_node = params_per_node
        self.d_model = d_model
        
        # Create separate convolutional layers for each variable
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding=kernel_size//2
            ) for _ in range(num_nodes * params_per_node)
        ])
        
        # Variable attention mechanism
        self.variable_attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.projection = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # x: [batch_size, seq_len, num_features]
        batch_size, seq_len, num_features = x.size()
        
        # Process each variable separately
        var_outputs = []
        
        for i in range(num_features):
            # Extract this variable's time series
            var_series = x[:, :, i].unsqueeze(1)  # [batch_size, 1, seq_len]
            
            # Apply 1D convolution
            conv_out = self.convs[i](var_series)  # [batch_size, d_model, seq_len]
            conv_out = conv_out.transpose(1, 2)  # [batch_size, seq_len, d_model]
            
            var_outputs.append(conv_out)
        
        # Stack variable outputs
        stacked_outputs = torch.stack(var_outputs, dim=1)  # [batch_size, num_features, seq_len, d_model]
        
        # Apply variable attention
        var_weights = self.variable_attention(stacked_outputs.mean(dim=2))  # [batch_size, num_features, 1]
        weighted_outputs = stacked_outputs * var_weights.unsqueeze(2)  # [batch_size, num_features, seq_len, d_model]
        
        # Aggregate across variables
        aggregated = weighted_outputs.sum(dim=1)  # [batch_size, seq_len, d_model]
        
        # Final projection
        output = self.projection(aggregated)
        
        return output

class MRTransformerPredictor(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 params_per_node, 
                 lookback_len,
                 d_model=256, 
                 nhead=8, 
                 num_encoder_layers=3, 
                 num_decoder_layers=3,
                 dim_feedforward=768, 
                 dropout=0.1, 
                 num_segments=8):
        super(MRTransformerPredictor, self).__init__()
        
        self.num_nodes = num_nodes
        self.params_per_node = params_per_node
        self.num_features = num_nodes * params_per_node
        self.lookback_len = lookback_len
        self.d_model = d_model
        
        # Parameter and node embeddings
        self.parameter_embeddings = nn.Parameter(torch.randn(params_per_node, d_model // 2))
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, d_model // 2))
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Long short-term attention layers for encoder
        self.encoder_lst_layers = nn.ModuleList([
            LongShortTermAttention(d_model, nhead, num_segments, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Variable-specific temporal convolution
        self.var_specific_conv = VariableSpecificConvolution(
            num_nodes, 
            params_per_node, 
            d_model,
            kernel_size=5
        )
        
        # Long short-term attention layers for decoder
        self.decoder_lst_layers = nn.ModuleList([
            LongShortTermAttention(d_model, nhead, num_segments, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Cross-attention for decoder
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward networks
        self.encoder_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_encoder_layers)
        ])
        
        self.decoder_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_decoder_layers)
        ])
        
        # Layer normalization
        self.encoder_norms1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_encoder_layers)
        ])
        self.encoder_norms2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_encoder_layers)
        ])
        
        self.decoder_norms1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_decoder_layers)
        ])
        self.decoder_norms2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_decoder_layers)
        ])
        self.decoder_norms3 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_decoder_layers)
        ])
        
        # Query embedding for decoder
        self.query_embed = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Output projection
        self.output_projection = nn.Linear(d_model, self.num_features)
        
        # Integration layer for combining variable-consistent and variable-specific features
        self.integration = nn.Linear(d_model * 2, d_model)
        
    def _prepare_inputs(self, x):
        # x shape: [batch_size, lookback_len, num_features]
        batch_size = x.size(0)
        
        # Reshape to separate node and parameter dimensions
        # [batch, time, node, param]
        x_reshaped = x.view(batch_size, self.lookback_len, self.num_nodes, self.params_per_node)
        
        # Process each time step
        time_features = []
        
        for t in range(self.lookback_len):
            # Get data for current time step: [batch, node, param]
            x_t = x_reshaped[:, t, :, :]
            
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
            
            # Combine embeddings with projected values
            x_proj_combined = torch.zeros_like(x_proj)
            x_proj_combined[:, :self.d_model//2] = node_emb
            x_proj_combined[:, self.d_model//2:] = param_emb
            
            # Reshape back: [batch, node*param, d_model]
            x_time = x_proj_combined.view(batch_size, self.num_nodes * self.params_per_node, self.d_model)
            
            # Average across features to get time representation
            x_time_avg = x_time.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
            time_features.append(x_time_avg)
        
        # Stack along time dimension: [batch, lookback_len, d_model]
        time_sequence = torch.cat(time_features, dim=1)
        
        # Apply positional encoding
        time_sequence = self.pos_encoder(time_sequence)
        
        return time_sequence, x
        
    def forward(self, x):
        # x shape: [batch_size, lookback_len, num_features]
        batch_size = x.size(0)
        
        # Prepare inputs for encoder
        encoder_input, orig_input = self._prepare_inputs(x)
        
        # Variable-specific processing with temporal convolution
        var_specific_features = self.var_specific_conv(x)
        
        # Encoder processing with long short-term attention
        encoder_output = encoder_input
        
        for i in range(len(self.encoder_lst_layers)):
            # Long short-term attention
            attn_output = self.encoder_lst_layers[i](encoder_output)
            encoder_output = self.encoder_norms1[i](encoder_output + attn_output)
            
            # Feed-forward
            ff_output = self.encoder_ffn[i](encoder_output)
            encoder_output = self.encoder_norms2[i](encoder_output + ff_output)
        
        # Prepare decoder input (query embedding)
        decoder_input = self.query_embed.repeat(batch_size, 1, 1)
        
        # Decoder processing
        decoder_output = decoder_input
        
        for i in range(len(self.decoder_lst_layers)):
            # Self-attention
            self_attn_output = self.decoder_lst_layers[i](decoder_output)
            decoder_output = self.decoder_norms1[i](decoder_output + self_attn_output)
            
            # Cross-attention with encoder output
            cross_attn_output, _ = self.cross_attention(
                decoder_output, encoder_output, encoder_output
            )
            decoder_output = self.decoder_norms2[i](decoder_output + cross_attn_output)
            
            # Feed-forward
            ff_output = self.decoder_ffn[i](decoder_output)
            decoder_output = self.decoder_norms3[i](decoder_output + ff_output)
        
        # Combine variable-consistent and variable-specific features
        # Adjust dimensions if needed
        if var_specific_features.size(1) != decoder_output.size(1):
            var_specific_features = var_specific_features[:, -1:, :]
            
        combined_features = torch.cat([decoder_output, var_specific_features], dim=-1)
        integrated_features = self.integration(combined_features)
        
        # Generate final predictions
        output = self.output_projection(integrated_features.squeeze(1))
        
        return output
    
class MRTransformerTimeSeriesPredictor:
    def __init__(self, num_nodes, params_per_node, lookback_len=5,
                 d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=768, dropout=0.1, num_segments=8):
        
        self.num_nodes = num_nodes
        self.params_per_node = params_per_node
        self.num_features = num_nodes * params_per_node
        self.lookback_len = lookback_len
        
        self.predictor = MRTransformerPredictor(
            num_nodes=num_nodes,
            params_per_node=params_per_node,
            lookback_len=lookback_len,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_segments=num_segments
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
        
        train_tensorX = torch.Tensor(np.array(x))
        train_tensorY = torch.Tensor(np.array(y))
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        
        # Early stopping parameters
        patience = 20
        best_loss = float('inf')
        patience_counter = 0
        min_delta = 0.0001
        
        self.predictor.train()
        epoch_losses = []
        
        for epoch_idx in range(epoch):
            total_loss = 0
            
            for i in range(len(x)):
                optimizer.zero_grad()
                
                _x, _y = x[i], y[i]
                outputs = self.predictor(torch.Tensor(_x).unsqueeze(0))
                loss = criterion(outputs, torch.Tensor(_y).unsqueeze(0))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(x)
            epoch_losses.append(avg_loss)
            
            if (epoch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch_idx+1}/{epoch}], Loss: {avg_loss:.4f}')
            
            # Early stopping
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience and epoch_idx > 300:
                print(f'Early stopping at epoch {epoch_idx+1}. Best loss: {best_loss:.4f}')
                break
        
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
        
        past_obs_np = past_observations.detach().cpu().numpy()
        recent_observations_np = recent_observations.reshape(1, -1)
        
        # Shift window to create new training example
        if len(past_obs_np.shape) == 3:
            x_update = past_obs_np[0, 1:, :]
            x_update = np.vstack([x_update, recent_observations_np])
            x_update = torch.tensor(x_update, dtype=torch.float32).unsqueeze(0)
        else:
            x_update = past_observations
        
        losses = []
        for epoch in range(epoch_update):
            optimizer.zero_grad()
            
            predicted_val = self.predictor(past_observations)
            target = torch.from_numpy(np.array(recent_observations)).float().unsqueeze(0)
            
            loss = criterion(predicted_val, target)
            
            # Add consistency regularization
            if 'x_update' in locals() and len(past_obs_np.shape) == 3:
                try:
                    consistency_weight = 0.3
                    with torch.no_grad():
                        pseudo_target = self.predictor(past_observations).detach()
                    
                    next_pred = self.predictor(x_update)
                    consistency_loss = consistency_weight * criterion(next_pred, pseudo_target)
                    loss += consistency_loss
                except:
                    pass
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Early stopping
            if len(losses) > 1 and losses[-1] > losses[-2] * 1.05:
                break