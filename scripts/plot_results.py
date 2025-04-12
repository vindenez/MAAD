import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob

# Configuration
logs_directory = 'results/Austevoll/'  # LSTM-AE logs directory

# Feature configuration from config.py
feature_columns = [
    "conductivity_conductivity", 
    "pressure_pressure",
    "pressure_temperature"
]

# Value range configuration for normalization
value_range_config = {
    "conductivity_conductivity": (25.0, 38.0),
    "pressure_pressure": (299.0, 321.0),
    "pressure_temperature": (5.0, 17.0)
}

# Dictionary to store anomalies for each feature
feature_anomalies = {}

# Check if system log file exists
system_log_path = os.path.join(logs_directory, 'system_alog.csv')
if os.path.exists(system_log_path):
    system_log_df = pd.read_csv(system_log_path)
    # Extract anomalies from system log (where is_anomalous is True)
    anomaly_indices = system_log_df[system_log_df['is_anomalous'] == True]['idx'].tolist()
    print(f"Found {len(anomaly_indices)} system-level anomalies")
else:
    # If no system log, collect anomalies from individual feature logs
    anomaly_indices = []
    
    for feature in feature_columns:
        feature_log_path = os.path.join(logs_directory, f"{feature}_log.csv")
        if os.path.exists(feature_log_path):
            log_df = pd.read_csv(feature_log_path)
            # Extract anomalies
            feature_anomalies[feature] = log_df[log_df['is_anomalous'] == True]['idx'].tolist()
            anomaly_indices.extend(feature_anomalies[feature])
    
    # Remove duplicates
    anomaly_indices = sorted(set(anomaly_indices))
    print(f"Found {len(anomaly_indices)} total anomalies from feature logs")

# Create a figure for normalized observed values
fig, ax = plt.subplots(figsize=(16, 8))

# Dictionary to store max index for each feature
max_idx = 0

# Define colors for different features
feature_colors = {
    "conductivity_conductivity": "blue",
    "pressure_pressure": "green",
    "pressure_temperature": "purple"
}

# Load observed values from log files and plot normalized values
for feature in feature_columns:
    feature_log_path = os.path.join(logs_directory, f"{feature}_log.csv")
    
    if not os.path.exists(feature_log_path):
        print(f"Log file for {feature} not found.")
        continue
    
    # Load the log file
    log_df = pd.read_csv(feature_log_path)
    
    # Get observed values and normalize
    min_val, max_val = value_range_config[feature]
    
    # Normalize the observed values to [0, 1] range
    normalized_values = (log_df['observed'] - min_val) / (max_val - min_val)
    normalized_values = np.clip(normalized_values, 0, 1)
    
    # Plot the normalized values
    ax.plot(log_df['idx'], normalized_values, label=feature, linewidth=1.5, color=feature_colors[feature])
    
    # Update max index
    max_idx = max(max_idx, log_df['idx'].max())

# Mark anomalies with vertical lines using feature-specific colors
for feature in feature_columns:
    if feature in feature_anomalies:
        for idx in feature_anomalies[feature]:
            ax.axvline(x=idx, color=feature_colors[feature], linestyle='-', alpha=0.3, linewidth=2.5)

# Add text label for the number of anomalies
ax.text(0.02, 0.02, f"Total anomalies: {len(anomaly_indices)}", transform=ax.transAxes, 
         bbox=dict(facecolor='white', alpha=0.7), fontsize=10)

# Create custom legend with anomaly markers
legend_handles = []
legend_labels = []

# Add feature lines to legend
for feature in feature_columns:
    legend_handles.append(Line2D([0], [0], color=feature_colors[feature], linewidth=1.5))
    legend_labels.append(feature)

# Add anomaly markers to legend
for feature in feature_columns:
    if feature in feature_anomalies:
        legend_handles.append(Line2D([0], [0], color=feature_colors[feature], linestyle='solid', linewidth=2.5, alpha=0.3))
        legend_labels.append(f'{feature} anomalies')

ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize=10)

# Set y-axis limits to the normalized range
ax.set_ylim(0, 1)

# Ensure x-axis spans the full dataset
ax.set_xlim(min(16, 1), max_idx)  # Use 16 as minimum if available (first log entry)

# Add reference lines at 0 and 1 to show the normalized range
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
ax.axhline(y=1, color='gray', linestyle='-', alpha=0.5, linewidth=1)

# Add title and labels
ax.set_title('Normalized Sensor Values with Feature-Specific Anomaly Detection (LSTM-AE)', fontsize=14)
ax.set_xlabel('Timestep', fontsize=12)
ax.set_ylabel('Normalized Value [0-1]', fontsize=12)

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Add instructional text
fig.text(0.5, 0.01, "Use the navigation toolbar to zoom and pan. Press 'h' to reset view.", 
         ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Enable tight layout
fig.tight_layout()

# Show the plot with block=True to keep window open
plt.show(block=True)

print("Visualization complete.")
