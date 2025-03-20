import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# File paths
original_data_path = 'data/Conductivity copy.csv'
anomaly_data_path = 'results/conductivity/att-lstm_0.01.csv'

# Configuration based on provided config file
feature_columns = [
    "conductivity_conductivity", 
    "conductivity_temperature", 
    "conductivity_salinity",
    "conductivity_density", 
    "conductivity_soundspeed"
]

# Value range configuration for normalization
value_range_config = {
    0: (25.0, 38.0),     # conductivity_conductivity
    1: (2.0, 20.0),      # conductivity_temperature
    2: (18.0, 32.0),     # conductivity_salinity
    3: (1008.0, 1030.0), # conductivity_density
    4: (1460.0, 1510.0)  # conductivity_soundspeed
}

# Load the datasets
original_df = pd.read_csv(original_data_path)
anomaly_df = pd.read_csv(anomaly_data_path)

print(f"Original dataset length: {len(original_df)}")
print(f"Anomaly dataset length: {len(anomaly_df)}")

# Clean the data: replace -999 values and handle NaN
for col in feature_columns:
    # Replace -999 values with NaN
    original_df[col] = original_df[col].replace(-999, np.nan)

# Simply add sequential timesteps to both datasets without trying to align them
original_df['timestep'] = np.arange(1, len(original_df) + 1)
anomaly_df['timestep'] = np.arange(1, len(anomaly_df) + 1)

# Check for anomalies in different ranges
anomalies = anomaly_df[anomaly_df['conductivity_conductivity_anomalous'] == True]
print(f"Total anomalies detected: {len(anomalies)}")
print(f"Min anomaly timestep: {anomalies['timestep'].min()}")
print(f"Max anomaly timestep: {anomalies['timestep'].max()}")

# Create a single figure
plt.figure(figsize=(16, 8))

# Normalize and plot each parameter
for i, col in enumerate(feature_columns):
    # Get valid data for this parameter (drop NaN values)
    valid_data = original_df.dropna(subset=[col])
    
    # Normalize using the config ranges
    min_val, max_val = value_range_config[i]
    
    # Proper normalization to [0,1] based on the config ranges
    normalized_values = np.clip((valid_data[col] - min_val) / (max_val - min_val), 0, 1)
    
    # Plot using timesteps on x-axis
    plt.plot(valid_data['timestep'], normalized_values, label=col, linewidth=1.5)

# Mark all anomalies with vertical lines
if not anomalies.empty:
    # Get current y-axis limits
    ylim = plt.gca().get_ylim()
    
    # Plot vertical lines for anomalies
    anomaly_timesteps = anomalies['timestep'].values
    
    for timestep in anomaly_timesteps:
        plt.axvline(x=timestep, color='red', linestyle='-', alpha=0.3, linewidth=2.5)
    
    # Reset the y-limits after adding vertical lines
    plt.gca().set_ylim(ylim)

    # Add text label for the number of anomalies
    plt.text(0.02, 0.02, f"Total anomalies: {len(anomalies)}", transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.7), fontsize=10)

# Add reference lines at 0 and 1 to show the normal range
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
plt.axhline(y=1, color='gray', linestyle='-', alpha=0.5, linewidth=1)

# Set y-axis limits explicitly to ensure [0,1] range is displayed
plt.ylim(0, 1)

# Create custom legend with anomaly marker
anomaly_line = Line2D([0], [0], color='red', linestyle='solid', linewidth=2.5, alpha=0.3)
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(anomaly_line)
labels.append('Anomalies')
plt.legend(handles=handles, labels=labels, loc='upper right', fontsize=10)

# Ensure x-axis spans the full dataset
plt.xlim(1, len(original_df))

# Add title and labels
plt.title('Normalized Conductivity Parameters with All Anomalies', fontsize=14)
plt.xlabel('Timestep', fontsize=12)
plt.ylabel('Normalized Value [0-1]', fontsize=12)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Tight layout and save
plt.tight_layout()
plt.savefig('graphs/multivariate_anomalies.png', dpi=300)

print("Visualization complete. All anomalies should now be properly plotted.")
print("The plot has been generated and saved as 'normalized_conductivity_with_all_anomalies.png'")

# Show the plot
plt.show()