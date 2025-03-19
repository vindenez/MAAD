import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob

# Configuration
original_data_path = 'data/Conductivity copy.csv'
logs_directory = 'results/adapad_logs/'

# Feature configuration
feature_columns = [
    "conductivity_conductivity", 
    "conductivity_temperature", 
    "conductivity_salinity",
    "conductivity_density", 
    "conductivity_soundspeed"
]

# Value range configuration for normalization
value_range_config = {
    "conductivity_conductivity": (25.0, 38.0),
    "conductivity_temperature": (2.0, 20.0),
    "conductivity_salinity": (18.0, 32.0),
    "conductivity_density": (1008.0, 1030.0),
    "conductivity_soundspeed": (1460.0, 1510.0)
}

# Load the original dataset
original_df = pd.read_csv(original_data_path)
print(f"Original dataset length: {len(original_df)}")

# Clean the data: replace -999 values and handle NaN
for col in feature_columns:
    if col in original_df.columns:
        # Replace -999 values with NaN
        original_df[col] = original_df[col].replace(-999, np.nan)

# Add sequential timesteps to original dataset
original_df['timestep'] = np.arange(1, len(original_df) + 1)

# Find all log files
log_files = glob.glob(os.path.join(logs_directory, '*.csv'))
print(f"Found {len(log_files)} log files: {log_files}")

# Create a dictionary to store anomalies for each feature
feature_anomalies = {}
feature_predictions = {}

# Load each log file and extract anomalies
for log_file in log_files:
    feature_name = os.path.basename(log_file).replace('_log.csv', '')
    print(f"Processing {feature_name} from {log_file}")
    
    try:
        # Load the log file
        log_df = pd.read_csv(log_file)
        
        # Add timesteps
        log_df['timestep'] = np.arange(1, len(log_df) + 1)
        
        # Extract anomalies
        anomalies = log_df[log_df['anomalous'] == True]
        feature_anomalies[feature_name] = anomalies['timestep'].tolist() if not anomalies.empty else []
        
        # Store predictions for plotting
        feature_predictions[feature_name] = {
            'observed': log_df['observed'].values,
            'predicted': log_df['predicted'].values,
            'low': log_df['low'].values,
            'high': log_df['high'].values,
            'timesteps': log_df['timestep'].values
        }
        
        print(f"  - Found {len(feature_anomalies[feature_name])} anomalies")
    except Exception as e:
        print(f"Error processing {log_file}: {e}")

# Create a figure for normalized values
plt.figure(figsize=(16, 8))

# Plot normalized original data for each feature
for feature in feature_columns:
    if feature in original_df.columns:
        # Get valid data for this parameter (drop NaN values)
        valid_data = original_df.dropna(subset=[feature])
        
        # Normalize using the config ranges
        min_val, max_val = value_range_config.get(feature, (0, 1))
        
        # Proper normalization to [0,1] based on the config ranges
        normalized_values = np.clip((valid_data[feature] - min_val) / (max_val - min_val), 0, 1)
        
        # Plot using timesteps on x-axis
        plt.plot(valid_data['timestep'], normalized_values, label=feature, linewidth=1.5)

# Mark all anomalies with vertical lines
all_anomaly_timesteps = []
for feature, anomalies in feature_anomalies.items():
    all_anomaly_timesteps.extend(anomalies)

# Remove duplicates and sort
all_anomaly_timesteps = sorted(set(all_anomaly_timesteps))

# Get current y-axis limits
ylim = plt.gca().get_ylim()

# Plot vertical lines for anomalies
for timestep in all_anomaly_timesteps:
    plt.axvline(x=timestep, color='red', linestyle='-', alpha=0.3, linewidth=2.5)

# Reset the y-limits after adding vertical lines
plt.gca().set_ylim(ylim)

# Add text label for the number of anomalies
plt.text(0.02, 0.02, f"Total anomalies: {len(all_anomaly_timesteps)}", transform=plt.gca().transAxes, 
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
plt.savefig('all_features_normalized_with_anomalies.png', dpi=300)

# Create individual plots for each feature showing observed vs predicted values
for feature_name, data in feature_predictions.items():
    plt.figure(figsize=(16, 6))
    
    # Get the original feature name from the log file name
    original_feature = None
    for col in feature_columns:
        if col in feature_name:
            original_feature = col
            break
    
    if original_feature is None:
        print(f"Could not determine original feature for {feature_name}, skipping...")
        continue
        
    # Get normalization range
    min_val, max_val = value_range_config.get(original_feature, (0, 1))
    
    # Plot observed values
    plt.plot(data['timesteps'], data['observed'], 'b-', label='Observed', linewidth=1.5)
    
    # Plot predicted values
    plt.plot(data['timesteps'], data['predicted'], 'g-', label='Predicted', linewidth=1.5)
    
    # Plot prediction bounds
    plt.fill_between(data['timesteps'], data['low'], data['high'], color='g', alpha=0.2, label='Prediction Bounds')
    
    # Mark anomalies
    if feature_name in feature_anomalies:
        for timestep in feature_anomalies[feature_name]:
            plt.axvline(x=timestep, color='red', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # Add title and labels
    plt.title(f'{original_feature} - Observed vs Predicted Values', fontsize=14)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    
    # Add legend
    anomaly_line = Line2D([0], [0], color='red', linestyle='solid', linewidth=1.5, alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(anomaly_line)
    labels.append('Anomalies')
    plt.legend(handles=handles, labels=labels, loc='best', fontsize=10)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{feature_name}_observed_vs_predicted.png', dpi=300)
    plt.close()

plt.show()

print("Visualization complete. All plots have been generated.")