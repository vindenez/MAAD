import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob

# Configuration
original_data_path = 'data/Austevoll_Autumn_2023_no_dcps copy.csv'
logs_directory = 'results/adapad_logs/'

# Feature configuration (only uncommented columns)
feature_columns = [
    "conductivity_conductivity", 
    #"conductivity_temperature", 
    #"conductivity_salinity",
    #"conductivity_density", 
    #"conductivity_soundspeed",
    "pressure_pressure",
    "pressure_temperature"
]

# Value range configuration for normalization (only uncommented columns)
value_range_config = {
    "conductivity_conductivity": (25.0, 38.0),
    #conductivity_temperature": (5.0, 17.0),
    #"conductivity_salinity": (18.0, 32.0),
    #"conductivity_density": (1008.0, 1030.0),
    #"conductivity_soundspeed": (1460.0, 1510.0),
    "pressure_pressure": (299.0, 321.0),
    "pressure_temperature": (5.0, 17.0)
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

# Load each log file and extract anomalies (only for uncommented columns)
for log_file in log_files:
    feature_name = os.path.basename(log_file).replace('_log.csv', '')
    
    # Only process uncommented columns
    if feature_name not in feature_columns:
        continue
    
    print(f"Processing {feature_name} from {log_file}")
    
    try:
        # Load the log file
        log_df = pd.read_csv(log_file)
        
        # Add timesteps
        log_df['timestep'] = np.arange(1, len(log_df) + 1)
        
        # Extract anomalies
        anomalies = log_df[log_df['anomalous'] == True]
        feature_anomalies[feature_name] = anomalies['timestep'].tolist() if not anomalies.empty else []
        
        print(f"  - Found {len(feature_anomalies[feature_name])} anomalies")
    except Exception as e:
        print(f"Error processing {log_file}: {e}")

# Create a figure for normalized values
plt.figure(figsize=(16, 8))

# Plot normalized original data for each uncommented feature
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

# Mark all anomalies with vertical lines (only for uncommented columns)
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
plt.title('Normalized Pressure Parameters with All Anomalies', fontsize=14)
plt.xlabel('Timestep', fontsize=12)
plt.ylabel('Normalized Value [0-1]', fontsize=12)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Tight layout and save
plt.tight_layout()
#plt.savefig('figures/sensor_parameters.png', dpi=300)
plt.show()

print("Visualization complete. Plot saved as 'all_features_normalized_with_anomalies.png'.")