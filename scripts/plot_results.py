import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob
from collections import defaultdict

# Configuration
logs_directory = 'results/SeaGuard' 

source = 'SeaGuard'

# Feature configuration from config.py
feature_columns = {
    "Austevoll": [
        "conductivity_conductivity", 
        "pressure_pressure",
        "pressure_temperature"
    ],
    "SeaGuard": [
        "SeaGuard_Nord_Conductivity_Sensor.Conductivity",
        "SeaGuard_Nord_Conductivity_Sensor.Temperature",
        "SeaGuard_Nord_Pressure_Sensor.Pressure",
        "SeaGuard_Nord_Pressure_Sensor.Temperature",
        "SeaGuard_Sor_Conductivity_Sensor.Conductivity",
        "SeaGuard_Sor_Conductivity_Sensor.Temperature",
        "SeaGuard_Sor_Pressure_Sensor.Pressure",
        "SeaGuard_Sor_Pressure_Sensor.Temperature"
    ]
}

# Value range configuration for normalization
value_range_config = {
    "Austevoll": {
        "conductivity_conductivity": (25.0, 38.0),
        "pressure_pressure": (299.0, 321.0),
        "pressure_temperature": (5.0, 17.0)
    },
    "SeaGuard": {
        "SeaGuard_Nord_Conductivity_Sensor.Conductivity": (32.0, 38.0),
        "SeaGuard_Nord_Conductivity_Sensor.Temperature": (5.0, 12.0),
        "SeaGuard_Nord_Pressure_Sensor.Pressure": (315.0, 355.0),
        "SeaGuard_Nord_Pressure_Sensor.Temperature": (5.0, 12.0),
        "SeaGuard_Sor_Conductivity_Sensor.Conductivity": (32.0, 38.0),
        "SeaGuard_Sor_Conductivity_Sensor.Temperature": (5.0, 12.0),
        "SeaGuard_Sor_Pressure_Sensor.Pressure": (710.0, 760.0),
        "SeaGuard_Sor_Pressure_Sensor.Temperature": (5.0, 12.0)
    }
}

# Dictionary to store anomalies for each feature
feature_anomalies = {}

# Create a mapping of feature to its index for offset calculation
feature_indices = {}
for i, feature in enumerate(feature_columns[source]):
    feature_indices[feature] = i

# Check if system log file exists
system_log_path = os.path.join(logs_directory, 'system_log.csv')
if os.path.exists(system_log_path):
    system_log_df = pd.read_csv(system_log_path)
    # Extract anomalies from system log (where is_anomalous is True)
    anomaly_indices = system_log_df[system_log_df['is_anomalous'] == True]['idx'].tolist()
    print(f"Found {len(anomaly_indices)} system-level anomalies")
    
    # When using system log, don't populate feature_anomalies
    # as we'll only display system-level anomalies
    feature_anomalies = {}
else:
    # If no system log, collect anomalies from individual feature logs
    anomaly_indices = []
    
    for feature in feature_columns[source]:
        # Create a shorter filename for the log file
        short_name = feature.replace("SeaGuard_", "").replace("_Sensor", "").replace(".", "_")
        feature_log_path = os.path.join(logs_directory, f"{short_name}_log.csv")
        if os.path.exists(feature_log_path):
            log_df = pd.read_csv(feature_log_path)
            # Extract anomalies
            feature_anomalies[feature] = log_df[log_df['is_anomalous'] == True]['idx'].tolist()
            anomaly_indices.extend(feature_anomalies[feature])
    
    # Remove duplicates
    anomaly_indices = sorted(set(anomaly_indices))
    print(f"Found {len(anomaly_indices)} total anomalies from feature logs")

# Create a figure for normalized observed values
# Use a standard style that works across matplotlib versions
plt.style.use('default')  # Reset to default style
fig, ax = plt.subplots(figsize=(16, 8))
ax.grid(True, linestyle='--', alpha=0.7)  # Add grid manually

# Dictionary to store max index for each feature
max_idx = 0

# Define colors for different features
feature_colors = {
    "Austevoll": {
        "conductivity_conductivity": "blue",
        "pressure_pressure": "green",
        "pressure_temperature": "purple"
    },
    "SeaGuard": {
        "SeaGuard_Nord_Conductivity_Sensor.Conductivity": "blue",
        "SeaGuard_Nord_Conductivity_Sensor.Temperature": "red",
        "SeaGuard_Nord_Pressure_Sensor.Pressure": "green",
        "SeaGuard_Nord_Pressure_Sensor.Temperature": "purple",
        "SeaGuard_Sor_Conductivity_Sensor.Conductivity": "cyan",
        "SeaGuard_Sor_Conductivity_Sensor.Temperature": "orange",
        "SeaGuard_Sor_Pressure_Sensor.Pressure": "lime",
        "SeaGuard_Sor_Pressure_Sensor.Temperature": "magenta",
    }
}

# Create a dictionary to count anomalies per timestep
timestep_anomaly_counts = defaultdict(int)

# Load observed values from log files and plot normalized values
for feature in feature_columns[source]:
    # Create a shorter filename for the log file
    short_name = feature.replace("SeaGuard_", "").replace("_Sensor", "").replace(".", "_")
    feature_log_path = os.path.join(logs_directory, f"{short_name}_log.csv")
    
    if not os.path.exists(feature_log_path):
        print(f"Log file for {feature} not found: {feature_log_path}")
        continue
    
    # Load the log file
    log_df = pd.read_csv(feature_log_path)
    
    # Get observed values and normalize
    min_val, max_val = value_range_config[source][feature]
    
    # Normalize the observed values to [0, 1] range
    normalized_values = (log_df['observed'] - min_val) / (max_val - min_val)
    normalized_values = np.clip(normalized_values, 0, 1)
    
    # Plot the normalized values
    ax.plot(log_df['idx'], normalized_values, label=feature, linewidth=1.5, color=feature_colors[source][feature])
    
    # Update max index
    max_idx = max(max_idx, log_df['idx'].max())
    
    # Count anomalies per timestep
    if feature in feature_anomalies:
        for idx in feature_anomalies[feature]:
            timestep_anomaly_counts[idx] += 1

# Define the offset parameters
max_offset = 0.5  # Maximum offset to apply
min_spacing = 0.4  # Minimum spacing between lines at the same timestep

# Mark anomalies with vertical lines depending on whether we're using system log or feature logs
if os.path.exists(system_log_path):
    # Plot vertical lines for system anomalies without offset - thicker and more transparent
    for idx in anomaly_indices:
        ax.axvline(x=idx, color='black', linestyle='-', alpha=0.5, linewidth=3.5)
else:
    # Mark feature anomalies with vertical lines using feature-specific colors and offsets
    for feature in feature_columns[source]:
        if feature in feature_anomalies:
            # Calculate offset based on feature index and total anomalies at each timestep
            feature_idx = feature_indices[feature]
            
            for idx in feature_anomalies[feature]:
                total_anomalies = timestep_anomaly_counts[idx]
                
                # Skip if no anomalies (shouldn't happen)
                if total_anomalies == 0:
                    continue
                    
                # Calculate offset: distribute anomalies evenly
                if total_anomalies > 1:
                    # Space anomalies evenly across a range of -max_offset to +max_offset
                    position = feature_idx % total_anomalies  # Ensure position is within range
                    offset = -max_offset + (2 * max_offset) * (position / (total_anomalies - 1))
                else:
                    offset = 0  # No offset needed if only one anomaly at this timestep
                    
                # Apply the offset to the x-position
                ax.axvline(x=idx + offset, color=feature_colors[source][feature], 
                          linestyle='-', alpha=0.6, linewidth=2.0)

# Add text label for the number of anomalies
ax.text(0.02, 0.02, f"Total anomalies: {len(anomaly_indices)}", transform=ax.transAxes, 
         bbox=dict(facecolor='white', alpha=0.7), fontsize=10)

# Create custom legend with anomaly markers
legend_handles = []
legend_labels = []

# Add feature lines to legend
for feature in feature_columns[source]:
    legend_handles.append(Line2D([0], [0], color=feature_colors[source][feature], linewidth=1.5))
    # Create more readable label by removing prefixes
    readable_label = feature.replace("SeaGuard_", "").replace("_Sensor", "")
    legend_labels.append(readable_label)

# Add appropriate anomaly markers to legend based on what we're using
if os.path.exists(system_log_path):
    # Add system anomalies to legend - thicker and more transparent to match
    legend_handles.append(Line2D([0], [0], color='black', linestyle='solid', linewidth=3.5, alpha=0.5))
    legend_labels.append('System Anomalies')
else:
    # Add feature anomaly markers to legend
    for feature in feature_columns[source]:
        if feature in feature_anomalies and feature_anomalies[feature]:  # Only add if anomalies exist
            legend_handles.append(Line2D([0], [0], color=feature_colors[source][feature], 
                                        linestyle='solid', linewidth=2.0, alpha=0.6))
            # Create more readable label
            readable_label = feature.replace("SeaGuard_", "").replace("_Sensor", "")
            legend_labels.append(f'{readable_label} anomalies')

# Create a more organized legend with columns
ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', 
          fontsize=9, ncol=2, framealpha=0.8)

# Set y-axis limits to the normalized range with a small margin
ax.set_ylim(-0.05, 1.05)

# Ensure x-axis spans the full dataset with a margin for offset anomaly markers
min_x = min(16, 1)  # Use 16 as minimum if available (first log entry)
ax.set_xlim(min_x - 2, max_idx + 2)

# Add reference lines at 0 and 1 to show the normalized range
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
ax.axhline(y=1, color='gray', linestyle='-', alpha=0.5, linewidth=1)

# Add title and labels
ax.set_title('Normalized Sensor Values with Non-Overlapping Anomaly Markers', fontsize=14)
ax.set_xlabel('Timestep', fontsize=12)
ax.set_ylabel('Normalized Value [0-1]', fontsize=12)

# Add grid for better readability (already added above)

# Set background color to white for better contrast
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Add instructional text
fig.text(0.5, 0.01, "Use the navigation toolbar to zoom and pan. Press 'h' to reset view.", 
         ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Enable tight layout
fig.tight_layout()

# Create the output directory if it doesn't exist
output_dir = 'visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Show the plot with block=True to keep window open
plt.show(block=True)

print("Visualization complete.")