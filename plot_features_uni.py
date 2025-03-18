import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# File path
log_data_path = 'results/adapad_logs/conductivity_conductivity_log.csv'

# Load the dataset
log_df = pd.read_csv(log_data_path)

# Add sequential timesteps
log_df['timestep'] = np.arange(1, len(log_df) + 1)

# Find anomalies
anomalies = log_df[log_df['anomalous'] == True]
print(f"Total anomalies detected: {len(anomalies)}")
if not anomalies.empty:
    print(f"Min anomaly timestep: {anomalies['timestep'].min()}")
    print(f"Max anomaly timestep: {anomalies['timestep'].max()}")

# Create a figure
plt.figure(figsize=(16, 8))

# Plot observed values
plt.plot(log_df['timestep'], log_df['observed'], label='Observed', linewidth=1.5, color='blue')

# Mark all anomalies with vertical lines
if not anomalies.empty:
    # Plot vertical lines for anomalies
    anomaly_timesteps = anomalies['timestep'].values
    
    for timestep in anomaly_timesteps:
        plt.axvline(x=timestep, color='red', linestyle='-', alpha=0.7)

    # Add text label for the number of anomalies
    plt.text(0.02, 0.02, f"Total anomalies: {len(anomalies)}", transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.7), fontsize=10)

# Create custom legend with anomaly marker
anomaly_line = Line2D([0], [0], color='red', linestyle='solid', linewidth=2)
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(anomaly_line)
labels.append('Anomalies')
plt.legend(handles=handles, labels=labels, loc='upper right', fontsize=10)

# Ensure x-axis spans the full dataset
plt.xlim(1, len(log_df))

# Set y-axis limits to (25.0, 36.0)
plt.ylim(25.0, 36.0)

# Add title and labels
plt.title('Conductivity Measurements with Anomaly Detection', fontsize=14)
plt.xlabel('Timestep', fontsize=12)
plt.ylabel('Conductivity Value', fontsize=12)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Tight layout and save
plt.tight_layout()
plt.savefig('conductivity_with_anomalies.png', dpi=300)

print("Visualization complete. Anomalies have been properly plotted.")
print("The plot has been generated and saved as 'conductivity_with_anomalies.png'")

# Show the plot
plt.show()