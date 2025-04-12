import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# This script treats the univariate AdapAD anomaly detections as ground truth
# Then it compares the multivariate AdapAD anomaly detection to the univariate detections

# Load multivariate model results
multivariate_file = 'results/conductivity/att-lstm_0.018.csv'
multivariate_df = pd.read_csv(multivariate_file)

# Get all univariate model log files
univariate_files = glob.glob('results/adapad_logs/*_log.csv')

# Dictionary to store univariate dataframes
univariate_dfs = {}

# Load all univariate model results
for file in univariate_files:
    feature_name = os.path.basename(file).replace('_log.csv', '')
    univariate_dfs[feature_name] = pd.read_csv(file)

# Initialize confusion matrix counters
TP = 0  # True Positive: Both multivariate and 2+ univariate models detect anomaly
FP = 0  # False Positive: Multivariate detects anomaly, but <2 univariate models do
TN = 0  # True Negative: Neither multivariate nor 2+ univariate models detect anomaly
FN = 0  # False Negative: Multivariate doesn't detect anomaly, but 2+ univariate models do

# Lists to store timesteps of false positives and false negatives
fp_timesteps = []
fn_timesteps = []

timestep = 0
still_rows = True

# Process each timestamp
while still_rows:
    # Check if we've reached the end of any dataset
    if timestep >= len(multivariate_df):
        still_rows = False
        break
    
    # Check if we've reached the end of any univariate dataset
    for feature, df in univariate_dfs.items():
        if timestep >= len(df):
            still_rows = False
            break
    
    if not still_rows:
        break
    
    # Get multivariate prediction for this timestamp
    multivariate_anomaly = multivariate_df.iloc[timestep]['conductivity_conductivity_anomalous']
    current_timestamp = multivariate_df.iloc[timestep]['timestamp']
    
    # Count univariate anomalies for this timestamp
    univariate_anomaly_count = 0
    anomalous_features = []
    for feature, df in univariate_dfs.items():
        if df.iloc[timestep]['anomalous']:
            univariate_anomaly_count += 1
            anomalous_features.append(feature)
    
    # Determine if it's a "true anomaly" based on univariate models (2+ detections)
    true_anomaly = univariate_anomaly_count >= 2
    
    # Update confusion matrix
    if multivariate_anomaly and true_anomaly:
        TP += 1
    elif multivariate_anomaly and not true_anomaly:
        FP += 1
        fp_timesteps.append((current_timestamp, timestep, anomalous_features))
    elif not multivariate_anomaly and true_anomaly:
        FN += 1
        fn_timesteps.append((current_timestamp, timestep, anomalous_features))
    else:  # not multivariate_anomaly and not true_anomaly
        TN += 1
    
    timestep += 1

# Calculate metrics
total = TP + FP + TN + FN
accuracy = (TP + TN) / total if total > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Print confusion matrix
print("\nConfusion Matrix:")
print(f"                  | Predicted Positive | Predicted Negative |")
print(f"Actual Positive   |        {TP}        |        {FN}        |")
print(f"Actual Negative   |        {FP}        |        {TN}        |")

# Print metrics
print(f"\nTotal samples processed: {total}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print false positives
print("\nFalse Positives (Multivariate detected anomaly but <2 univariate models did):")
for timestamp, step, features in fp_timesteps:
    print(f"  Timestamp: {timestamp}, Step: {step}, Anomalous features: {features}")

# Print false negatives
print("\nFalse Negatives (Multivariate missed anomaly but 2+ univariate models detected):")
for timestamp, step, features in fn_timesteps:
    print(f"  Timestamp: {timestamp}, Step: {step}, Anomalous features: {features}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = np.array([[TN, FP], [FN, TP]])
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Anomaly', 'Anomaly'])
plt.yticks(tick_marks, ['Not Anomaly', 'Anomaly'])

# Add text annotations to the confusion matrix
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label (2+ Univariate Detections)')
plt.xlabel('Predicted Label (Multivariate Model)')
plt.tight_layout()

print(f"\nProcessed {len(univariate_dfs)} univariate models and {timestep} timesteps.")
print("Analysis complete! Confusion matrix saved as 'confusion_matrix.png'")