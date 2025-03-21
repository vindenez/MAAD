import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the data and inspect column names
df = pd.read_csv('data/Austevoll_Autumn_2023_no_dcps.csv', skipinitialspace=True)
print("Columns in DataFrame:", df.columns.tolist())  # Debug: Print column names

# Strip column names and verify
df.columns = df.columns.str.strip()
print("Stripped columns:", df.columns.tolist())  # Debug: Print stripped column names

# Convert timestamp column
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Explicitly convert timestamp

# Create the plot
plt.figure(figsize=(12, 6))

# Plot both temperature parameters
plt.plot(df['timestamp'], df['conductivity_conductivity'], label='Conductivity', linestyle='--')
plt.plot(df['timestamp'], df['pressure'], label='Conductivity Temperature')
plt.plot(df['timestamp'], df['pressure_temperature'], label='Pressure Temperature', linestyle='--')


# Formatting
plt.title('Temperature Measurements - Austevoll Autumn 2023')
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

# Format x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate()

# Show plot
plt.tight_layout()
plt.show()