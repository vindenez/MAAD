import pandas as pd

# Load the CSV file
input_file = 'data/Austevoll_Autumn_2023.csv'
output_file = 'data/Austevoll_Autumn_2023_no_dcps.csv'

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)

# Remove columns that start with 'dcps'
df = df.loc[:, ~df.columns.str.startswith('dcps')]

# Save the modified DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Removed DCPS columns. Saved to {output_file}")