import pandas as pd

# Load the CSV
df = pd.read_csv('iris_data_adapted_for_feast.csv')

# Convert timestamps to proper datetime objects
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df['created_timestamp'] = pd.to_datetime(df['created_timestamp'])

# Save to the Feast data directory
df.to_parquet('iris_feature_repo/feature_repo/data/iris_dataset.parquet')
print("Data successfully saved to Parquet!")