import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage
import os

# --- CONFIGURATION ---
PROJECT_ID = "project-0e0c3cc6-a540-41ce-8f5"            
BUCKET_NAME = "mlops-week-1-harshalgunjal"   
LOCAL_DATA_PATH = "data/iris.csv"

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File uploaded to gs://{bucket_name}/{destination_blob_name}")

# 1. Load data
print("Loading local data...")
data = pd.read_csv(LOCAL_DATA_PATH)

# 2. Split data
print("Splitting data into train and eval sets...")
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)

# 3. Save locally temporarily
train.to_csv("train.csv", index=False)
test.to_csv("eval.csv", index=False)

# 4. Create bucket if it doesn't exist
storage_client = storage.Client(project=PROJECT_ID)
try:
    bucket = storage_client.get_bucket(BUCKET_NAME)
except:
    print(f"Creating bucket {BUCKET_NAME}...")
    bucket = storage_client.create_bucket(BUCKET_NAME, location="US")

# 5. Upload original and splits to GCS
print("Uploading files to GCS...")
upload_to_gcs(BUCKET_NAME, LOCAL_DATA_PATH, "data/iris.csv")
upload_to_gcs(BUCKET_NAME, "train.csv", "data/train.csv")
upload_to_gcs(BUCKET_NAME, "eval.csv", "data/eval.csv")

print("Data preparation complete!")