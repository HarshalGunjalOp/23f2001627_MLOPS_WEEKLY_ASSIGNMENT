import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
from google.cloud import storage
from datetime import datetime
import os
import pytz

# --- CONFIGURATION ---
PROJECT_ID = "project-0e0c3cc6-a540-41ce-8f5"             
BUCKET_NAME = "mlops-week-1-harshalgunjal"    

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

# 1. Fetch training data from GCS
print("Downloading training data from GCS...")
download_from_gcs(BUCKET_NAME, "data/train.csv", "train_local.csv")
train_data = pd.read_csv("train_local.csv")

X_train = train_data[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train_data['species']

# 2. Train model
print("Training Decision Tree model...")
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)

# 3. Save model locally
joblib.dump(mod_dt, "model.joblib")

# 4. Generate timestamp and upload to GCS
ist_timezone = pytz.timezone('Asia/Kolkata')
timestamp = datetime.now(ist_timezone).strftime("%Y%m%d_%H%M%S")
gcs_model_path = f"artifacts/{timestamp}/model.joblib"

print(f"Uploading model to gs://{BUCKET_NAME}/{gcs_model_path}...")
upload_to_gcs(BUCKET_NAME, "model.joblib", gcs_model_path)

# Save timestamp locally so the inference script knows which one to grab
with open("latest_run.txt", "w") as f:
    f.write(timestamp)

print(f"Training complete! Model stored under timestamp: {timestamp}")