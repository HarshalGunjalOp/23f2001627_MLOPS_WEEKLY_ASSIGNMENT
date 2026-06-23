import pandas as pd
import joblib
from google.cloud import storage
from sklearn import metrics
import sys
import os

# --- CONFIGURATION ---
PROJECT_ID = "project-0e0c3cc6-a540-41ce-8f5"             
BUCKET_NAME = "mlops-week-1-harshalgunjal"    

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

# 1. Get the latest timestamp folder
timestamp = None
if len(sys.argv) > 1:
    timestamp = sys.argv[1] # Allow passing timestamp manually via command line
elif os.path.exists("latest_run.txt"):
    with open("latest_run.txt", "r") as f:
        timestamp = f.read().strip()

if not timestamp:
    print("Error: Could not find latest_run.txt or timestamp argument.")
    sys.exit(1)

print(f"Starting inference using model from run: {timestamp}")

# 2. Fetch model and eval data from GCS
print("Downloading model and eval data from GCS...")
download_from_gcs(BUCKET_NAME, f"artifacts/{timestamp}/model.joblib", "model_local.joblib")
download_from_gcs(BUCKET_NAME, "data/eval.csv", "eval_local.csv")

# 3. Load data and model
test_data = pd.read_csv("eval_local.csv")
X_test = test_data[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test_data['species']

model = joblib.load("model_local.joblib")

# 4. Predict and evaluate
predictions = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)

print("\n===================================")
print(f" INFERENCE COMPLETE")
print(f" Model Timestamp: {timestamp}")
print(f" Model Accuracy:  {accuracy:.3f}")
print("===================================\n")