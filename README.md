# Week 1 MLOps Assignment: Vertex AI & GCS Pipeline

This repository contains the bare-bone ML pipeline using GCP components to train an IRIS classification model. 

## Files Included:
1. **`data_prep.py`**: Reads local `iris.csv`, splits it into train/eval sets (0.6/0.4), creates a GCS bucket, and uploads the datasets to the cloud.
2. **`train.py`**: Fetches `train.csv` from GCS, trains a Decision Tree model (`max_depth=3`), saves it locally as a `.joblib` file, and uploads the artifact to a dynamically generated timestamped folder (`artifacts/<timestamp>/model.joblib`) in GCS.
3. **`inference.py`**: Identifies the latest model execution timestamp, downloads the corresponding `model.joblib` and `eval.csv` from GCS, runs predictions, and prints the accuracy score.

## Execution Instructions:
Run the files in the following order:
```bash
python data_prep.py
python train.py
python inference.py
```