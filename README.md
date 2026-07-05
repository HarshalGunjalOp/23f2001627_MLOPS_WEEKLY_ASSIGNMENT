# Week 3 MLOps Assignment: Feast Feature Store Integration

This repository contains the Week 3 assignment for integrating a Feast feature store into an ML pipeline to solve training-serving skew.

## File Utility

*   **`prep_data.py`**: A script used to convert the raw time-aware CSV dataset into a Parquet format required by Feast.
*   **`iris_feature_repo/feature_repo/features.py`**: Contains the Feast definitions, including the data source, the entity (`iris_id`), and the Feature View with a 10-year TTL to prevent data staleness.
*   **`iris_feature_repo/feature_repo/feature_store.yaml`**: The configuration file that tells Feast to use a local provider and an SQLite database for the online store.
*   **`train.py`**: The model training script. It connects to the Feast offline store to fetch historical, point-in-time correct features for training a Decision Tree classifier, preventing data leakage.
*   **`inference.py`**: The real-time serving script. It connects to the Feast online store (SQLite) to fetch the latest feature values for a specific entity and makes a prediction using the trained model.

## Execution Steps
1. Navigate to the Feast directory: `cd iris_feature_repo/feature_repo`
2. Apply definitions: `feast apply`
3. Materialize the online store: `feast materialize 2010-01-01T00:00:00 2030-01-01T00:00:00`
4. Return to the root directory.
5. Train the model: `python train.py`
6. Run inference: `python inference.py`