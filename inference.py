from feast import FeatureStore
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore") # Hides standard sklearn/numpy warnings

# Connect to the Feature Store
store = FeatureStore(repo_path="iris_feature_repo/feature_repo")

# Load the trained model
model = joblib.load("model.joblib")

# Simulate an inference request for a specific entity ID
entity_rows = [{"iris_id": 1001}]

# Fetch Online Features (Real-Time)
online_features = store.get_online_features(
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
    entity_rows=entity_rows
).to_dict()

# Convert to DataFrame
features_df = pd.DataFrame.from_dict(online_features)
features_df = features_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Predict
prediction = model.predict(features_df)

print(f"\nFeatures retrieved from Online Store for Iris 1001: \n{features_df}")
print(f"Prediction: {prediction[0]}")