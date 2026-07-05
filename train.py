import pandas as pd
from feast import FeatureStore
from sklearn.tree import DecisionTreeClassifier
import joblib

# Connect to the Feast Feature Store
store = FeatureStore(repo_path="iris_feature_repo/feature_repo")

# Define the entity dataframe (IDs and timestamps we want to train on)
raw_data = pd.read_parquet("iris_feature_repo/feature_repo/data/iris_dataset.parquet")
entity_df = raw_data[['iris_id', 'event_timestamp', 'species']]

# Fetch Historical Features (Offline Store)
print("Fetching historical features...")
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ]
).to_df()

# Train the Model
print("Training model...")
X = training_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = training_df['species']

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "model.joblib")
print("Model trained and saved!")