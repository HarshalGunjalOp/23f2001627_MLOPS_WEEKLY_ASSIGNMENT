import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def poison_data(X, y, poison_rate):
    """Replaces a percentage of data with random values and random labels."""
    if poison_rate == 0:
        return X, y
        
    n_samples = X.shape[0]
    n_poison = int(n_samples * poison_rate)
    
    # Pick random indices to poison
    poison_indices = np.random.choice(n_samples, n_poison, replace=False)
    
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    
    # Generate random noise for features (using typical IRIS dataset bounds)
    for idx in poison_indices:
        X_poisoned[idx] = [
            np.random.uniform(4.0, 8.0), # sepal length
            np.random.uniform(2.0, 4.5), # sepal width
            np.random.uniform(1.0, 7.0), # petal length
            np.random.uniform(0.1, 2.5)  # petal width
        ]
        # Assign random label (0, 1, or 2)
        y_poisoned[idx] = np.random.randint(0, 3)
        
    return X_poisoned, y_poisoned

def run_experiment(poison_rate):
    # Load Data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Apply Data Poisoning
    X_poisoned, y_poisoned = poison_data(X, y, poison_rate)
    
    # Split Data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_poisoned, y_poisoned, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"Poison_Rate_{int(poison_rate*100)}%"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        # Log params and metrics
        mlflow.log_param("poison_rate", poison_rate)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        print(f"Run completed for {int(poison_rate*100)}% poisoning: Accuracy = {accuracy:.4f}")

if __name__ == "__main__":
    mlflow.set_experiment("IRIS_MLSecOps_Poisoning")
    
    # Run for 0%, 5%, 10%, and 50% corruption levels
    rates = [0.0, 0.05, 0.10, 0.50]
    for rate in rates:
        run_experiment(rate)
