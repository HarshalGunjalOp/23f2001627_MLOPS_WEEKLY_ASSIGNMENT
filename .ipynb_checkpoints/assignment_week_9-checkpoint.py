import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

np.random.seed(42)
location = pd.Series(np.random.choice([0, 1], size=len(X)), name='location')

X_train, X_test, y_train, y_test, loc_train, loc_test = train_test_split(
    X, y, location, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
y_pred = clf.predict(X_test)

metrics = {
    'accuracy': accuracy_score,
    'precision': lambda y_t, y_p: precision_score(y_t, y_p, average='macro', zero_division=0),
    'recall': lambda y_t, y_p: recall_score(y_t, y_p, average='macro', zero_division=0)
}

mf = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=loc_test)
print("--- Fairness Audit by Location ---")
print(mf.by_group)

explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_test)

print("\nGenerating SHAP summary plot for Virginica...")
shap.summary_plot(shap_values[:, :, 2], X_test, show=False)
plt.savefig("shap_virginica.png", bbox_inches='tight')
plt.clf()

X_drift = X_test.copy()
X_drift['petal length (cm)'] += 2.0 

print("\n--- Data Drift Detection ---")
print(f"Original Petal Length Mean: {X_test['petal length (cm)'].mean():.2f}")
print(f"Drifted Petal Length Mean:  {X_drift['petal length (cm)'].mean():.2f}")
