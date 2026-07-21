import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH        = "data/iris.csv"
EXPERIMENT_NAME  = "IRIS_Classification"
REGISTERED_NAME  = "iris_classifier"
FEATURE_COLS     = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# ── Hyperparameter Grid (Task 1) ──────────────────────────────────────────────
# Varying: max_depth, min_samples_split, criterion — all for DecisionTree
# Plus one RandomForest run for comparison
EXPERIMENTS = [
    {
        "run_name"         : "DT_depth3_gini",
        "model_type"       : "DecisionTree",
        "max_depth"        : 3,
        "min_samples_split": 2,
        "criterion"        : "gini",
    },
    {
        "run_name"         : "DT_depth5_gini",
        "model_type"       : "DecisionTree",
        "max_depth"        : 5,
        "min_samples_split": 2,
        "criterion"        : "gini",
    },
    {
        "run_name"         : "DT_depth3_entropy",
        "model_type"       : "DecisionTree",
        "max_depth"        : 3,
        "min_samples_split": 5,
        "criterion"        : "entropy",
    },
    {
        "run_name"         : "DT_depth7_entropy",
        "model_type"       : "DecisionTree",
        "max_depth"        : 7,
        "min_samples_split": 3,
        "criterion"        : "entropy",
    },
    {
        "run_name"         : "RF_100trees",
        "model_type"       : "RandomForest",
        "n_estimators"     : 100,
        "max_depth"        : 5,
        "min_samples_split": 2,
    },
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df["species"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def build_model(config):
    if config["model_type"] == "DecisionTree":
        return DecisionTreeClassifier(
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            criterion=config["criterion"],
            random_state=42,
        )
    else:
        return RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=42,
        )


def run_experiment(config, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=config["run_name"]) as run:
        # ── Log all hyperparameters ───────────────────────────────────────────
        mlflow.log_param("model_type",        config["model_type"])
        mlflow.log_param("max_depth",         config.get("max_depth"))
        mlflow.log_param("min_samples_split", config.get("min_samples_split"))
        if config["model_type"] == "DecisionTree":
            mlflow.log_param("criterion",     config.get("criterion"))
        if config["model_type"] == "RandomForest":
            mlflow.log_param("n_estimators",  config.get("n_estimators"))

        # ── Train ─────────────────────────────────────────────────────────────
        model = build_model(config)
        model.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred    = model.predict(X_test)
        y_pred_tr = model.predict(X_train)

        train_acc = accuracy_score(y_train, y_pred_tr)
        test_acc  = accuracy_score(y_test,  y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall    = recall_score(y_test,    y_pred, average="weighted")
        f1        = f1_score(y_test,        y_pred, average="weighted")

        # ── Log metrics ───────────────────────────────────────────────────────
        mlflow.log_metric("train_accuracy", round(train_acc, 4))
        mlflow.log_metric("test_accuracy",  round(test_acc,  4))
        mlflow.log_metric("precision",      round(precision, 4))
        mlflow.log_metric("recall",         round(recall,    4))
        mlflow.log_metric("f1_score",       round(f1,        4))

        # ── Log model to MLflow ───────────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_NAME,
        )

        print(f"  Run: {config['run_name']}")
        print(f"    Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {f1:.4f}")
        print(f"    Run ID: {run.info.run_id}")

        return run.info.run_id, test_acc, f1


def main():
    print("=== IRIS MLflow Experiment Tracking ===\n")

    # ── Set MLflow experiment ─────────────────────────────────────────────────
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Registry   : {REGISTERED_NAME}")
    print(f"Runs       : {len(EXPERIMENTS)}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # ── Run all experiments ───────────────────────────────────────────────────
    results = []
    for config in EXPERIMENTS:
        run_id, acc, f1 = run_experiment(
            config, X_train, X_test, y_train, y_test
        )
        results.append((config["run_name"], run_id, acc, f1))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Experiment Summary ===")
    print(f"{'Run Name':<25} {'Test Acc':>10} {'F1':>10}")
    print("-" * 50)
    best = max(results, key=lambda x: x[3])
    for name, rid, acc, f1 in sorted(results, key=lambda x: -x[3]):
        marker = " ← BEST" if name == best[0] else ""
        print(f"{name:<25} {acc:>10.4f} {f1:>10.4f}{marker}")

    print(f"\n✅ All {len(EXPERIMENTS)} experiments logged to MLflow!")
    print(f"   Best run : {best[0]} (F1={best[3]:.4f})")
    print(f"\nTo view UI: mlflow ui  (then open http://127.0.0.1:5000)")


if __name__ == "__main__":
    main()
