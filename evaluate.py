import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report
)

REGISTERED_NAME = "iris_classifier"
DATA_PATH       = "data/iris.csv"
FEATURE_COLS    = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_model(version=None, stage=None):
    """Load model from MLflow registry by version or stage."""
    if stage:
        model_uri = f"models:/{REGISTERED_NAME}/{stage}"
        print(f"  Loading model: {model_uri}")
    elif version:
        model_uri = f"models:/{REGISTERED_NAME}/{version}"
        print(f"  Loading model: {model_uri}")
    else:
        # Load latest version
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
        latest = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{REGISTERED_NAME}/{latest.version}"
        print(f"  Loading latest model version {latest.version}: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)
    return model, model_uri


def main(version=None, stage=None):
    print("=== Evaluation via MLflow Model Registry ===\n")

    # ── 1. Load model from registry ───────────────────────────────────────────
    print("[1/4] Fetching model from MLflow registry ...")
    model, model_uri = load_model(version, stage)
    print(f"      Model type : {type(model).__name__}")
    print(f"      Params     : {model.get_params()}")

    # ── 2. Load evaluation data ───────────────────────────────────────────────
    print("\n[2/4] Loading evaluation data ...")
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURE_COLS]
    y  = df["species"]
    print(f"      Samples : {len(df)}")

    # ── 3. Run evaluation ─────────────────────────────────────────────────────
    print("\n[3/4] Running evaluation ...")
    y_pred    = model.predict(X)
    accuracy  = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted")
    recall    = recall_score(y, y_pred, average="weighted")
    f1        = f1_score(y, y_pred, average="weighted")

    print(f"\n      Accuracy  : {accuracy:.4f}")
    print(f"      Precision : {precision:.4f}")
    print(f"      Recall    : {recall:.4f}")
    print(f"      F1 Score  : {f1:.4f}")

    # ── 4. Log evaluation run to MLflow ──────────────────────────────────────
    print("\n[4/4] Logging evaluation run to MLflow ...")
    mlflow.set_experiment("IRIS_Evaluation")
    with mlflow.start_run(run_name="evaluation_run"):
        mlflow.log_param("model_uri",     model_uri)
        mlflow.log_param("model_type",    type(model).__name__)
        mlflow.log_metric("eval_accuracy",  round(accuracy,  4))
        mlflow.log_metric("eval_precision", round(precision, 4))
        mlflow.log_metric("eval_recall",    round(recall,    4))
        mlflow.log_metric("eval_f1",        round(f1,        4))

    print("\n=== Classification Report ===")
    print(classification_report(y, y_pred))

    print(f"\n✅ Evaluation complete!")
    print(f"   Model URI : {model_uri}")
    print(f"   Accuracy  : {accuracy:.4f}")
    print(f"   F1 Score  : {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None, help="Model version number")
    parser.add_argument("--stage",   default=None, help="Model stage (Staging/Production)")
    args = parser.parse_args()
    main(args.version, args.stage)
