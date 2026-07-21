import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "IRIS_Classification"
REGISTERED_NAME = "iris_classifier"


def main():
    print("=== Promoting Best Model to Production ===\n")
    client = MlflowClient()

    # ── Find best run by test_accuracy ───────────────────────────────────────
    print("[1/3] Finding best experiment run ...")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )

    if not runs:
        print("No runs found! Run train_mlflow.py first.")
        return

    best_run = runs[0]
    print(f"      Best run  : {best_run.info.run_name}")
    print(f"      Run ID    : {best_run.info.run_id}")
    print(f"      F1 Score  : {best_run.data.metrics.get('f1_score', 'N/A')}")
    print(f"      Test Acc  : {best_run.data.metrics.get('test_accuracy', 'N/A')}")

    # ── Find corresponding model version ────────────────────────────────────
    print(f"\n[2/3] Finding model version for this run ...")
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    matched  = [v for v in versions if v.run_id == best_run.info.run_id]

    if not matched:
        print("No model version found for best run!")
        return

    best_version = matched[0].version
    print(f"      Model version : {best_version}")

    # ── Promote to Production ─────────────────────────────────────────────────
    print(f"\n[3/3] Promoting version {best_version} to Production ...")
    client.transition_model_version_stage(
        name=REGISTERED_NAME,
        version=best_version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"      ✅ Version {best_version} is now in Production!")

    print(f"\n✅ Promotion complete!")
    print(f"   Model     : {REGISTERED_NAME}")
    print(f"   Version   : {best_version}")
    print(f"   Stage     : Production")
    print(f"\nNow run: python evaluate_mlflow.py --stage Production")


if __name__ == "__main__":
    main()
