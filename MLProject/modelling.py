import os
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn


def _set_tracking_uri(tracking_dir: Path) -> None:
    """
    Prioritas:
    1) Jika workflow/runner sudah set MLFLOW_TRACKING_URI, pakai itu (agar konsisten dengan mlflow run).
    2) Jika tidak ada, gunakan file store lokal dari tracking_dir.
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        mlflow.set_tracking_uri(env_uri)
        return

    tracking_dir = tracking_dir.expanduser().resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())


def main():
    # ---- Konfigurasi path (CSV di root folder MLProject) ----
    base_dir = Path(__file__).parent

    # CSV default sesuai keinginan kamu: tidak di folder preprocessing
    data_path = base_dir / "Stroke_preprocessing.csv"

    # Local tracking default (kalau bukan via workflow)
    tracking_dir = base_dir / "mlruns_ci"

    # ---- Set tracking (lokal/file-based) ----
    _set_tracking_uri(tracking_dir)

    # Experiment name (rapi di UI)
    mlflow.set_experiment("stroke-prediction")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    X = df.drop("stroke", axis=1)
    y = df["stroke"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Autolog (wajib, dan posisinya benar: sebelum start_run & fit) ----
    mlflow.sklearn.autolog(log_models=True)

    # Jika dijalankan lewat `mlflow run`, MLflow sudah membuat parent run dan set MLFLOW_RUN_ID.
    # Resume run itu agar tidak "Run not found".
    run_id = os.getenv("MLFLOW_RUN_ID")

    # ---- Training (tanpa hyperparameter tuning) ----
    with mlflow.start_run(run_id=run_id, run_name="logreg_baseline"):
        model = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Metric utama (autolog biasanya juga log, tapi ini memperjelas)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        print("Training complete.")
        print("Tracking URI:", mlflow.get_tracking_uri())
        print("Accuracy:", acc)
        print("F1:", f1)


if __name__ == "__main__":
    main()
