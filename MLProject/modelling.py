
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stroke prediction model with MLflow autolog (local tracking).")
    p.add_argument(
        "--data_path",
        type=str,
        default=str("Stroke_preprocessed.csv"),
        help="Path to preprocessed CSV produced by Eksperimen.ipynb.",
    )
    p.add_argument(
        "--target",
        type=str,
        default="stroke",
        help="Name of the target column (default: stroke).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "random_forest"],
        help="Which model to train (no tuning).",
    )
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split size. Default: 0.2",
    )
    p.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    p.add_argument(
        "--experiment_name",
        type=str,
        default="stroke-prediction",
        help="MLflow experiment name.",
    )
    p.add_argument(
        "--tracking_dir",
        type=str,
        default="mlruns",
        help="Local MLflow tracking directory. Default: ./mlruns",
    )
    return p.parse_args()


def build_model(name: str, random_state: int):
    # Single, sensible defaults — no hyperparameter search/tuning.
    if name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # handle class imbalance (stroke is typically rare)
            n_jobs=None,
            solver="lbfgs",
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {name}")


def main() -> int:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at: {data_path.resolve()}\n"
            "Make sure you already ran Eksperimen.ipynb and produced stroke_preprocessed.csv."
        )

    # Configure local tracking (file-based) so it is stored in the project folder.
    tracking_dir = Path(args.tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    mlflow.set_experiment(args.experiment_name)

    # Enable autologging for sklearn (must happen before fit).
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True,
        silent=True,
    )

    df = pd.read_csv(data_path)

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found. Available columns: {list(df.columns)}"
        )

    # Separate features and target
    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target])

    # Defensive conversion: if any non-numeric columns remain, fail clearly.
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(
            "Non-numeric columns still exist after preprocessing. "
            "Ensure Eksperimen.ipynb outputs fully numeric features. "
            f"Non-numeric columns: {non_numeric}"
        )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model = build_model(args.model, args.random_state)

    run_name = f"{args.model}-no-tuning"
    with mlflow.start_run(run_name=run_name):
        # Log a couple of dataset facts explicitly (autolog won't capture these).
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("n_rows", int(df.shape[0]))
        mlflow.log_param("n_features", int(X.shape[1]))
        mlflow.log_param("positive_rate", float(y.mean()))

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Probabilities for AUC if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_proba is not None and y_test.nunique() > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Confusion matrix + report as artifacts
        cm = confusion_matrix(y_test, y_pred)
        cm_path = Path("confusion_matrix.txt")
        cm_path.write_text(
            "Confusion Matrix (rows=true, cols=pred):\n"
            + np.array2string(cm)
            + "\n"
        )

        report_path = Path("classification_report.txt")
        report_path.write_text(classification_report(y_test, y_pred, digits=4, zero_division=0))

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(report_path))

        # Clean up local artifact files
        cm_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)

    print("Training complete.")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print("To view runs: mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
