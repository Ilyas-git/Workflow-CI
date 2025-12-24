import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn


def main():
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_PATH = BASE_DIR / "Stroke_preprocessing.csv"

    # Local MLflow tracking store (explicit)
    TRACKING_DIR = BASE_DIR / "mlruns_ci"
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_DIR.resolve().as_uri())

    # Optional: set experiment name (helps UI organization)
    mlflow.set_experiment("stroke-prediction")

    # Load dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    X = df.drop("stroke", axis=1)
    y = df["stroke"].astype(int)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Autolog (should be called before start_run & before fit)
    mlflow.sklearn.autolog(log_models=True)

    # Train (no hyperparameter tuning)
    with mlflow.start_run(run_name="logreg_baseline"):
        model = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log key metrics explicitly (autolog usually logs metrics too, but this is safe/clear)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        print("Tracking URI:", mlflow.get_tracking_uri())
        print("Accuracy:", acc)
        print("F1 score:", f1)


if __name__ == "__main__":
    main()
