import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# Load dataset
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Stroke_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop("stroke", axis=1)
y = df["stroke"].astype(int)

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline Modelling
with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("F1 score:", f1)