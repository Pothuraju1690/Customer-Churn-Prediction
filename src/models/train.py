
from pathlib import Path


import logging
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.data.preprocess import create_preprocessor


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42


# --------------------------------------------------
# Load training data
# --------------------------------------------------

def load_training_data():
    X = pd.read_csv(DATA_DIR / "X_train_raw.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET_COLUMN]
    return X, y


# --------------------------------------------------
# Candidate models
# --------------------------------------------------

def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, probabilities),
    }


# --------------------------------------------------
# Train, validate, select, and save
# --------------------------------------------------

def train_models():
    X, y = load_training_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Use SQLite for MLflow tracking.
    mlflow_db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db_path.as_posix()}"
    )
    mlflow.set_experiment("Customer_Churn_Prediction")

    results = []
    best_f1 = -1
    best_model_name = None
    best_classifier = None

    for model_name, classifier in get_models().items():
        preprocessor = create_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", classifier),
            ]
        )

        with mlflow.start_run(run_name=f"{model_name}_validation"):
            pipeline.fit(X_train, y_train)

            # Select models using validation metrics—not training metrics.
            metrics = calculate_metrics(pipeline, X_val, y_val)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(classifier.get_params())
            mlflow.log_metrics({
                f"validation_{key}": value
                for key, value in metrics.items()
            })

            logger.info(f"\nModel: {model_name} (validation)")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")

            results.append({
                "model": model_name,
                **{
                    f"validation_{key}": value
                    for key, value in metrics.items()
                },
            })

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model_name = model_name
                best_classifier = clone(classifier)

    # Retrain the selected model using the entire training split.
    final_preprocessor = create_preprocessor(X)
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(final_preprocessor)),
            ("classifier", best_classifier),
        ]
    )

    final_pipeline.fit(X, y)

    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(final_pipeline, model_path)

    pd.DataFrame(results).to_csv(
        PROJECT_ROOT / "artifacts" / "model_comparison.csv",
        index=False,
    )

    logger.info("\n------------------------------")
    logger.info(f"Selected model: {best_model_name}")
    logger.info(f"Validation F1-score: {best_f1:.4f}")
    logger.info(f"Saved final pipeline: {model_path}")
    logger.info("------------------------------")


if __name__ == "__main__":
    train_models()
