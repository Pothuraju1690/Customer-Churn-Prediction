
from pathlib import Path


import logging
import joblib
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
TARGET_COLUMN = "Churn"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load test data and trained model
# --------------------------------------------------

def load_test_data():
    X_test = pd.read_csv(DATA_DIR / "X_test_raw.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")[TARGET_COLUMN]

    return X_test, y_test


def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

def evaluate_model():
    X_test, y_test = load_test_data()
    model = load_model()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    # Save evaluation metrics.
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        REPORT_DIR / "test_metrics.csv",
        index=False,
    )

    # Save confusion matrix.
    matrix = confusion_matrix(y_test, predictions)

    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual_No", "Actual_Yes"],
        columns=["Predicted_No", "Predicted_Yes"],
    )

    confusion_df.to_csv(
        REPORT_DIR / "confusion_matrix.csv"
    )

    # Save classification report.
    report = classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    # Display results.
    logger.info("\nTest Set Evaluation")
    logger.info("------------------------------")

    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    logger.info("\nConfusion Matrix")
    logger.info(confusion_df)

    logger.info("\nClassification Report")
    logger.info(classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    ))

    logger.info("\nEvaluation files saved to:")
    logger.info(REPORT_DIR)


if __name__ == "__main__":
    evaluate_model()
