from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "model"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# --------------------------------------------------
# Load and clean raw data
# --------------------------------------------------

def load_and_clean_data(data_path: Path = RAW_DATA_PATH):
    """Load the raw dataset and perform basic data cleaning."""

    df = pd.read_csv(data_path)

    # Convert TotalCharges to numeric.
    # Blank or invalid values become NaN and are imputed later.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove duplicate records.
    df = df.drop_duplicates()

    # Remove customer identifier because it is not a predictive feature.
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Convert target labels to numeric values.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1
    })

    # Remove rows where target is missing or invalid.
    df = df.dropna(subset=[TARGET_COLUMN])

    return df


# --------------------------------------------------
# Create preprocessing pipeline
# --------------------------------------------------

def create_preprocessor(X: pd.DataFrame):
    """Create pipelines for numerical and categorical features."""

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns)
        ],
        remainder="drop"
    )

    return preprocessor


# --------------------------------------------------
# Split data and preprocess
# --------------------------------------------------

def preprocess_data():
    """Split, preprocess, and save the churn dataset."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    # Split before fitting preprocessing to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Save raw split data for later model training and evaluation.
    X_train.to_csv(PROCESSED_DIR / "X_train_raw.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test_raw.csv", index=False)

    y_train.to_frame(name=TARGET_COLUMN).to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False
    )

    y_test.to_frame(name=TARGET_COLUMN).to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False
    )

    # Fit preprocessing ONLY on training data.
    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get names of transformed features.
    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names
    )

    # Save processed feature matrices.
    X_train_processed.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False
    )

    X_test_processed.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False
    )

    # Save the fitted preprocessing pipeline.
    joblib.dump(
        preprocessor,
        ARTIFACT_DIR / "preprocessor.joblib"
    )

    print("Preprocessing completed successfully!")
    print(f"Training rows: {X_train_processed.shape[0]}")
    print(f"Testing rows: {X_test_processed.shape[0]}")
    print(f"Processed feature count: {X_train_processed.shape[1]}")
    print(f"Training churn rate: {y_train.mean():.3f}")
    print(f"Testing churn rate: {y_test.mean():.3f}")
    print(f"Processed files saved to: {PROCESSED_DIR}")
    print(f"Preprocessor saved to: {ARTIFACT_DIR}")


# --------------------------------------------------
# Main entry point
# --------------------------------------------------

if __name__ == "__main__":
    preprocess_data()