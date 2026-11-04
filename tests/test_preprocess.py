import pandas as pd
import pytest

from src.data.preprocess import (
    load_and_clean_data,
    create_preprocessor,
)


@pytest.fixture
def sample_data(tmp_path):
    """Create a small raw dataset for testing."""

    data = {
        "customerID": ["C001", "C002", "C003", "C004", "C004"],
        "gender": ["Female", "Male", "Female", "Male", "Male"],
        "tenure": [1, 12, 24, 36, 36],
        "MonthlyCharges": [29.85, 56.95, 70.35, 89.10, 89.10],
        "TotalCharges": ["29.85", "684.40", "", "3204.00", "3204.00"],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "Month-to-month",
        ],
        "Churn": ["Yes", "No", "Yes", "No", "No"],
    }

    df = pd.DataFrame(data)
    file_path = tmp_path / "sample_dataset.csv"
    df.to_csv(file_path, index=False)

    return file_path


def test_load_and_clean_data_removes_customer_id(sample_data):
    df = load_and_clean_data(sample_data)

    assert "customerID" not in df.columns


def test_load_and_clean_data_converts_total_charges(sample_data):
    df = load_and_clean_data(sample_data)

    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert pd.isna(df.loc[df["tenure"] == 24, "TotalCharges"]).all()


def test_load_and_clean_data_maps_churn_to_binary(sample_data):
    df = load_and_clean_data(sample_data)

    assert set(df["Churn"].unique()).issubset({0, 1})
    assert df.loc[df["tenure"] == 1, "Churn"].iloc[0] == 1
    assert df.loc[df["tenure"] == 12, "Churn"].iloc[0] == 0


def test_load_and_clean_data_removes_duplicates(sample_data):
    df = load_and_clean_data(sample_data)

    assert len(df) == 4


def test_create_preprocessor_handles_missing_values_and_categories():
    X = pd.DataFrame({
        "tenure": [1, 12, None, 36],
        "MonthlyCharges": [29.85, 56.95, 70.35, None],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
        ],
    })

    preprocessor = create_preprocessor(X)

    X_processed = preprocessor.fit_transform(X)

    assert X_processed.shape[0] == 4
    assert X_processed.shape[1] > X.shape[1]
    assert not pd.isna(X_processed).any()


def test_preprocessor_ignores_unseen_categories():
    X_train = pd.DataFrame({
        "tenure": [1, 12, 24],
        "Contract": ["Month-to-month", "One year", "Two year"],
    })

    X_test = pd.DataFrame({
        "tenure": [6],
        "Contract": ["New contract type"],
    })

    preprocessor = create_preprocessor(X_train)
    preprocessor.fit(X_train)

    X_processed = preprocessor.transform(X_test)

    assert X_processed.shape[0] == 1
    assert X_processed.shape[1] == preprocessor.transform(X_train).shape[1]