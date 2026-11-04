import pytest
from fastapi.testclient import TestClient

import api.main as api_module



client = TestClient(api_module.app)


@pytest.fixture
def valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2,
    }


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Customer Churn Prediction API is running"
    )
    assert "model_loaded" in response.json()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "model_loaded" in response.json()


def test_predict_endpoint_returns_expected_fields(valid_customer):
    response = client.post("/predict", json=valid_customer)

    # If the model artifact is unavailable, the API correctly returns 503.
    if api_module.model is None:
        assert response.status_code == 503
        return

    assert response.status_code == 200

    result = response.json()

    assert result["churn_prediction"] in [0, 1]
    assert result["churn_label"] in ["Yes", "No"]
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["no_churn_probability"] <= 1

    assert abs(
        result["churn_probability"]
        + result["no_churn_probability"]
        - 1
    ) <= 0.001


def test_predict_endpoint_rejects_invalid_gender(valid_customer):
    valid_customer["gender"] = "Other"

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_tenure(valid_customer):
    valid_customer["tenure"] = 150

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422


def test_predict_endpoint_rejects_negative_monthly_charges(valid_customer):
    valid_customer["MonthlyCharges"] = -10

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422