
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "best_model.joblib"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a telecom customer is likely to churn.",
    version="1.0.0",
)


# --------------------------------------------------
# Load the trained pipeline
# --------------------------------------------------

model = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
except Exception as exc:
    print(f"Model loading error: {exc}")


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class CustomerData(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "model_loaded": model is not None,
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train the model first.",
        )

    try:
        # Convert validated customer data to a DataFrame.
        input_data = pd.DataFrame([customer.model_dump()])

        # Predict churn class and probability.
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        churn_probability = float(probabilities[1])

        return {
            "churn_prediction": prediction,
            "churn_label": "Yes" if prediction == 1 else "No",
            "churn_probability": round(churn_probability, 4),
            "no_churn_probability": round(float(probabilities[0]), 4),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
