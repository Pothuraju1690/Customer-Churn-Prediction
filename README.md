

# Customer Churn Prediction – End-to-End MLOps Project

An end-to-end Machine Learning application that predicts customer churn risk using a trained ML pipeline, FastAPI prediction service, and Streamlit dashboard. The project is containerized with Docker and deployed on AWS EC2 using Amazon Elastic Container Registry (ECR).

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Predicting churn risk helps businesses identify customers who may leave and take proactive retention measures.

This project provides a web application where users can enter customer details and receive a churn prediction along with the estimated churn probability.

### Key Features

- Data preprocessing and feature preparation
- Machine Learning model training and evaluation
- Saved model pipeline using Joblib
- REST API for predictions using FastAPI
- Interactive dashboard built with Streamlit
- FastAPI-first prediction with local model fallback
- Dockerized API and frontend services
- Docker networking between containers
- AWS ECR image storage
- AWS EC2 deployment
- API documentation through Swagger UI

## Architecture

```text
                 User
                  |
                  v
        Streamlit Web Application
             Port 8501
                  |
          Prediction Request
                  |
                  v
          FastAPI Prediction API
             Port 8000
                  |
                  v
       Saved ML Pipeline / Model
                  |
                  v
       Churn Prediction + Probability
                  |
                  v
          Streamlit Dashboard

Fallback:
If FastAPI is unavailable, Streamlit
uses the saved ML pipeline locally.
```

## Technology Stack

| Area                | Technologies     |
| ------------------- | ---------------- |
| Programming         | Python           |
| Data Processing     | Pandas, NumPy    |
| Machine Learning    | Scikit-learn     |
| Model Serialization | Joblib           |
| API                 | FastAPI, Uvicorn |
| Frontend            | Streamlit        |
| API Requests        | Requests         |
| Containerization    | Docker           |
| Container Registry  | Amazon ECR       |
| Cloud Deployment    | Amazon EC2       |
| API Testing         | Swagger UI       |

## Project Structure

```text
Customer Churn Prediction/
│
├── artifacts/
│   └── model/
│       └── best_model.joblib
│
├── data/
│   └── raw/
│       └── dataset.csv
│
├── frontend/
│   └── app.py
│
├── backend/
│   └── main.py
│
├── requirements-streamlit.txt
├── Dockerfile.streamlit
├── Dockerfile
├── README.md
└── ...
```

> The exact backend and training filenames may vary depending on the final repository structure.

## Machine Learning Workflow

1. Load the customer churn dataset.
2. Inspect and prepare the data.
3. Perform preprocessing and feature preparation.
4. Train and evaluate Machine Learning models.
5. Select the best-performing model pipeline.
6. Save the trained pipeline using Joblib.
7. Load the saved pipeline for prediction through the API or local fallback.

The saved pipeline contains the required preprocessing and model steps used during prediction.

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Pothuraju1690/Customer-Churn-Prediction
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the backend dependencies from the relevant requirements file:

```powershell
pip install -r requirements.txt
```

Install the Streamlit dependencies:

```powershell
pip install -r requirements-streamlit.txt
```

### 4. Start the FastAPI backend

```powershell
python -m uvicorn main:app --reload --port 8000
```

Open the API documentation:

```text
http://localhost:8000/docs
```

> Run this command from the directory containing the FastAPI application, and adjust the module path if your backend file is located elsewhere.

### 5. Start the Streamlit frontend

In a separate terminal:

```powershell
streamlit run frontend/app.py --server.port 8501
```

Open the application:

```text
http://localhost:8501
```

## Docker Deployment

The project uses separate Docker containers for the FastAPI backend and Streamlit frontend.

### Build the API image

```bash
docker build -t customer-churn-api .
```

### Build the Streamlit image

```bash
docker build -f Dockerfile.streamlit -t customer-churn-streamlit .
```

### Run the containers

Create a Docker network:

```bash
docker network create churn-network
```

Run the API:

```bash
docker run -d \
  --name customer-churn-api \
  --network churn-network \
  -p 8000:8000 \
  customer-churn-api
```

Run Streamlit:

```bash
docker run -d \
  --name customer-churn-streamlit \
  --network churn-network \
  -p 8501:8501 \
  -e API_URL=http://customer-churn-api:8000/predict \
  customer-churn-streamlit
```

Open:

- Streamlit: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`

## Prediction Fallback

The Streamlit application attempts to send prediction requests to FastAPI first.

- **FastAPI available:** prediction is served by the API backend.
- **FastAPI unavailable:** Streamlit loads the saved ML pipeline and performs prediction locally.
- **API restored:** the application can use the FastAPI backend again.

This fallback helps keep the prediction interface usable when the API service is temporarily unavailable.

## AWS Deployment

The application is deployed on an AWS EC2 instance, with Docker images stored in Amazon ECR.

### Deployed Services

| Service                       | URL                                                              |
| ----------------------------- | ---------------------------------------------------------------- |
| Streamlit Application         | [http://3.110.114.138:8501](http://3.110.114.138:8501)           |
| FastAPI Swagger Documentation | [http://3.110.114.138:8000/docs](http://3.110.114.138:8000/docs) |

### AWS Deployment Flow

1. Build Docker images.
2. Authenticate Docker with Amazon ECR.
3. Push API and Streamlit images to ECR.
4. Connect to the EC2 instance.
5. Pull the latest images from ECR.
6. Run the containers on the shared Docker network.
7. Access the deployed application through the EC2 public IP.

> These URLs use the EC2 public IP configured during deployment. If the instance IP changes, update the links accordingly.

## Testing the Fallback

To verify the fallback behavior on the deployment:

1. Stop the FastAPI container.
2. Submit a prediction through the Streamlit application.
3. Confirm the prediction source indicates **Local Model**.
4. Start the FastAPI container again.
5. Submit another prediction.
6. Confirm the prediction source indicates **FastAPI backend**.

## Future Improvements

- Automated testing integrated into CI/CD
- GitHub Actions deployment pipeline
- Application logging and monitoring
- Model performance and data drift monitoring
- Automated model retraining
- HTTPS and custom domain configuration

## Disclaimer

This project is intended for educational and portfolio demonstration purposes. Predictions are based on the dataset and trained model and should not be treated as guaranteed outcomes.

## Author

**Nani**
AI / Machine Learning Engineer – Portfolio Project
