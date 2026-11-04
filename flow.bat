@echo off
title Customer Churn Prediction - MLOps Project Setup

echo.
echo ==========================================
echo   Customer Churn Prediction MLOps Setup
echo ==========================================
echo.

REM Create project directories
mkdir data 2>nul
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir notebooks 2>nul

mkdir src 2>nul
mkdir src\data 2>nul
mkdir src\models 2>nul

mkdir api 2>nul
mkdir frontend 2>nul

mkdir artifacts 2>nul
mkdir artifacts\model 2>nul

mkdir tests 2>nul
mkdir logs 2>nul
mkdir configs 2>nul

mkdir .github 2>nul
mkdir .github\workflows 2>nul

REM Create initial project files
type nul > README.md
type nul > requirements.txt
type nul > .gitignore
type nul > Dockerfile
type nul > docker-compose.yml
type nul > dvc.yaml
type nul > params.yaml

type nul > notebooks\01_eda.ipynb

type nul > src\data\preprocess.py
type nul > src\models\train.py
type nul > src\models\evaluate.py

type nul > api\main.py
type nul > frontend\app.py

type nul > tests\test_preprocess.py
type nul > tests\test_api.py

type nul > .github\workflows\ci.yml

echo.
echo Project folder structure created successfully!
echo.
tree /F
echo.
pause