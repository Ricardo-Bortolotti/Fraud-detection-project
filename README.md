# 💳 Credit Card Fraud Detection

Fraud detection project for credit card transactions using Machine Learning, FastAPI, PostgreSQL, MLflow, and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-orange)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)](https://www.postgresql.org/)

## 🎯 Overview

This project implements a complete Machine Learning pipeline for credit card fraud detection, including experiment tracking, model versioning, and containerized deployment:

- **Experimentation**: MLflow for experiment tracking and model versioning
- **Deployment**: FastAPI API with Docker Compose for orchestration
- **Monitoring**: Persistence of predictions for operational inspection and exploratory analysis
- **Persistence**: PostgreSQL for storing predictions and metadata
- **Scalability**: Containerization with Docker Compose for isolation and local reproducibility

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   FastAPI       │────▶│   PostgreSQL    │
│   App (8501)    │     │   API (8000)    │     │   (5432)        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Components

- **PostgreSQL**: Relational database with two schemas:
  - `fraud_detection`: Stores API prediction history
  - `mlflow`: Backend for MLflow experiment tracking

- **MLflow Server**: MLOps platform for:
  - Experiment and hyperparameter tracking
  - Model versioning
  - Metric comparison across runs

- **FastAPI API**: REST service for:
  - Predictions via API
  - `/predict` endpoint with Pydantic validation
  - Automatic prediction logging to PostgreSQL
  - Health checks and auto-generated documentation

- **Streamlit App**: Web interface with:
  - Manual feature input for single predictions
  - Batch upload (JSON)
  - Monitoring dashboard with statistics and charts
  - Champion model information

## 📁 Project Structure

```
.
├── api/                    # FastAPI application
│   ├── main.py            # Endpoints, schemas, prediction logic
│   └── Dockerfile         # Build configuration
├── app/                    # Streamlit interface
│   ├── streamlit_app.py   # Interactive dashboard
│   └── Dockerfile
├── config/                 # Configuration files
│   └── config.yaml        # Hyperparameters and paths
├── data/                   # Data storage
│   ├── external/          # External datasets
│   ├── processed/         # Processed data
│   └── raw/               # Original dataset
├── docker/                 # Docker configurations
│   ├── init-db.sql        # PostgreSQL initialization
│   └── mlflow.Dockerfile  # MLflow server build
├── models/                 # Model storage
│   └── champion/          # Production model
│       ├── model/         # Model artifacts
│       └── metadata.json  # Champion metadata
├── notebooks/              # Jupyter notebooks
│   └── 01_eda.ipynb       # Exploratory analysis
├── scripts/                # Training scripts
│   ├── select_champion.py # Champion model selection
│   ├── train_logistic_regression.py
│   ├── train_knn.py
│   ├── train_random_forest.py
│   └── train_xgboost.py
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── evaluation/        # Metrics and evaluation
│   ├── features/          # Feature engineering
│   └── models/            # Model definitions
├── tests/                  # Unit tests
├── docker-compose.yml      # Orchestration
├── Dockerfile             # Multi-stage build
├── pyproject.toml         # Dependencies (uv)
└── uv.lock                # Locked dependencies
```

## 🔐 Environment Variables

For security, credentials are managed via environment variables:

1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit `.env` with your credentials:
```bash
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=fraud_detection
DATABASE_URL=postgresql://postgres:your_secure_password_here@postgres:5432/fraud_detection
MLFLOW_BACKEND_STORE_URI=postgresql://postgres:your_secure_password_here@postgres:5432/mlflow
```

3. The `.env` file is listed in `.gitignore` and will not be committed.

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed
- Docker Compose v2+

### Start all services

```bash
docker compose up --build
```

This starts:
- **PostgreSQL** (port 5432) — Backend for MLflow and predictions
- **MLflow Server** (port 5000) — Experiment tracking and UI
- **FastAPI** (port 8000) — Prediction API
- **Streamlit** (port 8501) — Web interface

### Access services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Streamlit**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

### Useful commands

```bash
# Stop services
docker compose down

# Stop and remove volumes (resets the database)
docker compose down -v

# View logs for all services
docker compose logs -f

# View logs for a specific service
docker compose logs -f api

# Rebuild a single service
docker compose up --build api
```

## 📊 Model Training

### Configuration

Hyperparameters are centralized in `config/config.yaml` for easy experimentation:

```yaml
model:
  logistic_regression:
    max_iter: 1000
    class_weight: balanced
    C: 1.0
    experiments:
      - name: lr_balanced_default
      - name: lr_balanced_strong_reg
        C: 0.01
```

### Training scripts

```bash
# Logistic regression
python scripts/train_logistic_regression.py                    # Default config
python scripts/train_logistic_regression.py --experiment lr_balanced_strong_reg
python scripts/train_logistic_regression.py --all-experiments  # All variants
python scripts/train_logistic_regression.py --C 0.1 --penalty l1 --solver saga  # CLI override

# KNN (uses stratified sample for performance)
python scripts/train_knn.py
python scripts/train_knn.py --experiment knn_k5_distance
python scripts/train_knn.py --n-neighbors 7 --weights distance

# Random forest
python scripts/train_random_forest.py
python scripts/train_random_forest.py --experiment rf_balanced_deep
python scripts/train_random_forest.py --n-estimators 200 --max-depth 12

# XGBoost
python scripts/train_xgboost.py
python scripts/train_xgboost.py --experiment xgb_auto_scale_deep
python scripts/train_xgboost.py --auto-scale-pos-weight --learning-rate 0.05
```

### Champion model selection

After training multiple models, select the champion based on PR-AUC:

```bash
python scripts/select_champion.py
```

This copies the best model to `models/champion/` and writes `metadata.json`.

## 🔧 API Endpoints

### POST /predict

Score a transaction for fraud.

**Request:**
```json
{
  "V1": -1.3598,
  "V2": 0.0724,
  "V3": 2.5363,
  ...
  "V28": 0.1234,
  "Amount": 149.62
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.0234,
  "is_fraud": false
}
```

### GET /model-info

Return champion model metadata.

**Response:**
```json
{
  "run_id": "abc123",
  "pr_auc": 0.8542,
  "model_type": "RandomForestClassifier",
  "metrics": {...},
  "params": {...}
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## 📈 Monitoring

The Streamlit app provides three tabs:

1. **Manual Input**: Single prediction with feature inputs
2. **JSON Upload**: Batch predictions via JSON file upload
3. **Monitoring**: Dashboard with:
   - Summary statistics (total, frauds, fraud rate)
   - Distribution chart (legitimate vs fraud)
   - Probability distribution by bucket
   - Prediction timeline by date
   - Table of the 50 most recent predictions

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **ML Framework**: scikit-learn, XGBoost
- **API**: FastAPI with Pydantic validation
- **MLOps**: MLflow for experiment tracking
- **Database**: PostgreSQL 15
- **ORM**: SQLAlchemy
- **Frontend**: Streamlit with Plotly visualizations
- **Containerization**: Docker & Docker Compose
- **Package Manager**: uv (fast Python package installer)

## 🧪 Tests

The project includes unit tests for the API, data modules, evaluation, and scripts.

### Install test dependencies

```bash
# Install development dependencies (dev group)
uv add --dev pytest pytest-cov pytest-asyncio httpx

# Sync all dependencies
uv sync
```

### Run tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src --cov=api --cov=app --cov=scripts --cov-report=html

# Run tests for a specific module
pytest tests/test_api.py

# Verbose output
pytest tests/ -v

# Show print statements
pytest tests/ -s
```

### Test layout

- `tests/test_api.py` — FastAPI endpoints and schemas
- `tests/test_data.py` — Data loading and preprocessing
- `tests/test_evaluation.py` — Evaluation metrics
- `tests/test_scripts.py` — Training scripts and champion selection

## 📝 Design Decisions

### PostgreSQL with two databases

Separation of concerns:
- `fraud_detection`: Production API predictions
- `mlflow`: Experiments and model metadata

Benefits:
- Data isolation
- Independent backups
- Separate scaling

### Multi-stage Docker build

Image optimization:
- **Base**: Dependencies with smart caching
- **API**: Only API-required dependencies
- **App**: Only Streamlit dependencies

### MLflow with PostgreSQL backend

Experiment persistence:
- Team-wide sharing
- Full history
- Easy run comparison

### PR-AUC as the primary metric

For imbalanced datasets (fraud is rare):
- PR-AUC is more informative than ROC-AUC
- Focuses on positive-class (fraud) performance
- Aligns with business goals

## ⚠️ Current Limitations

- **Highly imbalanced dataset**: Fraud is ~0.17% of transactions; balancing techniques are required
- **No production drift detection**: The system does not monitor distribution shift over time
- **No authentication yet**: API and dashboard are public; not suitable for production as-is
- **Synchronous predictions may limit throughput**: The API processes one request at a time without batching

## 📊 Model Performance

- **Best PR-AUC achieved**: 0.8542
- **Recall @ 5% FPR**: 0.78

## 🚧 Roadmap

- [ ] Add unit and integration tests
- [ ] Implement CI/CD with GitHub Actions
- [ ] Add API authentication
- [ ] Implement rate limiting
- [ ] Add Prometheus/Grafana monitoring
- [ ] Add production drift detection
- [ ] Add more models (CatBoost, LightGBM)
- [ ] Implement automatic retraining
- [ ] Add A/B testing for models

## 📄 License

MIT

## 👤 Author

Ricardo Bortolotti  
Data Scientist | Machine Learning Engineering
