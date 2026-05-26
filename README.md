# Detecção de Fraude em Cartão de Crédito

Projeto de Machine Learning com o dataset [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB MLG).

## Estrutura

- `data/raw/` — dataset original
- `notebooks/` — análise exploratória
- `src/` — código de pré-processamento, modelos e métricas
- `scripts/` — treino e pipelines
- `mlruns/` — experimentos MLflow (local)
- `api/` — API FastAPI para predições
- `app/` — Interface Streamlit
- `docker/` — Scripts de inicialização do banco

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Docker

### Pré-requisitos

- Docker Desktop instalado
- Docker Compose

### Subir todos os serviços

```bash
docker compose up --build
```

Isso iniciará:
- **PostgreSQL** (porta 5432) — Backend para MLflow e armazenamento de predições
- **MLflow Server** (porta 5000) — Tracking e UI de experimentos
- **API FastAPI** (porta 8000) — Serviço de predição
- **Streamlit App** (porta 8501) — Interface web interativa

### Acessar os serviços

- API: http://localhost:8000
- Streamlit: http://localhost:8501
- MLflow UI: http://localhost:5000
- Documentação API: http://localhost:8000/docs

### Comandos úteis

```bash
# Parar serviços
docker compose down

# Parar e remover volumes
docker compose down -v

# Ver logs
docker compose logs -f

# Ver logs de um serviço específico
docker compose logs -f api
```

## Uso rápido

```bash
# EDA
jupyter notebook notebooks/01_eda.ipynb

# Regressão logística (parâmetros padrão do config)
python scripts/train_logistic_regression.py

# Variante do config (model.logistic_regression.experiments)
python scripts/train_logistic_regression.py --experiment lr_balanced_strong_reg

# Todas as variantes do config
python scripts/train_logistic_regression.py --all-experiments

# Parâmetros pela CLI (sobrescrevem config)
python scripts/train_logistic_regression.py --C 0.1 --penalty l1 --solver saga --class-weight balanced --run-name lr_custom

# KNN (usa amostra estratificada de treino por padrão — ver config)
python scripts/train_knn.py
python scripts/train_knn.py --experiment knn_k5_distance
python scripts/train_knn.py --all-experiments
python scripts/train_knn.py --n-neighbors 7 --weights distance --train-sample-size 30000

# Random Forest
python scripts/train_random_forest.py
python scripts/train_random_forest.py --experiment rf_balanced_deep
python scripts/train_random_forest.py --all-experiments
python scripts/train_random_forest.py --n-estimators 200 --max-depth 12 --class-weight balanced

# XGBoost
python scripts/train_xgboost.py
python scripts/train_xgboost.py --experiment xgb_auto_scale_deep
python scripts/train_xgboost.py --all-experiments
python scripts/train_xgboost.py --auto-scale-pos-weight --learning-rate 0.05 --max-depth 8
```

## Modelos planejados

Regressão logística, Random Forest, SVM, XGBoost, CatBoost, KNN — com tracking via MLflow.
