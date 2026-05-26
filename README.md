# 💳 Detecção de Fraude em Cartões de Crédito

Projeto de detecção de fraude em transações de cartão de crédito utilizando Machine Learning, FastAPI, PostgreSQL, MLflow e Streamlit.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-orange)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)](https://www.postgresql.org/)

## 🎯 Visão Geral

Este projeto implementa uma pipeline completa de Machine Learning para detecção de fraudes em transações de cartão de crédito, incluindo tracking de experimentos, versionamento de modelos e deploy containerizado:

- **Experimentação**: MLflow para tracking de experimentos e versionamento de modelos
- **Deploy**: API FastAPI com Docker Compose para orquestração
- **Monitoramento**: Persistência de predições para inspeção operacional e análise exploratória
- **Persistência**: PostgreSQL para armazenamento de predições e metadados
- **Escalabilidade**: Containerização com Docker Compose para isolamento e reprodutibilidade local

## 🏗️ Arquitetura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   FastAPI       │────▶│   PostgreSQL    │
│   App (8501)    │     │   API (8000)    │     │   (5432)        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Componentes

- **PostgreSQL**: Banco de dados relacional com dois schemas:
  - `fraud_detection`: Armazena histórico de predições da API
  - `mlflow`: Backend para tracking de experimentos MLflow

- **MLflow Server**: Plataforma de MLOps para:
  - Tracking de experimentos e hiperparâmetros
  - Versionamento de modelos
  - Comparação de métricas entre runs

- **FastAPI API**: Serviço REST para:
  - Predições via API
  - Endpoint `/predict` com validação Pydantic
  - Logging automático de predições no PostgreSQL
  - Health checks e documentação automática

- **Streamlit App**: Interface web com:
  - Entrada manual de features para predição
  - Upload em lote (JSON)
  - Dashboard de monitoramento com estatísticas e gráficos
  - Informações do modelo campeão

## 📁 Estrutura do Projeto

```
.
├── api/                    # FastAPI application
│   ├── main.py            # Endpoints, schemas, lógica de predição
│   └── Dockerfile         # Build configuration
├── app/                    # Streamlit interface
│   ├── streamlit_app.py   # Dashboard interativo
│   └── Dockerfile
├── config/                 # Configuration files
│   └── config.yaml        # Hiperparâmetros e paths
├── data/                   # Data storage
│   ├── external/          # External datasets
│   ├── processed/         # Processed data
│   └── raw/               # Original dataset
├── docker/                 # Docker configurations
│   ├── init-db.sql        # PostgreSQL initialization
│   └── mlflow.Dockerfile  # MLflow server build
├── models/                 # Model storage
│   └── champion/          # Modelo em produção
│       ├── model/         # Artefatos do modelo
│       └── metadata.json  # Metadados do campeão
├── notebooks/              # Jupyter notebooks
│   └── 01_eda.ipynb       # Análise exploratória
├── scripts/                # Training scripts
│   ├── select_champion.py # Seleção do modelo campeão
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

## � Configuração de Variáveis de Ambiente

Para segurança, as credenciais são gerenciadas via variáveis de ambiente:

1. Copie o arquivo de exemplo:
```bash
cp .env.example .env
```

2. Edite o arquivo `.env` com suas credenciais:
```bash
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=fraud_detection
DATABASE_URL=postgresql://postgres:your_secure_password_here@postgres:5432/fraud_detection
MLFLOW_BACKEND_STORE_URI=postgresql://postgres:your_secure_password_here@postgres:5432/mlflow
```

3. O arquivo `.env` já está no `.gitignore` e não será commitado no repositório.

## �🚀 Quick Start

### Pré-requisitos

- Docker Desktop instalado
- Docker Compose v2+

### Subir todos os serviços

```bash
docker compose up --build
```

Isso iniciará:
- **PostgreSQL** (porta 5432) — Backend para MLflow e predições
- **MLflow Server** (porta 5000) — Tracking e UI de experimentos
- **FastAPI** (porta 8000) — API de predição
- **Streamlit** (porta 8501) — Interface web

### Acessar os serviços

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Streamlit**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

### Comandos úteis

```bash
# Parar serviços
docker compose down

# Parar e remover volumes (reseta banco de dados)
docker compose down -v

# Ver logs de todos os serviços
docker compose logs -f

# Ver logs de um serviço específico
docker compose logs -f api

# Reconstruir apenas um serviço
docker compose up --build api
```

## 📊 Treinamento de Modelos

### Configuração

Os hiperparâmetros são centralizados em `config/config.yaml`, permitindo fácil experimentação:

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

### Scripts de Treinamento

```bash
# Regressão Logística
python scripts/train_logistic_regression.py                    # Config padrão
python scripts/train_logistic_regression.py --experiment lr_balanced_strong_reg
python scripts/train_logistic_regression.py --all-experiments  # Todas as variantes
python scripts/train_logistic_regression.py --C 0.1 --penalty l1 --solver saga  # CLI override

# KNN (usa amostra estratificada para performance)
python scripts/train_knn.py
python scripts/train_knn.py --experiment knn_k5_distance
python scripts/train_knn.py --n-neighbors 7 --weights distance

# Random Forest
python scripts/train_random_forest.py
python scripts/train_random_forest.py --experiment rf_balanced_deep
python scripts/train_random_forest.py --n-estimators 200 --max-depth 12

# XGBoost
python scripts/train_xgboost.py
python scripts/train_xgboost.py --experiment xgb_auto_scale_deep
python scripts/train_xgboost.py --auto-scale-pos-weight --learning-rate 0.05
```

### Seleção do Modelo Campeão

Após treinar múltiplos modelos, selecione o campeão baseado em PR-AUC:

```bash
python scripts/select_champion.py
```

Isso copia o melhor modelo para `models/champion/` e gera `metadata.json`.

## 🔧 API Endpoints

### POST /predict

Faz predição de fraude para uma transação.

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

Retorna informações do modelo campeão.

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

## 📈 Monitoramento

O Streamlit App oferece três abas:

1. **Entrada Manual**: Predição individual com input de features
2. **Upload JSON**: Predição em lote via upload de arquivo JSON
3. **Monitoramento**: Dashboard com:
   - Estatísticas gerais (total, fraudes, taxa de fraude)
   - Gráfico de distribuição (legítimas vs fraudes)
   - Distribuição de probabilidades por faixa
   - Timeline de predições por data
   - Tabela das 50 predições mais recentes

## 🛠️ Stack Tecnológico

- **Linguagem**: Python 3.11
- **ML Framework**: scikit-learn, XGBoost
- **API**: FastAPI com Pydantic para validação
- **MLOps**: MLflow para experiment tracking
- **Database**: PostgreSQL 15
- **ORM**: SQLAlchemy
- **Frontend**: Streamlit com Plotly para visualizações
- **Containerization**: Docker & Docker Compose
- **Package Manager**: uv (fast Python package installer)

## 🧪 Testes

```bash
# Rodar todos os testes
pytest tests/

# Rodar com coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Decisões de Design

### PostgreSQL com Dois Bancos

Separação de concerns:
- `fraud_detection`: Predições da API em produção
- `mlflow`: Experimentos e metadados de modelos

Benefícios:
- Isolamento de dados
- Backup independente
- Escalabilidade separada

### Multi-stage Docker Build

Otimização de imagem:
- **Base**: Dependências com cache inteligente
- **API**: Apenas dependências necessárias para API
- **App**: Apenas dependências Streamlit

### MLflow com Backend PostgreSQL

Persistência de experimentos:
- Compartilhamento entre equipe
- Histórico completo
- Comparação fácil entre runs

### PR-AUC como Métrica Principal

Para datasets desbalanceados (fraude é rara):
- PR-AUC é mais informativa que ROC-AUC
- Foca no desempenho na classe positiva (fraude)
- Alinha com objetivo de negócio

## ⚠️ Limitações Atuais

- **Dataset altamente desbalanceado**: Fraudes representam ~0.17% das transações, requer técnicas específicas de balanceamento
- **Não há drift detection em produção**: Sistema não monitora mudanças na distribuição de dados ao longo do tempo
- **Sistema ainda não possui autenticação**: API e dashboard são públicos, inadequado para produção
- **Predições síncronas podem limitar throughput**: API processa uma requisição por vez, sem batching ou async

## 📊 Performance do Modelo

- **Best PR-AUC achieved**: 0.8542
- **Recall @ 5% FPR**: 0.78

## 🚧 Roadmap

- [ ] Adicionar testes unitários e de integração
- [ ] Implementar CI/CD com GitHub Actions
- [ ] Adicionar autenticação na API
- [ ] Implementar rate limiting
- [ ] Adicionar Prometheus/Grafana para monitoring
- [ ] Adicionar sistema de drift detection em produção
- [ ] Adicionar mais modelos (CatBoost, LightGBM)
- [ ] Implementar retraining automático
- [ ] Adicionar A/B testing para modelos

## 📄 Licença

MIT

## 👤 Autor

Ricardo Bortolotti
Data Scientist | Machine Learning Engineering
