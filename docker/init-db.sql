-- Criar banco de dados para armazenar predições
CREATE DATABASE IF NOT EXISTS fraud_detection;

-- Conectar ao banco de dados fraud_detection
\c fraud_detection;

-- Criar tabela para logging de predições
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    v1 FLOAT,
    v2 FLOAT,
    v3 FLOAT,
    v4 FLOAT,
    v5 FLOAT,
    v6 FLOAT,
    v7 FLOAT,
    v8 FLOAT,
    v9 FLOAT,
    v10 FLOAT,
    v11 FLOAT,
    v12 FLOAT,
    v13 FLOAT,
    v14 FLOAT,
    v15 FLOAT,
    v16 FLOAT,
    v17 FLOAT,
    v18 FLOAT,
    v19 FLOAT,
    v20 FLOAT,
    v21 FLOAT,
    v22 FLOAT,
    v23 FLOAT,
    v24 FLOAT,
    v25 FLOAT,
    v26 FLOAT,
    v27 FLOAT,
    v28 FLOAT,
    amount FLOAT,
    prediction INTEGER,
    probability FLOAT,
    is_fraud BOOLEAN
);

-- Criar índice para consultas mais rápidas
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_is_fraud ON predictions(is_fraud);

-- Conceder permissões
GRANT ALL PRIVILEGES ON DATABASE fraud_detection TO mlflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlflow;
