"""FastAPI service for the champion fraud detection model."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_DIR = PROJECT_ROOT / "models" / "champion"


class PredictionRequest(BaseModel):
    """Request body for a fraud prediction."""

    V1: float = Field(..., description="Feature V1")
    V2: float = Field(..., description="Feature V2")
    V3: float = Field(..., description="Feature V3")
    V4: float = Field(..., description="Feature V4")
    V5: float = Field(..., description="Feature V5")
    V6: float = Field(..., description="Feature V6")
    V7: float = Field(..., description="Feature V7")
    V8: float = Field(..., description="Feature V8")
    V9: float = Field(..., description="Feature V9")
    V10: float = Field(..., description="Feature V10")
    V11: float = Field(..., description="Feature V11")
    V12: float = Field(..., description="Feature V12")
    V13: float = Field(..., description="Feature V13")
    V14: float = Field(..., description="Feature V14")
    V15: float = Field(..., description="Feature V15")
    V16: float = Field(..., description="Feature V16")
    V17: float = Field(..., description="Feature V17")
    V18: float = Field(..., description="Feature V18")
    V19: float = Field(..., description="Feature V19")
    V20: float = Field(..., description="Feature V20")
    V21: float = Field(..., description="Feature V21")
    V22: float = Field(..., description="Feature V22")
    V23: float = Field(..., description="Feature V23")
    V24: float = Field(..., description="Feature V24")
    V25: float = Field(..., description="Feature V25")
    V26: float = Field(..., description="Feature V26")
    V27: float = Field(..., description="Feature V27")
    V28: float = Field(..., description="Feature V28")
    Amount: float = Field(..., description="Transaction amount")


class PredictionResponse(BaseModel):
    """Response body for a fraud prediction."""

    prediction: int = Field(..., description="0: legitimate, 1: fraud")
    probability: float = Field(..., description="Fraud probability")
    is_fraud: bool = Field(..., description="True when classified as fraud")


class ModelInfo(BaseModel):
    """Metadata for the champion model."""

    run_id: str
    pr_auc: float
    model_type: str
    metrics: dict[str, Any]
    params: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, scaler, and metadata on startup; dispose DB on shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        Control to the running application.

    Raises:
        FileNotFoundError: If champion model artifacts are missing.
    """
    print("Starting API...")

    if not CHAMPION_DIR.exists():
        raise FileNotFoundError(
            f"Champion model directory not found: {CHAMPION_DIR}. "
            "Run 'python scripts/select_champion.py' first."
        )

    model_path = CHAMPION_DIR / "model"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print("Loading model...")

    app.state.model = joblib.load(CHAMPION_DIR / "model" / "model" / "model.pkl")

    print("Model loaded successfully.")

    scaler_path = model_path / "scaler.joblib"

    if scaler_path.exists():
        app.state.scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    else:
        app.state.scaler = None
        print("Warning: scaler.joblib not found. Using unscaled features.")

    metadata_path = CHAMPION_DIR / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            app.state.metadata = json.load(f)

        print("Metadata loaded successfully.")
    else:
        app.state.metadata = None
        print("Warning: metadata.json not found.")

    print("API ready to accept requests.")

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("PostgreSQL connection established.")
    except Exception as e:
        print(f"Warning: Could not connect to PostgreSQL: {e}")

    yield

    print("Shutting down API...")
    engine.dispose()


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    """Return a simple status message.

    Returns:
        JSON message confirming the API is running.
    """
    return {"message": "Fraud Detection API is running"}


@app.get("/model-info", response_model=ModelInfo)
def get_model_info(request: Request) -> ModelInfo:
    """Return metadata for the champion model.

    Args:
        request: Incoming request (reads ``app.state.metadata``).

    Returns:
        Champion model metadata.

    Raises:
        HTTPException: 404 when metadata is not available.
    """
    metadata = request.app.state.metadata

    if metadata is None:
        raise HTTPException(status_code=404, detail="Metadata not available")

    return ModelInfo(
        run_id=metadata["run_id"],
        pr_auc=metadata["pr_auc"],
        model_type=metadata["model_type"],
        metrics=metadata["metrics"],
        params=metadata["params"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request_data: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Score a transaction and optionally persist the prediction.

    Args:
        request_data: Validated feature payload.
        request: Incoming request (reads model and scaler from app state).

    Returns:
        Prediction label, fraud probability, and fraud flag.

    Raises:
        HTTPException: 503 when the model is not loaded.
    """
    model = request.app.state.model
    scaler = request.app.state.scaler

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    features = np.array([[getattr(request_data, col) for col in feature_cols]])

    if scaler is not None:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1]

    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO predictions (
                        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                        v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                        v21, v22, v23, v24, v25, v26, v27, v28, amount,
                        prediction, probability, is_fraud
                    ) VALUES (
                        :v1, :v2, :v3, :v4, :v5, :v6, :v7, :v8, :v9, :v10,
                        :v11, :v12, :v13, :v14, :v15, :v16, :v17, :v18, :v19, :v20,
                        :v21, :v22, :v23, :v24, :v25, :v26, :v27, :v28, :amount,
                        :prediction, :probability, :is_fraud
                    )
                """),
                {
                    "v1": request_data.V1,
                    "v2": request_data.V2,
                    "v3": request_data.V3,
                    "v4": request_data.V4,
                    "v5": request_data.V5,
                    "v6": request_data.V6,
                    "v7": request_data.V7,
                    "v8": request_data.V8,
                    "v9": request_data.V9,
                    "v10": request_data.V10,
                    "v11": request_data.V11,
                    "v12": request_data.V12,
                    "v13": request_data.V13,
                    "v14": request_data.V14,
                    "v15": request_data.V15,
                    "v16": request_data.V16,
                    "v17": request_data.V17,
                    "v18": request_data.V18,
                    "v19": request_data.V19,
                    "v20": request_data.V20,
                    "v21": request_data.V21,
                    "v22": request_data.V22,
                    "v23": request_data.V23,
                    "v24": request_data.V24,
                    "v25": request_data.V25,
                    "v26": request_data.V26,
                    "v27": request_data.V27,
                    "v28": request_data.V28,
                    "amount": request_data.Amount,
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "is_fraud": bool(prediction == 1),
                },
            )
    except Exception as e:
        import traceback

        print("ERROR SAVING TO DATABASE:")
        print(traceback.format_exc())
        print(e)

    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability),
        is_fraud=bool(prediction == 1),
    )
