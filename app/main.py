# app/main.py

import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Dict
from datetime import datetime

from src.models.baseline_trainer import BaselineTrainer
from app.schemas import PredictionRequest, PredictionResponse

# --- КОНФИГУРАЦИЯ ---
MODEL_NAME = 'lightgbm'
DEFAULT_THRESHOLD = 0.42

app = FastAPI(
    title="ML Credit Default Predictions API",
    version="1.0.0",
    description="API для скоринга кредитных заявок с использованием разных моделей"
)

MODEL_PIPELINE = None
MODEL_METRICS = {}

def load_model_on_startup():
    global MODEL_PIPELINE, MODEL_METRICS

    try:
        pipeline, metrics = BaselineTrainer.load_model(MODEL_NAME)

        MODEL_PIPELINE = pipeline
        MODEL_METRICS = metrics

        print(f"[{datetime.now().strftime('%H:%M:%S')}]  Model '{MODEL_NAME}' loaded successfully.")
        print(f"Loaded Test ROC-AUC: {MODEL_METRICS.get('test', {}).get('roc_auc', 'N/A')}")

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}]  FATAL ERROR: Could not load model '{MODEL_NAME}'.")
        print(f"Error: {e}")
        raise RuntimeError(f"Failed to load model: {e}") from e

# загружаем модель при запуске FastAPI
load_model_on_startup()


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Проверка состояния сервиса."""
    if MODEL_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "test_roc_auc": f"{MODEL_METRICS.get('test', {}).get('roc_auc', 'N/A'):.4f}"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Выполняет предсказание вероятности дефолта для одного клиента.
    """
    if MODEL_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model pipeline not available.")

    try:
        # --- 1. Подготовка данных ---

        # Объединяем основные поля с all_features
        input_data = {
            "AMT_CREDIT": request.AMT_CREDIT,
            "AMT_ANNUITY": request.AMT_ANNUITY,
            "DAYS_EMPLOYED": request.DAYS_EMPLOYED,
            **request.all_features
        }

        input_df = pd.DataFrame([input_data])

        # --- 2. Предсказание ---

        proba = MODEL_PIPELINE.predict_proba(input_df)[:, 1][0]

        # --- 3. Бинаризация по порогу ---

        prediction = int(proba >= DEFAULT_THRESHOLD)

        # --- 4. Возврат ответа ---
        return PredictionResponse(
            model_name=MODEL_NAME,
            prediction=prediction,
            probability=float(proba),
            threshold=DEFAULT_THRESHOLD
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")