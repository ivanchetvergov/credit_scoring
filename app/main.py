# app/main.py

import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Dict, Optional, Any
from datetime import datetime

from src.models.sklearn_trainer import SklearnTrainer
from app.schemas import RawApplicationData, PredictionResponse
from configs.processeed_features_config import FEATURE_ORDER
import logging

logger = logging.getLogger(__name__)

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

def _build_all_features(data: RawApplicationData) -> Dict[str, Any]:
    """"
    Преобразует RawApplicationData (17 фичей) в полный набор из 40 фичей,
    необходимых для модели.

    ВНИМАНИЕ: В этой заглушке мы генерируем фиктивные значения
    для агрегированных/производных фичей.
    """
    N: int = 40
    # 1 исходные фичи в словарь
    raw_features = data.model_dump(exclude_none=True)

    # 2 создаем временную заглушку для аггрегированных фич
    mock_aggregated_features = {

    }

    for i in range(N - len(mock_aggregated_features)):
        mock_aggregated_features[f"mock_agg_feature_{i}"] = 0.0

    full_features = {
        **{k.lower(): v for k, v in raw_features.items()},
        **mock_aggregated_features
    }

    return full_features


def load_model_on_startup():
    global MODEL_PIPELINE, MODEL_METRICS

    try:
        pipeline, metrics = SklearnTrainer.load_model(MODEL_NAME)

        MODEL_PIPELINE = pipeline
        MODEL_METRICS = metrics

        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}]  Model '{MODEL_NAME}' loaded successfully.")
        logger.info(f"Loaded Test ROC-AUC: {MODEL_METRICS.get('test', {}).get('roc_auc', 'N/A')}")

    except Exception as e:
        logger.error(f"FATAL ERROR: Could not load model '{MODEL_NAME}'. Error: {e}")
        raise RuntimeError(f"Failed to load model: {e}") from e


@app.on_event("startup")
def startup_event():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
async def predict(request: RawApplicationData, threshold: float = DEFAULT_THRESHOLD):
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
        }

        input_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)

        # --- 2. Предсказание ---

        proba = MODEL_PIPELINE.predict_proba(input_df)[:, 1][0]

        # --- 3. Бинаризация по порогу ---

        prediction = int(proba >= threshold)

        # --- 4. Возврат ответа ---
        return PredictionResponse(
            model_name=MODEL_NAME,
            prediction=prediction,
            probability=float(proba),
            threshold=DEFAULT_THRESHOLD
        )

    except Exception as e:
        logger.info(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")