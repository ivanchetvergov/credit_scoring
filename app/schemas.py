from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictionRequest(BaseModel):
    """"
    Схема для входных данных клиента (для предсказания)
    Включает ключевые фичи и поле для оставшихся фичей
    """
    # ключевые фичи
    AMT_CREDIT: float = Field(..., description="Сумма кредита")
    AMT_ANNUITY: float = Field(..., description="Ежегодный аннуитет")
    DAYS_EMPLOYED: float = Field(..., description="Кол-во дней работы (отрицательное)")

    # общее поле для остальных фичей (передаются как есть)
    all_features: Dict[str, Any] = Field(
        {},
        description="Словарь, содержащий все остальные фичи, которые нужны модели"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "DAYS_EMPLOYED": -457,
                "all_features": {
                    "REGION_POPULATION_RELATIVE": 0.0188,
                    "OWN_CAR_AGE": 12.0,
                }
            }
        }

class PredictionResponse(BaseModel):
    """
    Схема для ответа API.
    """
    model_name: str = Field(..., description="Название использованной модели")
    prediction: int = Field(..., description="Бинарное предсказание (0 или 1)")
    probability: float = Field(..., description="Вероятность положительного класса")
    threshold: float = Field(..., description="Порог, использованный для бинаризации")