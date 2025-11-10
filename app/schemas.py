# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# =========================================================================
# 1. ВХОДНАЯ СХЕМА (для пользователя/веб-сайта)
# Принимает только необработанные данные заявки.
# =========================================================================

class RawApplicationData(BaseModel):
    """
    Минимальный контракт для входных данных, которые может предоставить клиент
    (без агрегаций Bureau/Previous).
    """
    # Основные финансовые и временные фичи
    AMT_CREDIT: float = Field(..., description="Сумма запрашиваемого кредита")
    AMT_ANNUITY: float = Field(..., description="Запрашиваемый годовой платеж")
    AMT_GOODS_PRICE: float = Field(..., description="Стоимость товара")
    AMT_INCOME_TOTAL: float = Field(..., description="Годовой доход клиента")

    DAYS_BIRTH: float = Field(..., description="Возраст клиента (в днях, отрицательное)")
    DAYS_EMPLOYED: float = Field(..., description="Дни работы (отрицательное)")
    DAYS_ID_PUBLISH: float = Field(..., description="Дни с момента публикации ID (отрицательное)")

    # Внешние оценки (могут быть пропущены)
    EXT_SOURCE_1: Optional[float] = Field(None, description="Внешний источник 1")
    EXT_SOURCE_2: Optional[float] = Field(None, description="Внешний источник 2")
    EXT_SOURCE_3: Optional[float] = Field(None, description="Внешний источник 3")

    # Категориальные фичи (OHE будет выполнено в API)
    CODE_GENDER: str = Field(..., description="Пол (M/F)")
    FLAG_OWN_CAR: str = Field(..., description="Наличие авто (Y/N)")
    NAME_EDUCATION_TYPE: str = Field(..., description="Образование")
    NAME_FAMILY_STATUS: str = Field(..., description="Семейный статус")
    NAME_CONTRACT_TYPE: str = Field(..., description="Тип контракта")
    REGION_RATING_CLIENT_W_CITY: int = Field(..., description="Рейтинг региона")

    # Внешняя фича
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(None, description="Кол-во дефолтов в соц. круге")

    class Config:
        json_schema_extra = {
            "example": {
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0,
                "AMT_INCOME_TOTAL": 200000.0,
                "DAYS_BIRTH": -9461.0,
                "DAYS_EMPLOYED": -457.0,
                "DAYS_ID_PUBLISH": -2120.0,
                "EXT_SOURCE_1": 0.505,
                "EXT_SOURCE_2": 0.654,
                "EXT_SOURCE_3": 0.231,
                "CODE_GENDER": "F",
                "FLAG_OWN_CAR": "N",
                "NAME_EDUCATION_TYPE": "Higher education",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_CONTRACT_TYPE": "Cash loans",
                "REGION_RATING_CLIENT_W_CITY": 2,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
            }
        }

# =========================================================================
# 2. СХЕМА ОТВЕТА
# =========================================================================
class PredictionResponse(BaseModel):
    """
    Схема для ответа API.
    """
    model_name: str = Field(..., description="Название использованной модели")
    prediction: int = Field(..., description="Бинарное предсказание (0 или 1)")
    probability: float = Field(..., description="Вероятность положительного класса")
    threshold: float = Field(..., description="Порог, использованный для бинаризации")