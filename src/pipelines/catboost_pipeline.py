# src/pipelines/catboost_pipeline.py (ОБНОВЛЕННЫЙ)

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from typing import Dict, Any

from src.pipelines.base_pipeline import BasePipelineBuilder
# Импортируем нашу фабричную функцию
from src.pipelines.preprocessor import get_model_specific_pipeline
from src.config import CATEGORICAL_FEATURES, BIN_CATEGORICAL_FEATURES


class CatBoostPipelineBuilder(BasePipelineBuilder):
    """
    Строитель для CatBoostClassifier.
    Использует минимальный препроцессинг (только FE) с помощью фабричной функции.
    Инкапсулирует передачу списка категориальных фичей в CatBoostClassifier.
    """

    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        """
        Получает специфичный для CatBoost пайплайн препроцессинга,
        используя фабричную функцию.
        """
        return get_model_specific_pipeline(
            model_name='catboost',
            include_feature_engineering=feature_engineering
        )

    def _get_model(self) -> CatBoostClassifier:
        """
        Создает инстанс CatBoostClassifier, используя self.random_state из Base.
        ВАЖНО: Инкапсулируем список категориальных фичей здесь.
        """
        params = {'random_state': self.random_state}
        params.update(self.model_params)

        # 1. рпределяем категориальные фичи, которые CatBoost должен обработать
        cat_features_for_model = CATEGORICAL_FEATURES + BIN_CATEGORICAL_FEATURES

        # 2. Добавляем их к параметрам инициализации модели
        params['cat_features'] = cat_features_for_model

        # verbose=False по умолчанию для чистоты логов
        return CatBoostClassifier(verbose=False, **params)