# src/pipelines/base_pipeline.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

# импорты для Feature Engineering (FE)
from src.features.custom_transformers import (
    AnomalyHandler,
    FeatureCreator,
    DataFrameCoercer
)
from src.config import SEED


class BasePipelineBuilder(ABC):
    """
    Абстрактный класс-строитель для создания end-to-end пайплайнов
    (препроцессинг + модель).
    Инкапсулирует логику Feature Engineering и random_state.
    """

    def __init__(self, model_params: Dict[str, Any]):
        self.model_params = model_params
        # инкапсулируем глобальный сид
        self.random_state = SEED

    @staticmethod
    def _get_feature_engineering_pipeline() -> Pipeline:
        """
        Собирает пайплайн, отвечающий за Feature Engineering (FE).
        """
        return Pipeline(
            steps=[
                # трансформеры инициализируются с параметрами по умолчанию
                ('coercer_pre', DataFrameCoercer()),
                ('anomaly_handler', AnomalyHandler()),
                ('feature_creator', FeatureCreator()),
            ]
        )

    @abstractmethod
    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        """
        Получает пайплайн препроцессинга.
        Включает FE, если feature_engineering=True.
        """
        pass

    @abstractmethod
    def _get_model(self) -> BaseEstimator:
        """
        Получает инстанс модели.
        """
        pass

    def build_pipeline(self, feature_engineering: bool = True) -> Pipeline:
        """
        Собирает финальный пайплайн.
        """
        # используем новый аргумент feature_engineering
        preprocessor_pipe = self._get_preprocessor(feature_engineering=feature_engineering)
        model = self._get_model()

        # добавляем модель к уже существующим шагам препроцессора
        steps = preprocessor_pipe.steps
        steps.append(('model', model))

        return Pipeline(steps)