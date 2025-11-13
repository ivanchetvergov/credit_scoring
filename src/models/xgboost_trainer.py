# src/models/xgboost_trainer.py (Обновленный)
from sklearn.pipeline import Pipeline

from src.models.trainer_interface import BaseTrainer
from xgboost import XGBClassifier
from typing import Dict, Tuple, Any, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XGBoostTrainer(BaseTrainer):
    """"
    Класс "учитель" для XGBoostClassifier.
    Использует полный пайплайн Sklearn (как и другие обычные модели),
    но добавляет аргументы для ранней остановки (early_stopping).
    """

    def __init__(self, **kwargs):
        # инициализируем родительский класс
        super().__init__(model_name='xgboost', **kwargs)

    def _get_model(self):
        """Возвращает инициализированный XGBClassifier."""
        # Используйте XGBRegressor, если это задача регрессии
        return XGBClassifier(**self.model_params)

    # ПЕРЕОПРЕДЕЛЯЕМ МЕТОД TRAIN
    def train(
            self,
            X_train: Any,
            y_train: np.ndarray,
            X_test: Any,
            y_test: np.ndarray,
            fit_kwargs: Optional[Dict] = None,
            external_preprocessor: Optional[Pipeline] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Переопределение метода train для добавления специфичных аргументов:
        - eval_set (для ранней остановки)
        """

        fit_kwargs = fit_kwargs or {}

        fit_kwargs.update({
            # передаем eval_set для ранней остановки, обернутый в 'model__'
            'model__eval_set': [(X_test, y_test)],
        })


        final_pipeline, metrics = super().train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fit_kwargs=fit_kwargs
        )

        return final_pipeline, metrics