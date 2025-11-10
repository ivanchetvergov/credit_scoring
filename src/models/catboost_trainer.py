# src/models/catboost_trainer.py
from catboost import CatBoostClassifier
from src.models.trainer_interface import BaseTrainer
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, Any, Optional
import numpy as np

from configs.catboost_config import CAT_FEATURES_COLS

class CatBoostTrainer(BaseTrainer):
    """"
    Класс "учитель" для CatBoostClassifier
    """
    def __init__(self, **kwargs):
        # инициализируем родительский класс, передавая имя модели
        super().__init__(model_name='catboost', **kwargs)

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            fit_kwargs: Optional[Dict] = None
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Переопределение метода train для добавления специфичных аргументов:
        - cat_features (список колонок для нативной обработки)
        - eval_set (для ранней остановки)
        """

        # аргументы для передачи в pipeline.fit()
        fit_kwargs = {
            # 1. CatBoost Specific: передаем список категориальных фичей
            'model__cat_features': CAT_FEATURES_COLS,

            # 2. Early Stopping: используем X_test как валидационный набор
            'model__eval_set': (X_test, y_test),
            'model__early_stopping_rounds': self.model_params.get('early_stopping_rounds', 100),
            'model__verbose': self.model_params.get('verbose', False)
        }

        # вызываем родительский метод с дополнительными аргументами
        return super().train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fit_kwargs=fit_kwargs  # передаем kwargs в BaseTrainer.train
        )

    def _get_model(self):
        """
        Реализация абстрактного метода: возвращает инициализированный LGBMClassifier.
        """
        return CatBoostClassifier(**self.model_params)