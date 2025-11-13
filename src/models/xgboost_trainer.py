# src/models/xgboost_trainer.py
from xgboost.sklearn import XGBClassifier
from src.models.trainer_interface import BaseTrainer
from src.features.pipelines import get_preprocessing_pipeline
from sklearn.pipeline import Pipeline
from typing import Optional, Dict, Tuple, Any
import numpy as np

class XGBoostTrainer(BaseTrainer):
    """
    Класс-тренер для модели XGBoostClassifier.
    Поддерживает раннюю остановку (early stopping) через переопределение метода train.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name='xgboost', **kwargs)

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            fit_kwargs: Optional[Dict] = None
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Переопределение метода train для добавления аргументов для ранней остановки.
        """
        fit_kwargs = fit_kwargs or {}
        preprocessor_for_test_transform = get_preprocessing_pipeline()

        # 1. обучаем препроцессор ТОЛЬКО на X_train
        preprocessor_for_test_transform.fit(X_train)

        # 2. трансформируем X_test, используя статистики, полученные ИЗ X_train
        X_test_transformed = preprocessor_for_test_transform.transform(X_test)

        # 3. создаем fit_kwargs с ПРЕОБРАЗОВАННЫМИ данными
        final_fit_kwargs = {
            'model__eval_set': [(X_test, y_test)],
            'model__verbose': False
        }

        fit_kwargs.update(final_fit_kwargs)
        print("DEBUG: Final fit_kwargs for XGBoost:", fit_kwargs)

        return super().train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fit_kwargs=fit_kwargs
        )

    def _get_model(self):
        """
        Реализация абстрактного метода: возвращает инициализированный XGBClassifier.
        """
        # XGBoost ожидает, что категориальные фичи уже преобразованы (OHE/Label Encoding).
        return XGBClassifier(**self.model_params)