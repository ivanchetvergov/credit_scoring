# src/models/xgboost_trainer.py
from xgboost.sklearn import XGBClassifier
from xgboost.callback import EarlyStopping
from src.models.trainer_interface import BaseTrainer
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
        # 1. извлекаем необходимое количество раундов из конфига
        stop_rounds = self.model_params.get('early_stopping_rounds', 200)

        # 2. cоздаем callback для ранней остановки
        early_stop = EarlyStopping(
            rounds=stop_rounds,
            save_best=True,
            metric_name='auc',      # указываем метрику для мониторинга
            data_name='validation'  # имя eval_set, которое мы передадим ниже
        )

        fit_kwargs = {
            # 3. early Stopping: X_test используется как eval_set
            'model__eval_set': [(X_test, y_test)],
            'model__callbacks': [early_stop],
            'model__verbose': False
        }

        # вызываем родительский метод с дополнительными аргументами
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