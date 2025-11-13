# src/models/xgboost_trainer.py
from xgboost.sklearn import XGBClassifier
from src.models.trainer_interface import BaseTrainer
from sklearn.pipeline import Pipeline
from typing import Optional, Dict, Tuple, Any
import numpy as np
from src.config import SEED

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

        ВАЖНО: передаем RAW X_test и y_test в eval_set, так как Pipeline
        сам применит препроцессинг через callbacks во время обучения.
        """
        it_kwargs = fit_kwargs or {}

        # конвертация object -> category
        for col in X_train.select_dtypes(include='object'):
            X_train[col] = X_train[col].astype('category')
        for col in X_test.select_dtypes(include='object'):
            X_test[col] = X_test[col].astype('category')


        # достаём препроцессор из пайплайна
        preprocessor = self.pipeline.named_steps.get('preprocessor')
        X_test_transformed = preprocessor.transform(X_test)

        final_fit_kwargs = {
            'model__eval_set': [(X_test_transformed, y_test)],
            'model__verbose': self.model_params.get('verbose', False)
        }

        fit_kwargs.update(final_fit_kwargs)

        return super().train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fit_kwargs=fit_kwargs
        )


    def _get_model(self):
        """Возвращает инициализированный XGBClassifier."""
        params = {'random_state': SEED}
        params.update(self.model_params or {})
        return XGBClassifier(**params)