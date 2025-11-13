# src/models/catboost_trainer.py

from catboost import CatBoostClassifier
from src.models.trainer_interface import BaseTrainer
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.features.custom_transformers import (
    DataFrameCoercer, AnomalyHandler, FeatureCreator
)

from src.pipelines.catboost_pipeline import get_catboost_preprocessing_pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CatBoostTrainer(BaseTrainer):
    """"
    Класс "учитель" для CatBoostClassifier, который выполняет FE вручную
    для сохранения метаданных DataFrame для CatBoost.
    """

    def __init__(self, **kwargs):
        super().__init__(model_name='catboost', **kwargs)
        # инициализируем пайплайн FE (он нам нужен для ручного fit/transform)
        self.fe_pipeline = Pipeline(steps=[
            ('coercer', DataFrameCoercer()),
            ('anomaly_handler', AnomalyHandler()),
            ('feature_creator', FeatureCreator()),
        ])

        self.minimal_preprocessor = get_catboost_preprocessing_pipeline(include_feature_engineering=False)

    def _apply_feature_engineering_externally(self, X: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Применяет FE-шаги вручную и подготавливает категориальные признаки."""

        if is_train:
            logger.info("Applying FE pipeline (fit_transform) to X_train.")
            X_processed = self.fe_pipeline.fit_transform(X)
        else:
            logger.info("Applying FE pipeline (transform) to X_test.")
            X_processed = self.fe_pipeline.transform(X)

        # После FE, CatBoost ожидает обработанные категории.
        cat_cols = list(X_processed.select_dtypes(include=['object', 'category']).columns)

        if cat_cols:
            logger.info("Preparing category columns for CatBoost: handling NaN and setting type.")
            X_processed[cat_cols] = X_processed[cat_cols].replace({None: np.nan})
            X_processed[cat_cols] = X_processed[cat_cols].fillna('Missing')

            # 3. Установка типа 'category' для CatBoost
        for col in cat_cols:
            X_processed[col] = X_processed[col].astype('category')

        return X_processed

    def train(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_test: pd.DataFrame,
            y_test: np.ndarray,
            fit_kwargs: Optional[Dict] = None,
            external_preprocessor: Optional[Pipeline] = None
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Переопределение train: применяет FE вручную, затем вызывает родительский метод
        с минимальным пайплайном и обработанными данными.
        """

        # 1. FE (ВНЕ ПАЙПЛАЙНА)
        X_train_processed = self._apply_feature_engineering_externally(X_train, is_train=True)
        X_test_processed = self._apply_feature_engineering_externally(X_test, is_train=False)

        # --- 2. ПОДГОТОВКА FIT_KWARGS ДЛЯ CATBOOST ---
        cat_features_for_fit = list(
            X_train_processed
            .select_dtypes(include=['object', 'category'])
            .columns
        )

        fit_kwargs = fit_kwargs or {}
        fit_kwargs.update({
            # CatBoost работает с cat_features в fit, поэтому используем префикс 'model__'
            'model__cat_features': cat_features_for_fit,
            'model__eval_set': (X_test_processed, y_test),
        })

        # --- 3. ВЫЗОВ РОДИТЕЛЬСКОГО МЕТОДА С ОБРАБОТАННЫМИ ДАННЫМИ ---

        return super().train(
            X_train=X_train_processed,
            y_train=y_train,
            X_test=X_test_processed,
            y_test=y_test,
            fit_kwargs=fit_kwargs,
            external_preprocessor=self.minimal_preprocessor
        )

    def _get_model(self):
        """Возвращает инициализированный CatBoostClassifier."""
        # Используйте CatBoostRegressor, если это задача регрессии
        return CatBoostClassifier(**self.model_params)