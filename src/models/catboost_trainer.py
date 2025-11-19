# src/models/catboost_trainer.py

from catboost import CatBoostClassifier
from src.models.trainer_interface import BaseTrainer
from sklearn.pipeline import Pipeline
from typing import Dict, Optional
import pandas as pd

from src.features.transformers import DataFrameCoercer, FeatureCreator, AnomalyHandler

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
        self.fe_pipeline = Pipeline(steps=[
            ('coercer', DataFrameCoercer()),
            ('anomaly_handler', AnomalyHandler()),
            ('feature_creator', FeatureCreator()),
        ])
        self.minimal_preprocessor = get_catboost_preprocessing_pipeline(include_feature_engineering=False)

    def _prepare_data(self, X: pd.DataFrame, y=None, is_train: bool = True):
        """Apply manual FE and coerce categoricals so CatBoost sees consistent metadata."""
        if is_train:
            logger.info("Applying FE pipeline (fit_transform) to training split.")
            X_processed = self.fe_pipeline.fit_transform(X)
        else:
            logger.info("Applying FE pipeline (transform) to evaluation split.")
            X_processed = self.fe_pipeline.transform(X)

        cat_cols = list(X_processed.select_dtypes(include=['object', 'category']).columns)
        if not cat_cols:
            return X_processed

        logger.info("Preparing categorical columns for CatBoost.")
        for col in cat_cols:
            series = X_processed[col].astype('category')
            if 'Missing' not in series.cat.categories:
                series = series.cat.add_categories(['Missing'])
            X_processed[col] = series.fillna('Missing')

        return X_processed

    def _build_preprocessor(self, external_preprocessor: Optional[Pipeline] = None) -> Pipeline:
        """CatBoost gets a minimal passthrough preprocessor unless explicitly overridden."""
        return super()._build_preprocessor(external_preprocessor or self.minimal_preprocessor)

    def _build_fit_kwargs(
            self,
            fit_kwargs: Optional[Dict],
            *,
            X_train,
            y_train,
            X_test,
            y_test,
            preprocessor: Pipeline
    ) -> Dict:
        cat_features = list(X_train.select_dtypes(include=['object', 'category']).columns)
        final_kwargs = dict(fit_kwargs or {})
        final_kwargs.setdefault('model__cat_features', cat_features)
        final_kwargs.setdefault('model__eval_set', (X_test, y_test))
        return final_kwargs

    def _get_model(self):
        """Возвращает инициализированный CatBoostClassifier."""
        return CatBoostClassifier(**self.model_params)