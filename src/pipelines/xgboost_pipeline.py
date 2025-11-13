# src/pipelines/xgboost_pipeline.py
from narwhals import DataFrame
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from sklearn.base import BaseEstimator
from src.pipelines.base_pipeline import BasePipelineBuilder
from src.pipelines.preprocessor import get_model_specific_pipeline


class XGBoostPipelineBuilder(BasePipelineBuilder):
    """
    Строитель для XGBoostClassifier.
    использует полный препроцессинг (OHE, Scaling).
    """

    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        return get_model_specific_pipeline(
            model_name= 'default',
            include_feature_engineering=feature_engineering
        )

    def _get_model(self) -> BaseEstimator:
        params = {'random_state': self.random_state}
        params.update(self.model_params)

        return XGBClassifier(**params)

