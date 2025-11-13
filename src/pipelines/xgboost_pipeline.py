# src/pipelines/xgboost_pipeline.py
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from sklearn.base import BaseEstimator
from src.pipelines.base_pipeline import BasePipelineBuilder
from src.pipelines.base_preprocessor import get_full_preprocessor


class XGBoostPipelineBuilder(BasePipelineBuilder):
    """
    Строитель для XGBoostClassifier.
    использует полный препроцессинг (OHE, Scaling).
    """

    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        # XGBoost требует полный численный и OHE набор, как и Sklearn
        steps = []

        if feature_engineering:
            steps.append(('feature_engineering', self._get_feature_engineering_pipeline()))

        # полный препроцессор Sklearn-стиля (OHE + Scaling)
        # get_full_preprocessor() возвращает ColumnTransformer, который мы инкапсулируем в Pipeline
        steps.append(('full_preprocessor', Pipeline([('column_transformer', get_full_preprocessor())])))

        return Pipeline(steps)

    def _get_model(self) -> BaseEstimator:
        params = {'random_state': self.random_state}
        params.update(self.model_params)

        return XGBClassifier(**params)