# src/pipelines/sklearn_pipeline.py

from typing import Dict, Any
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.pipelines.base_pipeline import BasePipelineBuilder
from src.pipelines.base_preprocessor import get_full_preprocessor

class SKLearnPipelineBuilder(BasePipelineBuilder):
    """
    Строитель для стандартных Sklearn-моделей.
    Использует полный препроцессинг (OHE, Scaling).
    """

    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        """
        Собирает полный пайплайн препроцессинга.
        FE (если нужно) -> Sklearn Full Preprocessor.
        """
        steps = []

        if feature_engineering:
            steps.append(('feature_engineering', self._get_feature_engineering_pipeline()))

        # полный препроцессор Sklearn-стиля (OHE + Scaling)
        # get_full_preprocessor() возвращает ColumnTransformer, который мы инкапсулируем в Pipeline
        steps.append(('full_preprocessor', Pipeline([('column_transformer', get_full_preprocessor())])))

        return Pipeline(steps)

    def _get_model(self) -> BaseEstimator:
        """
        Получает инстанс модели, используя self.random_state из Base.
        """
        params = {'random_state': self.random_state}
        params.update(self.model_params)

        # в реальном коде тут должна быть логика извлечения model_class
        return LogisticRegression(**params)